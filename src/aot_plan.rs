// ═══════════════════════════════════════════════════════════════════════
// CRYS-L — AOT Plan Compiler v2.0 (Level 2: Full Plan JIT)
// Percy Rojas M. · Qomni AI Lab · 2026
//
// Level 1 (v1.0): Pre-resolved dispatch tables — ~1,400 ns/plan
// Level 2 (v2.0): Entire plan compiled as ONE Cranelift function
//   - All oracle bodies INLINED as Cranelift IR
//   - Zero per-step overhead: no fn_ptr calls, no arg marshaling
//   - No HashMap, no Vec, no String, no Instant in hot path
//   - Target: 15-30 ns per plan execution
//
// Architecture:
//   At startup, for each plan:
//     1. Walk pre-resolved execution order (same as Level 1)
//     2. For each step, read the oracle's bytecode from bytecode::Module
//     3. Lower each oracle's opcodes to Cranelift IR (same as jit.rs)
//     4. Chain them: step[i] output feeds step[i+1] inputs via SSA values
//     5. Compile into ONE native function per plan
//
//   At request time:
//     1. Single indirect call to the compiled plan function
//     2. Results written to stack array — zero heap allocation
//     3. Return code indicates success
//
// JIT Plan ABI:
//   extern "C" fn(params: *const f64, n_params: usize, results: *mut f64) -> i32
//   - params: input parameter values (ordered by plan param declaration)
//   - n_params: number of input params
//   - results: output buffer for step results (caller allocates)
//   - returns: number of steps written to results
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::time::Instant;

use crate::ast::{PlanDecl, Expr, BinaryOp};
use crate::plan::{PlanResult, StepResult, parallel_groups};
use crate::bytecode::{self, Const, Op};

// ── Cranelift imports (for Level 2 JIT) ───────────────────────────────
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlags, UserFuncName,
    types::{F64, I64, I32},
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

// ── Arg source: where each oracle argument comes from ─────────────────

#[derive(Clone, Copy, Debug)]
enum ArgSource {
    Param(usize),   // from input params[i]
    Step(usize),    // from results[i] (output of a previous step)
    Const(f64),     // literal constant
}

// ── Pre-resolved step ─────────────────────────────────────────────────

#[derive(Clone)]
struct FastStep {
    oracle_idx: usize,         // index into oracle_fns array
    oracle_name: String,       // for bytecode lookup
    args:       Vec<ArgSource>,
}

// ── Pre-compiled plan ─────────────────────────────────────────────────

struct FastPlan {
    param_names:  Vec<String>,
    exec_steps:   Vec<FastStep>,     // topological execution order
    exec_to_decl: Vec<usize>,       // exec index → original declaration index
    step_names:   Vec<String>,       // for output only
    oracle_names: Vec<String>,       // for output only
    n_groups:     usize,
    jit_plan_fn:  Option<usize>,     // Level 2: compiled plan function pointer
    n_steps:      usize,             // number of steps (for JIT result buffer)
}

// ── Oracle entry: direct fn pointer or JIT address ────────────────────

#[derive(Clone)]
struct OracleFn {
    builtin:      fn(&[f64]) -> f64,
    jit_addr:     Option<usize>,
    jit_n_params: usize,
}

type JitOracleFn = unsafe extern "C" fn(*const f64, usize) -> f64;

// Level 2: JIT plan function type
// (params_ptr, n_params, results_ptr) -> n_results
type JitPlanFn = unsafe extern "C" fn(*const f64, usize, *mut f64) -> i32;

// ── Math shims (same as jit.rs) ──────────────────────────────────────
extern "C" fn fast_pow(base: f64, exp: f64) -> f64 {
    if base > 0.0 { (exp * base.log2()).exp2() } else { base.powf(exp) }
}

// ═══════════════════════════════════════════════════════════════════════
// AotPlanCache — the fast execution engine
// ═══════════════════════════════════════════════════════════════════════

/// Level 3: Turbo index entry -- zero HashMap, zero allocation at execution.
pub struct TurboPlan {
    pub name: String,
    pub fn_ptr: usize,
    pub reg_fn_ptr: Option<usize>,  // Level 4: register-ABI fn_ptr
    pub n_params: usize,
    pub n_steps: usize,
    pub param_names: Vec<String>,
    pub step_names: Vec<String>,
    pub oracle_names: Vec<String>,
    pub exec_to_decl: Vec<usize>,
}

pub struct AotPlanCache {
    plans:      HashMap<String, FastPlan>,
    oracle_fns: Vec<OracleFn>,
    turbo_table: Vec<TurboPlan>,
    turbo_index: HashMap<String, usize>,
}

unsafe impl Send for AotPlanCache {}
unsafe impl Sync for AotPlanCache {}

impl AotPlanCache {
    /// Build the cache ONCE at startup.
    pub fn compile(
        plans:    &[PlanDecl],
        builtins: &HashMap<String, fn(&[f64]) -> f64>,
        jit_map:  &Option<HashMap<String, (usize, usize)>>,
    ) -> Self {
        let mut oracle_fns: Vec<OracleFn> = Vec::new();
        let mut oracle_idx: HashMap<String, usize> = HashMap::new();

        // Register builtins
        for (name, f) in builtins {
            let idx = oracle_fns.len();
            let jit = jit_map.as_ref().and_then(|m| m.get(name));
            oracle_fns.push(OracleFn {
                builtin:      *f,
                jit_addr:     jit.map(|&(a, _)| a),
                jit_n_params: jit.map(|&(_, n)| n).unwrap_or(0),
            });
            oracle_idx.insert(name.clone(), idx);
        }

        // Register JIT-only oracles
        if let Some(jm) = jit_map {
            for (name, &(addr, n)) in jm.iter() {
                if !oracle_idx.contains_key(name) {
                    let idx = oracle_fns.len();
                    oracle_fns.push(OracleFn {
                        builtin:      |_| 0.0,
                        jit_addr:     Some(addr),
                        jit_n_params: n,
                    });
                    oracle_idx.insert(name.clone(), idx);
                }
            }
        }

        // Compile each plan (Level 1)
        let mut compiled: HashMap<String, FastPlan> = HashMap::new();
        for plan in plans {
            if let Some(fp) = Self::compile_plan(plan, &oracle_idx) {
                compiled.insert(plan.name.clone(), fp);
            }
        }

        let n_plans = compiled.len();
        let n_oracles = oracle_fns.len();
        eprintln!("  AOT Plan Cache: {} plans compiled (Level 1), {} oracles indexed", n_plans, n_oracles);

        Self { plans: compiled, oracle_fns, turbo_table: Vec::new(), turbo_index: HashMap::new() }
    }

    /// Level 2: Compile entire plans as single Cranelift functions.
    /// Must be called AFTER compile() and takes the bytecode module to inline oracle bodies.
    pub fn compile_plans_jit(&mut self, bc: &bytecode::Module) {
        // Build Cranelift JIT module
        let mut flag_builder = settings::builder();
        if let Err(e) = flag_builder.set("use_colocated_libcalls", "false") {
            eprintln!("  [AOT L2] cranelift flag error: {}", e); return;
        }
        if let Err(e) = flag_builder.set("is_pic", "false") {
            eprintln!("  [AOT L2] cranelift flag error: {}", e); return;
        }
        if let Err(e) = flag_builder.set("opt_level", "speed") {
            eprintln!("  [AOT L2] cranelift flag error: {}", e); return;
        }
        let flags = settings::Flags::new(flag_builder);
        let isa = match cranelift_codegen::isa::lookup_by_name("x86_64") {
            Ok(b) => match b.finish(flags) {
                Ok(i) => i,
                Err(e) => { eprintln!("  [AOT L2] isa finish: {}", e); return; }
            },
            Err(e) => { eprintln!("  [AOT L2] isa lookup: {}", e); return; }
        };

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        builder.symbol("fast_pow", fast_pow as *const u8);
        let mut jit_module = JITModule::new(builder);

        let mut compiled_count = 0u32;
        let mut func_ids: Vec<(String, cranelift_module::FuncId)> = Vec::new();
        let plan_names: Vec<String> = self.plans.keys().cloned().collect();

        for plan_name in &plan_names {
            match self.compile_one_plan_jit(plan_name, bc, &mut jit_module) {
                Ok(func_id) => {
                    func_ids.push((plan_name.clone(), func_id));
                    compiled_count += 1;
                }
                Err(e) => {
                    eprintln!("  [AOT L2] plan '{}' JIT failed: {}", plan_name, e);
                }
            }
        }

        if compiled_count == 0 {
            eprintln!("  [AOT L2] no plans compiled");
            return;
        }

        // Finalize all at once
        if let Err(e) = jit_module.finalize_definitions() {
            eprintln!("  [AOT L2] finalize error: {}", e);
            return;
        }

        // Resolve function pointers
        for (plan_name, func_id) in &func_ids {
            let raw_ptr = jit_module.get_finalized_function(*func_id);
            if let Some(plan) = self.plans.get_mut(plan_name) {
                plan.jit_plan_fn = Some(raw_ptr as usize);
                eprintln!("  [AOT L2] plan '{}' -> {:p} ({} steps inlined)",
                    plan_name, raw_ptr, plan.n_steps);
            }
        }

        // Leak the JIT module to keep code alive forever
        Box::leak(Box::new(jit_module));

        eprintln!("  AOT Level 2: {} plans compiled as single native functions", compiled_count);
    }

    /// Compile one plan into a single Cranelift function that inlines all oracle bodies.
    fn compile_one_plan_jit(
        &self,
        plan_name: &str,
        bc: &bytecode::Module,
        jit_module: &mut JITModule,
    ) -> Result<cranelift_module::FuncId, String> {
        let plan = self.plans.get(plan_name)
            .ok_or_else(|| format!("plan '{}' not found", plan_name))?;

        let sym = format!("crysl_plan_{}", plan_name.replace(['-', '.', ' '], "_"));

        // Function signature: fn(params: *const f64, n_params: usize, results: *mut f64) -> i32
        let mut sig = jit_module.make_signature();
        sig.params.push(AbiParam::new(I64));  // params ptr
        sig.params.push(AbiParam::new(I64));  // n_params
        sig.params.push(AbiParam::new(I64));  // results ptr
        sig.returns.push(AbiParam::new(I32)); // n_results

        let func_id = jit_module
            .declare_function(&sym, Linkage::Export, &sig)
            .map_err(|e| format!("declare: {}", e))?;

        // Declare fast_pow shim
        let mut fp_sig = jit_module.make_signature();
        fp_sig.params.push(AbiParam::new(F64));
        fp_sig.params.push(AbiParam::new(F64));
        fp_sig.returns.push(AbiParam::new(F64));
        let fast_pow_id = jit_module
            .declare_function("fast_pow", Linkage::Import, &fp_sig)
            .map_err(|e| format!("declare fast_pow: {}", e))?;

        // Build function body
        let mut ctx = jit_module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        {
            let mut func_ctx = FunctionBuilderContext::new();
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = bcx.create_block();
            bcx.append_block_params_for_function_params(entry_block);
            bcx.switch_to_block(entry_block);
            bcx.seal_block(entry_block);

            let params_ptr  = bcx.block_params(entry_block)[0]; // *const f64
            let _n_params   = bcx.block_params(entry_block)[1]; // usize (unused)
            let results_ptr = bcx.block_params(entry_block)[2]; // *mut f64

            let fast_pow_ref = jit_module.declare_func_in_func(fast_pow_id, bcx.func);

            // Load plan input params from params_ptr
            let n_plan_params = plan.param_names.len();
            let mut param_vals: Vec<cranelift_codegen::ir::Value> = Vec::with_capacity(n_plan_params);
            for i in 0..n_plan_params {
                let offset = (i * 8) as i32;
                let val = bcx.ins().load(F64, MemFlags::trusted(), params_ptr, offset);
                param_vals.push(val);
            }

            // step_results[i] = SSA value of step i's output
            let mut step_results: Vec<cranelift_codegen::ir::Value> = Vec::new();

            // Variable counter for branching oracles (each gets unique range)
            let mut var_counter: usize = 0;

            // For each step in execution order, inline the oracle body
            for (step_idx, step) in plan.exec_steps.iter().enumerate() {
                // Build the argument SSA values for this oracle call
                let mut arg_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
                for src in &step.args {
                    let val = match *src {
                        ArgSource::Param(j) => param_vals[j],
                        ArgSource::Step(j)  => step_results[j],
                        ArgSource::Const(v) => bcx.ins().f64const(v),
                    };
                    arg_vals.push(val);
                }

                // Try to find oracle in bytecode module for inlining
                let oracle_meta = bc.oracles.iter().find(|o| o.name == step.oracle_name);

                let result_val = if let Some(meta) = oracle_meta {
                    // Check if oracle has branches
                    let oracle_end = {
                        let idx = bc.oracles.iter().position(|o| o.entry_ip == meta.entry_ip).unwrap_or(0);
                        bc.oracles.get(idx + 1).map(|o| o.entry_ip).unwrap_or(bc.code.len())
                    };
                    let has_branches = {
                        let mut found = false;
                        let mut scan = meta.entry_ip;
                        while scan < oracle_end {
                            match bc.code[scan].op {
                                Op::JumpFalse | Op::Jump => { found = true; break; }
                                Op::Halt => break,
                                _ => {}
                            }
                            scan += 1;
                        }
                        found
                    };

                    if has_branches {
                        Self::inline_oracle_body_branching(
                            bc, meta.entry_ip, oracle_end, meta.n_params,
                            &arg_vals, &mut bcx, fast_pow_ref, &mut var_counter,
                        )
                    } else {
                        Self::inline_oracle_body(
                            bc, meta.entry_ip, meta.n_params,
                            &arg_vals, &mut bcx, fast_pow_ref,
                        )
                    }
                } else {
                    // Oracle not in bytecode (builtin-only) — call via fn_ptr
                    let of = &self.oracle_fns[step.oracle_idx];
                    if let Some(addr) = of.jit_addr {
                        Self::emit_indirect_oracle_call(addr, &arg_vals, &mut bcx)
                    } else {
                        Self::emit_indirect_oracle_call(of.builtin as usize, &arg_vals, &mut bcx)
                    }
                };

                // Store result to results buffer
                let offset = (step_idx * 8) as i32;
                bcx.ins().store(MemFlags::trusted(), result_val, results_ptr, offset);

                step_results.push(result_val);
            }

            // Return number of steps
            let n_steps = bcx.ins().iconst(I32, plan.exec_steps.len() as i64);
            bcx.ins().return_(&[n_steps]);
            bcx.finalize();
        }

        jit_module
            .define_function(func_id, &mut ctx)
            .map_err(|e| format!("define '{}': {}", plan_name, e))?;
        jit_module.clear_context(&mut ctx);

        Ok(func_id)
    }

    /// Inline an oracle's bytecode body as Cranelift IR within the plan function.
    /// Returns the SSA value of the oracle's return value.
    fn inline_oracle_body(
        bc: &bytecode::Module,
        entry_ip: usize,
        n_params: usize,
        arg_vals: &[cranelift_codegen::ir::Value],
        bcx: &mut FunctionBuilder,
        fast_pow_ref: cranelift_codegen::ir::FuncRef,
    ) -> cranelift_codegen::ir::Value {
        use cranelift_codegen::ir::condcodes::FloatCC;

        let mut reg_vals: HashMap<u16, cranelift_codegen::ir::Value> = HashMap::new();
        let mut var_vals: HashMap<u16, cranelift_codegen::ir::Value> = HashMap::new();
        let mut const_regs: HashMap<u16, f64> = HashMap::new();

        // Pre-load oracle params from arg_vals
        for (i, val) in arg_vals.iter().enumerate().take(n_params) {
            reg_vals.insert(i as u16, *val);
        }

        macro_rules! reg {
            ($r:expr) => {
                *reg_vals.entry($r).or_insert_with(|| bcx.ins().f64const(0.0))
            }
        }

        let mut ip = entry_ip;
        let mut return_val: Option<cranelift_codegen::ir::Value> = None;

        while ip < bc.code.len() && return_val.is_none() {
            let instr = &bc.code[ip];
            match instr.op {
                Op::LoadConst => {
                    let ci = instr.b as usize;
                    let fval: f64 = match bc.consts.get(ci) {
                        Some(Const::Float(f)) => *f,
                        Some(Const::Int(n))   => *n as f64,
                        Some(Const::Bool(b))  => if *b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    };
                    const_regs.insert(instr.a, fval);
                    reg_vals.insert(instr.a, bcx.ins().f64const(fval));
                }
                Op::LoadTrit => {
                    let tv = (instr.b as i16) as f64;
                    const_regs.insert(instr.a, tv);
                    reg_vals.insert(instr.a, bcx.ins().f64const(tv));
                }
                Op::StoreVar => {
                    let src = reg!(instr.b);
                    var_vals.insert(instr.a, src);
                }
                Op::LoadVar => {
                    let val = *var_vals.entry(instr.b)
                        .or_insert_with(|| {
                            reg_vals.get(&instr.b).copied()
                                .unwrap_or_else(|| bcx.ins().f64const(0.0))
                        });
                    reg_vals.insert(instr.a, val);
                }
                Op::Move => {
                    let src = reg!(instr.b);
                    reg_vals.insert(instr.a, src);
                }
                Op::Add => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    reg_vals.insert(instr.a, bcx.ins().fadd(l, r));
                }
                Op::Sub => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    reg_vals.insert(instr.a, bcx.ins().fsub(l, r));
                }
                Op::Mul => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    reg_vals.insert(instr.a, bcx.ins().fmul(l, r));
                }
                Op::Div => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    let zero = bcx.ins().f64const(0.0);
                    let is_zero = bcx.ins().fcmp(FloatCC::Equal, r, zero);
                    let quot = bcx.ins().fdiv(l, r);
                    let result = bcx.ins().select(is_zero, zero, quot);
                    reg_vals.insert(instr.a, result);
                }
                Op::Pow => {
                    let base = reg!(instr.b);
                    let static_exp = detect_const_reg(bc, ip, entry_ip, instr.c)
                        .or_else(|| const_regs.get(&instr.c).copied());
                    let res = if let Some(exp_val) = static_exp {
                        inline_pow_const(exp_val, base, bcx, fast_pow_ref)
                    } else {
                        let exp = reg!(instr.c);
                        let call = bcx.ins().call(fast_pow_ref, &[base, exp]);
                        bcx.inst_results(call)[0]
                    };
                    reg_vals.insert(instr.a, res);
                }
                Op::Neg => {
                    let v = reg!(instr.b);
                    reg_vals.insert(instr.a, bcx.ins().fneg(v));
                }
                Op::Eq => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    let cmp = bcx.ins().fcmp(FloatCC::Equal, l, r);
                    let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                    reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                }
                Op::Lt => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    let cmp = bcx.ins().fcmp(FloatCC::LessThan, l, r);
                    let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                    reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                }
                Op::Gt => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    let cmp = bcx.ins().fcmp(FloatCC::GreaterThan, l, r);
                    let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                    reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                }
                Op::Not => {
                    let v = reg!(instr.b);
                    let zero = bcx.ins().f64const(0.0);
                    let cmp = bcx.ins().fcmp(FloatCC::Equal, v, zero);
                    let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                    reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                }
                Op::And => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    let zero = bcx.ins().f64const(0.0);
                    let cl = bcx.ins().fcmp(FloatCC::NotEqual, l, zero);
                    let cr = bcx.ins().fcmp(FloatCC::NotEqual, r, zero);
                    let both = bcx.ins().band(cl, cr);
                    let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                    reg_vals.insert(instr.a, bcx.ins().select(both, t, f));
                }
                Op::Or => {
                    let l = reg!(instr.b); let r = reg!(instr.c);
                    let zero = bcx.ins().f64const(0.0);
                    let cl = bcx.ins().fcmp(FloatCC::NotEqual, l, zero);
                    let cr = bcx.ins().fcmp(FloatCC::NotEqual, r, zero);
                    let either = bcx.ins().bor(cl, cr);
                    let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                    reg_vals.insert(instr.a, bcx.ins().select(either, t, f));
                }
                Op::Return => {
                    let ret = reg!(instr.a);
                    let zero = bcx.ins().f64const(0.0);
                    let is_ordered = bcx.ins().fcmp(FloatCC::Ordered, ret, ret);
                    let safe_ret = bcx.ins().select(is_ordered, ret, zero);
                    return_val = Some(safe_ret);
                }
                Op::Halt => {
                    return_val = Some(bcx.ins().f64const(0.0));
                }
                _ => { /* skip unknown ops */ }
            }
            ip += 1;
        }

        return_val.unwrap_or_else(|| bcx.ins().f64const(0.0))
    }

    /// Inline a branching oracle body using Cranelift Variables.
    fn inline_oracle_body_branching(
        bc: &bytecode::Module,
        entry_ip: usize,
        oracle_end: usize,
        n_params: usize,
        arg_vals: &[cranelift_codegen::ir::Value],
        bcx: &mut FunctionBuilder,
        fast_pow_ref: cranelift_codegen::ir::FuncRef,
        var_counter: &mut usize,
    ) -> cranelift_codegen::ir::Value {
        use cranelift_codegen::ir::condcodes::FloatCC;

        // Collect branch targets
        let mut branch_targets: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        {
            let mut scan = entry_ip;
            while scan < oracle_end {
                let si = &bc.code[scan];
                match si.op {
                    Op::JumpFalse => {
                        branch_targets.insert(scan + 1);
                        branch_targets.insert(si.b as usize);
                    }
                    Op::Jump => { branch_targets.insert(si.b as usize); }
                    Op::Halt => break,
                    _ => {}
                }
                scan += 1;
            }
        }

        // Find max register in this oracle
        let max_reg = bc.code[entry_ip..oracle_end].iter()
            .flat_map(|i| [i.a, i.b, i.c])
            .max().unwrap_or(0) as usize + 1;

        // Unique variable base for this inline instance
        let var_base = *var_counter;
        *var_counter += max_reg + 1; // +1 for result var

        let cvars: Vec<Variable> = (0..max_reg).map(|i| {
            let v = Variable::from_u32((var_base + i) as u32);
            bcx.declare_var(v, F64);
            v
        }).collect();

        // Result variable
        let result_var = Variable::from_u32((var_base + max_reg) as u32);
        bcx.declare_var(result_var, F64);
        let zero_v = bcx.ins().f64const(0.0);
        bcx.def_var(result_var, zero_v);

        // Initialize all variables to 0
        for v in &cvars { bcx.def_var(*v, zero_v); }

        // Initialize param registers
        for (i, val) in arg_vals.iter().enumerate().take(n_params) {
            if i < cvars.len() { bcx.def_var(cvars[i], *val); }
        }

        // Create blocks
        let oracle_entry_block = bcx.create_block();
        let merge_block = bcx.create_block();

        let mut block_map: HashMap<usize, cranelift_codegen::ir::Block> = HashMap::new();
        block_map.insert(entry_ip, oracle_entry_block);
        for &target in &branch_targets {
            block_map.entry(target).or_insert_with(|| bcx.create_block());
        }

        // Jump to oracle entry
        bcx.ins().jump(oracle_entry_block, &[]);
        bcx.seal_block(oracle_entry_block);
        bcx.switch_to_block(oracle_entry_block);

        let mut const_regs: HashMap<u16, f64> = HashMap::new();
        let mut var_vals: HashMap<u16, cranelift_codegen::ir::Value> = HashMap::new();

        let get_v = |r: u16, bcx2: &mut FunctionBuilder| -> cranelift_codegen::ir::Value {
            if r < cvars.len() as u16 { bcx2.use_var(cvars[r as usize]) }
            else { bcx2.ins().f64const(0.0) }
        };
        let def_v = |r: u16, val: cranelift_codegen::ir::Value, bcx2: &mut FunctionBuilder| {
            if r < cvars.len() as u16 { bcx2.def_var(cvars[r as usize], val); }
        };

        let mut ip = entry_ip;
        let mut block_filled = false;

        while ip < oracle_end {
            if let Some(&tgt_block) = block_map.get(&ip) {
                if tgt_block != oracle_entry_block || ip != entry_ip {
                    if !block_filled {
                        bcx.ins().jump(tgt_block, &[]);
                    }
                    bcx.seal_block(tgt_block);
                    bcx.switch_to_block(tgt_block);
                    block_filled = false;
                }
            }
            if block_filled { ip += 1; continue; }

            let instr = bc.code[ip];
            match instr.op {
                Op::LoadConst => {
                    let ci = instr.b as usize;
                    let fval: f64 = match bc.consts.get(ci) {
                        Some(Const::Float(f)) => *f,
                        Some(Const::Int(n))   => *n as f64,
                        Some(Const::Bool(b))  => if *b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    };
                    const_regs.insert(instr.a, fval);
                    def_v(instr.a, bcx.ins().f64const(fval), bcx);
                }
                Op::LoadTrit => {
                    let tv = (instr.b as i16) as f64;
                    const_regs.insert(instr.a, tv);
                    def_v(instr.a, bcx.ins().f64const(tv), bcx);
                }
                Op::StoreVar => {
                    let sv = get_v(instr.b, bcx);
                    var_vals.insert(instr.a, sv);
                }
                Op::LoadVar => {
                    let lv = *var_vals.entry(instr.b).or_insert(zero_v);
                    def_v(instr.a, lv, bcx);
                }
                Op::Move => {
                    let mv = get_v(instr.b, bcx);
                    def_v(instr.a, mv, bcx);
                }
                Op::Add  => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); def_v(instr.a, bcx.ins().fadd(l,r), bcx); }
                Op::Sub  => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); def_v(instr.a, bcx.ins().fsub(l,r), bcx); }
                Op::Mul  => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); def_v(instr.a, bcx.ins().fmul(l,r), bcx); }
                Op::Div  => {
                    let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx);
                    let z=bcx.ins().f64const(0.0);
                    let is_zero=bcx.ins().fcmp(FloatCC::Equal,r,z);
                    let quot=bcx.ins().fdiv(l,r);
                    let res=bcx.ins().select(is_zero,z,quot);
                    def_v(instr.a, res, bcx);
                }
                Op::Neg  => { let v=get_v(instr.b,bcx); def_v(instr.a, bcx.ins().fneg(v), bcx); }
                Op::Eq   => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); let c=bcx.ins().fcmp(FloatCC::Equal,l,r); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),bcx); }
                Op::Lt   => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); let c=bcx.ins().fcmp(FloatCC::LessThan,l,r); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),bcx); }
                Op::Gt   => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); let c=bcx.ins().fcmp(FloatCC::GreaterThan,l,r); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),bcx); }
                Op::Not  => { let v=get_v(instr.b,bcx); let z=bcx.ins().f64const(0.0); let c=bcx.ins().fcmp(FloatCC::Equal,v,z); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),bcx); }
                Op::And  => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); let z=bcx.ins().f64const(0.0); let cl=bcx.ins().fcmp(FloatCC::NotEqual,l,z); let cr=bcx.ins().fcmp(FloatCC::NotEqual,r,z); let both=bcx.ins().band(cl,cr); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(both,t,f),bcx); }
                Op::Or   => { let l=get_v(instr.b,bcx); let r=get_v(instr.c,bcx); let z=bcx.ins().f64const(0.0); let cl=bcx.ins().fcmp(FloatCC::NotEqual,l,z); let cr=bcx.ins().fcmp(FloatCC::NotEqual,r,z); let either=bcx.ins().bor(cl,cr); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(either,t,f),bcx); }
                Op::Pow  => {
                    let base = get_v(instr.b, bcx);
                    let static_exp = detect_const_reg(bc, ip, entry_ip, instr.c)
                        .or_else(|| const_regs.get(&instr.c).copied());
                    let res = if let Some(exp_val) = static_exp {
                        inline_pow_const(exp_val, base, bcx, fast_pow_ref)
                    } else {
                        let exp = get_v(instr.c, bcx);
                        let call = bcx.ins().call(fast_pow_ref, &[base, exp]);
                        bcx.inst_results(call)[0]
                    };
                    def_v(instr.a, res, bcx);
                }
                Op::JumpFalse => {
                    let cond_f = get_v(instr.a, bcx);
                    let z = bcx.ins().f64const(0.0);
                    let is_true = bcx.ins().fcmp(FloatCC::NotEqual, cond_f, z);
                    let then_blk = *block_map.get(&(ip + 1))
                        .unwrap_or(&oracle_entry_block);
                    let else_blk = block_map[&(instr.b as usize)];
                    bcx.ins().brif(is_true, then_blk, &[], else_blk, &[]);
                    block_filled = true;
                }
                Op::Jump => {
                    let tgt = *block_map.get(&(instr.b as usize))
                        .unwrap_or(&oracle_entry_block);
                    bcx.ins().jump(tgt, &[]);
                    block_filled = true;
                }
                Op::Return => {
                    let ret = get_v(instr.a, bcx);
                    let z = bcx.ins().f64const(0.0);
                    let ok = bcx.ins().fcmp(FloatCC::Ordered, ret, ret);
                    let safe = bcx.ins().select(ok, ret, z);
                    bcx.def_var(result_var, safe);
                    bcx.ins().jump(merge_block, &[]);
                    block_filled = true;
                }
                Op::Halt => {
                    let z = bcx.ins().f64const(0.0);
                    bcx.def_var(result_var, z);
                    bcx.ins().jump(merge_block, &[]);
                    block_filled = true;
                }
                _ => {}
            }
            ip += 1;
        }

        if !block_filled {
            let z = bcx.ins().f64const(0.0);
            bcx.def_var(result_var, z);
            bcx.ins().jump(merge_block, &[]);
        }

        bcx.seal_block(merge_block);
        bcx.switch_to_block(merge_block);
        bcx.use_var(result_var)
    }

    /// Emit an indirect call to a pre-compiled JIT oracle function.
    fn emit_indirect_oracle_call(
        addr: usize,
        arg_vals: &[cranelift_codegen::ir::Value],
        bcx: &mut FunctionBuilder,
    ) -> cranelift_codegen::ir::Value {
        use cranelift_codegen::ir::condcodes::FloatCC;
        // Create a stack slot for args
        let n_args = arg_vals.len().max(1);
        let slot = bcx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            (n_args * 8) as u32,
            3, // 8-byte alignment
        ));
        for (i, &val) in arg_vals.iter().enumerate() {
            bcx.ins().stack_store(val, slot, (i * 8) as i32);
        }
        let slot_addr = bcx.ins().stack_addr(I64, slot, 0);
        let call_sig = bcx.func.dfg.signatures.push(cranelift_codegen::ir::Signature {
            params: vec![AbiParam::new(I64), AbiParam::new(I64)],
            returns: vec![AbiParam::new(F64)],
            call_conv: CallConv::SystemV,
        });
        let fn_addr_val = bcx.ins().iconst(I64, addr as i64);
        let n_args_val = bcx.ins().iconst(I64, arg_vals.len() as i64);
        let call = bcx.ins().call_indirect(call_sig, fn_addr_val, &[slot_addr, n_args_val]);
        let result = bcx.inst_results(call)[0];
        // NaN clamp
        let zero = bcx.ins().f64const(0.0);
        let ok = bcx.ins().fcmp(FloatCC::Ordered, result, result);
        bcx.ins().select(ok, result, zero)
    }

    fn compile_plan(plan: &PlanDecl, oracle_idx: &HashMap<String, usize>) -> Option<FastPlan> {
        let groups = parallel_groups(plan);
        let n_groups = groups.len();

        let param_map: HashMap<&str, usize> = plan.params.iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();

        let exec_order: Vec<usize> = groups.iter().flat_map(|g| g.iter().copied()).collect();

        let mut step_map: HashMap<&str, usize> = HashMap::new();
        for (ei, &di) in exec_order.iter().enumerate() {
            step_map.insert(plan.steps[di].name.as_str(), ei);
        }

        let mut exec_steps = Vec::new();
        let mut step_names = Vec::new();
        let mut oracle_names = Vec::new();
        let mut exec_to_decl = Vec::new();

        for &di in &exec_order {
            let step = &plan.steps[di];
            let oidx = oracle_idx.get(&step.oracle)?;

            let args: Vec<ArgSource> = step.args.iter()
                .map(|e| resolve_arg(e, &param_map, &step_map))
                .collect::<Option<Vec<_>>>()?;

            exec_steps.push(FastStep {
                oracle_idx: *oidx,
                oracle_name: step.oracle.clone(),
                args,
            });
            step_names.push(step.name.clone());
            oracle_names.push(step.oracle.clone());
            exec_to_decl.push(di);
        }

        let n_steps = exec_steps.len();

        Some(FastPlan {
            param_names: plan.params.iter().map(|p| p.name.clone()).collect(),
            exec_steps, exec_to_decl, step_names, oracle_names, n_groups,
            jit_plan_fn: None,
            n_steps,
        })
    }

    /// Level 2: Execute plan via single JIT-compiled function. Zero overhead.
    #[inline]
    pub fn execute_jit(&self, plan_name: &str, param_vals: &[f64])
        -> Option<([f64; 32], usize)>
    {
        let plan = self.plans.get(plan_name)?;
        let addr = plan.jit_plan_fn?;
        let mut results = [0.0f64; 32];
        let n = unsafe {
            let f: JitPlanFn = std::mem::transmute(addr);
            f(param_vals.as_ptr(), param_vals.len(), results.as_mut_ptr())
        };
        Some((results, n as usize))
    }

    /// Execute a plan. Zero HashMap lookups in hot path.
    pub fn execute(&self, plan_name: &str, params: &HashMap<String, f64>) -> Result<PlanResult, String> {
        let plan = self.plans.get(plan_name)
            .ok_or_else(|| format!("AOT: plan '{}' not compiled", plan_name))?;

        // Build ordered param array
        let param_vals: Vec<f64> = plan.param_names.iter()
            .map(|n| *params.get(n).unwrap_or(&0.0))
            .collect();

        // ── Try Level 2 JIT first ──
        if plan.jit_plan_fn.is_some() {
            let t0 = Instant::now();
            if let Some((results, n)) = self.execute_jit(plan_name, &param_vals) {
                let total_ns = t0.elapsed().as_nanos() as f64;
                let n = n.min(plan.exec_steps.len());

                let mut srs: Vec<(usize, StepResult)> = (0..n).map(|i| {
                    (plan.exec_to_decl[i], StepResult {
                        step:       plan.step_names[i].clone(),
                        oracle:     plan.oracle_names[i].clone(),
                        value:      results[i],
                        latency_ns: 0.0,
                    })
                }).collect();
                srs.sort_by_key(|(idx, _)| *idx);

                return Ok(PlanResult {
                    plan_name:             plan_name.to_string(),
                    inputs:                params.clone(),
                    steps:                 srs.into_iter().map(|(_, s)| s).collect(),
                    total_ns,
                    parallel_groups_count: plan.n_groups,
                    cache_hits:            0,
                });
            }
        }

        let t0 = Instant::now();

        // ── Level 1 fallback: array-indexed dispatch ──
        let n = plan.exec_steps.len();
        let mut results = Vec::with_capacity(n);
        let mut abuf = [0.0f64; 8];

        for step in &plan.exec_steps {
            let na = step.args.len().min(8);
            for (i, src) in step.args.iter().enumerate().take(8) {
                abuf[i] = match *src {
                    ArgSource::Param(j) => param_vals[j],
                    ArgSource::Step(j)  => results[j],
                    ArgSource::Const(v) => v,
                };
            }

            let of = &self.oracle_fns[step.oracle_idx];
            let val = if let Some(addr) = of.jit_addr {
                unsafe {
                    let f: JitOracleFn = std::mem::transmute(addr);
                    f(abuf.as_ptr(), na)
                }
            } else {
                (of.builtin)(&abuf[..na])
            };
            results.push(val);
        }

        let total_ns = t0.elapsed().as_nanos() as f64;

        let mut srs: Vec<(usize, StepResult)> = (0..n).map(|i| {
            (plan.exec_to_decl[i], StepResult {
                step:       plan.step_names[i].clone(),
                oracle:     plan.oracle_names[i].clone(),
                value:      results[i],
                latency_ns: 0.0,
            })
        }).collect();
        srs.sort_by_key(|(idx, _)| *idx);

        Ok(PlanResult {
            plan_name:             plan_name.to_string(),
            inputs:                params.clone(),
            steps:                 srs.into_iter().map(|(_, s)| s).collect(),
            total_ns,
            parallel_groups_count: plan.n_groups,
            cache_hits:            0,
        })
    }

    /// Stack-only execution for <=8 step plans. Zero heap in hot path.
    #[inline]
    pub fn execute_stack(&self, plan_name: &str, param_vals: &[f64])
        -> Result<([f64; 8], usize, f64), String>
    {
        let plan = self.plans.get(plan_name)
            .ok_or_else(|| format!("AOT: plan '{}' not compiled", plan_name))?;

        let t0 = Instant::now();
        let n = plan.exec_steps.len().min(8);
        let mut res = [0.0f64; 8];
        let mut abuf = [0.0f64; 8];

        for (si, step) in plan.exec_steps.iter().enumerate().take(8) {
            let na = step.args.len().min(8);
            for (ai, src) in step.args.iter().enumerate().take(8) {
                abuf[ai] = match *src {
                    ArgSource::Param(j) => param_vals[j],
                    ArgSource::Step(j)  => res[j],
                    ArgSource::Const(v) => v,
                };
            }
            let of = &self.oracle_fns[step.oracle_idx];
            res[si] = if let Some(addr) = of.jit_addr {
                unsafe {
                    let f: JitOracleFn = std::mem::transmute(addr);
                    f(abuf.as_ptr(), na)
                }
            } else {
                (of.builtin)(&abuf[..na])
            };
        }
        Ok((res, n, t0.elapsed().as_nanos() as f64))
    }

    pub fn has_plan(&self, name: &str) -> bool { self.plans.contains_key(name) }
    pub fn plan_count(&self) -> usize { self.plans.len() }
    pub fn oracle_count(&self) -> usize { self.oracle_fns.len() }

    // ======================================================================
    // Level 3: Turbo -- zero-overhead execution
    // ======================================================================

    /// Build turbo index table. Call after compile_plans_jit().
    pub fn build_turbo_table(&mut self) {
        let mut table = Vec::new();
        let mut index = HashMap::new();
        for (name, plan) in &self.plans {
            if let Some(fn_ptr) = plan.jit_plan_fn {
                let idx = table.len();
                table.push(TurboPlan {
                    name: name.clone(),
                    fn_ptr,
                    reg_fn_ptr: None,
                    n_params: plan.param_names.len(),
                    n_steps: plan.n_steps,
                    param_names: plan.param_names.clone(),
                    step_names: plan.step_names.clone(),
                    oracle_names: plan.oracle_names.clone(),
                    exec_to_decl: plan.exec_to_decl.clone(),
                });
                index.insert(name.clone(), idx);
            }
        }
        eprintln!("  AOT Level 3: {} plans in turbo table", table.len());
        self.turbo_table = table;
        self.turbo_index = index;
    }

    /// Level 3 turbo execution -- NO HashMap, NO Instant, NO Vec, NO String.
    #[inline(always)]
    pub fn execute_turbo(&self, turbo_idx: usize, params: &[f64]) -> Option<([f64; 32], usize)> {
        let tp = self.turbo_table.get(turbo_idx)?;
        let mut results = [0.0f64; 32];
        let n = unsafe {
            let f: JitPlanFn = std::mem::transmute(tp.fn_ptr);
            f(params.as_ptr(), params.len(), results.as_mut_ptr())
        };
        Some((results, n as usize))
    }

    /// Resolve plan name to turbo index. Do this ONCE at request parse time.
    #[inline]
    pub fn turbo_index(&self, name: &str) -> Option<usize> {
        self.turbo_index.get(name).copied()
    }

    /// Get turbo plan info by index.
    #[inline]
    pub fn turbo_plan(&self, idx: usize) -> Option<&TurboPlan> {
        self.turbo_table.get(idx)
    }

    /// Inline benchmark: run plan N times, return average ns.
    pub fn bench_turbo(&self, turbo_idx: usize, params: &[f64], iterations: usize) -> Option<f64> {
        let tp = self.turbo_table.get(turbo_idx)?;
        let addr = tp.fn_ptr;
        let mut results = [0.0f64; 32];

        // Warmup
        for _ in 0..100 {
            unsafe {
                let f: JitPlanFn = std::mem::transmute(addr);
                f(params.as_ptr(), params.len(), results.as_mut_ptr());
            }
        }

        // Timed run
        let t0 = std::time::Instant::now();
        for _ in 0..iterations {
            unsafe {
                let f: JitPlanFn = std::mem::transmute(addr);
                std::hint::black_box(f(
                    std::hint::black_box(params.as_ptr()),
                    std::hint::black_box(params.len()),
                    std::hint::black_box(results.as_mut_ptr()),
                ));
            }
        }
        let elapsed = t0.elapsed().as_nanos() as f64;
        Some(elapsed / iterations as f64)
    }


    // ======================================================================
    // Level 4: Register ABI -- params in XMM registers, zero memory loads
    // ======================================================================

    /// Level 4: Compile plans with register-passing ABI.
    /// Params passed directly as f64 in XMM0-XMM7 registers.
    /// For plans with <=8 params, this eliminates ALL memory loads for inputs.
    pub fn compile_plans_register_abi(&mut self, bc: &bytecode::Module) {
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("use_colocated_libcalls", "false");
        let _ = flag_builder.set("is_pic", "false");
        let _ = flag_builder.set("opt_level", "speed");
        let flags = settings::Flags::new(flag_builder);
        let isa = match cranelift_codegen::isa::lookup_by_name("x86_64") {
            Ok(b) => match b.finish(flags) {
                Ok(i) => i,
                Err(e) => { eprintln!("  [AOT L4] isa error: {}", e); return; }
            },
            Err(e) => { eprintln!("  [AOT L4] isa error: {}", e); return; }
        };

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        builder.symbol("fast_pow", fast_pow as *const u8);
        let mut jit_module = JITModule::new(builder);

        let mut func_ids: Vec<(String, cranelift_module::FuncId, usize)> = Vec::new();

        for (turbo_idx, tp) in self.turbo_table.iter().enumerate() {
            if tp.n_params > 8 { continue; }

            let plan = match self.plans.get(&tp.name) {
                Some(p) => p,
                None => continue,
            };

            let sym = format!("crysl_reg_{}", tp.name.replace(['-', '.', ' '], "_"));

            // Signature: (p0: f64, ..., pN-1: f64, results: *mut f64) -> i32
            let mut sig = jit_module.make_signature();
            for _ in 0..tp.n_params {
                sig.params.push(AbiParam::new(F64));
            }
            sig.params.push(AbiParam::new(I64));  // results ptr
            sig.returns.push(AbiParam::new(I32));  // n_results

            let func_id = match jit_module.declare_function(&sym, Linkage::Export, &sig) {
                Ok(id) => id,
                Err(e) => { eprintln!("  [AOT L4] declare {}: {}", tp.name, e); continue; }
            };

            // Declare fast_pow shim
            let mut fp_sig = jit_module.make_signature();
            fp_sig.params.push(AbiParam::new(F64));
            fp_sig.params.push(AbiParam::new(F64));
            fp_sig.returns.push(AbiParam::new(F64));
            let fast_pow_id = match jit_module.declare_function("fast_pow", Linkage::Import, &fp_sig) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let mut ctx = jit_module.make_context();
            ctx.func.signature = sig;
            ctx.func.name = UserFuncName::user(0, func_id.as_u32());

            let compile_ok = {
                let mut func_ctx = FunctionBuilderContext::new();
                let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
                let entry_block = bcx.create_block();
                bcx.append_block_params_for_function_params(entry_block);
                bcx.switch_to_block(entry_block);
                bcx.seal_block(entry_block);

                let block_params: Vec<cranelift_codegen::ir::Value> =
                    (0..tp.n_params + 1).map(|i| bcx.block_params(entry_block)[i]).collect();

                // Params directly from registers -- NO memory loads!
                let mut param_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
                for i in 0..tp.n_params {
                    param_vals.push(block_params[i]);
                }
                let results_ptr = block_params[tp.n_params];

                let fast_pow_ref = jit_module.declare_func_in_func(fast_pow_id, bcx.func);

                let mut step_results: Vec<cranelift_codegen::ir::Value> = Vec::new();
                let mut var_counter: usize = 0;

                for (step_idx, step) in plan.exec_steps.iter().enumerate() {
                    let mut arg_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
                    for src in &step.args {
                        let val = match *src {
                            ArgSource::Param(j) => param_vals[j],
                            ArgSource::Step(j)  => step_results[j],
                            ArgSource::Const(v) => bcx.ins().f64const(v),
                        };
                        arg_vals.push(val);
                    }

                    let oracle_meta = bc.oracles.iter().find(|o| o.name == step.oracle_name);

                    let result_val = if let Some(meta) = oracle_meta {
                        let oracle_end = {
                            let idx = bc.oracles.iter().position(|o| o.entry_ip == meta.entry_ip).unwrap_or(0);
                            bc.oracles.get(idx + 1).map(|o| o.entry_ip).unwrap_or(bc.code.len())
                        };
                        let has_branches = {
                            let mut found = false;
                            let mut scan = meta.entry_ip;
                            while scan < oracle_end {
                                match bc.code[scan].op {
                                    Op::JumpFalse | Op::Jump => { found = true; break; }
                                    Op::Halt => break,
                                    _ => {}
                                }
                                scan += 1;
                            }
                            found
                        };
                        if has_branches {
                            Self::inline_oracle_body_branching(
                                bc, meta.entry_ip, oracle_end, meta.n_params,
                                &arg_vals, &mut bcx, fast_pow_ref, &mut var_counter,
                            )
                        } else {
                            Self::inline_oracle_body(
                                bc, meta.entry_ip, meta.n_params,
                                &arg_vals, &mut bcx, fast_pow_ref,
                            )
                        }
                    } else {
                        let of = &self.oracle_fns[step.oracle_idx];
                        if let Some(addr) = of.jit_addr {
                            Self::emit_indirect_oracle_call(addr, &arg_vals, &mut bcx)
                        } else {
                            Self::emit_indirect_oracle_call(of.builtin as usize, &arg_vals, &mut bcx)
                        }
                    };

                    let offset = (step_idx * 8) as i32;
                    bcx.ins().store(MemFlags::trusted(), result_val, results_ptr, offset);
                    step_results.push(result_val);
                }

                let n_steps = bcx.ins().iconst(I32, plan.exec_steps.len() as i64);
                bcx.ins().return_(&[n_steps]);
                bcx.finalize();
                true
            };

            if !compile_ok { continue; }

            match jit_module.define_function(func_id, &mut ctx) {
                Ok(_) => { func_ids.push((tp.name.clone(), func_id, turbo_idx)); }
                Err(e) => { eprintln!("  [AOT L4] define {}: {}", tp.name, e); }
            }
            jit_module.clear_context(&mut ctx);
        }

        if func_ids.is_empty() {
            eprintln!("  [AOT L4] no plans compiled");
            return;
        }

        if let Err(e) = jit_module.finalize_definitions() {
            eprintln!("  [AOT L4] finalize error: {}", e);
            return;
        }

        let mut count = 0;
        for (_name, func_id, turbo_idx) in &func_ids {
            let raw_ptr = jit_module.get_finalized_function(*func_id);
            if let Some(tp) = self.turbo_table.get_mut(*turbo_idx) {
                tp.reg_fn_ptr = Some(raw_ptr as usize);
                count += 1;
            }
        }

        Box::leak(Box::new(jit_module));
        eprintln!("  AOT Level 4: {} plans compiled with register ABI", count);
    }

    /// Level 4: Execute with register ABI. Params in CPU registers.
    #[inline(always)]
    pub fn execute_register(&self, turbo_idx: usize, params: &[f64]) -> Option<([f64; 32], usize)> {
        let tp = self.turbo_table.get(turbo_idx)?;
        let addr = tp.reg_fn_ptr?;
        if params.len() < tp.n_params { return None; }
        let mut results = [0.0f64; 32];

        let n = match tp.n_params {
            0 => unsafe {
                type F = unsafe extern "C" fn(*mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(results.as_mut_ptr())
            },
            1 => unsafe {
                type F = unsafe extern "C" fn(f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], results.as_mut_ptr())
            },
            2 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], results.as_mut_ptr())
            },
            3 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], params[2], results.as_mut_ptr())
            },
            4 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], params[2], params[3], results.as_mut_ptr())
            },
            5 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], params[2], params[3], params[4], results.as_mut_ptr())
            },
            6 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], params[2], params[3], params[4], params[5], results.as_mut_ptr())
            },
            7 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], params[2], params[3], params[4], params[5], params[6], results.as_mut_ptr())
            },
            8 => unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                f(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], results.as_mut_ptr())
            },
            _ => unsafe {
                // >8 params: fallback to L3 pointer ABI
                type F = unsafe extern "C" fn(*const f64, usize, *mut f64) -> i32;
                let f: F = std::mem::transmute(tp.fn_ptr);
                f(params.as_ptr(), params.len(), results.as_mut_ptr())
            },
        };
        Some((results, n as usize))
    }

    /// Benchmark Level 4 register ABI -- minimal overhead measurement
    pub fn bench_register(&self, turbo_idx: usize, params: &[f64], iterations: usize) -> Option<f64> {
        let tp = self.turbo_table.get(turbo_idx)?;
        let addr = tp.reg_fn_ptr?;
        if params.len() < tp.n_params { return None; }
        let mut results = [0.0f64; 32];
        let rp = results.as_mut_ptr();

        // Warmup
        for _ in 0..100 {
            let _ = self.execute_register(turbo_idx, params);
        }

        // Direct dispatch: resolve fn type ONCE, then tight loop
        // Only black_box the return value to prevent dead code elimination
        // but let the compiler keep params in registers across iterations
        let t0 = std::time::Instant::now();
        match tp.n_params {
            0 => { unsafe {
                type F = unsafe extern "C" fn(*mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(rp)); }
            }},
            1 => { unsafe {
                type F = unsafe extern "C" fn(f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0], rp)); }
            }},
            2 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1], rp)); }
            }},
            3 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1],params[2], rp)); }
            }},
            4 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1],params[2],params[3], rp)); }
            }},
            5 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1],params[2],params[3],params[4], rp)); }
            }},
            6 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1],params[2],params[3],params[4],params[5], rp)); }
            }},
            7 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1],params[2],params[3],params[4],params[5],params[6], rp)); }
            }},
            8 => { unsafe {
                type F = unsafe extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64, *mut f64) -> i32;
                let f: F = std::mem::transmute(addr);
                for _ in 0..iterations { std::hint::black_box(f(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7], rp)); }
            }},
            _ => {
                for _ in 0..iterations {
                    let _ = self.execute_register(turbo_idx, params);
                }
            },
        }
        let elapsed = t0.elapsed().as_nanos() as f64;
        Some(elapsed / iterations as f64)
    }

    /// Get number of turbo plans

    // ======================================================================
    // Level 5: Batch Turbo -- SIMD-style batch execution with prefetch
    // ======================================================================

    /// Execute the same plan with N different parameter sets.
    /// Uses L1 prefetch hints and processes params in groups of 4.
    pub fn batch_execute_turbo(&self, turbo_idx: usize, param_sets: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let tp = match self.turbo_table.get(turbo_idx) {
            Some(t) => t,
            None => return Vec::new(),
        };
        let n = param_sets.len();
        if n == 0 { return Vec::new(); }

        let mut all_results: Vec<Vec<f64>> = Vec::with_capacity(n);

        // Process in groups of 4 for pipeline-style execution
        let chunks = n / 4;
        let remainder = n % 4;

        let f: JitPlanFn = unsafe { std::mem::transmute(tp.fn_ptr) };

        for chunk in 0..chunks {
            let base = chunk * 4;

            // Prefetch next chunk's params while we execute current
            #[cfg(target_arch = "x86_64")]
            {
                let next_base = base + 4;
                if next_base < n {
                    unsafe {
                        use std::arch::x86_64::_mm_prefetch;
                        use std::arch::x86_64::_MM_HINT_T0;
                        _mm_prefetch(
                            param_sets[next_base].as_ptr() as *const i8,
                            _MM_HINT_T0,
                        );
                        if next_base + 1 < n {
                            _mm_prefetch(
                                param_sets[next_base + 1].as_ptr() as *const i8,
                                _MM_HINT_T0,
                            );
                        }
                    }
                }
            }

            // Execute 4 plans in tight sequence
            for i in 0..4 {
                let idx = base + i;
                let params = std::hint::black_box(&param_sets[idx]);
                let mut results = [0.0f64; 32];
                let steps = unsafe {
                    std::hint::black_box(f(
                        std::hint::black_box(params.as_ptr()),
                        std::hint::black_box(params.len()),
                        std::hint::black_box(results.as_mut_ptr()),
                    ))
                } as usize;
                let steps = steps.min(tp.n_steps).min(32);
                all_results.push(std::hint::black_box(results[..steps].to_vec()));
            }
        }

        // Handle remainder
        for i in 0..remainder {
            let idx = chunks * 4 + i;
            let params = std::hint::black_box(&param_sets[idx]);
            let mut results = [0.0f64; 32];
            let steps = unsafe {
                std::hint::black_box(f(
                    std::hint::black_box(params.as_ptr()),
                    std::hint::black_box(params.len()),
                    std::hint::black_box(results.as_mut_ptr()),
                ))
            } as usize;
            let steps = steps.min(tp.n_steps).min(32);
            all_results.push(std::hint::black_box(results[..steps].to_vec()));
        }

        all_results
    }

    /// Get the dependency graph for a plan's steps.
    /// Returns for each step, which param indices it depends on.
    pub fn step_dependencies(&self, plan_name: &str) -> Option<Vec<Vec<usize>>> {
        let plan = self.plans.get(plan_name)?;
        let n_steps = plan.exec_steps.len();

        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n_steps];

        for (si, step) in plan.exec_steps.iter().enumerate() {
            let mut param_deps: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for arg in &step.args {
                match arg {
                    ArgSource::Param(pi) => { param_deps.insert(*pi); }
                    ArgSource::Step(prev_si) => {
                        // Transitive: inherit all param deps from the referenced step
                        if *prev_si < si {
                            for &pd in &deps[*prev_si] {
                                param_deps.insert(pd);
                            }
                        }
                    }
                    ArgSource::Const(_) => {}
                }
            }
            deps[si] = param_deps.into_iter().collect();
            deps[si].sort();
        }

        Some(deps)
    }

    /// Get param names for a plan
    pub fn plan_param_names(&self, plan_name: &str) -> Option<&Vec<String>> {
        self.plans.get(plan_name).map(|p| &p.param_names)
    }

    /// Get step names for a plan (in execution order)
    pub fn plan_step_names(&self, plan_name: &str) -> Option<&Vec<String>> {
        self.plans.get(plan_name).map(|p| &p.step_names)
    }


    pub fn turbo_count(&self) -> usize { self.turbo_table.len() }
}

// ── Arg resolution (compile-time only) ────────────────────────────────

fn resolve_arg(
    expr:  &Expr,
    params: &HashMap<&str, usize>,
    steps:  &HashMap<&str, usize>,
) -> Option<ArgSource> {
    match expr {
        Expr::Float(f) => Some(ArgSource::Const(*f)),
        Expr::Int(i)   => Some(ArgSource::Const(*i as f64)),
        Expr::Ident(name) => {
            if let Some(&i) = params.get(name.as_str()) {
                Some(ArgSource::Param(i))
            } else if let Some(&i) = steps.get(name.as_str()) {
                Some(ArgSource::Step(i))
            } else {
                None
            }
        }
        Expr::Binary(op, lhs, rhs) => {
            let l = resolve_arg(lhs, params, steps)?;
            let r = resolve_arg(rhs, params, steps)?;
            match (l, r) {
                (ArgSource::Const(lv), ArgSource::Const(rv)) => {
                    Some(ArgSource::Const(match op {
                        BinaryOp::Add => lv + rv,
                        BinaryOp::Sub => lv - rv,
                        BinaryOp::Mul => lv * rv,
                        BinaryOp::Div => if rv == 0.0 { 0.0 } else { lv / rv },
                        BinaryOp::Pow => lv.powf(rv),
                        _ => return None,
                    }))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// ── Pow inlining (same as jit.rs) ────────────────────────────────────

fn detect_const_reg(
    bc:         &bytecode::Module,
    current_ip: usize,
    entry_ip:   usize,
    target_reg: u16,
) -> Option<f64> {
    if current_ip == 0 { return None; }
    for scan in (entry_ip..current_ip).rev() {
        let prev = &bc.code[scan];
        if prev.a == target_reg {
            return match prev.op {
                Op::LoadConst => match bc.consts.get(prev.b as usize) {
                    Some(Const::Float(f)) => Some(*f),
                    Some(Const::Int(n))   => Some(*n as f64),
                    Some(Const::Bool(b))  => Some(if *b { 1.0 } else { 0.0 }),
                    _                     => None,
                },
                Op::LoadTrit => Some((prev.b as i16) as f64),
                _ => None,
            };
        }
    }
    None
}

fn inline_pow_const(
    exp_val:      f64,
    base:         cranelift_codegen::ir::Value,
    bcx:          &mut FunctionBuilder,
    fast_pow_ref: cranelift_codegen::ir::FuncRef,
) -> cranelift_codegen::ir::Value {
    const EPS: f64 = 1e-9;

    if (exp_val - 0.5).abs() < EPS {
        bcx.ins().sqrt(base)
    } else if (exp_val - 1.0).abs() < EPS {
        base
    } else if (exp_val - 2.0).abs() < EPS {
        bcx.ins().fmul(base, base)
    } else if (exp_val - 3.0).abs() < EPS {
        let sq = bcx.ins().fmul(base, base);
        bcx.ins().fmul(sq, base)
    } else if (exp_val - 4.0).abs() < EPS {
        let sq = bcx.ins().fmul(base, base);
        bcx.ins().fmul(sq, sq)
    } else if (exp_val - 0.25).abs() < EPS {
        let r = bcx.ins().sqrt(base);
        bcx.ins().sqrt(r)
    } else if (exp_val + 1.0).abs() < EPS {
        let one = bcx.ins().f64const(1.0);
        bcx.ins().fdiv(one, base)
    } else {
        let exp_c = bcx.ins().f64const(exp_val);
        let call  = bcx.ins().call(fast_pow_ref, &[base, exp_c]);
        bcx.inst_results(call)[0]
    }
}

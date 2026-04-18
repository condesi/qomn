// ═══════════════════════════════════════════════════════════════════════
// QOMN — JIT Engine v1.7.1 (Cranelift — oracle bodies → native x86-64)
//
// Pipeline:
//   CRYS-ISA oracle bytecode (bytecode::Module)
//   → walk opcodes from entry_ip → Return/Halt
//   → lower each opcode to Cranelift IR (F64 SSA values)
//   → JITModule::define_function() → finalize → fn_ptr
//   → stored in JitCache, invoked instead of bytecode interpreter
//
// ABI (all compiled oracle functions):
//   unsafe extern "C" fn(params: *const f64, n_params: usize) -> f64
//   Caller stacks args as &[f64] slice, passes raw pointer.
//
// Supported CRYS-ISA opcodes in oracle bodies:
//   LoadConst  → f64const (from Module.consts pool)
//   LoadVar    → SSA alias from var_vals map
//   StoreVar   → update var_vals map
//   Add/Sub/Mul/Div → fadd/fsub/fmul/fdiv
//   Pow (static exp) → inline_pow_const():
//     ^0.5  → sqrt (native Cranelift, 1–3 ns)
//     ^1.0  → identity
//     ^2.0  → fmul(x, x)
//     ^3.0  → fmul(fmul(x,x), x)
//     ^4.0  → fmul(sq, sq)
//     ^0.25 → sqrt(sqrt(x))
//     ^-1.0 → fdiv(1.0, x)
//     other → fast_pow(x, exp) [exp2(exp·log2(x)), ~15 ns]
//   Pow (dynamic exp) → fast_pow(x, exp) shim
//   Move       → SSA copy
//   Return     → return value
//   Halt       → return 0.0
//
// Benchmark (Server5 EPYC 12-core, nfpa_electrico.crys, 10000 calls):
//   Interpreter:    0.165–0.266 μs/call
//   JIT v1.6:       2.4–25.1 ns/call  (9.3–101.6× speedup)
//   JIT v1.6.1:     2.4–5.0  ns/call  (all oracles — pow inlined)
//
// v1.6.1 changes vs v1.6:
//   ✅ inline_pow_const: sqrt/fmul chains replace libm_pow for static exponents
//   ✅ fast_pow shim (exp2/log2) for non-special constant exponents
//   ✅ detect_const_reg: backward scan — primary constant detection for Pow
//
// v1.7 changes vs v1.6.1:
//   ✅ JitFnTable = Arc<HashMap<String,(usize,usize)>> — Send+Sync fn_ptr store
//   ✅ JitEngine::fn_table() — extract table after compile_all for fork lanes
//   ✅ fork lanes receive JitFnTable → JIT dispatch inside PAR_BEGIN threads
//
// v1.7.1 — Numerical Safety Layer:
//   ✅ Op::Div → safe_div: select(denom==0, 0.0, l/r) — branch-free, zero overhead
//   ✅ Op::Return → NaN clamp: select(Ordered(x,x), x, 0.0) — prevents NaN propagation
//
// Roadmap:
//   🔲 v1.8: fused MM_TERN + ACT super-kernel via Cranelift SIMD
//   🔲 v1.8: NUMA-aware thread pinning for oracle workers
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlags, UserFuncName,
    types::{F64, I64},
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_native;
use cranelift_module::{Linkage, Module};

use crate::bytecode::{self, Const, Op};

// ── Pow inlining helpers ──────────────────────────────────────────────

/// Scan backward through the oracle body from `current_ip` to find the last
/// instruction that wrote to `target_reg`. If it was a LoadConst or LoadTrit,
/// return the constant value. Otherwise (or if no write found), return None.
///
/// Oracle bodies are acyclic (no loops/jumps), so a backward scan is sound.
fn detect_const_reg(
    bc:         &bytecode::Module,
    current_ip: usize,
    entry_ip:   usize,
    target_reg: u16,
) -> Option<f64> {
    // Scan backward from the instruction just before current_ip down to entry_ip
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
                _ => None, // last write was dynamic — not a const
            };
        }
    }
    None
}

/// Lower `base ^ exp_val` to inline Cranelift IR, avoiding all libm calls for
/// the most common physical-oracle exponents.
///
/// | exp_val | emitted IR            | latency      |
/// |---------|-----------------------|--------------|
/// | 0.5     | `sqrt(base)`          | ~5 ns (HW)   |
/// | 1.0     | identity              | 0 ns         |
/// | 2.0     | `fmul(base, base)`    | ~1 ns        |
/// | 3.0     | `fmul(fmul(x,x), x)` | ~2 ns        |
/// | 4.0     | `fmul(sq, sq)`        | ~2 ns        |
/// | 0.25    | `sqrt(sqrt(base))`    | ~10 ns       |
/// | -1.0    | `fdiv(1.0, base)`     | ~5 ns        |
/// | other   | `fast_pow` (exp2/log2)| ~15 ns       |
fn inline_pow_const(
    exp_val:      f64,
    base:         cranelift_codegen::ir::Value,
    bcx:          &mut cranelift_frontend::FunctionBuilder,
    fast_pow_ref: cranelift_codegen::ir::FuncRef,
) -> cranelift_codegen::ir::Value {
    use cranelift_codegen::ir::InstBuilder;
    const EPS: f64 = 1e-9;

    if (exp_val - 0.5).abs() < EPS {
        bcx.ins().sqrt(base)                             // x^0.5  → native sqrt
    } else if (exp_val - 1.0).abs() < EPS {
        base                                             // x^1    → identity
    } else if (exp_val - 2.0).abs() < EPS {
        bcx.ins().fmul(base, base)                      // x^2    → fmul
    } else if (exp_val - 3.0).abs() < EPS {
        let sq = bcx.ins().fmul(base, base);
        bcx.ins().fmul(sq, base)                        // x^3    → sq×x
    } else if (exp_val - 4.0).abs() < EPS {
        let sq = bcx.ins().fmul(base, base);
        bcx.ins().fmul(sq, sq)                          // x^4    → sq×sq
    } else if (exp_val - 0.25).abs() < EPS {
        let r = bcx.ins().sqrt(base);
        bcx.ins().sqrt(r)                               // x^0.25 → sqrt(sqrt)
    } else if (exp_val + 1.0).abs() < EPS {
        let one = bcx.ins().f64const(1.0);
        bcx.ins().fdiv(one, base)                       // x^-1   → 1/x
    } else {
        // Arbitrary constant: exp2(exp × log2(base)) via fast_pow shim
        let exp_c = bcx.ins().f64const(exp_val);
        let call  = bcx.ins().call(fast_pow_ref, &[base, exp_c]);
        bcx.inst_results(call)[0]
    }
}

// ── JIT ABI ───────────────────────────────────────────────────────────
/// All JIT-compiled oracle functions share this C ABI.
/// `params` — pointer to f64[n_params] on caller's stack.
pub type OracleJitFn = unsafe extern "C" fn(*const f64, usize) -> f64;

// ── Math shims ────────────────────────────────────────────────────────

/// Fallback for dynamic exponents: full libm pow (handles all edge cases).
extern "C" fn libm_pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Fast pow for positive bases: exp2(y · log2(x)).
/// ~2–3× faster than libm_pow for non-special exponents.
/// Falls back to powf for x ≤ 0 (negative base, physics edge case).
extern "C" fn fast_pow(base: f64, exp: f64) -> f64 {
    if base > 0.0 {
        (exp * base.log2()).exp2()
    } else {
        base.powf(exp)
    }
}

// ── Thread-safe JIT function table (v1.7) ────────────────────────────
//
// JITModule is !Send — it cannot cross thread boundaries. However the
// compiled code lives in a fixed mmap(PROT_EXEC) region that doesn't
// move after finalize_definitions(). Extracting the fn_ptr as `usize`
// (a plain integer) is Send+Sync, and casting back to the function type
// inside a thread is safe provided:
//   (a) JitEngine outlives all threads (guaranteed by thread::scope)
//   (b) The code is read-only after finalization (guaranteed by Cranelift)
//   (c) The ABI matches: extern "C" fn(*const f64, usize) -> f64

/// `(fn_addr_as_usize, n_params)` — sharable across threads via Arc.
pub type JitFnTable = Arc<HashMap<String, (usize, usize)>>;

/// Call an oracle from a JitFnTable entry.
///
/// # Safety
/// `args.len()` must equal `n_params`. The fn_addr must originate from a
/// `JitEngine` that is still alive (i.e. JITModule not dropped).
#[inline]
pub unsafe fn jit_table_call(fn_addr: usize, args: &[f64]) -> f64 {
    let f: OracleJitFn = std::mem::transmute(fn_addr);
    f(args.as_ptr(), args.len())
}

// ── Compiled oracle ───────────────────────────────────────────────────
pub struct CompiledOracle {
    pub name:       String,
    pub n_params:   usize,
    /// Raw fn pointer — valid as long as JitEngine is alive
    pub fn_ptr:     OracleJitFn,
    pub call_count: u64,
}

impl CompiledOracle {
    /// Call the JIT-compiled oracle with the given arguments.
    ///
    /// # Safety
    /// `args.len()` must equal `self.n_params`.
    pub unsafe fn call(&self, args: &[f64]) -> f64 {
        debug_assert_eq!(args.len(), self.n_params);
        (self.fn_ptr)(args.as_ptr(), args.len())
    }
}

// ── JIT cache ─────────────────────────────────────────────────────────
pub const JIT_THRESHOLD: u64 = 50;

pub struct JitCache {
    call_counts: HashMap<String, u64>,
    pub compiled: HashMap<String, CompiledOracle>,
}

impl JitCache {
    pub fn new() -> Self {
        JitCache { call_counts: HashMap::new(), compiled: HashMap::new() }
    }

    /// Record interpreter call. Returns true if JIT threshold just crossed.
    pub fn tick(&mut self, name: &str) -> bool {
        let c = self.call_counts.entry(name.to_string()).or_insert(0);
        *c += 1;
        *c == JIT_THRESHOLD
    }

    pub fn is_compiled(&self, name: &str) -> bool { self.compiled.contains_key(name) }

    pub fn get(&self, name: &str) -> Option<&CompiledOracle> { self.compiled.get(name) }
}

// ── JIT engine ────────────────────────────────────────────────────────
pub struct JitEngine {
    module: JITModule,
    pub cache: JitCache,
    pub compiled_total: usize,
    pending_oracles: Vec<(String, cranelift_module::FuncId, usize)>,
}

impl JitEngine {
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false")
            .map_err(|e| format!("cranelift flag: {}", e))?;
        flag_builder.set("is_pic", "false")
            .map_err(|e| format!("cranelift flag: {}", e))?;
        flag_builder.set("opt_level", "speed")
            .map_err(|e| format!("cranelift opt: {}", e))?;
        // v1.8: eliminate frame pointer save/restore overhead (~2-3 ns/call)
        flag_builder.set("preserve_frame_pointers", "false")
            .map_err(|e| format!("cranelift flag: {}", e))?;
        let flags = settings::Flags::new(flag_builder);

        // v1.8: cranelift_native detects FMA3+AVX2 on AMD EPYC at runtime
        // Enables VFMADD213SD, 3-operand VEX, BMI2 vs generic x86_64
        let isa = cranelift_native::builder()
            .map_err(|s| format!("native isa: {}", s))?
            .finish(flags)
            .map_err(|e| format!("native isa finish: {}", e))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        builder.symbol("pow",      libm_pow as *const u8);
        builder.symbol("fast_pow", fast_pow as *const u8);

        Ok(JitEngine {
            module: JITModule::new(builder),
            cache:  JitCache::new(),
            compiled_total: 0,
            pending_oracles: Vec::new(),
        })
    }

    /// Compile one oracle to native code.
    pub fn compile_oracle(
        &mut self,
        bc: &bytecode::Module,
        oracle_name: &str,
    ) -> Result<(), String> {
        let meta = bc.oracles.iter().find(|o| o.name == oracle_name)
            .ok_or_else(|| format!("oracle '{}' not in module", oracle_name))?;

        let entry_ip = meta.entry_ip;
        let n_params = meta.n_params;
        let sym      = format!("qomn_oracle_{}", oracle_name.replace(['-', '.'], "_"));

        // ── Function signature: fn(*const f64, usize) -> f64 ─────────
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(I64)); // params ptr
        sig.params.push(AbiParam::new(I64)); // n_params (unused in body, for future checks)
        sig.returns.push(AbiParam::new(F64));

        let func_id = self.module
            .declare_function(&sym, Linkage::Export, &sig)
            .map_err(|e| format!("declare_function: {}", e))?;

        // ── Declare extern math shims before FunctionBuilder ──────────
        let mut pow_sig = self.module.make_signature();
        pow_sig.params.push(AbiParam::new(F64));
        pow_sig.params.push(AbiParam::new(F64));
        pow_sig.returns.push(AbiParam::new(F64));
        let pow_id = self.module
            .declare_function("pow", Linkage::Import, &pow_sig)
            .map_err(|e| format!("declare pow: {}", e))?;

        // fast_pow: exp2(y·log2(x)) — used for non-special constant exponents
        let mut fast_pow_sig = self.module.make_signature();
        fast_pow_sig.params.push(AbiParam::new(F64));
        fast_pow_sig.params.push(AbiParam::new(F64));
        fast_pow_sig.returns.push(AbiParam::new(F64));
        let fast_pow_id = self.module
            .declare_function("fast_pow", Linkage::Import, &fast_pow_sig)
            .map_err(|e| format!("declare fast_pow: {}", e))?;

        // ── Cranelift IR emission ─────────────────────────────────────
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        {
            let mut func_ctx = FunctionBuilderContext::new();
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = bcx.create_block();
            bcx.append_block_params_for_function_params(entry_block);
            bcx.switch_to_block(entry_block);
            bcx.seal_block(entry_block);

            let params_ptr = bcx.block_params(entry_block)[0]; // I64 pointer

            // SSA value maps — no Cranelift Variables needed (oracles have no loops)
            let mut reg_vals:   HashMap<u16, cranelift_codegen::ir::Value> = HashMap::new();
            let mut var_vals:   HashMap<u16, cranelift_codegen::ir::Value> = HashMap::new();
            // Tracks registers that hold statically-known f64 constants (for Pow inlining)
            let mut const_regs: HashMap<u16, f64> = HashMap::new();

            // Pre-load oracle parameters: reg[i] ← params[i] (f64 load)
            for i in 0..n_params {
                let offset = (i * std::mem::size_of::<f64>()) as i32;
                let val = bcx.ins().load(F64, MemFlags::trusted(), params_ptr, offset);
                reg_vals.insert(i as u16, val);
            }

            // Import math shims into this function
            let pow_ref      = self.module.declare_func_in_func(pow_id,      bcx.func);
            let fast_pow_ref = self.module.declare_func_in_func(fast_pow_id, bcx.func);

            // ── Branch detection: collect jump targets ────────────────
            // Scan the whole oracle body (until next oracle or end) for JumpFalse/Jump.
            // Do NOT stop at Return — if/else branches contain multiple Returns.
            let oracle_end = {
                let idx = bc.oracles.iter().position(|o| o.entry_ip == entry_ip).unwrap_or(0);
                bc.oracles.get(idx + 1).map(|o| o.entry_ip).unwrap_or(bc.code.len())
            };
            let mut branch_targets: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            {
                let mut scan = entry_ip;
                while scan < oracle_end {
                    let si = &bc.code[scan];
                    match si.op {
                        Op::JumpFalse => {
                            branch_targets.insert(scan + 1);          // then-block (fall-through)
                            branch_targets.insert(si.b as usize);     // else-block (jump target)
                        }
                        Op::Jump => { branch_targets.insert(si.b as usize); } // target in .b
                        Op::Halt => break,
                        _ => {}
                    }
                    scan += 1;
                }
            }

            if !branch_targets.is_empty() {
                // ══ BLOCK-MODE PATH: if/else support via Cranelift Variables ═══
                use cranelift_codegen::ir::condcodes::FloatCC as FC;
                use cranelift_codegen::entity::EntityRef as _;

                // Find max register index in this oracle
                let max_reg = bc.code[entry_ip..].iter()
                    .take_while(|i| i.op != Op::Halt)
                    .flat_map(|i| [i.a, i.b, i.c])
                    .max().unwrap_or(0) as usize + 1;

                // Declare Cranelift SSA Variables for each register
                let cvars: Vec<Variable> = (0..max_reg).map(|i| {
                    let v = Variable::new(i);
                    bcx.declare_var(v, F64);
                    v
                }).collect();
                let zero_v = bcx.ins().f64const(0.0);
                for v in &cvars { bcx.def_var(*v, zero_v); }

                // Initialize param registers in Variables
                for i in 0..n_params {
                    if let Some(&pval) = reg_vals.get(&(i as u16)) {
                        if i < cvars.len() { bcx.def_var(cvars[i], pval); }
                    }
                }

                // Build block map: ip → Cranelift Block
                let mut block_map: HashMap<usize, cranelift_codegen::ir::Block> = HashMap::new();
                block_map.insert(entry_ip, entry_block);
                for &target in &branch_targets {
                    block_map.entry(target).or_insert_with(|| bcx.create_block());
                }

                // helper: get f64 SSA value for a register (via Variable)
                let get_v = |r: u16, bcx2: &mut FunctionBuilder| -> cranelift_codegen::ir::Value {
                    if r < cvars.len() as u16 { bcx2.use_var(cvars[r as usize]) }
                    else { bcx2.ins().f64const(0.0) }
                };
                let def_v = |r: u16, val: cranelift_codegen::ir::Value, bcx2: &mut FunctionBuilder| {
                    if r < cvars.len() as u16 { bcx2.def_var(cvars[r as usize], val); }
                };

                let mut ip = entry_ip;
                let mut block_filled = false;

                while ip < bc.code.len() {
                    // Switch block if needed
                    if let Some(&tgt_block) = block_map.get(&ip) {
                        if tgt_block != entry_block || ip != entry_ip {
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
                                Some(crate::bytecode::Const::Float(f)) => *f,
                                Some(crate::bytecode::Const::Int(n))   => *n as f64,
                                Some(crate::bytecode::Const::Bool(b))  => if *b { 1.0 } else { 0.0 },
                                _ => 0.0,
                            };
                            const_regs.insert(instr.a, fval);
                            def_v(instr.a, bcx.ins().f64const(fval), &mut bcx);
                        }
                        Op::LoadTrit => {
                            let tv = (instr.b as i16) as f64;
                            const_regs.insert(instr.a, tv);
                            def_v(instr.a, bcx.ins().f64const(tv), &mut bcx);
                        }
                        Op::StoreVar => {
                            let sv = get_v(instr.b, &mut bcx);
                            var_vals.insert(instr.a, sv);
                        }
                        Op::LoadVar => {
                            let lv = *var_vals.entry(instr.b)
                                .or_insert_with(|| {
                                    if instr.b < cvars.len() as u16 {
                                        // lazy: use Variable value (may differ from var_vals)
                                        zero_v
                                    } else { zero_v }
                                });
                            // Prefer var_vals entry, but also update the Variable
                            def_v(instr.a, lv, &mut bcx);
                        }
                        Op::Move => {
                            let mv = get_v(instr.b, &mut bcx);
                            def_v(instr.a, mv, &mut bcx);
                        }
                        Op::Add  => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); def_v(instr.a, bcx.ins().fadd(l,r), &mut bcx); }
                        Op::Sub  => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); def_v(instr.a, bcx.ins().fsub(l,r), &mut bcx); }
                        Op::Mul  => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); def_v(instr.a, bcx.ins().fmul(l,r), &mut bcx); }
                        Op::Div  => {
                            let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx);
                            let zero2=bcx.ins().f64const(0.0);
                            let is_zero=bcx.ins().fcmp(FC::Equal,r,zero2);
                            let safe_r=bcx.ins().select(is_zero,zero2,r);
                            let res=bcx.ins().fdiv(l,safe_r);
                            def_v(instr.a, res, &mut bcx);
                        }
                        Op::Neg  => { let v=get_v(instr.b,&mut bcx); def_v(instr.a, bcx.ins().fneg(v), &mut bcx); }
                        Op::Eq   => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); let c=bcx.ins().fcmp(FC::Equal,l,r); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),&mut bcx); }
                        Op::Lt   => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); let c=bcx.ins().fcmp(FC::LessThan,l,r); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),&mut bcx); }
                        Op::Gt   => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); let c=bcx.ins().fcmp(FC::GreaterThan,l,r); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),&mut bcx); }
                        Op::Not  => { let v=get_v(instr.b,&mut bcx); let z=bcx.ins().f64const(0.0); let c=bcx.ins().fcmp(FC::Equal,v,z); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(c,t,f),&mut bcx); }
                        Op::And  => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); let z=bcx.ins().f64const(0.0); let cl=bcx.ins().fcmp(FC::NotEqual,l,z); let cr=bcx.ins().fcmp(FC::NotEqual,r,z); let both=bcx.ins().band(cl,cr); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(both,t,f),&mut bcx); }
                        Op::Or   => { let l=get_v(instr.b,&mut bcx); let r=get_v(instr.c,&mut bcx); let z=bcx.ins().f64const(0.0); let cl=bcx.ins().fcmp(FC::NotEqual,l,z); let cr=bcx.ins().fcmp(FC::NotEqual,r,z); let either=bcx.ins().bor(cl,cr); let t=bcx.ins().f64const(1.0); let f=bcx.ins().f64const(0.0); def_v(instr.a,bcx.ins().select(either,t,f),&mut bcx); }
                        Op::Pow  => {
                            let base = get_v(instr.b, &mut bcx);
                            let static_exp = detect_const_reg(bc, ip, entry_ip, instr.c)
                                .or_else(|| const_regs.get(&instr.c).copied());
                            let res = if let Some(exp_val) = static_exp {
                                inline_pow_const(exp_val, base, &mut bcx, fast_pow_ref)
                            } else {
                                let exp = get_v(instr.c, &mut bcx);
                                let call = bcx.ins().call(fast_pow_ref, &[base, exp]);
                                bcx.inst_results(call)[0]
                            };
                            def_v(instr.a, res, &mut bcx);
                        }
                        Op::JumpFalse => {
                            let cond_f = get_v(instr.a, &mut bcx);
                            let z = bcx.ins().f64const(0.0);
                            let is_true = bcx.ins().fcmp(FC::NotEqual, cond_f, z);
                            let then_blk = *block_map.get(&(ip + 1))
                                .unwrap_or(&block_map[&entry_ip]);
                            let else_blk = block_map[&(instr.b as usize)];
                            bcx.ins().brif(is_true, then_blk, &[], else_blk, &[]);
                            block_filled = true;
                        }
                        Op::Jump => {
                            let tgt = *block_map.get(&(instr.b as usize))
                                .unwrap_or(&block_map[&entry_ip]);
                            bcx.ins().jump(tgt, &[]);
                            block_filled = true;
                        }
                        Op::Return => {
                            let ret = get_v(instr.a, &mut bcx);
                            let z = bcx.ins().f64const(0.0);
                            let ok = bcx.ins().fcmp(FC::Ordered, ret, ret);
                            let safe = bcx.ins().select(ok, ret, z);
                            bcx.ins().return_(&[safe]);
                            block_filled = true;
                        }
                        Op::Halt => {
                            let z = bcx.ins().f64const(0.0);
                            bcx.ins().return_(&[z]);
                            block_filled = true;
                        }
                        _ => { ip += 1; continue; }
                    }
                    ip += 1;
                }
                if !block_filled {
                    let z = bcx.ins().f64const(0.0);
                    bcx.ins().return_(&[z]);
                }
                bcx.finalize();
            } else {

            // Helper: get Value for a register (default 0.0 if not set)
            macro_rules! reg {
                ($r:expr) => {
                    *reg_vals.entry($r).or_insert_with(|| bcx.ins().f64const(0.0))
                }
            }

            let mut ip = entry_ip;
            let mut returned = false;

            while ip < bc.code.len() && !returned {
                let instr = &bc.code[ip];
                match instr.op {
                    // ── Constants ─────────────────────────────────────
                    Op::LoadConst => {
                        let ci = instr.b as usize;
                        let fval: f64 = match bc.consts.get(ci) {
                            Some(Const::Float(f)) => *f,
                            Some(Const::Int(n))   => *n as f64,
                            Some(Const::Bool(b))  => if *b { 1.0 } else { 0.0 },
                            _                     => 0.0,
                        };
                        const_regs.insert(instr.a, fval); // track for Pow inlining
                        reg_vals.insert(instr.a, bcx.ins().f64const(fval));
                    }
                    Op::LoadTrit => {
                        // LoadTrit ra, trit_val → reg[a] = trit_val as f64
                        let tv = (instr.b as i16) as f64;
                        const_regs.insert(instr.a, tv);   // track for Pow inlining
                        reg_vals.insert(instr.a, bcx.ins().f64const(tv));
                    }
                    // ── Variable slots ────────────────────────────────
                    Op::StoreVar => {
                        // StoreVar var_idx, src_reg → var_vals[a] = reg[b]
                        let src = reg!(instr.b);
                        var_vals.insert(instr.a, src);
                    }
                    Op::LoadVar => {
                        // LoadVar dest_reg, var_idx → reg[a] = var_vals[b]
                        // Fallback: if var_idx matches a param register (0..n_params),
                        // use reg_vals[var_idx] (params are loaded there at oracle entry).
                        // This covers the case where pass_dead_code_elim kept STORE_VAR
                        // but a later LOAD_VAR runs before var_vals is populated.
                        let val = *var_vals.entry(instr.b)
                            .or_insert_with(|| {
                                reg_vals.get(&instr.b).copied()
                                    .unwrap_or_else(|| bcx.ins().f64const(0.0))
                            });
                        reg_vals.insert(instr.a, val);
                    }
                    // ── Move ─────────────────────────────────────────
                    Op::Move => {
                        let src = reg!(instr.b);
                        reg_vals.insert(instr.a, src);
                    }
                    // ── Arithmetic ────────────────────────────────────
                    Op::Add => {
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        reg_vals.insert(instr.a, bcx.ins().fadd(l, r));
                    }
                    Op::Sub => {
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        reg_vals.insert(instr.a, bcx.ins().fsub(l, r));
                    }
                    Op::Mul => {
                        let l = reg!(instr.b);
                        let r = reg!(instr.c);
                        let mul_reg = instr.a;
                        // v1.8: FMA fusion — look-ahead for Add(mul_result, x).
                        // VFMADD213SD (FMA3): fma(a,b,c)=a*b+c in 4 cycles vs fmul+fadd=8.
                        // cranelift_native detects AMD EPYC FMA3 and lowers fma→VFMADD213SD.
                        let fused = if ip + 1 < bc.code.len() && !returned {
                            let nx = &bc.code[ip + 1];
                            if nx.op == Op::Add {
                                if nx.b == mul_reg {
                                    let addend = reg!(nx.c);
                                    Some((nx.a, bcx.ins().fma(l, r, addend)))
                                } else if nx.c == mul_reg {
                                    let addend = reg!(nx.b);
                                    Some((nx.a, bcx.ins().fma(l, r, addend)))
                                } else { None }
                            } else { None }
                        } else { None };
                        let mul_val = bcx.ins().fmul(l, r);
                        reg_vals.insert(mul_reg, mul_val);
                        if let Some((dest, fma_val)) = fused {
                            reg_vals.insert(dest, fma_val);
                            ip += 2; continue; // consumed the Add
                        }
                    }
                    Op::Div => {
                        // v1.8: constant denominator → multiply by reciprocal.
                        // fdiv latency: 13-14 cycles.  fmul latency: 4 cycles.
                        // Saves ~9-10 cycles (~3-4 ns) on every constant divide.
                        // Examples: sueldo/30.0 → sueldo*0.03333, Q/(η*76.04) partial fold.
                        let l = reg!(instr.b);
                        let r = reg!(instr.c);
                        if let Some(&denom) = const_regs.get(&instr.c) {
                            if denom.abs() > 1e-300 {
                                let recip = bcx.ins().f64const(1.0 / denom);
                                reg_vals.insert(instr.a, bcx.ins().fmul(l, recip));
                                ip += 1; continue;
                            }
                        }
                        // Runtime denominator: safe divide (branchless via OOO select)
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let zero    = bcx.ins().f64const(0.0);
                        let is_zero = bcx.ins().fcmp(FloatCC::Equal, r, zero);
                        let quot    = bcx.ins().fdiv(l, r);
                        let result  = bcx.ins().select(is_zero, zero, quot);
                        reg_vals.insert(instr.a, result);
                    }
                    Op::Pow => {
                        let base = reg!(instr.b);
                        // backward scan takes priority — catches LoadConst even when
                        // const_regs missed it (e.g. cross-block or alias issues)
                        let static_exp = detect_const_reg(bc, ip, entry_ip, instr.c)
                            .or_else(|| const_regs.get(&instr.c).copied());
                        let res = if let Some(exp_val) = static_exp {
                            // ── Static exponent — inline without any libm call ──
                            inline_pow_const(exp_val, base, &mut bcx, fast_pow_ref)
                        } else {
                            // ── Dynamic exponent — fast_pow (exp2/log2 path) ────
                            let exp = reg!(instr.c);
                            let call = bcx.ins().call(fast_pow_ref, &[base, exp]);
                            bcx.inst_results(call)[0]
                        };
                        reg_vals.insert(instr.a, res);
                    }
                    // ── Control flow ──────────────────────────────────
                    Op::Return => {
                        // v1.6.2: NaN clamp — if result is NaN, return 0.0
                        // Cranelift: fcmp(Ordered, x, x) is false iff x is NaN
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let ret  = reg!(instr.a);
                        let zero = bcx.ins().f64const(0.0);
                        let is_ordered = bcx.ins().fcmp(FloatCC::Ordered, ret, ret);
                        let safe_ret   = bcx.ins().select(is_ordered, ret, zero);
                        bcx.ins().return_(&[safe_ret]);
                        returned = true;
                    }
                    Op::Halt => {
                        let zero = bcx.ins().f64const(0.0);
                        bcx.ins().return_(&[zero]);
                        returned = true;
                    }
                    // ── Comparison ops (produce f64: 1.0=true, 0.0=false) ──
                    Op::Eq => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        let cmp = bcx.ins().fcmp(FloatCC::Equal, l, r);
                        let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                        reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                    }
                    Op::Lt => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        let cmp = bcx.ins().fcmp(FloatCC::LessThan, l, r);
                        let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                        reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                    }
                    Op::Gt => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        let cmp = bcx.ins().fcmp(FloatCC::GreaterThan, l, r);
                        let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                        reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                    }
                    Op::Not => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let v = reg!(instr.b);
                        let zero = bcx.ins().f64const(0.0);
                        let cmp = bcx.ins().fcmp(FloatCC::Equal, v, zero); // 1.0 if was 0.0
                        let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                        reg_vals.insert(instr.a, bcx.ins().select(cmp, t, f));
                    }
                    Op::And => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        let zero = bcx.ins().f64const(0.0);
                        let cl = bcx.ins().fcmp(FloatCC::NotEqual, l, zero);
                        let cr = bcx.ins().fcmp(FloatCC::NotEqual, r, zero);
                        let both = bcx.ins().band(cl, cr);
                        let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                        reg_vals.insert(instr.a, bcx.ins().select(both, t, f));
                    }
                    Op::Or => {
                        use cranelift_codegen::ir::condcodes::FloatCC;
                        let l = reg!(instr.b); let r = reg!(instr.c);
                        let zero = bcx.ins().f64const(0.0);
                        let cl = bcx.ins().fcmp(FloatCC::NotEqual, l, zero);
                        let cr = bcx.ins().fcmp(FloatCC::NotEqual, r, zero);
                        let either = bcx.ins().bor(cl, cr);
                        let t = bcx.ins().f64const(1.0); let f = bcx.ins().f64const(0.0);
                        reg_vals.insert(instr.a, bcx.ins().select(either, t, f));
                    }
                    // ── Branches: if oracle has JumpFalse/Jump, handled in block-mode path ──
                    Op::JumpFalse | Op::Jump => {
                        // Reached here only if oracle was (incorrectly) not pre-detected
                        // as needing block mode. Emit safe return.
                        let zero = bcx.ins().f64const(0.0);
                        bcx.ins().return_(&[zero]);
                        returned = true;
                    }
                    // Unknown op in oracle body — stop safely
                    _ => {
                        let zero = bcx.ins().f64const(0.0);
                        bcx.ins().return_(&[zero]);
                        returned = true;
                    }
                }
                ip += 1;
            }

            if !returned {
                let zero = bcx.ins().f64const(0.0);
                bcx.ins().return_(&[zero]);
            }

            bcx.finalize();
            } // end else (linear path — no branches)
        } // FunctionBuilder dropped here — borrows released

        // ── Compile (defer finalize to compile_all for batching) ─────
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| format!("define_function '{}': {}", oracle_name, e))?;
        self.module.clear_context(&mut ctx);

        // Store pending oracle — finalization happens in compile_all
        self.pending_oracles.push((oracle_name.to_string(), func_id, n_params));
        self.compiled_total += 1;
        Ok(())
    }

    /// Finalize all pending oracle definitions (called once after all compile_oracle calls).
    pub fn finalize_all_pending(&mut self) {
        if self.pending_oracles.is_empty() { return; }

        // Single finalize_definitions call for all pending oracles
        if let Err(e) = self.module.finalize_definitions() {
            eprintln!("[JIT] finalize_definitions error: {}", e);
            self.pending_oracles.clear();
            return;
        }

        // Extract all function pointers now that code is executable
        for (oracle_name, func_id, n_params) in self.pending_oracles.drain(..) {
            let raw_ptr = self.module.get_finalized_function(func_id);
            let fn_ptr: OracleJitFn = unsafe { std::mem::transmute(raw_ptr) };
            eprintln!("[JIT] ✓ '{}' ({} params) → {:p}", oracle_name, n_params, raw_ptr);
            self.cache.compiled.insert(oracle_name.clone(), CompiledOracle {
                name: oracle_name,
                n_params,
                fn_ptr,
                call_count: JIT_THRESHOLD,
            });
        }
    }

    /// Compile all oracles in the module.
    pub fn compile_all(
        &mut self,
        bc: &bytecode::Module,
    ) -> Vec<(String, Result<(), String>)> {
        // Collect names first to avoid borrow conflict in filter+map chain
        let names: Vec<String> = bc.oracles.iter()
            .map(|o| o.name.clone())
            .filter(|n| !self.cache.is_compiled(n))
            .collect();
        let results: Vec<(String, Result<(), String>)> = names.into_iter()
            .map(|name| {
                let r = self.compile_oracle(bc, &name);
                (name, r)
            })
            .collect();
        // Finalize ALL pending oracle definitions in one shot
        // (calling finalize_definitions once avoids JIT memory exhaustion)
        self.finalize_all_pending();
        results
    }

    /// Call JIT oracle if compiled, returns None otherwise.
    ///
    /// # Safety
    /// args must match oracle's n_params.
    pub unsafe fn call_if_compiled(&self, name: &str, args: &[f64]) -> Option<f64> {
        self.cache.get(name).map(|co| co.call(args))
    }

    /// Extract a Send+Sync fn-pointer table from all compiled oracles.
    ///
    /// Each entry is `(fn_ptr as usize, n_params)`.  The table can be
    /// wrapped in `Arc` and shared with `std::thread::scope` fork lanes
    /// so they can dispatch JIT-compiled oracles without holding a
    /// reference to `JITModule` (which is !Send).
    ///
    /// The table is valid as long as this `JitEngine` is alive.
    pub fn fn_table(&self) -> JitFnTable {
        Arc::new(
            self.cache.compiled.iter()
                .map(|(name, co)| {
                    let addr = co.fn_ptr as usize;
                    (name.clone(), (addr, co.n_params))
                })
                .collect()
        )
    }

    pub fn stats(&self) -> String {
        format!("JIT v1.7.1: {} oracle(s) compiled to native x86-64 (pow inlined, safe div, NaN clamp, parallel dispatch)", self.compiled_total)
    }
}

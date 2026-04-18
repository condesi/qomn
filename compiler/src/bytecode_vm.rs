// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v1.6 — Bytecode VM (wired to Runtime + Backend CPU + JIT)
//
// Arquitectura:
//   dispatch loop plano (cero recursión AST)
//   ↕ crystal cache (mmap lazy load, L1_PIN / STREAM / PREFETCH)
//   ↕ AsyncOracleEngine (ORACLE_CALL → ticket, ORACLE_WAIT → result)
//   ↕ backend_cpu::tgemv_ternary (AVX2 sign-blend f32, v1.5)
//   ↕ MemoryPool (zero-copy buffers)
//   ↕ Profiler (ns per opcode)
//   ↕ PAR_BEGIN → std::thread::scope (real parallel fork lanes, v1.5)
//   ↕ JitEngine (Cranelift JIT — oracle bodies → native x86-64, v1.6)
//      threshold=50 interpreter calls → auto-compile → native dispatch
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use crate::bytecode::{Module, Op, Const, CrysLoadMode, ActFn};
use crate::backend_cpu::{tgemv_ternary, apply_activation, unpack_2bit, ActFunc};
use crate::runtime::{CrysRuntime, OracleTicket};
use crate::jit::{JitEngine, JitFnTable, jit_table_call};

// ── Runtime Value ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub enum BVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Trit(i8),
    Tvec(Vec<i8>),
    Fvec(Vec<f32>),
    /// Oracle async ticket handle
    Ticket(OracleTicket),
    #[default]
    Null,
}

impl BVal {
    pub fn as_f64(&self) -> f64 {
        match self {
            BVal::Float(f) => *f,
            BVal::Int(n)   => *n as f64,
            BVal::Bool(b)  => if *b { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }
    fn as_bool(&self) -> bool {
        match self {
            BVal::Bool(b)  => *b,
            BVal::Int(n)   => *n != 0,
            BVal::Float(f) => f.abs() > 1e-12,
            _ => false,
        }
    }
    fn as_trit(&self) -> i8 {
        match self {
            BVal::Trit(t)  => *t,
            BVal::Int(n)   => n.signum() as i8,
            BVal::Float(f) => if *f > 1e-9 { 1 } else if *f < -1e-9 { -1 } else { 0 },
            _ => 0,
        }
    }
    fn as_fvec(&self) -> Option<&Vec<f32>> {
        if let BVal::Fvec(v) = self { Some(v) } else { None }
    }
}

impl std::fmt::Display for BVal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BVal::Int(n)    => write!(f, "{}", n),
            BVal::Float(v)  => write!(f, "{:.6}", v),
            BVal::Bool(b)   => write!(f, "{}", b),
            BVal::Str(s)    => write!(f, "{}", s),
            BVal::Trit(t)   => write!(f, "trit({})", t),
            BVal::Tvec(v)   => write!(f, "tvec[{}..len={}]",
                v.iter().take(4).map(|x| x.to_string()).collect::<Vec<_>>().join(","), v.len()),
            BVal::Fvec(v)   => write!(f, "fvec[{:.4}..len={}]",
                v.first().unwrap_or(&0.0), v.len()),
            BVal::Ticket(t) => write!(f, "ticket({})", t),
            BVal::Null      => write!(f, "null"),
        }
    }
}

// ── Call Frame ────────────────────────────────────────────────────────

struct Frame {
    ret_ip:     usize,
    saved_regs: Box<[BVal; 256]>,
    result_reg: u16,
}

// ── Oracle closure registry ───────────────────────────────────────────
// Evaluated oracle functions for async dispatch.
// Built from the Module at VM init time.

type OracleFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static>;

// ── Bytecode VM ───────────────────────────────────────────────────────

pub struct BytecodeVm {
    vars:           Vec<BVal>,
    output:         Vec<String>,
    oracle_entries: Vec<usize>,
    /// Cached oracle functions for async dispatch
    oracle_fns:     Vec<Option<OracleFn>>,
    /// JIT engine — auto-compiles hot oracles after JIT_THRESHOLD calls
    jit:            Option<JitEngine>,
}

impl BytecodeVm {
    pub fn new() -> Self {
        Self {
            vars:           Vec::new(),
            output:         Vec::new(),
            oracle_entries: Vec::new(),
            oracle_fns:     Vec::new(),
            jit:            None,
        }
    }

    /// Attach a pre-initialized JIT engine (e.g. from `jit::JitEngine::new()`).
    /// Already-compiled oracles in the engine's cache are used immediately.
    pub fn with_jit(mut self, engine: JitEngine) -> Self {
        self.jit = Some(engine);
        self
    }

    pub fn run(
        &mut self,
        module:  &Module,
        runtime: &mut CrysRuntime,
    ) -> Result<Vec<String>, String> {
        self.output.clear();
        self.vars.resize(module.vars.len().max(64), BVal::Null);
        self.oracle_entries = module.oracles.iter().map(|o| o.entry_ip).collect();

        // Register file
        let mut regs: Box<[BVal; 256]> =
            Box::new(std::array::from_fn(|_| BVal::Null));
        let mut ip = 0usize;
        let mut call_stack: Vec<Frame> = Vec::new();
        let code_len = module.code.len();

        while ip < code_len {
            let instr = module.code[ip];
            ip += 1;

            let t0 = std::time::Instant::now();

            match instr.op {

                // ── Data movement ──────────────────────────────────
                Op::LoadConst => {
                    regs[instr.a as usize] =
                        const_to_bval(module.consts.get(instr.b as usize));
                }
                Op::LoadTrit => {
                    regs[instr.a as usize] = BVal::Trit(instr.b as i16 as i8);
                }
                Op::Move => {
                    let v = regs[instr.b as usize].clone();
                    regs[instr.a as usize] = v;
                }
                Op::LoadVar => {
                    regs[instr.a as usize] =
                        self.vars.get(instr.b as usize).cloned().unwrap_or(BVal::Null);
                }
                Op::StoreVar => {
                    let vi = instr.a as usize;
                    if vi >= self.vars.len() { self.vars.resize(vi + 1, BVal::Null); }
                    self.vars[vi] = regs[instr.b as usize].clone();
                }

                // ── Crystal load (with cache mode) ─────────────────
                Op::LoadCrys => {
                    let cid  = instr.b as usize;
                    let mode = CrysLoadMode::from(instr.flags & 0x03);
                    let path = module.crystals.get(cid)
                        .map(|c| c.path.clone())
                        .unwrap_or_default();
                    let name = module.crystals.get(cid)
                        .map(|c| c.name.clone())
                        .unwrap_or_default();

                    // Issue prefetch for PREFETCH mode (async, non-blocking)
                    if mode == CrysLoadMode::Prefetch && runtime.crystal_cache.is_loaded(cid) {
                        // Already cached — no-op
                    }

                    // Load into cache (lazy mmap)
                    match runtime.crystal_cache.load(cid, &path, mode) {
                        Ok(_) => {
                            regs[instr.a as usize] = BVal::Str(
                                format!("crystal:{}", name)
                            );
                        }
                        Err(e) => {
                            // Crystal not on disk (e.g. not compiled yet) — use stub
                            regs[instr.a as usize] = BVal::Str(
                                format!("crystal:{}[stub]", name)
                            );
                            let _ = e;
                        }
                    }
                }

                // ── Arithmetic ─────────────────────────────────────
                Op::Add => {
                    let l = regs[instr.b as usize].as_f64();
                    let r = regs[instr.c as usize].as_f64();
                    regs[instr.a as usize] = BVal::Float(l + r);
                }
                Op::Sub => {
                    let l = regs[instr.b as usize].as_f64();
                    let r = regs[instr.c as usize].as_f64();
                    regs[instr.a as usize] = BVal::Float(l - r);
                }
                Op::Mul => {
                    let (l, r) = (regs[instr.b as usize].clone(), regs[instr.c as usize].clone());
                    regs[instr.a as usize] = eval_arith_bval(&l, &r, Op::Mul);
                }
                Op::Div => {
                    let l = regs[instr.b as usize].as_f64();
                    let r = regs[instr.c as usize].as_f64();
                    regs[instr.a as usize] =
                        BVal::Float(if r.abs() < 1e-300 { 0.0 } else { l / r });
                }
                Op::Pow => {
                    let l = regs[instr.b as usize].as_f64();
                    let r = regs[instr.c as usize].as_f64();
                    regs[instr.a as usize] = BVal::Float(l.powf(r));
                }
                Op::Neg => {
                    let v = regs[instr.b as usize].as_f64();
                    regs[instr.a as usize] = BVal::Float(-v);
                }

                // ── Comparison ─────────────────────────────────────
                Op::Eq  => { let r = (regs[instr.b as usize].as_f64() - regs[instr.c as usize].as_f64()).abs() < 1e-9; regs[instr.a as usize] = BVal::Bool(r); }
                Op::Lt  => { let r = regs[instr.b as usize].as_f64() < regs[instr.c as usize].as_f64(); regs[instr.a as usize] = BVal::Bool(r); }
                Op::Gt  => { let r = regs[instr.b as usize].as_f64() > regs[instr.c as usize].as_f64(); regs[instr.a as usize] = BVal::Bool(r); }
                Op::Not => { let r = !regs[instr.b as usize].as_bool(); regs[instr.a as usize] = BVal::Bool(r); }
                Op::And => { let r = regs[instr.b as usize].as_bool() && regs[instr.c as usize].as_bool(); regs[instr.a as usize] = BVal::Bool(r); }
                Op::Or  => { let r = regs[instr.b as usize].as_bool() || regs[instr.c as usize].as_bool(); regs[instr.a as usize] = BVal::Bool(r); }

                // ── Ternary ────────────────────────────────────────
                Op::TritMul => {
                    let a = regs[instr.b as usize].as_trit();
                    let b = regs[instr.c as usize].as_trit();
                    let r: i8 = match (a, b) { (0,_)|(_,0) => 0, (x,y) if x==y => 1, _ => -1 };
                    regs[instr.a as usize] = BVal::Trit(r);
                }
                Op::Encode => {
                    let scalar = regs[instr.b as usize].as_f64() as f32;
                    let dim    = if instr.c == 0 { 4864 } else { instr.c as usize };
                    let fvec: Vec<f32> = (0..dim)
                        .map(|i| (scalar * (i as f32 * 0.001 + 1.0)).sin())
                        .collect();
                    regs[instr.a as usize] = BVal::Fvec(fvec);
                }
                Op::Quantize => {
                    let v = regs[instr.b as usize].clone();
                    regs[instr.a as usize] = match v {
                        BVal::Fvec(fv) => {
                            let mean: f32 = fv.iter().map(|x| x.abs()).sum::<f32>() / fv.len() as f32;
                            let trits: Vec<i8> = fv.iter().map(|&x| {
                                if x > mean { 1 } else if x < -mean { -1 } else { 0 }
                            }).collect();
                            BVal::Tvec(trits)
                        }
                        other => other,
                    };
                }

                // ── ACT (activation function) ──────────────────────
                Op::Act => {
                    let src = regs[instr.b as usize].clone();
                    let func = ActFn::from(instr.c as u8);
                    let act_fn = match func {
                        ActFn::Step    => ActFunc::Step,
                        ActFn::ReLU    => ActFunc::ReLU,
                        ActFn::Sigmoid => ActFunc::Sigmoid,
                        ActFn::GeLU    => ActFunc::GeLU,
                        ActFn::Tanh    => ActFunc::Tanh,
                        ActFn::Lut     => ActFunc::ReLU, // fallback
                    };
                    regs[instr.a as usize] = match src {
                        BVal::Fvec(mut v) => { apply_activation(&mut v, act_fn); BVal::Fvec(v) }
                        BVal::Float(f)    => {
                            let mut v = vec![f as f32];
                            apply_activation(&mut v, act_fn);
                            BVal::Float(v[0] as f64)
                        }
                        BVal::Tvec(tv) => {
                            // ACT STEP on tvec = identity (already ternary)
                            BVal::Tvec(tv)
                        }
                        other => other,
                    };
                }

                // ── MM_TERN (AVX2 via backend_cpu) ─────────────────
                Op::MatMulTern => {
                    let rc    = instr.a as usize;
                    let cid   = instr.b as usize;
                    let x_reg = instr.c as usize;

                    let crystal_name = module.crystals.get(cid)
                        .map(|c| c.name.clone())
                        .unwrap_or_default();
                    let crystal_path = module.crystals.get(cid)
                        .map(|c| c.path.clone())
                        .unwrap_or_default();
                    let mode = module.crystals.get(cid)
                        .map(|c| c.mode)
                        .unwrap_or(CrysLoadMode::Stream);

                    // Get input vector
                    let x_vec: Vec<f32> = match &regs[x_reg] {
                        BVal::Fvec(v) => v.clone(),
                        BVal::Tvec(v) => v.iter().map(|&t| t as f32).collect(),
                        BVal::Float(f) => vec![*f as f32],
                        _ => vec![],
                    };

                    if x_vec.is_empty() {
                        regs[rc] = BVal::Str(format!("[{}] empty input", crystal_name));
                    } else {
                        // Load crystal (uses cache if already loaded)
                        match runtime.crystal_cache.load(cid, &crystal_path, mode) {
                            Ok(crys) => {
                                let t_mm = std::time::Instant::now();
                                // Determine actual input dim (pad/truncate x to cols)
                                let cols = crys.cols;
                                let rows = crys.rows;
                                let mut x_padded = x_vec.clone();
                                x_padded.resize(cols, 0.0);

                                // ── AVX2 tgemv ──
                                let result = tgemv_ternary(
                                    &crys.packed,
                                    &crys.scales,
                                    &x_padded,
                                    rows,
                                    cols,
                                );

                                // Fused ACT (if flag bit2 set)
                                let mut out = result.data;
                                if instr.flags & 0x04 != 0 {
                                    let act_fn = match (instr.flags >> 4) & 0x0F {
                                        1 => ActFunc::ReLU,
                                        2 => ActFunc::Sigmoid,
                                        3 => ActFunc::GeLU,
                                        _ => ActFunc::Step,
                                    };
                                    apply_activation(&mut out, act_fn);
                                }

                                let mm_ns = t_mm.elapsed().as_nanos() as u64;
                                runtime.profiler.record("MM_TERN", mm_ns);
                                runtime.profiler.record(&format!("MM_TERN[{}]", crystal_name), mm_ns);

                                regs[rc] = BVal::Fvec(out);
                            }
                            Err(_) => {
                                // Crystal not available — return stub
                                let norm: f32 = x_vec.iter().map(|f| f*f).sum::<f32>().sqrt();
                                regs[rc] = BVal::Str(format!(
                                    "[{}] |x|={:.3} → stub (crystal not loaded)", crystal_name, norm
                                ));
                            }
                        }
                    }
                }

                // ── SCALE_F (dequantize tvec × scale) ─────────────
                Op::ScaleF => {
                    let src = regs[instr.b as usize].clone();
                    let sc  = module.consts.get(instr.c as usize)
                        .and_then(|c| if let Const::Float(f) = c { Some(*f as f32) } else { None })
                        .unwrap_or(1.0);
                    regs[instr.a as usize] = match src {
                        BVal::Tvec(v) => BVal::Fvec(v.iter().map(|&t| t as f32 * sc).collect()),
                        BVal::Fvec(v) => BVal::Fvec(v.iter().map(|&f| f * sc).collect()),
                        BVal::Float(f) => BVal::Float(f * sc as f64),
                        other => other,
                    };
                }

                // ── ORACLE_CALL (async via AsyncOracleEngine) ──────
                Op::OracleCall => {
                    let ticket_reg = instr.a as usize;
                    let oi         = instr.b as usize;
                    let args_base  = instr.c as usize;
                    let n_args     = (instr.flags & 0x0F) as usize;
                    let is_async   = instr.flags & 0x08 != 0;

                    // Collect args as f64
                    let args: Vec<f64> = (0..n_args)
                        .map(|i| regs[(args_base + i).min(255)].as_f64())
                        .collect();

                    let entry_ip = *self.oracle_entries.get(oi)
                        .ok_or_else(|| format!("Unknown oracle idx {}", oi))?;

                    // ── JIT fast path (sync only) ──────────────────────
                    let oracle_name = module.oracles.get(oi)
                        .map(|o| o.name.clone())
                        .unwrap_or_default();

                    if !is_async {
                        if let Some(ref jit) = self.jit {
                            if let Some(co) = jit.cache.get(&oracle_name) {
                                let result = unsafe { co.call(&args) };
                                regs[ticket_reg] = BVal::Float(result);
                                runtime.profiler.record("ORACLE_CALL[JIT]",
                                    t0.elapsed().as_nanos() as u64);
                                continue;
                            }
                        }
                    }

                    if is_async {
                        // Async: submit job to engine, store ticket in register
                        let oracle_fn = build_oracle_fn(module, entry_ip, oi);
                        let ticket = runtime.oracle_engine.submit(oi, args, oracle_fn);
                        regs[ticket_reg] = BVal::Ticket(ticket);
                    } else {
                        // Sync interpreter path
                        let result = exec_oracle_sync(module, entry_ip, &args)?;
                        regs[ticket_reg] = BVal::Float(result);

                        // ── JIT threshold: tick → auto-compile when hot ──
                        if let Some(ref mut jit) = self.jit {
                            if !oracle_name.is_empty() && jit.cache.tick(&oracle_name) {
                                // Threshold crossed — compile to native x86-64
                                match jit.compile_oracle(module, &oracle_name) {
                                    Ok(()) => eprintln!(
                                        "[JIT] auto-compiled '{}' after {} interpreter calls",
                                        oracle_name, crate::jit::JIT_THRESHOLD
                                    ),
                                    Err(e) => eprintln!(
                                        "[JIT] compile '{}' failed: {}", oracle_name, e
                                    ),
                                }
                            }
                        }
                    }

                    runtime.profiler.record("ORACLE_CALL",
                        t0.elapsed().as_nanos() as u64);
                }

                // ── ORACLE_WAIT (block until ticket ready) ─────────
                Op::OracleWait => {
                    let (rd, rs) = (instr.a as usize, instr.b as usize);
                    regs[rd] = match regs[rs].clone() {
                        BVal::Ticket(t) => {
                            let v = runtime.oracle_engine.wait(t);
                            BVal::Float(v)
                        }
                        // Already resolved (sync path)
                        other => other,
                    };
                    runtime.profiler.record("ORACLE_WAIT",
                        t0.elapsed().as_nanos() as u64);
                }

                // ── ORACLE_FUSED (single register window) ──────────
                Op::OracleFused => {
                    let rc   = instr.a as usize;
                    let oi_a = instr.b as usize;
                    let oi_b = instr.c as usize;

                    let entry_a = *self.oracle_entries.get(oi_a).unwrap_or(&0);
                    let entry_b = *self.oracle_entries.get(oi_b).unwrap_or(&0);

                    let name_a = module.oracles.get(oi_a).map(|o| o.name.clone()).unwrap_or_default();
                    let name_b = module.oracles.get(oi_b).map(|o| o.name.clone()).unwrap_or_default();

                    // Collect args from current register window (R0..R7)
                    let args_a: Vec<f64> = (0..8)
                        .map(|i| regs[i].as_f64())
                        .take_while(|&v| v != 0.0)
                        .collect();

                    // Execute oracle_a — JIT if available
                    let r_a = if let Some(ref jit) = self.jit {
                        if let Some(co) = jit.cache.get(&name_a) {
                            unsafe { co.call(&args_a) }
                        } else {
                            exec_oracle_sync(module, entry_a, &args_a).unwrap_or(0.0)
                        }
                    } else {
                        exec_oracle_sync(module, entry_a, &args_a).unwrap_or(0.0)
                    };

                    // Feed r_a as first arg to oracle_b — JIT if available
                    let r_b = if let Some(ref jit) = self.jit {
                        if let Some(co) = jit.cache.get(&name_b) {
                            unsafe { co.call(&[r_a]) }
                        } else {
                            exec_oracle_sync(module, entry_b, &[r_a]).unwrap_or(0.0)
                        }
                    } else {
                        exec_oracle_sync(module, entry_b, &[r_a]).unwrap_or(0.0)
                    };

                    regs[rc] = BVal::Float(r_b);
                    runtime.profiler.record("ORACLE_FUSED",
                        t0.elapsed().as_nanos() as u64);
                }

                // ── Control flow ───────────────────────────────────
                Op::JumpFalse => {
                    if !regs[instr.a as usize].as_bool() { ip = instr.b as usize; }
                }
                Op::Jump => { ip = instr.a as usize; }

                Op::Return => {
                    let ret_val = regs[instr.a as usize].clone();
                    if let Some(frame) = call_stack.pop() {
                        *regs = *frame.saved_regs;
                        ip    = frame.ret_ip;
                        regs[frame.result_reg as usize] = ret_val;
                    } else {
                        break;
                    }
                }

                // ── Oracle call (via CALL opcode, sync frame push) ─
                Op::Call => {
                    let oi        = instr.a as usize;
                    let args_base = instr.b as usize;
                    let n_args    = instr.flags as usize;
                    let entry_ip  = *self.oracle_entries.get(oi).ok_or("bad oracle")?;
                    let rc        = self.oracle_entries.len() as u16; // result → R[n_oracles]

                    let oracle_name = module.oracles.get(oi)
                        .map(|o| o.name.clone())
                        .unwrap_or_default();

                    // ── JIT fast path: skip frame push entirely ────────
                    if let Some(ref jit) = self.jit {
                        if let Some(co) = jit.cache.get(&oracle_name) {
                            let args: Vec<f64> = (0..n_args)
                                .map(|i| regs[(args_base + i).min(255)].as_f64())
                                .collect();
                            let result = unsafe { co.call(&args) };
                            regs[rc as usize] = BVal::Float(result);
                            continue;
                        }
                    }

                    // Interpreter path: save frame, jump to entry
                    let mut saved = Box::new(std::array::from_fn(|_| BVal::Null));
                    for i in 0..256 { saved[i] = regs[i].clone(); }
                    // Load args into R[0..n_args]
                    for i in 0..n_args {
                        regs[i] = saved[(args_base + i).min(255)].clone();
                    }
                    call_stack.push(Frame { ret_ip: ip, saved_regs: saved, result_reg: rc });
                    ip = entry_ip;
                }

                // ── PAR_BEGIN — real parallel oracle dispatch ──────
                //
                // Encoding:
                //   ParBegin { a: n_lanes, b: join_ip }
                //   Fork     { a: lane_mask, b: target_ip }  (one per lane)
                //   ...lane body (ORACLE_CALL, ORACLE_WAIT, Respond, ...)...
                //   Join     (rendez-vous, skipped by fast-forward)
                //   ParEnd
                //
                // Execution:
                //   1. Scan forward to collect all Fork targets
                //   2. std::thread::scope: each lane gets clone of regs + fresh runtime
                //   3. Merge lane outputs into main output vec
                //   4. ip = join_ip (fast-forward past lane bodies)
                Op::ParBegin => {
                    let join_ip     = instr.b as usize;
                    let code        = &module.code;
                    let code_len    = code.len();

                    // Collect Fork targets in the parallel region
                    let mut fork_targets: Vec<usize> = Vec::new();
                    let mut scan = ip;  // ip already past ParBegin
                    while scan < code_len && scan < join_ip {
                        if code[scan].op == Op::Fork {
                            fork_targets.push(code[scan].b as usize);
                        }
                        scan += 1;
                    }

                    if fork_targets.is_empty() {
                        // No forks — just run sequentially (fall through)
                    } else {
                        // Build JIT fn-table once — Send+Sync, shared across lanes (v1.7)
                        let jit_table: Option<JitFnTable> =
                            self.jit.as_ref().map(|jit| jit.fn_table());

                        // Clone register snapshot for each lane
                        let reg_snaps: Vec<Box<[BVal; 256]>> = fork_targets
                            .iter()
                            .map(|_| Box::new(std::array::from_fn(|i| regs[i].clone())))
                            .collect();

                        // Run lanes in parallel using scoped threads
                        let lane_results: Vec<Vec<String>> = std::thread::scope(|scope| {
                            let handles: Vec<_> = fork_targets.iter().zip(reg_snaps.into_iter())
                                .map(|(&target_ip, lane_regs)| {
                                    let tbl = jit_table.clone();
                                    scope.spawn(move || {
                                        run_fork_lane(module, target_ip, lane_regs, tbl)
                                    })
                                })
                                .collect();
                            handles.into_iter()
                                .map(|h| h.join().unwrap_or_default())
                                .collect()
                        });

                        // Merge all lane outputs into main output
                        for lane_out in lane_results {
                            self.output.extend(lane_out);
                        }

                        // Fast-forward past all lane bodies to join_ip
                        ip = join_ip;
                    }

                    runtime.profiler.record("PAR_BEGIN", t0.elapsed().as_nanos() as u64);
                }

                // PAR_END / FORK / JOIN — handled by PAR_BEGIN logic above
                Op::ParEnd | Op::Fork | Op::Join => {}

                // ── Output ─────────────────────────────────────────
                Op::Respond => {
                    self.output.push(format!("{}", regs[instr.a as usize]));
                    runtime.profiler.record("RESPOND", t0.elapsed().as_nanos() as u64);
                }

                Op::Halt => break,
                Op::Nop  => {}
                _ => {}
            }
        }

        Ok(self.output.clone())
    }
}

// ── Fork lane executor (runs in its own scoped thread) ────────────────

/// Execute one PAR_BEGIN fork lane from `entry_ip` to the first Join/ParEnd.
/// Each lane gets its own register snapshot and a fresh lightweight runtime.
///
/// `jit_table` (v1.7): if provided, oracle calls dispatch to JIT-compiled
/// native code via the fn-address table instead of the bytecode interpreter.
/// The table is `Arc<HashMap<String,(usize,usize)>>` — Send+Sync — built
/// from `JitEngine::fn_table()` before the scope, so `JITModule` (!Send)
/// never crosses the thread boundary.
///
/// Returns the Respond outputs produced by this lane.
fn run_fork_lane(
    module:    &Module,
    entry_ip:  usize,
    mut regs:  Box<[BVal; 256]>,
    jit_table: Option<JitFnTable>,
) -> Vec<String> {
    // Fresh runtime: 2 oracle workers per lane — keeps thread overhead low
    let mut rt   = crate::runtime::CrysRuntime::new(2);
    let mut out  = Vec::new();
    let mut ip   = entry_ip;
    let code_len = module.code.len();

    while ip < code_len {
        let instr = module.code[ip];
        ip += 1;

        match instr.op {
            // Stop at lane boundary
            Op::Join | Op::ParEnd => break,

            // Lightweight arithmetic (mirrors main dispatch)
            Op::LoadConst => {
                regs[instr.a as usize] = const_to_bval(module.consts.get(instr.b as usize));
            }
            Op::Add => {
                regs[instr.a as usize] =
                    BVal::Float(regs[instr.b as usize].as_f64() + regs[instr.c as usize].as_f64());
            }
            Op::Sub => {
                regs[instr.a as usize] =
                    BVal::Float(regs[instr.b as usize].as_f64() - regs[instr.c as usize].as_f64());
            }
            Op::Mul => {
                let (l, r) = (regs[instr.b as usize].clone(), regs[instr.c as usize].clone());
                regs[instr.a as usize] = eval_arith_bval(&l, &r, Op::Mul);
            }
            Op::Div => {
                let (l, r) = (regs[instr.b as usize].as_f64(), regs[instr.c as usize].as_f64());
                regs[instr.a as usize] = BVal::Float(if r.abs() < 1e-300 { 0.0 } else { l / r });
            }
            Op::Pow => {
                regs[instr.a as usize] = BVal::Float(
                    regs[instr.b as usize].as_f64().powf(regs[instr.c as usize].as_f64())
                );
            }
            Op::Move => {
                let v = regs[instr.b as usize].clone();
                regs[instr.a as usize] = v;
            }

            // Oracle call — v1.7: JIT dispatch if table available, else interpreter
            Op::OracleCall | Op::OracleFused => {
                let oi        = instr.b as usize;
                let args_base = instr.c as usize;
                let n_args    = (instr.flags & 0x0F) as usize;
                let args: Vec<f64> = (0..n_args)
                    .map(|i| regs[(args_base + i).min(255)].as_f64())
                    .collect();

                // v1.7: try JIT table first (native dispatch, ~2.5 ns/call)
                let oracle_name = module.oracles.get(oi).map(|o| o.name.as_str()).unwrap_or("");
                let result = if let Some(ref tbl) = jit_table {
                    if let Some(&(fn_addr, _n_params)) = tbl.get(oracle_name) {
                        unsafe { jit_table_call(fn_addr, &args) }
                    } else {
                        let entry = module.oracles.get(oi).map(|o| o.entry_ip).unwrap_or(0);
                        exec_oracle_sync(module, entry, &args).unwrap_or(0.0)
                    }
                } else {
                    let entry = module.oracles.get(oi).map(|o| o.entry_ip).unwrap_or(0);
                    exec_oracle_sync(module, entry, &args).unwrap_or(0.0)
                };
                regs[instr.a as usize] = BVal::Float(result);
            }

            Op::OracleWait => {
                // In fork lane, tickets already resolved (sync path above)
                let v = regs[instr.b as usize].clone();
                regs[instr.a as usize] = v;
            }

            Op::Respond => {
                out.push(format!("{}", regs[instr.a as usize]));
            }

            Op::LoadCrys => {
                let cid  = instr.b as usize;
                let mode = CrysLoadMode::from(instr.flags & 0x03);
                let path = module.crystals.get(cid).map(|c| c.path.clone()).unwrap_or_default();
                let name = module.crystals.get(cid).map(|c| c.name.clone()).unwrap_or_default();
                let _ = rt.crystal_cache.load(cid, &path, mode);
                regs[instr.a as usize] = BVal::Str(format!("crystal:{}", name));
            }

            Op::MatMulTern => {
                let cid  = instr.b as usize;
                let x_reg = instr.c as usize;
                let path = module.crystals.get(cid).map(|c| c.path.clone()).unwrap_or_default();
                let mode = module.crystals.get(cid).map(|c| c.mode).unwrap_or(CrysLoadMode::Stream);
                let x_vec: Vec<f32> = match &regs[x_reg] {
                    BVal::Fvec(v) => v.clone(),
                    BVal::Tvec(v) => v.iter().map(|&t| t as f32).collect(),
                    BVal::Float(f) => vec![*f as f32],
                    _ => vec![],
                };
                if !x_vec.is_empty() {
                    if let Ok(crys) = rt.crystal_cache.load(cid, &path, mode) {
                        let mut x_padded = x_vec;
                        x_padded.resize(crys.cols, 0.0);
                        let result = crate::backend_cpu::tgemv_ternary(
                            &crys.packed, &crys.scales, &x_padded, crys.rows, crys.cols
                        );
                        regs[instr.a as usize] = BVal::Fvec(result.data);
                    }
                }
            }

            Op::Halt => break,
            Op::Nop  => {}
            // Skip other instructions not needed in fork context
            _ => {}
        }
    }

    let _ = rt;
    out
}

// ── Oracle sync executor (inline, no thread) ──────────────────────────

/// Execute oracle body starting at entry_ip with given args.
/// Used for ORACLE_FUSED, sync fallback, and JIT benchmarking.
pub fn exec_oracle_sync(
    module:   &Module,
    entry_ip: usize,
    args:     &[f64],
) -> Result<f64, String> {
    let mut regs = vec![0.0f64; 64];
    for (i, &v) in args.iter().enumerate().take(32) { regs[i] = v; }
    // Also store into vars (oracle params are stored via STORE_VAR at entry)
    let mut vars = vec![0.0f64; module.vars.len().max(32)];

    let mut ip = entry_ip;
    let code_len = module.code.len();

    while ip < code_len {
        let instr = module.code[ip];
        ip += 1;
        match instr.op {
            Op::LoadConst => {
                regs[instr.a as usize % 64] = match module.consts.get(instr.b as usize) {
                    Some(Const::Float(f)) => *f,
                    Some(Const::Int(n))   => *n as f64,
                    _ => 0.0,
                };
            }
            Op::LoadVar => {
                regs[instr.a as usize % 64] =
                    vars.get(instr.b as usize).copied().unwrap_or(0.0);
            }
            Op::StoreVar => {
                let vi = instr.a as usize;
                if vi >= vars.len() { vars.resize(vi + 1, 0.0); }
                vars[vi] = regs[instr.b as usize % 64];
            }
            Op::Add  => { regs[instr.a as usize%64] = regs[instr.b as usize%64] + regs[instr.c as usize%64]; }
            Op::Sub  => { regs[instr.a as usize%64] = regs[instr.b as usize%64] - regs[instr.c as usize%64]; }
            Op::Mul  => { regs[instr.a as usize%64] = regs[instr.b as usize%64] * regs[instr.c as usize%64]; }
            Op::Div  => {
                let r = regs[instr.c as usize % 64];
                regs[instr.a as usize%64] = if r.abs() < 1e-300 { 0.0 } else { regs[instr.b as usize%64] / r };
            }
            Op::Pow  => { regs[instr.a as usize%64] = regs[instr.b as usize%64].powf(regs[instr.c as usize%64]); }
            Op::Neg  => { regs[instr.a as usize%64] = -regs[instr.b as usize%64]; }
            Op::Move => { regs[instr.a as usize%64] = regs[instr.b as usize%64]; }
            Op::Return => { return Ok(regs[instr.a as usize % 64]); }
            Op::Halt   => break,
            _ => {}
        }
    }
    Ok(0.0)
}

/// Build a closure over the oracle bytecode for async dispatch.
/// The closure is self-contained (captures a copy of the module code slice).
fn build_oracle_fn(module: &Module, entry_ip: usize, _oi: usize) -> impl Fn(&[f64]) -> f64 + Send + 'static {
    // Clone just the code + consts + vars needed for this oracle
    let code:   Vec<_> = module.code.iter().skip(entry_ip).cloned().collect();
    let consts: Vec<_> = module.consts.clone();
    let n_vars          = module.vars.len();

    move |args: &[f64]| -> f64 {
        let mut regs = vec![0.0f64; 64];
        let mut vars = vec![0.0f64; n_vars.max(32)];
        for (i, &v) in args.iter().enumerate().take(32) { regs[i] = v; }

        for instr in &code {
            match instr.op {
                Op::LoadConst => {
                    regs[instr.a as usize%64] = match consts.get(instr.b as usize) {
                        Some(Const::Float(f)) => *f,
                        Some(Const::Int(n))   => *n as f64,
                        _ => 0.0,
                    };
                }
                Op::LoadVar  => { regs[instr.a as usize%64] = vars.get(instr.b as usize).copied().unwrap_or(0.0); }
                Op::StoreVar => {
                    let vi = instr.a as usize;
                    if vi >= vars.len() { vars.resize(vi+1, 0.0); }
                    vars[vi] = regs[instr.b as usize%64];
                }
                Op::Add  => { regs[instr.a as usize%64] = regs[instr.b as usize%64] + regs[instr.c as usize%64]; }
                Op::Sub  => { regs[instr.a as usize%64] = regs[instr.b as usize%64] - regs[instr.c as usize%64]; }
                Op::Mul  => { regs[instr.a as usize%64] = regs[instr.b as usize%64] * regs[instr.c as usize%64]; }
                Op::Div  => {
                    let r = regs[instr.c as usize%64];
                    regs[instr.a as usize%64] = if r.abs() < 1e-300 { 0.0 } else { regs[instr.b as usize%64] / r };
                }
                Op::Pow  => { regs[instr.a as usize%64] = regs[instr.b as usize%64].powf(regs[instr.c as usize%64]); }
                Op::Neg  => { regs[instr.a as usize%64] = -regs[instr.b as usize%64]; }
                Op::Move => { regs[instr.a as usize%64] = regs[instr.b as usize%64]; }
                Op::Return => { return regs[instr.a as usize%64]; }
                Op::Halt   => break,
                _ => {}
            }
        }
        0.0
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn const_to_bval(c: Option<&Const>) -> BVal {
    match c {
        Some(Const::Int(n))    => BVal::Int(*n),
        Some(Const::Float(f))  => BVal::Float(*f),
        Some(Const::Bool(b))   => BVal::Bool(*b),
        Some(Const::Str(s))    => BVal::Str(s.clone()),
        None                    => BVal::Null,
    }
}

fn eval_arith_bval(l: &BVal, r: &BVal, op: Op) -> BVal {
    match (l, r) {
        (BVal::Int(a), BVal::Int(b)) => match op {
            Op::Add => BVal::Int(a + b),
            Op::Sub => BVal::Int(a - b),
            Op::Mul => BVal::Int(a * b),
            _ => BVal::Float((*a as f64) * (*b as f64)),
        },
        _ => BVal::Float(match op {
            Op::Add => l.as_f64() + r.as_f64(),
            Op::Sub => l.as_f64() - r.as_f64(),
            Op::Mul => l.as_f64() * r.as_f64(),
            Op::Div => { let d = r.as_f64(); if d.abs() < 1e-300 { 0.0 } else { l.as_f64() / d } }
            Op::Pow => l.as_f64().powf(r.as_f64()),
            _ => 0.0,
        }),
    }
}

// ── Public API ────────────────────────────────────────────────────────

/// Run a program via the Bytecode VM using the full Runtime (v1.4).
pub fn run_bytecode(module: &Module, runtime: &mut CrysRuntime) -> Result<Vec<String>, String> {
    BytecodeVm::new().run(module, runtime)
}

/// Run a program via the Bytecode VM with JIT oracle dispatch enabled (v1.6).
///
/// Hot oracles are auto-compiled to native x86-64 after `JIT_THRESHOLD` interpreter
/// calls. Subsequent calls bypass the interpreter entirely via the Cranelift fn_ptr.
///
/// `engine` — pre-initialized JitEngine; may already hold pre-compiled oracles
///            (e.g. from `JitEngine::compile_all`).
pub fn run_bytecode_jit(
    module:  &Module,
    runtime: &mut CrysRuntime,
    engine:  JitEngine,
) -> Result<Vec<String>, String> {
    BytecodeVm::new().with_jit(engine).run(module, runtime)
}

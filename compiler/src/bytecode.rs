// ═══════════════════════════════════════════════════════════════════════
// QOMN v1.4 — CRYS-ISA  (Bytecode IR + Optimizer)
//
// ISA de dos niveles inspirada en MLIR/XLA/oneDNN:
//
//   Nivel 1 (Instrucción, 8 bytes fijos):
//     [op:u8][flags:u8][a:u16][b:u16][c:u16]
//     a/b/c son IDs (registro virtual, descriptor, tensor, ticket…)
//
//   Nivel 2 (Descriptores en memoria — TensorDesc):
//     struct TensorDesc { shape, stride, layout, quantization }
//     Las instrucciones referencian descriptores por ID, igual que
//     CUDA, oneDNN y XLA — sin codificar dimensiones en la instrucción.
//
// Registros virtuales:
//   %t0–%t15  tensor ternario (tvec/tmat, lazy-mmap backing)
//   %f0–%f15  float32 escalar / fvec
//   %i0–%i7   control / índices
//   %o0–%o7   oracle handles (ticket async)
//
// Mejoras v1.4 sobre v1.3:
//   1. TensorDesc table — shape/stride/layout/quantization desacoplados
//   2. LOAD_CRYS mode   — L1_PIN | STREAM | PREFETCH (EPYC prefetch aware)
//   3. PAR_BEGIN/PAR_END + FORK/JOIN — paralelismo explícito de lanes
//   4. ORACLE_CALL (→ ticket) + ORACLE_WAIT — async por defecto
//   5. ACT %reg, func_id — ReLU / STEP / SIGMOID / CUSTOM_LUT
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use crate::ast::{
    Program, Decl, Stmt, Expr, BinaryOp, UnaryOp,
    OracleDecl, CrystalDecl,
};

// ── Tensor Descriptor (Nivel 2) ───────────────────────────────────────

/// Memory layout for tensors referenced by ISA instructions.
/// Instructions encode desc_id (u16); the runtime resolves shape at exec time.
/// Inspired by: CUDA's cudaTensorMapEncode, oneDNN's memory descriptor.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// Logical shape [rows, cols] (or [len] for vectors)
    pub shape:        [u32; 4],
    /// Stride in elements per axis (row-major default)
    pub stride:       [u32; 4],
    /// Memory layout
    pub layout:       TensorLayout,
    /// Quantization scheme for ternary tensors
    pub quantization: QuantMode,
    /// Scale factors (one per row for BitNet absmean)
    pub scales:       Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Standard row-major (C order)
    RowMajor,
    /// Column-major (Fortran order)
    ColMajor,
    /// Blocked 32×32 tiles (aligned to AVX2 cache lines)
    Blocked32,
    /// Sparse coordinate format (COO)
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Float32,
    /// BitNet b1.58 — {-1,0,+1} per-row absmean scale
    Ternary,
    /// 4-bit NF4 (future)
    NF4,
}

impl Default for TensorDesc {
    fn default() -> Self {
        Self {
            shape:        [0; 4],
            stride:       [1; 4],
            layout:       TensorLayout::RowMajor,
            quantization: QuantMode::Float32,
            scales:       vec![],
        }
    }
}

// ── Crystal Load Mode (EPYC prefetch aware) ───────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QomnoadMode {
    /// Pin crystal into L1/L2 cache (small crystals ≤ 4 MB)
    L1Pin    = 0,
    /// Stream — read sequentially without polluting cache (large crystals)
    Stream   = 1,
    /// Prefetch next crystal while current inference runs
    Prefetch = 2,
}

impl From<u8> for QomnoadMode {
    fn from(v: u8) -> Self {
        match v { 1 => Self::Stream, 2 => Self::Prefetch, _ => Self::L1Pin }
    }
}

// ── Activation Function IDs ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActFn {
    Step    = 0,   // Heaviside step (ternary → {-1,0,+1})
    ReLU    = 1,
    Sigmoid = 2,
    GeLU    = 3,
    Tanh    = 4,
    Lut     = 15,  // Custom lookup table (lut_id in `c`)
}

impl From<u8> for ActFn {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::ReLU, 2 => Self::Sigmoid,
            3 => Self::GeLU, 4 => Self::Tanh, 15 => Self::Lut,
            _ => Self::Step,
        }
    }
}

// ── Opcodes ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    // ── Data movement ──────────────────────────────────────────────
    /// LOAD_CONST  %ra  const_id          %ra = consts[const_id]
    LoadConst   = 0x01,
    /// LOAD_TRIT   %ti  trit_val          %ti = Trit(val as i8)
    LoadTrit    = 0x02,
    /// MOVE        %dst  %src
    Move        = 0x03,
    /// LOAD_VAR    %ra  var_id            %ra = vars[var_id]
    LoadVar     = 0x04,
    /// STORE_VAR   var_id  %ra            vars[var_id] = %ra
    StoreVar    = 0x05,

    // ── Crystal load (with cache mode) ─────────────────────────────
    /// LOAD_CRYS   %ti  crys_id  mode     %ti ← mmap crystal[crys_id]
    /// flags encodes QomnoadMode: 0=L1_PIN 1=STREAM 2=PREFETCH
    LoadCrys    = 0x06,

    // ── Arithmetic ─────────────────────────────────────────────────
    Add         = 0x10,
    Sub         = 0x11,
    Mul         = 0x12,
    Div         = 0x13,
    Pow         = 0x14,
    Neg         = 0x15,   // unary

    // ── Comparison / boolean ────────────────────────────────────────
    Eq          = 0x20,
    Lt          = 0x21,
    Gt          = 0x22,
    Not         = 0x23,
    And         = 0x24,
    Or          = 0x25,

    // ── Ternary kernel ──────────────────────────────────────────────
    /// TRIT_MUL  %rc  %ra  %rb      ternary product
    TritMul     = 0x30,
    /// ENCODE    %rc  %ra  dim      scalar → sinusoidal Fvec
    Encode      = 0x31,
    /// QUANTIZE  %rc  %ra           Fvec → Tvec (absmean BitNet)
    Quantize    = 0x32,

    // ── Matrix-Ternary multiply (AVX2-ready) ───────────────────────
    /// MM_TERN   %t_res  %t_mat  %t_vec
    ///   flags bit0 = AVX2_ALIGNED (32B boundary, enables _mm256_sign_epi8)
    ///   flags bit1 = BLOCKED32   (use TensorLayout::Blocked32 tile path)
    ///   Descriptor for %t_mat resolved from tensor_descs[mat_id]
    MatMulTern  = 0x40,

    // ── Activation ─────────────────────────────────────────────────
    /// ACT   %res  %src  func_id
    /// func_id: 0=STEP 1=RELU 2=SIGMOID 3=GELU 4=TANH 15=LUT(c=lut_id)
    Act         = 0x41,

    // ── Scale / dequantize ─────────────────────────────────────────
    /// SCALE_F  %f_res  %t_src  scale_const_id
    /// Dequantize: f_res = t_src * consts[scale_const_id]
    ScaleF      = 0x42,

    // ── Oracle (async, ticket-based) ───────────────────────────────
    /// ORACLE_CALL  %oi  oid  args_base
    /// Issues oracle computation, returns immediately with ticket in %oi.
    /// flags = n_args
    OracleCall  = 0x50,
    /// ORACLE_WAIT  %fi  %oi
    /// Block until oracle ticket %oi completes, result → %fi.
    OracleWait  = 0x51,
    /// ORACLE_FUSED %res  oa_id  ob_id   (args_base implicit from regs)
    /// Fused oracle pair in single register window.
    OracleFused = 0x52,

    // ── Parallelism ─────────────────────────────────────────────────
    /// PAR_BEGIN  n_lanes  label_join
    /// Spawns n_lanes parallel execution lanes; they all jump to label_join
    /// when done. Each lane gets a unique %i0 = lane_id.
    ParBegin    = 0x60,
    /// PAR_END     (implicit barrier — all lanes rendez-vous here)
    ParEnd      = 0x61,
    /// FORK  lane_mask  target_ip
    /// Dispatch a subset of lanes to target_ip (bitmask in `a`).
    Fork        = 0x62,
    /// JOIN  (barrier for forked lanes)
    Join        = 0x63,

    // ── Control flow ────────────────────────────────────────────────
    /// JUMP_IF_FALSE  %cond  target_ip
    JumpFalse   = 0x70,
    /// JUMP  target_ip
    Jump        = 0x71,
    /// CALL  oracle_id  args_base  n_args  → push frame
    Call        = 0x72,
    /// RETURN  %ra
    Return      = 0x73,

    // ── I/O ─────────────────────────────────────────────────────────
    /// RESPOND  %ra   push %ra to output buffer
    Respond     = 0x80,

    // ── Meta ────────────────────────────────────────────────────────
    Nop         = 0xF0,
    Halt        = 0xFF,
}

// ── Instruction (8 bytes fixed-width) ────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Instr {
    pub op:    Op,
    /// bit0 = avx2_aligned
    /// bit1 = blocked32
    /// bit2 = tail_call
    /// bit3 = async (for ORACLE_CALL)
    /// bits[4..7] = n_args (for ORACLE_CALL)
    pub flags: u8,
    pub a:     u16,
    pub b:     u16,
    pub c:     u16,
}

impl Instr {
    pub fn new(op: Op, a: u16, b: u16, c: u16) -> Self {
        Self { op, flags: 0, a, b, c }
    }
    pub fn with_flags(mut self, f: u8) -> Self { self.flags = f; self }
}

// ── Constant pool ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Const {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

// ── Compiled Module ───────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct Module {
    /// Flat instruction stream
    pub code:          Vec<Instr>,
    /// Constant pool
    pub consts:        Vec<Const>,
    /// Oracle table: name → (entry_ip, n_params)
    pub oracles:       Vec<OracleMeta>,
    /// Crystal table: name → path + desc_id
    pub crystals:      Vec<CrystalMeta>,
    /// Variable name table
    pub vars:          Vec<String>,
    /// Tensor descriptor table (Level-2: shape/stride/layout/quant)
    pub tensor_descs:  Vec<TensorDesc>,
}

#[derive(Debug)]
pub struct OracleMeta {
    pub name:     String,
    pub entry_ip: usize,
    pub n_params: usize,
}

#[derive(Debug)]
pub struct CrystalMeta {
    pub name:    String,
    pub path:    String,
    /// Index into tensor_descs for this crystal's weight matrix
    pub desc_id: usize,
    /// Default load mode
    pub mode:    QomnoadMode,
}

// ── Compiler ──────────────────────────────────────────────────────────

pub struct Compiler {
    module:     Module,
    next_reg:   u16,
    var_map:    HashMap<String, u16>,
    oracle_map: HashMap<String, u16>,
    /// Pending oracle tickets: %oi → result register in caller
    ticket_map: HashMap<u16, u16>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            module:     Module::default(),
            next_reg:   0,
            var_map:    HashMap::new(),
            oracle_map: HashMap::new(),
            ticket_map: HashMap::new(),
        }
    }

    fn alloc_reg(&mut self) -> u16 {
        let r = self.next_reg;
        self.next_reg = self.next_reg.saturating_add(1);
        r
    }

    fn reset_regs(&mut self) { self.next_reg = 0; }

    fn add_const(&mut self, c: Const) -> u16 {
        for (i, e) in self.module.consts.iter().enumerate() {
            if const_eq(e, &c) { return i as u16; }
        }
        let idx = self.module.consts.len() as u16;
        self.module.consts.push(c);
        idx
    }

    fn emit(&mut self, instr: Instr) -> usize {
        let ip = self.module.code.len();
        self.module.code.push(instr);
        ip
    }

    fn patch_jump(&mut self, ip: usize, target: u16) {
        self.module.code[ip].b = target;
    }

    fn var_idx(&mut self, name: &str) -> u16 {
        if let Some(&i) = self.var_map.get(name) { return i; }
        let i = self.module.vars.len() as u16;
        self.module.vars.push(name.to_string());
        self.var_map.insert(name.to_string(), i);
        i
    }

    /// Add a TensorDesc for a crystal weight matrix (ROWS×COLS ternary).
    fn add_tensor_desc(&mut self, rows: u32, cols: u32) -> usize {
        let id = self.module.tensor_descs.len();
        self.module.tensor_descs.push(TensorDesc {
            shape:        [rows, cols, 0, 0],
            stride:       [cols, 1, 0, 0],
            layout:       TensorLayout::Blocked32,  // AVX2 tile-friendly
            quantization: QuantMode::Ternary,
            scales:       vec![1.0f32; rows as usize],
        });
        id
    }

    // ── Public entry ─────────────────────────────────────────────
    pub fn compile(mut self, prog: &Program) -> Module {
        // Pass 1: register declarations
        for decl in &prog.decls {
            match decl {
                Decl::Oracle(o)  => self.register_oracle(o),
                Decl::Crystal(c) => self.register_crystal(c),
                _ => {}
            }
        }
        // Pass 2: top-level code
        for decl in &prog.decls {
            match decl {
                Decl::Let(name, _, expr) => {
                    let ra = self.compile_expr(expr);
                    let vi = self.var_idx(name);
                    self.emit(Instr::new(Op::StoreVar, vi, ra, 0));
                }
                Decl::Stmt(s) => { self.compile_stmt(s); }
                _ => {}
            }
        }
        self.emit(Instr::new(Op::Halt, 0, 0, 0));
        optimize(&mut self.module);
        self.module
    }

    fn register_oracle(&mut self, oracle: &OracleDecl) {
        let idx = self.module.oracles.len() as u16;
        self.oracle_map.insert(oracle.name.clone(), idx);
        let saved = self.next_reg;
        self.reset_regs();
        let entry_ip = self.module.code.len();
        self.module.oracles.push(OracleMeta {
            name:     oracle.name.clone(),
            entry_ip,
            n_params: oracle.params.len(),
        });
        for (i, param) in oracle.params.iter().enumerate() {
            let vi = self.var_idx(&param.name);
            self.emit(Instr::new(Op::StoreVar, vi, i as u16, 0));
        }
        for stmt in &oracle.body {
            if let Some(ra) = self.compile_stmt(stmt) {
                self.emit(Instr::new(Op::Return, ra, 0, 0));
            }
        }
        self.emit(Instr::new(Op::Return, 0, 0, 0));
        self.next_reg = saved;
    }

    fn register_crystal(&mut self, crystal: &CrystalDecl) {
        // Default: Qwen-0.5B FFN dimensions (ROWS=896, COLS=4864)
        let desc_id = self.add_tensor_desc(896, 4864);
        // Large crystals (86 MB) → STREAM mode to avoid cache pollution
        let mode = if std::path::Path::new(&crystal.path)
            .metadata().map(|m| m.len()).unwrap_or(0) > 4 * 1024 * 1024
        { QomnoadMode::Stream } else { QomnoadMode::L1Pin };

        self.module.crystals.push(CrystalMeta {
            name: crystal.name.clone(),
            path: crystal.path.clone(),
            desc_id,
            mode,
        });
    }

    // ── Statement compiler ───────────────────────────────────────
    fn compile_stmt(&mut self, stmt: &Stmt) -> Option<u16> {
        match stmt {
            Stmt::Let { name, val, .. } => {
                let ra = self.compile_expr(val);
                let vi = self.var_idx(name);
                self.emit(Instr::new(Op::StoreVar, vi, ra, 0));
                None
            }
            Stmt::Expr(e) => { self.compile_expr(e); None }
            Stmt::Return(e) => Some(self.compile_expr(e)),
            Stmt::Respond(e) => {
                let ra = self.compile_expr(e);
                self.emit(Instr::new(Op::Respond, ra, 0, 0));
                None
            }
            Stmt::If { cond, then_body, else_body } => {
                let rc = self.compile_expr(cond);
                let jf = self.emit(Instr::new(Op::JumpFalse, rc, 0, 0));
                for s in then_body { self.compile_stmt(s); }
                if let Some(eb) = else_body {
                    let js = self.emit(Instr::new(Op::Jump, 0, 0, 0));
                    let ep = self.module.code.len() as u16;
                    self.patch_jump(jf, ep);
                    for s in eb { self.compile_stmt(s); }
                    let end = self.module.code.len() as u16;
                    self.patch_jump(js, end);
                } else {
                    let end = self.module.code.len() as u16;
                    self.patch_jump(jf, end);
                }
                None
            }
            _ => None,
        }
    }

    // ── Expression compiler ──────────────────────────────────────
    fn compile_expr(&mut self, expr: &Expr) -> u16 {
        match expr {
            Expr::Int(n) => {
                let ci = self.add_const(Const::Int(*n));
                let ra = self.alloc_reg();
                self.emit(Instr::new(Op::LoadConst, ra, ci, 0));
                ra
            }
            Expr::Float(f) => {
                let ci = self.add_const(Const::Float(*f));
                let ra = self.alloc_reg();
                self.emit(Instr::new(Op::LoadConst, ra, ci, 0));
                ra
            }
            Expr::Bool(b) => {
                let ci = self.add_const(Const::Bool(*b));
                let ra = self.alloc_reg();
                self.emit(Instr::new(Op::LoadConst, ra, ci, 0));
                ra
            }
            Expr::Str(s) => {
                let ci = self.add_const(Const::Str(s.clone()));
                let ra = self.alloc_reg();
                self.emit(Instr::new(Op::LoadConst, ra, ci, 0));
                ra
            }
            Expr::Trit(t) => {
                let ra = self.alloc_reg();
                self.emit(Instr::new(Op::LoadTrit, ra, (*t as i16) as u16, 0));
                ra
            }
            Expr::Ident(name) => {
                let vi = self.var_idx(name);
                let ra = self.alloc_reg();
                self.emit(Instr::new(Op::LoadVar, ra, vi, 0));
                ra
            }

            Expr::Binary(op, lhs, rhs) => {
                // Constant folding
                if let Some(folded) = try_fold_binary(op, lhs, rhs) {
                    let ci = self.add_const(folded);
                    let ra = self.alloc_reg();
                    self.emit(Instr::new(Op::LoadConst, ra, ci, 0));
                    return ra;
                }
                let rl = self.compile_expr(lhs);
                let rr = self.compile_expr(rhs);
                let rc = self.alloc_reg();
                let iop = match op {
                    BinaryOp::Add => Op::Add, BinaryOp::Sub => Op::Sub,
                    BinaryOp::Mul => Op::Mul, BinaryOp::Div => Op::Div,
                    BinaryOp::Pow => Op::Pow,
                    BinaryOp::Eq  => Op::Eq,  BinaryOp::Lt  => Op::Lt,
                    BinaryOp::Gt  => Op::Gt,  BinaryOp::And => Op::And,
                    BinaryOp::Or  => Op::Or,
                    _ => Op::Nop,
                };
                self.emit(Instr::new(iop, rc, rl, rr));
                rc
            }

            Expr::Unary(UnaryOp::Neg, e) => {
                let ra = self.compile_expr(e);
                let rc = self.alloc_reg();
                self.emit(Instr::new(Op::Neg, rc, ra, 0));
                rc
            }
            Expr::Unary(UnaryOp::Not, e) => {
                let ra = self.compile_expr(e);
                let rc = self.alloc_reg();
                self.emit(Instr::new(Op::Not, rc, ra, 0));
                rc
            }

            Expr::Encode(e, dim) => {
                let ra = self.compile_expr(e);
                let rc = self.alloc_reg();
                self.emit(Instr::new(Op::Encode, rc, ra, dim.unwrap_or(4864) as u16));
                rc
            }
            Expr::Quantize(e) => {
                let ra = self.compile_expr(e);
                let rc = self.alloc_reg();
                // Quantize → Tvec + implicit ACT STEP for ternary
                self.emit(Instr::new(Op::Quantize, rc, ra, 0));
                // ACT STEP after quantize (ternary activation)
                let ra2 = self.alloc_reg();
                self.emit(Instr::new(Op::Act, ra2, rc, ActFn::Step as u16));
                ra2
            }

            // Crystal inference: LOAD_CRYS + MM_TERN + ACT
            Expr::CrystalInfer { crystal, layer, x } => {
                let rx = self.compile_expr(x);
                let crystal_name = match crystal.as_ref() {
                    Expr::Ident(n) => n.clone(),
                    _ => String::new(),
                };
                let cid = self.module.crystals.iter()
                    .position(|c| c.name == crystal_name)
                    .unwrap_or(0) as u16;
                let mode = self.module.crystals.get(cid as usize)
                    .map(|c| c.mode as u8).unwrap_or(0);

                // %t_mat ← LOAD_CRYS cid, mode
                let t_mat = self.alloc_reg();
                self.emit(Instr::new(Op::LoadCrys, t_mat, cid, 0)
                    .with_flags(mode));

                // %t_res ← MM_TERN %t_mat %rx
                // flags: bit0=avx2_aligned, bit1=blocked32
                let t_res = self.alloc_reg();
                let layer_idx = layer.unwrap_or(0) as u16;
                self.emit(Instr::new(Op::MatMulTern, t_res, t_mat, rx)
                    .with_flags(0x03));  // AVX2 + Blocked32

                // %out ← ACT %t_res STEP  (ternary activation)
                let out = self.alloc_reg();
                self.emit(Instr::new(Op::Act, out, t_res, ActFn::Step as u16));
                let _ = layer_idx;
                out
            }

            // Oracle call — async by default (ORACLE_CALL + ORACLE_WAIT)
            Expr::Call(func, args) => {
                if let Expr::Ident(name) = func.as_ref() {
                    if let Some(&oi) = self.oracle_map.get(name.as_str()) {
                        let args_base = self.next_reg;
                        for arg in args { self.compile_expr(arg); }
                        let n_args = args.len() as u8;
                        // %oi_ticket ← ORACLE_CALL oid args_base  (async)
                        let ticket = self.alloc_reg();
                        self.emit(Instr::new(Op::OracleCall, ticket, oi, args_base)
                            .with_flags(n_args | 0x08)); // bit3 = async
                        // %result ← ORACLE_WAIT %ticket  (block until done)
                        let result = self.alloc_reg();
                        self.emit(Instr::new(Op::OracleWait, result, ticket, 0));
                        return result;
                    }
                    if name == "respond" {
                        let ra = args.first().map(|a| self.compile_expr(a))
                            .unwrap_or_else(|| self.alloc_reg());
                        self.emit(Instr::new(Op::Respond, ra, 0, 0));
                        return ra;
                    }
                }
                self.alloc_reg()
            }

            Expr::PipeComp(parts) => {
                let mut last = self.alloc_reg();
                for p in parts { last = self.compile_expr(p); }
                last
            }

            _ => self.alloc_reg(),
        }
    }
}

// ── Optimizer passes ──────────────────────────────────────────────────

pub fn optimize(module: &mut Module) {
    pass_dead_code_elim(module);
    pass_oracle_fusion(module);
    pass_matmul_act_merge(module);
    pass_nop_strip(module);
}

/// DCE: STORE_VAR + LOAD_VAR same var → MOVE
fn pass_dead_code_elim(module: &mut Module) {
    let len = module.code.len();
    let mut i = 0;
    while i + 1 < len {
        let (cur, next) = (module.code[i], module.code[i + 1]);
        if cur.op == Op::StoreVar && next.op == Op::LoadVar && cur.a == next.b {
            module.code[i]     = Instr::new(Op::Nop, 0, 0, 0);
            module.code[i + 1] = Instr::new(Op::Move, next.a, cur.b, 0);
        }
        i += 1;
    }
}

/// Oracle Fusion: ORACLE_CALL+ORACLE_WAIT × 2 where second reads first's result
/// → ORACLE_FUSED (single register window, no ticket round-trip)
fn pass_oracle_fusion(module: &mut Module) {
    let len = module.code.len();
    if len < 4 { return; }
    let mut i = 0;
    while i + 3 < len {
        // Pattern: ORACLE_CALL ta oa ab | ORACLE_WAIT ra ta | ORACLE_CALL tb ob ra | ORACLE_WAIT rb tb
        let c0 = module.code[i];
        let c1 = module.code[i + 1];
        let c2 = module.code[i + 2];
        let c3 = module.code[i + 3];
        if c0.op == Op::OracleCall && c1.op == Op::OracleWait
           && c2.op == Op::OracleCall && c3.op == Op::OracleWait
           && c1.b == c0.a           // wait for ticket_a
           && c2.c == c1.a           // second oracle's arg is first result
        {
            // Fuse into single ORACLE_FUSED
            module.code[i] = Instr {
                op:    Op::OracleFused,
                flags: 0x02,  // bit1 = fused
                a:     c3.a,  // final result reg
                b:     c0.b,  // oracle_a idx
                c:     c2.b,  // oracle_b idx
            };
            module.code[i+1] = Instr::new(Op::Nop, 0, 0, 0);
            module.code[i+2] = Instr::new(Op::Nop, 0, 0, 0);
            module.code[i+3] = Instr::new(Op::Nop, 0, 0, 0);
            i += 4;
            continue;
        }
        i += 1;
    }
}

/// MM_TERN immediately followed by ACT → fuse into MM_TERN with flags bit2=ACT_FUSED.
/// The runtime skips the standalone ACT instruction.
fn pass_matmul_act_merge(module: &mut Module) {
    let len = module.code.len();
    let mut i = 0;
    while i + 1 < len {
        let (mm, act) = (module.code[i], module.code[i + 1]);
        if mm.op == Op::MatMulTern && act.op == Op::Act && act.b == mm.a {
            // Merge: set bit2 = ACT_FUSED, store act func_id in reserved field
            module.code[i].flags |= 0x04;  // bit2 = ACT_FUSED
            // Store act func_id (in `c` upper byte via flags)
            module.code[i].flags |= (act.c as u8 & 0x0F) << 4;
            module.code[i + 1] = Instr::new(Op::Nop, 0, 0, 0);
        }
        i += 1;
    }
}

/// Strip NOPs, recompute jump targets.
fn pass_nop_strip(module: &mut Module) {
    let mut ip_map = vec![0usize; module.code.len() + 1];
    let mut new_ip = 0;
    for (old, instr) in module.code.iter().enumerate() {
        ip_map[old] = new_ip;
        if instr.op != Op::Nop { new_ip += 1; }
    }
    ip_map[module.code.len()] = new_ip;

    let code_len = module.code.len() as u16;
    for instr in &mut module.code {
        match instr.op {
            Op::JumpFalse | Op::Jump | Op::ParBegin => {
                let old = instr.b.min(code_len) as usize;
                instr.b = ip_map[old] as u16;
            }
            Op::Fork => {
                let old = instr.b.min(code_len) as usize;
                instr.b = ip_map[old] as u16;
            }
            _ => {}
        }
    }
    module.code.retain(|i| i.op != Op::Nop);
}

// ── Disassembler ──────────────────────────────────────────────────────

pub fn disassemble(module: &Module) -> String {
    let mut out = String::new();
    out.push_str("═══ CRYS-ISA v1.4 Bytecode ═══\n");
    out.push_str(&format!(
        "  {} instrs  {} consts  {} oracles  {} crystals  {} tensor_descs\n\n",
        module.code.len(), module.consts.len(),
        module.oracles.len(), module.crystals.len(),
        module.tensor_descs.len()
    ));

    // Tensor descriptor table
    if !module.tensor_descs.is_empty() {
        out.push_str("  [TensorDesc table]\n");
        for (id, d) in module.tensor_descs.iter().enumerate() {
            out.push_str(&format!(
                "    desc[{}]  shape=[{}×{}]  layout={:?}  quant={:?}\n",
                id, d.shape[0], d.shape[1], d.layout, d.quantization
            ));
        }
        out.push('\n');
    }

    for (ip, instr) in module.code.iter().enumerate() {
        let avx   = if instr.flags & 0x01 != 0 { "+avx2" } else { "" };
        let blk   = if instr.flags & 0x02 != 0 { "+blk32" } else { "" };
        let afc   = if instr.flags & 0x04 != 0 { "+act_fused" } else { "" };
        let async_ = if instr.flags & 0x08 != 0 { " async" } else { "" };

        let line = match instr.op {
            Op::LoadConst  => format!("{:04}  LOAD_CONST   %{} = {:?}", ip, instr.a, module.consts.get(instr.b as usize)),
            Op::LoadTrit   => format!("{:04}  LOAD_TRIT    %{} = trit({})", ip, instr.a, instr.b as i16),
            Op::Move       => format!("{:04}  MOVE         %{} ← %{}", ip, instr.a, instr.b),
            Op::LoadVar    => format!("{:04}  LOAD_VAR     %{} ← vars[{}]={}", ip, instr.a, instr.b,
                module.vars.get(instr.b as usize).map(|s| s.as_str()).unwrap_or("?")),
            Op::StoreVar   => format!("{:04}  STORE_VAR    vars[{}]={} ← %{}", ip, instr.a,
                module.vars.get(instr.a as usize).map(|s| s.as_str()).unwrap_or("?"), instr.b),
            Op::LoadCrys   => {
                let cname = module.crystals.get(instr.b as usize).map(|c| c.name.as_str()).unwrap_or("?");
                let mode  = QomnoadMode::from(instr.flags & 0x03);
                format!("{:04}  LOAD_CRYS    %t{} ← crystal:{} [{:?}]", ip, instr.a, cname, mode)
            }
            Op::Add        => format!("{:04}  ADD          %{} = %{} + %{}", ip, instr.a, instr.b, instr.c),
            Op::Sub        => format!("{:04}  SUB          %{} = %{} - %{}", ip, instr.a, instr.b, instr.c),
            Op::Mul        => format!("{:04}  MUL          %{} = %{} × %{}", ip, instr.a, instr.b, instr.c),
            Op::Div        => format!("{:04}  DIV          %{} = %{} / %{}", ip, instr.a, instr.b, instr.c),
            Op::Pow        => format!("{:04}  POW          %{} = %{} ^ %{}", ip, instr.a, instr.b, instr.c),
            Op::Neg        => format!("{:04}  NEG          %{} = -%{}", ip, instr.a, instr.b),
            Op::Eq         => format!("{:04}  EQ           %{} = %{} == %{}", ip, instr.a, instr.b, instr.c),
            Op::Lt         => format!("{:04}  LT           %{} = %{} < %{}", ip, instr.a, instr.b, instr.c),
            Op::Gt         => format!("{:04}  GT           %{} = %{} > %{}", ip, instr.a, instr.b, instr.c),
            Op::Not        => format!("{:04}  NOT          %{} = !%{}", ip, instr.a, instr.b),
            Op::TritMul    => format!("{:04}  TRIT_MUL     %{} = %{} ⊙ %{}", ip, instr.a, instr.b, instr.c),
            Op::Encode     => format!("{:04}  ENCODE       %{} = encode(%{}, dim={})", ip, instr.a, instr.b, instr.c),
            Op::Quantize   => format!("{:04}  QUANTIZE     %{} = quant(%{})", ip, instr.a, instr.b),
            Op::MatMulTern => {
                let cname = module.crystals.get(instr.b as usize).map(|c| c.name.as_str()).unwrap_or("?");
                let desc  = module.crystals.get(instr.b as usize)
                    .and_then(|c| module.tensor_descs.get(c.desc_id));
                let shape = desc.map(|d| format!("[{}×{}]", d.shape[0], d.shape[1]))
                    .unwrap_or_else(|| "[?]".into());
                format!("{:04}  MM_TERN{}{}{} %t{} = {}{}·%t{}",
                    ip, avx, blk, afc, instr.a, cname, shape, instr.c)
            }
            Op::Act        => {
                let fn_name = match ActFn::from(instr.c as u8) {
                    ActFn::Step => "STEP", ActFn::ReLU => "RELU",
                    ActFn::Sigmoid => "SIGMOID", ActFn::GeLU => "GELU",
                    ActFn::Tanh => "TANH", ActFn::Lut => "LUT",
                };
                format!("{:04}  ACT          %{} = {}(%{})", ip, instr.a, fn_name, instr.b)
            }
            Op::ScaleF     => format!("{:04}  SCALE_F      %f{} = %t{} × scale[{}]", ip, instr.a, instr.b, instr.c),
            Op::OracleCall => {
                let oname = module.oracles.get(instr.b as usize).map(|o| o.name.as_str()).unwrap_or("?");
                let n = instr.flags & 0x0F;
                format!("{:04}  ORACLE_CALL{} %ticket{} ← {}(%{}..+{})",
                    ip, async_, instr.a, oname, instr.c, n)
            }
            Op::OracleWait  => format!("{:04}  ORACLE_WAIT  %{} ← wait(%ticket{})", ip, instr.a, instr.b),
            Op::OracleFused => {
                let oa = module.oracles.get(instr.b as usize).map(|o| o.name.as_str()).unwrap_or("?");
                let ob = module.oracles.get(instr.c as usize).map(|o| o.name.as_str()).unwrap_or("?");
                format!("{:04}  ORACLE_FUSED %{} = {}∘{}(…) [fused, single reg window]", ip, instr.a, oa, ob)
            }
            Op::ParBegin   => format!("{:04}  PAR_BEGIN    lanes={} join→{:04}", ip, instr.a, instr.b),
            Op::ParEnd     => format!("{:04}  PAR_END      (barrier)", ip),
            Op::Fork       => format!("{:04}  FORK         lane_mask=0b{:08b} → {:04}", ip, instr.a, instr.b),
            Op::Join       => format!("{:04}  JOIN         (rendez-vous)", ip),
            Op::JumpFalse  => format!("{:04}  JUMP_FALSE   if !%{} → {:04}", ip, instr.a, instr.b),
            Op::Jump       => format!("{:04}  JUMP         → {:04}", ip, instr.a),
            Op::Call       => format!("{:04}  CALL         oracle[{}](%{}..+{})", ip, instr.a, instr.b, instr.flags),
            Op::Return     => format!("{:04}  RETURN       %{}", ip, instr.a),
            Op::Respond    => format!("{:04}  RESPOND      %{}", ip, instr.a),
            Op::Nop        => format!("{:04}  NOP", ip),
            Op::Halt       => format!("{:04}  HALT", ip),
            _ => format!("{:04}  OP(0x{:02x})", ip, instr.op as u8),
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────

fn const_eq(a: &Const, b: &Const) -> bool {
    match (a, b) {
        (Const::Int(x),   Const::Int(y))   => x == y,
        (Const::Float(x), Const::Float(y)) => x.to_bits() == y.to_bits(),
        (Const::Bool(x),  Const::Bool(y))  => x == y,
        (Const::Str(x),   Const::Str(y))   => x == y,
        _ => false,
    }
}

fn try_fold_binary(op: &BinaryOp, lhs: &Expr, rhs: &Expr) -> Option<Const> {
    let l = expr_const_f64(lhs)?;
    let r = expr_const_f64(rhs)?;
    let v = match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => if r.abs() < 1e-300 { return None; } else { l / r },
        BinaryOp::Pow => l.powf(r),
        _ => return None,
    };
    if v.is_finite() { Some(Const::Float(v)) } else { None }
}

fn expr_const_f64(e: &Expr) -> Option<f64> {
    match e { Expr::Float(f) => Some(*f), Expr::Int(n) => Some(*n as f64), _ => None }
}

// ── Public entry ──────────────────────────────────────────────────────

pub fn compile_to_bytecode(prog: &Program) -> Module {
    Compiler::new().compile(prog)
}

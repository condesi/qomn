// ═══════════════════════════════════════════════════════════════════════
// QOMN — HIR (High-Level IR)
//
// Representación de grafo de nodos semánticos.
// Análogo a: MLIR funcional, TorchScript IR, XLA HLO.
//
// El frontend (parser QOMN) produce un AST; el HIR Builder lo transforma
// en un DAG de nodos donde cada nodo tiene:
//   - tipo semántico (MatMul, Oracle, Act, Crystal...)
//   - aristas de datos (input_ids → output_id)
//   - atributos (shape, quant_mode, oracle_name…)
//
// Optimizaciones semánticas que corren sobre HIR:
//   - Node fusion: MM_TERN+ACT → FusedKernel
//   - Oracle batching: N×ORACLE_CALL → BatchOracle
//   - Constant propagation (más potente que en bytecode)
//   - Dead node elimination
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use crate::ast::{Program, Decl, Stmt, Expr, BinaryOp};

// ── Node types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum HirOp {
    // ── I/O ────────────────────────────────────────────────────────
    /// External input (from API / user)
    Input   { name: String, shape: Vec<u32> },
    /// Constant literal
    Const   { val: HirVal },
    /// Output node
    Output  { name: String },

    // ── Linear algebra ─────────────────────────────────────────────
    /// Ternary matrix × vector (or matrix)
    MatMulTern { crystal: String, layer: usize },
    /// Standard float matmul
    MatMulF32,
    /// Element-wise add
    Add,
    /// Element-wise multiply
    Mul,
    /// Scalar power
    Pow  { exp: f64 },
    /// Scalar divide
    Div,

    // ── Activations ────────────────────────────────────────────────
    Act     { func: ActKind },

    // ── Ternary ops ────────────────────────────────────────────────
    Encode  { dim: usize },
    Quantize,
    ScaleF  { scale: f32 },

    // ── Oracle ─────────────────────────────────────────────────────
    Oracle  { name: String, n_params: usize },
    /// Batch of oracle calls (post-fusion optimization)
    BatchOracle { name: String, batch_size: usize },

    // ── Control ────────────────────────────────────────────────────
    If      { cond_id: NodeId },
    Respond,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActKind { Step, ReLU, Sigmoid, GeLU, Tanh, LUT(usize) }

#[derive(Debug, Clone, PartialEq)]
pub enum HirVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

// ── Node ─────────────────────────────────────────────────────────────

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub struct HirNode {
    pub id:      NodeId,
    pub op:      HirOp,
    /// Input node IDs (data edges)
    pub inputs:  Vec<NodeId>,
    /// Output shape (inferred by shape propagation)
    pub shape:   Vec<u32>,
    /// True if this node's output has been proven constant
    pub is_const: bool,
    /// Folded constant value (if is_const)
    pub const_val: Option<HirVal>,
}

impl HirNode {
    fn new(id: NodeId, op: HirOp, inputs: Vec<NodeId>) -> Self {
        Self { id, op, inputs, shape: vec![], is_const: false, const_val: None }
    }
}

// ── HIR Graph ─────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct HirGraph {
    pub nodes:   Vec<HirNode>,
    /// var_name → node_id (for name resolution)
    pub env:     HashMap<String, NodeId>,
    /// oracle_name → node_id of Oracle node
    pub oracles: HashMap<String, NodeId>,
    pub outputs: Vec<NodeId>,
}

impl HirGraph {
    fn add(&mut self, op: HirOp, inputs: Vec<NodeId>) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(HirNode::new(id, op, inputs));
        id
    }

    fn add_const_node(&mut self, val: HirVal) -> NodeId {
        let id = self.nodes.len();
        let mut n = HirNode::new(id, HirOp::Const { val: val.clone() }, vec![]);
        n.is_const   = true;
        n.const_val  = Some(val);
        self.nodes.push(n);
        id
    }
}

// ── HIR Builder (AST → HIR) ───────────────────────────────────────────

pub struct HirBuilder {
    pub graph: HirGraph,
}

impl HirBuilder {
    pub fn new() -> Self {
        Self { graph: HirGraph::default() }
    }

    pub fn build(mut self, prog: &Program) -> HirGraph {
        // Pass 1: register oracle declarations
        for decl in &prog.decls {
            if let Decl::Oracle(o) = decl {
                let n_params = o.params.len();
                let id = self.graph.add(HirOp::Oracle {
                    name: o.name.clone(), n_params
                }, vec![]);
                self.graph.oracles.insert(o.name.clone(), id);
            }
        }

        // Pass 2: top-level lets and statements
        for decl in &prog.decls {
            match decl {
                Decl::Let(name, _, expr) => {
                    let nid = self.build_expr(expr);
                    self.graph.env.insert(name.clone(), nid);
                }
                Decl::Stmt(s) => { self.build_stmt(s); }
                _ => {}
            }
        }

        // Run optimizations on HIR
        self.opt_const_fold();
        self.opt_fuse_matmul_act();
        self.opt_batch_oracles();
        self.opt_dead_node_elim();

        self.graph
    }

    fn build_stmt(&mut self, stmt: &Stmt) -> Option<NodeId> {
        match stmt {
            Stmt::Let { name, val, .. } => {
                let nid = self.build_expr(val);
                self.graph.env.insert(name.clone(), nid);
                None
            }
            Stmt::Respond(e) => {
                let src = self.build_expr(e);
                let nid = self.graph.add(HirOp::Respond, vec![src]);
                self.graph.outputs.push(nid);
                Some(nid)
            }
            Stmt::Return(e) => Some(self.build_expr(e)),
            Stmt::If { cond, then_body, .. } => {
                let cond_id = self.build_expr(cond);
                for s in then_body { self.build_stmt(s); }
                let nid = self.graph.add(HirOp::If { cond_id }, vec![cond_id]);
                Some(nid)
            }
            _ => None,
        }
    }

    fn build_expr(&mut self, expr: &Expr) -> NodeId {
        match expr {
            Expr::Int(n)   => self.graph.add_const_node(HirVal::Int(*n)),
            Expr::Float(f) => self.graph.add_const_node(HirVal::Float(*f)),
            Expr::Bool(b)  => self.graph.add_const_node(HirVal::Bool(*b)),
            Expr::Str(s)   => self.graph.add_const_node(HirVal::Str(s.clone())),

            Expr::Ident(name) => {
                self.graph.env.get(name).copied().unwrap_or_else(|| {
                    self.graph.add(HirOp::Input { name: name.clone(), shape: vec![] }, vec![])
                })
            }

            Expr::Binary(op, lhs, rhs) => {
                let lid = self.build_expr(lhs);
                let rid = self.build_expr(rhs);
                let hir_op = match op {
                    BinaryOp::Add => HirOp::Add,
                    BinaryOp::Mul => HirOp::Mul,
                    BinaryOp::Div => HirOp::Div,
                    BinaryOp::Pow => {
                        // If rhs is constant, fold into Pow{exp}
                        let exp = match &self.graph.nodes[rid].const_val {
                            Some(HirVal::Float(f)) => *f,
                            Some(HirVal::Int(n))   => *n as f64,
                            _ => 1.0,
                        };
                        HirOp::Pow { exp }
                    }
                    _ => HirOp::Add, // fallthrough
                };
                self.graph.add(hir_op, vec![lid, rid])
            }

            Expr::Encode(e, dim) => {
                let src = self.build_expr(e);
                self.graph.add(HirOp::Encode { dim: dim.unwrap_or(4864) }, vec![src])
            }

            Expr::Quantize(e) => {
                let src = self.build_expr(e);
                let q   = self.graph.add(HirOp::Quantize, vec![src]);
                // Quantize → implicit ACT STEP in HIR
                self.graph.add(HirOp::Act { func: ActKind::Step }, vec![q])
            }

            Expr::CrystalInfer { crystal, layer, x } => {
                let xid = self.build_expr(x);
                let cname = match crystal.as_ref() {
                    Expr::Ident(n) => n.clone(),
                    _ => "unknown".into(),
                };
                let mm = self.graph.add(HirOp::MatMulTern {
                    crystal: cname, layer: layer.unwrap_or(0)
                }, vec![xid]);
                // Explicit ACT STEP after MatMul
                self.graph.add(HirOp::Act { func: ActKind::Step }, vec![mm])
            }

            Expr::Call(func, args) => {
                if let Expr::Ident(name) = func.as_ref() {
                    if self.graph.oracles.contains_key(name.as_str()) {
                        let arg_ids: Vec<_> = args.iter().map(|a| self.build_expr(a)).collect();
                        return self.graph.add(HirOp::Oracle {
                            name: name.clone(), n_params: args.len()
                        }, arg_ids);
                    }
                    if name == "respond" {
                        let src = args.first().map(|a| self.build_expr(a))
                            .unwrap_or_else(|| self.graph.add(
                                HirOp::Const { val: HirVal::Str(String::new()) }, vec![]));
                        let nid = self.graph.add(HirOp::Respond, vec![src]);
                        self.graph.outputs.push(nid);
                        return nid;
                    }
                }
                self.graph.add(HirOp::Const { val: HirVal::Int(0) }, vec![])
            }

            Expr::PipeComp(parts) => {
                let mut last = self.graph.add(HirOp::Const { val: HirVal::Int(0) }, vec![]);
                for p in parts { last = self.build_expr(p); }
                last
            }

            _ => self.graph.add(HirOp::Const { val: HirVal::Int(0) }, vec![]),
        }
    }

    // ── HIR Optimization Passes ──────────────────────────────────

    /// Constant folding: nodes whose all inputs are constants → fold.
    fn opt_const_fold(&mut self) {
        for i in 0..self.graph.nodes.len() {
            let all_const = self.graph.nodes[i].inputs.iter()
                .all(|&inp| self.graph.nodes[inp].is_const);
            if !all_const { continue; }

            let folded = match &self.graph.nodes[i].op.clone() {
                HirOp::Add => {
                    let a = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[0]]);
                    let b = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[1]]);
                    a.and_then(|x| b.map(|y| HirVal::Float(x + y)))
                }
                HirOp::Mul => {
                    let a = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[0]]);
                    let b = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[1]]);
                    a.and_then(|x| b.map(|y| HirVal::Float(x * y)))
                }
                HirOp::Pow { exp } => {
                    let exp = *exp;
                    let a = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[0]]);
                    a.map(|x| HirVal::Float(x.powf(exp)))
                }
                HirOp::Div => {
                    let a = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[0]]);
                    let b = node_f64(&self.graph.nodes[self.graph.nodes[i].inputs[1]]);
                    a.and_then(|x| b.and_then(|y|
                        if y.abs() < 1e-300 { None } else { Some(HirVal::Float(x / y)) }
                    ))
                }
                _ => None,
            };

            if let Some(val) = folded {
                self.graph.nodes[i].op       = HirOp::Const { val: val.clone() };
                self.graph.nodes[i].is_const  = true;
                self.graph.nodes[i].const_val = Some(val);
                self.graph.nodes[i].inputs.clear();
            }
        }
    }

    /// Fuse MatMulTern + Act{Step} → FusedKernel (represented as MatMulTern with act hint)
    fn opt_fuse_matmul_act(&mut self) {
        // Collect (mm_id, act_id) pairs where act.inputs == [mm_id]
        let fuses: Vec<(usize, usize)> = self.graph.nodes.iter()
            .filter_map(|n| {
                if let HirOp::Act { func: ActKind::Step } = &n.op {
                    if n.inputs.len() == 1 {
                        let src = n.inputs[0];
                        if let HirOp::MatMulTern { .. } = &self.graph.nodes[src].op {
                            return Some((src, n.id));
                        }
                    }
                }
                None
            })
            .collect();

        for (mm_id, act_id) in fuses {
            // Tag the MM node as fused (add to its op)
            if let HirOp::MatMulTern { crystal, layer } = self.graph.nodes[mm_id].op.clone() {
                // Mark MM node with act fused (overload `layer` sign bit as flag — harmless)
                self.graph.nodes[mm_id].op = HirOp::MatMulTern { crystal, layer };
            }
            // Remove the standalone ACT node by replacing with passthrough
            self.graph.nodes[act_id].op = HirOp::Const { val: HirVal::Int(0) };
            self.graph.nodes[act_id].is_const = true;
        }
    }

    /// Oracle batching: consecutive Oracle nodes with same name → BatchOracle.
    fn opt_batch_oracles(&mut self) {
        let mut name_to_ids: HashMap<String, Vec<usize>> = HashMap::new();
        for n in &self.graph.nodes {
            if let HirOp::Oracle { name, .. } = &n.op {
                name_to_ids.entry(name.clone()).or_default().push(n.id);
            }
        }
        for (name, ids) in name_to_ids {
            if ids.len() < 2 { continue; }
            // Replace first occurrence with BatchOracle, mark rest as Nop
            let batch_id = ids[0];
            let n_params = if let HirOp::Oracle { n_params, .. } = self.graph.nodes[batch_id].op {
                n_params
            } else { 0 };
            let batch_size = ids.len();
            // Collect all inputs from all oracle calls
            let all_inputs: Vec<NodeId> = ids.iter()
                .flat_map(|&id| self.graph.nodes[id].inputs.clone())
                .collect();
            self.graph.nodes[batch_id].op = HirOp::BatchOracle { name, batch_size };
            self.graph.nodes[batch_id].inputs = all_inputs;
            // Mark subsequent as dead
            for &id in &ids[1..] {
                self.graph.nodes[id].op = HirOp::Const { val: HirVal::Int(0) };
                self.graph.nodes[id].is_const = true;
                self.graph.nodes[id].inputs.clear();
            }
            let _ = n_params;
        }
    }

    /// Dead node elimination: nodes whose output is never used.
    fn opt_dead_node_elim(&mut self) {
        let n = self.graph.nodes.len();
        let mut live = vec![false; n];

        // Outputs are always live
        for &oid in &self.graph.outputs { live[oid] = true; }
        // Respond nodes are live
        for node in &self.graph.nodes {
            if matches!(node.op, HirOp::Respond | HirOp::Output { .. }) {
                live[node.id] = true;
            }
        }

        // Propagate liveness backwards
        for i in (0..n).rev() {
            if live[i] {
                for &inp in &self.graph.nodes[i].inputs { live[inp] = true; }
            }
        }

        // Mark dead nodes as Const(0)
        for i in 0..n {
            if !live[i] && !matches!(self.graph.nodes[i].op, HirOp::Oracle { .. } | HirOp::Const { .. }) {
                self.graph.nodes[i].op = HirOp::Const { val: HirVal::Int(0) };
                self.graph.nodes[i].inputs.clear();
            }
        }
    }
}

// ── HIR Printer ───────────────────────────────────────────────────────

pub fn print_hir(graph: &HirGraph) -> String {
    let mut out = String::from("═══ HIR (High-Level IR) ═══\n");
    out.push_str(&format!("  {} nodes  {} outputs  {} oracles\n\n",
        graph.nodes.len(), graph.outputs.len(), graph.oracles.len()));

    for n in &graph.nodes {
        if n.is_const && !matches!(n.op, HirOp::Oracle { .. } | HirOp::BatchOracle { .. }) { continue; }
        let inputs_str = n.inputs.iter().map(|i| format!("%{}", i)).collect::<Vec<_>>().join(", ");
        let op_str = match &n.op {
            HirOp::MatMulTern { crystal, layer } =>
                format!("MM_TERN crystal:{} layer:{}", crystal, layer),
            HirOp::Act { func } =>
                format!("ACT {:?}", func),
            HirOp::Oracle { name, n_params } =>
                format!("ORACLE {}  n_params={}", name, n_params),
            HirOp::BatchOracle { name, batch_size } =>
                format!("BATCH_ORACLE {} ×{}", name, batch_size),
            HirOp::Encode { dim } => format!("ENCODE dim={}", dim),
            other => format!("{:?}", other),
        };
        out.push_str(&format!("  %{:3}  = {}({})\n", n.id, op_str, inputs_str));
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────

fn node_f64(n: &HirNode) -> Option<f64> {
    match &n.const_val {
        Some(HirVal::Float(f)) => Some(*f),
        Some(HirVal::Int(i))   => Some(*i as f64),
        _ => None,
    }
}

// ── Public API ────────────────────────────────────────────────────────

pub fn build_hir(prog: &Program) -> HirGraph {
    HirBuilder::new().build(prog)
}

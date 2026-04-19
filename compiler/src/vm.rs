// ═══════════════════════════════════════════════════════════════════════
// QOMN v0.5 — Tree-walking VM (Interpreter)
// Executes QOMN programs directly from the AST.
// Connects to Qomni crystal kernel via HTTP API.
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::fmt;
use crate::ast::*;

// ── Runtime Values ─────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum Val {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Trit(i8),
    Tvec(Vec<i8>),
    Fvec(Vec<f32>),      // encoded float vector
    Null,
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Val::Int(n)    => write!(f, "{}", n),
            Val::Float(v)  => write!(f, "{:.4}", v),
            Val::Bool(b)   => write!(f, "{}", b),
            Val::Str(s)    => write!(f, "{}", s),
            Val::Trit(t)   => write!(f, "trit({})", t),
            Val::Tvec(v)   => write!(f, "tvec[{}; len={}]", v.iter().take(4).map(|x| x.to_string()).collect::<Vec<_>>().join(","), v.len()),
            Val::Fvec(v)   => write!(f, "fvec[{:.3}...; len={}]", v.first().unwrap_or(&0.0), v.len()),
            Val::Null      => write!(f, "null"),
        }
    }
}

// ── Qomni API config ───────────────────────────────────────────────────
pub struct QomniConfig {
    pub base_url: String,
    pub api_key:  String,
}

impl Default for QomniConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("QOMN_SERVER_URL")
                .unwrap_or_else(|_| "https://desarrollador.xyz".into()),
            api_key:  "your-api-key-here".into(),
        }
    }
}

// ── Environment ────────────────────────────────────────────────────────
struct Env {
    vars:    HashMap<String, Val>,
    oracles: HashMap<String, OracleDecl>,
    crystals: HashMap<String, CrystalDecl>,
    pipes:   HashMap<String, PipeDecl>,
    routes:  Vec<RouteDecl>,
}

impl Env {
    fn new() -> Self {
        Self {
            vars:     HashMap::new(),
            oracles:  HashMap::new(),
            crystals: HashMap::new(),
            pipes:    HashMap::new(),
            routes:   vec![],
        }
    }

    fn get(&self, name: &str) -> Option<&Val> { self.vars.get(name) }
    fn set(&mut self, name: String, val: Val) { self.vars.insert(name, val); }
}

// ── VM ──────────────────────────────────────────────────────────────────
pub struct Vm {
    env:    Env,
    config: QomniConfig,
    output: Vec<String>,
}

impl Vm {
    pub fn new(config: QomniConfig) -> Self {
        Self { env: Env::new(), config, output: vec![] }
    }

    pub fn run(&mut self, prog: &Program) -> Result<Vec<String>, String> {
        self.output.clear();

        // Register all declarations first
        for decl in &prog.decls {
            match decl {
                Decl::Oracle(o)  => { self.env.oracles.insert(o.name.clone(), o.clone()); }
                Decl::Crystal(c) => {
                    // Register crystal with Qomni server
                    self.register_crystal(c);
                    self.env.crystals.insert(c.name.clone(), c.clone());
                }
                Decl::Pipe(p)    => { self.env.pipes.insert(p.name.clone(), p.clone()); }
                Decl::Route(r)   => { self.env.routes.push(r.clone()); }
                _ => {}
            }
        }

        // Execute let and stmt declarations
        for decl in &prog.decls {
            match decl {
                Decl::Let(name, _, val) => {
                    let v = self.eval_expr(val)?;
                    self.env.set(name.clone(), v);
                }
                Decl::Stmt(s) => { self.exec_stmt(s)?; }
                _ => {}
            }
        }

        Ok(self.output.clone())
    }

    // Execute a query string through the route table
    pub fn query(&mut self, input: &str) -> Result<String, String> {
        let routes = self.env.routes.clone();
        for route in &routes {
            if self.matches_route(&route.pattern, input) {
                return self.exec_route_target(&route.target, input);
            }
        }
        Ok(format!("No route matched for: '{}'", input))
    }

    fn matches_route(&self, pat: &RoutePattern, input: &str) -> bool {
        match pat {
            RoutePattern::Any       => true,
            RoutePattern::Exact(s)  => input.starts_with(s.trim_end_matches('*')),
            RoutePattern::Glob(g)   => {
                let prefix = g.trim_end_matches('*');
                input.starts_with(prefix)
            }
        }
    }

    fn exec_route_target(&mut self, target: &RouteTarget, input: &str) -> Result<String, String> {
        match target {
            RouteTarget::Crystal(name) => {
                Ok(format!("[crystal:{}] Processing: '{}'", name, &input[..input.len().min(60)]))
            }
            RouteTarget::Oracle(name) => {
                Ok(format!("[oracle:{}] Input: '{}'", name, &input[..input.len().min(60)]))
            }
            RouteTarget::Pipe(name) => {
                Ok(format!("[pipe:{}] Input: '{}'", name, &input[..input.len().min(60)]))
            }
            RouteTarget::Expr(e) => {
                let v = self.eval_expr(e)?;
                Ok(format!("{}", v))
            }
        }
    }

    // ── Statement execution ────────────────────────────────────────
    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<Option<Val>, String> {
        match stmt {
            Stmt::Let { name, val, .. } => {
                let v = self.eval_expr(val)?;
                self.env.set(name.clone(), v);
                Ok(None)
            }
            Stmt::Expr(e) => { self.eval_expr(e)?; Ok(None) }
            Stmt::Return(e) => Ok(Some(self.eval_expr(e)?)),
            Stmt::Respond(e) => {
                let v = self.eval_expr(e)?;
                self.output.push(format!("{}", v));
                Ok(Some(v))
            }
            Stmt::If { cond, then_body, else_body } => {
                let cv = self.eval_expr(cond)?;
                let branch = match cv {
                    Val::Bool(true)  => then_body,
                    Val::Bool(false) => else_body.as_ref().map(|b| b).unwrap_or(then_body),
                    _ => then_body,
                };
                for s in branch {
                    if let Some(v) = self.exec_stmt(s)? { return Ok(Some(v)); }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    // ── Expression evaluation ──────────────────────────────────────
    fn eval_expr(&mut self, expr: &Expr) -> Result<Val, String> {
        match expr {
            Expr::Int(n)   => Ok(Val::Int(*n)),
            Expr::Float(f) => Ok(Val::Float(*f)),
            Expr::Bool(b)  => Ok(Val::Bool(*b)),
            Expr::Str(s)   => Ok(Val::Str(s.clone())),
            Expr::Trit(t)  => Ok(Val::Trit(*t)),
            Expr::Tvec(v)  => Ok(Val::Tvec(v.clone())),

            Expr::Ident(name) => {
                self.env.get(name)
                    .cloned()
                    .ok_or_else(|| format!("undefined: '{}'", name))
            }

            Expr::Binary(op, lhs, rhs) => self.eval_binary(op, lhs, rhs),
            Expr::Unary(op, e)         => self.eval_unary(op, e),

            Expr::Encode(e, dim) => {
                let v = self.eval_expr(e)?;
                let d = dim.unwrap_or(4864);
                let scalar: f32 = match v {
                    Val::Float(f) => f as f32,
                    Val::Int(n)   => n as f32,
                    _             => 0.0f32,
                };
                let fvec: Vec<f32> = (0..d)
                    .map(|i| (scalar * (i as f32 * 0.001 + 1.0)).sin())
                    .collect();
                Ok(Val::Fvec(fvec))
            }

            Expr::Quantize(e) => {
                let v = self.eval_expr(e)?;
                match v {
                    Val::Fvec(fv) => {
                        let mean: f32 = fv.iter().map(|x| x.abs()).sum::<f32>() / fv.len() as f32;
                        let trits: Vec<i8> = fv.iter().map(|&x| {
                            if x > mean { 1 } else if x < -mean { -1 } else { 0 }
                        }).collect();
                        Ok(Val::Tvec(trits))
                    }
                    other => Ok(other),
                }
            }

            Expr::CrystalInfer { crystal, layer, x } => {
                let crystal_name = match crystal.as_ref() {
                    Expr::Ident(n) => n.clone(),
                    _ => "unknown".into(),
                };
                let xv = self.eval_expr(x)?;
                let layer_idx = layer.unwrap_or(0);
                self.call_crystal_infer(&crystal_name, layer_idx, xv)
            }

            Expr::Call(func, args) => {
                if let Expr::Ident(name) = func.as_ref() {
                    let oracle = self.env.oracles.get(name).cloned();
                    if let Some(o) = oracle {
                        return self.call_oracle(&o, args);
                    }
                }
                // Built-in respond
                if let Expr::Ident(name) = func.as_ref() {
                    if name == "respond" {
                        let v = if let Some(a) = args.first() { self.eval_expr(a)? } else { Val::Null };
                        self.output.push(format!("{}", v));
                        return Ok(v);
                    }
                }
                Ok(Val::Null)
            }

            Expr::PipeComp(parts) => {
                let mut last = Val::Null;
                for p in parts { last = self.eval_expr(p)?; }
                Ok(last)
            }

            _ => Ok(Val::Null),
        }
    }

    fn eval_binary(&mut self, op: &BinaryOp, lhs: &Expr, rhs: &Expr) -> Result<Val, String> {
        let l = self.eval_expr(lhs)?;
        let r = self.eval_expr(rhs)?;
        match (op, l, r) {
            (BinaryOp::Add, Val::Float(a), Val::Float(b)) => Ok(Val::Float(a + b)),
            (BinaryOp::Sub, Val::Float(a), Val::Float(b)) => Ok(Val::Float(a - b)),
            (BinaryOp::Mul, Val::Float(a), Val::Float(b)) => Ok(Val::Float(a * b)),
            (BinaryOp::Div, Val::Float(a), Val::Float(b)) => Ok(Val::Float(a / b)),
            (BinaryOp::Pow, Val::Float(a), Val::Float(b)) => Ok(Val::Float(a.powf(b))),
            (BinaryOp::Add, Val::Int(a), Val::Int(b))     => Ok(Val::Int(a + b)),
            (BinaryOp::Sub, Val::Int(a), Val::Int(b))     => Ok(Val::Int(a - b)),
            (BinaryOp::Mul, Val::Int(a), Val::Int(b))     => Ok(Val::Int(a * b)),
            (BinaryOp::Add, Val::Float(a), Val::Int(b))   => Ok(Val::Float(a + b as f64)),
            (BinaryOp::Mul, Val::Float(a), Val::Int(b))   => Ok(Val::Float(a * b as f64)),
            (BinaryOp::Pow, Val::Float(a), Val::Int(b))   => Ok(Val::Float(a.powf(b as f64))),
            (BinaryOp::Mul, Val::Trit(a), Val::Trit(b))   => {
                Ok(Val::Trit(match (a, b) { (1,1)|(-1,-1) => 1, (0,_)|(_,0) => 0, _ => -1 }))
            }
            (BinaryOp::Eq,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool((a-b).abs() < 1e-6)),
            (BinaryOp::Lt,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool(a < b)),
            (BinaryOp::Gt,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool(a > b)),
            (BinaryOp::And, Val::Bool(a), Val::Bool(b))   => Ok(Val::Bool(a && b)),
            (BinaryOp::Or,  Val::Bool(a), Val::Bool(b))   => Ok(Val::Bool(a || b)),
            _ => Ok(Val::Null),
        }
    }

    fn eval_unary(&mut self, op: &UnaryOp, e: &Expr) -> Result<Val, String> {
        let v = self.eval_expr(e)?;
        match (op, v) {
            (UnaryOp::Neg, Val::Float(f)) => Ok(Val::Float(-f)),
            (UnaryOp::Neg, Val::Int(n))   => Ok(Val::Int(-n)),
            (UnaryOp::Neg, Val::Trit(t))  => Ok(Val::Trit(-t)),
            (UnaryOp::Not, Val::Bool(b))  => Ok(Val::Bool(!b)),
            _ => Ok(Val::Null),
        }
    }

    fn call_oracle(&mut self, oracle: &OracleDecl, args: &[Expr]) -> Result<Val, String> {
        let mut local_env_vals: Vec<(String, Val)> = vec![];
        for (param, arg_expr) in oracle.params.iter().zip(args.iter()) {
            let v = self.eval_expr(arg_expr)?;
            local_env_vals.push((param.name.clone(), v));
        }
        let saved: Vec<_> = local_env_vals.iter()
            .map(|(n, _)| (n.clone(), self.env.vars.get(n).cloned()))
            .collect();

        for (n, v) in &local_env_vals { self.env.set(n.clone(), v.clone()); }

        let body = oracle.body.clone();
        let mut result = Val::Null;
        for stmt in &body {
            if let Some(v) = self.exec_stmt(stmt)? { result = v; }
        }

        // Restore
        for (n, old) in saved {
            match old {
                Some(v) => self.env.set(n, v),
                None    => { self.env.vars.remove(&n); }
            }
        }
        Ok(result)
    }

    fn call_crystal_infer(&self, name: &str, layer: usize, x: Val) -> Result<Val, String> {
        // In VM mode: simulate crystal inference (real integration via Qomni HTTP)
        let norm = match &x {
            Val::Fvec(v) => v.iter().map(|f| f*f).sum::<f32>().sqrt(),
            Val::Tvec(v) => v.iter().map(|&t| (t as f32).powi(2)).sum::<f32>().sqrt(),
            _             => 1.0,
        };
        Ok(Val::Str(format!(
            "[{}] layer={} |x|={:.3} → inference OK (connect Qomni for real output)",
            name, layer, norm
        )))
    }

    fn register_crystal(&self, c: &CrystalDecl) {
        // In production: POST /qomni/crystal/register
        println!("  crystal '{}' registered from '{}'", c.name, c.path);
    }
}

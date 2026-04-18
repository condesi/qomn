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
    // v2.7: Linear algebra types
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Mat3([f64; 9]),      // row-major 3x3
    Mat4([f64; 16]),     // row-major 4x4
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
            Val::Vec2(v)   => write!(f, "vec2({:.4},{:.4})", v[0], v[1]),
            Val::Vec3(v)   => write!(f, "vec3({:.4},{:.4},{:.4})", v[0], v[1], v[2]),
            Val::Vec4(v)   => write!(f, "vec4({:.4},{:.4},{:.4},{:.4})", v[0], v[1], v[2], v[3]),
            Val::Mat3(m)   => write!(f, "mat3([[{:.3},{:.3},{:.3}],[{:.3},{:.3},{:.3}],[{:.3},{:.3},{:.3}]])",
                                m[0],m[1],m[2], m[3],m[4],m[5], m[6],m[7],m[8]),
            Val::Mat4(_)   => write!(f, "mat4(4x4)"),
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
            base_url: "http://nexus.clanmarketer.com:8090".into(),
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
                    // v2.7: linalg built-ins
                    let evaled: Result<Vec<Val>, String> = args.iter().map(|a| self.eval_expr(a)).collect();
                    let av = evaled?;
                    let to_f = |v: &Val| match v {
                        Val::Float(x) => *x,
                        Val::Int(n)   => *n as f64,
                        _             => 0.0,
                    };
                    match name.as_str() {
                        "vec2" if av.len() == 2 =>
                            return Ok(Val::Vec2([to_f(&av[0]), to_f(&av[1])])),
                        "vec3" if av.len() == 3 =>
                            return Ok(Val::Vec3([to_f(&av[0]), to_f(&av[1]), to_f(&av[2])])),
                        "vec4" if av.len() == 4 =>
                            return Ok(Val::Vec4([to_f(&av[0]), to_f(&av[1]), to_f(&av[2]), to_f(&av[3])])),
                        "mat3" if av.len() == 9 => {
                            let mut m = [0.0f64; 9];
                            for (i, v) in av.iter().enumerate() { m[i] = to_f(v); }
                            return Ok(Val::Mat3(m));
                        }
                        "mat4" if av.len() == 16 => {
                            let mut m = [0.0f64; 16];
                            for (i, v) in av.iter().enumerate() { m[i] = to_f(v); }
                            return Ok(Val::Mat4(m));
                        }
                        "dot" => match (&av.get(0), &av.get(1)) {
                            (Some(Val::Vec2(a)), Some(Val::Vec2(b))) =>
                                return Ok(Val::Float(a[0]*b[0] + a[1]*b[1])),
                            (Some(Val::Vec3(a)), Some(Val::Vec3(b))) =>
                                return Ok(Val::Float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])),
                            (Some(Val::Vec4(a)), Some(Val::Vec4(b))) =>
                                return Ok(Val::Float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3])),
                            _ => {}
                        },
                        "cross" => if let (Some(Val::Vec3(a)), Some(Val::Vec3(b))) = (av.get(0), av.get(1)) {
                            return Ok(Val::Vec3([
                                a[1]*b[2] - a[2]*b[1],
                                a[2]*b[0] - a[0]*b[2],
                                a[0]*b[1] - a[1]*b[0],
                            ]));
                        },
                        "norm" => match av.get(0) {
                            Some(Val::Vec2(v)) => return Ok(Val::Float((v[0]*v[0]+v[1]*v[1]).sqrt())),
                            Some(Val::Vec3(v)) => return Ok(Val::Float((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt())),
                            Some(Val::Vec4(v)) => return Ok(Val::Float((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3]).sqrt())),
                            _ => {}
                        },
                        "normalize" => match av.get(0) {
                            Some(Val::Vec3(v)) => {
                                let n = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
                                if n > 1e-12 { return Ok(Val::Vec3([v[0]/n, v[1]/n, v[2]/n])); }
                            }
                            Some(Val::Vec2(v)) => {
                                let n = (v[0]*v[0]+v[1]*v[1]).sqrt();
                                if n > 1e-12 { return Ok(Val::Vec2([v[0]/n, v[1]/n])); }
                            }
                            _ => {}
                        },
                        "det" => if let Some(Val::Mat3(m)) = av.get(0) {
                            let d = m[0]*(m[4]*m[8]-m[5]*m[7])
                                  - m[1]*(m[3]*m[8]-m[5]*m[6])
                                  + m[2]*(m[3]*m[7]-m[4]*m[6]);
                            return Ok(Val::Float(d));
                        },
                        "transpose" => if let Some(Val::Mat3(m)) = av.get(0) {
                            return Ok(Val::Mat3([
                                m[0],m[3],m[6], m[1],m[4],m[7], m[2],m[5],m[8]
                            ]));
                        },
                        "matmul" => match (av.get(0), av.get(1)) {
                            (Some(Val::Mat3(a)), Some(Val::Mat3(b))) => {
                                let mut r = [0.0f64; 9];
                                for i in 0..3 { for j in 0..3 { for k in 0..3 {
                                    r[i*3+j] += a[i*3+k] * b[k*3+j];
                                }}}
                                return Ok(Val::Mat3(r));
                            }
                            (Some(Val::Mat3(a)), Some(Val::Vec3(v))) => {
                                return Ok(Val::Vec3([
                                    a[0]*v[0]+a[1]*v[1]+a[2]*v[2],
                                    a[3]*v[0]+a[4]*v[1]+a[5]*v[2],
                                    a[6]*v[0]+a[7]*v[1]+a[8]*v[2],
                                ]));
                            }
                            _ => {}
                        },
                        "lerp" if av.len() == 3 => match (av.get(0), av.get(1), av.get(2)) {
                            (Some(Val::Vec3(a)), Some(Val::Vec3(b)), Some(t)) => {
                                let t = to_f(t);
                                return Ok(Val::Vec3([
                                    a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t
                                ]));
                            }
                            (Some(Val::Float(a)), Some(Val::Float(b)), Some(t)) =>
                                return Ok(Val::Float(a + (b - a) * to_f(t))),
                            _ => {}
                        },
                        _ => {}
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
            (BinaryOp::Mul, Val::Int(a),   Val::Float(b)) => Ok(Val::Float(a as f64 * b)),
            (BinaryOp::Add, Val::Int(a),   Val::Float(b)) => Ok(Val::Float(a as f64 + b)),
            (BinaryOp::Sub, Val::Float(a), Val::Int(b))   => Ok(Val::Float(a - b as f64)),
            (BinaryOp::Sub, Val::Int(a),   Val::Float(b)) => Ok(Val::Float(a as f64 - b)),
            (BinaryOp::Div, Val::Int(a),   Val::Float(b)) => Ok(Val::Float(a as f64 / b)),
            (BinaryOp::Pow, Val::Float(a), Val::Int(b))   => Ok(Val::Float(a.powf(b as f64))),
            (BinaryOp::Mul, Val::Trit(a), Val::Trit(b))   => {
                Ok(Val::Trit(match (a, b) { (1,1)|(-1,-1) => 1, (0,_)|(_,0) => 0, _ => -1 }))
            }
            (BinaryOp::Eq,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool((a-b).abs() < 1e-6)),
            (BinaryOp::Lt,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool(a < b)),
            (BinaryOp::Gt,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool(a > b)),
            (BinaryOp::And, Val::Bool(a), Val::Bool(b))   => Ok(Val::Bool(a && b)),
            (BinaryOp::Or,  Val::Bool(a), Val::Bool(b))   => Ok(Val::Bool(a || b)),
            (BinaryOp::Ge,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool(a >= b)),
            (BinaryOp::Le,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool(a <= b)),
            (BinaryOp::Ne,  Val::Float(a), Val::Float(b)) => Ok(Val::Bool((a-b).abs() >= 1e-6)),
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

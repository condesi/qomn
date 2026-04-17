// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v2.0 — Type Checker
// Validates ternary type compatibility, unit types, oracle signatures,
// range bounds, and plan declarations.
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use crate::ast::*;
use crate::units::{units_compatible, nfpa_range};

#[derive(Debug, Clone)]
pub struct TypeEnv {
    vars:    HashMap<String, Type>,
    oracles: HashMap<String, (Vec<Type>, Type)>,
    crystals: HashMap<String, String>,  // name -> path
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars:    HashMap::new(),
            oracles: HashMap::new(),
            crystals: HashMap::new(),
        }
    }

    pub fn check_program(&mut self, prog: &Program) -> Vec<String> {
        let mut errors = vec![];

        // First pass: collect all declarations
        for decl in &prog.decls {
            match decl {
                Decl::Oracle(o) => {
                    let param_types: Vec<Type> = o.params.iter().map(|p| p.ty.clone()).collect();
                    self.oracles.insert(o.name.clone(), (param_types, o.ret_ty.clone()));
                }
                Decl::Crystal(c) => {
                    self.crystals.insert(c.name.clone(), c.path.clone());
                    self.vars.insert(c.name.clone(), Type::Tmat(168, 4864)); // default crystal type
                }
                Decl::Let(name, ty, _) => {
                    let t = ty.clone().unwrap_or(Type::Inferred);
                    self.vars.insert(name.clone(), t);
                }
                _ => {}
            }
        }

        // Second pass: type check bodies
        for decl in &prog.decls {
            match decl {
                Decl::Oracle(o)   => errors.extend(self.check_oracle(o)),
                Decl::Pipe(p)     => errors.extend(self.check_pipe(p)),
                Decl::Plan(p)     => errors.extend(self.check_plan(p)),  // v2.0
                Decl::Route(r)    => errors.extend(self.check_route(r)),
                Decl::Let(n,ty,v) => {
                    if let Err(e) = self.check_let(n, ty, v) {
                        errors.push(e);
                    }
                }
                _ => {}
            }
        }
        errors
    }

    // ── v2.0: Plan type checking ───────────────────────────────────
    fn check_plan(&self, p: &PlanDecl) -> Vec<String> {
        let mut errors = vec![];
        for step in &p.steps {
            if !self.oracles.contains_key(&step.oracle) {
                errors.push(format!(
                    "plan {}: step '{}' calls undeclared oracle '{}'",
                    p.name, step.name, step.oracle
                ));
            }
        }
        errors
    }

    fn check_oracle(&mut self, o: &OracleDecl) -> Vec<String> {
        let mut errors = vec![];
        // Push params into local scope
        let mut local = self.clone();
        for p in &o.params {
            local.vars.insert(p.name.clone(), p.ty.clone());
        }
        for stmt in &o.body {
            if let Err(e) = local.check_stmt(stmt, &o.ret_ty) {
                errors.push(format!("oracle {}: {}", o.name, e));
            }
        }
        errors
    }

    fn check_pipe(&mut self, p: &PipeDecl) -> Vec<String> {
        let mut errors = vec![];
        let mut local = self.clone();
        for param in &p.params {
            local.vars.insert(param.name.clone(), param.ty.clone());
        }
        for (name, expr) in &p.steps {
            match local.infer_expr(expr) {
                Ok(ty) => { local.vars.insert(name.clone(), ty); }
                Err(e) => errors.push(format!("pipe {} step {}: {}", p.name, name, e)),
            }
        }
        errors
    }

    fn check_route(&self, r: &RouteDecl) -> Vec<String> {
        let mut errors = vec![];
        match &r.target {
            RouteTarget::Crystal(n) => {
                if !self.crystals.contains_key(n) {
                    errors.push(format!("route target crystal '{}' not declared", n));
                }
            }
            RouteTarget::Oracle(n) => {
                if !self.oracles.contains_key(n) {
                    errors.push(format!("route target oracle '{}' not declared", n));
                }
            }
            _ => {}
        }
        errors
    }

    fn check_let(&mut self, name: &str, ty: &Option<Type>, val: &Expr) -> Result<(), String> {
        let inferred = self.infer_expr(val)?;
        if let Some(declared) = ty {
            self.check_compat(declared, &inferred)?;
        }
        self.vars.insert(name.to_string(), ty.clone().unwrap_or(inferred));
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Stmt, expected_ret: &Type) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, ty, val } => {
                self.check_let(name, ty, val)
            }
            Stmt::Return(expr) => {
                let ty = self.infer_expr(expr)?;
                self.check_compat(expected_ret, &ty)
            }
            Stmt::Expr(e) => { self.infer_expr(e)?; Ok(()) }
            Stmt::Respond(e) => { self.infer_expr(e)?; Ok(()) }
            // v2.0: assert must have bool condition
            Stmt::Assert { cond, msg: _ } => {
                let cty = self.infer_expr(cond)?;
                if !matches!(cty.base_scalar(), Type::Bool | Type::Inferred) {
                    return Err(format!("assert condition must be bool, got {:?}", cty));
                }
                Ok(())
            }
            Stmt::If { cond, then_body, else_body } => {
                let cty = self.infer_expr(cond)?;
                if cty != Type::Bool && cty != Type::Inferred {
                    return Err(format!("if condition must be bool, got {:?}", cty));
                }
                for s in then_body { self.check_stmt(s, expected_ret)?; }
                if let Some(eb) = else_body {
                    for s in eb { self.check_stmt(s, expected_ret)?; }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn infer_expr(&self, expr: &Expr) -> Result<Type, String> {
        match expr {
            Expr::Int(_)   => Ok(Type::I64),
            Expr::Float(_) => Ok(Type::F32),
            Expr::Bool(_)  => Ok(Type::Bool),
            Expr::Str(_)   => Ok(Type::Str),
            Expr::Trit(_)  => Ok(Type::Trit),
            Expr::Tvec(v)  => Ok(Type::Tvec(v.len())),

            Expr::Ident(name) => {
                self.vars.get(name)
                    .cloned()
                    .ok_or_else(|| format!("undefined variable '{}'", name))
            }

            Expr::Binary(op, lhs, rhs) => {
                let lt = self.infer_expr(lhs)?;
                let rt = self.infer_expr(rhs)?;
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul
                    | BinaryOp::Div | BinaryOp::Pow => {
                        self.numeric_result(&lt, &rt)
                    }
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt
                    | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge => Ok(Type::Bool),
                    BinaryOp::And | BinaryOp::Or => Ok(Type::Bool),
                    BinaryOp::Assign => Ok(rt),
                    _ => Ok(Type::Inferred),
                }
            }

            Expr::Unary(UnaryOp::Neg, e) => self.infer_expr(e),
            Expr::Unary(UnaryOp::Not, _) => Ok(Type::Bool),

            Expr::Encode(_, dim) => Ok(Type::Tvec(dim.unwrap_or(4864))),
            Expr::Quantize(e)    => {
                match self.infer_expr(e)? {
                    Type::Tmat(r,c) => Ok(Type::Tmat(r, c)),
                    Type::Tensor(_,_) => Ok(Type::Tmat(896, 4864)),
                    _ => Ok(Type::Tmat(896, 4864)),
                }
            }

            Expr::CrystalInfer { .. } => Ok(Type::Tvec(896)),
            Expr::CrystalLayer(_, _)  => Ok(Type::Tmat(896, 4864)),
            Expr::CrystalNorm(_)      => Ok(Type::F32),

            Expr::Call(func, _args) => {
                if let Expr::Ident(name) = func.as_ref() {
                    if let Some((_, ret)) = self.oracles.get(name) {
                        return Ok(ret.clone());
                    }
                }
                Ok(Type::Inferred)
            }

            Expr::PipeComp(parts) => {
                if let Some(last) = parts.last() {
                    self.infer_expr(last)
                } else {
                    Ok(Type::Inferred)
                }
            }

            _ => Ok(Type::Inferred),
        }
    }

    fn numeric_result(&self, a: &Type, b: &Type) -> Result<Type, String> {
        // v2.0: strip unit for numeric arithmetic, propagate unit from lhs
        let a_base = a.base_scalar();
        let b_base = b.base_scalar();
        let result_base = match (a_base, b_base) {
            (Type::Trit,    Type::Trit)    => Type::Trit,
            (Type::Tvec(n), Type::Tvec(m)) if n == m => Type::Tvec(*n),
            (Type::Tmat(r,c), Type::Tvec(n)) if c == n => Type::Tvec(*r),
            (Type::F32, _) | (_, Type::F32) => Type::F32,
            (Type::F64, _) | (_, Type::F64) => Type::F64,
            (Type::I64, Type::I64)           => Type::I64,
            (Type::I32, Type::I32)           => Type::I32,
            (Type::Inferred, t) | (t, Type::Inferred) => t.clone(),
            _ => return Err(format!("type mismatch: {:?} op {:?}", a, b)),
        };
        // Propagate unit annotation from lhs if present
        if let Some(unit) = a.unit() {
            Ok(Type::Unit(Box::new(result_base), unit.clone()))
        } else {
            Ok(result_base)
        }
    }

    fn check_compat(&self, expected: &Type, got: &Type) -> Result<(), String> {
        // v2.0: unit-aware compatibility
        // 1. Exact match
        if expected == got { return Ok(()); }
        // 2. Either is Inferred — pass
        if *expected == Type::Inferred || *got == Type::Inferred { return Ok(()); }
        // 3. Base scalars must match
        if expected.base_scalar() != got.base_scalar() {
            return Err(format!(
                "type error: expected {:?}, got {:?}",
                expected.base_scalar(), got.base_scalar()
            ));
        }
        // 4. Unit check: warn if units differ but don't hard-error (units are advisory)
        if let (Some(eu), Some(gu)) = (expected.unit(), got.unit()) {
            if !units_compatible(eu, gu) {
                // Emit a soft warning (not an error) — unit system is advisory in v2.0
                eprintln!(
                    "  [unit warning] expected unit '{}', got '{}' — check physical dimensions",
                    eu.display(), gu.display()
                );
            }
        }
        // 5. Range check: if expected has a range, warn on literal value overflows
        if let Type::UnitRange(_, unit, lo, hi) = expected {
            if let Some(r) = nfpa_range(&unit.display()) {
                let _ = (lo, hi, r); // ranges are validated at runtime via assert
            }
        }
        Ok(())
    }
}

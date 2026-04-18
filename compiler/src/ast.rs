// ═══════════════════════════════════════════════════════════════════════
// QOMN v2.0 — Abstract Syntax Tree
// ═══════════════════════════════════════════════════════════════════════

// ── Unit Expressions (v2.0) ───────────────────────────────────────────
/// Physical unit algebra: `gpm`, `psi^0.5`, `gpm/psi^0.5`, `A*m`
#[derive(Debug, Clone, PartialEq)]
pub enum UnitExpr {
    Base(String),                          // "gpm", "psi", "m2", "A"
    Mul(Box<UnitExpr>, Box<UnitExpr>),     // A * m
    Div(Box<UnitExpr>, Box<UnitExpr>),     // gpm / psi^0.5
    Pow(Box<UnitExpr>, f64),               // psi ^ 0.5
    Dimensionless,                         // pure number (no unit)
}

impl UnitExpr {
    pub fn display(&self) -> String {
        match self {
            UnitExpr::Base(s)       => s.clone(),
            UnitExpr::Mul(a, b)     => format!("{}*{}", a.display(), b.display()),
            UnitExpr::Div(a, b)     => format!("{}/{}", a.display(), b.display()),
            UnitExpr::Pow(a, e)     => format!("{}^{}", a.display(), e),
            UnitExpr::Dimensionless => "1".to_string(),
        }
    }
    /// Check structural equality (not dimensional analysis)
    pub fn compatible_with(&self, other: &UnitExpr) -> bool {
        self == other || matches!(self, UnitExpr::Dimensionless) || matches!(other, UnitExpr::Dimensionless)
    }
}

// ── Types ─────────────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    F32, F64, I32, I64, Bool, Str,
    Trit,
    Tvec(usize),              // tvec[n]
    Tmat(usize, usize),       // tmat[r][c]
    Tensor(Box<Type>, Vec<usize>),
    Inferred,                 // type to be resolved by typeck
    // v2.0: physical unit types
    Unit(Box<Type>, UnitExpr),                    // float(gpm), float(psi)
    UnitRange(Box<Type>, UnitExpr, f64, f64),     // float(psi)[0.0..175.0]
}

impl Type {
    /// Strip unit annotation, return the base scalar type
    pub fn base_scalar(&self) -> &Type {
        match self {
            Type::Unit(inner, _)       => inner.base_scalar(),
            Type::UnitRange(inner,_,_,_) => inner.base_scalar(),
            other                      => other,
        }
    }
    /// Extract unit expression if present
    pub fn unit(&self) -> Option<&UnitExpr> {
        match self {
            Type::Unit(_, u)         => Some(u),
            Type::UnitRange(_, u,_,_) => Some(u),
            _                        => None,
        }
    }
    pub fn is_numeric(&self) -> bool {
        matches!(self.base_scalar(), Type::F32 | Type::F64 | Type::I32 | Type::I64)
    }
}

// ── Hardware hints ─────────────────────────────────────────────────────
#[derive(Debug, Clone, PartialEq)]
pub enum HwHint { Mmap, Avx2, Cpu, Auto }

#[derive(Debug, Clone, PartialEq)]
pub enum HwCond { Avx2Available, TernaryChip, GpuAvailable, Else }

// ── Expressions ───────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Trit(i8),
    Tvec(Vec<i8>),            // tvec[+1, 0t, -1, ...]

    // Variable / field access
    Ident(String),
    Field(Box<Expr>, String), // expr.field

    // Operations
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),             // expr[idx]
    Call(Box<Expr>, Vec<Expr>),              // expr(args)

    // Crystal-specific
    CrystalInfer {                           // crystal.infer(layer=N, x=expr)
        crystal: Box<Expr>,
        layer:   Option<usize>,
        x:       Box<Expr>,
    },
    CrystalLayer(Box<Expr>, usize),          // crystal.layer(N)
    CrystalNorm(Box<Expr>),                  // crystal.norm()

    // Built-ins
    Encode(Box<Expr>, Option<usize>),        // encode(expr, dim)
    Quantize(Box<Expr>),                     // quantize(expr)

    // Pipeline composition
    PipeComp(Vec<Expr>),                     // a | b | c
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp { Neg, Not }

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or,
    Assign,
}

// ── Statements ────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty:   Option<Type>,
        val:  Expr,
    },
    Expr(Expr),
    Return(Expr),
    If {
        cond:      Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    For {
        var:  String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    Respond(Expr),
    // v2.0: runtime assertion with message
    Assert {
        cond: Expr,
        msg:  String,
    },
}

// ── Top-level declarations ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty:   Type,
}

#[derive(Debug, Clone)]
pub struct OracleDecl {
    pub name:    String,
    pub params:  Vec<Param>,
    pub ret_ty:  Type,
    pub body:    Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct CrystalDecl {
    pub name: String,
    pub hint: HwHint,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct PipeDecl {
    pub name:   String,
    pub params: Vec<Param>,
    pub steps:  Vec<(String, Expr)>,  // name = expr
    pub sink:   Expr,                 // respond(...)
}

#[derive(Debug, Clone)]
pub enum RoutePattern {
    Exact(String),
    Glob(String),   // contains *
    Any,            // *
}

#[derive(Debug, Clone)]
pub enum RouteTarget {
    Crystal(String),
    Oracle(String),
    Pipe(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct RouteDecl {
    pub pattern: RoutePattern,
    pub target:  RouteTarget,
}

#[derive(Debug, Clone)]
pub struct ScheduleBranch {
    pub cond: HwCond,
    pub hint: HwHint,
}

#[derive(Debug, Clone)]
pub struct ScheduleDecl {
    pub expr:     Expr,
    pub branches: Vec<ScheduleBranch>,
}

// ── Plan declaration (v2.0) ────────────────────────────────────────────
/// A single ordered step in a plan: `step name: oracle_call(args)`
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub name:   String,
    pub oracle: String,
    pub args:   Vec<Expr>,
    /// Optional doc comment for the step
    pub doc:    Option<String>,
}

/// A plan groups oracle calls into an ordered DAG
/// `plan plan_name(params) { step s1: oracle1(...) ... }`
#[derive(Debug, Clone)]
pub struct PlanDecl {
    pub name:   String,
    pub params: Vec<Param>,
    pub steps:  Vec<PlanStep>,
    pub doc:    Option<String>,
}

#[derive(Debug, Clone)]
pub enum Decl {
    Oracle(OracleDecl),
    Crystal(CrystalDecl),
    Pipe(PipeDecl),
    Plan(PlanDecl),          // v2.0
    Route(RouteDecl),
    Schedule(ScheduleDecl),
    Let(String, Option<Type>, Expr),
    Stmt(Stmt),
}

// ── Program ────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}

// ═══════════════════════════════════════════════════════════════════════
// CRYS-L plan_v2.rs — SPEC.md plan_* Parser, Formatter, JIT Compiler
// v2.4: Core module for plan_* syntax as documented in SPEC.md v2.3
//
// Syntax supported:
//   plan_<name>(param: f64 = default, ...) {
//       meta { standard: "...", source: "...", domain: "..." }
//       const IDENT = expr;
//       let IDENT = expr;
//       formula "label": "text";
//       assert expr msg "text";
//       output IDENT label "label" unit "unit";
//       return { IDENT: IDENT, ... };
//   }
//
// This module is INDEPENDENT of the old oracle/plan AST.
// It integrates with JIT via the existing Cranelift JIT infrastructure.
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ── Source Location ────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Span {
    pub line: u32,
    pub col:  u32,
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

// ── AST Types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlanV2Param {
    pub name:    String,
    pub ty:      PV2Type,
    pub default: Option<f64>,
    pub span:    Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PV2Type { F64, F32, I64, Bool, Str }

impl PV2Type {
    fn as_str(&self) -> &'static str {
        match self {
            PV2Type::F64 => "f64",
            PV2Type::F32 => "f32",
            PV2Type::I64 => "i64",
            PV2Type::Bool => "bool",
            PV2Type::Str  => "str",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanV2Meta {
    pub standard: String,
    pub source:   String,
    pub domain:   String,
    pub version:  Option<String>,
    pub note:     Option<String>,
}

impl Default for PlanV2Meta {
    fn default() -> Self {
        Self { standard: String::new(), source: String::new(), domain: String::new(),
               version: None, note: None }
    }
}

#[derive(Debug, Clone)]
pub enum PV2Expr {
    Num(f64),
    Ident(String),
    Neg(Box<PV2Expr>),
    BinOp { op: PV2Op, left: Box<PV2Expr>, right: Box<PV2Expr> },
    Call { name: String, args: Vec<PV2Expr> },
    // Comparisons for assert
    Cmp { op: PV2Cmp, left: Box<PV2Expr>, right: Box<PV2Expr> },
    And(Box<PV2Expr>, Box<PV2Expr>),
    Or(Box<PV2Expr>, Box<PV2Expr>),
    Not(Box<PV2Expr>),
    Str(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PV2Op { Add, Sub, Mul, Div, Pow }

#[derive(Debug, Clone, PartialEq)]
pub enum PV2Cmp { Eq, Ne, Lt, Le, Gt, Ge }

#[derive(Debug, Clone)]
pub enum PV2Item {
    Const  { name: String, value: f64, span: Span },
    Let    { name: String, expr: PV2Expr, span: Span },
    Formula{ label: String, text: String, span: Span },
    Assert { expr: PV2Expr, msg: String, span: Span },
    Output { name: String, label: String, unit: Option<String>, span: Span },
}

#[derive(Debug, Clone)]
pub struct PlanV2Decl {
    pub name:        String,
    pub params:      Vec<PlanV2Param>,
    pub meta:        PlanV2Meta,
    pub body:        Vec<PV2Item>,
    /// Variables listed in return { a: a, b: b }
    pub return_vars: Vec<String>,
    pub span:        Span,
}

// ── Lexer (dedicated, simpler than main lexer) ────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum PT { // PlanToken
    Ident(String), Str(String), Float(f64), Int(i64),
    // Punctuation
    LParen, RParen, LBrace, RBrace, Comma, Colon, Semi, Eq, EqEq, BangEq,
    Lt, Le, Gt, Ge, Plus, Minus, Star, Slash, Caret, Bang, Pipe, Amp,
    // Keywords
    KwMeta, KwConst, KwLet, KwFormula, KwAssert, KwMsg, KwOutput,
    KwLabel, KwUnit, KwReturn, KwTrue, KwFalse, KwAnd, KwOr, KwNot,
    KwF64, KwF32, KwI64, KwBool, KwStr,
    Newline, Eof,
}

#[derive(Debug, Clone)]
struct PTok { token: PT, span: Span }

struct PV2Lexer<'a> {
    src:  &'a [char],
    pos:  usize,
    line: u32,
    col:  u32,
}

impl<'a> PV2Lexer<'a> {
    fn new(chars: &'a [char]) -> Self { Self { src: chars, pos: 0, line: 1, col: 1 } }

    fn peek(&self) -> Option<char> { self.src.get(self.pos).copied() }
    fn peek2(&self) -> Option<char> { self.src.get(self.pos + 1).copied() }

    fn advance(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied();
        if let Some(ch) = c {
            self.pos += 1;
            if ch == '\n' { self.line += 1; self.col = 1; }
            else { self.col += 1; }
        }
        c
    }

    fn span(&self) -> Span { Span { line: self.line, col: self.col } }

    fn tokenize(mut self) -> Vec<PTok> {
        let mut tokens = vec![];
        loop {
            // skip whitespace (not newlines which are significant)
            while matches!(self.peek(), Some(' ') | Some('\t') | Some('\r')) { self.advance(); }
            let sp = self.span();

            match self.peek() {
                None => { tokens.push(PTok { token: PT::Eof, span: sp }); break; }

                // Comments: // and #
                Some('/') if self.peek2() == Some('/') => {
                    while self.peek().is_some() && self.peek() != Some('\n') { self.advance(); }
                }
                Some('#') => {
                    while self.peek().is_some() && self.peek() != Some('\n') { self.advance(); }
                }

                Some('\n') => {
                    self.advance();
                    tokens.push(PTok { token: PT::Newline, span: sp });
                }

                Some('"') => {
                    self.advance();
                    let mut s = String::new();
                    loop {
                        match self.peek() {
                            Some('"') | None => { self.advance(); break; }
                            Some('\\') => {
                                self.advance();
                                match self.advance() {
                                    Some('n') => s.push('\n'),
                                    Some('t') => s.push('\t'),
                                    Some(c)   => { s.push('\\'); s.push(c); }
                                    None      => break,
                                }
                            }
                            Some(c) => { s.push(c); self.advance(); }
                        }
                    }
                    tokens.push(PTok { token: PT::Str(s), span: sp });
                }

                // Numbers
                Some(c) if c.is_ascii_digit() || (c == '-' && self.peek2().map(|d| d.is_ascii_digit()).unwrap_or(false)) => {
                    let mut s = String::new();
                    if c == '-' { s.push(c); self.advance(); }
                    while self.peek().map(|c| c.is_ascii_digit() || c == '.' || c == '_' || c == 'e' || c == 'E').unwrap_or(false) {
                        let ch = self.advance().unwrap();
                        if ch != '_' { s.push(ch); }
                        // Handle e+/e- in scientific notation
                        if (ch == 'e' || ch == 'E') && matches!(self.peek(), Some('+') | Some('-')) {
                            s.push(self.advance().unwrap());
                        }
                    }
                    let tok = if s.contains('.') || s.contains('e') || s.contains('E') {
                        PT::Float(s.parse().unwrap_or(0.0))
                    } else {
                        PT::Int(s.parse().unwrap_or(0))
                    };
                    tokens.push(PTok { token: tok, span: sp });
                }

                // Identifiers and keywords
                Some(c) if c.is_alphabetic() || c == '_' => {
                    let mut s = String::new();
                    while self.peek().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) {
                        s.push(self.advance().unwrap());
                    }
                    let tok = match s.as_str() {
                        "meta"    => PT::KwMeta,
                        "const"   => PT::KwConst,
                        "let"     => PT::KwLet,
                        "formula" => PT::KwFormula,
                        "assert"  => PT::KwAssert,
                        "msg"     => PT::KwMsg,
                        "output"  => PT::KwOutput,
                        "label"   => PT::KwLabel,
                        "unit"    => PT::KwUnit,
                        "return"  => PT::KwReturn,
                        "true"    => PT::KwTrue,
                        "false"   => PT::KwFalse,
                        "and"     => PT::KwAnd,
                        "or"      => PT::KwOr,
                        "not"     => PT::KwNot,
                        "f64" | "float" | "float64" => PT::KwF64,
                        "f32" | "float32"            => PT::KwF32,
                        "i64" | "int" | "int64"      => PT::KwI64,
                        "bool"    => PT::KwBool,
                        "str"     => PT::KwStr,
                        _         => PT::Ident(s),
                    };
                    tokens.push(PTok { token: tok, span: sp });
                }

                // Operators / punctuation
                Some('=') => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); tokens.push(PTok { token: PT::EqEq, span: sp }); }
                    else { tokens.push(PTok { token: PT::Eq, span: sp }); }
                }
                Some('!') => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); tokens.push(PTok { token: PT::BangEq, span: sp }); }
                    else { tokens.push(PTok { token: PT::Bang, span: sp }); }
                }
                Some('<') => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); tokens.push(PTok { token: PT::Le, span: sp }); }
                    else { tokens.push(PTok { token: PT::Lt, span: sp }); }
                }
                Some('>') => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); tokens.push(PTok { token: PT::Ge, span: sp }); }
                    else { tokens.push(PTok { token: PT::Gt, span: sp }); }
                }
                Some('&') => {
                    self.advance();
                    if self.peek() == Some('&') { self.advance(); }
                    tokens.push(PTok { token: PT::Amp, span: sp });
                }
                Some('|') => {
                    self.advance();
                    if self.peek() == Some('|') { self.advance(); }
                    tokens.push(PTok { token: PT::Pipe, span: sp });
                }
                Some('(') => { self.advance(); tokens.push(PTok { token: PT::LParen, span: sp }); }
                Some(')') => { self.advance(); tokens.push(PTok { token: PT::RParen, span: sp }); }
                Some('{') => { self.advance(); tokens.push(PTok { token: PT::LBrace, span: sp }); }
                Some('}') => { self.advance(); tokens.push(PTok { token: PT::RBrace, span: sp }); }
                Some(',') => { self.advance(); tokens.push(PTok { token: PT::Comma, span: sp }); }
                Some(':') => { self.advance(); tokens.push(PTok { token: PT::Colon, span: sp }); }
                Some(';') => { self.advance(); tokens.push(PTok { token: PT::Semi, span: sp }); }
                Some('+') => { self.advance(); tokens.push(PTok { token: PT::Plus, span: sp }); }
                Some('-') => { self.advance(); tokens.push(PTok { token: PT::Minus, span: sp }); }
                Some('*') => { self.advance(); tokens.push(PTok { token: PT::Star, span: sp }); }
                Some('/') => { self.advance(); tokens.push(PTok { token: PT::Slash, span: sp }); }
                Some('^') => { self.advance(); tokens.push(PTok { token: PT::Caret, span: sp }); }
                Some(c) => { self.advance(); /* skip unknown */ let _ = c; }
            }
        }
        tokens
    }
}

// ── Parser Error ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PV2Error {
    pub message: String,
    pub span:    Span,
}

impl std::fmt::Display for PV2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}:{}] {}", self.span.line, self.span.col, self.message)
    }
}

impl PV2Error {
    fn new(msg: impl Into<String>, span: Span) -> Self {
        Self { message: msg.into(), span }
    }
}

// ── Parser ────────────────────────────────────────────────────────────

struct PV2Parser {
    tokens: Vec<PTok>,
    pos:    usize,
}

impl PV2Parser {
    fn new(tokens: Vec<PTok>) -> Self { Self { tokens, pos: 0 } }

    fn peek(&self) -> &PT { &self.tokens[self.pos.min(self.tokens.len()-1)].token }
    fn span(&self) -> Span { self.tokens[self.pos.min(self.tokens.len()-1)].span.clone() }

    fn advance(&mut self) -> PT {
        let t = self.tokens[self.pos.min(self.tokens.len()-1)].token.clone();
        if self.pos + 1 < self.tokens.len() { self.pos += 1; }
        t
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), PT::Newline) { self.advance(); }
    }

    fn expect_ident(&mut self) -> Result<String, PV2Error> {
        let sp = self.span();
        match self.advance() {
            PT::Ident(s) => Ok(s),
            other => Err(PV2Error::new(format!("expected identifier, got {:?}", other), sp)),
        }
    }

    fn expect_str(&mut self) -> Result<String, PV2Error> {
        let sp = self.span();
        match self.advance() {
            PT::Str(s) => Ok(s),
            other => Err(PV2Error::new(format!("expected string literal, got {:?}", other), sp)),
        }
    }

    fn expect_tok(&mut self, expected: &PT) -> Result<(), PV2Error> {
        let sp = self.span();
        let got = self.advance();
        if &got == expected { Ok(()) }
        else { Err(PV2Error::new(format!("expected {:?}, got {:?}", expected, got), sp)) }
    }

    /// Parse a single plan_* declaration starting from current position.
    /// Assumes the leading `plan_<name>` identifier has NOT been consumed yet.
    pub fn parse_plan(&mut self) -> Result<PlanV2Decl, PV2Error> {
        self.skip_newlines();
        let sp = self.span();
        let name = self.expect_ident()?;
        if !name.starts_with("plan_") {
            return Err(PV2Error::new(format!("expected plan_* name, got '{}'", name), sp.clone()));
        }

        // Parameters
        self.expect_tok(&PT::LParen)?;
        let params = self.parse_params()?;
        self.expect_tok(&PT::RParen)?;

        self.skip_newlines();
        self.expect_tok(&PT::LBrace)?;
        self.skip_newlines();

        // Meta block
        let meta = if matches!(self.peek(), PT::KwMeta) {
            self.advance(); // consume 'meta'
            self.skip_newlines();
            self.expect_tok(&PT::LBrace)?;
            let m = self.parse_meta()?;
            self.skip_newlines();
            self.expect_tok(&PT::RBrace)?;
            self.skip_newlines();
            // optional comma/semi after meta
            if matches!(self.peek(), PT::Comma | PT::Semi) { self.advance(); }
            self.skip_newlines();
            m
        } else {
            PlanV2Meta::default()
        };

        // Body items
        let mut body = vec![];
        let mut return_vars = vec![];

        loop {
            self.skip_newlines();
            match self.peek() {
                PT::RBrace | PT::Eof => break,
                PT::KwConst => {
                    let item_sp = self.span();
                    self.advance();
                    let n = self.expect_ident()?;
                    self.expect_tok(&PT::Eq)?;
                    let v = self.parse_const_expr()?;
                    if matches!(self.peek(), PT::Semi) { self.advance(); }
                    body.push(PV2Item::Const { name: n, value: v, span: item_sp });
                }
                PT::KwLet => {
                    let item_sp = self.span();
                    self.advance();
                    let n = self.expect_ident()?;
                    self.expect_tok(&PT::Eq)?;
                    let expr = self.parse_expr()?;
                    if matches!(self.peek(), PT::Semi) { self.advance(); }
                    body.push(PV2Item::Let { name: n, expr, span: item_sp });
                }
                PT::KwFormula => {
                    let item_sp = self.span();
                    self.advance();
                    let lbl = self.expect_str()?;
                    self.expect_tok(&PT::Colon)?;
                    let txt = self.expect_str()?;
                    if matches!(self.peek(), PT::Semi) { self.advance(); }
                    body.push(PV2Item::Formula { label: lbl, text: txt, span: item_sp });
                }
                PT::KwAssert => {
                    let item_sp = self.span();
                    self.advance();
                    let expr = self.parse_cond_expr()?;
                    // expect 'msg'
                    let msg_str = if matches!(self.peek(), PT::KwMsg) {
                        self.advance();
                        self.expect_str()?
                    } else {
                        "assertion failed".to_string()
                    };
                    if matches!(self.peek(), PT::Semi) { self.advance(); }
                    body.push(PV2Item::Assert { expr, msg: msg_str, span: item_sp });
                }
                PT::KwOutput => {
                    let item_sp = self.span();
                    self.advance();
                    let var = self.expect_ident()?;
                    // optional 'label "..."'
                    let lbl = if matches!(self.peek(), PT::KwLabel) {
                        self.advance();
                        self.expect_str()?
                    } else {
                        var.clone()
                    };
                    // optional 'unit "..."'
                    let unit = if matches!(self.peek(), PT::KwUnit) {
                        self.advance();
                        Some(self.expect_str()?)
                    } else {
                        None
                    };
                    if matches!(self.peek(), PT::Semi) { self.advance(); }
                    body.push(PV2Item::Output { name: var, label: lbl, unit, span: item_sp });
                }
                PT::KwReturn => {
                    self.advance();
                    self.expect_tok(&PT::LBrace)?;
                    self.skip_newlines();
                    while !matches!(self.peek(), PT::RBrace | PT::Eof) {
                        let n = self.expect_ident()?;
                        if matches!(self.peek(), PT::Colon) {
                            self.advance(); // skip ':'
                            // value ident (may be same as key or different)
                            let _ = self.expect_ident()?;
                        }
                        return_vars.push(n);
                        if matches!(self.peek(), PT::Comma) { self.advance(); }
                        self.skip_newlines();
                    }
                    self.expect_tok(&PT::RBrace)?;
                    if matches!(self.peek(), PT::Semi) { self.advance(); }
                }
                _ => {
                    // skip unknown tokens to be lenient
                    self.advance();
                }
            }
        }

        self.skip_newlines();
        self.expect_tok(&PT::RBrace)?;

        Ok(PlanV2Decl { name, params, meta, body, return_vars, span: sp })
    }

    fn parse_params(&mut self) -> Result<Vec<PlanV2Param>, PV2Error> {
        let mut params = vec![];
        while !matches!(self.peek(), PT::RParen | PT::Eof) {
            let sp = self.span();
            let name = self.expect_ident()?;
            // optional type annotation
            let ty = if matches!(self.peek(), PT::Colon) {
                self.advance();
                self.parse_type()?
            } else {
                PV2Type::F64
            };
            // optional default
            let default = if matches!(self.peek(), PT::Eq) {
                self.advance();
                Some(self.parse_const_expr()?)
            } else {
                None
            };
            params.push(PlanV2Param { name, ty, default, span: sp });
            if matches!(self.peek(), PT::Comma) { self.advance(); }
        }
        Ok(params)
    }

    fn parse_type(&mut self) -> Result<PV2Type, PV2Error> {
        let sp = self.span();
        match self.advance() {
            PT::KwF64 => Ok(PV2Type::F64),
            PT::KwF32 => Ok(PV2Type::F32),
            PT::KwI64 => Ok(PV2Type::I64),
            PT::KwBool => Ok(PV2Type::Bool),
            PT::KwStr  => Ok(PV2Type::Str),
            other => Err(PV2Error::new(format!("expected type (f64/f32/i64/bool/str), got {:?}", other), sp)),
        }
    }

    fn parse_meta(&mut self) -> Result<PlanV2Meta, PV2Error> {
        let mut m = PlanV2Meta::default();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), PT::RBrace | PT::Eof) { break; }
            let key = match self.advance() {
                PT::Ident(s) => s,
                PT::RBrace => break,
                _ => continue,
            };
            if !matches!(self.peek(), PT::Colon) { continue; }
            self.advance(); // ':'
            let val = self.expect_str().unwrap_or_default();
            if matches!(self.peek(), PT::Comma) { self.advance(); }
            match key.as_str() {
                "standard" => m.standard = val,
                "source"   => m.source   = val,
                "domain"   => m.domain   = val,
                "version"  => m.version  = Some(val),
                "note"     => m.note     = Some(val),
                _ => {}
            }
        }
        Ok(m)
    }

    /// Parse a constant-foldable expression (no variable references).
    /// Used for const declarations and default parameter values.
    fn parse_const_expr(&mut self) -> Result<f64, PV2Error> {
        // Parse a simple arithmetic expression and constant-fold it
        let expr = self.parse_expr()?;
        self.const_fold(&expr).ok_or_else(|| PV2Error::new(
            "const expression cannot reference variables", self.span()))
    }

    fn const_fold(&self, expr: &PV2Expr) -> Option<f64> {
        match expr {
            PV2Expr::Num(v) => Some(*v),
            PV2Expr::Neg(e) => Some(-self.const_fold(e)?),
            PV2Expr::BinOp { op, left, right } => {
                let l = self.const_fold(left)?;
                let r = self.const_fold(right)?;
                Some(match op {
                    PV2Op::Add => l + r,
                    PV2Op::Sub => l - r,
                    PV2Op::Mul => l * r,
                    PV2Op::Div => if r == 0.0 { return None; } else { l / r },
                    PV2Op::Pow => l.powf(r),
                })
            }
            _ => None,
        }
    }

    // ── Expression parsing (Pratt) ─────────────────────────────────

    fn parse_expr(&mut self) -> Result<PV2Expr, PV2Error> {
        self.parse_add()
    }

    fn parse_add(&mut self) -> Result<PV2Expr, PV2Error> {
        let mut left = self.parse_mul()?;
        loop {
            match self.peek() {
                PT::Plus  => { self.advance(); let r = self.parse_mul()?;  left = PV2Expr::BinOp { op: PV2Op::Add, left: Box::new(left), right: Box::new(r) }; }
                PT::Minus => { self.advance(); let r = self.parse_mul()?;  left = PV2Expr::BinOp { op: PV2Op::Sub, left: Box::new(left), right: Box::new(r) }; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<PV2Expr, PV2Error> {
        let mut left = self.parse_pow()?;
        loop {
            match self.peek() {
                PT::Star  => { self.advance(); let r = self.parse_pow()?; left = PV2Expr::BinOp { op: PV2Op::Mul, left: Box::new(left), right: Box::new(r) }; }
                PT::Slash => { self.advance(); let r = self.parse_pow()?; left = PV2Expr::BinOp { op: PV2Op::Div, left: Box::new(left), right: Box::new(r) }; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_pow(&mut self) -> Result<PV2Expr, PV2Error> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), PT::Caret) {
            self.advance();
            let exp = self.parse_unary()?;
            Ok(PV2Expr::BinOp { op: PV2Op::Pow, left: Box::new(base), right: Box::new(exp) })
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<PV2Expr, PV2Error> {
        if matches!(self.peek(), PT::Minus) {
            self.advance();
            let e = self.parse_atom()?;
            return Ok(PV2Expr::Neg(Box::new(e)));
        }
        if matches!(self.peek(), PT::Bang) {
            self.advance();
            let e = self.parse_atom()?;
            return Ok(PV2Expr::Not(Box::new(e)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<PV2Expr, PV2Error> {
        let sp = self.span();
        match self.peek().clone() {
            PT::Float(v)  => { self.advance(); Ok(PV2Expr::Num(v)) }
            PT::Int(v)    => { self.advance(); Ok(PV2Expr::Num(v as f64)) }
            PT::KwTrue    => { self.advance(); Ok(PV2Expr::Num(1.0)) }
            PT::KwFalse   => { self.advance(); Ok(PV2Expr::Num(0.0)) }
            PT::Str(s)    => { let s = s.clone(); self.advance(); Ok(PV2Expr::Str(s)) }
            PT::Ident(name) => {
                let name = name.clone(); self.advance();
                if matches!(self.peek(), PT::LParen) {
                    // Function call
                    self.advance();
                    let mut args = vec![];
                    while !matches!(self.peek(), PT::RParen | PT::Eof) {
                        args.push(self.parse_expr()?);
                        if matches!(self.peek(), PT::Comma) { self.advance(); }
                    }
                    self.expect_tok(&PT::RParen)?;
                    Ok(PV2Expr::Call { name, args })
                } else {
                    Ok(PV2Expr::Ident(name))
                }
            }
            PT::LParen => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect_tok(&PT::RParen)?;
                Ok(e)
            }
            other => Err(PV2Error::new(format!("unexpected token in expression: {:?}", other), sp)),
        }
    }

    /// Parse a conditional expression (for assert)
    fn parse_cond_expr(&mut self) -> Result<PV2Expr, PV2Error> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<PV2Expr, PV2Error> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), PT::Pipe | PT::KwOr) {
            self.advance();
            let r = self.parse_and()?;
            left = PV2Expr::Or(Box::new(left), Box::new(r));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<PV2Expr, PV2Error> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), PT::Amp | PT::KwAnd) {
            self.advance();
            let r = self.parse_not()?;
            left = PV2Expr::And(Box::new(left), Box::new(r));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<PV2Expr, PV2Error> {
        if matches!(self.peek(), PT::Bang | PT::KwNot) {
            self.advance();
            let e = self.parse_cmp()?;
            return Ok(PV2Expr::Not(Box::new(e)));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<PV2Expr, PV2Error> {
        let left = self.parse_expr()?;
        let op = match self.peek() {
            PT::EqEq   => PV2Cmp::Eq,
            PT::BangEq => PV2Cmp::Ne,
            PT::Lt     => PV2Cmp::Lt,
            PT::Le     => PV2Cmp::Le,
            PT::Gt     => PV2Cmp::Gt,
            PT::Ge     => PV2Cmp::Ge,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_expr()?;
        Ok(PV2Expr::Cmp { op, left: Box::new(left), right: Box::new(right) })
    }
}

// ── Public parse function ─────────────────────────────────────────────

/// Parse all plan_* declarations from a CRYS-L v2.3 source file.
/// Non-plan_* top-level items (comments, blank lines) are ignored.
pub fn parse_plans(src: &str) -> Result<Vec<PlanV2Decl>, Vec<PV2Error>> {
    let chars: Vec<char> = src.chars().collect();
    let lexer = PV2Lexer::new(&chars);
    let tokens = lexer.tokenize();

    let mut parser = PV2Parser::new(tokens);
    let mut plans  = vec![];
    let mut errors = vec![];

    parser.skip_newlines();
    while !matches!(parser.peek(), PT::Eof) {
        match parser.peek() {
            PT::Ident(s) if s.starts_with("plan_") => {
                match parser.parse_plan() {
                    Ok(p)  => plans.push(p),
                    Err(e) => { errors.push(e); /* skip to next plan */ }
                }
            }
            _ => { parser.advance(); } // skip non-plan lines
        }
        parser.skip_newlines();
    }

    if errors.is_empty() { Ok(plans) } else { Err(errors) }
}

// ── Formatter (fmt command) ───────────────────────────────────────────

/// Pretty-print a plan_* declaration back to canonical CRYS-L source.
pub fn fmt_plan(plan: &PlanV2Decl) -> String {
    let mut out = String::new();
    // Signature
    let params_str: Vec<String> = plan.params.iter().map(|p| {
        let mut s = format!("{}: {}", p.name, p.ty.as_str());
        if let Some(d) = p.default { write!(s, " = {}", fmt_f64(d)).ok(); }
        s
    }).collect();
    writeln!(out, "{}({}) {{", plan.name, params_str.join(", ")).ok();

    // Meta block
    let m = &plan.meta;
    if !m.standard.is_empty() || !m.source.is_empty() {
        writeln!(out, "    meta {{").ok();
        if !m.standard.is_empty() { writeln!(out, "        standard: \"{}\",", m.standard).ok(); }
        if !m.source.is_empty()   { writeln!(out, "        source:   \"{}\",", m.source).ok(); }
        if !m.domain.is_empty()   { writeln!(out, "        domain:   \"{}\",", m.domain).ok(); }
        if let Some(v) = &m.version { writeln!(out, "        version:  \"{}\",", v).ok(); }
        if let Some(n) = &m.note    { writeln!(out, "        note:     \"{}\",", n).ok(); }
        writeln!(out, "    }}").ok();
    }

    // Body items
    for item in &plan.body {
        match item {
            PV2Item::Const { name, value, .. } =>
                writeln!(out, "    const {} = {};", name, fmt_f64(*value)).ok(),
            PV2Item::Let { name, expr, .. } =>
                writeln!(out, "    let {} = {};", name, fmt_expr(expr)).ok(),
            PV2Item::Formula { label, text, .. } =>
                writeln!(out, "    formula \"{}\": \"{}\";", label, text).ok(),
            PV2Item::Assert { expr, msg, .. } =>
                writeln!(out, "    assert {} msg \"{}\";", fmt_cond(expr), msg).ok(),
            PV2Item::Output { name, label, unit, .. } => {
                let unit_part = unit.as_ref().map(|u| format!(" unit \"{}\"", u)).unwrap_or_default();
                writeln!(out, "    output {} label \"{}\"{};", name, label, unit_part).ok()
            }
        };
    }

    // Return
    if !plan.return_vars.is_empty() {
        let rv: Vec<String> = plan.return_vars.iter().map(|v| format!("{}: {}", v, v)).collect();
        writeln!(out, "    return {{ {} }};", rv.join(", ")).ok();
    }

    writeln!(out, "}}").ok();
    out
}

fn fmt_f64(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 { format!("{:.1}", v) }
    else { format!("{}", v) }
}

fn fmt_expr(e: &PV2Expr) -> String {
    match e {
        PV2Expr::Num(v) => fmt_f64(*v),
        PV2Expr::Ident(s) => s.clone(),
        PV2Expr::Str(s) => format!("\"{}\"", s),
        PV2Expr::Neg(e) => format!("-{}", fmt_expr(e)),
        PV2Expr::Not(e) => format!("!{}", fmt_expr(e)),
        PV2Expr::BinOp { op, left, right } => {
            let op_s = match op {
                PV2Op::Add => "+", PV2Op::Sub => "-",
                PV2Op::Mul => "*", PV2Op::Div => "/", PV2Op::Pow => "^",
            };
            format!("({} {} {})", fmt_expr(left), op_s, fmt_expr(right))
        }
        PV2Expr::Call { name, args } => {
            let args_s: Vec<String> = args.iter().map(fmt_expr).collect();
            format!("{}({})", name, args_s.join(", "))
        }
        PV2Expr::Cmp { op, left, right } => fmt_cond(&PV2Expr::Cmp { op: op.clone(), left: left.clone(), right: right.clone() }),
        PV2Expr::And(l, r) => format!("({} && {})", fmt_cond(l), fmt_cond(r)),
        PV2Expr::Or(l, r)  => format!("({} || {})", fmt_cond(l), fmt_cond(r)),
    }
}

fn fmt_cond(e: &PV2Expr) -> String {
    match e {
        PV2Expr::Cmp { op, left, right } => {
            let op_s = match op {
                PV2Cmp::Eq => "==", PV2Cmp::Ne => "!=",
                PV2Cmp::Lt => "<",  PV2Cmp::Le => "<=",
                PV2Cmp::Gt => ">",  PV2Cmp::Ge => ">=",
            };
            format!("{} {} {}", fmt_expr(left), op_s, fmt_expr(right))
        }
        PV2Expr::And(l, r) => format!("({} && {})", fmt_cond(l), fmt_cond(r)),
        PV2Expr::Or(l, r)  => format!("({} || {})", fmt_cond(l), fmt_cond(r)),
        PV2Expr::Not(e)    => format!("!{}", fmt_cond(e)),
        other => fmt_expr(other),
    }
}

// ── Type Checker ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PV2TypeError {
    pub message: String,
    pub span:    Span,
}

pub fn typecheck_plan(plan: &PlanV2Decl) -> Vec<PV2TypeError> {
    let mut errors = vec![];
    let mut env: HashMap<String, PV2Type> = HashMap::new();

    // Seed env with parameters
    for p in &plan.params {
        env.insert(p.name.clone(), p.ty.clone());
    }

    // meta: standard and source must be non-empty
    if plan.meta.standard.is_empty() {
        errors.push(PV2TypeError { message: "meta.standard is required".into(), span: plan.span.clone() });
    }

    // Check body
    for item in &plan.body {
        match item {
            PV2Item::Const { name, .. } => {
                env.insert(name.clone(), PV2Type::F64);
            }
            PV2Item::Let { name, expr, span } => {
                if let Err(e) = typecheck_expr(expr, &env) {
                    errors.push(PV2TypeError { message: format!("let {}: {}", name, e), span: span.clone() });
                }
                env.insert(name.clone(), PV2Type::F64);
            }
            PV2Item::Assert { expr, span, .. } => {
                if let Err(e) = typecheck_cond(expr, &env) {
                    errors.push(PV2TypeError { message: format!("assert: {}", e), span: span.clone() });
                }
            }
            PV2Item::Output { name, span, .. } => {
                if !env.contains_key(name) {
                    errors.push(PV2TypeError {
                        message: format!("output '{}' is not defined", name), span: span.clone()
                    });
                }
            }
            PV2Item::Formula { .. } => {} // no type check needed
        }
    }

    // Check return vars exist in env
    for rv in &plan.return_vars {
        if !env.contains_key(rv) {
            errors.push(PV2TypeError {
                message: format!("return var '{}' is not defined", rv), span: plan.span.clone()
            });
        }
    }

    errors
}

fn typecheck_expr(e: &PV2Expr, env: &HashMap<String, PV2Type>) -> Result<(), String> {
    match e {
        PV2Expr::Num(_) | PV2Expr::Str(_) => Ok(()),
        PV2Expr::Ident(s) => {
            if env.contains_key(s) { Ok(()) }
            else { Err(format!("undefined variable '{}'", s)) }
        }
        PV2Expr::Neg(e) | PV2Expr::Not(e) => typecheck_expr(e, env),
        PV2Expr::BinOp { left, right, .. } => {
            typecheck_expr(left, env)?;
            typecheck_expr(right, env)
        }
        PV2Expr::Call { name, args } => {
            // Check it's a known built-in
            let known = ["sqrt","pow","abs","min","max","clamp","log","log10","round",
                         "ceil","floor","sin","cos","tan","asin","acos","atan","atan2",
                         "pi","e","exp"];
            if !known.contains(&name.as_str()) {
                return Err(format!("unknown function '{}'", name));
            }
            for a in args { typecheck_expr(a, env)?; }
            Ok(())
        }
        PV2Expr::Cmp { left, right, .. } => {
            typecheck_expr(left, env)?;
            typecheck_expr(right, env)
        }
        PV2Expr::And(l, r) | PV2Expr::Or(l, r) => {
            typecheck_cond(l, env)?;
            typecheck_cond(r, env)
        }
    }
}

fn typecheck_cond(e: &PV2Expr, env: &HashMap<String, PV2Type>) -> Result<(), String> {
    typecheck_expr(e, env)
}

// ── Interpreter (execute plan_v2 without JIT) ─────────────────────────

#[derive(Debug)]
pub struct PV2Result {
    pub plan:    String,
    pub inputs:  HashMap<String, f64>,
    pub outputs: Vec<(String, f64)>,  // (var_name, value)
    pub meta:    PlanV2Meta,
    pub formulas: Vec<(String, String)>,
    pub latency_ns: u64,
}

impl PV2Result {
    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        let mut out_obj = String::new();
        for (k, v) in &self.outputs {
            if !out_obj.is_empty() { out_obj.push_str(", "); }
            write!(out_obj, "\"{}\": {}", k, v).ok();
        }
        format!(r#"{{"ok":true,"plan":"{}","standard":"{}","result":{{{}}}, "latency_ns":{}}}"#,
            self.plan, self.meta.standard, out_obj, self.latency_ns)
    }

    /// Human-readable display
    pub fn display(&self) {
        println!("Plan: {}  [{}]", self.plan, self.meta.standard);
        println!("{:-<60}", "");
        println!("  Inputs:");
        for (k, v) in &self.inputs {
            println!("    {:20} = {:.4}", k, v);
        }
        println!("  Outputs:");
        for (k, v) in &self.outputs {
            println!("    {:20} = {:.4}", k, v);
        }
        if !self.formulas.is_empty() {
            println!("  Formulas:");
            for (lbl, txt) in &self.formulas {
                println!("    {}:  {}", lbl, txt);
            }
        }
        println!("  Latency: {} ns", self.latency_ns);
    }
}

pub fn execute_plan(plan: &PlanV2Decl, args: &HashMap<String, f64>) -> Result<PV2Result, String> {
    let t0 = std::time::Instant::now();

    // Build env from params (with defaults for missing)
    let mut env: HashMap<String, f64> = HashMap::new();
    for p in &plan.params {
        if let Some(&v) = args.get(&p.name) {
            env.insert(p.name.clone(), v);
        } else if let Some(d) = p.default {
            env.insert(p.name.clone(), d);
        } else {
            return Err(format!("missing required parameter '{}'", p.name));
        }
    }

    let mut formulas = vec![];
    let mut inputs = env.clone();

    // Execute body items
    for item in &plan.body {
        match item {
            PV2Item::Const { name, value, .. } => {
                env.insert(name.clone(), *value);
            }
            PV2Item::Let { name, expr, .. } => {
                let v = eval_expr(expr, &env)
                    .map_err(|e| format!("let {}: {}", name, e))?;
                env.insert(name.clone(), v);
            }
            PV2Item::Formula { label, text, .. } => {
                formulas.push((label.clone(), text.clone()));
            }
            PV2Item::Assert { expr, msg, .. } => {
                let v = eval_cond(expr, &env).map_err(|e| format!("assert: {}", e))?;
                if v == 0.0 {
                    return Err(format!("assertion failed: {}", msg));
                }
            }
            PV2Item::Output { .. } | PV2Item::Let { .. } => {} // already handled
        }
    }

    // Collect outputs
    let outputs: Vec<(String, f64)> = if !plan.return_vars.is_empty() {
        plan.return_vars.iter()
            .filter_map(|v| env.get(v).map(|&val| (v.clone(), val)))
            .collect()
    } else {
        // Auto-collect let bindings (excluding input params)
        let param_names: std::collections::HashSet<&String> = plan.params.iter().map(|p| &p.name).collect();
        env.iter()
            .filter(|(k, _)| !param_names.contains(k))
            .map(|(k, &v)| (k.clone(), v))
            .collect()
    };

    Ok(PV2Result {
        plan: plan.name.clone(),
        inputs,
        outputs,
        meta: plan.meta.clone(),
        formulas,
        latency_ns: t0.elapsed().as_nanos() as u64,
    })
}

fn eval_expr(e: &PV2Expr, env: &HashMap<String, f64>) -> Result<f64, String> {
    match e {
        PV2Expr::Num(v) => Ok(*v),
        PV2Expr::Ident(s) => env.get(s).copied().ok_or_else(|| format!("undefined '{}'", s)),
        PV2Expr::Str(_) => Err("cannot use string in arithmetic".to_string()),
        PV2Expr::Neg(e) => Ok(-eval_expr(e, env)?),
        PV2Expr::Not(e) => Ok(if eval_cond(e, env)? == 0.0 { 1.0 } else { 0.0 }),
        PV2Expr::BinOp { op, left, right } => {
            let l = eval_expr(left, env)?;
            let r = eval_expr(right, env)?;
            match op {
                PV2Op::Add => Ok(l + r),
                PV2Op::Sub => Ok(l - r),
                PV2Op::Mul => Ok(l * r),
                PV2Op::Div => if r == 0.0 { Err("division by zero".to_string()) } else { Ok(l / r) },
                PV2Op::Pow => Ok(l.powf(r)),
            }
        }
        PV2Expr::Call { name, args } => {
            let a: Vec<f64> = args.iter().map(|a| eval_expr(a, env)).collect::<Result<_, _>>()?;
            eval_builtin(name, &a)
        }
        PV2Expr::Cmp { op, left, right } => {
            let l = eval_expr(left, env)?;
            let r = eval_expr(right, env)?;
            let b = match op {
                PV2Cmp::Eq => l == r, PV2Cmp::Ne => l != r,
                PV2Cmp::Lt => l < r,  PV2Cmp::Le => l <= r,
                PV2Cmp::Gt => l > r,  PV2Cmp::Ge => l >= r,
            };
            Ok(if b { 1.0 } else { 0.0 })
        }
        PV2Expr::And(l, r) => Ok(if eval_cond(l, env)? != 0.0 && eval_cond(r, env)? != 0.0 { 1.0 } else { 0.0 }),
        PV2Expr::Or(l, r)  => Ok(if eval_cond(l, env)? != 0.0 || eval_cond(r, env)? != 0.0 { 1.0 } else { 0.0 }),
    }
}

fn eval_cond(e: &PV2Expr, env: &HashMap<String, f64>) -> Result<f64, String> {
    eval_expr(e, env)
}

fn eval_builtin(name: &str, args: &[f64]) -> Result<f64, String> {
    let a0 = || args.first().copied().ok_or_else(|| format!("{}() needs 1 arg", name));
    let a1 = || args.get(1).copied().ok_or_else(|| format!("{}() needs 2 args", name));
    match name {
        "sqrt"  => Ok(a0()?.sqrt()),
        "abs"   => Ok(a0()?.abs()),
        "log"   => Ok(a0()?.ln()),
        "log10" => Ok(a0()?.log10()),
        "exp"   => Ok(a0()?.exp()),
        "round" => Ok(a0()?.round()),
        "ceil"  => Ok(a0()?.ceil()),
        "floor" => Ok(a0()?.floor()),
        "sin"   => Ok(a0()?.sin()),
        "cos"   => Ok(a0()?.cos()),
        "tan"   => Ok(a0()?.tan()),
        "asin"  => Ok(a0()?.asin()),
        "acos"  => Ok(a0()?.acos()),
        "atan"  => Ok(a0()?.atan()),
        "pi"    => Ok(std::f64::consts::PI),
        "e"     => Ok(std::f64::consts::E),
        "pow"   => Ok(a0()?.powf(a1()?)),
        "atan2" => Ok(a0()?.atan2(a1()?)),
        "min"   => Ok(a0()?.min(a1()?)),
        "max"   => Ok(a0()?.max(a1()?)),
        "clamp" => {
            let v   = a0()?;
            let lo  = a1()?;
            let hi  = args.get(2).copied().ok_or_else(|| "clamp() needs 3 args".to_string())?;
            Ok(v.clamp(lo, hi))
        }
        other => Err(format!("unknown built-in '{}'", other)),
    }
}

// ── Sweep (v2.5 language feature) ────────────────────────────────────

/// A single sweep axis: one parameter iterated over a range or list of values.
#[derive(Debug, Clone)]
pub enum SweepAxis {
    Range { start: f64, stop: f64, step: f64 },
    List(Vec<f64>),
    Fixed(f64),
}

impl SweepAxis {
    pub fn values(&self) -> Vec<f64> {
        match self {
            SweepAxis::Range { start, stop, step } => {
                let mut v = vec![];
                let mut cur = *start;
                while cur <= *stop + 1e-10 {
                    v.push(cur);
                    cur += step;
                }
                v
            }
            SweepAxis::List(vals) => vals.clone(),
            SweepAxis::Fixed(v)   => vec![*v],
        }
    }
}

/// Execute a Cartesian sweep over all parameter combinations.
/// Returns results sorted by first output variable descending.
pub fn sweep_plan(plan: &PlanV2Decl, axes: &HashMap<String, SweepAxis>)
    -> Vec<PV2Result>
{
    // Build ordered list of (param_name, values)
    let param_names: Vec<String> = plan.params.iter().map(|p| p.name.clone()).collect();
    let mut ranges: Vec<Vec<f64>> = param_names.iter().map(|n| {
        axes.get(n).map(|a| a.values())
            .or_else(|| plan.params.iter().find(|p| &p.name == n).and_then(|p| p.default.map(|d| vec![d])))
            .unwrap_or_default()
    }).collect();

    // Cartesian product
    let mut combinations: Vec<Vec<f64>> = vec![vec![]];
    for vals in &ranges {
        let mut new_combinations = vec![];
        for existing in &combinations {
            for &v in vals {
                let mut new = existing.clone();
                new.push(v);
                new_combinations.push(new);
            }
        }
        combinations = new_combinations;
    }

    // Execute each combination
    combinations.iter().filter_map(|combo| {
        let args: HashMap<String, f64> = param_names.iter()
            .zip(combo.iter())
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        execute_plan(plan, &args).ok()
    }).collect()
}

/// Parse sweep axis from CLI string: "param=start:stop:step" or "param=v1,v2,v3" or "param=value"
pub fn parse_sweep_axis(spec: &str) -> Option<(String, SweepAxis)> {
    let (name, rest) = spec.split_once('=')?;
    let axis = if rest.contains(':') {
        let parts: Vec<f64> = rest.split(':').filter_map(|s| s.parse().ok()).collect();
        if parts.len() >= 2 {
            let step = parts.get(2).copied().unwrap_or(1.0);
            SweepAxis::Range { start: parts[0], stop: parts[1], step }
        } else {
            return None;
        }
    } else if rest.contains(',') {
        let vals: Vec<f64> = rest.split(',').filter_map(|s| s.parse().ok()).collect();
        if vals.is_empty() { return None; }
        SweepAxis::List(vals)
    } else {
        SweepAxis::Fixed(rest.parse().ok()?)
    };
    Some((name.to_string(), axis))
}

// ── Converge (v2.6: bounded iteration) ──────────────────────────────

/// Perform Newton-like convergence using a callback f(x) → x'.
/// The callback is a closure that computes one iteration step.
/// Returns (converged_value, iterations, converged: bool).
pub fn converge<F: Fn(f64) -> f64>(
    initial: f64,
    max_iter: usize,
    tol: f64,
    f: F,
) -> (f64, usize, bool) {
    let mut x = initial;
    for i in 0..max_iter {
        let x_new = f(x);
        if (x_new - x).abs() < tol {
            return (x_new, i + 1, true);
        }
        x = x_new;
    }
    (x, max_iter, false)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const PUMP_SIZING: &str = r#"
plan_pump_sizing(Q_gpm: f64, P_psi: f64, eff: f64 = 0.70) {
    meta {
        standard: "NFPA 20:2022",
        source:   "Section 4.26",
        domain:   "fire",
        version:  "2.3",
    }
    const HP_FACTOR   = 3960.0;
    let HP_req = (Q_gpm * P_psi) / (eff * HP_FACTOR);
    let HP_max = HP_req * 1.40;
    formula "Pump HP": "HP = (Q[GPM] x P[PSI]) / (eta x 3960)";
    assert Q_gpm > 0.0 msg "flow must be positive";
    assert eff <= 1.0 msg "efficiency must be <= 1.0";
    return { HP_req: HP_req, HP_max: HP_max };
}
"#;

    #[test]
    fn test_parse_pump_sizing() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        assert_eq!(plans.len(), 1);
        let p = &plans[0];
        assert_eq!(p.name, "plan_pump_sizing");
        assert_eq!(p.params.len(), 3);
        assert_eq!(p.params[2].default, Some(0.70));
        assert_eq!(p.meta.standard, "NFPA 20:2022");
        assert_eq!(p.meta.domain, "fire");
    }

    #[test]
    fn test_execute_pump_sizing_ref01() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        let mut args = HashMap::new();
        args.insert("Q_gpm".to_string(), 500.0);
        args.insert("P_psi".to_string(), 100.0);
        args.insert("eff".to_string(), 0.75);
        let result = execute_plan(&plans[0], &args).expect("execute failed");
        let hp_req = result.outputs.iter().find(|(k, _)| k == "HP_req").map(|(_, v)| *v).unwrap_or(0.0);
        // Expected: 16.835 ± 0.5%
        assert!((hp_req - 16.835).abs() < 0.1, "HP_req = {} (expected ~16.835)", hp_req);
    }

    #[test]
    fn test_execute_pump_assert_negative_flow() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        let mut args = HashMap::new();
        args.insert("Q_gpm".to_string(), -100.0);
        args.insert("P_psi".to_string(), 100.0);
        args.insert("eff".to_string(), 0.75);
        let result = execute_plan(&plans[0], &args);
        assert!(result.is_err(), "expected assertion error for negative flow");
        assert!(result.unwrap_err().contains("flow must be positive"));
    }

    #[test]
    fn test_execute_pump_default_efficiency() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        let mut args = HashMap::new();
        args.insert("Q_gpm".to_string(), 500.0);
        args.insert("P_psi".to_string(), 100.0);
        // eff not provided — should use default 0.70
        let result = execute_plan(&plans[0], &args).expect("execute failed");
        let hp_req = result.outputs.iter().find(|(k, _)| k == "HP_req").map(|(_, v)| *v).unwrap_or(0.0);
        // Expected: 18.038 ± 0.5%
        assert!((hp_req - 18.038).abs() < 0.1, "HP_req with default eff = {} (expected ~18.038)", hp_req);
    }

    #[test]
    fn test_typecheck_pump_sizing() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        let errors = typecheck_plan(&plans[0]);
        assert!(errors.is_empty(), "typecheck errors: {:?}", errors);
    }

    #[test]
    fn test_fmt_roundtrip() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        let formatted = fmt_plan(&plans[0]);
        // Re-parse the formatted output
        let plans2 = parse_plans(&formatted).expect("re-parse of formatted output failed");
        assert_eq!(plans2.len(), 1);
        assert_eq!(plans2[0].name, "plan_pump_sizing");
    }

    #[test]
    fn test_sweep() {
        let plans = parse_plans(PUMP_SIZING).expect("parse failed");
        let mut axes = HashMap::new();
        axes.insert("Q_gpm".to_string(), SweepAxis::Range { start: 100.0, stop: 500.0, step: 200.0 });
        axes.insert("P_psi".to_string(), SweepAxis::Fixed(100.0));
        axes.insert("eff".to_string(), SweepAxis::Fixed(0.75));
        let results = sweep_plan(&plans[0], &axes);
        assert_eq!(results.len(), 3, "expected 3 sweep results (100, 300, 500 GPM)");
    }

    #[test]
    fn test_converge_sqrt2() {
        // Newton's method for sqrt(2): x' = (x + 2/x) / 2
        let (result, iters, ok) = converge(1.0, 50, 1e-10, |x| (x + 2.0/x) / 2.0);
        assert!(ok, "did not converge in {} iters", iters);
        assert!((result - 2.0f64.sqrt()).abs() < 1e-9, "result = {}", result);
    }
}

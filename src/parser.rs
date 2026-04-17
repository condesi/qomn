// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v0.3 — Parser  (Token stream → AST)
// Recursive descent, handles INDENT/DEDENT for blocks.
// ═══════════════════════════════════════════════════════════════════════

use crate::lexer::{Tok, Token};
use crate::ast::*;

pub struct Parser {
    tokens: Vec<Tok>,
    pos:    usize,
}

impl Parser {
    pub fn new(tokens: Vec<Tok>) -> Self {
        // filter bare newlines at top level between declarations
        Self { tokens, pos: 0 }
    }

    // ── Token helpers ──────────────────────────────────────────────
    fn peek(&self) -> &Token { &self.tokens[self.pos].token }

    fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos].token;
        if self.pos + 1 < self.tokens.len() { self.pos += 1; }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, self.peek()))
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) { self.advance(); }
    }

    fn at_eof(&self) -> bool { matches!(self.peek(), Token::Eof) }

    // ── Program ────────────────────────────────────────────────────
    pub fn parse(&mut self) -> Result<Program, String> {
        let mut decls = vec![];
        self.skip_newlines();
        while !self.at_eof() {
            decls.push(self.parse_decl()?);
            self.skip_newlines();
        }
        Ok(Program { decls })
    }

    fn parse_decl(&mut self) -> Result<Decl, String> {
        match self.peek() {
            Token::KwOracle   => Ok(Decl::Oracle(self.parse_oracle()?)),
            Token::KwCrystal  => Ok(Decl::Crystal(self.parse_crystal()?)),
            Token::KwPipe     => Ok(Decl::Pipe(self.parse_pipe()?)),
            Token::KwPlan     => Ok(Decl::Plan(self.parse_plan()?)),    // v2.0
            Token::KwRoute    => Ok(Decl::Route(self.parse_route()?)),
            Token::KwSchedule => Ok(Decl::Schedule(self.parse_schedule()?)),
            Token::KwLet      => {
                let (n, ty, v) = self.parse_let()?;
                Ok(Decl::Let(n, ty, v))
            }
            _ => Ok(Decl::Stmt(self.parse_stmt()?)),
        }
    }

    // ── Oracle declaration ─────────────────────────────────────────
    // oracle name(params) -> type:
    //     body
    fn parse_oracle(&mut self) -> Result<OracleDecl, String> {
        self.advance(); // 'oracle'
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::Arrow)?;
        let ret_ty = self.parse_type()?;
        self.expect(&Token::Colon)?;
        self.skip_newlines();
        let body = self.parse_block()?;
        Ok(OracleDecl { name, params, ret_ty, body })
    }

    // ── Crystal declaration ────────────────────────────────────────
    // crystal name = load @hint "path"
    fn parse_crystal(&mut self) -> Result<CrystalDecl, String> {
        self.advance(); // 'crystal'
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        self.expect(&Token::KwLoad)?;
        let hint = self.parse_hw_hint()?;
        let path = self.expect_str()?;
        Ok(CrystalDecl { name, hint, path })
    }

    // ── Pipe declaration ───────────────────────────────────────────
    // pipe name(params):
    //     steps
    //     -> respond(expr)
    fn parse_pipe(&mut self) -> Result<PipeDecl, String> {
        self.advance(); // 'pipe'
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::Colon)?;
        self.skip_newlines();
        self.expect(&Token::Indent)?;
        self.skip_newlines();

        let mut steps: Vec<(String, Expr)> = vec![];
        let mut sink = Expr::Bool(false);

        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }

            if matches!(self.peek(), Token::Arrow) {
                self.advance(); // '->'
                self.expect(&Token::KwRespond)?;
                self.expect(&Token::LParen)?;
                sink = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                self.skip_newlines();
                break;
            }

            // step: name = expr
            if let Token::Ident(_) = self.peek() {
                let n = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let e = self.parse_pipe_expr()?;
                steps.push((n, e));
            } else {
                break;
            }
        }

        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        Ok(PipeDecl { name, params, steps, sink })
    }

    // ── Plan declaration (v2.0) ────────────────────────────────────
    // plan name(params):
    //     step step_name: oracle_name(args)
    //     step step_name: oracle_name(args)
    fn parse_plan(&mut self) -> Result<PlanDecl, String> {
        self.advance(); // 'plan'
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::Colon)?;
        self.skip_newlines();
        self.expect(&Token::Indent)?;
        self.skip_newlines();

        let mut steps: Vec<PlanStep> = vec![];
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            // Expect: step <name>: <oracle>(args)
            if matches!(self.peek(), Token::KwStep) {
                self.advance(); // 'step'
                let step_name = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let oracle_name = self.expect_ident()?;
                self.expect(&Token::LParen)?;
                let mut args = vec![];
                while !matches!(self.peek(), Token::RParen | Token::Eof) {
                    args.push(self.parse_expr()?);
                    if matches!(self.peek(), Token::Comma) { self.advance(); }
                }
                self.expect(&Token::RParen)?;
                steps.push(PlanStep { name: step_name, oracle: oracle_name, args, doc: None });
            } else {
                break;
            }
        }
        // consume dedent
        if matches!(self.peek(), Token::Dedent) { self.advance(); }

        Ok(PlanDecl { name, params, steps, doc: None })
    }

    // ── Route declaration ──────────────────────────────────────────
    // route pattern -> target
    fn parse_route(&mut self) -> Result<RouteDecl, String> {
        self.advance(); // 'route'
        let pattern = match self.peek() {
            Token::Str(s) => { let s = s.clone(); self.advance(); RoutePattern::Exact(s) }
            Token::Glob    => { self.advance(); RoutePattern::Glob("*".into()) }
            Token::Star    => { self.advance(); RoutePattern::Any }
            _              => return Err(format!("Expected route pattern, got {:?}", self.peek())),
        };
        self.expect(&Token::Arrow)?;

        let target = match self.peek() {
            Token::KwCrystal  => {
                self.advance();
                self.expect(&Token::Colon)?;
                let n = self.expect_ident()?;
                RouteTarget::Crystal(n)
            }
            Token::KwOracle   => {
                self.advance();
                self.expect(&Token::Colon)?;
                let n = self.expect_ident()?;
                RouteTarget::Oracle(n)
            }
            Token::KwPipe     => {
                self.advance();
                self.expect(&Token::Colon)?;
                let n = self.expect_ident()?;
                RouteTarget::Pipe(n)
            }
            _ => RouteTarget::Expr(self.parse_pipe_expr()?),
        };

        Ok(RouteDecl { pattern, target })
    }

    // ── Schedule declaration ───────────────────────────────────────
    // schedule expr:  OR  schedule *:
    //     if cond: @hint
    fn parse_schedule(&mut self) -> Result<ScheduleDecl, String> {
        self.advance(); // 'schedule'
        // Allow wildcard `*` as schedule target meaning "all expressions"
        let expr = if matches!(self.peek(), Token::Star) {
            self.advance();
            Expr::Ident("*".into())
        } else {
            self.parse_expr()?
        };
        self.expect(&Token::Colon)?;
        self.skip_newlines();
        self.expect(&Token::Indent)?;
        self.skip_newlines();

        let mut branches = vec![];
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            self.expect(&Token::KwIf)?;
            let cond = self.parse_hw_cond()?;
            self.expect(&Token::Colon)?;
            let hint = self.parse_hw_hint()?;
            self.skip_newlines();
            branches.push(ScheduleBranch { cond, hint });
        }
        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        Ok(ScheduleDecl { expr, branches })
    }

    // ── Block ──────────────────────────────────────────────────────
    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(&Token::Indent)?;
        let mut stmts = vec![];
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            stmts.push(self.parse_stmt()?);
        }
        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        Ok(stmts)
    }

    // ── Statements ─────────────────────────────────────────────────
    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Token::KwLet    => { let (n,t,v) = self.parse_let()?; Ok(Stmt::Let{name:n,ty:t,val:v}) }
            Token::KwReturn => { self.advance(); Ok(Stmt::Return(self.parse_expr()?)) }
            Token::KwIf     => self.parse_if(),
            Token::KwFor    => self.parse_for(),
            // v2.0: assert condition, "message"
            Token::KwAssert => {
                self.advance(); // 'assert'
                let cond = self.parse_expr()?;
                self.expect(&Token::Comma)?;
                let msg = self.expect_str()?;
                Ok(Stmt::Assert { cond, msg })
            }
            Token::Arrow    => {
                self.advance();
                self.expect(&Token::KwRespond)?;
                self.expect(&Token::LParen)?;
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Stmt::Respond(e))
            }
            _ => Ok(Stmt::Expr(self.parse_expr()?)),
        }
    }

    fn parse_let(&mut self) -> Result<(String, Option<Type>, Expr), String> {
        self.advance(); // 'let'
        let name = self.expect_ident()?;
        let ty = if matches!(self.peek(), Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else { None };
        self.expect(&Token::Eq)?;
        let val = self.parse_expr()?;
        Ok((name, ty, val))
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        self.advance(); // 'if'
        let cond = self.parse_expr()?;
        self.expect(&Token::Colon)?;
        self.skip_newlines();
        let then_body = self.parse_block()?;
        let else_body = if matches!(self.peek(), Token::KwElse) {
            self.advance();
            self.expect(&Token::Colon)?;
            self.skip_newlines();
            Some(self.parse_block()?)
        } else { None };
        Ok(Stmt::If { cond, then_body, else_body })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        self.advance(); // 'for'
        let var = self.expect_ident()?;
        self.expect(&Token::KwIn)?;
        let iter = self.parse_expr()?;
        self.expect(&Token::Colon)?;
        self.skip_newlines();
        let body = self.parse_block()?;
        Ok(Stmt::For { var, iter, body })
    }

    // ── Pipe expression (a | b | c) ────────────────────────────────
    fn parse_pipe_expr(&mut self) -> Result<Expr, String> {
        let first = self.parse_expr()?;
        if !matches!(self.peek(), Token::Pipe) { return Ok(first); }
        let mut parts = vec![first];
        while matches!(self.peek(), Token::Pipe) {
            self.advance();
            parts.push(self.parse_expr()?);
        }
        Ok(Expr::PipeComp(parts))
    }

    // ── Expressions (Pratt-style precedence) ───────────────────────
    fn parse_expr(&mut self) -> Result<Expr, String> { self.parse_assign() }

    fn parse_assign(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_or()?;
        if matches!(self.peek(), Token::Eq) {
            self.advance();
            let rhs = self.parse_assign()?;
            return Ok(Expr::Binary(BinaryOp::Assign, Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Token::KwOr) {
            self.advance();
            lhs = Expr::Binary(BinaryOp::Or, Box::new(lhs), Box::new(self.parse_and()?));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.peek(), Token::KwAnd) {
            self.advance();
            lhs = Expr::Binary(BinaryOp::And, Box::new(lhs), Box::new(self.parse_cmp()?));
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Token::EqEq  => BinaryOp::Eq,
                Token::BangEq=> BinaryOp::Ne,
                Token::Lt    => BinaryOp::Lt,
                Token::Gt    => BinaryOp::Gt,
                Token::LtEq  => BinaryOp::Le,
                Token::GtEq  => BinaryOp::Ge,
                _ => break,
            };
            self.advance();
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(self.parse_add()?));
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Token::Plus  => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(self.parse_mul()?));
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_pow()?;
        loop {
            let op = match self.peek() {
                Token::Star    => BinaryOp::Mul,
                Token::Slash   => BinaryOp::Div,
                Token::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.advance();
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(self.parse_pow()?));
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), Token::Caret) {
            self.advance();
            let exp = self.parse_unary()?;
            return Ok(Expr::Binary(BinaryOp::Pow, Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Token::Minus  => { self.advance(); Ok(Expr::Unary(UnaryOp::Neg, Box::new(self.parse_unary()?))) }
            Token::KwNot  => { self.advance(); Ok(Expr::Unary(UnaryOp::Not, Box::new(self.parse_unary()?))) }
            _ => self.parse_call(),
        }
    }

    fn parse_call(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Token::LParen => {
                    self.advance();
                    let args = self.parse_args()?;
                    self.expect(&Token::RParen)?;
                    expr = Expr::Call(Box::new(expr), args);
                }
                Token::LBracket => {
                    self.advance();
                    let idx = self.parse_expr()?;
                    self.expect(&Token::RBracket)?;
                    expr = Expr::Index(Box::new(expr), Box::new(idx));
                }
                Token::Dot => {
                    self.advance();
                    let field = self.expect_ident()?;
                    // special crystal methods
                    if field == "infer" {
                        self.expect(&Token::LParen)?;
                        let (layer, x) = self.parse_infer_args()?;
                        self.expect(&Token::RParen)?;
                        expr = Expr::CrystalInfer { crystal: Box::new(expr), layer, x: Box::new(x) };
                    } else if field == "layer" {
                        self.expect(&Token::LParen)?;
                        let n = self.expect_int()? as usize;
                        self.expect(&Token::RParen)?;
                        expr = Expr::CrystalLayer(Box::new(expr), n);
                    } else if field == "norm" {
                        self.expect(&Token::LParen)?;
                        self.expect(&Token::RParen)?;
                        expr = Expr::CrystalNorm(Box::new(expr));
                    } else {
                        expr = Expr::Field(Box::new(expr), field);
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_infer_args(&mut self) -> Result<(Option<usize>, Expr), String> {
        let mut layer = None;
        let mut x_expr = None;
        loop {
            if matches!(self.peek(), Token::RParen) { break; }
            let key = self.expect_ident()?;
            self.expect(&Token::Eq)?;
            if key == "layer" {
                layer = Some(self.expect_int()? as usize);
            } else if key == "x" {
                x_expr = Some(self.parse_expr()?);
            }
            if matches!(self.peek(), Token::Comma) { self.advance(); }
        }
        Ok((layer, x_expr.unwrap_or(Expr::Bool(false))))
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Token::Int(n)   => { self.advance(); Ok(Expr::Int(n)) }
            Token::Float(f) => { self.advance(); Ok(Expr::Float(f)) }
            Token::Str(s)   => { self.advance(); Ok(Expr::Str(s)) }
            Token::Bool(b)  => { self.advance(); Ok(Expr::Bool(b)) }
            Token::Trit(t)  => { self.advance(); Ok(Expr::Trit(t)) }
            Token::KwEncode => {
                self.advance();
                self.expect(&Token::LParen)?;
                let e = self.parse_expr()?;
                let dim = if matches!(self.peek(), Token::Comma) {
                    self.advance();
                    Some(self.expect_int()? as usize)
                } else { None };
                self.expect(&Token::RParen)?;
                Ok(Expr::Encode(Box::new(e), dim))
            }
            Token::KwQuantize => {
                self.advance();
                self.expect(&Token::LParen)?;
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Quantize(Box::new(e)))
            }
            Token::TyTvec => {
                self.advance();
                self.expect(&Token::LBracket)?;
                let mut trits = vec![];
                loop {
                    if matches!(self.peek(), Token::RBracket) { break; }
                    match self.peek().clone() {
                        Token::Trit(t) => { self.advance(); trits.push(t); }
                        _ => break,
                    }
                    if matches!(self.peek(), Token::Comma) { self.advance(); }
                }
                self.expect(&Token::RBracket)?;
                Ok(Expr::Tvec(trits))
            }
            Token::LParen => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            Token::Ident(name) => { self.advance(); Ok(Expr::Ident(name)) }
            other => Err(format!("Unexpected token in expression: {:?}", other)),
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = vec![];
        while !matches!(self.peek(), Token::RParen | Token::Eof) {
            args.push(self.parse_expr()?);
            if matches!(self.peek(), Token::Comma) { self.advance(); }
        }
        Ok(args)
    }

    // ── Types ──────────────────────────────────────────────────────
    fn parse_type(&mut self) -> Result<Type, String> {
        let base = match self.peek().clone() {
            Token::TyF32    => { self.advance(); Type::F32 }
            Token::TyF64    => { self.advance(); Type::F64 }
            Token::TyI32    => { self.advance(); Type::I32 }
            Token::TyI64    => { self.advance(); Type::I64 }
            Token::TyBool   => { self.advance(); Type::Bool }
            Token::TyStr    => { self.advance(); Type::Str }
            Token::TyTrit   => { self.advance(); Type::Trit }
            Token::TyTvec   => {
                self.advance();
                self.expect(&Token::LBracket)?;
                let n = self.expect_int()? as usize;
                self.expect(&Token::RBracket)?;
                Type::Tvec(n)
            }
            Token::TyTmat   => {
                self.advance();
                self.expect(&Token::LBracket)?;
                let r = self.expect_int()? as usize;
                self.expect(&Token::RBracket)?;
                self.expect(&Token::LBracket)?;
                let c = self.expect_int()? as usize;
                self.expect(&Token::RBracket)?;
                Type::Tmat(r, c)
            }
            other => return Err(format!("Expected type, got {:?}", other)),
        };

        // v2.0: optional unit annotation `float(unit_expr)`
        if matches!(self.peek(), Token::LParen) && base.is_numeric() {
            self.advance(); // '('
            let unit = self.parse_unit_expr()?;
            self.expect(&Token::RParen)?;
            let typed = Type::Unit(Box::new(base), unit);

            // v2.0: optional range `[lo..hi]`
            if matches!(self.peek(), Token::LBracket) {
                self.advance(); // '['
                let lo = self.expect_float()?;
                self.expect(&Token::DotDot)?;
                let hi = self.expect_float()?;
                self.expect(&Token::RBracket)?;
                if let Type::Unit(inner, u) = typed {
                    return Ok(Type::UnitRange(inner, u, lo, hi));
                }
            }
            return Ok(typed);
        }

        Ok(base)
    }

    /// Parse a unit expression: `gpm`, `psi^0.5`, `gpm/psi^0.5`, `A*m`
    /// Grammar (right-to-left precedence: ^ > * > /)
    fn parse_unit_expr(&mut self) -> Result<UnitExpr, String> {
        self.parse_unit_div()
    }

    fn parse_unit_div(&mut self) -> Result<UnitExpr, String> {
        let mut lhs = self.parse_unit_mul()?;
        while matches!(self.peek(), Token::Slash) {
            self.advance();
            let rhs = self.parse_unit_mul()?;
            lhs = UnitExpr::Div(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unit_mul(&mut self) -> Result<UnitExpr, String> {
        let mut lhs = self.parse_unit_pow()?;
        while matches!(self.peek(), Token::Star) {
            self.advance();
            let rhs = self.parse_unit_pow()?;
            lhs = UnitExpr::Mul(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unit_pow(&mut self) -> Result<UnitExpr, String> {
        let base_unit = self.parse_unit_atom()?;
        if matches!(self.peek(), Token::Caret) {
            self.advance();
            let exp = self.expect_float()?;
            return Ok(UnitExpr::Pow(Box::new(base_unit), exp));
        }
        Ok(base_unit)
    }

    fn parse_unit_atom(&mut self) -> Result<UnitExpr, String> {
        match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                Ok(UnitExpr::Base(s))
            }
            // Allow type keywords as unit names (e.g. "m2" parsed as ident)
            Token::LParen => {
                self.advance();
                let inner = self.parse_unit_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(format!("Expected unit identifier, got {:?}", other)),
        }
    }

    /// Expect a float literal (or int that becomes float)
    fn expect_float(&mut self) -> Result<f64, String> {
        match self.peek().clone() {
            Token::Float(f) => { self.advance(); Ok(f) }
            Token::Int(i)   => { self.advance(); Ok(i as f64) }
            other => Err(format!("Expected float literal, got {:?}", other)),
        }
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, String> {
        let mut params = vec![];
        while !matches!(self.peek(), Token::RParen | Token::Eof) {
            let name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let ty = self.parse_type()?;
            params.push(Param { name, ty });
            if matches!(self.peek(), Token::Comma) { self.advance(); }
        }
        Ok(params)
    }

    fn parse_hw_hint(&mut self) -> Result<HwHint, String> {
        match self.peek() {
            Token::HwMmap => { self.advance(); Ok(HwHint::Mmap) }
            Token::HwAvx2 => { self.advance(); Ok(HwHint::Avx2) }
            Token::HwCpu  => { self.advance(); Ok(HwHint::Cpu) }
            Token::HwAuto => { self.advance(); Ok(HwHint::Auto) }
            other => Err(format!("Expected hardware hint @mmap/@avx2/@cpu/@auto, got {:?}", other)),
        }
    }

    fn parse_hw_cond(&mut self) -> Result<HwCond, String> {
        match self.peek() {
            Token::HwCondAvx2    => { self.advance(); Ok(HwCond::Avx2Available) }
            Token::HwCondTernary => { self.advance(); Ok(HwCond::TernaryChip) }
            Token::HwCondGpu     => { self.advance(); Ok(HwCond::GpuAvailable) }
            Token::KwElse        => { self.advance(); Ok(HwCond::Else) }
            other => Err(format!("Expected hw condition, got {:?}", other)),
        }
    }

    // ── Helpers ────────────────────────────────────────────────────
    fn expect_ident(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Token::Ident(s) => { self.advance(); Ok(s) }
            other => Err(format!("Expected identifier, got {:?}", other)),
        }
    }

    fn expect_str(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Token::Str(s) => { self.advance(); Ok(s) }
            other => Err(format!("Expected string, got {:?}", other)),
        }
    }

    fn expect_int(&mut self) -> Result<i64, String> {
        match self.peek().clone() {
            Token::Int(n) => { self.advance(); Ok(n) }
            other => Err(format!("Expected integer, got {:?}", other)),
        }
    }
}

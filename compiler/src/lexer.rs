// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v0.2 — Lexer (Tokenizer)
// Converts source text into a flat token stream with position info.
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ── Literals ──────────────────────────────────────────────
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Trit(i8),        // +1 | 0t | -1  (ternary literal)

    // ── Identifiers & Keywords ────────────────────────────────
    Ident(String),

    // Declarations
    KwOracle,
    KwCrystal,
    KwPipe,
    KwPlan,          // v2.0 plan declaration
    KwStep,          // v2.0 plan step
    KwRoute,
    KwSchedule,
    KwLoad,
    KwLet,

    // Control flow
    KwIf,
    KwElse,
    KwFor,
    KwIn,
    KwReturn,
    KwRespond,
    KwAssert,        // v2.0 assert statement

    // Range operator (v2.0: [lo..hi])
    DotDot,          // ..

    // Expressions
    KwEncode,
    KwQuantize,
    KwAnd,
    KwOr,
    KwNot,

    // Scalar types
    TyF32, TyF64, TyI32, TyI64, TyBool, TyStr,

    // Ternary types
    TyTrit,
    TyTvec,
    TyTmat,
    TyTensor,

    // Hardware hints
    HwMmap,
    HwAvx2,
    HwCpu,
    HwAuto,

    // Hardware conditions
    HwCondAvx2,
    HwCondTernary,
    HwCondGpu,
    HwCondElse,

    // ── Operators ─────────────────────────────────────────────
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Caret,      // ^
    Eq,         // =
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    Gt,         // >
    LtEq,       // <=
    GtEq,       // >=
    Arrow,      // ->
    Pipe,       // |
    Dot,        // .
    Colon,      // :
    Comma,      // ,
    Glob,       // * (in route patterns)

    // ── Delimiters ────────────────────────────────────────────
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]

    // ── Indentation ───────────────────────────────────────────
    Newline,
    Indent,
    Dedent,

    // ── End ───────────────────────────────────────────────────
    Eof,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub line: usize,
    pub col:  usize,
}

#[derive(Debug, Clone)]
pub struct Tok {
    pub token: Token,
    pub span:  Span,
}

pub struct Lexer {
    src:    Vec<char>,
    pos:    usize,
    line:   usize,
    col:    usize,
    indent_stack: Vec<usize>,
    pending: Vec<Tok>,
}

impl Lexer {
    pub fn new(src: &str) -> Self {
        Self {
            src: src.chars().collect(),
            pos: 0, line: 1, col: 1,
            indent_stack: vec![0],
            pending: vec![],
        }
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied()?;
        self.pos += 1;
        if c == '\n' { self.line += 1; self.col = 1; }
        else         { self.col += 1; }
        Some(c)
    }

    fn span(&self) -> Span { Span { line: self.line, col: self.col } }

    fn skip_comment(&mut self) {
        while self.peek().map_or(false, |c| c != '\n') {
            self.advance();
        }
    }

    fn read_string(&mut self) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c == '"' { self.advance(); break; }
            s.push(c);
            self.advance();
        }
        s
    }

    fn read_number(&mut self, first: char) -> Token {
        let mut s = String::from(first);
        while self.peek().map_or(false, |c| c.is_ascii_digit()) {
            s.push(self.advance().unwrap());
        }
        if self.peek() == Some('.') && self.peek2().map_or(false, |c| c.is_ascii_digit()) {
            s.push(self.advance().unwrap()); // '.'
            while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                s.push(self.advance().unwrap());
            }
            Token::Float(s.parse().unwrap_or(0.0))
        } else {
            Token::Int(s.parse().unwrap_or(0))
        }
    }

    fn read_ident(&mut self, first: char) -> Token {
        let mut s = String::from(first);
        while self.peek().map_or(false, |c| c.is_alphanumeric() || c == '_') {
            s.push(self.advance().unwrap());
        }
        match s.as_str() {
            "oracle"           => Token::KwOracle,
            "crystal"          => Token::KwCrystal,
            "pipe"             => Token::KwPipe,
            "plan"             => Token::KwPlan,
            "step"             => Token::KwStep,
            "assert"           => Token::KwAssert,
            "route"            => Token::KwRoute,
            "schedule"         => Token::KwSchedule,
            "load"             => Token::KwLoad,
            "let"              => Token::KwLet,
            "if"               => Token::KwIf,
            "else"             => Token::KwElse,
            "for"              => Token::KwFor,
            "in"               => Token::KwIn,
            "return"           => Token::KwReturn,
            "respond"          => Token::KwRespond,
            "encode"           => Token::KwEncode,
            "quantize"         => Token::KwQuantize,
            "and"              => Token::KwAnd,
            "or"               => Token::KwOr,
            "not"              => Token::KwNot,
            "true"             => Token::Bool(true),
            "false"            => Token::Bool(false),
            "f32" | "float32"  => Token::TyF32,
            "f64" | "float"    => Token::TyF64,   // float is alias for f64
            "i32" | "int32"    => Token::TyI32,
            "i64" | "int" | "int64" => Token::TyI64,
            "bool"             => Token::TyBool,
            "str"              => Token::TyStr,
            "trit"             => Token::TyTrit,
            "tvec"             => Token::TyTvec,
            "tmat"             => Token::TyTmat,
            "tensor"           => Token::TyTensor,
            "avx2_available"   => Token::HwCondAvx2,
            "ternary_chip"     => Token::HwCondTernary,
            "gpu_available"    => Token::HwCondGpu,
            "0t"               => Token::Trit(0),
            _                  => Token::Ident(s),
        }
    }

    fn handle_indent(&mut self, indent: usize) -> Vec<Tok> {
        let mut toks = vec![];
        let top = *self.indent_stack.last().unwrap();
        let sp = self.span();
        if indent > top {
            self.indent_stack.push(indent);
            toks.push(Tok { token: Token::Indent, span: sp.clone() });
        } else {
            while *self.indent_stack.last().unwrap() > indent {
                self.indent_stack.pop();
                toks.push(Tok { token: Token::Dedent, span: sp.clone() });
            }
        }
        toks
    }

    pub fn tokenize(&mut self) -> Vec<Tok> {
        let mut tokens: Vec<Tok> = vec![];
        let mut at_line_start = true;

        loop {
            // flush pending (indent/dedent)
            tokens.extend(self.pending.drain(..));

            if self.pos >= self.src.len() {
                // emit remaining DEDENTs
                let sp = self.span();
                while self.indent_stack.len() > 1 {
                    self.indent_stack.pop();
                    tokens.push(Tok { token: Token::Dedent, span: sp.clone() });
                }
                tokens.push(Tok { token: Token::Eof, span: sp });
                break;
            }

            let sp = self.span();
            let c = self.advance().unwrap();

            // Handle indentation at start of line
            if at_line_start && c != '\n' && c != '#' {
                at_line_start = false;
                let mut col = 1usize;
                // count spaces already consumed — backtrack via col counter
                // We track by measuring from col=1
                let spaces = if c == ' ' {
                    let mut n = 1;
                    while self.peek() == Some(' ') { self.advance(); n += 1; }
                    n
                } else if c == '\t' {
                    let mut n = 4;
                    while self.peek() == Some('\t') { self.advance(); n += 4; }
                    n
                } else {
                    col = 0;
                    0
                };

                let indent_toks = self.handle_indent(spaces);
                tokens.extend(indent_toks);

                if col == 0 {
                    // c was not whitespace, process it below (fall through)
                    // but we already consumed c — re-process
                    let tok = self.lex_char(c, &sp);
                    if let Some(t) = tok { tokens.push(t); }
                }
                continue;
            }

            match c {
                '\n' => {
                    tokens.push(Tok { token: Token::Newline, span: sp });
                    at_line_start = true;
                }
                ' ' | '\t' | '\r' => {}
                '#' => self.skip_comment(),
                '"' => {
                    let s = self.read_string();
                    let is_glob = s.contains('*');
                    tokens.push(Tok {
                        token: if is_glob { Token::Glob } else { Token::Str(s) },
                        span: sp
                    });
                }
                '@' => {
                    // hardware hints
                    let mut kw = String::new();
                    while self.peek().map_or(false, |c| c.is_alphanumeric() || c == '_') {
                        kw.push(self.advance().unwrap());
                    }
                    let tok = match kw.as_str() {
                        "mmap" => Token::HwMmap,
                        "avx2" => Token::HwAvx2,
                        "cpu"  => Token::HwCpu,
                        "auto" => Token::HwAuto,
                        _      => Token::Ident(format!("@{kw}")),
                    };
                    tokens.push(Tok { token: tok, span: sp });
                }
                '+' => {
                    // +1 is trit literal
                    if self.peek() == Some('1') {
                        self.advance();
                        tokens.push(Tok { token: Token::Trit(1), span: sp });
                    } else {
                        tokens.push(Tok { token: Token::Plus, span: sp });
                    }
                }
                '-' => {
                    if self.peek() == Some('>') {
                        self.advance();
                        tokens.push(Tok { token: Token::Arrow, span: sp });
                    } else if self.peek() == Some('1') {
                        self.advance();
                        tokens.push(Tok { token: Token::Trit(-1), span: sp });
                    } else {
                        tokens.push(Tok { token: Token::Minus, span: sp });
                    }
                }
                '*' => tokens.push(Tok { token: Token::Star, span: sp }),
                '/' => {
                    if self.peek() == Some('/') {
                        self.skip_comment(); // // line comment
                    } else {
                        tokens.push(Tok { token: Token::Slash, span: sp });
                    }
                }
                '%' => tokens.push(Tok { token: Token::Percent, span: sp }),
                '^' => tokens.push(Tok { token: Token::Caret, span: sp }),
                '|' => tokens.push(Tok { token: Token::Pipe, span: sp }),
                '.' => {
                    if self.peek() == Some('.') {
                        self.advance(); // consume second '.'
                        tokens.push(Tok { token: Token::DotDot, span: sp });
                    } else {
                        tokens.push(Tok { token: Token::Dot, span: sp });
                    }
                }
                ':' => tokens.push(Tok { token: Token::Colon, span: sp }),
                ',' => tokens.push(Tok { token: Token::Comma, span: sp }),
                '(' => tokens.push(Tok { token: Token::LParen, span: sp }),
                ')' => tokens.push(Tok { token: Token::RParen, span: sp }),
                '[' => tokens.push(Tok { token: Token::LBracket, span: sp }),
                ']' => tokens.push(Tok { token: Token::RBracket, span: sp }),
                '=' => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push(Tok { token: Token::EqEq, span: sp });
                    } else {
                        tokens.push(Tok { token: Token::Eq, span: sp });
                    }
                }
                '!' => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push(Tok { token: Token::BangEq, span: sp });
                    }
                }
                '<' => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push(Tok { token: Token::LtEq, span: sp });
                    } else {
                        tokens.push(Tok { token: Token::Lt, span: sp });
                    }
                }
                '>' => {
                    if self.peek() == Some('=') {
                        self.advance();
                        tokens.push(Tok { token: Token::GtEq, span: sp });
                    } else {
                        tokens.push(Tok { token: Token::Gt, span: sp });
                    }
                }
                c if c.is_ascii_digit() => {
                    let tok = self.read_number(c);
                    tokens.push(Tok { token: tok, span: sp });
                }
                c if c.is_alphabetic() || c == '_' => {
                    let tok = self.read_ident(c);
                    tokens.push(Tok { token: tok, span: sp });
                }
                _ => {}
            }
        }
        tokens
    }

    fn lex_char(&mut self, c: char, sp: &Span) -> Option<Tok> {
        let tok = match c {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '.' => Token::Dot,
            ':' => Token::Colon,
            // Handle // line comments (called when / is at start of line)
            '/' => {
                if self.peek() == Some('/') {
                    self.skip_comment(); // consume rest of line
                    return None;
                }
                Token::Slash
            }
            '#' => { self.skip_comment(); return None; }
            _ if c.is_alphabetic() || c == '_' => self.read_ident(c),
            _ if c.is_ascii_digit() => self.read_number(c),
            _ => return None,
        };
        Some(Tok { token: tok, span: sp.clone() })
    }
}

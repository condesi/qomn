// ═══════════════════════════════════════════════════════════════════════════
// CRYS-L v3.1 — WebAssembly (WASM) Backend
// Emits WebAssembly Text Format (WAT) from CRYS-L oracle definitions.
// WAT can be assembled to .wasm via:
//   wat2wasm out.wat -o out.wasm
// Or loaded directly in a browser via WebAssembly.instantiateStreaming().
// ═══════════════════════════════════════════════════════════════════════════

use crate::ast::{Expr, BinaryOp, UnaryOp, OracleDecl};

/// WAT stack-machine code generation context
pub struct WatCtx {
    lines:  Vec<String>,
    params: Vec<String>,
    depth:  usize,          // indentation depth
    locals: Vec<String>,    // declared local vars
}

impl WatCtx {
    fn new(depth: usize) -> Self {
        WatCtx { lines: Vec::new(), params: Vec::new(), depth, locals: Vec::new() }
    }

    fn ind(&self) -> String { "    ".repeat(self.depth) }

    fn emit(&mut self, s: impl Into<String>) {
        self.lines.push(format!("{}{}", self.ind(), s.into()));
    }

    fn emit_comment(&mut self, s: impl Into<String>) {
        self.lines.push(format!("{}  ;; {}", self.ind(), s.into()));
    }

    /// Emit WAT instructions to push expr result onto the f64 stack.
    /// WAT is a stack machine: each instruction pops operands and pushes result.
    fn compile_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Float(f) => {
                self.emit(format!("f64.const {:?}", f));
            }
            Expr::Int(n) => {
                self.emit(format!("f64.const {:?}", *n as f64));
            }
            Expr::Ident(name) => {
                if self.params.contains(name) {
                    self.emit(format!("local.get ${}", name));
                } else {
                    self.emit_comment(format!("undefined var: {}", name));
                    self.emit("f64.const 0.0");
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                match op {
                    BinaryOp::Div => {
                        // Safe div: divisor + epsilon
                        self.compile_expr(lhs);
                        self.compile_expr(rhs);
                        self.emit("f64.const 1e-12");
                        self.emit("f64.add");
                        self.emit("f64.div");
                    }
                    BinaryOp::Pow => {
                        // Inline common exponents
                        let exp_val = match rhs.as_ref() {
                            Expr::Float(e) => Some(*e),
                            Expr::Int(n)   => Some(*n as f64),
                            _              => None,
                        };
                        match exp_val {
                            Some(e) if (e - 0.5).abs() < 1e-9 => {
                                self.compile_expr(lhs);
                                self.emit("f64.sqrt");
                            }
                            Some(e) if (e - 2.0).abs() < 1e-9 => {
                                self.compile_expr(lhs);
                                self.emit("local.tee $__sq");
                                self.ensure_local("__sq");
                                self.emit("local.get $__sq");
                                self.emit("f64.mul");
                            }
                            Some(e) if (e - 3.0).abs() < 1e-9 => {
                                self.compile_expr(lhs);
                                self.emit("local.tee $__cb");
                                self.ensure_local("__cb");
                                self.emit("local.get $__cb");
                                self.emit("local.get $__cb");
                                self.emit("f64.mul");
                                self.emit("f64.mul");
                            }
                            Some(e) if (e - 0.25).abs() < 1e-9 => {
                                self.compile_expr(lhs);
                                self.emit("f64.sqrt");
                                self.emit("f64.sqrt");
                            }
                            Some(e) if (e - (-1.0)).abs() < 1e-9 => {
                                self.emit("f64.const 1.0");
                                self.compile_expr(lhs);
                                self.emit("f64.const 1e-12");
                                self.emit("f64.add");
                                self.emit("f64.div");
                            }
                            _ => {
                                // General pow via WASM import (host must provide Math.pow)
                                self.compile_expr(lhs);
                                self.compile_expr(rhs);
                                self.emit("call $math_pow");
                            }
                        }
                    }
                    _ => {
                        self.compile_expr(lhs);
                        self.compile_expr(rhs);
                        let instr = match op {
                            BinaryOp::Add => "f64.add",
                            BinaryOp::Sub => "f64.sub",
                            BinaryOp::Mul => "f64.mul",
                            BinaryOp::Eq  => "f64.eq",
                            BinaryOp::Lt  => "f64.lt",
                            BinaryOp::Gt  => "f64.gt",
                            BinaryOp::Le  => "f64.le",
                            BinaryOp::Ge  => "f64.ge",
                            BinaryOp::Ne  => "f64.ne",
                            _             => "f64.add",  // fallback
                        };
                        self.emit(instr);
                    }
                }
            }
            Expr::Unary(UnaryOp::Neg, inner) => {
                self.compile_expr(inner);
                self.emit("f64.neg");
            }
            Expr::Call(func, args) => {
                if let Expr::Ident(name) = func.as_ref() {
                    match name.as_str() {
                        "sqrt" => {
                            if let Some(a) = args.first() { self.compile_expr(a); }
                            self.emit("f64.sqrt");
                        }
                        "abs" => {
                            if let Some(a) = args.first() { self.compile_expr(a); }
                            self.emit("f64.abs");
                        }
                        "floor" => {
                            if let Some(a) = args.first() { self.compile_expr(a); }
                            self.emit("f64.floor");
                        }
                        "ceil" => {
                            if let Some(a) = args.first() { self.compile_expr(a); }
                            self.emit("f64.ceil");
                        }
                        "trunc" => {
                            if let Some(a) = args.first() { self.compile_expr(a); }
                            self.emit("f64.trunc");
                        }
                        "nearest" | "round" => {
                            if let Some(a) = args.first() { self.compile_expr(a); }
                            self.emit("f64.nearest");
                        }
                        "min" => {
                            if let Some(a) = args.get(0) { self.compile_expr(a); }
                            if let Some(b) = args.get(1) { self.compile_expr(b); }
                            self.emit("f64.min");
                        }
                        "max" => {
                            if let Some(a) = args.get(0) { self.compile_expr(a); }
                            if let Some(b) = args.get(1) { self.compile_expr(b); }
                            self.emit("f64.max");
                        }
                        "clamp" => {
                            // clamp(x, lo, hi) = max(lo, min(hi, x))
                            if let Some(x)  = args.get(0) { self.compile_expr(x); }
                            if let Some(hi) = args.get(2) { self.compile_expr(hi); }
                            self.emit("f64.min");
                            if let Some(lo) = args.get(1) { self.compile_expr(lo); }
                            self.emit("f64.max");
                        }
                        "copysign" => {
                            if let Some(a) = args.get(0) { self.compile_expr(a); }
                            if let Some(b) = args.get(1) { self.compile_expr(b); }
                            self.emit("f64.copysign");
                        }
                        // Math functions via import
                        "sin" | "cos" | "tan" | "log" | "log2" | "log10"
                        | "exp" | "atan" | "atan2" | "asin" | "acos" | "pow" => {
                            for a in args { self.compile_expr(a); }
                            self.emit(format!("call $math_{}", name));
                        }
                        _ => {
                            // Unknown function: try to call it
                            for a in args { self.compile_expr(a); }
                            self.emit(format!("call ${}  ;; user-defined or unknown", name));
                        }
                    }
                } else {
                    self.emit("f64.const 0.0  ;; complex call expr");
                }
            }
            _ => {
                self.emit("f64.const 0.0  ;; unhandled expr");
            }
        }
    }

    fn ensure_local(&mut self, name: &str) {
        if !self.locals.contains(&name.to_string()) {
            self.locals.push(name.to_string());
        }
    }
}

/// Track which math imports are needed
fn collect_math_imports(expr: &Expr, needed: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Call(func, args) => {
            if let Expr::Ident(name) = func.as_ref() {
                match name.as_str() {
                    "sin"|"cos"|"tan"|"log"|"log2"|"log10"
                    |"exp"|"atan"|"atan2"|"asin"|"acos"|"pow" => {
                        needed.insert(name.clone());
                    }
                    _ => {}
                }
            }
            for a in args { collect_math_imports(a, needed); }
        }
        Expr::Binary(_, l, r) => {
            // Check for general pow (non-integer exponent)
            if let Expr::Binary(BinaryOp::Pow, lhs, rhs) = expr {
                match rhs.as_ref() {
                    Expr::Float(e) if ![0.5,2.0,3.0,0.25,-1.0].iter().any(|x| (x-e).abs()<1e-9)
                        => { needed.insert("pow".into()); }
                    Expr::Int(_) => {} // handled inline
                    _ => { needed.insert("pow".into()); }
                }
                collect_math_imports(lhs, needed);
                collect_math_imports(rhs, needed);
                return;
            }
            collect_math_imports(l, needed);
            collect_math_imports(r, needed);
        }
        Expr::Unary(_, inner) => collect_math_imports(inner, needed),
        _ => {}
    }
}

/// Emit WAT module for an oracle declaration.
/// Returns the complete WAT text.
pub fn oracle_to_wat(oracle: &OracleDecl) -> String {
    let mut out = Vec::<String>::new();
    out.push(format!(";; CRYS-L v3.1 WAT — oracle: {}", oracle.name));
    out.push(";; Generated by crysl wasm_backend".into());
    out.push(";; Assemble: wat2wasm oracle.wat -o oracle.wasm".into());
    out.push("".into());
    out.push("(module".into());

    // Collect math imports needed
    let mut math_needed = std::collections::HashSet::<String>::new();
    for stmt in &oracle.body {
        use crate::ast::Stmt;
        match stmt {
            Stmt::Let { val, .. } | Stmt::Return(val) => {
                collect_math_imports(val, &mut math_needed);
            }
            _ => {}
        }
    }

    // Emit math imports
    let nparams: Vec<(&str, usize)> = vec![
        ("sin",1),("cos",1),("tan",1),("log",1),("log2",1),("log10",1),
        ("exp",1),("atan",1),("asin",1),("acos",1),("pow",2),("atan2",2),
    ];
    for (name, n) in &nparams {
        if math_needed.contains(*name) {
            let params = (0..*n).map(|_| "f64").collect::<Vec<_>>().join(" ");
            out.push(format!(
                "  (import \"Math\" \"{}\" (func $math_{} (param {}) (result f64)))",
                name, name, params
            ));
        }
    }
    // Always import pow if needed for general case
    if math_needed.contains("pow") && !out.iter().any(|l| l.contains("$math_pow")) {
        out.push("  (import \"Math\" \"pow\" (func $math_pow (param f64 f64) (result f64)))".into());
    } else if !math_needed.contains("pow") {
        // Add it anyway for general ^ operator fallback
        out.push("  (import \"Math\" \"pow\" (func $math_pow (param f64 f64) (result f64)))".into());
    }

    if !math_needed.is_empty() { out.push("".into()); }

    // Function signature
    let param_list: Vec<String> = oracle.params.iter()
        .map(|p| format!("(param ${} f64)", p.name))
        .collect();
    out.push(format!("  (func $oracle_{} (export \"oracle_{}\")", oracle.name, oracle.name));
    for p in &param_list { out.push(format!("    {}", p)); }
    out.push("    (result f64)".into());

    // Compile body
    let mut ctx = WatCtx::new(2);
    ctx.params = oracle.params.iter().map(|p| p.name.clone()).collect();

    let mut last_expr_compiled = false;

    for stmt in &oracle.body {
        use crate::ast::Stmt;
        match stmt {
            Stmt::Let { name, val, .. } => {
                // Declare local + store
                ctx.ensure_local(name);
                ctx.compile_expr(val);
                ctx.emit(format!("local.set ${}", name));
                last_expr_compiled = false;
            }
            Stmt::Return(expr2) => {
                ctx.compile_expr(expr2);
                last_expr_compiled = true;
            }
            _ => {}
        }
    }

    // If no explicit return, push last local
    if !last_expr_compiled {
        if let Some(last_local) = ctx.locals.last().cloned() {
            ctx.emit(format!("local.get ${}", last_local));
        } else {
            ctx.emit("f64.const 0.0");
        }
    }

    // Emit local declarations first
    let all_locals: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        let mut v = Vec::new();
        for l in &ctx.locals {
            if seen.insert(l.clone()) { v.push(l.clone()); }
        }
        v
    };
    for local in &all_locals {
        out.push(format!("    (local ${} f64)", local));
    }
    if !all_locals.is_empty() { out.push("".into()); }

    // Emit compiled instructions
    for line in &ctx.lines {
        out.push(format!("    {}", line.trim_start_matches("        ").trim_start_matches("    ")));
    }

    out.push("  )".into());
    out.push("".into());
    out.push(")".into());

    out.join("\n")
}

/// Try to assemble WAT to WASM binary (requires wat2wasm in PATH).
/// Returns base64-encoded .wasm, or None if wat2wasm not available.
pub fn wat_to_wasm_base64(wat: &str, name: &str) -> Option<String> {
    let dir = "/tmp/crysl_wasm";
    std::fs::create_dir_all(dir).ok()?;
    let wat_path  = format!("{}/{}.wat", dir, name);
    let wasm_path = format!("{}/{}.wasm", dir, name);
    std::fs::write(&wat_path, wat).ok()?;

    let res = std::process::Command::new("wat2wasm")
        .args([&wat_path, "-o", &wasm_path])
        .output().ok()?;
    if !res.status.success() { return None; }

    let bytes = std::fs::read(&wasm_path).ok()?;
    // base64 encode manually (no dep)
    Some(base64_encode(&bytes))
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() * 4 / 3) + 4);
    let chunks = data.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        out.push(CHARS[(b0 >> 2)] as char);
        out.push(CHARS[((b0 & 0x3) << 4) | (b1 >> 4)] as char);
        if chunk.len() > 1 { out.push(CHARS[((b1 & 0xf) << 2) | (b2 >> 6)] as char); } else { out.push('='); }
        if chunk.len() > 2 { out.push(CHARS[b2 & 0x3f] as char); } else { out.push('='); }
    }
    out
}

/// Route handler: POST /compile?target=wasm
/// Body: {"src":"oracle pump_size(...) = ..."}
/// Returns: {"ok":true,"wat":"...", "wasm_b64":"...", "size_bytes":N}
pub fn handle_compile_wasm(body: &str, prog: &crate::ast::Program) -> String {
    let oracles: Vec<&crate::ast::OracleDecl> = prog.decls.iter().filter_map(|d| {
        if let crate::ast::Decl::Oracle(o) = d { Some(o) } else { None }
    }).collect();
    if oracles.is_empty() {
        return r#"{"ok":false,"error":"no oracle declarations found in source"}"#.to_string();
    }
    let oracle = oracles[0];
    let wat = oracle_to_wat(oracle);
    let wat_escaped = wat.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");

    let wasm_info = match wat_to_wasm_base64(&wat, &oracle.name) {
        Some(b64) => {
            let sz = b64.len() * 3 / 4;
            format!(r#","wasm_b64":"{}","wasm_size_bytes":{}"#, b64, sz)
        }
        None => r#","wasm_b64":null,"note":"wat2wasm not in PATH; WAT returned only""#.to_string(),
    };

    format!(
        r#"{{"ok":true,"oracle":"{}","target":"wasm","wat_lines":{}{}}}"#,
        oracle.name,
        wat.lines().count(),
        wasm_info
    )
}

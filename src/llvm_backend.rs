// ═══════════════════════════════════════════════════════════════════════════
// CRYS-L v3.0 — LLVM IR Backend
// Emits LLVM IR 18 text from CRYS-L oracle definitions.
// The IR can be compiled to native code via:
//   llc-18 -O3 -filetype=obj out.ll -o out.o
//   clang-18 -shared -fPIC out.o -o libcryslo.so
// ═══════════════════════════════════════════════════════════════════════════

use crate::ast::{Expr, BinaryOp, UnaryOp, OracleDecl};

/// LLVM IR code generation context for one function.
pub struct LlvmCtx {
    regs:   usize,               // next SSA register index
    lines:  Vec<String>,         // IR lines
    params: Vec<String>,         // parameter names (without %)
}

impl LlvmCtx {
    fn new() -> Self { LlvmCtx { regs: 0, lines: Vec::new(), params: Vec::new() } }

    fn fresh(&mut self) -> String {
        let r = format!("%t{}", self.regs);
        self.regs += 1;
        r
    }

    fn emit(&mut self, s: impl Into<String>) { self.lines.push(s.into()); }

    /// Compile expression to SSA value, return register name
    fn compile_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Float(f) => {
                let r = self.fresh();
                // LLVM uses hex float literals for exactness, but decimal works fine
                self.emit(format!("  {} = fadd double 0.0, {:.17e}", r, f));
                r
            }
            Expr::Int(n) => {
                let r = self.fresh();
                self.emit(format!("  {} = fadd double 0.0, {:.1}", r, *n as f64));
                r
            }
            Expr::Ident(name) => {
                // Map to parameter or zero
                if self.params.contains(name) {
                    format!("%{}", name)
                } else {
                    let r = self.fresh();
                    self.emit(format!("  {} = fadd double 0.0, 0.0  ; undefined: {}", r, name));
                    r
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let lv = self.compile_expr(lhs);
                let rv = self.compile_expr(rhs);
                let r  = self.fresh();
                let instr = match op {
                    BinaryOp::Add => format!("  {} = fadd double {}, {}", r, lv, rv),
                    BinaryOp::Sub => format!("  {} = fsub double {}, {}", r, lv, rv),
                    BinaryOp::Mul => format!("  {} = fmul double {}, {}", r, lv, rv),
                    BinaryOp::Div => {
                        // Safe div: guard against divide-by-zero
                        let eps = self.fresh();
                        let safe = self.fresh();
                        self.emit(format!("  {} = fadd double {}, 1.0e-12", eps, rv));
                        let tmp = format!("  {} = fdiv double {}, {}", safe, lv, eps);
                        self.emit(tmp);
                        return safe;
                    }
                    BinaryOp::Pow => {
                        // Inline common exponents; fall back to llvm.pow intrinsic
                        let exponent = match rhs.as_ref() {
                            Expr::Float(e) => Some(*e),
                            Expr::Int(n)   => Some(*n as f64),
                            _              => None,
                        };
                        match exponent {
                            Some(e) if (e - 0.5).abs() < 1e-9 => {
                                // x^0.5 → llvm.sqrt
                                self.emit(format!("  {} = call double @llvm.sqrt.f64(double {})", r, lv));
                                return r;
                            }
                            Some(e) if (e - 2.0).abs() < 1e-9 => {
                                self.emit(format!("  {} = fmul double {}, {}", r, lv, lv));
                                return r;
                            }
                            Some(e) if (e - 3.0).abs() < 1e-9 => {
                                let sq = self.fresh();
                                self.emit(format!("  {} = fmul double {}, {}", sq, lv, lv));
                                self.emit(format!("  {} = fmul double {}, {}", r, sq, lv));
                                return r;
                            }
                            Some(e) if (e - 0.25).abs() < 1e-9 => {
                                let sq = self.fresh();
                                self.emit(format!("  {} = call double @llvm.sqrt.f64(double {})", sq, lv));
                                self.emit(format!("  {} = call double @llvm.sqrt.f64(double {})", r, sq));
                                return r;
                            }
                            _ => {
                                self.emit(format!(
                                    "  {} = call double @llvm.pow.f64(double {}, double {})",
                                    r, lv, rv
                                ));
                                return r;
                            }
                        }
                    }
                    BinaryOp::Eq => {
                        self.emit(format!("  %cmp_{} = fcmp oeq double {}, {}", self.regs, lv, rv));
                        self.emit(format!("  {} = uitofp i1 %cmp_{} to double", r, self.regs - 1));
                        // adjust regs since we used self.regs inline
                        return r;
                    }
                    BinaryOp::Lt => {
                        let cr = self.regs;
                        self.emit(format!("  %cmp_{} = fcmp olt double {}, {}", cr, lv, rv));
                        self.regs += 1;
                        self.emit(format!("  {} = uitofp i1 %cmp_{} to double", r, cr));
                        return r;
                    }
                    BinaryOp::Gt => {
                        let cr = self.regs;
                        self.emit(format!("  %cmp_{} = fcmp ogt double {}, {}", cr, lv, rv));
                        self.regs += 1;
                        self.emit(format!("  {} = uitofp i1 %cmp_{} to double", r, cr));
                        return r;
                    }
                    _ => format!("  {} = fadd double {}, 0.0  ; unsupported op", r, lv),
                };
                self.emit(instr);
                r
            }
            Expr::Unary(UnaryOp::Neg, inner) => {
                let iv = self.compile_expr(inner);
                let r  = self.fresh();
                self.emit(format!("  {} = fneg double {}", r, iv));
                r
            }
            Expr::Call(func, args) => {
                if let Expr::Ident(name) = func.as_ref() {
                    let compiled_args: Vec<String> = args.iter()
                        .map(|a| self.compile_expr(a))
                        .collect();
                    let r = self.fresh();
                    let ir = match name.as_str() {
                        "sqrt"  => format!("  {} = call double @llvm.sqrt.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "abs"   => format!("  {} = call double @llvm.fabs.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "floor" => format!("  {} = call double @llvm.floor.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "ceil"  => format!("  {} = call double @llvm.ceil.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "sin"   => format!("  {} = call double @llvm.sin.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "cos"   => format!("  {} = call double @llvm.cos.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "log"   => format!("  {} = call double @llvm.log.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "log2"  => format!("  {} = call double @llvm.log2.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "exp"   => format!("  {} = call double @llvm.exp.f64(double {})", r, compiled_args.get(0).cloned().unwrap_or("0.0".into())),
                        "min"   => format!("  {} = call double @llvm.minnum.f64(double {}, double {})",
                            r,
                            compiled_args.get(0).cloned().unwrap_or("0.0".into()),
                            compiled_args.get(1).cloned().unwrap_or("0.0".into())),
                        "max"   => format!("  {} = call double @llvm.maxnum.f64(double {}, double {})",
                            r,
                            compiled_args.get(0).cloned().unwrap_or("0.0".into()),
                            compiled_args.get(1).cloned().unwrap_or("0.0".into())),
                        "clamp" => {
                            // clamp(x, lo, hi) = min(max(x, lo), hi)
                            let x  = compiled_args.get(0).cloned().unwrap_or("0.0".into());
                            let lo = compiled_args.get(1).cloned().unwrap_or("0.0".into());
                            let hi = compiled_args.get(2).cloned().unwrap_or("1.0".into());
                            let mx = self.fresh();
                            self.emit(format!("  {} = call double @llvm.maxnum.f64(double {}, double {})", mx, x, lo));
                            format!("  {} = call double @llvm.minnum.f64(double {}, double {})", r, mx, hi)
                        }
                        _ => format!("  {} = fadd double 0.0, 0.0  ; unknown fn: {}", r, name),
                    };
                    self.emit(ir);
                    return r;
                }
                let r = self.fresh();
                self.emit(format!("  {} = fadd double 0.0, 0.0  ; complex call", r));
                r
            }
            _ => {
                let r = self.fresh();
                self.emit(format!("  {} = fadd double 0.0, 0.0  ; unhandled expr", r));
                r
            }
        }
    }
}

/// Emit LLVM IR module text for a single oracle declaration.
/// Returns the complete .ll text ready for llc-18.
pub fn oracle_to_llvm_ir(oracle: &OracleDecl) -> String {
    let mut out = Vec::<String>::new();

    // Module header
    out.push(format!("; CRYS-L v3.0 LLVM IR — oracle: {}", oracle.name));
    out.push("; Generated by crysl llvm_backend".into());
    out.push("; Compile: llc-18 -O3 -filetype=obj out.ll -o out.o".into());
    out.push(";          clang-18 -shared -fPIC out.o -o libcrysl_oracle.so".into());
    out.push("".into());
    out.push("source_filename = \"crysl_oracle\"".into());
    out.push("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"".into());
    out.push("target triple = \"x86_64-pc-linux-gnu\"".into());
    out.push("".into());

    // LLVM intrinsic declarations
    for intr in &[
        "declare double @llvm.sqrt.f64(double %Val)",
        "declare double @llvm.fabs.f64(double %Val)",
        "declare double @llvm.floor.f64(double %Val)",
        "declare double @llvm.ceil.f64(double %Val)",
        "declare double @llvm.sin.f64(double %Val)",
        "declare double @llvm.cos.f64(double %Val)",
        "declare double @llvm.log.f64(double %Val)",
        "declare double @llvm.log2.f64(double %Val)",
        "declare double @llvm.exp.f64(double %Val)",
        "declare double @llvm.pow.f64(double %Val, double %Power)",
        "declare double @llvm.minnum.f64(double %Val1, double %Val2)",
        "declare double @llvm.maxnum.f64(double %Val1, double %Val2)",
    ] {
        out.push(intr.to_string());
    }
    out.push("".into());

    // Function signature: double @oracle_NAME(double %param1, double %param2, ...)
    let param_list: Vec<String> = oracle.params.iter()
        .map(|p| format!("double %{}", p.name))
        .collect();
    out.push(format!("define double @oracle_{}({}) {{", oracle.name, param_list.join(", ")));
    out.push("entry:".into());

    // Compile body
    let mut ctx = LlvmCtx::new();
    ctx.params = oracle.params.iter().map(|p| p.name.clone()).collect();

    // Find the return expression — the last binding or 'result' var
    let mut result_reg = "%t_zero".to_string();
    ctx.emit("  %t_zero = fadd double 0.0, 0.0".to_string());

    // Compile each binding statement's expression, last one is return
    for stmt in &oracle.body {
        use crate::ast::Stmt;
        match stmt {
            Stmt::Let { name, val, .. } => {
                let reg = ctx.compile_expr(val);
                // Bind name → reg by aliasing
                let alias = format!("  %v_{} = fadd double {}, 0.0", name, reg);
                ctx.emit(alias);
                result_reg = format!("%v_{}", name);
            }

            Stmt::Return(expr) => {
                result_reg = ctx.compile_expr(expr);
            }
            _ => {}
        }
    }

    // Emit compiled lines
    for line in &ctx.lines {
        out.push(line.clone());
    }

    // Return
    out.push(format!("  ret double {}", result_reg));
    out.push("}".into());
    out.push("".into());

    // Attributes
    out.push(format!(
        "attributes #0 = {{ nounwind readnone speculatable willreturn \"frame-pointer\"=\"none\" \"no-trapping-math\"=\"true\" }}"
    ));

    out.join("\n")
}

/// Compile LLVM IR text to native object (requires llc-18 + clang-18).
/// Returns path to produced .so file, or error string.
pub fn compile_ir_to_so(ir_text: &str, oracle_name: &str) -> Result<String, String> {
    let dir = "/tmp/crysl_llvm";
    std::fs::create_dir_all(dir).ok();
    let ll_path = format!("{}/{}.ll", dir, oracle_name);
    let obj_path = format!("{}/{}.o", dir, oracle_name);
    let so_path  = format!("{}/lib{}.so", dir, oracle_name);

    std::fs::write(&ll_path, ir_text)
        .map_err(|e| format!("write .ll: {}", e))?;

    // llc-18 → object
    let llc = std::process::Command::new("llc-18")
        .args(["-O3", "-filetype=obj", &ll_path, "-o", &obj_path])
        .output()
        .map_err(|e| format!("llc-18 not found: {}", e))?;
    if !llc.status.success() {
        return Err(format!("llc-18 error: {}", String::from_utf8_lossy(&llc.stderr)));
    }

    // clang-18 → shared library
    let clang = std::process::Command::new("clang-18")
        .args(["-shared", "-fPIC", "-O3", &obj_path, "-o", &so_path, "-lm"])
        .output()
        .map_err(|e| format!("clang-18 not found: {}", e))?;
    if !clang.status.success() {
        return Err(format!("clang-18 error: {}", String::from_utf8_lossy(&clang.stderr)));
    }

    Ok(so_path)
}

/// Route handler: POST /compile?backend=llvm
/// Body: {"src":"oracle pump_size(...) = ..."}
/// Returns: {"ok":true,"ir":"...", "so_path":"...", "size_bytes":N}
pub fn handle_compile_llvm(body: &str, prog: &crate::ast::Program) -> String {
    // Find first oracle in program
    let oracles: Vec<&crate::ast::OracleDecl> = prog.decls.iter().filter_map(|d| {
        if let crate::ast::Decl::Oracle(o) = d { Some(o) } else { None }
    }).collect();
    if oracles.is_empty() {
        return r#"{"ok":false,"error":"no oracle declarations found in source"}"#.to_string();
    }
    let oracle = oracles[0];
    let ir = oracle_to_llvm_ir(oracle);
    let ir_escaped = ir.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");

    // Try native compile (best-effort)
    match compile_ir_to_so(&ir, &oracle.name) {
        Ok(so_path) => {
            let size = std::fs::metadata(&so_path).map(|m| m.len()).unwrap_or(0);
            format!(
                r#"{{"ok":true,"oracle":"{}","backend":"llvm18","ir_lines":{},"so_path":"{}","size_bytes":{}}}"#,
                oracle.name,
                ir.lines().count(),
                so_path,
                size
            )
        }
        Err(e) => {
            // Return IR text even if compile failed
            format!(
                r#"{{"ok":true,"oracle":"{}","backend":"llvm18","ir_lines":{},"compile_error":"{}","ir":"{}"}}"#,
                oracle.name,
                ir.lines().count(),
                e.replace('"', "'"),
                ir_escaped
            )
        }
    }
}

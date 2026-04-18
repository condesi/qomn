// ═══════════════════════════════════════════════════════════════════════
// QOMN v1.0 — REPL (Read-Eval-Print Loop)
// Interactive shell for QOMN programs.
// ═══════════════════════════════════════════════════════════════════════

use std::io::{self, BufRead, Write};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::typeck::TypeEnv;
use crate::vm::{Vm, QomniConfig};

const BANNER: &str = r#"
  ╔═══════════════════════════════════════════════════╗
  ║   QOMN v1.0  — QOMN Language REPL           ║
  ║   Qomni AI Lab · Condesi Perú · 2026             ║
  ║                                                   ║
  ║   Tipos: trit  tvec[n]  tmat[r][c]               ║
  ║   Cmds:  :help  :crystals  :routes  :quit        ║
  ╚═══════════════════════════════════════════════════╝
"#;

const HELP: &str = r#"
QOMN REPL Commands:
  :help              this help
  :quit / :q         exit
  :crystals          list registered crystals
  :routes            list route table
  :load <file>       load and execute a .crys file
  :query <text>      route text through crystal router

QOMN Syntax Quick Reference:
  oracle f(x: f32) -> f32:      define physics oracle
      return x * 0.18

  crystal c = load @mmap "c.crystal"   load crystal
  pipe p(x: f32):               define pipeline
      v = encode(x, 4864)
      -> respond(c.infer(x=v))

  route "IGV*" -> crystal:c    define route
  let y = f(1000.0)             evaluate expression

Types: f32  f64  i32  i64  bool  str  trit  tvec[n]  tmat[r][c]
"#;

pub fn run_repl(qomni_url: Option<String>, qomni_key: Option<String>) {
    println!("{}", BANNER);

    let config = QomniConfig {
        base_url: qomni_url.unwrap_or_else(|| "http://109.123.245.234:8090".into()),
        api_key:  qomni_key.unwrap_or_else(|| "your-api-key-here".into()),
    };

    let mut vm      = Vm::new(config);
    let mut typenv  = TypeEnv::new();
    let mut history: Vec<String> = vec![];
    let mut multiline = String::new();
    let mut in_block  = false;

    let stdin  = io::stdin();
    let stdout = io::stdout();

    loop {
        {
            let mut out = stdout.lock();
            if in_block {
                write!(out, "  ... ").unwrap();
            } else {
                write!(out, "crys> ").unwrap();
            }
            out.flush().unwrap();
        }

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            _ => {}
        }
        let line = line.trim_end_matches('\n').trim_end_matches('\r');

        // ── REPL commands ──────────────────────────────────────────
        if !in_block {
            match line.trim() {
                ":quit" | ":q" | "exit" | "quit" => {
                    println!("  Bye.");
                    break;
                }
                ":help" => { println!("{}", HELP); continue; }
                ":crystals" => { println!("  (crystals registered in VM)"); continue; }
                ":routes"   => { println!("  (route table)"); continue; }
                "" => continue,
                cmd if cmd.starts_with(":query ") => {
                    let q = &cmd[7..];
                    println!("  → routing: '{}'", q);
                    continue;
                }
                cmd if cmd.starts_with(":load ") => {
                    let path = cmd[6..].trim();
                    match std::fs::read_to_string(path) {
                        Ok(src) => eval_source(&src, &mut vm, &mut typenv),
                        Err(e)  => eprintln!("  Error loading '{}': {}", path, e),
                    }
                    continue;
                }
                _ => {}
            }
        }

        // ── Multi-line block detection ─────────────────────────────
        multiline.push_str(line);
        multiline.push('\n');

        let trimmed = line.trim();
        if trimmed.ends_with(':') || in_block {
            in_block = true;
            // Empty line ends the block
            if trimmed.is_empty() || (in_block && !line.starts_with("    ") && !line.starts_with('\t') && !trimmed.is_empty() && !trimmed.ends_with(':')) {
                in_block = false;
                let src = multiline.clone();
                multiline.clear();
                history.push(src.clone());
                eval_source(&src, &mut vm, &mut typenv);
            }
            continue;
        }

        let src = multiline.clone();
        multiline.clear();
        history.push(src.clone());
        eval_source(&src, &mut vm, &mut typenv);
    }
}

fn eval_source(src: &str, vm: &mut Vm, typenv: &mut TypeEnv) {
    // Lex
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize();

    // Parse
    let mut parser = Parser::new(tokens);
    let prog = match parser.parse() {
        Ok(p)  => p,
        Err(e) => { eprintln!("  Parse error: {}", e); return; }
    };

    // Type check
    let errors = typenv.check_program(&prog);
    if !errors.is_empty() {
        for e in &errors { eprintln!("  Type error: {}", e); }
        // continue anyway (warn, not halt)
    }

    // Execute
    match vm.run(&prog) {
        Ok(out) => {
            for line in out {
                println!("  → {}", line);
            }
        }
        Err(e) => eprintln!("  Runtime error: {}", e),
    }
}

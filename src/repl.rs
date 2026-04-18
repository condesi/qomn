// ═══════════════════════════════════════════════════════════════════════
// QOMN v3.2 — REPL (Read-Eval-Print Loop)
//
// Dual-mode interactive shell:
//   • Legacy mode:  oracle / crystal / pipe syntax (old AST)
//   • Plan mode:    plan_* syntax (plan_v2 module, SPEC.md v2.3)
//
// Commands:
//   .help                         this help
//   .quit / .q / exit             exit
//   .bench <plan_name> [k=v ...]  benchmark a loaded plan_* plan (1000 iters)
//   .explain <plan_name>          show formula sources, standards, params
//   .load <file>                  load & execute legacy .crys file
//   .load-plan <file>             load plan_* .qomn file into REPL
//   .load-dir <dir>               load all .qomn files from directory
//   .plans                        list loaded plan_* plans
//   .reload                       hot-reload last loaded plan file
//   .check <file>                 type-check a .qomn file
//   .history                      show command history
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::typeck::TypeEnv;
use crate::vm::{Vm, QomniConfig};
use crate::plan_v2;

const BANNER: &str = r#"
  ╔══════════════════════════════════════════════════════╗
  ║   QOMN v3.2  — QOMN Language REPL              ║
  ║   Qomni AI Lab · Condesi Perú · 2026                ║
  ║                                                     ║
  ║   plan_pump_sizing(Q_gpm=500, P_psi=100, eff=0.75) ║
  ║   .help  .plans  .bench  .explain  .load-plan       ║
  ╚══════════════════════════════════════════════════════╝
"#;

const HELP: &str = r#"
QOMN v3.2 REPL — Commands:

  .help                         this help
  .quit / .q / exit             exit
  .plans                        list loaded plan_* plans
  .bench <plan> [k=v ...]       benchmark plan (1000 iterations)
  .explain <plan>               show formulas + standards for plan
  .load <file>                  load legacy .crys file (oracle syntax)
  .load-plan <file>             load plan_* .qomn file into session
  .reload                       reload last .qomn file
  .check <file>                 type-check a .qomn file

Plan_* Syntax Quick Reference:
  plan_pump_sizing(Q_gpm: f64, P_psi: f64, eff: f64 = 0.70) {
      meta { standard: "NFPA 20:2022", source: "§4.26", ... }
      const K = 3960.0;
      let HP = (Q_gpm * P_psi) / (eff * K);
      assert Q_gpm > 0.0 msg "flow must be positive";
      output HP label "Pump HP" unit "HP";
      return { HP: HP };
  }

Calling a plan (inline syntax):
  plan_pump_sizing(Q_gpm=500, P_psi=100, eff=0.75)

Legacy oracle syntax still works for .crys files via .load.
"#;

struct ReplState {
    vm:       Vm,
    typenv:   TypeEnv,
    plans:    Vec<plan_v2::PlanV2Decl>,
    last_file: Option<String>,
    history:  Vec<String>,
}

impl ReplState {
    fn new(config: QomniConfig) -> Self {
        ReplState {
            vm:       Vm::new(config),
            typenv:   TypeEnv::new(),
            plans:    vec![],
            last_file: None,
            history:  vec![],
        }
    }

    fn load_plan_file(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Err(e) => eprintln!("  Error: cannot read '{}': {}", path, e),
            Ok(src) => {
                match plan_v2::parse_plans(&src) {
                    Err(errs) => {
                        for e in &errs { eprintln!("  Parse error {}: {}", e.span, e.message); }
                    }
                    Ok(new_plans) => {
                        let n = new_plans.len();
                        // Replace plans with the same name, append new ones
                        for p in new_plans {
                            if let Some(idx) = self.plans.iter().position(|x| x.name == p.name) {
                                self.plans[idx] = p;
                            } else {
                                self.plans.push(p);
                            }
                        }
                        self.last_file = Some(path.to_string());
                        println!("  Loaded {} plan(s) from '{}'", n, path);
                        for p in &self.plans {
                            println!("    • {} ({} params)", p.name, p.params.len());
                        }
                    }
                }
            }
        }
    }

    fn find_plan(&self, name: &str) -> Option<&plan_v2::PlanV2Decl> {
        self.plans.iter().find(|p| p.name == name)
    }

    fn bench_plan(&self, name: &str, args: &HashMap<String, f64>) {
        let plan = match self.find_plan(name) {
            Some(p) => p,
            None => { eprintln!("  Unknown plan '{}'", name); return; }
        };
        const N: usize = 1000;
        // Warmup
        for _ in 0..10 {
            let _ = plan_v2::execute_plan(plan, args);
        }
        let t0 = std::time::Instant::now();
        let mut last_ok = true;
        for _ in 0..N {
            if plan_v2::execute_plan(plan, args).is_err() { last_ok = false; }
        }
        let total = t0.elapsed();
        let ns = total.as_nanos() as f64 / N as f64;
        if !last_ok {
            eprintln!("  Warning: some iterations returned errors");
        }
        println!("  .bench {}", name);
        println!("    iterations : {}", N);
        println!("    total      : {:?}", total);
        println!("    median ns  : {:.1} ns/call", ns);
        if ns < 1000.0 {
            println!("    throughput : {:.0} M calls/s", 1_000_000_000.0 / ns / 1_000_000.0);
        }
    }

    fn explain_plan(&self, name: &str) {
        let plan = match self.find_plan(name) {
            Some(p) => p,
            None => { eprintln!("  Unknown plan '{}'", name); return; }
        };
        println!("  .explain {}", plan.name);
        println!();
        println!("  Standard : {}", plan.meta.standard);
        println!("  Source   : {}", plan.meta.source);
        println!("  Domain   : {}", plan.meta.domain);
        println!("  Version  : {}", plan.meta.version.as_deref().unwrap_or("-"));
        println!();
        println!("  Parameters:");
        for p in &plan.params {
            match p.default {
                Some(d) => println!("    {:20} : {:?}  (default = {})", p.name, p.ty, d),
                None    => println!("    {:20} : {:?}  (required)", p.name, p.ty),
            }
        }
        println!();
        println!("  Formulas:");
        let mut has_formula = false;
        for item in &plan.body {
            if let plan_v2::PV2Item::Formula { label, text, .. } = item {
                println!("    [{}] {}", label, text);
                has_formula = true;
            }
        }
        if !has_formula { println!("    (none documented)"); }
        println!();
        println!("  Asserts:");
        let mut has_assert = false;
        for item in &plan.body {
            if let plan_v2::PV2Item::Assert { msg, .. } = item {
                println!("    \"{}\"", msg);
                has_assert = true;
            }
        }
        if !has_assert { println!("    (none)"); }
    }

    fn call_plan_inline(&self, src: &str) {
        // Detect: plan_name(k=v, ...) or plan_name(v1, v2, ...)
        // Simple approach: find first '(' and last ')'
        let src = src.trim();
        let lparen = match src.find('(') {
            Some(i) => i,
            None => { eprintln!("  Expected '(' in plan call"); return; }
        };
        let rparen = match src.rfind(')') {
            Some(i) => i,
            None => { eprintln!("  Expected ')' in plan call"); return; }
        };
        let name = src[..lparen].trim();
        let args_str = &src[lparen+1..rparen];
        let plan = match self.find_plan(name) {
            Some(p) => p,
            None => { eprintln!("  Unknown plan '{}' — use .load-plan to load it first", name); return; }
        };

        // Parse args: positional or k=v
        let mut args: HashMap<String, f64> = HashMap::new();
        if !args_str.trim().is_empty() {
            let parts: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
            for (i, part) in parts.iter().enumerate() {
                if let Some(eq) = part.find('=') {
                    let k = part[..eq].trim().to_string();
                    let v: f64 = part[eq+1..].trim().parse().unwrap_or_else(|_| {
                        eprintln!("  Warning: cannot parse value for '{}'", k);
                        0.0
                    });
                    args.insert(k, v);
                } else if let Ok(v) = part.parse::<f64>() {
                    if i < plan.params.len() {
                        args.insert(plan.params[i].name.clone(), v);
                    }
                } else {
                    eprintln!("  Warning: cannot parse arg '{}'", part);
                }
            }
        }

        let t0 = std::time::Instant::now();
        match plan_v2::execute_plan(plan, &args) {
            Err(e) => eprintln!("  Execution error: {}", e),
            Ok(result) => {
                let elapsed = t0.elapsed();
                for (k, v) in &result.outputs {
                    println!("  {} = {:.6}", k, v);
                }
                println!("  Latency: {:?}", elapsed);
            }
        }
    }
}

pub fn run_repl(qomni_url: Option<String>, qomni_key: Option<String>) {
    println!("{}", BANNER);

    let config = QomniConfig {
        base_url: qomni_url.unwrap_or_else(|| "http://109.123.245.234:8090".into()),
        api_key:  qomni_key.unwrap_or_else(|| "your-api-key-here".into()),
    };

    let mut state = ReplState::new(config);
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
                write!(out, "qomn> ").unwrap();
            }
            out.flush().unwrap();
        }

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            _ => {}
        }
        let line = line.trim_end_matches('\n').trim_end_matches('\r');

        // ── REPL dot-commands ──────────────────────────────────────
        if !in_block {
            let trimmed = line.trim();
            match trimmed {
                ".quit" | ".q" | ":quit" | ":q" | "exit" | "quit" => {
                    println!("  Bye.");
                    break;
                }
                ".help" | ":help" => { println!("{}", HELP); continue; }
                ".plans" | ":crystals" => {
                    if state.plans.is_empty() {
                        println!("  (no plans loaded — use .load-plan <file>)");
                    } else {
                        for p in &state.plans {
                            println!("  {} ({} params)", p.name, p.params.len());
                        }
                    }
                    continue;
                }
                ".reload" => {
                    if let Some(path) = state.last_file.clone() {
                        state.load_plan_file(&path);
                    } else {
                        eprintln!("  Nothing loaded yet — use .load-plan <file>");
                    }
                    continue;
                }
                "" => continue,
                ".history" => {
                    if state.history.is_empty() {
                        println!("  (no history yet)");
                    } else {
                        for (i, h) in state.history.iter().enumerate().rev().take(20) {
                            println!("  [{:3}] {}", i, h.trim());
                        }
                    }
                    continue;
                }
                cmd if cmd.starts_with(".load-dir ") => {
                    let dir = cmd[10..].trim();
                    match std::fs::read_dir(dir) {
                        Err(e) => eprintln!("  Error: {}", e),
                        Ok(entries) => {
                            let mut loaded = 0;
                            let mut paths: Vec<String> = entries
                                .filter_map(|e| e.ok())
                                .map(|e| e.path())
                                .filter(|p| p.extension().map(|x| x == "qomn").unwrap_or(false))
                                .map(|p| p.to_string_lossy().into_owned())
                                .collect();
                            paths.sort();
                            for path in &paths {
                                state.load_plan_file(path);
                                loaded += 1;
                            }
                            if loaded == 0 { println!("  (no .qomn files found in '{}')", dir); }
                        }
                    }
                    continue;
                }
                cmd if cmd.starts_with(".load-plan ") => {
                    let path = cmd[11..].trim();
                    state.load_plan_file(path);
                    continue;
                }
                cmd if cmd.starts_with(":load ") || cmd.starts_with(".load ") => {
                    let path = if cmd.starts_with(":load ") { &cmd[6..] } else { &cmd[6..] }.trim();
                    match std::fs::read_to_string(path) {
                        Ok(src) => eval_legacy_source(&src, &mut state.vm, &mut state.typenv),
                        Err(e)  => eprintln!("  Error loading '{}': {}", path, e),
                    }
                    continue;
                }
                cmd if cmd.starts_with(".check ") => {
                    let path = cmd[7..].trim();
                    match std::fs::read_to_string(path) {
                        Err(e) => eprintln!("  Error: {}", e),
                        Ok(src) => {
                            match plan_v2::parse_plans(&src) {
                                Err(errs) => { for e in &errs { eprintln!("  Parse error {}: {}", e.span, e.message); } }
                                Ok(plans) => {
                                    let mut any_err = false;
                                    for plan in &plans {
                                        let errs = plan_v2::typecheck_plan(plan);
                                        if errs.is_empty() { println!("  OK  {}", plan.name); }
                                        else { for e in &errs { eprintln!("  Error {}: {}", e.span, e.message); } any_err = true; }
                                    }
                                    if !any_err { println!("  --- {} plan(s) OK", plans.len()); }
                                }
                            }
                        }
                    }
                    continue;
                }
                cmd if cmd.starts_with(".explain ") => {
                    let name = cmd[9..].trim();
                    state.explain_plan(name);
                    continue;
                }
                cmd if cmd.starts_with(".bench ") => {
                    let rest = cmd[7..].trim();
                    // Split first word = plan name, rest = k=v args
                    let (plan_name, args_str) = rest.split_once(' ').unwrap_or((rest, ""));
                    let mut args: HashMap<String, f64> = HashMap::new();
                    for part in args_str.split_whitespace() {
                        if let Some(eq) = part.find('=') {
                            let k = part[..eq].to_string();
                            let v: f64 = part[eq+1..].parse().unwrap_or(0.0);
                            args.insert(k, v);
                        }
                    }
                    state.bench_plan(plan_name, &args);
                    continue;
                }
                // plan_* inline call detection: plan_name(...)
                cmd if cmd.starts_with("plan_") && cmd.contains('(') => {
                    state.call_plan_inline(cmd);
                    continue;
                }
                _ => {}
            }
        }

        // ── Multi-line block detection (legacy oracle syntax) ──────
        multiline.push_str(line);
        multiline.push('\n');

        let trimmed = line.trim();
        if trimmed.ends_with(':') || in_block {
            in_block = true;
            if trimmed.is_empty() || (in_block && !line.starts_with("    ") && !line.starts_with('\t') && !trimmed.is_empty() && !trimmed.ends_with(':')) {
                in_block = false;
                let src = multiline.clone();
                multiline.clear();
                state.history.push(src.clone());
                eval_legacy_source(&src, &mut state.vm, &mut state.typenv);
            }
            continue;
        }

        // ── Plan_* multi-line detection ─────────────────────────────
        // If source accumulates a plan_* definition (starts with plan_, ends with })
        let src = multiline.clone();
        multiline.clear();
        let src_trimmed = src.trim();
        if src_trimmed.starts_with("plan_") {
            match plan_v2::parse_plans(src_trimmed) {
                Ok(plans) if !plans.is_empty() => {
                    for p in plans {
                        if let Some(idx) = state.plans.iter().position(|x| x.name == p.name) {
                            println!("  Updated plan: {}", p.name);
                            state.plans[idx] = p;
                        } else {
                            println!("  Loaded plan: {}", p.name);
                            state.plans.push(p);
                        }
                    }
                }
                _ => {
                    eval_legacy_source(src_trimmed, &mut state.vm, &mut state.typenv);
                }
            }
        } else {
            state.history.push(src.clone());
            eval_legacy_source(src_trimmed, &mut state.vm, &mut state.typenv);
        }
    }
}

fn eval_legacy_source(src: &str, vm: &mut Vm, typenv: &mut TypeEnv) {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let prog = match parser.parse() {
        Ok(p)  => p,
        Err(e) => { eprintln!("  Parse error: {}", e); return; }
    };
    let errors = typenv.check_program(&prog);
    if !errors.is_empty() {
        for e in &errors { eprintln!("  Type error: {}", e); }
    }
    match vm.run(&prog) {
        Ok(out) => { for line in out { println!("  → {}", line); } }
        Err(e) => eprintln!("  Runtime error: {}", e),
    }
}

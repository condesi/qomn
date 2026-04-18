// ═══════════════════════════════════════════════════════════════════════
// QOMN v2.0 — Cognitive Execution Engine + JIT + Typed Units + Plans
//
// Pipeline: QOMN source
//   → Lexer → Parser → AST
//   → HIR (High-Level IR graph: node fusion, oracle batching)
//   → MIR (Tensor + Control IR)
//   → CRYS-ISA Bytecode (v1.4: async oracle, PAR_BEGIN, LOAD_CRYS mode)
//   → Optimizer (DCE, Oracle Fusion, MM+ACT merge, NOP strip)
//   → Runtime (async oracle engine, crystal cache, mem pool, profiler)
//   → Backend CPU (AVX2 sign-blend MM_TERN — 15.62 GOPS on EPYC)
//   → JIT (Cranelift v1.6: oracle bodies → native x86-64)
//
// CLI:
//   qomn                         REPL
//   qomn run     <file.crys>     execute (tree-walk VM)
//   qomn run-jit <file.crys>     execute with JIT oracle dispatch (v1.6)
//   qomn check <file.crys>       type-check only
//   qomn lex   <file.crys>       dump tokens
//   qomn hir   <file.crys>       dump High-Level IR graph
//   qomn ir    <file.crys>       dump CRYS-ISA Bytecode IR
//   qomn jit   <file.crys>       compile oracles → native x86-64 (v1.6)
//   qomn bench [rows] [cols]     AVX2 MM_TERN 3-way benchmark
//   qomn eval  <expr>            evaluate inline
//   qomn compile <file.crys> [out_dir]   oracle → .crystal (RFF PaO)
//   qomn serve <file.crys> [port]        HTTP API
// ═══════════════════════════════════════════════════════════════════════

mod lexer;
mod ast;
mod parser;
mod typeck;
mod units;           // v2.0
mod plan;            // v2.0
mod intent_parser;   // v2.0
mod vm;
mod repl;
mod server;
mod crystal_compiler;
mod cognitive_memory;   // Cognitive Memory
mod bytecode;
mod bytecode_vm;
mod hir;
mod backend_cpu;
mod runtime;
mod jit;
mod batch_oracle;
mod batch_plan;
mod simulation_engine;
mod aot_plan;           // v3.1: AOT pre-compiled plan dispatch
mod plan_v2;            // v2.4: SPEC.md plan_* syntax
mod llvm_backend;       // v3.0: LLVM IR 18 emission
mod wasm_backend;
mod benchmark_proofs;   // Commander-Level Benchmark Proofs       // v3.1: WAT/WASM emission

use lexer::Lexer;
use parser::Parser;
use typeck::TypeEnv;
use vm::{Vm, QomniConfig};
use runtime::QomnRuntime;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        None | Some("repl") => {
            repl::run_repl(None, None);
        }

        Some("run") => {
            let path = args.get(2).expect("Usage: qomn run <file.crys>");
            let src  = read_file(path);
            let prog = parse_src(&src);
            let mut vm = Vm::new(QomniConfig::default());
            match vm.run(&prog) {
                Ok(out) => { for line in out { println!("{}", line); } }
                Err(e)  => { eprintln!("Runtime error: {}", e); std::process::exit(1); }
            }
        }

        // qomn run-jit <file.crys>
        // Pre-compiles all oracle bodies to native x86-64 (Cranelift v1.6),
        // then executes via bytecode VM with JIT dispatch wired in.
        // After JIT_THRESHOLD=50 interpreter calls any hot oracle auto-compiles.
        Some("run-jit") => {
            let path = args.get(2).expect("Usage: qomn run-jit <file.crys>");
            let src    = read_file(path);
            let prog   = parse_src(&src);
            let module = bytecode::compile_to_bytecode(&prog);

            let mut engine = match jit::JitEngine::new() {
                Ok(e)  => e,
                Err(e) => {
                    eprintln!("JIT init failed: {}", e);
                    std::process::exit(1);
                }
            };

            // Pre-compile all oracle bodies eagerly
            let results = engine.compile_all(&module);
            let ok: usize = results.iter().filter(|(_, r)| r.is_ok()).count();
            eprintln!("[JIT] pre-compiled {}/{} oracle(s) to native x86-64",
                ok, results.len());

            let mut runtime = QomnRuntime::new(4);
            match bytecode_vm::run_bytecode_jit(&module, &mut runtime, engine) {
                Ok(out) => { for line in out { println!("{}", line); } }
                Err(e)  => { eprintln!("Runtime error: {}", e); std::process::exit(1); }
            }
        }

        Some("check") => {
            let path   = args.get(2).expect("Usage: qomn check <file.crys>");
            let src    = read_file(path);
            let prog   = parse_src(&src);
            let mut tc = TypeEnv::new();
            let errors = tc.check_program(&prog);
            if errors.is_empty() {
                println!("OK — no type errors");
            } else {
                for e in &errors { eprintln!("Type error: {}", e); }
                std::process::exit(1);
            }
        }

        Some("lex") => {
            let path = args.get(2).expect("Usage: qomn lex <file.crys>");
            let src  = read_file(path);
            let mut lexer = Lexer::new(&src);
            for tok in lexer.tokenize() {
                println!("{:3}:{:2}  {:?}", tok.span.line, tok.span.col, tok.token);
            }
        }

        Some("hir") => {
            // qomn hir <file.crys>  — dump HIR graph after optimizations
            let path = args.get(2).expect("Usage: qomn hir <file.crys>");
            let src  = read_file(path);
            let prog = parse_src(&src);
            let graph = hir::build_hir(&prog);
            println!("{}", hir::print_hir(&graph));
        }

        Some("ir") => {
            // qomn ir <file.crys>  — dump CRYS-ISA Bytecode IR (v1.4)
            let path = args.get(2).expect("Usage: qomn ir <file.crys>");
            let src  = read_file(path);
            let prog = parse_src(&src);
            let module = bytecode::compile_to_bytecode(&prog);
            println!("{}", bytecode::disassemble(&module));
        }

        Some("jit") => {
            // qomn jit <file.crys>  — compile all oracle bodies → native x86-64
            let path = args.get(2).expect("Usage: qomn jit <file.crys>");
            let src  = read_file(path);
            let prog = parse_src(&src);
            let module = bytecode::compile_to_bytecode(&prog);

            println!("QOMN JIT v1.6 — Cranelift oracle compilation");
            println!("  Source:  {}", path);
            println!("  Oracles: {}", module.oracles.len());
            println!();

            match jit::JitEngine::new() {
                Err(e) => {
                    eprintln!("  ✗ JIT init failed: {}", e);
                    std::process::exit(1);
                }
                Ok(mut engine) => {
                    let results = engine.compile_all(&module);
                    let mut ok = 0;
                    for (name, result) in &results {
                        match result {
                            Ok(())  => { println!("  ✓ {}", name); ok += 1; }
                            Err(e)  => { println!("  ✗ {} → {}", name, e); }
                        }
                    }
                    println!();
                    println!("{} oracle(s) JIT-compiled to native x86-64", ok);
                    println!("{}", engine.stats());

                    // Self-test: call each compiled oracle with zero args
                    println!();
                    println!("Self-test (params=0.0):");
                    for (name, result) in &results {
                        if result.is_ok() {
                            if let Some(co) = engine.cache.get(name) {
                                let zeros: Vec<f64> = vec![0.0; co.n_params];
                                let out = unsafe { co.call(&zeros) };
                                println!("  {}({}) → {:.6}", name,
                                    zeros.iter().map(|_| "0.0").collect::<Vec<_>>().join(", "),
                                    out);
                            }
                        }
                    }

                    // ── JIT vs Interpreter timing benchmark ───────────
                    println!();
                    println!("Timing benchmark (10 000 calls per oracle):");
                    println!("  {:<26}  {:>12}  {:>12}  {:>8}", "oracle", "interp μs/call", "JIT ns/call", "speedup");
                    println!("  {}", "-".repeat(65));

                    const BENCH_N: usize = 10_000;

                    for (name, r) in &results {
                        if r.is_err() { continue; }
                        let co = match engine.cache.get(name) { Some(c) => c, None => continue };
                        let args: Vec<f64> = vec![1.0; co.n_params];
                        let entry_ip = module.oracles.iter()
                            .find(|o| &o.name == name)
                            .map(|o| o.entry_ip)
                            .unwrap_or(0);

                        // Interpreter timing
                        let t0 = std::time::Instant::now();
                        for _ in 0..BENCH_N {
                            let _ = bytecode_vm::exec_oracle_sync(&module, entry_ip, &args);
                        }
                        let interp_us = t0.elapsed().as_micros() as f64 / BENCH_N as f64;

                        // JIT timing
                        let t1 = std::time::Instant::now();
                        for _ in 0..BENCH_N {
                            let _ = unsafe { co.call(&args) };
                        }
                        let jit_ns = t1.elapsed().as_nanos() as f64 / BENCH_N as f64;

                        let speedup = (interp_us * 1_000.0) / jit_ns.max(0.001);
                        println!("  {:<26}  {:>12.3}  {:>11.1}  {:>7.1}×",
                            name, interp_us, jit_ns, speedup);
                    }
                    println!();
                }
            }
        }

        Some("bench") => {
            // qomn bench [rows] [cols] [n_runs]
            // 3-way benchmark: scalar vs sign-blend v1.5.1 vs 2-bit+FMA v1.5.2
            let rows   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(896usize);
            let cols   = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4864usize);
            let n_runs = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100usize);

            println!("QOMN MM_TERN Benchmark v1.5.2 (3-way: scalar | sign-blend | 2bit+FMA)");
            println!("  Matrix: {}×{}  ({} trits = {} KB packed)",
                rows, cols, rows * cols, (rows * cols + 3) / 4 / 1024);
            println!("  Tile:   {}×{}  Prefetch distance: {} tiles",
                64usize, 256usize, 2usize);
            println!("  Runs:   {}", n_runs);
            println!();
            backend_cpu::benchmark_compare(rows, cols, n_runs);
        }

        // qomn batch <file.crys> [batch_size] [n_iters]
        // v1.8 dual-path benchmark: scalar JIT vs AVX2 batch (vs AVX-512 if available)
        Some("batch") => {
            let path       = args.get(2).expect("Usage: qomn batch <file.crys> [batch_size] [n_iters]");
            let batch_size = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64usize);
            let n_iters    = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000usize);

            let src    = read_file(path);
            let prog   = parse_src(&src);
            let module = bytecode::compile_to_bytecode(&prog);
            let mut runtime = QomnRuntime::new(4);

            let engine = batch_oracle::BatchOracleEngine::new();

            println!("QOMN Batch Oracle Engine v1.8 — Dual-Path Execution");
            println!("  Source:     {}", path);
            println!("  Batch size: {}", batch_size);
            println!("  Iterations: {}", n_iters);
            println!("  AVX-512:    {}", if engine.avx512_available { "✓ available" } else { "✗ not available (KVM masked)" });
            println!("  AVX2:       {}", if engine.avx2_available { "✓ available" } else { "✗ not available" });
            println!("  Active path: {}", engine.select_path(batch_size).name());
            println!();
            println!("  Note: AVX-512 is NOT used for single-call oracle speedup.");
            println!("        It vectorizes ACROSS inputs: 16 oracle calls in parallel.");
            println!("        Single-call → scalar JIT (3–4 ns). Batch≥8 → SIMD kernel.");
            println!();

            // Compile all oracles to JIT
            let mut jit_engine = match jit::JitEngine::new() {
                Ok(e)  => e,
                Err(e) => { eprintln!("[JIT] init failed: {}", e); std::process::exit(1); }
            };
            jit_engine.compile_all(&module);
            let fn_table = jit_engine.fn_table();

            println!("  {:<26}  {:>12}  {:>14}  {:>10}",
                "oracle", "scalar ns/call", "batch ns/result", "speedup");
            println!("  {}", "-".repeat(68));

            for oracle in &module.oracles {
                let name     = &oracle.name;
                let n_params = oracle.n_params;
                if let Some(&(fn_addr, _)) = fn_table.get(name.as_str()) {
                    let (batch_ns, scalar_ns, speedup) = unsafe {
                        batch_oracle::bench_batch_vs_scalar(
                            name, fn_addr, n_params, batch_size, n_iters
                        )
                    };
                    println!("  {:<26}  {:>12.1}  {:>14.2}  {:>9.1}×",
                        name, scalar_ns, batch_ns, speedup);
                }
            }
            println!();
            println!("  Batch path: {} ({} inputs/SIMD)",
                engine.select_path(batch_size).name(),
                engine.select_path(batch_size).width());
            let _ = runtime;
        }

        Some("eval") => {
            let src  = args.get(2).expect("Usage: qomn eval <expr>");
            let prog = parse_src(src);
            let mut vm = Vm::new(QomniConfig::default());
            match vm.run(&prog) {
                Ok(out) => { for line in out { println!("{}", line); } }
                Err(e)  => eprintln!("Error: {}", e),
            }
        }

        Some("compile") => {
            let path    = args.get(2).expect("Usage: qomn compile <file.crys> [out_dir]");
            let out_dir = args.get(3).map(|s| s.as_str()).unwrap_or(".");
            let src     = read_file(path);
            let prog    = parse_src(&src);

            println!("QOMN Compiler v1.4 — oracle → .crystal");
            println!("  Physics-as-Oracle: RFF multi-scale projection");
            println!("  Input:   {}", path);
            println!("  Out dir: {}", out_dir);
            println!();

            let results = crystal_compiler::compile_oracles(&prog, out_dir);
            if results.is_empty() {
                println!("No oracle declarations found in '{}'.", path);
                std::process::exit(0);
            }

            let mut ok = 0;
            for r in results {
                match r {
                    Ok(c) => {
                        let kb = c.file_size / 1024;
                        let v  = &c.validation;
                        println!(
                            "  ✓ oracle {} → {} ({}KB, sparsity={:.1}%, coverage={:.1}%, sign_agree={:.0}%)",
                            c.oracle_name, c.out_path, kb,
                            c.sparsity * 100.0,
                            v.coverage * 100.0,
                            v.sign_agree_rate * 100.0,
                        );
                        ok += 1;
                    }
                    Err(e) => eprintln!("  ✗ ERROR: {}", e),
                }
            }
            println!();
            println!("{} oracle(s) compiled to .crystal", ok);
        }

        // ── v2.0: Plan execution ──────────────────────────────────
        // Usage: qomn plan <file.crys> <plan_name> [key=value ...]
        // Example: qomn plan nfpa.crys plan_sistema_incendios area=1200 K=5.6 P_disponible=60
        Some("plan") => {
            let path      = args.get(2).expect("Usage: qomn plan <file.crys> <plan_name> [key=value ...]");
            let plan_name = args.get(3).expect("Specify plan name");
            let src  = read_file(path);
            let prog = parse_src(&src);

            // Collect key=value params
            let mut params = std::collections::HashMap::new();
            for arg in args.iter().skip(4) {
                if let Some((k, v)) = arg.split_once('=') {
                    if let Ok(f) = v.parse::<f64>() {
                        params.insert(k.to_string(), f);
                    }
                }
            }

            // Extract plan declarations
            let plans: Vec<ast::PlanDecl> = prog.decls.iter().filter_map(|d| {
                if let ast::Decl::Plan(p) = d { Some(p.clone()) } else { None }
            }).collect();

            if plans.is_empty() {
                eprintln!("No plan declarations found in '{}'", path);
                std::process::exit(1);
            }

            // Build JIT engine for fast oracle dispatch
            let jit_engine = match jit::JitEngine::new() {
                Ok(mut e) => {
                    let bc = bytecode::compile_to_bytecode(&prog);
                    e.compile_all(&bc);
                    Some(e)
                }
                Err(e) => { eprintln!("JIT init warning: {}", e); None }
            };

            // Execute plan
            let executor_base = plan::PlanExecutor::new(&plans);
            let result = if let Some(ref jit) = jit_engine {
                // Extract fn-address map from JitEngine (JitFnTable = Arc<HashMap>)
                let fn_map: plan::JitFnMap = jit.fn_table().as_ref().clone();
                executor_base.with_jit_map(fn_map).execute(plan_name, params)
            } else {
                executor_base.execute(plan_name, params)
            };

            match result {
                Ok(r) => {
                    r.display();
                    println!();
                    println!("{}", r.to_json());
                }
                Err(e) => {
                    eprintln!("Plan execution error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        // ── v2.0: Intent parsing ──────────────────────────────────
        // Usage: qomn intent <file.crys> "<natural language query>"
        // Uses MockBackend (no API key needed). Set QOMNI_LLM_URL to use real LLM.
        Some("intent") => {
            let path  = args.get(2).expect("Usage: qomn intent <file.crys> <query>");
            let query = args.get(3).expect("Provide query string in quotes");
            let src   = read_file(path);
            let prog  = parse_src(&src);

            // Collect available plan names
            let available_plans: Vec<String> = prog.decls.iter().filter_map(|d| {
                if let ast::Decl::Plan(p) = d { Some(p.name.clone()) } else { None }
            }).collect();

            // Use mock backend (real backend: plug in reqwest-based LlmBackend impl)
            let backend = Box::new(intent_parser::MockBackend);
            let parser  = intent_parser::IntentParser::new(backend);

            match parser.parse(query) {
                Ok(intent) => {
                    println!("Intent parsed:");
                    println!("  domain:      {}", intent.domain.as_str());
                    println!("  plan:        {:?}", intent.plan_name);
                    println!("  params:      {:?}", intent.params);
                    println!("  units:       {:?}", intent.units);
                    println!("  constraints: {:?}", intent.constraints);

                    if let Some(plan_name) = intent_parser::route_to_plan(&intent, &available_plans) {
                        println!("\nRouted to plan: {}", plan_name);
                        println!("Run: qomn plan {} {} {}",
                            path, plan_name,
                            intent.params.iter()
                                .map(|(k,v)| format!("{}={}", k, v))
                                .collect::<Vec<_>>().join(" ")
                        );
                    } else {
                        println!("\nNo matching plan found for domain '{}'", intent.domain.as_str());
                        println!("Available plans: {:?}", available_plans);
                    }
                }
                Err(e) => {
                    eprintln!("Intent parse error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Some("serve") => {
            // QOMN_NO_FMA=1 → force SSE2-only path for cross-arch hash portability
            if std::env::var("QOMN_NO_FMA").map(|v| v == "1").unwrap_or(false) {
                crate::server::set_no_fma_mode(true);
                eprintln!("  [QOMN_NO_FMA=1] FMA fusion disabled — VMULSD+VADDSD enforced");
            }
            let path = args.get(2).expect("Usage: qomn serve <file.crys> [port]");
            let port: u16 = args.get(3).and_then(|p| p.parse().ok()).unwrap_or(9000);
            let src  = read_file(path);
            let prog = parse_src(&src);
            let mut vm_inst = vm::Vm::new(vm::QomniConfig::default());
            let _ = vm_inst.run(&prog);
            println!("  QOMN Cognitive Engine — serving '{}'", path);
            // Pre-compile all oracles to JIT for fast plan dispatch
            let bc  = bytecode::compile_to_bytecode(&prog);
            let srv = match jit::JitEngine::new() {
                Ok(mut engine) => {
                    let _ = engine.compile_all(&bc);
                    let fn_map: plan::JitFnMap = engine.fn_table().as_ref().clone();
                    // ── AOT Plan Cache: pre-compile all plans for ~50ns execution ──
                    {
                        let plan_decls: Vec<crate::ast::PlanDecl> = prog.decls.iter()
                            .filter_map(|d| if let crate::ast::Decl::Plan(p) = d { Some(p.clone()) } else { None })
                            .collect();
                        let builtins = crate::plan::PlanExecutor::builtin_oracles();
                        let mut aot = crate::aot_plan::AotPlanCache::compile(&plan_decls, &builtins, &Some(fn_map.clone()));
                        aot.compile_plans_jit(&bc);
                        aot.build_turbo_table();
                        aot.compile_plans_register_abi(&bc);
                        crate::server::init_aot_cache(aot);
                        eprintln!("  AOT Plan Cache initialized (L1 + L2 + L3 Turbo + L4 Register ABI)");
                    }
                    // Keep JIT memory alive for the entire server lifetime.
                    // JitEngine owns the mmap region; dropping it frees the code → use-after-free.
                    let _jit_guard = Box::leak(Box::new(engine));
                    server::QomnServer::new(vm_inst, prog, port).with_jit_map(fn_map)
                }
                Err(_) => server::QomnServer::new(vm_inst, prog, port),
            };
            // Start self-healing watchdog background thread
            crate::server::start_watchdog();
            srv.run();
        }


        // ── plan_v2 commands (SPEC.md plan_* syntax) ────────────────────────

        // qomn check-plan <file.qomn>  — type-check plan_* syntax
        Some("check-plan") => {
            let path = args.get(2).expect("Usage: qomn check-plan <file.qomn>");
            let src  = read_file(path);
            match plan_v2::parse_plans(&src) {
                Err(errs) => {
                    for e in &errs { eprintln!("Parse error {}: {}", e.span, e.message); }
                    std::process::exit(1);
                }
                Ok(plans) => {
                    let mut any_err = false;
                    for plan in &plans {
                        let errs = plan_v2::typecheck_plan(plan);
                        if errs.is_empty() {
                            println!("OK  {}", plan.name);
                        } else {
                            for e in &errs { eprintln!("Type error {}: {}", e.span, e.message); }
                            any_err = true;
                        }
                    }
                    if any_err { std::process::exit(1); }
                    println!("--- {} plan(s) OK", plans.len());
                }
            }
        }

        // qomn fmt <file.qomn>  — pretty-print plan_* syntax
        Some("fmt") => {
            let path = args.get(2).expect("Usage: qomn fmt <file.qomn>");
            let src  = read_file(path);
            match plan_v2::parse_plans(&src) {
                Err(errs) => {
                    for e in &errs { eprintln!("Parse error {}: {}", e.span, e.message); }
                    std::process::exit(1);
                }
                Ok(plans) => {
                    for plan in &plans {
                        println!("{}", plan_v2::fmt_plan(plan));
                    }
                }
            }
        }

        // qomn run-plan <file.qomn> [name=value ...]  — execute plan_* plan
        Some("run-plan") => {
            let path = args.get(2).expect("Usage: qomn run-plan <file.qomn> [name=value ...]");
            let src  = read_file(path);
            // parse key=value args
            let mut call_args: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
            for arg in args.iter().skip(3) {
                if let Some(eq) = arg.find('=') {
                    let name = arg[..eq].to_string();
                    let val: f64 = arg[eq+1..].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid value in '{}'", arg);
                        std::process::exit(1);
                    });
                    call_args.insert(name, val);
                }
            }
            match plan_v2::parse_plans(&src) {
                Err(errs) => {
                    for e in &errs { eprintln!("Parse error {}: {}", e.span, e.message); }
                    std::process::exit(1);
                }
                Ok(plans) => {
                    if plans.is_empty() { eprintln!("No plans found in '{}'", path); std::process::exit(1); }
                    let plan = &plans[0];
                    let t0 = std::time::Instant::now();
                    match plan_v2::execute_plan(plan, &call_args) {
                        Err(e) => { eprintln!("Execution error: {}", e); std::process::exit(1); }
                        Ok(result) => {
                            let elapsed = t0.elapsed();
                            println!("QOMN run-plan: {}", plan.name);
                            println!();
                            for (k, v) in &result.outputs {
                                println!("  {:20} = {:.6}", k, v);
                            }
                            println!();
                            println!("  Latency: {:?}", elapsed);
                        }
                    }
                }
            }
        }

        // qomn sweep <file.qomn> [param=start:stop:step | param=[v1,v2] ...]
        Some("sweep") => {
            let path = args.get(2).expect("Usage: qomn sweep <file.qomn> [param=start:stop:step ...]");
            let src  = read_file(path);
            let mut axes: std::collections::HashMap<String, plan_v2::SweepAxis> = std::collections::HashMap::new();
            for arg in args.iter().skip(3) {
                if let Some((name, axis)) = plan_v2::parse_sweep_axis(arg) {
                    axes.insert(name, axis);
                } else {
                    eprintln!("Invalid sweep axis '{}' — expected: name=start:stop:step or name=[v1,v2,v3]", arg);
                    std::process::exit(1);
                }
            }
            match plan_v2::parse_plans(&src) {
                Err(errs) => {
                    for e in &errs { eprintln!("Parse error {}: {}", e.span, e.message); }
                    std::process::exit(1);
                }
                Ok(plans) => {
                    if plans.is_empty() { eprintln!("No plans found in '{}'", path); std::process::exit(1); }
                    let plan = &plans[0];
                    let results = plan_v2::sweep_plan(plan, &axes);
                    println!("QOMN sweep: {} — {} scenarios", plan.name, results.len());
                    println!();
                    for (i, r) in results.iter().enumerate() {
                        // print first output key as summary
                        if let Some((k, v)) = r.outputs.iter().next() {
                            println!("  [{:4}]  {} = {:.6}", i, k, v);
                        }
                    }
                    println!();
                    println!("  {} scenarios computed", results.len());
                }
            }
        }

        Some(cmd) => {
            eprintln!(
                "Unknown command: '{}'\nUsage: repl | run | run-jit | check | check-plan | fmt | run-plan | sweep | lex | hir | ir | jit | bench | eval | compile | serve | plan | intent",
                cmd
            );
            std::process::exit(1);
        }
    }
}

fn read_file(path: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| { eprintln!("Error reading '{}': {}", path, e); std::process::exit(1) })
}

fn parse_src(src: &str) -> ast::Program {
    let mut lexer  = Lexer::new(src);
    let tokens     = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse().unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        std::process::exit(1)
    })
}

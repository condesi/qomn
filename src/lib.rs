pub mod lexer;
pub mod ast;
pub mod parser;
pub mod typeck;
pub mod units;           // v2.0: physical unit algebra + range table
pub mod plan;            // v2.0: plan execution engine
pub mod intent_parser;   // v2.0: NL -> IntentAST interface
pub mod vm;
pub mod repl;
pub mod server;
pub mod crystal_compiler;
pub mod cognitive_memory;   // Cognitive Memory: persist execution experiences
pub mod bytecode;       // v1.4: CRYS-ISA bytecode IR + optimizer
pub mod aot_plan;       // v3.1: AOT pre-compiled plan dispatch (Level 1 + Level 2 JIT)
pub mod batch_plan;
pub mod simulation_engine; // v2.2: Continuous sim loop, SoA AVX2, physics layer, decision optimizer
pub mod plan_v2;
pub mod llvm_backend;   // v3.0: LLVM IR 18 emission backend
pub mod wasm_backend;
pub mod benchmark_proofs;   // Commander-Level Benchmark Proofs   // v3.1: WAT/WASM emission backend
        // v2.4: SPEC.md plan_* syntax — parser, formatter, typechecker, interpreter

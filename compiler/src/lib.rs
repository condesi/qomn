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

# Changelog

All notable changes to QOMN will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows semantic versioning adapted for a language runtime (major = breaking language or ABI change, minor = new feature / plan set, patch = bug fix / perf improvement).

## [Unreleased]

## [3.2] — 2026-04

### Added
- Full rename from CRYS-L to QOMN: language, ISA, bytecode magic (`b"QOMN"`), file extension `.qomn`, plan syntax references.
- Global integrity hash (SHA-256, 10K-scenario sweep) embedded in README as reproducibility anchor.
- Additional adversarial tests (12.8M total inputs across `tests/adversarial.rs`).
- `QOMN_SERVER_URL` environment variable for runtime endpoint override.
- `QOMN_NO_FMA=1` flag for cross-architecture bit-exact determinism.
- Related Repositories section in README linking to `qomn-paper` and archived `crysl-lang`.

### Changed
- Default `base_url` in `vm` and `repl` now uses `https://desarrollador.xyz` (or `QOMN_SERVER_URL`) instead of a direct hostname.
- Measured throughput upgraded to the 449M–540M scenarios/sec range (live measurements, median ~510M, SIMD utilization 60-73%).
- 57 plans across 10 engineering domains (up from 56 in v2.3).
- Brand unified: `.crysl` → `.qomn`, `LOAD_CRYS` → `LOAD_QOMN`, CRYS-ISA → QOMN-ISA.

### Fixed
- Removed internal hostname exposure from `compiler/src/vm.rs` and `compiler/src/repl.rs` (security).
- Canonicalized commit history via `.mailmap`.
- Cleaned `.git/config` of hardcoded tokens.

## [3.1] — 2026-04
### Added
- AOT pre-compiled plan dispatch (`aot_plan.rs`).
- LLVM IR 18 emission backend (`llvm_backend.rs`).
- WebAssembly text (WAT) backend (`wasm_backend.rs`).

## [3.0] — 2026-03
### Added
- Cranelift JIT with frame-pointer elimination and constant-exponent power inlining.
- Physical unit algebra with NFPA/IEC range validation (`units.rs`).

## [2.3] — 2026-03
### Added
- L4 Register ABI for hot-path plan execution.
- OracleCache with FNV-1a hash (measured ~12 ns cache probe).
- Continuous simulation engine with SoA AVX2 sweep kernel.
- Multi-objective Pareto front exploration (170 solutions per call, ~1.8 ms).

## [2.2] — 2026-02
### Added
- HTTP server on port 8090 (production).
- JSON request/response surface for `/api/plan/execute`, `/api/plans`, `/api/health`.
- Intent parser with optional LLM front-end (not used for compute).

## [2.0] — 2026-02
### Added
- Typed unit system (flow, pressure, power, current, voltage, area, ratio).
- `plan` construct with typed parameters and named step results.

## [1.6] — 2026-01
### Added
- Initial Cranelift JIT integration; oracle bodies compiled to native x86-64.

## [1.4] — 2025-12
### Added
- Bytecode IR with optimizer (dead-code elimination, inlining of short oracles).
- Interpreted bytecode VM as pre-JIT fallback.

## [1.0] — 2025-10
### Initial release
- Parser, AST, type checker for the engineering DSL.
- First plans: fire pump sizing, sprinkler hydraulics, pipe losses (Hazen-Williams, Manning).

[Unreleased]: https://github.com/condesi/qomn/compare/v3.2...HEAD
[3.2]: https://github.com/condesi/qomn/releases/tag/v3.2

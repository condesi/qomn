# Contributing to QOMN

Thank you for considering a contribution. QOMN is an open Rust implementation of a deterministic compute engine for critical systems; contributions are welcome in several categories.

## What kinds of contributions are welcome?

### 1. New engineering plans

If you are a licensed or credentialed professional in a domain covered by QOMN (fire, electrical, structural, hydraulic, HVAC, financial, medical, etc.), plan contributions are the highest-impact form of contribution.

A good plan contribution:
- Implements a formula from a cited governing standard (NFPA, IEC, AISC, ASCE, FAO-56, etc.).
- Places the citation in an inline comment in the `.qomn` source: `// NFPA 20:2022 §4.26`.
- Adds at least one regression test in `tests/` with a known-good input/output pair.
- Includes an adversarial test case if the plan admits a bounded parameter range.

Open an issue first if your plan replaces an existing one, covers a new domain, or touches shared oracles.

### 2. Runtime fixes

- Bugs: correctness of generated code, JIT compilation, dimensional type checking, HTTP handler behavior.
- Performance: measurable improvements to the AVX2 sweep kernel, JIT codegen, or cache paths.
- Portability: ARM64 support, macOS Apple Silicon validation, `no_std` reduction.

### 3. Tests and adversarial corpus

Reproduction reports, additional adversarial inputs, stress tests, and cross-platform verification are welcome. Please document the hardware and OS for any measurement.

### 4. Documentation

Corrections, clarifications, and new examples in `docs/`, `plans/`, or top-level `.qomn` files. Keep the citation-in-source discipline for any domain-specific content.

## What is out of scope for this repository?

- Features of the cognitive orchestration layer (Qomni Cognitive OS). That system is maintained privately and is not the subject of this repository.
- Wrappers or SDKs for specific host languages (Python, JS, Go). These may become separate companion repositories; issues proposing them are welcome, PRs changing this repo to host them are not.
- Paper corrections. Open those against [`condesi/qomn-paper`](https://github.com/condesi/qomn-paper).

## How to open a good pull request

1. Fork and create a branch from `main`.
2. Ensure the build compiles cleanly: `cargo build --release`.
3. Ensure tests pass: `cargo test --release`.
4. For plan contributions, verify: `cargo test --release --test all_56_plans`.
5. Keep commits focused (one idea per commit).
6. Sign your commits with a real name and verified email.
7. Submit a PR with a clear description of what changed and why.

## Coding conventions

- Rust edition 2021.
- Prefer `#[inline]` on trivial accessors in hot paths.
- No `unsafe` blocks outside `backend_cpu.rs`, `hdc_turbo.rs`, and a short list of other performance-critical modules.
- Public API surface is minimized; prefer adding internal helpers over new public types.
- Follow the existing module-per-concern structure.

## Plan authoring conventions

- Plan files use the `.qomn` extension (not `.crys`).
- Unit dimensions are declared in parameter types (`Q_gpm: flow`, `P_psi: pressure`).
- Magic numbers are named constants with citations.
- Oracles are pure functions; side effects belong in the runtime, never in a plan.

## Security

If you discover a security issue (command injection, path traversal, panic on adversarial input, crash under untrusted parameters), please email `percy.rojas@condesi.pe` privately rather than opening a public issue. We will credit you in the release notes if the fix ships.

## Code of conduct

See `CODE_OF_CONDUCT.md`. In short: focus on technical content, be respectful, keep discussions on-topic.

## Questions

For anything this document does not cover:
- **Percy Rojas Masgo** — `percy.rojas@condesi.pe`
- Open a Discussion in the repository.

Thank you for helping improve QOMN.

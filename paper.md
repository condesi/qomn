---
title: 'CRYS-L: A Deterministic Execution Engine for Safety-Critical Engineering Computations'
tags:
  - Rust
  - deterministic computing
  - IEEE-754
  - JIT compilation
  - safety-critical systems
  - engineering calculations
  - NFPA
authors:
  - name: Percy Rojas Masgo
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Condesi Perú / Qomni AI Lab, Lima, Peru
    index: 1
date: 17 April 2026
bibliography: paper.bib
---

# Summary

Engineering certification standards—including NFPA 20 (fire pump sizing),
IEC 60364 (electrical installations), and ASCE 7 (structural loads)—require
that computational results be **mathematically reproducible**: given identical
inputs, every server must produce bit-identical outputs under any load.

**CRYS-L** is an execution engine that enforces this property through three
mechanisms: (1) the *branchless oracle pattern*, which eliminates undefined
behavior and enables SIMD vectorization; (2) Cranelift JIT compilation
targeting AVX2; and (3) strict IEEE-754 enforcement at the compiler level.
The engine exposes 56 physics plans across 6 engineering domains via a
public REST API, achieving 1.53 billion scenarios/second on commodity
hardware ($80/month, AMD EPYC 12-core). All results are publicly
verifiable without credentials at <https://desarrollador.xyz>.

# Statement of Need

Three classes of tools widely used for engineering calculations fail the
certified reproducibility requirement:

1. **Large Language Models**: stochastic sampling produces different outputs
   on repeated queries, even at temperature 0 across different hardware.
2. **Python/NumPy**: floating-point results vary across NumPy versions,
   BLAS backends, and compiler optimization levels [@goldberg1991].
3. **Unsafe C++**: `if (flow < 1.0) return NAN` introduces a branch that
   prevents SIMD vectorization and propagates NaN silently on invalid input.

CRYS-L fills this gap by providing a domain-specific language for physics
formulas that compiles to IEEE-754-exact machine code with provably zero
variance across runs, servers, and loads.

# The Branchless Oracle Pattern

The central design innovation is expressing conditional validation as a
floating-point predicate, eliminating branches entirely:

```
oracle nfpa20_pump_hp(
    flow: float, head: float, eff: float) -> float:
  let valid = (flow >= 1.0) * (flow <= 50000.0) * (eff >= 0.10)
  ((flow * 0.06309 * head * 0.70307) / (eff * 76.04 + 0.0001)) * valid
```

The predicate `valid` evaluates to `1.0` when all constraints hold and
`0.0` otherwise. This eliminates undefined behavior on invalid inputs and
allows AVX2 to evaluate 4 scenarios in a single `VMULPD` instruction.

# Architecture

CRYS-L is implemented in 2,843 lines of Rust. Physics expressions compile
to Cranelift IR [@cranelift], which generates AVX2 machine code with
FMA contraction disabled to preserve IEEE-754 reproducibility. Compiled
plans are cached keyed by expression hash.

The `/verify` endpoint provides a live determinism proof: it executes the
voltage-drop oracle N times and returns a FNV-1a hash of the IEEE-754 bit
pattern, which is identical across all runs:

```bash
curl "https://desarrollador.xyz/verify?runs=20"
# -> {"variance":0.000000000000, "all_identical":true, "hash_match":true}
```

# Performance

| System | Throughput | Determinism |
|--------|-----------|-------------|
| CRYS-L v3.2 (AVX2) | 1.53 B scenarios/s | IEEE-754 exact |
| C++ GCC -O3 | ~5 M/s | UB on NaN inputs |
| Python/NumPy | ~0.2 M/s | Float drift across versions |
| GPT-4 Turbo | 0.08 answers/s | Stochastic |

All benchmarks are publicly verifiable:

```bash
curl https://desarrollador.xyz/simulation/jitter_bench
curl https://desarrollador.xyz/simulation/adversarial
curl https://desarrollador.xyz/benchmark/vs_llm
```

# Plan Coverage

Version 3.2 includes 56 physics plans: fire protection (NFPA 20, 8 plans),
electrical (NEC/IEC 60364, 12 plans), structural (ASCE 7, 9 plans),
HVAC (ASHRAE, 11 plans), finance (8 plans), and medical (IEC 60601, 8 plans).

# References

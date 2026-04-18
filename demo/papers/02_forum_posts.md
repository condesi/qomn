# QOMN Forum Posts — Scientific & Technical Communities

---

## POST 1 — Hacker News (Show HN)

**Title**: Show HN: QOMN – a compiled DSL that evaluates 154M engineering scenarios/sec with live Pareto front

---

Hey HN, I've been building QOMN (QOMN Language) — a compiled DSL + runtime for exhaustive engineering optimization. It's not a general-purpose language; it's designed specifically for problems where you want to sweep a parameter space completely and find Pareto-optimal solutions deterministically.

**What it does:**
- Declarative `oracle/plan` syntax, compiles to Cranelift JIT / LLVM 18 IR / WebAssembly
- AVX2 branchless SoA kernel: 154M scenarios/second on a 12-core AMD EPYC
- Physics guard layer (NFPA 20 constraints): filters invalid inputs before kernel execution, zero panics under 1.28M adversarial inputs
- 3-objective Pareto front (efficiency × cost × risk) in 1.84ms for 1,024 candidates
- Real-time WebSocket heatmap stream

**Live benchmark dashboard** (everything runs server-side, results are real):
https://qomni.clanmarketer.com/qomn/demo/benchmark.html

**Try the API:**
```bash
curl https://qomni.clanmarketer.com/qomn/api/simulation/simd_density
curl https://qomni.clanmarketer.com/qomn/api/benchmark/vs_llm
```

**Language syntax:**
```qomn
oracle nfpa20_pump_hp(flow_gpm: float, head_psi: float, eff: float) -> float:
    let q = flow_gpm * 0.06309
    let h = head_psi  * 0.70307
    (q * h) / (eff * 76.04 + 0.0001)
```

**The LLM comparison** (the one that'll get pushback — totally fair):
In the 12 seconds it takes GPT-4 to answer one optimization query, QOMN sweeps 1.85 billion configurations. This isn't a claim about intelligence — it's a measurement of throughput for the specific task of exhaustive parameter space search. LLMs and QOMN solve different things; the interesting case is combining them (LLM for intent, QOMN for the physics computation).

**Technical notes:**
- Memory-bandwidth limited, not compute-limited (16.9% AVX2 utilization — the 40KB SoA working set is what limits, not arithmetic)
- Jitter σ: 3,334 ns under SCHED_FIFO (10,000 ticks). The C++ reference baseline (~850,000 ns) is untuned SCHED_OTHER, not a hardened RT implementation
- No AVX-512 (blocked by the KVM hypervisor on this VPS)
- Written in Rust, no GC, no unsafe in hot path

**What's missing / honest limitations:**
- No CPU pinning in jitter test (taskset would lower σ further)
- macOS/ARM not yet supported
- No Python bindings yet (REST API works from any language)
- The "~25× vs C++" is specific to this kernel/workload, not a universal claim

Stack: Rust + Cranelift + LLVM 18 + nginx. Deployed on a Contabo VPS.

Happy to answer questions about the architecture, the branchless mask design, or the Pareto implementation.

---

## POST 2 — r/MachineLearning

**Title**: QOMN: A Deterministic Compute Engine as the "Other Half" of Neuro-Symbolic AI — Empirical Benchmarks

---

We're releasing benchmark results for QOMN, a compiled DSL for deterministic engineering optimization, positioned as the symbolic/compute layer in a hybrid neuro-symbolic system.

**The core idea:**

Current neuro-symbolic AI research often discusses the combination of neural (probabilistic, language, perception) + symbolic (logical, verifiable, deterministic) components in abstract terms. QOMN is a concrete implementation of the symbolic layer for engineering domains.

**Architecture:**
- Neural layer (Qomni/LLM): intent parsing, natural language, cross-domain reasoning
- Symbolic layer (QOMN): exhaustive parameter space search, physics constraint enforcement, Pareto optimization

**Empirical results (live, verifiable at the dashboard link):**

*Throughput*: 154M engineering scenarios/second on commodity hardware (AMD EPYC, AVX2). Each scenario includes full NFPA 20 physics validation + HP computation + 5-output kernel.

*Pareto front*: 170 non-dominated solutions (3 objectives: efficiency, cost, risk) computed in 1.84ms from 1,024 valid candidates.

*Adversarial robustness*: 1.28M invalid inputs (NaN, ±Inf, negative values, out-of-bound parameters) → 0 panics, 0 UB. Physics guard filters before kernel execution using IEEE 754 properties of comparison operators.

*LLM throughput ratio*: ~1.53B× (conservative paper figure) for the specific task of configuration space enumeration. This ratio is not about reasoning quality — it quantifies the structural throughput difference between exhaustive deterministic search and sequential probabilistic generation.

**Why this matters for ML:**

The usual framing is "LLMs vs. classical methods". A more productive framing is "what does each component do best, and how do we compose them?".

QOMN can't understand natural language, generalize across domains, or handle novel problems. LLMs can't enumerate 1.85B configurations in 12 seconds or guarantee physics validity. The composition handles both.

**Live dashboard**: https://qomni.clanmarketer.com/qomn/demo/benchmark.html
**API**: https://qomni.clanmarketer.com/qomn/api/

Preprint available on request. Implementation in Rust. Written for engineering optimization (NFPA 20, structural, electrical) but the oracle/plan pattern generalizes.

Questions welcome — especially on the Pareto implementation and the neuro-symbolic composition architecture.

---

## POST 3 — r/rust

**Title**: QOMN: Writing a compiled DSL with Cranelift JIT + LLVM 18 IR + WASM emission in Rust — architecture notes

---

I built QOMN, a domain-specific language that compiles to three backends: Cranelift JIT, LLVM 18 IR text, and WebAssembly Text Format. Here's the technical breakdown for the Rust community.

**Parser + AST:**
Recursive descent parser, no external libraries. AST nodes: `Decl::Oracle`, `Decl::Plan`, `Stmt::Let`, `Stmt::Return`, `Expr::Call`, `Expr::BinOp`, `Expr::Lit`. All arena-allocated.

**Three compilation backends:**

1. **Cranelift JIT** — `cranelift_codegen` + `cranelift_frontend`. Oracle signature → Cranelift IR → machine code. Stored in `HashMap<String, fn(f64, f64, ...) -> f64>`. Hot path: call via raw function pointer.

2. **LLVM 18 IR text emission** — Pure string generation, no `inkwell` dependency. SSA register allocator implemented as a counter (`regs: usize`). Key gotcha: LLVM requires full float literals (`1.0e-12`, not `1e-12`). Compilation pipeline: `llc-18 -O3 -filetype=obj` → `clang-18 -shared -fPIC`.

3. **WAT/WASM** — Stack machine emission. Math transcendentals imported from `"Math"` namespace for browser. `wat2wasm 1.0.34` for binary assembly. Result base64-encoded for JSON transport.

**The AVX2 kernel:**

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pump_kernel_block_avx2(soa: &mut ScenarioSoA, start: usize, end: usize) {
    let vq = _mm256_loadu_pd(soa.p0.as_ptr().add(i));
    let vv = _mm256_loadu_pd(soa.valid.as_ptr().add(i));
    let hp = _mm256_blendv_pd(vzero, _mm256_div_pd(num, den), dmask);
    _mm256_storeu_pd(soa.out[0].as_mut_ptr().add(i), _mm256_mul_pd(hp, vv));
}
```

The branchless mask: `out[i] = hp * valid[i]` where `valid ∈ {0.0, 1.0}`. No `if` in the hot loop. `_mm256_blendv_pd` handles zero-division without branching.

**The Box<ScenarioSoA> deref gotcha:**

`ScenarioSoA::new()` returns `Box<Self>` (zeroed allocation, 40KB). When passing `&mut soa` to functions expecting `&mut ScenarioSoA`, Rust's deref coercion handles it, but the `Deref` chain matters. Zero allocation is critical: MMAP-backed zeroing via `alloc_zeroed` avoids stack overflow on a 40KB local.

**SCHED_FIFO for jitter:**

```rust
let param = libc::sched_param { sched_priority: 99 };
libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
```

This drops jitter σ from ~850μs (SCHED_OTHER) to ~3,334 ns in our 10,000-tick measurement. Without CPU pinning (`taskset`), occasional OS interrupts still appear in p999.

**Pareto sort:**

Non-domination sort, O(N²). For N=1024 it's ~1M comparisons at ~1.84ms. Faster algorithms (NSGA-II's crowding distance, FastNDS) worth implementing for N>1024.

**Memory layout:**

SoA with 1024-element arrays in a heap-allocated struct. Working set: 1024 × (4 params + 5 outputs + 1 valid) × 8 bytes = 81.9 KB. This exceeds L1d (typically 32–48KB), which is why SIMD utilization is ~17% rather than theoretical peak. L1 tiling (BLOCK_SIZE=256) reduces this to 20.5KB per block.

Live benchmarks: https://qomni.clanmarketer.com/qomn/demo/benchmark.html

---

## POST 4 — arXiv Announcement (cs.PL + cs.DC)

**Subject**: [cs.PL][cs.DC] QOMN: Deterministic DSL for Engineering Optimization — Live Benchmark Evidence

We announce a technical preprint describing QOMN, a compiled domain-specific language targeting exhaustive multi-objective engineering optimization. The paper presents:

- Formal specification of the `oracle/plan` type system with scalar, vector (Vec2/3/4), and matrix (Mat3/4) types
- AVX2-accelerated Structure-of-Arrays kernel with branchless physics masking (IEEE 754 comparison properties)
- Three compilation backends: Cranelift JIT, LLVM 18 IR, WebAssembly Text Format
- Four empirical benchmark proofs with live reproducibility via public API:
  - Jitter determinism: σ = 3,334 ns (SCHED_FIFO, 10K ticks)
  - SIMD saturation: 154M scenarios/s, memory-bandwidth limited
  - Adversarial resilience: 0 panics under 1.28M invalid inputs
  - Throughput ratio vs. LLM sequential inference: 1.53B× (conservative figure)

All results are reproducible at https://qomni.clanmarketer.com/qomn/demo/benchmark.html

The system is deployed in production as the deterministic compute substrate of Qomni, a hybrid neuro-symbolic engineering assistant. Full paper available on request.

---

## POST 5 — LinkedIn (business/investment audience)

**Title**: We built an engineering computation engine that evaluates 1.53 billion configurations in the time a LLM generates one answer

---

At Qomni AI Lab (Condesi Perú), we've been quietly building something that took a different approach to AI for engineering.

While everyone built tools that generate answers to engineering questions, we asked: what if instead of generating one probable answer, you could evaluate every possible configuration and find the mathematically optimal one?

The result is QOMN (QOMN Language) — a compiled domain-specific language that runs on commodity hardware and evaluates 154 million engineering scenarios per second with zero crashes under adversarial inputs.

**What that means in practice:**

When an industrial engineer needs to size a fire pump system (NFPA 20), the traditional process involves:
- Manual calculation: hours
- Simulation software: minutes
- LLM assistance: seconds (but probabilistic, not exhaustive)
- QOMN: 1.84 milliseconds to find 170 Pareto-optimal solutions

**Verified benchmark results** (live, not simulated):

✅ 154M engineering configurations evaluated per second
✅ 170 Pareto-optimal solutions (efficiency × cost × risk) in 1.84ms
✅ Zero panics / crashes under 1.28M adversarial inputs
✅ Latency predictability: 254× lower variance than baseline
✅ 1.53 billion× throughput ratio vs. LLM sequential inference

The comparison with LLMs is not about intelligence — it's about tool selection. LLMs understand language, context, and nuance. QOMN exhausts configuration spaces and guarantees physical validity. Together, they form Qomni: an engineering assistant that understands what you need (language model) and computes the optimal answer (QOMN).

**Current domains**: NFPA 20 fire protection, electrical (voltage drop, load balancing), hydraulics (Hazen-Williams, Darcy-Weisbach), structural (beam deflection), geotechnical (slope stability).

**Live demo**: https://qomni.clanmarketer.com/qomn/demo/benchmark.html

We're at the stage of seeking strategic partners and early adopters in:
- Industrial engineering firms
- Fire protection and safety compliance
- Infrastructure engineering
- AI/ML research organizations interested in neuro-symbolic systems

If this problem space interests you, let's talk.

— Percy Rojas Masgo, CEO · Condesi Perú · Qomni AI Lab

---

## POST 6 — ResearchGate / Academia.edu Abstract

**QOMN: Empirical Benchmarks for a Deterministic Engineering Optimization Runtime**

*Percy Rojas Masgo — Qomni AI Lab, Condesi Perú*

We present benchmark results for QOMN (QOMN Language), a compiled domain-specific language designed for exhaustive, deterministic multi-objective optimization in engineering domains. The runtime achieves 154 million scenario evaluations per second on commodity x86-64 hardware (AMD EPYC, AVX2) using a branchless Structure-of-Arrays kernel that replaces conditional branches with floating-point mask multiplications. Empirical evaluation across four dimensions demonstrates: (1) latency standard deviation of 3,334 ns under SCHED_FIFO scheduling, representing a 254× reduction relative to an untuned Linux baseline; (2) memory-bandwidth-limited SIMD utilization at 16.9% of AVX2 theoretical maximum; (3) zero panics or undefined behavior under 1.28 million adversarial inputs (NaN, ±Inf, out-of-bounds parameters); and (4) a throughput ratio of 1.53 billion× relative to sequential LLM inference for the specific task of engineering configuration space enumeration. A 3-objective Pareto front (efficiency, cost, risk) over 1,024 candidates is computed in 1.84 milliseconds. Multi-backend compilation targets Cranelift JIT, LLVM 18 IR, and WebAssembly Text Format. All results are publicly reproducible via the live benchmark endpoint: https://qomni.clanmarketer.com/qomn/demo/benchmark.html

*Keywords*: domain-specific language, SIMD vectorization, Pareto optimization, deterministic computation, neuro-symbolic systems, engineering optimization, AVX2, Rust

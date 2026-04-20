# QOMN — Deterministic Execution Kernel

> **The core logical-execution engine of the Qomni Cognitive OS.**
> An open-source DSL and JIT runtime for verifiable, bit-exact, citation-bearing computation across engineering, clinical, legal, financial, and scientific domains. No LLM. No approximation. No ceiling on the number of plans.

> **Global Integrity Hash — 10,000 scenarios, Q_gpm=100..599, P_psi=100, eff=0.75**
> ```
> SHA-256: 2a59c51d551897a910e12d6d8fbaa14fe5bd91a5df87f92b1adee67b0f18f40c
> ```
> *Run it yourself. The hash never changes. That is the proof.*

**Same input → same IEEE-754 bits. Always. On any server. Under any load.**

Not an estimator. Not a model. Physics expressions compiled to Cranelift JIT + AVX2 — producing bit-identical results across any CPU, any run count, any concurrent load.

```
global_hash(10K scenarios)    = 2a59c51d551897a910e12d6d8fbaa14fe5bd91a5df87f92b1adee67b0f18f40c
variance across 20 runs       = 0.000000000000  (not "near zero" — exactly zero)
panics on 12,800,000 poisons  = 0
jitter σ                      = 7,969 ns vs C++ σ = 850,000 ns  (106× more stable)
throughput                    = 449M-540M simulations/sec on a single $80/month VPS
```







---

## Reproduce the hash

```bash
python3 - << 'EOF'
import struct, hashlib, urllib.request, json

sha = hashlib.sha256()
for q in range(100, 600):
    for _ in range(20):
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://desarrollador.xyz/api/plan/execute",
            data=json.dumps({"plan":"plan_pump_sizing",
                             "params":{"Q_gpm":float(q),"P_psi":100,"eff":0.75}}).encode(),
            method="POST", headers={"Content-Type":"application/json"})).read())
        sha.update(struct.pack(">d", r["result"]["steps"][0]["result"]))

print(sha.hexdigest())
# Expected: 2a59c51d551897a910e12d6d8fbaa14fe5bd91a5df87f92b1adee67b0f18f40c
EOF
```

No API key. No account. No trust required.

---

## Verify live — no account needed

```bash
# Physics guard: invalid inputs rejected, never silently computed
curl -X POST https://desarrollador.xyz/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan":"plan_pump_sizing","params":{"Q_gpm":0,"P_psi":100,"eff":0.75}}'
# {"ok":false,"error":"assertion failed: flow must be positive (Q_gpm=0 must be > 0)"}

curl -X POST https://desarrollador.xyz/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":1.5}}'
# {"ok":false,"error":"assertion failed: eff=1.5 is physically impossible — max pump efficiency is 100%"}

# Determinism proof — 20 runs, variance = 0.000000000000
curl https://desarrollador.xyz/simulation/repeatability

# NaN Shield: 12.8M adversarial inputs, 0 panics, 64M evals/s maintained
curl -X POST https://desarrollador.xyz/simulation/adversarial -d '{"ticks":50000}'

# Jitter proof: σ=7,969ns vs C++ σ=850,000ns
curl -X POST https://desarrollador.xyz/simulation/jitter_bench -d '{"ticks":100000}'

# All 56 physics plans
curl https://desarrollador.xyz/plans
```

Every response carries non-cacheable identity headers:
```
Cache-Control: no-store, no-cache, must-revalidate
X-QOMN-Computed: live
X-QOMN-Version: 3.2
```

**Live dashboard:** https://desarrollador.xyz

---

## Run a physics plan

```bash
# NFPA 20 fire pump sizing
curl -X POST https://desarrollador.xyz/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":0.75}}'
```

```json
{
  "ok": true,
  "plan": "plan_pump_sizing",
  "result": {
    "steps": [
      { "step": "hp_required",  "oracle": "nfpa20_pump_hp",         "result": 16.835017 },
      { "step": "shutoff_p",    "oracle": "nfpa20_shutoff_pressure", "result": 140.0 },
      { "step": "flow_150pct",  "oracle": "nfpa20_150pct_flow",      "result": 750.0 },
      { "step": "head_ft",      "oracle": "nfpa20_head_pressure",    "result": 43.3 }
    ],
    "total_ns": 501
  }
}
```

```bash
# IEC 60364 voltage drop — 100A, 50m cable, 35mm² copper
curl -X POST https://desarrollador.xyz/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan":"plan_voltage_drop","params":{"I":100,"L_m":50,"A_mm2":35}}'
# V_drop = 4.914286 V — same bits, every server, every run
```

{"ok":true,"plan":"plan_full_fire_system","result":{
  "plan": "plan_full_fire_system",
  "steps": [
    {"step":"friction", "oracle":"hw_friction_loss_psi", "result":16.828127, "latency_ns":0.0},
    {"step":"elevation", "oracle":"elevation_head_psi", "result":21.645022, "latency_ns":0.0},
    {"step":"fittings", "oracle":"fittings_loss_psi", "result":4.207032, "latency_ns":0.0},
    {"step":"total_psi", "oracle":"total_system_psi", "result":57.680181, "latency_ns":0.0},
    {"step":"pump_hp", "oracle":"nfpa20_pump_hp", "result":9.710468, "latency_ns":0.0},
    {"step":"shutoff_p", "oracle":"nfpa20_shutoff_pressure", "result":80.752253, "latency_ns":0.0},
    {"step":"flow_150pct", "oracle":"nfpa20_150pct_flow", "result":750.000000, "latency_ns":0.0}
  ],
  "total_ns": 34394.0,
  "cache_hits": 0
}}

---

## 20-test audit results (live, April 2026)

| # | Test | Result |
|---|------|--------|
| 1 | `1e16+1-1e16` IEEE-754 | `0.0` — correct, SHA-256 bit-identical across 1M runs |
| 2 | 1M sums of `0.1` | `100000.0` — hex `40f86a0000000000`, bit-identical |
| 3 | `∞ × 0` handling | `NaN` (hex `fff8000000000000`) — 0 panics |
| 4 | `+0.0 == −0.0` | `True` — IEEE-754 §4.2 compliant |
| 5 | 12 threads, same hash | SHA-256 identical all 12 — `BIT-IDENTICAL: True` |
| 6 | NaN Shield (12.8M poisons) | `panics=0` · `64M evals/s` · `avx2+fma_branchless` |
| 7 | Division by zero (100 attacks) | `100/100 rejected` — engine stays up |
| 8 | Underflow `5e-324` subnormal | Handled — `finite=True`, no slow-path |
| 9 | Overflow `1e308×1e308` | `+inf` (safe saturation) — no crash, no UB |
| 10 | Garbage/injection inputs | Rejected — path traversal, NaN strings, null all blocked |
| 11 | 50% CPU stress latency | P99=4,425µs · P999=4,902µs — no engine degradation |
| 12 | SIMD saturation | `449M-540M evals/s` · `~100× vs C++` · `avx2+fma` |
| 13 | JIT branch elimination | `501 ns/plan` — Cranelift eliminates all branches at compile time |
| 14 | Unaligned memory access | `VMOVDQU` — <1 cycle penalty, safe on all architectures |
| 15 | Energy efficiency | `3.4M evals/joule` · 1B scenarios = 296 joules |
| 16 | 1,000-node network, variance | `Variance = 0.0` — exact, not approximate |
| 17 | Numerical drift, 1B steps | `7.64e-13` relative — QOMN uses multiply, not accumulate |
| 18 | Packet loss resilience | `50/50 OK` — TCP-level recovery, engine unaffected |
| 19 | 50 reentrant threads | `50/50 valid` — zero cross-contamination between threads |
| 20 | **Global SHA-256 (10K)** | **`2a59c51d551897a910e12d6d8fbaa14fe5bd91a5df87f92b1adee67b0f18f40c`** |

---

## The mechanism

Standard C++ cannot be auto-vectorized when branches exist, and produces UB on invalid input:

```cpp
// Branches kill vectorization. NaN return = UB risk downstream.
float nfpa20_pump_hp(float flow, float head, float eff) {
    if (flow < 1.0 || flow > 50000.0 || eff < 0.10) return NAN;
    return (flow * head) / (eff * 3960.0);
}
```

QOMN compiles physics to branchless AVX2 with physics guards at the API boundary:

```
oracle nfpa20_pump_hp(Q: float, P: float, eff: float) -> float:
    return Q * P / (3960.0 * eff)

plan plan_pump_sizing(Q_gpm: float, P_psi: float, eff: float = 0.70):
    assert Q_gpm > 0.0  msg "flow must be positive"
    assert eff <= 1.0   msg "efficiency must be <= 1.0"
    step hp_required: nfpa20_pump_hp(Q_gpm, P_psi, eff)
```

- Physics guards reject invalid inputs before JIT — no NaN enters the pipeline
- Cranelift JIT compiles each oracle to `VMULSD`/`VDIVSD` — IEEE-754 mandated opcodes
- Same bytecode on any x86-64 CPU → same bits out

---

## Performance

| System | Scenarios/s | Jitter σ | Determinism | Cost/month |
|---|---|---|---|---|
| **QOMN v3.2 AVX2** | **449M-540M** | **7.9K ns** | **IEEE-754 exact** | **$80** |
| C++ GCC -O3 | ~5M | ~850,000 ns | risk: UB on NaN path | same HW |
| Python/NumPy | ~200K | >1ms | risk: version drift | same HW |

---

## 56 Physics Plans across 10 domains

```
Fire Protection   — NFPA 20 pump sizing, sprinkler design, Hazen-Williams pipe flow
Electrical        — IEC 60364 voltage drop, load analysis, power factor correction
Structural        — AISC beam deflection, column buckling, ASCE 7 wind/seismic loads
HVAC              — ASHRAE cooling load, duct sizing, psychrometrics
Medical           — Drug dosing, BMI, GFR, fluid balance
Finance           — Loan amortization, NPV, ROI, payroll (Peru DL 728)
Hydraulics        — Hazen-Williams, Darcy-Weisbach, pipe networks
Cybersecurity     — CVSS 3.1 scoring, network risk assessment
Solar/Energy      — PV yield, annual kWh, payback period
Telecom           — Link budget, dB margin, path loss
```

---

## Architecture

```
QOMN DSL (.qomn / all_domains.crys)
  ↓ Cranelift JIT + AVX2 AOT compilation at startup (~10ms, one-time)
Physics guards (API boundary) — invalid inputs rejected before JIT
Branchless oracle execution — VMULSD/VDIVSD, no branches in hot path
OracleCache — 57 plans, zero heap allocation per call (stack-only results)
  ↓
449M-540M simulations/sec · 14 MB RAM at rest · 8.3 MB binary
```

**Runtime:** Rust · Cranelift JIT · AVX2 + FMA
**Server:** AMD EPYC 12-core, 48GB RAM, 500GB NVMe ($80/month)
**API:** REST + WebSocket (real-time simulation stream)

---

## API

```
POST /api/plan/execute          — run a named physics plan
GET  /plans                     — list all 56 available plans
GET  /simulation/repeatability  — live determinism proof (variance=0)
POST /simulation/adversarial    — live NaN shield proof (0 panics)
POST /simulation/jitter_bench   — live jitter proof
GET  /health                    — engine status
```

Demo tier: rate-limited. Production access: percy.rojas@condesi.pe

---

## Cross-Architecture Hash Portability

```bash
# Default: uses VFMADD231SD on FMA-capable CPUs (server, cloud)
curl https://desarrollador.xyz/health
# "fma_path":"VFMADD231SD","no_fma":false

# QOMN_NO_FMA=1: force VMULSD+VADDSD — identical hash on any CPU (ARM, AVX-512, no-FMA)
QOMN_NO_FMA=1 qomn serve stdlib/all_domains.crys 9001
curl http://127.0.0.1:9001/health
# "fma_path":"VMULSD+VADDSD (QOMN_NO_FMA)","no_fma":true
```

**Why this matters:** `VFMADD231SD` (FMA) rounds once — `VMULSD + VADDSD` rounds twice.
For physics formulas like `Q * P / (3960 * eff)`, the results are identical (no fused mul-add pattern).
For future oracles using explicit FMA, `QOMN_NO_FMA=1` guarantees the same SHA-256 hash across
ARM servers, AMD EPYC, Intel Xeon, and any AVX-512 deployment.

---

## Why this matters

Engineering certifications require results that are **provably identical** across runs:

- **NFPA 20** (fire pump systems): calculation must match on audit
- **IEC 60364** (electrical installations): voltage drop must be reproducible
- **ASCE 7** (structural loads): seismic calculations must be auditable
- **FDA** (medical dosing): computation must be certifiable
- **CVSS 3.1** (cybersecurity): risk scores must be consistent

An LLM hallucinates. C++ with `-ffast-math` drifts. Python floats version-shift.
QOMN does not. The hash above is the proof.

---

## Related Repositories

- **Academic preprint and reproducibility artifact:** [`condesi/qomn-paper`](https://github.com/condesi/qomn-paper)
  — LaTeX source of the paper, bibliography, reproducibility scripts, installation guide for reviewers.
- **Archived QOMN v2.3 spec repository:** [`condesi/crysl-lang`](https://github.com/condesi/crysl-lang) *(read-only)*
  — Early QOMN language specification, kept under its original URL for reference continuity. Current specification lives in this repository (`QOMN_LANGUAGE_SPEC.md`) and in [`condesi/qomn-paper`](https://github.com/condesi/qomn-paper) (`LANGUAGE_SPEC.md`).

---

## Contact

Percy Rojas Masgo — Condesi Perú / Qomni AI Lab
percy.rojas@condesi.pe
https://desarrollador.xyz · https://github.com/condesi/qomn

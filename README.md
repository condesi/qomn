# CRYS-L — Execution Engine Without Variability for Critical Systems

**Mathematically reproducible in production under any load. Same input → same IEEE-754 bits. Always.**

Not an estimator. Not a model. A compiled oracle engine — physics expressions compile to Cranelift JIT + AVX2 and produce bit-identical results across any server, any load, any number of runs.

```
variance across 20 repeated runs  = 0.000000000000  (not "near zero" — exactly zero)
panics on 100,000 adversarial inputs = 0
jitter σ = 4,922 ns vs C++ σ = 850,000 ns  (173× more stable)
throughput = 77M+ simulations/sec
```

---

## Verify yourself — no account needed

```bash
# Live simulation status
curl https://qomni.clanmarketer.com/crysl/simulation/status

# IEEE-754 determinism: 20 runs, variance = 0.000000000000
curl https://qomni.clanmarketer.com/crysl/simulation/repeatability

# Adversarial resilience: 100,000 corrupt inputs (NaN/inf/zero/negative), 0 panics
curl https://qomni.clanmarketer.com/crysl/simulation/adversarial

# Jitter stability: σ=4,922ns vs C++ σ=850,000ns
curl https://qomni.clanmarketer.com/crysl/simulation/jitter_bench

# Determinism proof with result hash — same input always produces same hash
curl "https://qomni.clanmarketer.com/crysl/verify?runs=20"

# All 56 physics plans
curl https://qomni.clanmarketer.com/crysl/plans
```

Every response carries identity headers:
```
X-Engine: CRYS-L v3.2
X-Deterministic: true
X-IEEE-754: enforced
```

**Live dashboard:** https://qomni.clanmarketer.com/crysl/demo/benchmark.html

---

## Run a physics plan

```bash
# NFPA 20 fire pump sizing — bit-identical on every run, every server
curl -X POST https://qomni.clanmarketer.com/crysl/api/plan/execute \
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
      { "step": "shutoff_p",    "oracle": "nfpa20_shutoff_pressure", "result": 130.0 },
      { "step": "service_head", "oracle": "nfpa20_service_head",     "result": 231.0 }
    ]
  }
}
```

```bash
# IEC 60364 voltage drop — 480V, 125A, 300ft run, AWG 2
curl -X POST https://qomni.clanmarketer.com/crysl/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan":"plan_voltage_drop","params":{"voltage_v":480,"current_a":125,"length_ft":300,"wire_size_awg":2}}'
# result: 11.695181 V drop — same value, every time
```

---

## The mechanism

Standard C++ cannot be auto-vectorized when branches exist, and produces UB on invalid input:

```cpp
// Branches kill vectorization. NaN return = UB risk downstream.
float nfpa20_pump_hp(float flow, float head, float eff) {
    if (flow < 1.0 || flow > 50000.0 || eff < 0.10) return NAN;
    return (flow * 0.06309 * head * 0.70307) / (eff * 76.04);
}
```

CRYS-L replaces every branch with a float predicate:

```
oracle nfpa20_pump_hp(flow: float, head: float, eff: float) -> float:
    let valid = (flow >= 1.0) * (flow <= 50000.0) * (eff >= 0.10)
    ((flow * 0.06309 * head * 0.70307) / (eff * 76.04 + 0.0001)) * valid
```

- `valid` is `1.0` when all conditions pass, `0.0` otherwise
- Invalid input returns `0.0` — no NaN, no UB, no panic
- 4 scenarios pack into one `VMULPD` AVX2 instruction
- Cranelift JIT compiles each oracle to native binary, cached per domain

---

## Performance

| System | Scenarios/s | Jitter σ | Determinism | Cost/month |
|---|---|---|---|---|
| **CRYS-L v3.2 AVX2** | **77M+** | **4,922 ns** | **IEEE-754 exact** | **$80** |
| C++ GCC -O3 | ~5M | ~850,000 ns | risk: UB on NaN path | same HW |
| Python/NumPy | ~200K | >1ms | risk: version drift | same HW |

---

## 56 Physics Plans across 6 domains

```
Fire Protection   — sprinkler design, pump sizing, pipe flow (Manning/Hazen-Williams), egress
Electrical        — load analysis, 3-phase, transformer sizing, voltage drop, power factor
Structural        — beam deflection, column buckling, wind/seismic loads, foundation bearing
HVAC              — cooling load, duct sizing, psychrometrics
Finance           — payroll, tax brackets, NPV, loan amortization
Medical           — BMI, GFR, dosing, fluid balance
```

```bash
curl https://qomni.clanmarketer.com/crysl/plans | jq '.[].name'
```

---

## Why this matters for critical systems

Engineering certifications require results that are **provably identical** across runs:

- **NFPA 20** (fire pump systems): calculation must match on audit
- **IEC 60364** (electrical installations): voltage drop must be reproducible
- **ASCE 7** (structural loads): wind/seismic calculations must be auditable

An LLM cannot provide this. C++ with different compiler flags cannot provide this. CRYS-L can — and proves it live via the `/verify` endpoint, which returns a result hash that is identical on every call.

---

## Architecture

```
CRYS-L DSL source (.crysl)
  ↓ compile
Cranelift JIT → AVX2 native binary
Branchless float predicates (no NaN, no UB)
OracleCache — plans served at nanosecond latency
  ↓ 77M+ simulations/sec
Pareto optimizer — full parameter space sweep
3-objective: efficiency · cost · compliance
Returns all Pareto-optimal solutions
```

**Runtime:** Rust · Cranelift JIT · AVX2 + FMA  
**Server:** AMD EPYC 12-core, 48GB RAM ($80/month)  
**API:** REST + WebSocket (real-time simulation stream)

---

## API

Demo tier: 3 req/min on compute-heavy endpoints  
Production access: percy.rojas@condesi.pe

---

## Contact

Percy Rojas Masgo — Condesi Perú / Qomni AI Lab  
percy.rojas@condesi.pe  
https://qomni.clanmarketer.com/crysl/demo/benchmark.html

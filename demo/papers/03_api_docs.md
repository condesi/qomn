# QOMN API Documentation v3.2

**QOMN Language — Deterministic Engineering Optimization Runtime**
Qomni AI Lab · Condesi Perú

Base URL: `https://qomni.clanmarketer.com/qomn/api`
Local (self-hosted): `http://localhost:9001`

---

## Quick Start

```bash
# Health check
curl https://qomni.clanmarketer.com/qomn/api/health

# Evaluate a built-in oracle
curl -X POST https://qomni.clanmarketer.com/qomn/api/eval \
  -H "Content-Type: application/json" \
  -d '{"expr": "nfpa20_pump_hp(500.0, 100.0, 0.75)"}'

# Execute a named plan
curl -X POST https://qomni.clanmarketer.com/qomn/api/plan/execute \
  -H "Content-Type: application/json" \
  -d '{"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 100, "eff": 0.75}}'
```

---

## Language Reference

### Oracle Declaration

```qomn
oracle name(param1: type, param2: type, ...) -> return_type:
    let x = expression
    let y = other_expression
    y
```

The last expression is the implicit return value.

### Types

| Type | Description | Example |
|------|-------------|---------|
| `float` | IEEE 754 double precision | `3.14159` |
| `int` | 64-bit signed integer | `42` |
| `string` | UTF-8 string | `"hello"` |
| `Vec2` | 2D float vector | `vec2(1.0, 2.0)` |
| `Vec3` | 3D float vector | `vec3(1.0, 2.0, 3.0)` |
| `Vec4` | 4D float vector | `vec4(1.0, 2.0, 3.0, 4.0)` |
| `Mat3` | 3×3 float matrix (row-major) | `mat3(...)` |
| `Mat4` | 4×4 float matrix (row-major) | `mat4(...)` |

### Built-in Functions

#### Scalar Math
```qomn
abs(x)          sqrt(x)         pow(x, n)
sin(x)          cos(x)          tan(x)
floor(x)        ceil(x)         round(x)
min(a, b)       max(a, b)       clamp(x, lo, hi)
log(x)          exp(x)          sign(x)
```

#### Vector Operations
```qomn
vec2(x, y)                   # construct Vec2
vec3(x, y, z)                # construct Vec3
vec4(x, y, z, w)             # construct Vec4
dot(a, b)                    # dot product (Vec2/3/4)
cross(a, b)                  # cross product (Vec3 → Vec3)
norm(v)                      # ||v|| (Euclidean norm)
normalize(v)                 # v / ||v||
lerp(a, b, t)                # linear interpolation
```

#### Matrix Operations
```qomn
mat3(m0..m8)                 # construct 3×3 matrix (9 values)
det(m)                       # 3×3 determinant
transpose(m)                 # transpose Mat3
matmul(a, b)                 # Mat3×Mat3 or Mat3×Vec3
```

#### Communication
```qomn
respond("message " + str(value))   # return string response
str(x)                             # convert to string
```

### Operators

| Operator | Description |
|----------|-------------|
| `+` `-` `*` `/` | Arithmetic |
| `>` `>=` `<` `<=` `==` `!=` | Comparison (returns float 0.0 or 1.0) |
| `and` `or` `not` | Logical |

### Branchless Pattern

QOMN comparisons return `float` (0.0 or 1.0), enabling branchless conditionals:

```qomn
# Instead of: if x > 0 then x else 0
let positive_part = x * (x > 0.0)

# Compound validation (all conditions must be true):
let valid = (flow >= 0.1) * (flow <= 50000.0) * (eff >= 0.10)
# valid = 1.0 if all true, 0.0 if any false
```

---

## REST API Endpoints

### Health

**GET** `/health`

```json
{
  "ok": true,
  "version": "3.2",
  "plans_loaded": 14,
  "jit_compiled": 6,
  "uptime_s": 3600
}
```

---

### Evaluate Expression

**POST** `/eval`

```json
{ "expr": "nfpa20_pump_hp(500.0, 100.0, 0.75)" }
```

Response:
```json
{ "ok": true, "result": "52.3841", "type": "float" }
```

---

### Execute Plan

**POST** `/plan/execute`

```json
{
  "plan": "plan_pump_sizing",
  "params": {
    "Q_gpm": 500.0,
    "P_psi": 100.0,
    "eff": 0.75
  }
}
```

Response:
```json
{
  "ok": true,
  "plan": "plan_pump_sizing",
  "result": "HP required: 52.38 | Efficiency valid: 1.0 | Flow valid: 1.0",
  "latency_ms": 0.12
}
```

---

### Compile Oracle

**POST** `/compile`

Compile a QOMN source to native code or WASM.

```json
{
  "src": "oracle double(x: float) -> float:\n    x * 2.0",
  "backend": "llvm"
}
```

Backends: `"jit"` (default), `"llvm"` (→ .so), `"wasm"` (→ .wasm base64)

Response (LLVM):
```json
{
  "ok": true,
  "backend": "llvm",
  "ir_preview": "define double @double(double %x) { ... }",
  "so_path": "/tmp/qomn_double.so"
}
```

Response (WASM):
```json
{
  "ok": true,
  "backend": "wasm",
  "wasm_base64": "AGFzbQEAAAA...",
  "wat_preview": "(module (func $double ...))"
}
```

---

### List Plans

**GET** `/plans`

```json
{
  "ok": true,
  "plans": [
    { "name": "plan_pump_sizing",    "params": ["Q_gpm", "P_psi", "eff"],   "domain": "fire_protection" },
    { "name": "plan_voltage_drop",   "params": ["I_a", "L_m", "R", "pf"],   "domain": "electrical" },
    { "name": "plan_beam_analysis",  "params": ["load_kn", "span_m", "E", "I"], "domain": "structural" }
  ]
}
```

---

### Intent Routing

**POST** `/intent`

Natural language query routed to the best matching plan:

```json
{ "q": "What HP pump do I need for 500 GPM at 100 PSI with 75% efficiency?" }
```

Response:
```json
{
  "ok": true,
  "intent": "pump_sizing",
  "plan": "plan_pump_sizing",
  "params": { "Q_gpm": 500.0, "P_psi": 100.0, "eff": 0.75 },
  "result": "HP required: 52.38",
  "latency_ms": 48.2
}
```

---

### Simulation Engine

**POST** `/simulation/start`

Start the parallel sweep engine:

```json
{ "mode": "stratified" }
```

Modes: `"stratified"` (uniform grid), `"stress"` (includes boundary violations), `"adversarial"` (40% invalid inputs)

**GET** `/simulation/status`

```json
{
  "ok": true,
  "per_s": 124300000,
  "peak_per_s": 136300000,
  "valid_frac": 1.0,
  "ticks": 13600000,
  "pareto_size": 602
}
```

**POST** `/simulation/stop`

---

### Benchmark Proofs

**GET** `/simulation/simd_density`

Returns SIMD saturation proof (runs ~2M scenarios, dedicated):

```json
{
  "ok": true,
  "proof": "simd_saturation",
  "cpu_mhz": 2794.7,
  "kernel": "avx2+fma",
  "measured": { "scenarios_per_s": 154439021, "scenarios_per_cycle": 0.0450 },
  "theoretical": { "max_scenarios_per_s": 745265600 },
  "simd_utilization_pct": 16.9
}
```

**POST** `/simulation/jitter_bench`

```json
{ "ticks": 10000 }
```

Returns jitter determinism proof with histogram.

**POST** `/simulation/adversarial`

```json
{ "ticks": 5000 }
```

Returns adversarial resilience proof.

**GET** `/benchmark/vs_llm`

Returns LLM throughput comparison.

**GET** `/benchmark/all`

Runs all 4 proofs sequentially and returns combined JSON.

---

### WebSocket — Real-Time Pareto Stream

**WS** `/ws/sim`

Connect to receive real-time simulation updates every 100ms:

```javascript
const ws = new WebSocket('wss://qomni.clanmarketer.com/qomn/api/ws/sim');
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  // d.type = "sim_tick"
  // d.per_s = scenarios per second
  // d.pareto_size = number of Pareto solutions
  // d.valid_frac = fraction of valid scenarios
  // d.heatmap = [[flow_gpm, head_psi, eff_score, cost_usd, risk_score], ...]
};
```

---

## Code Examples

### Python

```python
import requests

BASE = "https://qomni.clanmarketer.com/qomn/api"

# Evaluate oracle
r = requests.post(f"{BASE}/eval", json={"expr": "nfpa20_pump_hp(500.0, 100.0, 0.75)"})
print(r.json())  # {"ok": true, "result": "52.3841"}

# Run full benchmark suite
r = requests.get(f"{BASE}/benchmark/all")
proofs = r.json()["proofs"]
print(f"SIMD: {proofs['simd']['measured']['scenarios_per_s']:,} scenarios/s")
print(f"LLM factor: {proofs['vs_llm']['speedup']['paper_figure']:,.0f}×")
```

### JavaScript

```javascript
const BASE = 'https://qomni.clanmarketer.com/qomn/api';

// Execute plan
const res = await fetch(`${BASE}/plan/execute`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ plan: 'plan_pump_sizing', params: { Q_gpm: 500, P_psi: 100, eff: 0.75 } })
});
const data = await res.json();
console.log(data.result);

// WebSocket stream
const ws = new WebSocket('wss://qomni.clanmarketer.com/qomn/api/ws/sim');
ws.onmessage = e => {
  const d = JSON.parse(e.data);
  console.log(`${(d.per_s/1e6).toFixed(1)}M scenarios/s, Pareto: ${d.pareto_size}`);
};
```

### Rust (via reqwest)

```rust
use reqwest::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let res = client
        .post("https://qomni.clanmarketer.com/qomn/api/eval")
        .json(&serde_json::json!({"expr": "nfpa20_pump_hp(500.0, 100.0, 0.75)"}))
        .send().await?
        .json::<serde_json::Value>().await?;
    println!("{}", res["result"]);
    Ok(())
}
```

### curl — Full Benchmark

```bash
#!/bin/bash
BASE="https://qomni.clanmarketer.com/qomn/api"

echo "=== QOMN Benchmark Suite ==="
echo ""
echo "--- Proof 2: SIMD Saturation ---"
curl -s "$BASE/simulation/simd_density" | python3 -m json.tool

echo ""
echo "--- Proof 1: Jitter (1000 ticks) ---"
curl -s -X POST "$BASE/simulation/jitter_bench" \
  -H "Content-Type: application/json" \
  -d '{"ticks":1000}' | python3 -m json.tool

echo ""
echo "--- Proof 4: LLM Factor ---"
curl -s "$BASE/benchmark/vs_llm" | python3 -m json.tool
```

---

## Self-Hosting

### Requirements

```
OS:       Linux x86-64 (Ubuntu 20.04+, Debian 11+)
CPU:      AVX2 support (Intel Haswell 2013+ / AMD Zen 2018+)
RAM:      4 GB minimum, 16 GB recommended
Rust:     1.75+ (cargo build --release)
Optional: llc-18, clang-18 (for LLVM backend)
Optional: wat2wasm 1.0.34 (for WASM backend)
```

### Build

```bash
git clone https://github.com/qomni-ai/qomn  # (planned public release)
cd qomn
cargo build --release
./target/release/qomn serve ./stdlib/all_domains.crys 9001
```

### systemd Service

```ini
[Unit]
Description=QOMN Optimization Engine
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/qomn serve /opt/qomn/stdlib/all_domains.crys 9001
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### nginx Proxy

```nginx
location /qomn/api/ {
    proxy_pass http://127.0.0.1:9001/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $http_upgrade;
    proxy_read_timeout 120s;
    add_header Access-Control-Allow-Origin *;
}
```

---

## Supported Engineering Domains

### Fire Protection (NFPA)

| Oracle | Standard | Parameters |
|--------|----------|-----------|
| `nfpa20_pump_hp` | NFPA 20 | flow_gpm, head_psi, efficiency |
| `nfpa20_shutoff_pressure` | NFPA 20 | rated_pressure_psi |
| `nfpa13_sprinkler` | NFPA 13 | k_factor, pressure_psi |
| `nfpa13_demand_flow` | NFPA 13 | area_m2, density, k |
| `nfpa72_detector_count` | NFPA 72 | area_m2, spacing_m |
| `nfpa101_egress_capacity` | NFPA 101 | width_m, occupant_load |

### Electrical

| Oracle | Description | Parameters |
|--------|-------------|-----------|
| `voltage_drop` | Single-phase | I_a, L_m, R_ohm_km, pf |
| `voltage_drop_3ph` | Three-phase | I_a, L_m, R_ohm_km, pf |

### Hydraulics

| Oracle | Method | Parameters |
|--------|--------|-----------|
| `hazen_williams_hf` | Hazen-Williams | Q, C, D, L |
| `darcy_head_loss` | Darcy-Weisbach | f, L, D, V |
| `manning_flow` | Manning | n, A, R, S |
| `pipe_velocity` | Continuity | Q, D |

### Structural

| Oracle | Description | Parameters |
|--------|-------------|-----------|
| `beam_deflection_udl` | Uniform distributed load | w, L, E, I |
| `bearing_capacity_terzaghi` | Soil bearing | c, phi, gamma, B, Df |

### Linear Algebra

All vector/matrix operations available as built-ins. See Language Reference section.

---

## Error Codes

| HTTP | Meaning |
|------|---------|
| 200 | Success |
| 400 | Parse error in QOMN source or missing parameters |
| 404 | Unknown plan name or endpoint |
| 405 | Method not allowed (GET vs POST) |
| 500 | Runtime error (compilation failure, etc.) |

All errors return JSON:
```json
{ "ok": false, "error": "description of what went wrong" }
```

---

## Version History

| Version | Features |
|---------|----------|
| v1.0 | Core oracle/plan syntax, JIT via Cranelift |
| v2.0 | REST API, intent routing, plan system |
| v2.4 | SPEC.md plan_* syntax, structured plans |
| v2.7 | Vec2/3/4, Mat3/4 types, linalg built-ins |
| v3.0 | LLVM 18 IR backend |
| v3.1 | WebAssembly (WAT) backend |
| v3.2 | Commander-level benchmark proofs, WebSocket heatmap, adversarial shield, Pareto fix |

---

*QOMN v3.2 · Qomni AI Lab · Condesi Perú · April 2026*
*Live dashboard: https://qomni.clanmarketer.com/qomn/demo/benchmark.html*

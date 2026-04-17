# CRYS-L — Crystal Language

**Compiled DSL for deterministic multi-objective engineering optimization**

[![Live Benchmarks](https://img.shields.io/badge/benchmarks-live-00e5ff)](https://qomni.clanmarketer.com/crysl/demo/benchmark.html)
[![API](https://img.shields.io/badge/API-v3.2-e040fb)](https://qomni.clanmarketer.com/crysl/api/health)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

## Performance

| Metric | Value |
|--------|-------|
| Scenarios/second | 154,439,021 |
| Pareto front (170 solutions) | 1.84 ms |
| Jitter σ (SCHED_FIFO, 10K ticks) | 3,334 ns |
| Panics under 1.28M adversarial inputs | 0 |
| Throughput vs. LLM sequential inference | 1.53B× |

All numbers are live and reproducible: [benchmark dashboard](https://qomni.clanmarketer.com/crysl/demo/benchmark.html)

## What is CRYS-L?

CRYS-L is a compiled domain-specific language for exhaustive engineering optimization. Instead of generating one probable answer, it evaluates every possible configuration in a parameter space and returns mathematically verified Pareto-optimal solutions.

```
oracle nfpa20_pump_hp(flow_gpm: float, head_psi: float, eff: float) -> float:
    let valid = (flow_gpm >= 1.0) * (flow_gpm <= 50000.0) * (eff >= 0.10)
    let q = flow_gpm * 0.06309
    let h = head_psi  * 0.70307
    ((q * h) / (eff * 76.04 + 0.0001)) * valid
```

Comparisons return `float` (0.0 or 1.0), enabling branchless physics validation that maps directly to AVX2 SIMD instructions.

## Architecture

```
oracle/plan source (.crys)
        ↓
    Lexer → Parser → AST → Type Checker
        ↓
   ┌────────────────────────────────┐
   │  Cranelift JIT (default)       │  sub-ms startup
   │  LLVM 18 IR text → .so        │  maximum optimization
   │  WebAssembly (WAT) → .wasm    │  browser deployment
   └────────────────────────────────┘
        ↓
   AVX2 SoA Sweep Engine
   (4 scenarios/instruction, SCHED_FIFO)
        ↓
   3-objective Pareto Front
   (efficiency × cost × risk)
```

## Quick Start

### Try the live API

```bash
# Health check
curl https://qomni.clanmarketer.com/crysl/api/health

# Evaluate an oracle
curl -X POST https://qomni.clanmarketer.com/crysl/api/eval \
  -H "Content-Type: application/json" \
  -d '{"expr": "nfpa20_pump_hp(500.0, 100.0, 0.75)"}'

# Run all 4 benchmark proofs
curl https://qomni.clanmarketer.com/crysl/api/benchmark/all | python3 -m json.tool

# Live SIMD proof
curl https://qomni.clanmarketer.com/crysl/api/simulation/simd_density
```

### Build from source

**Requirements:**
- Linux x86-64 (Ubuntu 20.04+ / Debian 11+)
- CPU with AVX2 support (Intel Haswell 2013+ / AMD Zen 2018+)
- Rust 1.75+
- Optional: `llc-18`, `clang-18` (LLVM backend), `wat2wasm` (WASM backend)

```bash
git clone https://github.com/condesi/crysl
cd crysl
cargo build --release
./target/release/crysl serve ./stdlib/all_domains.crys 9001
```

### systemd service

```ini
[Unit]
Description=CRYS-L Optimization Engine
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/crysl serve /opt/crysl/stdlib/all_domains.crys 9001
Restart=always
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

## Language Reference

### Oracle declaration

```
oracle name(param1: type, ...) -> return_type:
    let x = expression
    x
```

### Types

| Type | Description |
|------|-------------|
| `float` | IEEE 754 double precision |
| `int` | 64-bit signed integer |
| `string` | UTF-8 |
| `Vec2/3/4` | Float vectors |
| `Mat3/4` | Float matrices (row-major) |

### Branchless pattern

```
# Comparisons return float 0.0 or 1.0
let valid = (flow >= 0.1) * (flow <= 50000.0) * (eff >= 0.10)
# valid = 1.0 if all conditions true, 0.0 if any false
let result = computation * valid  # masks invalid outputs without branching
```

## Engineering Domains

- **Fire Protection**: NFPA 20 (pump sizing), NFPA 13 (sprinklers), NFPA 72 (detectors), NFPA 101 (egress)
- **Electrical**: voltage drop (1ph/3ph), load current, transformer sizing, conductor resistance
- **Hydraulics**: Hazen-Williams, Darcy-Weisbach, Manning flow, pipe velocity
- **Structural**: beam deflection (UDL), Terzaghi bearing capacity
- **Linear Algebra**: Vec2/3/4, Mat3/4, dot, cross, norm, normalize, matmul, det

## REST API

Full documentation: [API Docs](https://qomni.clanmarketer.com/crysl/demo/papers/papers_index.html)

Base URL: `https://qomni.clanmarketer.com/crysl/api`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status |
| `/eval` | POST | Evaluate expression |
| `/plan/execute` | POST | Execute named plan |
| `/compile` | POST | Compile to LLVM/WASM |
| `/plans` | GET | List available plans |
| `/simulation/simd_density` | GET | SIMD benchmark proof |
| `/benchmark/all` | GET | All 4 proofs |
| `/ws/sim` | WS | Real-time Pareto stream |

## Papers

- [Full Technical Paper](https://qomni.clanmarketer.com/crysl/demo/papers/01_main_paper.md)
- [API Documentation](https://qomni.clanmarketer.com/crysl/demo/papers/03_api_docs.md)
- [Papers Index](https://qomni.clanmarketer.com/crysl/demo/papers/papers_index.html)
- [Language Specification (SPEC.md)](https://github.com/condesi/crysl-lang/blob/main/SPEC.md)
- [Originality Statement](https://github.com/condesi/crysl-lang/blob/main/ORIGINALITY.md)

## About

CRYS-L is the deterministic compute substrate of **Qomni** — a hybrid neuro-symbolic AI platform for engineering decision support.

- **Author**: Percy Rojas Masgo — [percy.rojas@condesi.pe](mailto:percy.rojas@condesi.pe)
- **Organization**: Qomni AI Lab · Condesi Perú
- **License**: Apache 2.0

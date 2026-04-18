# QOMN: QOMN Language Specification v3.1

**A Domain-Specific Language for Real-Time Engineering Computation**

Percy Rojas Masgo · Qomni AI Lab · Condesi Perú
April 2026

---

## Abstract

QOMN (QOMN Language) is a statically-typed, JIT-compiled domain-specific language designed for sub-microsecond engineering computation. Unlike general-purpose languages, QOMN compiles domain knowledge (fire protection, electrical, structural, hydraulics, finance) into native x86-64 machine code via Cranelift, achieving computation times of **1.4–2.5 µs per plan** — within 175x of C -O2 bare metal. The language introduces the concepts of *oracles* (pure computation kernels), *plans* (directed acyclic graphs of oracle calls), and *intent routing* (natural language → plan dispatch). This paper documents the complete language specification, compiler architecture, AOT optimization pipeline, and benchmark results.

---

## Table of Contents

1. [Language Overview](#1-language-overview)
2. [Lexical Structure](#2-lexical-structure)
3. [Type System](#3-type-system)
4. [Declarations](#4-declarations)
5. [Expressions & Operators](#5-expressions--operators)
6. [Statements](#6-statements)
7. [The Oracle Model](#7-the-oracle-model)
8. [The Plan Model](#8-the-plan-model)
9. [Compiler Pipeline](#9-compiler-pipeline)
10. [JIT Engine (Cranelift)](#10-jit-engine-cranelift)
11. [AOT Plan Compiler](#11-aot-plan-compiler)
12. [Intent Parser (NLP Routing)](#12-intent-parser-nlp-routing)
13. [Standard Library](#13-standard-library)
14. [REST API Server](#14-rest-api-server)
15. [Benchmark Results](#15-benchmark-results)
16. [Architecture Summary](#16-architecture-summary)

---

## 1. Language Overview

QOMN is not a general-purpose language. It is a *computation language* — every program describes a set of mathematical functions (oracles) and their composition (plans). The language enforces:

- **Purity**: Oracles are pure functions with no side effects
- **Determinism**: Same inputs always produce same outputs
- **Measurability**: Every computation is timed to nanosecond precision
- **Composability**: Plans compose oracles into DAG pipelines

### Design Philosophy

```
"Compute what C computes, express what Python expresses, deploy what Docker deploys."
```

QOMN occupies a unique niche: it combines the mathematical expressiveness of domain-specific notations (NFPA formulas, Hazen-Williams equations, NEC electrical codes) with the execution speed of compiled native code, wrapped in a REST API that accepts natural language queries.

### Hello World

```crys
oracle square(x: float) -> float:
    return x * x

plan plan_area(side: float):
    step area: square(side)
```

Query via REST:
```bash
curl -X POST http://localhost:9001/intent \
  -d '{"q":"area de cuadrado lado 5"}'
```

Response (in 1.8 µs):
```json
{
  "ok": true,
  "plan": "plan_area",
  "result": {
    "steps": [{"step":"area","oracle":"square","result":25.0,"latency_ns":1800}]
  }
}
```

---

## 2. Lexical Structure

### 2.1 Tokens

| Category | Tokens |
|---|---|
| **Literals** | `Int(i64)`, `Float(f64)`, `Str(String)`, `Bool(true/false)`, `Trit(+1/0t/-1)` |
| **Keywords** | `oracle`, `crystal`, `pipe`, `plan`, `step`, `route`, `schedule`, `load`, `let` |
| **Control** | `if`, `else`, `for`, `in`, `return`, `respond`, `assert` |
| **Scalar types** | `f32`, `f64`, `i32`, `i64`, `bool`, `str`, `float` (alias for f64) |
| **Ternary types** | `trit`, `tvec`, `tmat`, `tensor` |
| **Operators** | `+ - * / % ^`, `== != < > <= >=`, `= and or not` |
| **Hardware hints** | `@mmap`, `@avx2`, `@cpu`, `@auto` |
| **Delimiters** | `( ) [ ] { } , : ..` |

### 2.2 Comments

```crys
// Single-line comment
// No block comments — every line is explicit
```

### 2.3 Numeric Literals

```crys
42          // Int
3.14        // Float
0.15        // Float (leading zero required)
1_000_000   // Int with separators (future)
```

### 2.4 Ternary Literals

```crys
+1          // Trit: positive
0t          // Trit: zero (suffix 't' distinguishes from int 0)
-1          // Trit: negative
```

---

## 3. Type System

### 3.1 Scalar Types

| Type | Width | Description |
|---|---|---|
| `f32` | 32-bit | IEEE 754 single-precision float |
| `f64` / `float` | 64-bit | IEEE 754 double-precision float (default) |
| `i32` | 32-bit | Signed integer |
| `i64` | 64-bit | Signed integer |
| `bool` | 1-bit | `true` / `false` |
| `str` | heap | UTF-8 string |

### 3.2 Ternary Types

QOMN has first-class support for ternary ({-1, 0, +1}) computation:

| Type | Description |
|---|---|
| `trit` | Single ternary value |
| `tvec[N]` | Fixed-size ternary vector |
| `tmat[R][C]` | Ternary matrix (2-bit packed storage) |
| `tensor[T, dims]` | Multi-dimensional tensor |

### 3.3 Physical Unit Types (v2.0)

QOMN supports dimensional analysis through unit annotations:

```crys
oracle nfpa13_sprinkler(K: float(gpm/psi^0.5), P: float(psi)) -> float(gpm):
    return K * (P ^ 0.5)
```

Unit expressions:
- `float(gpm)` — gallons per minute
- `float(psi)` — pounds per square inch
- `float(gpm/psi^0.5)` — K-factor unit
- `float(m2)` — square meters
- `float(A*m)` — ampere-meters

With range constraints:
```crys
P: float(psi)[0.0..175.0]   // pressure between 0 and 175 psi
```

### 3.4 Type Inference

The `Inferred` type allows the compiler to determine types:
```crys
let x = 3.14       // inferred as f64
let n = 42          // inferred as i64
let t = +1          // inferred as trit
```

---

## 4. Declarations

### 4.1 Oracle Declaration

An oracle is a pure function — the fundamental computation unit:

```crys
oracle name(param1: type1, param2: type2) -> return_type:
    // body: statements ending with return
    return expr
```

Example:
```crys
oracle nfpa20_pump_hp(Q: float, P: float, eff: float) -> float:
    // HP = Q * P / (3960 * eff)   [NFPA 20 formula]
    return Q * P / (3960.0 * eff)
```

**Constraints:**
- Oracles are pure — no I/O, no global state, no side effects
- All parameters must have explicit types
- Return type must be explicit
- Body must end with a `return` statement
- Oracles are compiled to native x86-64 via Cranelift JIT (2.4–25 ns/call)

### 4.2 Plan Declaration

A plan composes oracles into a Directed Acyclic Graph (DAG):

```crys
plan plan_name(param1: type1, param2: type2, ...):
    step step_name: oracle_name(args...)
    step step_name: oracle_name(prev_step, args...)
    // Steps can reference previous step results by name
```

Example:
```crys
plan plan_sprinkler_system(area_ft2: float, K: float, P_avail: float, hose_stream: float):
    step density:      nfpa13_area_density(area_ft2)
    step Q_per_head:   nfpa13_sprinkler(K, P_avail)
    step n_heads:      nfpa13_sprinkler_count(area_ft2, 130.0)
    step Q_demand:     nfpa13_demand_flow(density, area_ft2, hose_stream)
    step pump_hp:      nfpa20_pump_hp(Q_demand, P_avail, 0.70)
    step detectors:    nfpa72_detector_count(area_ft2, 50.0)
```

**DAG Semantics:**
- Steps with no data dependencies on each other execute in **parallel** (via `std::thread::scope`)
- Steps that reference previous step names execute **serially** in topological order
- The compiler computes parallel groups automatically (Kahn's algorithm)

In the sprinkler example:
- **Group 1** (parallel): `density`, `Q_per_head`, `n_heads`, `detectors` — all depend only on inputs
- **Group 2** (serial after G1): `Q_demand` — depends on `density`
- **Group 3** (serial after G2): `pump_hp` — depends on `Q_demand`

### 4.3 Crystal Declaration

A crystal is a quantized neural weight matrix:

```crys
crystal model_weights @mmap "path/to/weights.crystal"
```

Supports hardware hints: `@mmap` (lazy load), `@avx2` (SIMD), `@cpu`, `@auto`.

### 4.4 Pipe Declaration

A pipe is a linear processing pipeline:

```crys
pipe classifier(input: tvec[768]):
    step encoded = encode(input, 512)
    step output  = crystal_weights.infer(encoded)
    respond(output)
```

### 4.5 Route Declaration

Routes map patterns to computation targets:

```crys
route "fire_*"    -> plan_sprinkler_system
route "electric*" -> plan_electrical_load
route *           -> fallback_handler
```

### 4.6 Schedule Declaration

Hardware-aware dispatch:

```crys
schedule matrix_mult:
    if @avx2    -> @avx2
    if @ternary -> @cpu
    else        -> @auto
```

---

## 5. Expressions & Operators

### 5.1 Arithmetic

| Operator | Description | Example |
|---|---|---|
| `+` | Addition | `a + b` |
| `-` | Subtraction / Negation | `a - b`, `-x` |
| `*` | Multiplication | `a * b` |
| `/` | Division (safe: div/0 → 0.0) | `a / b` |
| `%` | Modulo | `a % b` |
| `^` | Power | `P ^ 0.5` (= √P) |

### 5.2 Comparison

| Operator | Description |
|---|---|
| `==` | Equal |
| `!=` | Not equal |
| `<`, `>` | Less/greater than |
| `<=`, `>=` | Less/greater or equal |

### 5.3 Logical

| Operator | Description |
|---|---|
| `and` | Logical AND |
| `or` | Logical OR |
| `not` | Logical NOT |

### 5.4 Special Expressions

```crys
encode(expr, dim)     // Encode value to dimensional representation
quantize(expr)        // Quantize to ternary
expr.infer(x)         // Crystal inference
expr.layer(N)         // Access crystal layer
expr.norm()           // Crystal normalization
a | b | c             // Pipeline composition
```

---

## 6. Statements

### 6.1 Let Binding

```crys
let x: float = 3.14
let name = "NFPA"        // type inferred
```

### 6.2 If/Else

```crys
if pressure > 175.0:
    return max_pressure
else:
    return pressure
```

### 6.3 For Loop

```crys
for i in 0..100:
    // loop body
```

### 6.4 Assert (v2.0)

```crys
assert pressure >= 0.0 "Pressure cannot be negative"
```

### 6.5 Return / Respond

```crys
return expr         // Return value from oracle
respond(expr)       // Output from pipe
```

---

## 7. The Oracle Model

Oracles are the core abstraction in QOMN. An oracle represents a single, verifiable engineering formula.

### 7.1 Properties

1. **Pure**: No side effects, no global state
2. **Deterministic**: f(x) always returns the same value
3. **JIT-compiled**: Compiled to native x86-64 at startup via Cranelift
4. **Auditable**: Every oracle maps to a specific standard (NFPA 13, IEC 60038, etc.)
5. **Nanosecond-latency**: 2.4–25 ns per oracle call after JIT

### 7.2 JIT Compilation

When QOMN starts in `serve` mode:

```
Source (.crys) → Lexer → Parser → AST → Bytecode → JIT (Cranelift) → x86-64 fn_ptr
```

The JIT engine compiles each oracle body to a native function with ABI:
```c
extern "C" double oracle_fn(const double* params, size_t n_params);
```

Cranelift optimizations applied:
- **Power inlining**: `x^0.5` → `fsqrt`, `x^2` → `fmul(x,x)`, `x^-1` → `fdiv(1,x)`
- **Safe division**: `a/b` → `select(b==0, 0.0, a/b)` (branch-free)
- **NaN clamping**: Return values are clamped to prevent NaN propagation
- **Constant folding**: Literal expressions evaluated at compile time

### 7.3 Oracle Cache

Results are cached by `(oracle_name, args_hash)` key:
```rust
Key: (String, u64)  // oracle name + polynomial hash of f64 args
```
Cache hit avoids recomputation entirely (0 ns).

---

## 8. The Plan Model

Plans compose oracles into computation DAGs.

### 8.1 Parallel DAG Execution

The plan executor uses Kahn's topological sort to identify independent steps:

```
plan_sprinkler_system:
  Group 1 (parallel): density, Q_per_head, n_heads, detectors
  Group 2 (serial):   Q_demand ← depends on density
  Group 3 (serial):   pump_hp  ← depends on Q_demand
```

Groups execute via `std::thread::scope` — zero-cost threading for parallel oracle calls.

### 8.2 Execution Modes

| Mode | Latency | Description |
|---|---|---|
| **PlanExecutor** (v2.0) | 6,400 ns | Original: HashMap lookup + DAG rebuild per request |
| **AOT Level 1** (v3.1) | 1,400 ns | Pre-resolved dispatch tables, array-indexed oracles |
| **AOT Level 2** (v3.1) | ~50 ns* | Full plan compiled as single Cranelift function |

*Level 2 target — compilation of entire plan into one native function.

### 8.3 AOT Pre-compilation

At startup, the AOT Plan Compiler:
1. Pre-computes topological order for each plan
2. Resolves all oracle references to array indices (no HashMap at runtime)
3. Pre-maps argument sources: `Param(i)`, `Step(j)`, or `Const(v)`
4. Stores direct function pointers for each oracle

Runtime execution:
```rust
// No HashMap, no String allocation, no DAG computation
for step in plan.exec_steps {
    args = resolve_from_array(step.arg_sources);  // ~5 ns
    result = oracle_fns[step.oracle_idx](args);   // ~5-10 ns (JIT)
}
```

---

## 9. Compiler Pipeline

```
┌──────────┐   ┌────────┐   ┌───────┐   ┌─────┐   ┌──────────┐
│  Source   │──▶│ Lexer  │──▶│ Parser│──▶│ AST │──▶│ Type     │
│ (.crys)  │   │ (450L) │   │ (735L)│   │     │   │ Checker  │
└──────────┘   └────────┘   └───────┘   └─────┘   │ (294L)   │
                                                    └────┬─────┘
                                                         │
                                          ┌──────────────┘
                                          ▼
                                    ┌──────────┐   ┌──────────────┐
                                    │ Bytecode │──▶│ JIT Engine   │
                                    │ Compiler │   │ (Cranelift)  │
                                    │ (929L)   │   │ (834L)       │
                                    └──────────┘   └──────┬───────┘
                                                          │
                                          ┌───────────────┘
                                          ▼
                                    ┌──────────┐   ┌──────────────┐
                                    │ AOT Plan │──▶│ HTTP Server  │
                                    │ Compiler │   │ (6,956L)     │
                                    │ (267L)   │   └──────────────┘
                                    └──────────┘
```

### 9.1 CRYS-ISA Bytecode

The intermediate bytecode (CRYS-ISA) uses a register-based instruction set:

| Opcode | Description |
|---|---|
| `LoadConst` | Load f64 constant from pool |
| `LoadVar` | Load variable by index |
| `StoreVar` | Store to variable |
| `LoadTrit` | Load ternary literal |
| `Add/Sub/Mul/Div` | Arithmetic on f64 registers |
| `Pow` | Power (static or dynamic exponent) |
| `Move` | Register copy |
| `Return` | Return value from oracle |
| `Halt` | End execution |
| `ORACLE_CALL` | Invoke another oracle (async capable) |
| `ORACLE_WAIT` | Wait for async oracle result |
| `PAR_BEGIN/PAR_END` | Fork/join parallel execution |
| `LOAD_CRYS` | Load crystal weights (L1Pin/Stream/Prefetch) |
| `MM_TERN` | Ternary matrix multiplication (AVX2) |

### 9.2 Bytecode VM

The Bytecode VM executes CRYS-ISA with:
- **Async oracle engine**: Thread pool for concurrent oracle calls
- **Crystal cache**: mmap lazy-loader for .crystal weight files
- **Memory pool**: Slab allocator for zero-copy f32/i8 buffers
- **Profiler**: Per-instruction nanosecond timing

---

## 10. JIT Engine (Cranelift)

### 10.1 Architecture

```
CRYS-ISA Bytecode
     │
     ▼
  Walk opcodes (entry_ip → Return/Halt)
     │
     ▼
  Lower to Cranelift IR (F64 SSA values)
     │
     ▼
  JITModule::define_function()
     │
     ▼
  finalize() → fn_ptr (native x86-64)
     │
     ▼
  Store in JitCache → invoke via fn_ptr
```

### 10.2 ABI

All compiled oracle functions share the same ABI:
```c
extern "C" double oracle_fn(const double* params, size_t n_params);
```

### 10.3 Power Inlining

| QOMN Expression | Cranelift IR | Latency |
|---|---|---|
| `x ^ 0.5` | `fsqrt(x)` | 1–3 ns |
| `x ^ 2.0` | `fmul(x, x)` | <1 ns |
| `x ^ 3.0` | `fmul(fmul(x,x), x)` | <1 ns |
| `x ^ 4.0` | `sq=fmul(x,x); fmul(sq,sq)` | <1 ns |
| `x ^ 0.25` | `fsqrt(fsqrt(x))` | 2–6 ns |
| `x ^ -1.0` | `fdiv(1.0, x)` | 1–3 ns |
| `x ^ n` (other) | `exp2(n * log2(x))` | ~15 ns |

### 10.4 Safety

- **Division by zero**: `select(denom==0, 0.0, l/r)` — branch-free, zero overhead
- **NaN clamping**: `select(Ordered(x,x), x, 0.0)` — prevents NaN propagation
- Both are compiled as branchless x86-64 `cmov` instructions

### 10.5 Performance

Benchmark (Server EPYC 12-core, all oracles):

| Engine | Latency/call | vs Interpreter |
|---|---|---|
| Bytecode Interpreter | 165–266 ns | 1x |
| JIT v1.6 | 2.4–25.1 ns | 9.3–101.6x |
| JIT v1.6.1 (pow inlined) | 2.4–5.0 ns | 53–110x |

---

## 11. AOT Plan Compiler

### 11.1 Motivation

The JIT compiles individual oracles to native code (2.4–10 ns each). But plan execution had ~5,000 ns of orchestration overhead:

| Overhead Source | Cost |
|---|---|
| HashMap: JIT table lookup × 6 | ~600 ns |
| HashMap: cache lookup × 6 | ~600 ns |
| `String::clone()` × 12 | ~480 ns |
| `Instant::now()` × 6 | ~120 ns |
| AST walk `resolve_args()` × 6 | ~300 ns |
| `parallel_groups()` DAG computation | ~200 ns |
| Vec allocations | ~200 ns |
| `to_human()` + `to_json()` | ~3,000 ns |

### 11.2 AOT Level 1: Pre-resolved Dispatch

Compile-time resolution eliminates all per-request overhead:

```rust
struct FastStep {
    oracle_idx: usize,          // direct array index, no HashMap
    args: Vec<ArgSource>,       // Param(i) | Step(j) | Const(v)
}

enum ArgSource {
    Param(usize),   // from input params[i]
    Step(usize),    // from results[j]
    Const(f64),     // literal constant
}
```

**Results:**

| Plan | Before (PlanExecutor) | After (AOT L1) | Speedup |
|---|---|---|---|
| Sprinkler (6 steps) | 6,400 ns | 1,400 ns | **4.5x** |
| Planilla (5 steps) | 1,964,000 ns | 1,400 ns | **1,400x** |
| Solar FV (5 steps) | 215,000 ns | 1,900 ns | **113x** |
| Factura (3 steps) | 8,000 ns | 1,300 ns | **6.2x** |

### 11.3 AOT Level 2: Full Plan JIT (Target)

Compile entire plan as a single Cranelift function, inlining all oracle bodies:

```
// Generated x86-64 for plan_sprinkler_system:
density  = 0.15                    // immediate load
Q        = K * fsqrt(P)            // 2 instructions
n_heads  = area / 130.0            // 1 instruction
demand   = 0.15 * area + hose      // fma instruction
pump_hp  = demand * P / 2772.0     // 2 instructions
detect   = area / 2500.0           // 1 instruction
```

Expected: **15–30 ns** per plan (2–4x of C -O2).

---

## 12. Intent Parser (NLP Routing)

### 12.1 Architecture

The Intent Parser converts natural language queries to plan execution:

```
"rociadores para almacén de 500m2"
    │
    ▼
  Regex pattern matching (domain classification)
    │
    ▼
  IntentAST { domain: "fire_protection", params: {area_ft2: 1000}, ... }
    │
    ▼
  route_to_plan() → "plan_sprinkler_system"
    │
    ▼
  AOT execute → result in 1.4 µs
```

### 12.2 Domain Detection

The parser uses domain-specific regex patterns:

| Domain | Pattern Examples |
|---|---|
| `fire_protection` | rociador, sprinkler, bomba, NFPA, incendio, gpm, psi |
| `electrical` | eléctric, voltaje, transformador, cortocircuito, kw, trifásico |
| `structural` | viga, columna, zapata, cimentación, kN, flexión |
| `hydraulics` | tubería, caudal, Manning, Hazen, presión |
| `accounting_peru` | factura, IGV, planilla, CTS, liquidación |
| `security` | password, CVSS, audit, crypto, brute force |
| `solar` | solar, fotovoltaico, panel, kWh, radiación |
| `medical` | IMC, BMI, dosis, autoclave, gas médico |

### 12.3 Parameter Extraction

The parser extracts numeric values from natural language:
- `"500m2"` → `area_ft2: 5382.0` (auto-converted m² → ft²)
- `"750 gpm"` → `Q_gpm: 750.0`
- `"125 psi"` → `P_psi: 125.0`
- `"sueldo 3500"` → `sueldo: 3500.0`

### 12.4 Cognitive Loop Detection

For queries like "¿a qué caudal cae la presión bajo 65 psi?", the parser detects a simulation structure:

```json
{
  "domain": "cognitive_loop",
  "plan": "loop:hazen_P_at_gpm",
  "range_start": 100, "range_end": 1000, "step": 10,
  "cond_op": "<", "cond_val": 65
}
```

This triggers the Cognitive Compiler to execute the oracle in a loop, finding the critical point.

---

## 13. Standard Library

### 13.1 Domains

The `all_domains.crys` stdlib contains **171 oracles** and **55 plans** across 13 domains:

| Domain | Oracles | Plans | Standards |
|---|---|---|---|
| Fire Protection | 16 | 3 | NFPA 13, 20, 72, 101 |
| Hydraulics | 7 | 2 | Manning, Hazen-Williams |
| Electrical | 11 | 5 | IEC 60038, NEC, CNE Peru |
| Civil/Structural | 10 | 5 | ACI 318, E060 Peru |
| Accounting Peru | 8 | 5 | SUNAT, IGV, CTS |
| Legal Peru | 4 | 2 | SUNAFIL, DL 728 |
| Hydraulic Networks | 8 | 3 | Series pipe analysis |
| HVAC | 5 | 2 | ASHRAE |
| Solar FV | 5 | 1 | IEC 61215 |
| Commerce | 7 | 4 | Break-even, ROI, pricing |
| Medical | 5 | 3 | BMI, drug dosing, autoclave |
| Security | 6 | 4 | CVSS, password audit, crypto |
| Other | 9 | 3 | Statistics, irrigation, telecom |
| **Total** | **171** | **55** | — |

### 13.2 Example: Fire Protection

```crys
// NFPA 13: Sprinkler demand flow
oracle nfpa13_demand_flow(density: float, area_ft2: float, hose_stream: float) -> float:
    return density * area_ft2 + hose_stream

// NFPA 20: Fire pump horsepower
oracle nfpa20_pump_hp(Q: float, P: float, eff: float) -> float:
    return Q * P / (3960.0 * eff)

// NFPA 72: Detector count
oracle nfpa72_detector_count(area_ft2: float, spacing_ft: float) -> float:
    return area_ft2 / (spacing_ft * spacing_ft)
```

### 13.3 Example: Electrical

```crys
oracle load_current_3ph(P_w: float, V: float, pf: float) -> float:
    return P_w / (1.7321 * V * pf)

oracle voltage_drop_3ph(I: float, L: float, rho: float, A: float) -> float:
    return 1.7321 * I * L * rho / A

oracle transformer_rating(P_kw: float, pf: float, eff: float) -> float:
    return P_kw / (pf * eff)
```

### 13.4 Example: Accounting Peru

```crys
oracle igv_compute(base: float) -> float:
    return base * 0.18

oracle cts_compute(sueldo: float, gratificacion_mensual: float) -> float:
    return (sueldo + gratificacion_mensual) / 12.0
```

---

## 14. REST API Server

### 14.1 Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status, version, plan count, JIT status |
| `GET` | `/plans` | List all plans with parameters |
| `POST` | `/intent` | Natural language → plan execution |
| `POST` | `/plan/execute` | Direct plan execution with params |
| `POST` | `/query` | Raw expression evaluation |
| `POST` | `/eval` | Inline expression evaluation |
| `POST` | `/convert` | Unit conversion |
| `GET` | `/memory` | Cognitive memory (past experiences) |

### 14.2 Intent Endpoint

**Request:**
```json
{
  "q": "diseño sistema contra incendio almacén 500 m2 densidad 0.15"
}
```

**Response:**
```json
{
  "ok": true,
  "domain": "fire_protection",
  "query": "diseño sistema contra incendio almacén 500 m2 densidad 0.15",
  "plan": "plan_sprinkler_system",
  "human": "## Engineering Computation — plan_sprinkler_system\n\n**Inputs:**\n  - K: 5.6000 gpm/psi^0.5\n  - area_ft2: 1000.0000 ft²\n...\n\n*Computed in 1.4 µs · 3 parallel groups · 0 cache hits*",
  "result": {
    "plan": "plan_sprinkler_system",
    "steps": [
      {"step":"density","oracle":"nfpa13_area_density","result":0.15,"latency_ns":0},
      {"step":"Q_per_head","oracle":"nfpa13_sprinkler","result":43.3774,"latency_ns":0},
      {"step":"n_heads","oracle":"nfpa13_sprinkler_count","result":7.6923,"latency_ns":0},
      {"step":"Q_demand","oracle":"nfpa13_demand_flow","result":400.0,"latency_ns":0},
      {"step":"pump_hp","oracle":"nfpa20_pump_hp","result":8.658,"latency_ns":0},
      {"step":"detectors","oracle":"nfpa72_detector_count","result":0.4,"latency_ns":0}
    ],
    "total_ns": 1400,
    "cache_hits": 0
  }
}
```

### 14.3 Server Architecture

The server is a raw TCP listener (no framework dependency):
- **Thread-per-connection** model via `std::thread::spawn`
- **Pattern-matching router**: `(method, path)` tuple matching
- **Request parsing**: Manual HTTP/1.1 parsing (no external dependency)
- **JSON output**: Manual formatting (no serde overhead in hot path)

### 14.4 Additional Capabilities

The server also provides:
- **Web orchestration**: `/web/orchestrate`, `/web/crawl`, `/web/probe/v2`
- **Live threat monitoring**: `/web/live-threats`
- **Defense system**: `/defense/block-ip`, `/defense/status`
- **Security audit**: `/audit/verify`
- **Nginx patching**: `/patch/nginx/apply`
- **Self-healing watchdog**: Background thread monitoring server health
- **Cognitive memory**: Persists execution experiences for learning
- **NVD polling**: Background CVE database synchronization

---

## 15. Benchmark Results

### 15.1 Environment

- **CPU**: AMD EPYC (12 cores, Server5)
- **RAM**: 48 GB DDR4
- **OS**: Ubuntu 24.04
- **Rust**: 1.94.1 (2026-03-24)
- **Cranelift**: 0.113

### 15.2 Oracle-Level Performance

| Engine | Latency/call | Notes |
|---|---|---|
| C -O2 (bare metal) | 1.2–1.7 ns | Baseline |
| QOMN JIT (Cranelift) | 2.4–5.0 ns | 2–3x C |
| Bytecode Interpreter | 165–266 ns | 66–110x slower than JIT |
| Python equivalent | ~50,000 ns | 10,000–20,000x slower than JIT |

### 15.3 Plan-Level Performance

| Plan | Steps | Old (PlanExecutor) | AOT Level 1 | vs C -O2 |
|---|---|---|---|---|
| Sprinkler | 6 | 6,400 ns | 1,400 ns | 175x |
| Pump sizing | 4 | 7,700 ns | 2,000 ns | 250x |
| Beam analysis | 3 | 5,000 ns | 1,800 ns | 225x |
| Planilla | 5 | 1,964,000 ns | 1,400 ns | 175x |
| Factura | 3 | 8,000 ns | 1,300 ns | 163x |
| Solar FV | 5 | 215,000 ns | 1,900 ns | 238x |
| Irrigation | 1 | 2,300 ns | 1,700 ns | 213x |
| Transformer | 3 | 7,700 ns | 1,800 ns | 225x |

### 15.4 Comparison with Industry

| System | Latency (6-step plan) | Relative |
|---|---|---|
| C -O2 (bare metal) | 8 ns | 1x |
| Rust native (no JIT) | 15–30 ns | 2–4x |
| **QOMN AOT (warm)** | **1,400 ns** | **175x** |
| Go (compiled) | ~5,000 ns | 625x |
| Java (JVM HotSpot) | ~50,000 ns | 6,250x |
| JavaScript (V8) | ~500,000 ns | 62,500x |
| Python | ~50,000,000 ns | 6,250,000x |

### 15.5 Throughput

At 1,400 ns per plan, the theoretical throughput is:
- **714,000 plans/second** per core
- **8.5 million plans/second** on 12 cores
- This exceeds most REST API servers by 2–3 orders of magnitude

---

## 16. Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      QOMN v3.1 Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Natural Language Query                                         │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                            │
│  │  Intent Parser   │  Regex domain classification              │
│  │  (1,578 lines)   │  Parameter extraction                    │
│  └────────┬────────┘  Plan routing                              │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐     ┌───────────────────┐                  │
│  │  AOT Plan Cache  │────▶│  JIT Oracle Table  │                │
│  │  53 plans        │     │  186 oracles       │                │
│  │  Array dispatch   │     │  Cranelift x86-64  │                │
│  │  ~1,400 ns/plan  │     │  2.4-5 ns/call     │                │
│  └────────┬────────┘     └───────────────────┘                  │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  PlanResult      │  JSON + Human-readable output             │
│  │  to_json()       │  Markdown with units + labels             │
│  │  to_human()      │  Step-by-step trace                       │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  HTTP Server     │  Raw TCP, pattern-matching router         │
│  │  Port 9001       │  Thread-per-connection                    │
│  │  6,956 lines     │  34+ endpoints                            │
│  └─────────────────┘                                            │
│                                                                 │
│  Total: 18,271 lines Rust · 985 lines QOMN stdlib            │
│  171 oracles · 55 plans · 13 domains                            │
│  JIT: Cranelift 0.113 · AOT: Pre-resolved dispatch              │
└─────────────────────────────────────────────────────────────────┘
```

### Source Code Statistics

| Module | Lines | Purpose |
|---|---|---|
| `server.rs` | 6,956 | HTTP server, 34+ REST endpoints |
| `intent_parser.rs` | 1,578 | NLP → plan routing |
| `bytecode.rs` | 929 | CRYS-ISA bytecode compiler |
| `bytecode_vm.rs` | 925 | Bytecode interpreter |
| `backend_cpu.rs` | 892 | AVX2 ternary GEMM |
| `jit.rs` | 834 | Cranelift JIT engine |
| `plan.rs` | 675 | DAG plan executor |
| `crystal_compiler.rs` | 532 | Crystal weight compiler |
| `hir.rs` | 465 | High-Level IR |
| `lexer.rs` | 450 | Tokenizer |
| `vm.rs` | 370 | Tree-walk VM |
| `runtime.rs` | 354 | Async oracle engine |
| `units.rs` | 315 | Physical unit algebra |
| `typeck.rs` | 294 | Type checker |
| `aot_plan.rs` | 267 | AOT pre-compiled dispatch |
| `parser.rs` | 735 | Parser |
| `ast.rs` | 252 | AST definitions |
| `repl.rs` | 162 | Interactive REPL |
| `batch_oracle.rs` | 593 | Batch oracle operations |
| `cognitive_memory.rs` | 113 | Experience persistence |
| `main.rs` | 515 | CLI entry point |
| **Total** | **18,271** | — |
| `all_domains.crys` | 985 | Standard library |

---

## Appendix A: CLI Usage

```
qomn                              Interactive REPL
qomn run <file.crys>              Execute (tree-walk VM)
qomn run-jit <file.crys>          Execute with JIT oracle dispatch
qomn check <file.crys>            Type-check only
qomn lex <file.crys>              Dump tokens
qomn hir <file.crys>              Dump High-Level IR graph
qomn ir <file.crys>               Dump CRYS-ISA Bytecode
qomn jit <file.crys>              Compile oracles → native x86-64
qomn bench [rows] [cols]          AVX2 MM_TERN benchmark
qomn eval <expr>                  Evaluate inline expression
qomn compile <file.crys> [dir]    Compile oracle → .crystal (RFF PaO)
qomn serve <file.crys> [port]     Start HTTP API server
qomn plan <file.crys> <plan> ...  Execute named plan
qomn intent <file.crys> <query>   Parse intent and execute
```

## Appendix B: Deployment

```bash
# Build
cd /opt/qomn
cargo build --release

# Deploy
systemctl stop qomn-nfpa
cp target/release/qomn /usr/local/bin/qomn
systemctl start qomn-nfpa

# Verify
curl http://localhost:9001/health
# {"status":"ok","lang":"QOMN","version":"3.0","plans":55,"jit":true,"watchdog":"healthy"}
```

Service configuration:
```ini
[Unit]
Description=QOMN NFPA Computation Engine
After=network.target

[Service]
ExecStart=/usr/local/bin/qomn serve /opt/qomn/stdlib/all_domains.crys 9001
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

---

*QOMN is developed by Qomni AI Lab, a division of Condesi Perú.*
*Percy Rojas Masgo — CEO & Lead Architect*
*April 2026*

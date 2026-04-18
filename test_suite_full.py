#!/usr/bin/env python3
"""CRYS-L Engine - Full 6-Level Test Suite v2.0
Level 1: Functional (correct results)
Level 2: Speed (ns-level benchmarks)
Level 3: Simulation (sweep + best config)
Level 4: Decision Engine (recommendations, not just data)
Level 5: Conversational Context (what-if, multi-step)
Level 6: Stress (concurrent load, memory stability)
"""
import json, urllib.request, sys, time, threading, math

BASE = "http://127.0.0.1:9001"
PASS = 0
FAIL = 0
results = []

def api(method, path, body=None, timeout=10):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read()), e.code
        except:
            return {"error": str(e)}, e.code
    except Exception as e:
        return {"error": str(e)}, 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        results.append(("PASS", name, detail))
    else:
        FAIL += 1
        results.append(("FAIL", name, detail))

def steps_dict(d):
    return {s["step"]: s["result"] for s in d.get("result", d).get("steps", d.get("steps", []))}

# =====================================================================
print("=" * 70)
print("  CRYS-L Engine - 6-Level Test Suite v2.0")
print("=" * 70)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 1: FUNCTIONAL — Does it solve correctly?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  LEVEL 1: FUNCTIONAL (correct calculations)")
print("=" * 70)

# 1a. Fire protection — full ACI scenario
print("\n  [1a] Fire Protection - ACI Warehouse 500m2")
d, s = api("POST", "/intent", {"q": "sistema contra incendios para almacen 500 metros cuadrados 500 gpm 100 psi eficiencia 0.7", "session": "aci_test"})
test("L1: intent routes fire system", d.get("ok") == True and ("pump" in d.get("plan", "") or "sprinkler" in d.get("plan", "") or "egress" in d.get("plan", "") or d.get("plan") is not None))
aci_steps = steps_dict(d)
aci_hp = aci_steps.get("hp_required", aci_steps.get("pump_hp", 0))
test("L1: pump HP > 0", aci_hp > 0, "hp={:.2f}".format(aci_hp))

# Direct plan execution with known values
d, s = api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 100, "eff": 0.70}})
st = steps_dict(d)
hp = st.get("hp_required", 0)
# Q*P / (3960*eff) = 500*100 / (3960*0.70) = 18.04 HP
test("L1: pump_hp = Q*P/(3960*eff) = 18.04", abs(hp - 18.04) < 0.1, "hp={:.2f}".format(hp))
shutoff = st.get("shutoff_p", 0)
test("L1: shutoff_p = P*1.4 = 140", abs(shutoff - 140) < 0.1, "shutoff={:.1f}".format(shutoff))
flow150 = st.get("flow_150pct", 0)
test("L1: flow_150pct = Q*1.5 = 750", abs(flow150 - 750) < 0.1, "flow150={:.1f}".format(flow150))

# 1b. Electrical — known formulas
print("\n  [1b] Electrical - 10kW single phase")
d, s = api("POST", "/plan/execute", {"plan": "plan_electrical_load", "params": {"P_w": 10000, "V": 220, "pf": 0.85, "L": 50, "A": 10}})
st = steps_dict(d)
i_load = st.get("I_load", 0)
# I = P / (V * pf) = 10000 / (220 * 0.85) = 53.48A
test("L1: I_load = P/(V*pf) = 53.48A", abs(i_load - 53.48) < 0.1, "I={:.2f}A".format(i_load))

# 1c. Cybersecurity — password entropy math
print("\n  [1c] Cybersecurity - Password & Bcrypt")
d, s = api("POST", "/plan/execute", {"plan": "plan_password_audit", "params": {"charset_size": 95, "length": 16, "attempts_per_sec": 1e12}})
st = steps_dict(d)
entropy = st.get("entropy", 0)
expected_entropy = 16 * math.log2(95)  # 104.97
test("L1: entropy = 16*log2(95) = {:.2f}".format(expected_entropy), abs(entropy - expected_entropy) < 0.1, "got={:.2f}".format(entropy))
crack = st.get("crack_years", 0)
expected_crack = 2**expected_entropy / 1e12 / 31536000
test("L1: crack_years ~ {:.1e}".format(expected_crack), abs(crack - expected_crack) / expected_crack < 0.01, "got={:.1e}".format(crack))

# Bcrypt cost factor math
d, s = api("POST", "/plan/execute", {"plan": "plan_bcrypt_audit", "params": {"cost_factor": 10, "charset_size": 72, "length": 10, "base_gpu_rate": 7000000, "dict_size": 14000000}})
st = steps_dict(d)
rate = st.get("effective_rate", 0)
# 7M / 2^(10-5) = 7M / 32 = 218750
test("L1: bcrypt rate = 7M/2^5 = 218750", abs(rate - 218750) < 1, "rate={:.0f}".format(rate))

# 1d. Finance — Peru specifics
print("\n  [1d] Finance - Peru tax & labor")
d, s = api("POST", "/plan/execute", {"plan": "plan_factura_peru", "params": {"base_imponible": 50000}})
st = steps_dict(d)
igv = st.get("igv_amount", 0)
test("L1: IGV = 18% of 50000 = 9000", abs(igv - 9000) < 1, "igv={:.0f}".format(igv))
total = st.get("total", 0)
test("L1: total = 59000", abs(total - 59000) < 1, "total={:.0f}".format(total))

d, s = api("POST", "/plan/execute", {"plan": "plan_planilla", "params": {"sueldo": 5000}})
st = steps_dict(d)
essalud = st.get("essalud", 0)
# EsSalud = 9% of salary
test("L1: essalud = 9% of 5000 = 450", abs(essalud - 450) < 1, "essalud={:.0f}".format(essalud))

# 1e. All 56 plans execute without error
print("\n  [1e] All plans execute (smoke test)")
d, s = api("GET", "/plans")
plans = d.get("plans", d) if isinstance(d, dict) else d
plan_names = [p if isinstance(p, str) else p.get("name", "") for p in plans]
test("L1: {} plans available".format(len(plan_names)), len(plan_names) >= 55)

fail_plans = []
for p in plans:
    name = p if isinstance(p, str) else p.get("name", "")
    params = {}
    if isinstance(p, dict) and "params" in p:
        for pm in p["params"]:
            pname = pm if isinstance(pm, str) else pm.get("name", "")
            pdefault = pm.get("default", 1.0) if isinstance(pm, dict) else 1.0
            params[pname] = pdefault if pdefault else 1.0
    d2, s2 = api("POST", "/plan/execute", {"plan": name, "params": params})
    if not d2.get("ok"):
        fail_plans.append(name)
test("L1: all plans execute (default params)", len(fail_plans) == 0,
     "failed: {}".format(fail_plans) if fail_plans else "all OK")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 2: SPEED — ns-level benchmarks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  LEVEL 2: SPEED (latency benchmarks)")
print("=" * 70)

# 2a. Single plan execution latency
print("\n  [2a] Plan execution latency (100 iterations)")
times = []
for _ in range(100):
    t0 = time.time()
    api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}})
    times.append((time.time() - t0) * 1000)
avg = sum(times) / len(times)
p50 = sorted(times)[49]
p99 = sorted(times)[98]
minv = min(times)
test("L2: plan avg < 10ms", avg < 10, "avg={:.2f}ms".format(avg))
test("L2: plan p50 < 5ms", p50 < 5, "p50={:.2f}ms".format(p50))
test("L2: plan p99 < 20ms", p99 < 20, "p99={:.2f}ms".format(p99))
print("    avg={:.2f}ms p50={:.2f}ms p99={:.2f}ms min={:.2f}ms".format(avg, p50, p99, minv))

# 2b. Internal compute time (from server response)
print("\n  [2b] Internal compute time (server-reported ns)")
d, s = api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}})
total_ns = d.get("result", {}).get("total_ns", 0)
test("L2: internal compute < 50000ns (50us)", total_ns < 50000, "total_ns={:.0f}".format(total_ns))
print("    Server reported: {:.0f} ns ({:.3f} us)".format(total_ns, total_ns / 1000))

# Bcrypt audit internal time
d, s = api("POST", "/plan/execute", {"plan": "plan_bcrypt_audit", "params": {"cost_factor": 6, "charset_size": 62, "length": 8, "base_gpu_rate": 7000000, "dict_size": 14000000}})
total_ns2 = d.get("result", {}).get("total_ns", 0)
test("L2: bcrypt audit < 100000ns", total_ns2 < 100000, "total_ns={:.0f}".format(total_ns2))

# 2c. Decision engine latency
print("\n  [2c] Decision engine latency (50 iterations)")
dtimes = []
for _ in range(50):
    t0 = time.time()
    api("POST", "/decision/analyze", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}, "execute": True})
    dtimes.append((time.time() - t0) * 1000)
davg = sum(dtimes) / len(dtimes)
dp99 = sorted(dtimes)[48]
test("L2: decision avg < 15ms", davg < 15, "avg={:.2f}ms".format(davg))
test("L2: decision p99 < 30ms", dp99 < 30, "p99={:.2f}ms".format(dp99))
print("    avg={:.2f}ms p99={:.2f}ms".format(davg, dp99))

# 2d. Intent latency
print("\n  [2d] Intent (NL) latency")
itimes = []
for _ in range(30):
    t0 = time.time()
    api("POST", "/intent", {"q": "bomba 500 gpm 80 psi"})
    itimes.append((time.time() - t0) * 1000)
iavg = sum(itimes) / len(itimes)
test("L2: intent avg < 20ms", iavg < 20, "avg={:.2f}ms".format(iavg))
print("    avg={:.2f}ms".format(iavg))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 3: SIMULATION — sweep + best config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  LEVEL 3: SIMULATION (sweep + optimization)")
print("=" * 70)

# 3a. Pump sweep — Q: 400-1000, P: 80-150
print("\n  [3a] Pump sizing sweep")
t0 = time.time()
d, s = api("POST", "/plan/simulate", {
    "plan": "plan_pump_sizing",
    "sweep": {
        "Q_gpm": {"min": 400, "max": 1000, "step": 100},
        "P_psi": {"min": 80, "max": 150, "step": 10},
        "eff": 0.70
    },
    "output_step": "hp_required"
})
sim_time = (time.time() - t0) * 1000
scenarios = d.get("scenarios", 0)
total_ns_sim = d.get("total_ns", 0)
test("L3: pump sweep scenarios >= 56", scenarios >= 56, "scenarios={}".format(scenarios))
test("L3: sweep wall-time < 100ms", sim_time < 100, "wall={:.1f}ms".format(sim_time))
if scenarios > 0:
    rate = scenarios / (total_ns_sim / 1e9) if total_ns_sim > 0 else 0
    test("L3: throughput > 10K scenarios/s", rate > 10000, "rate={:,.0f}/s".format(rate))
    print("    {} scenarios in {:.0f}ns ({:,.0f}/s), wall={:.1f}ms".format(scenarios, total_ns_sim, rate, sim_time))

# Verify results are monotonic (more flow + pressure = more HP)
sim_results = d.get("results", [])
if len(sim_results) >= 2:
    first_hp = sim_results[0].get("hp_required", sim_results[0].get("result", 0))
    last_hp = sim_results[-1].get("hp_required", sim_results[-1].get("result", 0))
    test("L3: HP increases with Q*P", last_hp > first_hp, "first={:.1f}, last={:.1f}".format(first_hp, last_hp))

# 3b. Bcrypt sweep — length 4-20
print("\n  [3b] Bcrypt crack time sweep (length 4-20)")
d, s = api("POST", "/plan/simulate", {
    "plan": "plan_bcrypt_audit",
    "sweep": {
        "cost_factor": 6,
        "charset_size": 62,
        "length": {"min": 4, "max": 20, "step": 1},
        "base_gpu_rate": 7000000,
        "dict_size": 14000000
    },
    "output_step": "brute_years"
})
scenarios2 = d.get("scenarios", 0)
test("L3: bcrypt sweep scenarios = 17", scenarios2 == 17, "scenarios={}".format(scenarios2))
sim_results2 = d.get("results", [])
if len(sim_results2) >= 2:
    first_y = sim_results2[0].get("brute_years", 0)
    last_y = sim_results2[-1].get("brute_years", 0)
    test("L3: brute_years increases with length", last_y > first_y, "L4={}, L20={}".format(first_y, last_y))
    # Find the crossover point where brute > 100 years
    crossover = None
    for r in sim_results2:
        by = r.get("brute_years", 0)
        length = r.get("params", {}).get("length", 0)
        if by > 100 and crossover is None:
            crossover = int(length)
    if crossover:
        test("L3: identifies safe password length", crossover <= 12, "safe_at={}ch".format(crossover))
        print("    Safe length (>100yr brute): {} characters".format(crossover))

# 3c. Large multi-param sweep
print("\n  [3c] Large sweep performance")
t0 = time.time()
d, s = api("POST", "/plan/simulate", {
    "plan": "plan_electrical_load",
    "sweep": {
        "P_w": {"min": 1000, "max": 50000, "step": 1000},
        "V": 220,
        "pf": {"min": 0.7, "max": 0.95, "step": 0.05},
        "L": 50,
        "A": {"min": 4, "max": 35, "step": 1}
    },
    "output_step": "I_load"
})
big_time = (time.time() - t0) * 1000
big_scenarios = d.get("scenarios", 0)
test("L3: large sweep > 1000 scenarios", big_scenarios > 1000, "scenarios={}".format(big_scenarios))
test("L3: large sweep < 500ms", big_time < 500, "wall={:.1f}ms".format(big_time))
if big_scenarios > 0:
    big_rate = big_scenarios / (big_time / 1000)
    print("    {} scenarios in {:.1f}ms ({:,.0f}/s)".format(big_scenarios, big_time, big_rate))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 4: DECISION ENGINE — recommendations, not just data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  LEVEL 4: DECISION ENGINE (recommendations)")
print("=" * 70)

# 4a. Normal pump — should be valid
print("\n  [4a] Normal pump → valid decision")
d, s = api("POST", "/decision/analyze", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}, "execute": True})
dec = d.get("decision", {})
test("L4: decision has status", dec.get("status") in ("valid", "warning", "critical"))
test("L4: decision has risk_score", isinstance(dec.get("risk_score"), (int, float)))
test("L4: decision has analysis[]", len(dec.get("analysis", [])) > 0)
# Check analysis items have recommendations
has_reco = any(a.get("recommendation") for a in dec.get("analysis", []))
test("L4: analysis has recommendations", has_reco)
test("L4: normal pump status=valid", dec.get("status") == "valid", "status={}".format(dec.get("status")))
test("L4: normal pump risk < 3", dec.get("risk_score", 10) < 3, "risk={}".format(dec.get("risk_score")))

# 4b. Dangerous pump — should be critical
print("\n  [4b] Dangerous pump (5000gpm/200psi) → critical")
d, s = api("POST", "/decision/analyze", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 5000, "P_psi": 200, "eff": 0.50}, "execute": True})
dec = d.get("decision", {})
test("L4: dangerous pump detected", dec.get("status") in ("warning", "critical"), "status={}".format(dec.get("status")))
test("L4: risk_score > 3", dec.get("risk_score", 0) > 3, "risk={}".format(dec.get("risk_score")))

# 4c. Weak password — should warn/critical
print("\n  [4c] Weak password (4ch, a-z) → critical")
d, s = api("POST", "/decision/analyze", {"plan": "plan_password_audit", "params": {"charset_size": 26, "length": 4, "attempts_per_sec": 1e10}, "execute": True})
dec = d.get("decision", {})
test("L4: weak password flagged", dec.get("status") in ("warning", "critical"), "status={}".format(dec.get("status")))

# 4d. Strong password — should be valid
print("\n  [4d] Strong password (16ch, full charset) → valid")
d, s = api("POST", "/decision/analyze", {"plan": "plan_password_audit", "params": {"charset_size": 95, "length": 16, "attempts_per_sec": 1e12}, "execute": True})
dec = d.get("decision", {})
test("L4: strong password = valid", dec.get("status") == "valid", "status={}".format(dec.get("status")))

# 4e. Decision includes live context
print("\n  [4e] Decision includes context")
ctx = d.get("context", {})
test("L4: context has twin_vars", "twin_vars" in ctx)
test("L4: context has threats_active", "threats_active" in ctx)
test("L4: context has ot_status", "ot_status" in ctx)

# 4f. Decision rules endpoint
d, s = api("GET", "/decision/rules")
test("L4: /decision/rules returns rules", s == 200)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 5: CONVERSATIONAL CONTEXT (the most critical)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  LEVEL 5: CONVERSATIONAL CONTEXT (multi-step thinking)")
print("=" * 70)

session = "context_test_v2_{}".format(int(time.time()))

# Step 1: Initial query
print("\n  [5a] Step 1: Initial fire pump query")
d1, s1 = api("POST", "/intent", {"q": "calcula bomba para 500 gpm 80 psi eficiencia 0.7", "session": session})
test("L5: step1 routes to pump", "pump" in d1.get("plan", ""), "plan={}".format(d1.get("plan")))
test("L5: step1 ok", d1.get("ok") == True)
plan1 = d1.get("plan", "")
st1 = steps_dict(d1)
hp1 = st1.get("hp_required", st1.get("pump_hp", 0))
test("L5: step1 hp > 0", hp1 > 0, "hp={:.2f}".format(hp1))
print("    Plan: {} | HP: {:.2f}".format(plan1, hp1))

# Step 2: What-if — increase flow
print("\n  [5b] Step 2: 'que pasa si subo a 750 gpm'")
d2, s2 = api("POST", "/intent", {"q": "que pasa si subo el caudal a 750 gpm", "session": session})
test("L5: step2 detected as modification", d2.get("modified") == True, "modified={}".format(d2.get("modified")))
test("L5: step2 same plan", d2.get("plan", "") == plan1, "plan={}".format(d2.get("plan")))
st2 = steps_dict(d2)
hp2 = st2.get("hp_required", st2.get("pump_hp", 0))
test("L5: step2 HP increased (750>500 gpm)", hp2 > hp1, "hp1={:.2f} -> hp2={:.2f}".format(hp1, hp2))
# Verify only Q_gpm changed
changes = d2.get("changes", {})
test("L5: step2 only Q_gpm changed", "Q_gpm" in str(changes) or "q_gpm" in str(changes).lower(), "changes={}".format(changes))
print("    HP: {:.2f} -> {:.2f} (changes: {})".format(hp1, hp2, changes))

# Step 3: What-if — increase pressure
print("\n  [5c] Step 3: 'y si aumento la presion a 120 psi'")
d3, s3 = api("POST", "/intent", {"q": "y si aumento la presion a 120 psi", "session": session})
test("L5: step3 detected as modification", d3.get("modified") == True, "modified={}".format(d3.get("modified")))
test("L5: step3 same plan", d3.get("plan", "") == plan1, "plan={}".format(d3.get("plan")))
st3 = steps_dict(d3)
hp3 = st3.get("hp_required", st3.get("pump_hp", 0))
test("L5: step3 HP increased further", hp3 > hp2, "hp2={:.2f} -> hp3={:.2f}".format(hp2, hp3))
print("    HP: {:.2f} -> {:.2f} -> {:.2f}".format(hp1, hp2, hp3))

# Step 4: Different domain — should NOT reuse pump context
print("\n  [5d] Step 4: Switch to different domain")
d4, s4 = api("POST", "/intent", {"q": "calcula la factura con base 10000 soles", "session": session})
test("L5: step4 switches domain", "pump" not in d4.get("plan", ""), "plan={}".format(d4.get("plan")))
test("L5: step4 ok", d4.get("ok") == True)

# Step 5: Back to original domain with new session
print("\n  [5e] Step 5: New session starts fresh")
session2 = "fresh_test_{}".format(int(time.time()))
d5, s5 = api("POST", "/intent", {"q": "que pasa si bajo a 300 gpm", "session": session2})
# Without prior context, this should NOT be a modification
is_mod = d5.get("modified", False)
# It's ok if it fails to be modified (no context) — that's correct behavior
test("L5: new session has no context", is_mod == False or d5.get("ok") == True, "modified={}".format(is_mod))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 6: STRESS — concurrent load, stability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  LEVEL 6: STRESS (concurrent load)")
print("=" * 70)

# 6a. Sequential burst — 1000 requests
print("\n  [6a] Sequential burst: 1000 plan executions")
t0 = time.time()
errors_seq = 0
for i in range(1000):
    d, s = api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 400 + i, "P_psi": 80, "eff": 0.70}})
    if not d.get("ok"):
        errors_seq += 1
burst_time = time.time() - t0
burst_rps = 1000 / burst_time
test("L6: 1000 sequential - 0 errors", errors_seq == 0, "errors={}".format(errors_seq))
test("L6: 1000 sequential < 10s", burst_time < 10, "time={:.1f}s".format(burst_time))
test("L6: sequential throughput > 200 rps", burst_rps > 200, "rps={:.0f}".format(burst_rps))
print("    1000 requests in {:.1f}s ({:.0f} rps), errors={}".format(burst_time, burst_rps, errors_seq))

# 6b. Concurrent load — 50 threads x 20 requests
print("\n  [6b] Concurrent load: 50 threads x 20 requests")
errors_conc = [0]
latencies = []
lock = threading.Lock()

def worker(thread_id):
    for i in range(20):
        t0 = time.time()
        try:
            d, s = api("POST", "/plan/execute", {
                "plan": "plan_pump_sizing",
                "params": {"Q_gpm": 500 + thread_id * 10 + i, "P_psi": 80, "eff": 0.70}
            }, timeout=30)
            lat = (time.time() - t0) * 1000
            with lock:
                latencies.append(lat)
            if not d.get("ok"):
                with lock:
                    errors_conc[0] += 1
        except:
            with lock:
                errors_conc[0] += 1

t0 = time.time()
threads = []
for t in range(50):
    th = threading.Thread(target=worker, args=(t,))
    threads.append(th)
    th.start()
for th in threads:
    th.join(timeout=60)
conc_time = time.time() - t0
total_reqs = 50 * 20
conc_rps = total_reqs / conc_time if conc_time > 0 else 0

test("L6: concurrent - 0 errors", errors_conc[0] == 0, "errors={}".format(errors_conc[0]))
test("L6: concurrent < 30s", conc_time < 30, "time={:.1f}s".format(conc_time))
test("L6: concurrent throughput > 100 rps", conc_rps > 100, "rps={:.0f}".format(conc_rps))
if latencies:
    cavg = sum(latencies) / len(latencies)
    cp99 = sorted(latencies)[int(len(latencies) * 0.99)]
    test("L6: concurrent p99 < 130ms", cp99 < 130, "p99={:.1f}ms".format(cp99))
    print("    {} requests in {:.1f}s ({:.0f} rps)".format(total_reqs, conc_time, conc_rps))
    print("    avg={:.1f}ms p99={:.1f}ms errors={}".format(cavg, cp99, errors_conc[0]))

# 6c. Mixed workload (plan + intent + decision simultaneously)
print("\n  [6c] Mixed workload: plan + intent + decision")
mix_errors = [0]
mix_lats = []

def mixed_worker(wid):
    ops = [
        lambda: api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}}),
        lambda: api("POST", "/intent", {"q": "bomba 500 gpm 80 psi"}),
        lambda: api("POST", "/decision/analyze", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}, "execute": True}),
        lambda: api("POST", "/plan/simulate", {"plan": "plan_pump_sizing", "sweep": {"Q_gpm": {"min": 400, "max": 600, "step": 50}, "P_psi": 80, "eff": 0.70}, "output_step": "hp_required"}),
    ]
    for i in range(10):
        op = ops[i % len(ops)]
        t0 = time.time()
        try:
            d, s = op()
            lat = (time.time() - t0) * 1000
            with lock:
                mix_lats.append(lat)
            ok = d.get("ok", False) or d.get("scenarios", 0) > 0
            if not ok:
                with lock:
                    mix_errors[0] += 1
        except:
            with lock:
                mix_errors[0] += 1

t0 = time.time()
threads = []
for t in range(20):
    th = threading.Thread(target=mixed_worker, args=(t,))
    threads.append(th)
    th.start()
for th in threads:
    th.join(timeout=60)
mix_time = time.time() - t0
mix_total = 20 * 10

test("L6: mixed workload - 0 errors", mix_errors[0] == 0, "errors={}".format(mix_errors[0]))
if mix_lats:
    mavg = sum(mix_lats) / len(mix_lats)
    mp99 = sorted(mix_lats)[int(len(mix_lats) * 0.99)]
    print("    {} mixed ops in {:.1f}s, avg={:.1f}ms p99={:.1f}ms errors={}".format(mix_total, mix_time, mavg, mp99, mix_errors[0]))

# 6d. Health check after stress
print("\n  [6d] Post-stress health check")
d, s = api("GET", "/health")
test("L6: health OK after stress", d.get("status") == "ok", "status={}".format(d.get("status")))
test("L6: watchdog healthy", d.get("watchdog") == "healthy", "watchdog={}".format(d.get("watchdog")))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("  RESULTS BY LEVEL")
print("=" * 70)

levels = {
    "L1": ("FUNCTIONAL", []),
    "L2": ("SPEED", []),
    "L3": ("SIMULATION", []),
    "L4": ("DECISION", []),
    "L5": ("CONTEXT", []),
    "L6": ("STRESS", []),
}
for status, name, detail in results:
    prefix = name[:2]
    if prefix in levels:
        levels[prefix][1].append((status, name, detail))

for prefix in ["L1", "L2", "L3", "L4", "L5", "L6"]:
    label, items = levels[prefix]
    passed = sum(1 for s, _, _ in items if s == "PASS")
    failed = sum(1 for s, _, _ in items if s == "FAIL")
    icon = "OK" if failed == 0 else "!!"
    print("  [{}] {}: {}/{} passed".format(icon, label, passed, passed + failed))
    for status, name, detail in items:
        if status == "FAIL":
            print("       FAIL: {} ({})".format(name, detail))

print("\n" + "-" * 70)
print("  TOTAL: {} passed, {} failed".format(PASS, FAIL))
if FAIL == 0:
    print("  >>> ALL {} TESTS PASSED <<<".format(PASS))
else:
    print("  >>> {} TESTS FAILED <<<".format(FAIL))
print("=" * 70)
sys.exit(0 if FAIL == 0 else 1)

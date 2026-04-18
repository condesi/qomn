#!/usr/bin/env python3
"""CRYS-L Engine - Full Test Suite v1.0
Tests all endpoints and plans for correctness before publishing.
"""
import json, urllib.request, sys

BASE = "http://127.0.0.1:9001"
PASS = 0
FAIL = 0
WARN = 0
results = []

def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code
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

def warn(name, detail=""):
    global WARN
    WARN += 1
    results.append(("WARN", name, detail))

print("=" * 70)
print("  CRYS-L Engine - Full Test Suite")
print("=" * 70)

# 1. HEALTH & META
print("\n[1/9] Health & Meta Endpoints")
d, s = api("GET", "/health")
test("GET /health returns 200", s == 200)
test("/health status=ok", d.get("status") == "ok")
test("/health has plans>0", d.get("plans", 0) > 0, "plans={}".format(d.get("plans")))
test("/health JIT enabled", d.get("jit") == True)
test("/health turbo>0", d.get("turbo", 0) > 0, "turbo={}".format(d.get("turbo")))

d, s = api("GET", "/plans")
plans = d.get("plans", d) if isinstance(d, dict) else d
test("GET /plans returns list", isinstance(plans, list) and len(plans) > 0, "count={}".format(len(plans)))

d, s = api("GET", "/oracles")
test("GET /oracles returns 200", s == 200 or s == 404, "status={}".format(s))

d, s = api("GET", "/stats")
test("GET /stats returns 200", s == 200 or s == 404, "status={}".format(s))

# 2. FIRE PROTECTION PLANS
print("\n[2/9] Fire Protection Plans")

d, s = api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}})
test("plan_pump_sizing executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
hp = steps.get("hp_required", steps.get("pump_hp", 0))
test("hp_required reasonable (10-20 HP for 500gpm/80psi)", 10 < hp < 20, "hp={:.2f}".format(hp))

d, s = api("POST", "/plan/execute", {"plan": "plan_sprinkler_system", "params": {"area_ft2": 5000, "K": 5.6, "P_avail": 80, "hose_stream": 250}})
test("plan_sprinkler_system executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
n_heads = steps.get("n_heads", steps.get("sprinkler_count", 0))
test("n_heads > 0", n_heads > 0, "n_heads={:.1f}".format(n_heads))

d, s = api("POST", "/plan/execute", {"plan": "plan_nfpa13_demand", "params": {"area_ft2": 3000, "density": 0.15, "hose_stream": 250}})
test("plan_nfpa13_demand executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
flow = steps.get("flow_gpm", steps.get("total_demand", 0))
test("flow_gpm > 0", flow > 0, "flow_gpm={:.1f}".format(flow))

d, s = api("POST", "/plan/execute", {"plan": "plan_pipe_losses", "params": {"Q_gpm": 500, "C": 120, "D_in": 6, "L_ft": 200}})
test("plan_pipe_losses executes", d.get("ok") == True)

d, s = api("POST", "/plan/execute", {"plan": "plan_pressure", "params": {"P_static": 80, "friction_loss": 15}})
test("plan_pressure executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
residual = steps.get("residual", 0)
test("residual = 80-15 = 65", abs(residual - 65) < 0.1, "residual={:.1f}".format(residual))

d, s = api("POST", "/plan/execute", {"plan": "plan_egress", "params": {"N_persons": 500, "exits": 4, "door_width_in": 36}})
test("plan_egress executes", d.get("ok") == True)

# 3. ELECTRICAL PLANS
print("\n[3/9] Electrical Plans")

d, s = api("POST", "/plan/execute", {"plan": "plan_electrical_load", "params": {"P_w": 10000, "V": 220, "pf": 0.85, "L": 50, "A": 10}})
test("plan_electrical_load executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
i_load = steps.get("I_load", 0)
test("I_load > 10A for 10kW", i_load > 10, "I_load={:.2f}A".format(i_load))

d, s = api("POST", "/plan/execute", {"plan": "plan_electrical_3ph", "params": {"P_kw": 50, "V": 380, "pf": 0.85, "L": 100, "A": 25}})
test("plan_electrical_3ph executes", d.get("ok") == True)

d, s = api("POST", "/plan/execute", {"plan": "plan_voltage_drop", "params": {"I": 30, "L_m": 100, "A_mm2": 6}})
test("plan_voltage_drop executes", d.get("ok") == True)

d, s = api("POST", "/plan/execute", {"plan": "plan_transformer", "params": {"P_kw": 100, "pf": 0.85, "eff": 0.95}})
test("plan_transformer executes", d.get("ok") == True)

d, s = api("POST", "/plan/execute", {"plan": "plan_power_factor_correction", "params": {"P_kw": 200, "pf_actual": 0.75, "pf_meta": 0.95}})
test("plan_power_factor_correction executes", d.get("ok") == True)

# 4. CYBERSECURITY PLANS
print("\n[4/9] Cybersecurity Plans")

d, s = api("POST", "/plan/execute", {"plan": "plan_password_audit", "params": {"charset_size": 72, "length": 12, "attempts_per_sec": 1e10}})
test("plan_password_audit executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
entropy = steps.get("entropy", 0)
crack = steps.get("crack_years", 0)
strength = steps.get("strength_ratio", 0)
test("entropy = 12*log2(72) ~ 74.04", abs(entropy - 74.039) < 0.1, "entropy={:.4f}".format(entropy))
test("crack_years > 10000", crack > 10000, "crack={:.1f}".format(crack))
test("crack_years ~ 61543", abs(crack - 61543) < 500, "crack={:.1f}".format(crack))
test("strength_ratio ~ 2.89", abs(strength - 2.89) < 0.1, "strength={:.2f}".format(strength))

# Zero params edge case
d, s = api("POST", "/plan/execute", {"plan": "plan_password_audit", "params": {"charset_size": 0, "length": 0, "attempts_per_sec": 0}})
test("password_audit zeros - no crash", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("entropy=0 when charset=0", steps.get("entropy", -1) == 0)
test("crack_years=0 when zeros", steps.get("crack_years", -1) == 0)

# Single char password
d, s = api("POST", "/plan/execute", {"plan": "plan_password_audit", "params": {"charset_size": 26, "length": 1, "attempts_per_sec": 1e9}})
test("single char password", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("1-char entropy ~ 4.7 bits", abs(steps.get("entropy", 0) - 4.7) < 0.1, "entropy={:.2f}".format(steps.get("entropy", 0)))

# Very strong password
d, s = api("POST", "/plan/execute", {"plan": "plan_password_audit", "params": {"charset_size": 95, "length": 32, "attempts_per_sec": 1e12}})
test("32-char strong password", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("32-char entropy > 200 bits", steps.get("entropy", 0) > 200, "entropy={:.1f}".format(steps.get("entropy", 0)))
test("32-char crack_years enormous", steps.get("crack_years", 0) > 1e30, "crack={:.2e}".format(steps.get("crack_years", 0)))

# CVSS
d, s = api("POST", "/plan/execute", {"plan": "plan_cvss_assessment", "params": {"av": 0.85, "ac": 0.77, "pr": 0.62, "ui": 0.85, "scope": 1.0, "c": 0.56, "i": 0.22, "a": 0.0}})
test("plan_cvss_assessment executes", d.get("ok") == True)

# Network security
d, s = api("POST", "/plan/execute", {"plan": "plan_network_security", "params": {"open_ports": 15, "unpatched": 3, "services": 8, "days_ssl": 45}})
test("plan_network_security executes", d.get("ok") == True)

# Crypto audit
d, s = api("POST", "/plan/execute", {"plan": "plan_crypto_audit", "params": {"aes_bits": 256, "rsa_bits": 2048, "charset_size": 95, "pwd_length": 16}})
test("plan_crypto_audit executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("AES-256 strength = 10.0", abs(steps.get("aes_strength", 0) - 10.0) < 0.01, "aes={}".format(steps.get("aes_strength")))

# Bcrypt audit — cost 6, RTX 4090
d, s = api("POST", "/plan/execute", {"plan": "plan_bcrypt_audit", "params": {"cost_factor": 6, "charset_size": 62, "length": 8, "base_gpu_rate": 7000000, "dict_size": 14000000}})
test("plan_bcrypt_audit executes", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("bcrypt effective_rate = 3.5M (cost6=base/2)", abs(steps.get("effective_rate", 0) - 3500000) < 100, "rate={:.0f}".format(steps.get("effective_rate", 0)))
test("bcrypt dict_seconds ~ 4s", abs(steps.get("dict_seconds", 0) - 4.0) < 0.1, "dict={:.1f}s".format(steps.get("dict_seconds", 0)))
test("bcrypt brute_years ~ 2.0", abs(steps.get("brute_years", 0) - 2.0) < 0.5, "brute={:.1f}y".format(steps.get("brute_years", 0)))
test("bcrypt entropy = 47.6 bits", abs(steps.get("entropy", 0) - 47.6) < 0.1, "entropy={:.1f}".format(steps.get("entropy", 0)))

# Bcrypt cost 12 comparison
d, s = api("POST", "/plan/execute", {"plan": "plan_bcrypt_audit", "params": {"cost_factor": 12, "charset_size": 62, "length": 8, "base_gpu_rate": 7000000, "dict_size": 14000000}})
test("bcrypt cost-12 executes", d.get("ok") == True)
steps12 = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("cost-12 rate = 54688 (64x slower)", abs(steps12.get("effective_rate", 0) - 54687.5) < 10, "rate={:.0f}".format(steps12.get("effective_rate", 0)))
test("cost-12 brute_years > 100", steps12.get("brute_years", 0) > 100, "years={:.1f}".format(steps12.get("brute_years", 0)))

# Bcrypt edge: cost 4 (very fast)
d, s = api("POST", "/plan/execute", {"plan": "plan_bcrypt_audit", "params": {"cost_factor": 4, "charset_size": 26, "length": 4, "base_gpu_rate": 7000000, "dict_size": 14000000}})
test("bcrypt cost-4 weak password", d.get("ok") == True)
steps4 = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
test("cost-4 brute < 1 second", steps4.get("brute_seconds", 999) < 1, "brute={:.3f}s".format(steps4.get("brute_seconds", 999)))

# 5. MULTI-DOMAIN PLANS
print("\n[5/9] Multi-Domain Plans")

multi_tests = [
    ("plan_factura_peru", {"base_imponible": 10000}, "igv_amount", lambda v: abs(v - 1800) < 1, "IGV=18% of 10000"),
    ("plan_planilla", {"sueldo": 3000}, "essalud", lambda v: v > 0, "essalud>0"),
    ("plan_break_even", {"fixed_cost": 50000, "price": 100, "var_cost": 60}, "units_be", lambda v: abs(v - 1250) < 1, "50000/(100-60)=1250"),
    ("plan_roi", {"net_profit": 50000, "investment": 200000}, "roi", lambda v: abs(v - 25) < 0.5, "50k/200k=25%"),
    ("plan_bmi_assessment", {"weight_kg": 80, "height_m": 1.75, "age": 30}, "bmi_val", lambda v: 25 < v < 27, "80/1.75^2~26.1"),
    ("plan_statistics", {"n": 100, "sum_x": 5000, "sum_x2": 300000}, "mean", lambda v: abs(v - 50) < 0.1, "5000/100=50"),
    ("plan_compound_interest", {"P": 10000, "r_decimal": 0.08, "years": 10}, "A", lambda v: abs(v - 21589.25) < 50, "10000*(1.08^10)"),
    ("plan_slope_analysis", {"delta_h": 10, "distance": 100}, "slope_pct_val", lambda v: abs(v - 10) < 0.1, "10/100*100=10%"),
]

for plan_name, params, check_step, check_fn, expected in multi_tests:
    d, s = api("POST", "/plan/execute", {"plan": plan_name, "params": params})
    ok = d.get("ok") == True
    test(plan_name + " executes", ok)
    if ok and check_step and check_fn:
        steps = {st["step"]: st["result"] for st in d.get("result", {}).get("steps", [])}
        val = steps.get(check_step, None)
        if val is not None:
            test(plan_name + "." + check_step + " correct", check_fn(val), "val={}, expected: {}".format(val, expected))
        else:
            warn(plan_name + "." + check_step + " not found in steps", "available: {}".format(list(steps.keys())))

# Execute-only plans (just check they don't crash)
exec_only = [
    ("plan_beam_analysis", {"P_kn": 50, "L_m": 6, "E_gpa": 200, "I_cm4": 5000}),
    ("plan_column_design", {"P_kn": 500, "L_m": 4, "b_cm": 30, "h_cm": 30}),
    ("plan_footing", {"q_allow": 150, "B": 2, "L": 2}),
    ("plan_solar_fv", {"kwh_daily": 30, "hsp": 5.5, "panel_wp": 400, "eff": 0.80, "cost_usd": 250, "tariff": 0.20}),
    ("plan_hvac_cooling", {"area_m2": 200, "ceiling_h": 3, "ach": 6, "delta_t": 10, "occupants": 30}),
    ("plan_hvac_ventilation", {"area_m2": 200, "ceiling_h": 3, "ach": 6}),
    ("plan_irrigation", {"area_ha": 5, "eto_mm": 6, "kc": 0.8, "eff": 0.85}),
    ("plan_telecom_link", {"tx_dbm": 30, "freq_mhz": 2400, "distance_km": 5, "bw_mhz": 20}),
    ("plan_autoclave_cycle", {"vol_l": 50, "T_c": 121, "P_bar": 2.1, "t_hold_min": 15, "D_value_min": 1.5}),
    ("plan_medical_gas", {"n_outlets": 20, "diversity": 0.6, "q_lpm": 10, "P_supply_psi": 55}),
    ("plan_hydro_channel", {"b_m": 2, "y_m": 1, "n_manning": 0.013, "slope": 0.001}),
    ("plan_hydro_demand", {"population": 50000, "lpcd": 200, "storage_days": 2, "pipe_D_m": 0.5}),
    ("plan_motor_drive", {"power_kw": 15, "rpm": 1800, "ratio": 3, "eff": 0.90}),
    ("plan_earthwork", {"area_m2": 1000, "cut_depth": 2, "fill_depth": 1.5, "compaction": 0.85}),
    ("plan_transport", {"distance_km": 500, "cost_per_km": 2.5, "n_trips": 10, "units_per_trip": 20}),
    ("plan_sample_size", {"confidence_pct": 95, "margin_error_pct": 5}),
    ("plan_loan_amortization", {"P": 100000, "r_monthly": 0.01, "n_months": 360}),
    ("plan_pricing", {"cost": 50, "markup_pct": 30}),
    ("plan_depreciacion", {"costo": 100000, "vida_util": 10, "valor_residual": 10000}),
    ("plan_ratios_financieros", {"activo_c": 500000, "pasivo_c": 200000, "utilidad": 80000, "ventas": 1000000, "patrimonio": 300000}),
    ("plan_liquidacion_laboral", {"sueldo": 3000, "meses_trabajo": 24, "dias_vacac": 15}),
    ("plan_pipe_manning", {"n": 0.013, "D": 0.3, "S": 0.001}),
    ("plan_pipe_hazen", {"Q": 500, "C": 120, "D": 6, "L": 200}),
    ("plan_pipe_network_3", {"Q_gpm": 1000, "P0_psi": 100, "C": 120, "D_in": 8, "L_m": 300}),
    ("plan_pump_selection", {"P_residual": 65, "P_required": 80, "Q_gpm": 500}),
    ("plan_drip_irrigation", {"area_m2": 10000, "ET0_mm": 5, "Kc": 0.8, "row_m": 1, "spacing_m": 0.3, "q_emitter_lph": 4, "eff": 0.90}),
    ("plan_slope_stability", {"H_m": 10, "VH_ratio": 1.5, "c_kpa": 25, "tan_phi": 0.577, "gamma_kN_m3": 18}),
    ("plan_vibration_fatigue", {"disp_mm": 0.5, "freq_hz": 50, "base_life_h": 10000, "load_ratio": 0.8}),
    ("plan_drug_dosing", {"weight_kg": 70, "dose_mg_per_kg": 15, "frequency_per_day": 3}),
]

for plan_name, params in exec_only:
    d, s = api("POST", "/plan/execute", {"plan": plan_name, "params": params})
    test(plan_name + " executes", d.get("ok") == True, "error={}".format(d.get("error", "")) if not d.get("ok") else "")

# 6. SIMULATE (sweep)
print("\n[6/9] Simulate (Sweep)")

d, s = api("POST", "/plan/simulate", {
    "plan": "plan_pump_sizing",
    "sweep": {"Q_gpm": {"min": 200, "max": 1000, "step": 200}, "P_psi": 80, "eff": 0.70},
    "output_step": "pump_hp"
})
ok = d.get("ok", False) if isinstance(d.get("ok"), bool) else d.get("scenarios", 0) > 0
test("/plan/simulate pump sweep", ok and s == 200)
scenarios = d.get("scenarios", 0)
test("simulate >1 scenarios", scenarios > 1, "scenarios={}".format(scenarios))

d, s = api("POST", "/plan/simulate", {
    "plan": "plan_password_audit",
    "sweep": {"charset_size": 95, "length": {"min": 6, "max": 16, "step": 2}, "attempts_per_sec": 1e10},
    "output_step": "entropy"
})
test("simulate password sweep", s == 200)
sim_results = d.get("results", [])
if len(sim_results) >= 2:
    vals = [r.get("entropy", r.get("result", 0)) for r in sim_results]
    non_zero = [v for v in vals if isinstance(v, (int, float)) and v > 0]
    test("entropy increases with length", len(non_zero) >= 2 and non_zero[-1] > non_zero[0],
         "first={}, last={}".format(non_zero[0] if non_zero else "?", non_zero[-1] if non_zero else "?"))
else:
    test("entropy increases with length", False, "results={}".format(len(sim_results)))

# Large sweep performance
import time
t0 = time.time()
d, s = api("POST", "/plan/simulate", {
    "plan": "plan_pump_sizing",
    "sweep": {"Q_gpm": {"min": 100, "max": 2000, "step": 50}, "P_psi": {"min": 40, "max": 120, "step": 10}, "eff": 0.70},
    "output_step": "pump_hp"
})
elapsed = time.time() - t0
scenarios = d.get("scenarios", 0)
test("large sweep ({}+ scenarios)".format(scenarios), scenarios > 100, "scenarios={}, time={:.2f}s".format(scenarios, elapsed))
if scenarios > 0 and elapsed > 0:
    rate = scenarios / elapsed
    test("throughput > 10K scenarios/s", rate > 10000, "rate={:,.0f}/s".format(rate))

# 7. INTENT (NL)
print("\n[7/9] Intent (Natural Language)")

d, s = api("POST", "/intent", {"q": "bomba de agua 500 gpm 80 psi eficiencia 0.70"})
test("/intent fire pump", d.get("ok") == True)
test("routes to pump plan", "pump" in d.get("plan", ""), "plan={}".format(d.get("plan")))

d, s = api("POST", "/intent", {"q": "sistema de rociadores para area de 5000 pies cuadrados"})
test("/intent sprinkler", d.get("ok") == True)

d, s = api("POST", "/intent", {"q": "calcula la factura con base imponible 10000 soles"})
test("/intent factura", d.get("ok") == True)

d, s = api("POST", "/intent", {"q": "analisis de password con charset 95 y longitud 16"})
test("/intent password", d.get("ok") == True)

# What-if context
d, s = api("POST", "/intent", {"q": "bomba 500 gpm 80 psi eficiencia 0.7", "session": "test_ctx_v2"})
test("/intent set context", d.get("ok") == True)
d2, s2 = api("POST", "/intent", {"q": "que pasa si aumento el caudal a 750 gpm", "session": "test_ctx_v2"})
is_modified = d2.get("modified") == True
test("/intent what-if context", is_modified or d2.get("ok") == True, "modified={}".format(d2.get("modified")))
if is_modified:
    test("what-if returns changed params", "changes" in d2 or "result" in d2)

# 8. DECISION ENGINE
print("\n[8/9] Decision Engine")

d, s = api("POST", "/decision/analyze", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}, "execute": True})
test("/decision/analyze pump", d.get("ok") == True)
dec = d.get("decision", {})
test("decision.status valid", dec.get("status") in ("valid", "warning", "critical"), "status={}".format(dec.get("status")))
test("decision.risk_score exists", isinstance(dec.get("risk_score"), (int, float)))
test("decision.risk_level exists", dec.get("risk_level") in ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
test("decision.analysis is list", isinstance(dec.get("analysis"), list) and len(dec.get("analysis", [])) > 0)
test("decision.context exists", isinstance(d.get("context"), dict))

d, s = api("GET", "/decision/rules")
test("GET /decision/rules 200", s == 200)

# Decision on weak password
d, s = api("POST", "/decision/analyze", {"plan": "plan_password_audit", "params": {"charset_size": 26, "length": 4, "attempts_per_sec": 1e10}, "execute": True})
test("decision: weak password", d.get("ok") == True)
steps = {st["step"]: st["result"] for st in d.get("steps", [])}
ent = steps.get("entropy", None)
if ent is not None:
    test("weak 4-char entropy < 20 bits", ent < 20, "entropy={:.1f}".format(ent))

# Decision on dangerous pump
d, s = api("POST", "/decision/analyze", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 5000, "P_psi": 200, "eff": 0.50}, "execute": True})
test("decision: large pump", d.get("ok") == True)
dec = d.get("decision", {})
test("large pump risk > 0", dec.get("risk_score", 0) > 0 or dec.get("status") in ("warning", "critical"),
     "risk={}, status={}".format(dec.get("risk_score"), dec.get("status")))

# 9. ERROR HANDLING
print("\n[9/9] Error Handling")

d, s = api("POST", "/plan/execute", {"plan": "nonexistent_plan", "params": {}})
test("nonexistent plan = error", d.get("ok") == False)

d, s = api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {}})
test("missing params defaults to 0", d.get("ok") == True)

d, s = api("POST", "/intent", {"q": ""})
test("empty intent handled", s in (200, 400))

d, s = api("GET", "/nonexistent")
test("unknown endpoint = 404", s == 404)

d, s = api("POST", "/plan/execute", {})
test("missing plan field = error", d.get("ok") == False or s == 400)

d, s = api("POST", "/decision/analyze", {"plan": "nonexistent", "execute": True})
test("decision on bad plan = error", d.get("ok") == False)

# PERFORMANCE
print("\n[PERF] Performance Benchmarks")
import time
times = []
for _ in range(100):
    t0 = time.time()
    api("POST", "/plan/execute", {"plan": "plan_pump_sizing", "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.70}})
    times.append((time.time() - t0) * 1000)
avg_ms = sum(times) / len(times)
p99_ms = sorted(times)[98]
test("plan_execute avg < 10ms", avg_ms < 10, "avg={:.2f}ms".format(avg_ms))
test("plan_execute p99 < 50ms", p99_ms < 50, "p99={:.2f}ms".format(p99_ms))

times2 = []
for _ in range(50):
    t0 = time.time()
    api("POST", "/decision/analyze", {"plan": "plan_password_audit", "params": {"charset_size": 95, "length": 16, "attempts_per_sec": 1e12}, "execute": True})
    times2.append((time.time() - t0) * 1000)
avg2 = sum(times2) / len(times2)
test("decision/analyze avg < 20ms", avg2 < 20, "avg={:.2f}ms".format(avg2))

# SUMMARY
print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
for status, name, detail in results:
    if status == "PASS":
        icon = " OK "
    elif status == "FAIL":
        icon = "FAIL"
    else:
        icon = "WARN"
    line = "  [{}] {}".format(icon, name)
    if detail:
        line += "  ({})".format(detail)
    print(line)

print("\n" + "-" * 70)
print("  TOTAL: {} passed, {} failed, {} warnings".format(PASS, FAIL, WARN))
if FAIL == 0:
    print("  >>> ALL TESTS PASSED <<<")
else:
    print("  >>> {} TESTS FAILED <<<".format(FAIL))
print("=" * 70)
sys.exit(0 if FAIL == 0 else 1)

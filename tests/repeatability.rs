//! CRYS-L Repeatability & Determinism Tests
//!
//! Validates the core CRYS-L guarantee: **same input → same output, always**.
//! These tests are the empirical basis for the determinism claims in the paper.
//!
//! Run with: `cargo test --test repeatability -- --nocapture`
//!
//! Paper reference: §4.2 "Deterministic Execution Guarantee"

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

// ── HTTP helpers ─────────────────────────────────────────────────────────────

fn post(path: &str, body: &str) -> String {
    let mut stream = TcpStream::connect("127.0.0.1:9001").expect("CRYS-L server not running");
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path, body.len(), body
    );
    stream.write_all(req.as_bytes()).unwrap();
    stream.shutdown(std::net::Shutdown::Write).ok();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).unwrap();
    buf.find("\r\n\r\n").map(|p| buf[p + 4..].to_string()).unwrap_or(buf)
}

fn extract_f64(json: &str, key: &str) -> f64 {
    let pattern = format!("\"{}\":", key);
    json.find(&pattern).and_then(|pos| {
        let after = json[pos + pattern.len()..].trim_start();
        let end = after.find(|c: char| c != '-' && c != '.' && !c.is_ascii_digit()).unwrap_or(after.len());
        after[..end].parse::<f64>().ok()
    }).unwrap_or(f64::NAN)
}

fn extract_bool(json: &str, key: &str) -> bool {
    let pattern = format!("\"{}\":", key);
    json.find(&pattern).map(|pos| {
        json[pos + pattern.len()..].trim_start().starts_with("true")
    }).unwrap_or(false)
}

// ── Core determinism test helper ──────────────────────────────────────────────

struct RepeatResult {
    values: Vec<f64>,
    timings_us: Vec<f64>,
    identical_count: usize,
}

fn run_n_times(plan: &str, params: &str, key: &str, n: usize) -> RepeatResult {
    let body = format!(r#"{{"plan":"{}","params":{}}}"#, plan, params);
    let mut values = Vec::with_capacity(n);
    let mut timings_us = Vec::with_capacity(n);

    for _ in 0..n {
        let t0 = Instant::now();
        let resp = post("/plan/execute", &body);
        let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
        let val = extract_f64(&resp, key);
        if !val.is_nan() {
            values.push(val);
        }
        timings_us.push(elapsed_us);
    }

    let unique: HashSet<String> = values.iter().map(|v| format!("{:.8}", v)).collect();
    let identical_count = if unique.len() == 1 { values.len() } else { 0 };

    RepeatResult { values, timings_us, identical_count }
}

// ── Test: Fire pump sizing — 10 runs ─────────────────────────────────────────

#[test]
fn fire_pump_determinism_10_runs() {
    let r = run_n_times(
        "plan_pump_sizing",
        r#"{"Q_gpm":500,"P_psi":100,"eff":0.75}"#,
        "nfpa20_pump_hp",
        10,
    );

    assert!(r.values.len() >= 9, "Too many failures: only {} responses", r.values.len());

    let variance = r.values.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Pump HP values: {:?}", r.values);
    println!("  Variance: {:.10}", variance);
    println!("  Identical: {}/{}", r.identical_count, r.values.len());

    assert_eq!(variance, 0.0,
        "Non-determinism detected! HP values varied by {:.10} over {} runs",
        variance, r.values.len()
    );
    assert_eq!(r.identical_count, r.values.len(),
        "Expected {} identical outputs, got {}", r.values.len(), r.identical_count
    );
}

// ── Test: Electrical load — 20 runs ─────────────────────────────────────────

#[test]
fn electrical_load_determinism_20_runs() {
    let r = run_n_times(
        "plan_electrical_load",
        r#"{"P_w":5000,"V":220,"pf":0.92,"L":50,"A":4}"#,
        "current_a",
        20,
    );

    let variance = r.values.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Current values (20 runs): {} unique values", {
        let u: HashSet<String> = r.values.iter().map(|v| format!("{:.8}", v)).collect();
        u.len()
    });
    println!("  Variance: {:.12}", variance);

    assert_eq!(variance, 0.0,
        "Electrical load non-determinism: variance={:.12}", variance);
}

// ── Test: Structural beam — 15 runs ─────────────────────────────────────────

#[test]
fn beam_analysis_determinism_15_runs() {
    let r = run_n_times(
        "plan_beam_analysis",
        r#"{"P_kn":50,"L_m":6,"E_gpa":200,"I_cm4":8000}"#,
        "deflection_mm",
        15,
    );

    let variance = r.values.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Deflection mm (15 runs): first={:.8}, last={:.8}", r.values[0], r.values[r.values.len()-1]);
    println!("  Variance: {:.12}", variance);

    assert_eq!(variance, 0.0, "Beam deflection non-determinism: variance={:.12}", variance);
}

// ── Test: Financial — 10 runs ────────────────────────────────────────────────

#[test]
fn financial_planilla_determinism_10_runs() {
    let r = run_n_times(
        "plan_planilla",
        r#"{"salario_bruto":5000,"horas_extras":10,"regimen":"general"}"#,
        "neto",
        10,
    );

    let variance = r.values.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Neto payroll (10 runs): {:.4}", r.values.get(0).copied().unwrap_or(f64::NAN));
    println!("  Variance: {:.12}", variance);

    assert_eq!(variance, 0.0, "Planilla non-determinism: variance={:.12}", variance);
}

// ── Test: NFPA 13 sprinkler — 10 runs ───────────────────────────────────────

#[test]
fn nfpa13_determinism_10_runs() {
    let r = run_n_times(
        "plan_nfpa13_demand",
        r#"{"area_ft2":1500,"density":0.15,"K":5.6,"hose_stream":250}"#,
        "total_flow_gpm",
        10,
    );

    let variance = r.values.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  NFPA13 total_flow (10 runs): {:.4}", r.values.get(0).copied().unwrap_or(f64::NAN));
    println!("  Variance: {:.12}", variance);

    assert_eq!(variance, 0.0, "NFPA13 non-determinism: variance={:.12}", variance);
}

// ── Test: Cross-domain determinism — 5 plans × 5 runs ───────────────────────

#[test]
fn cross_domain_determinism_5x5() {
    let cases: &[(&str, &str, &str)] = &[
        ("plan_solar_fv",            r#"{"area_m2":20,"irrad":5.5,"eff":0.18,"pf":0.9}"#,          "annual_kwh"),
        ("plan_loan_amortization",   r#"{"principal":100000,"rate_annual":0.12,"months":36}"#,      "monthly_payment"),
        ("plan_slope_stability",     r#"{"c_kpa":10,"phi_deg":30,"H_m":8,"gamma_kn":18,"beta_deg":45}"#, "fos"),
        ("plan_statistics",          r#"{"n":100,"sum_x":5000,"sum_x2":300000}"#,                   "mean"),
        ("plan_bmi_assessment",      r#"{"weight_kg":75,"height_m":1.75,"age":30}"#,                "bmi"),
    ];

    let mut all_pass = true;
    for (plan, params, key) in cases {
        let r = run_n_times(plan, params, key, 5);
        let var = r.values.iter().cloned()
            .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let pass = var == 0.0 && r.values.len() == 5;
        println!("  [{:<6}] {} → key={} val={:.6} variance={:.12}",
            if pass { "PASS" } else { "FAIL" }, plan, key,
            r.values.get(0).copied().unwrap_or(f64::NAN), var);
        if !pass { all_pass = false; }
    }

    assert!(all_pass, "Cross-domain determinism failed for one or more plans");
}

// ── Test: Timing variance (jitter OK, compute deterministic) ─────────────────

#[test]
fn timing_jitter_compute_still_deterministic() {
    let r = run_n_times(
        "plan_pump_sizing",
        r#"{"Q_gpm":750,"P_psi":120,"eff":0.80}"#,
        "nfpa20_pump_hp",
        20,
    );

    // Timing WILL vary (OS scheduler jitter) — that's expected and OK
    let t_min = r.timings_us.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = r.timings_us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_median = {
        let mut ts = r.timings_us.clone();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    };

    // But the compute result must be identical
    let val_var = r.values.iter().cloned()
        .fold(f64::NEG_INFINITY, f64::max) - r.values.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Timing: min={:.1}µs max={:.1}µs median={:.1}µs (jitter OK)", t_min, t_max, t_median);
    println!("  Compute variance: {:.12} (must be 0)", val_var);

    // Jitter of timing is expected — up to 100ms is fine
    assert!(t_max < 100_000.0, "Unexpectedly slow: max roundtrip {}µs", t_max);

    // But compute must be bit-exact
    assert_eq!(val_var, 0.0,
        "CRITICAL: compute varied despite timing jitter! variance={:.12}", val_var);
}

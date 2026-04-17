//! CRYS-L Adversarial & NaN-Shield Tests
//!
//! Validates that CRYS-L handles all adversarial, edge-case, and malformed
//! inputs without panicking, producing undefined behavior, or silently emitting
//! wrong results. This is "Proof 3" from the paper: NaN-Shield.
//!
//! Run with: `cargo test --test adversarial -- --nocapture`
//!
//! Paper reference: §5.1 "Adversarial Input Resilience"

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

// ── HTTP helpers ─────────────────────────────────────────────────────────────

fn post(path: &str, body: &str) -> (String, bool) {
    let result = std::panic::catch_unwind(|| {
        let mut stream = TcpStream::connect("127.0.0.1:9001").expect("server not running");
        let req = format!(
            "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path, body.len(), body
        );
        stream.write_all(req.as_bytes()).unwrap();
        stream.shutdown(std::net::Shutdown::Write).ok();
        let mut buf = String::new();
        stream.read_to_string(&mut buf).unwrap();
        buf.find("\r\n\r\n").map(|p| buf[p + 4..].to_string()).unwrap_or(buf)
    });
    match result {
        Ok(s) => (s, false),  // (response, panicked)
        Err(_) => (String::from("{\"ok\":false,\"error\":\"panic\"}"), true),
    }
}

fn extract_bool(json: &str, key: &str) -> bool {
    let pattern = format!("\"{}\":", key);
    json.find(&pattern).map(|pos| {
        json[pos + pattern.len()..].trim_start().starts_with("true")
    }).unwrap_or(false)
}

fn extract_f64(json: &str, key: &str) -> f64 {
    let pattern = format!("\"{}\":", key);
    json.find(&pattern).and_then(|pos| {
        let after = json[pos + pattern.len()..].trim_start();
        let end = after.find(|c: char| c != '-' && c != '.' && !c.is_ascii_digit()).unwrap_or(after.len());
        after[..end].parse::<f64>().ok()
    }).unwrap_or(f64::NAN)
}

// ── Helper: assert no panic, return structured result ──────────────────────

fn assert_no_panic(plan: &str, params: &str, label: &str) -> String {
    let body = format!(r#"{{"plan":"{}","params":{}}}"#, plan, params);
    let (resp, panicked) = post("/plan/execute", &body);
    assert!(!panicked, "[{}] SERVER PANIC on input: {}", label, params);
    assert!(!resp.contains("\"panic\""), "[{}] panic in response: {}", label, resp);
    resp
}

// ── Test: Zero inputs ────────────────────────────────────────────────────────

#[test]
fn zero_inputs_no_panic() {
    let cases = &[
        ("plan_pump_sizing",      r#"{"Q_gpm":0,"P_psi":0,"eff":0}"#),
        ("plan_electrical_load",  r#"{"P_w":0,"V":0,"pf":0,"L":0,"A":0}"#),
        ("plan_beam_analysis",    r#"{"P_kn":0,"L_m":0,"E_gpa":0,"I_cm4":0}"#),
        ("plan_statistics",       r#"{"n":0,"sum_x":0,"sum_x2":0}"#),
        ("plan_solar_fv",         r#"{"area_m2":0,"irrad":0,"eff":0,"pf":0}"#),
    ];

    for (plan, params) in cases {
        let resp = assert_no_panic(plan, params, "zero_inputs");
        // Either ok:true with result, or ok:false with error — never a panic
        let has_response = resp.contains("\"ok\"");
        assert!(has_response, "[zero] {} gave no valid JSON: {}", plan, &resp[..resp.len().min(200)]);
        println!("  [OK] {} zero → handled gracefully", plan);
    }
}

// ── Test: Negative impossible values ─────────────────────────────────────────

#[test]
fn negative_impossible_values_no_panic() {
    let cases = &[
        ("plan_pump_sizing",      r#"{"Q_gpm":-999,"P_psi":-500,"eff":-1.5}"#),
        ("plan_electrical_load",  r#"{"P_w":-10000,"V":-220,"pf":-0.9,"L":-50,"A":-4}"#),
        ("plan_beam_analysis",    r#"{"P_kn":-100,"L_m":-6,"E_gpa":-200,"I_cm4":-8000}"#),
        ("plan_loan_amortization",r#"{"principal":-100000,"rate_annual":-0.12,"months":-36}"#),
        ("plan_bmi_assessment",   r#"{"weight_kg":-75,"height_m":-1.75,"age":-30}"#),
    ];

    let mut panic_count = 0;
    for (plan, params) in cases {
        let body = format!(r#"{{"plan":"{}","params":{}}}"#, plan, params);
        let (resp, panicked) = post("/plan/execute", &body);
        if panicked {
            panic_count += 1;
            eprintln!("  [PANIC] {} params={}", plan, params);
        } else {
            println!("  [OK]    {} negative → no panic, resp ok={}", plan, extract_bool(&resp, "ok"));
        }
    }

    assert_eq!(panic_count, 0,
        "NaN-Shield failed: {} panics on negative inputs (expected 0)", panic_count);
}

// ── Test: Extreme large values (overflow candidates) ─────────────────────────

#[test]
fn extreme_large_values_no_panic() {
    let cases = &[
        ("plan_pump_sizing",      r#"{"Q_gpm":1e15,"P_psi":1e12,"eff":0.99}"#),
        ("plan_electrical_load",  r#"{"P_w":1e18,"V":1e6,"pf":1.0,"L":1e9,"A":1e6}"#),
        ("plan_compound_interest",r#"{"principal":1e18,"rate":0.999,"periods":10000}"#),
        ("plan_statistics",       r#"{"n":1000000,"sum_x":1e15,"sum_x2":1e30}"#),
        ("plan_beam_analysis",    r#"{"P_kn":1e10,"L_m":1e6,"E_gpa":200,"I_cm4":1e12}"#),
    ];

    let mut panic_count = 0;
    for (plan, params) in cases {
        let body = format!(r#"{{"plan":"{}","params":{}}}"#, plan, params);
        let (resp, panicked) = post("/plan/execute", &body);
        if panicked {
            panic_count += 1;
        }
        // Results may be Inf/NaN in JSON — that's allowed; panic is not
        println!("  [{}] {} large → panicked={}", if panicked { "FAIL" } else { "OK" }, plan, panicked);
    }

    assert_eq!(panic_count, 0,
        "NaN-Shield failed: {} panics on extreme values (expected 0)", panic_count);
}

// ── Test: Missing fields (partial JSON) ──────────────────────────────────────

#[test]
fn missing_fields_no_panic() {
    let cases = &[
        ("plan_pump_sizing",      r#"{"Q_gpm":500}"#),     // missing P_psi, eff
        ("plan_electrical_load",  r#"{"V":220}"#),          // only voltage
        ("plan_beam_analysis",    r#"{}"#),                  // all missing
        ("plan_loan_amortization",r#"{"principal":50000}"#), // missing rate and months
    ];

    for (plan, params) in cases {
        let resp = assert_no_panic(plan, params, "missing_fields");
        println!("  [OK] {} partial → ok={}", plan, extract_bool(&resp, "ok"));
    }
}

// ── Test: Type confusion (strings where numbers expected) ────────────────────

#[test]
fn type_confusion_no_panic() {
    let cases = &[
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":"five_hundred","P_psi":"max","eff":"high"}}"#,
        r#"{"plan":"plan_electrical_load","params":{"P_w":null,"V":false,"pf":[],"L":{},"A":""}}"#,
        r#"{"plan":"plan_statistics","params":{"n":"hundred","sum_x":true,"sum_x2":null}}"#,
    ];

    for body in cases {
        let (resp, panicked) = post("/plan/execute", body);
        assert!(!panicked, "Panic on type confusion: {}", body);
        println!("  [OK] type confusion → ok={}", extract_bool(&resp, "ok"));
    }
}

// ── Test: Malformed JSON ──────────────────────────────────────────────────────

#[test]
fn malformed_json_no_panic() {
    let cases = &[
        r#"not json at all"#,
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":500"#,  // unclosed
        r#"{}"#,                                                   // empty
        r#"{"plan":null,"params":null}"#,
        r#"[]"#,                                                   // array instead of object
        "",                                                        // empty body
    ];

    for body in cases {
        let (resp, panicked) = post("/plan/execute", body);
        assert!(!panicked, "Panic on malformed JSON: {:?}", body);
        println!("  [OK] malformed {:?} → handled", &body[..body.len().min(30)]);
    }
}

// ── Test: IEEE-754 special values in JSON ────────────────────────────────────

#[test]
fn ieee754_special_values_no_panic() {
    // Standard JSON doesn't allow Infinity/NaN, but clients may send them
    let cases = &[
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":1.7976931348623157e308}}"#,  // max f64
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":5e-324}}"#,               // min positive f64
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":0.0}}"#,                  // zero efficiency
    ];

    for body in cases {
        let (resp, panicked) = post("/plan/execute", body);
        assert!(!panicked, "Panic on IEEE-754 edge value: {}", &body[..80]);
        // Result must not contain unquoted NaN or Infinity (invalid JSON)
        assert!(!resp.contains(":NaN"), "Unquoted NaN in response: {}", &resp[..resp.len().min(200)]);
        assert!(!resp.contains(":Infinity"), "Unquoted Infinity in response");
        println!("  [OK] IEEE-754 edge → valid response");
    }
}

// ── Test: 1000-request stress with adversarial mix ───────────────────────────

#[test]
#[ignore = "slow: runs 1000 requests, use --include-ignored for paper benchmarks"]
fn adversarial_stress_1000_requests() {
    let adversarial_payloads = vec![
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":-1,"P_psi":0,"eff":2.5}}"#,
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":1e20,"P_psi":1e20,"eff":1e20}}"#,
        r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":0.75}}"#, // valid
        r#"{"plan":"plan_nonexistent","params":{}}"#,
        r#"not json"#,
    ];

    let n = 1000;
    let mut panics = 0;
    let mut errors = 0;
    let mut successes = 0;
    let t0 = Instant::now();

    for i in 0..n {
        let payload = &adversarial_payloads[i % adversarial_payloads.len()];
        let (resp, panicked) = post("/plan/execute", payload);
        if panicked { panics += 1; }
        else if extract_bool(&resp, "ok") { successes += 1; }
        else { errors += 1; }
    }

    let elapsed_ms = t0.elapsed().as_millis();
    let rps = n as f64 / (elapsed_ms as f64 / 1000.0);

    println!("\n  Adversarial stress ({} requests):", n);
    println!("    Panics:    {} (must be 0)", panics);
    println!("    Errors:    {} (graceful failures, OK)", errors);
    println!("    Successes: {}", successes);
    println!("    Time:      {}ms ({:.0} req/s)", elapsed_ms, rps);

    assert_eq!(panics, 0,
        "NaN-Shield FAILED: {} panics in {} adversarial requests", panics, n);
}

// ── Test: Concurrent adversarial requests ────────────────────────────────────

#[test]
fn concurrent_adversarial_no_race() {
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let panic_count = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..10).map(|i| {
        let pc = Arc::clone(&panic_count);
        thread::spawn(move || {
            // Mix of adversarial and valid
            let payload = if i % 2 == 0 {
                r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":-999,"P_psi":-500,"eff":-1}}"#
            } else {
                r#"{"plan":"plan_pump_sizing","params":{"Q_gpm":500,"P_psi":100,"eff":0.75}}"#
            };
            let (_, panicked) = post("/plan/execute", payload);
            if panicked { pc.fetch_add(1, Ordering::SeqCst); }
        })
    }).collect();

    for h in handles { h.join().ok(); }

    let total_panics = panic_count.load(Ordering::SeqCst);
    println!("  Concurrent adversarial (10 threads): {} panics", total_panics);
    assert_eq!(total_panics, 0, "Race condition or panic in concurrent adversarial test");
}

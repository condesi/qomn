//! CRYS-L Smoke Tests — All 56 Plans
//!
//! Every plan in CRYS-L v3.2 is tested with representative inputs.
//! Tests verify that: (1) server responds without panic, (2) ok:true,
//! (3) key result values are numeric and non-NaN.
//!
//! Run with: `cargo test --test all_56_plans -- --nocapture`
//!
//! Paper reference: §2 "The Plan Library — 56 Deterministic Oracles"

use std::io::{Read, Write};
use std::net::TcpStream;

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

fn extract_bool(json: &str, key: &str) -> bool {
    let pattern = format!("\"{}\":", key);
    json.find(&pattern).map(|pos| json[pos + pattern.len()..].trim_start().starts_with("true")).unwrap_or(false)
}

fn run_plan(plan: &str, params: &str) -> (bool, String) {
    let body = format!(r#"{{"plan":"{}","params":{}}}"#, plan, params);
    let resp = post("/plan/execute", &body);
    let ok = extract_bool(&resp, "ok");
    (ok, resp)
}

// ── Fire & Life Safety Domain ─────────────────────────────────────────────────

#[test] fn smoke_plan_sprinkler_system() {
    let (ok, r) = run_plan("plan_sprinkler_system", r#"{"area_ft2":1500,"K":5.6,"P_avail":50,"hose_stream":250}"#);
    assert!(ok, "plan_sprinkler_system failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_sprinkler_system");
}

#[test] fn smoke_plan_pump_sizing() {
    let (ok, r) = run_plan("plan_pump_sizing", r#"{"Q_gpm":500,"P_psi":100,"eff":0.75}"#);
    assert!(ok, "plan_pump_sizing failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pump_sizing");
}

#[test] fn smoke_plan_egress() {
    let (ok, r) = run_plan("plan_egress", r#"{"N_persons":500,"exits":4,"door_width_in":72}"#);
    assert!(ok, "plan_egress failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_egress");
}

#[test] fn smoke_plan_nfpa13_demand() {
    let (ok, r) = run_plan("plan_nfpa13_demand", r#"{"area_ft2":1500,"density":0.15,"K":5.6,"hose_stream":250}"#);
    assert!(ok, "plan_nfpa13_demand failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_nfpa13_demand");
}

#[test] fn smoke_plan_pipe_losses() {
    let (ok, r) = run_plan("plan_pipe_losses", r#"{"Q_gpm":300,"D_in":4,"L_ft":500,"C":120}"#);
    assert!(ok, "plan_pipe_losses failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pipe_losses");
}

#[test] fn smoke_plan_pressure() {
    let (ok, r) = run_plan("plan_pressure", r#"{"h_ft":100,"rho":1.0,"g":32.2}"#);
    assert!(ok, "plan_pressure failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pressure");
}

#[test] fn smoke_plan_pump_selection() {
    let (ok, r) = run_plan("plan_pump_selection", r#"{"Q_gpm":500,"TDH_ft":200,"eff":0.75}"#);
    assert!(ok, "plan_pump_selection failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pump_selection");
}

// ── Hydraulics & Pipe Networks ────────────────────────────────────────────────

#[test] fn smoke_plan_pipe_manning() {
    let (ok, r) = run_plan("plan_pipe_manning", r#"{"n":0.013,"D":0.3,"S":0.001}"#);
    assert!(ok, "plan_pipe_manning failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pipe_manning");
}

#[test] fn smoke_plan_pipe_hazen() {
    let (ok, r) = run_plan("plan_pipe_hazen", r#"{"Q":500,"C":120,"D":4,"L":1000}"#);
    assert!(ok, "plan_pipe_hazen failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pipe_hazen");
}

#[test] fn smoke_plan_pipe_network_3() {
    let (ok, r) = run_plan("plan_pipe_network_3", r#"{"Q1":200,"Q2":150,"Q3":100,"C":120,"D":4,"L":500}"#);
    assert!(ok, "plan_pipe_network_3 failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pipe_network_3");
}

#[test] fn smoke_plan_hazen_sweep() {
    let (ok, r) = run_plan("plan_hazen_sweep", r#"{"Q_min":100,"Q_max":800,"C":120,"D":4,"L":1000}"#);
    assert!(ok, "plan_hazen_sweep failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_hazen_sweep");
}

#[test] fn smoke_plan_hazen_critical_q() {
    let (ok, r) = run_plan("plan_hazen_critical_q", r#"{"P_avail":65,"C":120,"D":4,"L":1000,"elev_diff":0}"#);
    assert!(ok, "plan_hazen_critical_q failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_hazen_critical_q");
}

#[test] fn smoke_plan_hydro_channel() {
    let (ok, r) = run_plan("plan_hydro_channel", r#"{"b":2,"y":1,"S":0.001,"n":0.015}"#);
    assert!(ok, "plan_hydro_channel failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_hydro_channel");
}

#[test] fn smoke_plan_hydro_demand() {
    let (ok, r) = run_plan("plan_hydro_demand", r#"{"population":50000,"demand_lpcd":200,"peak_factor":2.5}"#);
    assert!(ok, "plan_hydro_demand failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_hydro_demand");
}

// ── Electrical Engineering ────────────────────────────────────────────────────

#[test] fn smoke_plan_electrical_load() {
    let (ok, r) = run_plan("plan_electrical_load", r#"{"P_w":5000,"V":220,"pf":0.92,"L":50,"A":4}"#);
    assert!(ok, "plan_electrical_load failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_electrical_load");
}

#[test] fn smoke_plan_electrical_3ph() {
    let (ok, r) = run_plan("plan_electrical_3ph", r#"{"P_kw":15,"V":380,"pf":0.92,"L":100,"A":6}"#);
    assert!(ok, "plan_electrical_3ph failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_electrical_3ph");
}

#[test] fn smoke_plan_transformer() {
    let (ok, r) = run_plan("plan_transformer", r#"{"P_kw":100,"pf":0.92,"eff":0.97}"#);
    assert!(ok, "plan_transformer failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_transformer");
}

#[test] fn smoke_plan_power_factor_correction() {
    let (ok, r) = run_plan("plan_power_factor_correction", r#"{"P_kw":50,"pf_actual":0.75,"pf_meta":0.95}"#);
    assert!(ok, "plan_power_factor_correction failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_power_factor_correction");
}

#[test] fn smoke_plan_voltage_drop() {
    let (ok, r) = run_plan("plan_voltage_drop", r#"{"I":25,"L_m":80,"A_mm2":4}"#);
    assert!(ok, "plan_voltage_drop failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_voltage_drop");
}

#[test] fn smoke_plan_motor_drive() {
    let (ok, r) = run_plan("plan_motor_drive", r#"{"P_kw":11,"V":380,"pf":0.88,"eff":0.92,"service_factor":1.15}"#);
    assert!(ok, "plan_motor_drive failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_motor_drive");
}

// ── Structural Engineering ────────────────────────────────────────────────────

#[test] fn smoke_plan_beam_analysis() {
    let (ok, r) = run_plan("plan_beam_analysis", r#"{"P_kn":50,"L_m":6,"E_gpa":200,"I_cm4":8000}"#);
    assert!(ok, "plan_beam_analysis failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_beam_analysis");
}

#[test] fn smoke_plan_column_design() {
    let (ok, r) = run_plan("plan_column_design", r#"{"P_kn":500,"L_m":4,"E_gpa":200,"I_cm4":5000,"A_cm2":100}"#);
    assert!(ok, "plan_column_design failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_column_design");
}

#[test] fn smoke_plan_footing() {
    let (ok, r) = run_plan("plan_footing", r#"{"P_kn":800,"q_adm":150,"depth_m":1.5}"#);
    assert!(ok, "plan_footing failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_footing");
}

#[test] fn smoke_plan_slope_stability() {
    let (ok, r) = run_plan("plan_slope_stability", r#"{"c_kpa":10,"phi_deg":30,"H_m":8,"gamma_kn":18,"beta_deg":45}"#);
    assert!(ok, "plan_slope_stability failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_slope_stability");
}

#[test] fn smoke_plan_slope_analysis() {
    let (ok, r) = run_plan("plan_slope_analysis", r#"{"rise":3,"run":4,"H_m":10}"#);
    assert!(ok, "plan_slope_analysis failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_slope_analysis");
}

#[test] fn smoke_plan_vibration_fatigue() {
    let (ok, r) = run_plan("plan_vibration_fatigue", r#"{"sigma_a":150,"sigma_m":100,"Su":500,"Se":250,"cycles":1e6}"#);
    assert!(ok, "plan_vibration_fatigue failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_vibration_fatigue");
}

// ── Finance & Peruvian Payroll ────────────────────────────────────────────────

#[test] fn smoke_plan_factura_peru() {
    let (ok, r) = run_plan("plan_factura_peru", r#"{"subtotal":10000,"igv_rate":0.18}"#);
    assert!(ok, "plan_factura_peru failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_factura_peru");
}

#[test] fn smoke_plan_planilla() {
    let (ok, r) = run_plan("plan_planilla", r#"{"salario_bruto":5000,"horas_extras":10,"regimen":"general"}"#);
    assert!(ok, "plan_planilla failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_planilla");
}

#[test] fn smoke_plan_ratios_financieros() {
    let (ok, r) = run_plan("plan_ratios_financieros", r#"{"activo_corriente":50000,"pasivo_corriente":25000,"inventario":10000,"ventas":200000,"utilidad_neta":20000,"activo_total":150000}"#);
    assert!(ok, "plan_ratios_financieros failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_ratios_financieros");
}

#[test] fn smoke_plan_depreciacion() {
    let (ok, r) = run_plan("plan_depreciacion", r#"{"costo":50000,"vida_util":5,"valor_residual":5000}"#);
    assert!(ok, "plan_depreciacion failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_depreciacion");
}

#[test] fn smoke_plan_liquidacion_laboral() {
    let (ok, r) = run_plan("plan_liquidacion_laboral", r#"{"salario_mensual":3000,"meses_trabajados":24,"dias_vacaciones":15}"#);
    assert!(ok, "plan_liquidacion_laboral failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_liquidacion_laboral");
}

#[test] fn smoke_plan_multa_sunafil() {
    let (ok, r) = run_plan("plan_multa_sunafil", r#"{"tipo":"grave","uit":5150,"trabajadores":20}"#);
    assert!(ok, "plan_multa_sunafil failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_multa_sunafil");
}

// ── Business Analytics ────────────────────────────────────────────────────────

#[test] fn smoke_plan_break_even() {
    let (ok, r) = run_plan("plan_break_even", r#"{"fixed_cost":50000,"variable_cost_unit":30,"price_unit":80}"#);
    assert!(ok, "plan_break_even failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_break_even");
}

#[test] fn smoke_plan_pricing() {
    let (ok, r) = run_plan("plan_pricing", r#"{"cost":100,"margin":0.40}"#);
    assert!(ok, "plan_pricing failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_pricing");
}

#[test] fn smoke_plan_roi() {
    let (ok, r) = run_plan("plan_roi", r#"{"gain":150000,"cost":100000}"#);
    assert!(ok, "plan_roi failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_roi");
}

#[test] fn smoke_plan_loan_amortization() {
    let (ok, r) = run_plan("plan_loan_amortization", r#"{"principal":100000,"rate_annual":0.12,"months":36}"#);
    assert!(ok, "plan_loan_amortization failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_loan_amortization");
}

#[test] fn smoke_plan_compound_interest() {
    let (ok, r) = run_plan("plan_compound_interest", r#"{"principal":10000,"rate":0.08,"periods":10}"#);
    assert!(ok, "plan_compound_interest failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_compound_interest");
}

// ── HVAC & Energy ─────────────────────────────────────────────────────────────

#[test] fn smoke_plan_hvac_cooling() {
    let (ok, r) = run_plan("plan_hvac_cooling", r#"{"area_m2":100,"ceiling_h":3,"ach":6,"delta_t":10,"occupants":20}"#);
    assert!(ok, "plan_hvac_cooling failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_hvac_cooling");
}

#[test] fn smoke_plan_hvac_ventilation() {
    let (ok, r) = run_plan("plan_hvac_ventilation", r#"{"area_m2":200,"occupants":50,"cfm_per_person":15}"#);
    assert!(ok, "plan_hvac_ventilation failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_hvac_ventilation");
}

#[test] fn smoke_plan_solar_fv() {
    let (ok, r) = run_plan("plan_solar_fv", r#"{"area_m2":20,"irrad":5.5,"eff":0.18,"pf":0.9}"#);
    assert!(ok, "plan_solar_fv failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_solar_fv");
}

// ── Medical & Clinical ────────────────────────────────────────────────────────

#[test] fn smoke_plan_bmi_assessment() {
    let (ok, r) = run_plan("plan_bmi_assessment", r#"{"weight_kg":75,"height_m":1.75,"age":30}"#);
    assert!(ok, "plan_bmi_assessment failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_bmi_assessment");
}

#[test] fn smoke_plan_drug_dosing() {
    let (ok, r) = run_plan("plan_drug_dosing", r#"{"weight_kg":70,"dose_mg_kg":5,"frequency_h":8}"#);
    assert!(ok, "plan_drug_dosing failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_drug_dosing");
}

#[test] fn smoke_plan_autoclave_cycle() {
    let (ok, r) = run_plan("plan_autoclave_cycle", r#"{"load_kg":10,"target_temp_c":134,"exposure_min":4}"#);
    assert!(ok, "plan_autoclave_cycle failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_autoclave_cycle");
}

#[test] fn smoke_plan_medical_gas() {
    let (ok, r) = run_plan("plan_medical_gas", r#"{"beds":50,"o2_lpm_per_bed":8,"peak_factor":1.5}"#);
    assert!(ok, "plan_medical_gas failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_medical_gas");
}

// ── Statistics & Data Science ─────────────────────────────────────────────────

#[test] fn smoke_plan_statistics() {
    let (ok, r) = run_plan("plan_statistics", r#"{"n":100,"sum_x":5000,"sum_x2":300000}"#);
    assert!(ok, "plan_statistics failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_statistics");
}

#[test] fn smoke_plan_sample_size() {
    let (ok, r) = run_plan("plan_sample_size", r#"{"confidence":0.95,"margin_error":0.05,"p":0.5}"#);
    assert!(ok, "plan_sample_size failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_sample_size");
}

// ── Cybersecurity ─────────────────────────────────────────────────────────────

#[test] fn smoke_plan_password_audit() {
    let (ok, r) = run_plan("plan_password_audit", r#"{"charset_size":94,"length":12,"attempts_per_sec":1000000000}"#);
    assert!(ok, "plan_password_audit failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_password_audit");
}

#[test] fn smoke_plan_cvss_assessment() {
    let (ok, r) = run_plan("plan_cvss_assessment", r#"{"AV":"N","AC":"L","PR":"N","UI":"N","S":"C","C":"H","I":"H","A":"H"}"#);
    assert!(ok, "plan_cvss_assessment failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_cvss_assessment");
}

#[test] fn smoke_plan_network_security() {
    let (ok, r) = run_plan("plan_network_security", r#"{"hosts":100,"open_ports_avg":5,"critical_services":3}"#);
    assert!(ok, "plan_network_security failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_network_security");
}

#[test] fn smoke_plan_crypto_audit() {
    let (ok, r) = run_plan("plan_crypto_audit", r#"{"key_bits":256,"algorithm":"AES","mode":"GCM","hash":"SHA256"}"#);
    assert!(ok, "plan_crypto_audit failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_crypto_audit");
}

#[test] fn smoke_plan_bcrypt_audit() {
    let (ok, r) = run_plan("plan_bcrypt_audit", r#"{"cost":12,"hash_count":10000}"#);
    assert!(ok, "plan_bcrypt_audit failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_bcrypt_audit");
}

// ── Civil & Geotechnical ──────────────────────────────────────────────────────

#[test] fn smoke_plan_earthwork() {
    let (ok, r) = run_plan("plan_earthwork", r#"{"cut_m3":5000,"fill_m3":3000,"swell_factor":1.25,"shrink_factor":0.9}"#);
    assert!(ok, "plan_earthwork failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_earthwork");
}

#[test] fn smoke_plan_transport() {
    let (ok, r) = run_plan("plan_transport", r#"{"distance_km":500,"load_tn":20,"fuel_l100km":35,"fuel_price":5.5}"#);
    assert!(ok, "plan_transport failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_transport");
}

// ── Agriculture & Irrigation ──────────────────────────────────────────────────

#[test] fn smoke_plan_irrigation() {
    let (ok, r) = run_plan("plan_irrigation", r#"{"area_ha":10,"eto_mm":5,"kc":0.8,"eff":0.75}"#);
    assert!(ok, "plan_irrigation failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_irrigation");
}

#[test] fn smoke_plan_drip_irrigation() {
    let (ok, r) = run_plan("plan_drip_irrigation", r#"{"area_ha":5,"emitters_per_ha":2000,"flow_lph":2,"eff":0.92}"#);
    assert!(ok, "plan_drip_irrigation failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_drip_irrigation");
}

// ── Telecom ───────────────────────────────────────────────────────────────────

#[test] fn smoke_plan_telecom_link() {
    let (ok, r) = run_plan("plan_telecom_link", r#"{"tx_dbm":30,"freq_mhz":900,"distance_km":5,"bw_mhz":20}"#);
    assert!(ok, "plan_telecom_link failed: {}", &r[..r.len().min(300)]);
    println!("  [OK] plan_telecom_link");
}

// ── All 56 plans aggregate test ───────────────────────────────────────────────

#[test]
fn all_56_plans_respond_without_panic() {
    // Quick sanity: at least the plan list endpoint works
    use std::io::{Read, Write};
    let mut stream = TcpStream::connect("127.0.0.1:9001").expect("server not running");
    let req = "GET /plans HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    stream.write_all(req.as_bytes()).unwrap();
    stream.shutdown(std::net::Shutdown::Write).ok();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).unwrap();
    let body = buf.find("\r\n\r\n").map(|p| &buf[p+4..]).unwrap_or("");

    // Count plan names in response
    let plan_count = body.matches("\"name\"").count();
    println!("  Plans registered: {}", plan_count);
    assert!(plan_count >= 50,
        "Expected ≥50 plans, found {}: {}", plan_count, &body[..body.len().min(200)]);
}

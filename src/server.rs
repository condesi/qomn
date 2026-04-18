// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v2.1 — HTTP Server Mode
// Percy Rojas M. · Qomni AI Lab · 2026
//
// Endpoints:
//   GET  /health              → status
//   GET  /plans               → list loaded plans
//   POST /plan/execute        → run plan  {"plan":"name","params":{...}}
//   POST /intent              → NL query → plan → result  {"q":"..."}
//   POST /query               → VM query  {"q":"expr"}
//   POST /eval                → eval expr {"expr":"expr"}
// ═══════════════════════════════════════════════════════════════════════

use tungstenite::{accept as ws_accept, Message as WsMessage};

// ── WebSocket client registry (Digital Twin push) ──────────────────────────
static WS_TWIN_CLIENTS: std::sync::OnceLock<
    std::sync::Mutex<Vec<std::sync::mpsc::SyncSender<String>>>
> = std::sync::OnceLock::new();

fn ws_clients() -> std::sync::MutexGuard<'static, Vec<std::sync::mpsc::SyncSender<String>>> {
    WS_TWIN_CLIENTS.get_or_init(|| std::sync::Mutex::new(Vec::new())).lock().unwrap()
}

/// Broadcast a JSON string to all connected WebSocket clients. Drops dead clients.
pub fn ws_broadcast(msg: &str) {
    let mut clients = ws_clients();
    clients.retain(|tx| tx.send(msg.to_string()).is_ok());
}

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use crate::vm::Vm;
use crate::ast::{Program, PlanDecl};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::plan::{self, PlanExecutor};
use crate::aot_plan::AotPlanCache;
use crate::intent_parser::{self, MockBackend};
use crate::batch_plan;
use crate::simulation_engine::{self, PlanId, SweepSpec, global_engine};
use crate::benchmark_proofs;
use crate::llvm_backend;
use crate::wasm_backend;
use crate::plan_v2::{parse_plans, execute_plan};


// ── AOT Plan Cache (global, initialized once at startup) ──────────────
static AOT_CACHE: std::sync::OnceLock<AotPlanCache> = std::sync::OnceLock::new();

/// Initialize the global AOT cache. Called once from main.rs after JIT compile.
pub fn set_no_fma_mode(enabled: bool) {
    NO_FMA_MODE.store(enabled, Ordering::Relaxed);
}

pub fn init_aot_cache(cache: AotPlanCache) {
    let _ = AOT_CACHE.set(cache);
}

// ── Reactive Sessions (global, RwLock-protected) ──────────────────────
// RwLock and LinkedList removed — replaced by DashMap session store

pub struct ReactiveSession {
    pub id: String,
    pub plan_name: String,
    pub plan_idx: usize,
    pub params: std::collections::HashMap<String, f64>,
    pub cached_results: Vec<f64>,
    pub step_deps: Vec<Vec<usize>>,
    pub last_access: std::time::Instant,
}

static SESSION_MAP: std::sync::OnceLock<DashMap<String, ReactiveSession>> = std::sync::OnceLock::new();

fn session_map() -> &'static DashMap<String, ReactiveSession> {
    SESSION_MAP.get_or_init(|| DashMap::new())
}

// ── Intent Session Context (conversational "what if" support) ─────────
struct IntentContext {
    last_plan: String,
    last_params: std::collections::HashMap<String, f64>,
    last_domain: String,
    timestamp: std::time::Instant,
}

static INTENT_CTX: std::sync::OnceLock<DashMap<String, IntentContext>> = std::sync::OnceLock::new();

fn intent_ctx_map() -> &'static DashMap<String, IntentContext> {
    INTENT_CTX.get_or_init(|| DashMap::new())
}

fn is_modify_intent(q: &str) -> bool {
    let lower = q.to_lowercase();
    let patterns = [
        "qu\u{00e9} pasa si", "que pasa si", "what if",
        "si aumento", "si cambio", "si reduzco", "si bajo",
        "si subo", "si pongo", "si uso",
        "y si ", "pero si ", "ahora con ",
        "cambia a", "cambialo a", "ponle",
        "sube a", "baja a", "aumenta a",
        "y con ", "pero con ",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

fn extract_param_changes(q: &str, known_params: &[String]) -> std::collections::HashMap<String, f64> {
    let mut changes = std::collections::HashMap::new();
    let lower = q.to_lowercase();

    // Unit-to-param mapping
    let unit_map: &[(&str, &str)] = &[
        ("gpm", "Q_gpm"), ("psi", "P_psi"), ("eficiencia", "eff"),
        ("hp", "hp"), ("area", "area_ft2"), ("m2", "area_m2"),
        ("caudal", "Q_gpm"), ("presion", "P_psi"),
        ("diametro", "D_in"), ("pulg", "D_in"),
        ("longitud", "L_m"), ("voltaje", "V"), ("corriente", "I"),
        ("metros", "L_m"), ("pulgadas", "D_in"), ("ft", "L_ft"),
        ("bar", "P_bar"), ("kpa", "P_kPa"), ("lps", "Q_lps"),
        ("m3/h", "Q_m3h"), ("c", "C"),
    ];

    let words: Vec<&str> = lower.split_whitespace().collect();
    for i in 0..words.len() {
        let word = words[i].trim_end_matches(|c: char| c == ',' || c == '?' || c == '.');
        if let Ok(num) = word.parse::<f64>() {
            // Check next word for unit
            if i + 1 < words.len() {
                let next = words[i + 1].trim_end_matches(|c: char| !c.is_alphanumeric());
                for (unit, param) in unit_map {
                    if next.contains(unit) && known_params.iter().any(|p| p == param) {
                        changes.insert(param.to_string(), num);
                    }
                }
            }
            // Check previous word for unit
            if i > 0 {
                let prev = words[i - 1].trim_end_matches(|c: char| !c.is_alphanumeric());
                for (unit, param) in unit_map {
                    if prev.contains(unit) && known_params.iter().any(|p| p == param) {
                        changes.insert(param.to_string(), num);
                    }
                }
            }
        }
    }
    changes
}

fn session_insert(session: ReactiveSession) {
    let map = session_map();
    // Evict old entries if over 1000
    if map.len() >= 1000 {
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = std::time::Instant::now();
        for entry in map.iter() {
            if entry.value().last_access < oldest_time {
                oldest_time = entry.value().last_access;
                oldest_key = Some(entry.key().clone());
            }
        }
        if let Some(k) = oldest_key { map.remove(&k); }
    }
    let id = session.id.clone();
    map.insert(id, session);
}

pub struct CrysServer {
    vm:      Arc<Mutex<Vm>>,
    prog:    Program,
    plans:   Arc<Vec<PlanDecl>>,
    jit_map: Arc<Option<plan::JitFnMap>>,
    port:    u16,
}

impl CrysServer {
    pub fn new(vm: Vm, prog: Program, port: u16) -> Self {
        let plans: Vec<PlanDecl> = prog.decls.iter()
            .filter_map(|d| if let crate::ast::Decl::Plan(p) = d { Some(p.clone()) } else { None })
            .collect();
        Self {
            vm:      Arc::new(Mutex::new(vm)),
            plans:   Arc::new(plans),
            jit_map: Arc::new(None),
            prog,
            port,
        }
    }

    /// Supply a JIT fn-address map for fast plan execution.
    pub fn with_jit_map(mut self, map: plan::JitFnMap) -> Self {
        self.jit_map = Arc::new(Some(map));
        self
    }

    pub fn run(&self) {
        // Load persisted DKP knowledge facts at startup
        load_dkp_store();
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr)
            .unwrap_or_else(|e| { eprintln!("Bind error: {}", e); std::process::exit(1) });

        println!("  CRYS-L server listening on {}", addr);
        // Store plans/jit for autonomous loop
        let _ = LIVE_PLANS.set(Arc::clone(&self.plans));
        let _ = LIVE_JIT.set(Arc::clone(&self.jit_map));
        spawn_nvd_poller();
        spawn_registry_syncer();
        println!("  Endpoints:");
        println!("    GET  /health");
        println!("    GET  /plans");
        println!("    GET  /memory             {{recent:[], total:N}}");
        println!("    POST /plan/execute  {{\"plan\":\"name\",\"params\":{{...}}}}");
        println!("    POST /intent        {{\"q\":\"natural language query\"}}");
        println!("    POST /query         {{\"q\":\"expr\"}}");
        println!("    POST /eval          {{\"expr\":\"expr\"}}");
        println!("    POST /web/fetch     {{\"url\":\"https://...\",\"selector\":\"optional\"}}");
        println!("    POST /convert       {{\"value\":100,\"from\":\"ft2\",\"to\":\"m2\"}}");
        println!("    POST /decision/analyze  {{\"plan\":\"name\",\"execute\":true}}");
        println!("    GET  /decision/rules");

        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let vm      = Arc::clone(&self.vm);
                    let prog    = self.prog.clone();
                    let plans   = Arc::clone(&self.plans);
                    let jit_map = Arc::clone(&self.jit_map);
                    std::thread::spawn(move || handle_conn(s, vm, prog, plans, jit_map));
                }
                Err(e) => eprintln!("Connection error: {}", e),
            }
        }
    }
}

/// WebSocket handler for /ws/twin — keeps connection alive, pushes twin updates
fn handle_ws_twin(mut stream: std::net::TcpStream, ws_key: &str) {
    // Compute Sec-WebSocket-Accept: SHA1(key + GUID) in base64
    // tungstenite exposes derive_accept_key for this
    let accept = tungstenite::handshake::derive_accept_key(ws_key.trim().as_bytes());
    let hs = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\r\n",
        accept
    );
    if stream.write_all(hs.as_bytes()).is_err() { return; }

    // Wrap in tungstenite WebSocket (server role, handshake already done)
    let mut ws = tungstenite::WebSocket::from_raw_socket(
        stream,
        tungstenite::protocol::Role::Server,
        None,
    );

    // Register channel for receiving broadcast messages
    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(64);
    ws_clients().push(tx);

    // Push-loop: forward channel messages → WebSocket frame
    loop {
        match rx.recv_timeout(std::time::Duration::from_secs(25)) {
            Ok(msg) => {
                if ws.send(tungstenite::Message::Text(msg)).is_err() { break; }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Keepalive ping
                if ws.send(tungstenite::Message::Ping(vec![])).is_err() { break; }
            }
            Err(_) => break,
        }
        // Drain incoming frames (pongs, close)
        match ws.read() {
            Ok(tungstenite::Message::Close(_)) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
}


fn handle_conn(
    mut stream: TcpStream,
    vm:         Arc<Mutex<Vm>>,
    prog:       Program,
    plans:      Arc<Vec<PlanDecl>>,
    jit_map:    Arc<Option<plan::JitFnMap>>,
) {
    use std::io::{Read, BufRead, BufReader};
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(5000)));
    let _ = stream.set_nodelay(true);

    // Buffered read: headers line-by-line, then exact Content-Length body
    let mut reader = BufReader::new(&stream);
    let mut header_buf = String::new();
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line == "\r\n" { break; }
                header_buf.push_str(&line);
            }
            Err(_) => break,
        }
    }
    let content_length: usize = header_buf.lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let mut body_bytes = vec![0u8; content_length.min(1048576)];
    if content_length > 0 {
        let _ = reader.read_exact(&mut body_bytes);
    }
    let body_raw = String::from_utf8_lossy(&body_bytes).to_string();
    let headers_part = header_buf.clone();
    let mut hdr_lines = headers_part.lines();
    let request_line  = hdr_lines.next().unwrap_or("");
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 { return; }
    let method = parts[0];
    let path   = parts[1];
    let body   = body_raw.trim_matches(char::from(0)).to_string();

    // WebSocket upgrade intercept for /ws/twin
    let is_ws_twin = path == "/ws/twin"
        && header_buf.lines().any(|l| l.to_lowercase().contains("upgrade: websocket"));
    let is_ws_sim = path == "/ws/sim"
        && header_buf.lines().any(|l| l.to_lowercase().contains("upgrade: websocket"));
    if is_ws_twin || is_ws_sim {
        let ws_key = header_buf.lines()
            .find(|l| l.to_lowercase().starts_with("sec-websocket-key:"))
            .and_then(|l| l.splitn(2, ':').nth(1))
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        drop(reader);
        if is_ws_sim {
            benchmark_proofs::handle_ws_sim(stream, &ws_key, simulation_engine::global_engine());
        } else {
            handle_ws_twin(stream, &ws_key);
        }
        return;
    }

    let _t_route = std::time::Instant::now();
    let (status, json) = route_request(method, path, &body, &vm, &prog, &plans, &jit_map);
    record_metric(_t_route.elapsed().as_nanos() as u64, status.starts_with("200"));

    let (real_status, ct) = if status == "200 SSE" {
        ("200 OK", "text/event-stream")
    } else {
        (status, "application/json")
    };
    let cache_hdr = "Cache-Control: no-store, no-cache, must-revalidate
";
    let response = format!(
        "HTTP/1.1 {}
Content-Type: {}
{}X-CRYS-Computed: live
X-CRYS-Version: 3.2
Access-Control-Allow-Origin: *
Content-Length: {}

{}",
        real_status, ct, cache_hdr, json.len(), json
    );
    let _ = stream.write_all(response.as_bytes());
}



// ── Canonicalize -0.0 to +0.0 for deterministic hashing ─────────────
#[inline(always)]
fn canon_f64(v: f64) -> f64 {
    // IEEE-754: -0.0 == +0.0 but they have different bits.
    // Physics results should never be -0.0, but we enforce it here.
    if v == 0.0 { 0.0 } else { v }
}

// ── Physics Guard: validate plan inputs before JIT execution ──────────
fn validate_plan_physics(plan_name: &str, params: &std::collections::HashMap<String, f64>) -> Result<(), String> {
    match plan_name {
        "plan_pump_sizing" => {
            let q = params.get("Q_gpm").copied().unwrap_or(0.0);
            let p = params.get("P_psi").copied().unwrap_or(0.0);
            let eff = params.get("eff").copied().unwrap_or(0.70);
            if q <= 0.0 { return Err(format!("assertion failed: flow must be positive (Q_gpm={} must be > 0)", q)); }
            if !q.is_finite() { return Err(format!("assertion failed: flow must be finite (Q_gpm={} — inf/NaN not allowed)", q)); }
            if q < 1.0 { return Err(format!("assertion failed: Q_gpm={} below practical minimum (fire pumps: 25 GPM per NFPA 20; non-fire: 1 GPM)", q)); }
            if p <= 0.0 { return Err(format!("assertion failed: pressure must be positive (P_psi={} must be > 0)", p)); }
            if !p.is_finite() { return Err(format!("assertion failed: pressure must be finite (P_psi={} — inf/NaN not allowed)", p)); }
            if eff <= 0.0 { return Err(format!("assertion failed: efficiency must be positive (eff={} must be > 0)", eff)); }
            if eff > 1.0 { return Err(format!("assertion failed: efficiency must be <= 1.0 (eff={} is physically impossible — max pump efficiency is 100%)", eff)); }
            if q > 100_000.0 { return Err(format!("assertion failed: flow Q_gpm={} exceeds physical maximum for single pump (100,000 GPM). Use parallel pump configuration.", q)); }
            if p > 5_000.0 { return Err(format!("assertion failed: pressure P_psi={} exceeds NFPA 20 maximum (5,000 PSI). Check units.", p)); }
        }
        "plan_sprinkler_system" => {
            let k = params.get("K").copied().unwrap_or(0.0);
            let p = params.get("P_avail").copied().unwrap_or(0.0);
            let area = params.get("area_ft2").copied().unwrap_or(0.0);
            if k <= 0.0 { return Err(format!("assertion failed: K-factor must be positive (K={})", k)); }
            if k < 1.4 { return Err(format!("assertion failed: K={} below NFPA 13 minimum (K=1.4). Standard K values: 1.4, 2.0, 2.8, 4.2, 5.6, 8.0, 11.2, 14.0, 16.8, 19.6, 22.4, 25.2, 28.0", k)); }
            if k > 28.0 { return Err(format!("assertion failed: K={} exceeds NFPA 13 max (K=28.0 ESFR). Check units.", k)); }
            if p < 7.0 { return Err(format!("assertion failed: P_avail={} PSI below minimum -- NFPA 13 absolute minimum 7 PSI; design minimum 15 PSI", p)); }
            if area <= 0.0 { return Err(format!("assertion failed: area_ft2={} must be positive", area)); }
            if area > 52_000.0 { return Err(format!("assertion failed: area_ft2={} exceeds NFPA 13 max (52,000 ft2). Split into zones.", area)); }
        }
        "plan_voltage_drop" => {
            // Plan params: I (current A), L_m (length m), A_mm2 (conductor area mm2)
            let i = params.get("I").copied().unwrap_or(0.0);
            let l = params.get("L_m").copied().unwrap_or(0.0);
            let a = params.get("A_mm2").copied().unwrap_or(0.0);
            if i <= 0.0 { return Err(format!("assertion failed: current must be positive (I={} A)", i)); }
            if l <= 0.0 { return Err(format!("assertion failed: cable length must be positive (L_m={} m)", l)); }
            if a <= 0.0 { return Err(format!("assertion failed: conductor area must be positive (A_mm2={} mm²) — e.g. 4 mm², 16 mm², 35 mm²", a)); }
        }
        "plan_beam_analysis" => {
            // Plan params: P_kn (load kN), L_m (span m), E_gpa (modulus GPa), I_cm4 (inertia cm4)
            let p = params.get("P_kn").copied().unwrap_or(0.0);
            let l = params.get("L_m").copied().unwrap_or(0.0);
            let e = params.get("E_gpa").copied().unwrap_or(200.0);  // default steel
            let i_val = params.get("I_cm4").copied().unwrap_or(0.0);
            if p <= 0.0 { return Err(format!("assertion failed: load must be positive (P_kn={} kN)", p)); }
            if l <= 0.0 { return Err(format!("assertion failed: span must be positive (L_m={} m)", l)); }
            if e <= 0.0 { return Err(format!("assertion failed: elastic modulus must be positive (E_gpa={} GPa) — steel=200, concrete=25", e)); }
            if i_val <= 0.0 { return Err(format!("assertion failed: moment of inertia must be positive (I_cm4={} cm4)", i_val)); }
        }
        "plan_pipe_hazen" => {
            let q = params.get("Q").copied().unwrap_or(0.0);
            let c = params.get("C").copied().unwrap_or(0.0);
            let d = params.get("D").copied().unwrap_or(0.0);
            let l = params.get("L").copied().unwrap_or(0.0);
            if q <= 0.0 { return Err(format!("assertion failed: flow must be positive (Q={})", q)); }
            if c <= 0.0 { return Err(format!("assertion failed: Hazen-Williams C must be positive (C={})", c)); }
            if d <= 0.0 { return Err(format!("assertion failed: pipe diameter must be positive (D={})", d)); }
            if d < 0.006 { return Err(format!("assertion failed: D={:.6}m = {:.3}in below engineering minimum. Smallest commercial pipe: NPS 1/8in = 6.35mm. NFPA 13 fire branch min: 1in = 0.0254m. Near-zero D causes overflow. — Smallest commercial pipe: NPS 1/8in = 6.35mm. NFPA 13 fire branch min: 1in = 0.0254m.", d, d*39.3701)); }
            if d > 10.0 { return Err(format!("assertion failed: D={}m exceeds practical max (10m / 33ft). Check units -- mm instead of m?", d)); }
            if l <= 0.0 { return Err(format!("assertion failed: pipe length must be positive (L={})", l)); }
            if l > 1_000_000.0 { return Err(format!("assertion failed: pipe length L={} km exceeds plausible engineering range (max 1,000 km)", l/1000.0)); }
        }
        "plan_drug_dosing" => {
            let wt = params.get("weight_kg").copied().unwrap_or(0.0);
            let dose = params.get("dose_mg_per_kg").copied().unwrap_or(0.0);
            let freq = params.get("frequency_h").copied().unwrap_or(0.0);
            if wt <= 0.0 { return Err(format!("assertion failed: patient weight must be positive (weight_kg={})", wt)); }
            if dose <= 0.0 { return Err(format!("assertion failed: dose must be positive (dose_mg_per_kg={})", dose)); }
            if freq <= 0.0 { return Err(format!("assertion failed: frequency must be positive (frequency_h={})", freq)); }
            if wt > 500.0 { return Err(format!("assertion failed: weight exceeds physiological maximum (weight_kg={} > 500)", wt)); }
        }
        "plan_loan_amortization" => {
            let p = params.get("principal").copied().unwrap_or(0.0);
            let r = params.get("annual_rate").copied().unwrap_or(0.0);
            let m = params.get("months").copied().unwrap_or(0.0);
            if p <= 0.0 { return Err(format!("assertion failed: principal must be positive (principal={})", p)); }
            if r < 0.0 { return Err(format!("assertion failed: interest rate must be non-negative (annual_rate={})", r)); }
            if r > 10.0 { return Err(format!("assertion failed: annual_rate={} — did you mean {} (decimal, not percent)?", r, r/100.0)); }
            if m <= 0.0 { return Err(format!("assertion failed: loan term must be positive (months={})", m)); }
        }
        "plan_cvss_assessment" => {
            let check_metric = |name: &str, val: f64| -> Result<(), String> {
                if val < 0.0 || val > 1.0 {
                    Err(format!("assertion failed: CVSS metric {} must be in [0,1] (got {})", name, val))
                } else { Ok(()) }
            };
            for (k, v) in params.iter() {
                check_metric(k, *v)?;
            }
        }
        "plan_full_fire_system" => {
            let q   = params.get("Q_gpm").copied().unwrap_or(0.0);
            let d   = params.get("D_in").copied().unwrap_or(0.0);
            let l   = params.get("L_ft").copied().unwrap_or(0.0);
            let elv = params.get("elev_ft").copied().unwrap_or(-1.0);
            let pr  = params.get("P_residual").copied().unwrap_or(-1.0);
            let eff = params.get("eff").copied().unwrap_or(0.0);
            let c_hw = params.get("C").copied().unwrap_or(0.0);
            if q <= 0.0 || !q.is_finite() { return Err(format!("assertion failed: Q_gpm={} must be positive and finite", q)); }
            if d < 0.75 { return Err(format!("assertion failed: D_in={} below minimum -- NFPA 13 branch min 1in; absolute min 3/4in (D_in=0.75)", d)); }
            if d > 36.0 { return Err(format!("assertion failed: D_in={} exceeds max (36in). Typical fire mains: 4-12in.", d)); }
            if l <= 0.0 || !l.is_finite() { return Err(format!("assertion failed: L_ft={} must be positive -- pipe equivalent length in feet", l)); }
            if l > 100_000.0 { return Err(format!("assertion failed: L_ft={} exceeds 100,000 ft. Segment the hydraulic calculation.", l)); }
            if elv < 0.0 { return Err(format!("assertion failed: elev_ft={} must be >= 0 -- elevation from pump to highest head in feet (0 for single-story)", elv)); }
            if elv > 2_000.0 { return Err(format!("assertion failed: elev_ft={} exceeds 2,000 ft (~200 stories). Split into pump zones.", elv)); }
            if pr < 0.0 { return Err(format!("assertion failed: P_residual={} PSI must be >= 0. NFPA 13: 15 PSI min; NFPA 14 standpipe: 65 PSI min", pr)); }
            if eff <= 0.0 || eff > 1.0 { return Err(format!("assertion failed: eff={} must be in (0, 1.0] -- typical centrifugal pump: 0.60-0.85", eff)); }
            if c_hw < 80.0 || c_hw > 160.0 { return Err(format!("assertion failed: C={} out of range [80,160]. C=80: corroded steel; C=100: old steel; C=120: new steel; C=150: PVC/CPVC", c_hw)); }
        }
        _ => {}
    }
    Ok(())
}

fn route_request(
    method:  &str,
    path:    &str,
    body:    &str,
    vm:      &Arc<Mutex<Vm>>,
    _prog:   &Program,
    plans:   &Arc<Vec<PlanDecl>>,
    jit_map: &Arc<Option<plan::JitFnMap>>,
) -> (&'static str, String) {
    match (method, path) {

        // ── Health ────────────────────────────────────────────────
        ("GET", "/health") => {
            let n_plans = plans.len();
            let has_jit = jit_map.is_some();
            let turbo_n = AOT_CACHE.get().map(|a| a.turbo_count()).unwrap_or(0);
            let (wstatus, _) = crate::server::watchdog_assess();
            // CPU feature detection for FMA consistency audit
            let has_fma  = is_x86_feature_detected!("fma");
            let has_avx2 = is_x86_feature_detected!("avx2");
            let no_fma   = NO_FMA_MODE.load(Ordering::Relaxed);
            let fma_path = if no_fma { "VMULSD+VADDSD (CRYS_NO_FMA)" } else if has_fma { "VFMADD231SD" } else { "VMULSD+VADDSD" };
            // -0.0 canonicalization: active
            // DAZ/FTZ: checked at runtime (MXCSR bits 6,15)
            ("200 OK", format!(
                r#"{{"status":"{}","lang":"CRYS-L","version":"3.2","plans":{},"jit":{},"turbo":{},"watchdog":"{}","cpu":{{"fma":{},"avx2":{},"fma_path":"{}","zero_canon":true,"daz_active":false,"nan_shield":"avx2+fma_branchless","rounding":"FE_TONEAREST","no_fma":{}}}}}"#,
                if wstatus == "healthy" { "ok" } else { wstatus },
                n_plans, has_jit, turbo_n, wstatus,
                has_fma, has_avx2, fma_path, no_fma
            ))
        }

        // ── Plan list ─────────────────────────────────────────────
        ("GET", "/plans") => {
            let entries: Vec<String> = plans.iter()
                .map(|p| format!(r#"{{"name":"{}","params":[{}]}}"#,
                    p.name,
                    p.params.iter().map(|pm| format!("\"{}\"", pm.name)).collect::<Vec<_>>().join(",")
                ))
                .collect();
            ("200 OK", format!(r#"{{"plans":[{}]}}"#, entries.join(",")))
        }

        // ── Cognitive Memory ─────────────────────────────────────
        ("GET", "/memory") => {
            let recent = crate::cognitive_memory::load_recent(20);
            let count  = crate::cognitive_memory::count();
            let entries = recent.iter().rev()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(",");
            ("200 OK", format!(
                r#"{{"ok":true,"total":{},"recent":[{}]}}"#, count, entries
            ))
        }

        // POST /memory/recall — find similar past experience
        ("POST", "/memory/recall") => {
            let q = extract_json_str(body, "q").unwrap_or_default();
            let hint = crate::cognitive_memory::recall_hint(&q);
            let count = crate::cognitive_memory::count();
            let recent = crate::cognitive_memory::load_recent(3);
            let entries = recent.iter().rev()
                .map(|s| s.as_str()).collect::<Vec<_>>().join(",");
            match hint {
                Some(h) => ("200 OK", format!(
                    r#"{{"ok":true,"hint":"{}","total":{},"recent":[{}]}}"#,
                    h.replace('"', "'"), count, entries
                )),
                None => ("200 OK", format!(
                    r#"{{"ok":true,"hint":null,"total":{},"recent":[{}]}}"#,
                    count, entries
                )),
            }
        }

        // ── Level 3: Turbo execution (zero-overhead) ──────────────
        // POST /plan/turbo
        // Body: {"plan":"plan_name","params":[1.0, 2.0, 3.0]}  (ordered array!)
        ("POST", "/plan/turbo") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan' field"}"#.into()),
            };
            let params = match extract_json_arr_float(body, "params") {
                Some(p) => p,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'params' array (ordered f64)"}"#.into()),
            };
            if let Some(aot) = AOT_CACHE.get() {
                if let Some(tidx) = aot.turbo_index(&plan_name) {
                    let t0 = std::time::Instant::now();
                    if let Some((results, n)) = aot.execute_turbo(tidx, &params) {
                        let ns = t0.elapsed().as_nanos() as f64;
                        let tp = aot.turbo_plan(tidx).unwrap();
                        let n = n.min(tp.n_steps);
                        // Build results in declaration order
                        let mut ordered: Vec<(usize, usize)> = (0..n).map(|i| (tp.exec_to_decl[i], i)).collect();
                        ordered.sort_by_key(|&(d, _)| d);
                        let steps_str: Vec<String> = ordered.iter().map(|&(_, ei)| {
                            format!(r#""{}""#, tp.step_names[ei])
                        }).collect();
                        let results_ordered: Vec<String> = ordered.iter().map(|&(_, ei)| {
                            let v = results[ei];
                            if v == v.floor() && v.abs() < 1e15 { format!("{:.1}", v) }
                            else { format!("{:.6}", v) }
                        }).collect();
                        return ("200 OK", format!(
                            r#"{{"ok":true,"level":3,"ns":{:.1},"results":[{}],"steps":[{}]}}"#,
                            ns, results_ordered.join(","), steps_str.join(",")
                        ));
                    }
                }
            }
            ("400 Bad Request", format!(r#"{{"ok":false,"error":"plan \'{}\' not in turbo table"}}"#, plan_name))
        }

        // ── Level 3: Benchmark ────────────────────────────────────
        // POST /plan/bench
        // Body: {"plan":"plan_name","params":[1.0,2.0,3.0],"iterations":100000}
        ("POST", "/plan/bench") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan' field"}"#.into()),
            };
            let params = match extract_json_arr_float(body, "params") {
                Some(p) => p,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'params' array"}"#.into()),
            };
            let iterations = extract_json_float(body, "iterations")
                .map(|v| v as usize)
                .unwrap_or(10000)
                .max(100).min(10_000_000);

            if let Some(aot) = AOT_CACHE.get() {
                if let Some(tidx) = aot.turbo_index(&plan_name) {
                    if let Some(avg_ns) = aot.bench_turbo(tidx, &params, iterations) {
                        let total_ms = (avg_ns * iterations as f64) / 1_000_000.0;
                        let l4_ns = aot.bench_register(tidx, &params, iterations);
                        let l4_str = match l4_ns {
                            Some(ns) => format!("{:.2}", ns),
                            None => "null".to_string(),
                        };
                        let speedup = match l4_ns {
                            Some(ns) if ns > 0.0 => format!("{:.1}x", avg_ns / ns),
                            _ => "null".to_string(),
                        };
                        return ("200 OK", format!(
                            r#"{{"ok":true,"plan":"{}","iterations":{},"avg_ns":{:.2},"l4_avg_ns":{},"speedup":"{}","total_ms":{:.3}}}"#,
                            plan_name, iterations, avg_ns, l4_str, speedup, total_ms
                        ));
                    }
                }
            }
            ("400 Bad Request", format!(r#"{{"ok":false,"error":"plan \'{}\' not in turbo table"}}"#, plan_name))
        }

        // ── Plan execution ────────────────────────────────────────
        // POST /plan/execute
        // Body: {"plan":"plan_name","params":{"area":1200,"K":5.6}}
        ("POST", "/plan/execute") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan' field"}"#.into()),
            };
            let params = match extract_json_obj_float(body, "params") {
                Some(p) => p,
                None    => std::collections::HashMap::new(),
            };

            // ── Physics guard ──
            if let Err(e) = validate_plan_physics(&plan_name, &params) {
                return ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e));
            }

            // ── AOT fast path ──
            if let Some(aot) = AOT_CACHE.get() {
                if aot.has_plan(&plan_name) {
                    if let Ok(r) = aot.execute(&plan_name, &params) {
                        let explain_flag = body.contains("\"explain\":true") || body.contains("\"explain\": true");
                        if explain_flag {
                            let explained = steps_with_explanations(&r.steps, &params);
                            let result_json = format!(
                                r#"{{"plan":"{}","steps":{},"total_ns":{:.1},"cache_hits":{}}}"#,
                                r.plan_name, explained, r.total_ns, r.cache_hits
                            );
                            return ("200 OK", format!(r#"{{"ok":true,"plan":"{}","result":{}}}"#, plan_name, result_json));
                        }
                        let json_out = r.to_json();
                        return ("200 OK", format!(r#"{{"ok":true,"plan":"{}","result":{}}}"#, plan_name, json_out));
                    }
                }
            }
            let executor = PlanExecutor::new(plans.as_slice());
            let executor = match jit_map.as_ref() {
                Some(map) => executor.with_jit_map(map.clone()),
                None      => executor,
            };

            let explain_flag = body.contains("\"explain\":true") || body.contains("\"explain\": true");
            match executor.execute(&plan_name, params.clone()) {
                Ok(r) => {
                    if explain_flag {
                        let explained = steps_with_explanations(&r.steps, &params);
                        let result_json = format!(
                            r#"{{"plan":"{}","steps":{},"total_ns":{:.1},"cache_hits":{}}}"#,
                            r.plan_name, explained, r.total_ns, r.cache_hits
                        );
                        ("200 OK", format!(r#"{{"ok":true,"plan":"{}","result":{}}}"#, plan_name, result_json))
                    } else {
                        ("200 OK", format!(r#"{{"ok":true,"plan":"{}","result":{}}}"#, plan_name, r.to_json()))
                    }
                }
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
            }
        }

        // ── Intent → Plan → Execute ───────────────────────────────
        // POST /intent
        // Body: {"q":"Disenar sistema incendios almacen 1200 m2 K=5.6"}
        ("POST", "/intent") => {
            let q = match extract_json_str(body, "q") {
                Some(q) => q,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'q' field"}"#.into()),
            };

            // ── "What if" / conversational context detection ──────────
            let session_key = extract_json_str(body, "session").unwrap_or_else(|| "default".to_string());

            if is_modify_intent(&q) {
                let ctx_hit = intent_ctx_map().get(&session_key).map(|ctx| {
                    let known: Vec<String> = ctx.last_params.keys().cloned().collect();
                    let changes = extract_param_changes(&q, &known);
                    (ctx.last_plan.clone(), ctx.last_params.clone(), ctx.last_domain.clone(), changes)
                });
                if let Some((plan_name, old_params, domain, changes)) = ctx_hit {
                    if !changes.is_empty() {
                        let mut new_params = old_params.clone();
                        for (k, v) in &changes {
                            new_params.insert(k.clone(), *v);
                        }
                        // Execute with modified params
                        // ── Physics guard ──
                        if let Err(e) = validate_plan_physics(&plan_name, &new_params) {
                            return ("400 Bad Request", format!(r#"{{"ok":false,"modified":true,"error":"{}"}}"#, e));
                        }
                        let aot_result: Option<crate::plan::PlanResult> = AOT_CACHE.get()
                            .and_then(|aot| if aot.has_plan(&plan_name) { aot.execute(&plan_name, &new_params).ok() } else { None });
                        let exec_result = if let Some(r) = aot_result {
                            Ok(r)
                        } else {
                            let executor = PlanExecutor::new(plans.as_slice());
                            let executor = match jit_map.as_ref() {
                                Some(map) => executor.with_jit_map(map.clone()),
                                None      => executor,
                            };
                            executor.execute(&plan_name, new_params.clone())
                        };
                        match exec_result {
                            Ok(r) => {
                                let human  = escape_json(&r.to_human());
                                let result = r.to_json();
                                // Build changes JSON
                                let changes_json: Vec<String> = changes.iter()
                                    .map(|(k,v)| format!("\"{}\":{}", k, v))
                                    .collect();
                                let prev_json: Vec<String> = changes.keys()
                                    .filter_map(|k| old_params.get(k).map(|v| format!("\"{}\":{}", k, v)))
                                    .collect();
                                // Update context
                                intent_ctx_map().insert(session_key, IntentContext {
                                    last_plan: plan_name.clone(),
                                    last_params: new_params,
                                    last_domain: domain.clone(),
                                    timestamp: std::time::Instant::now(),
                                });
                                return ("200 OK", format!(
                                    r#"{{"ok":true,"modified":true,"domain":"{}","query":"{}","plan":"{}","human":"{}","changes":{{{}}},"previous":{{{}}},"result":{}}}"#,
                                    domain, q, plan_name, human,
                                    changes_json.join(","), prev_json.join(","), result
                                ));
                            }
                            Err(e) => {
                                return ("400 Bad Request", format!(
                                    r#"{{"ok":false,"modified":true,"error":"plan re-exec: {}"}}"#, e
                                ));
                            }
                        }
                    }
                }
            }

            let backend: Box<dyn intent_parser::LlmBackend> = Box::new(MockBackend);
            let parser = intent_parser::IntentParser::new(backend);
            let available: Vec<String> = plans.iter().map(|p| p.name.clone()).collect();

            match parser.parse(&q) {
                Ok(intent) => {
                    let domain = intent.domain.as_str().to_string();

                    // ── Cognitive Compiler: Loop/Simulation path ──────────
                    // If structure detector identified a loop/simulation
                    if domain == "cognitive_loop" {
                        // Extract loop spec from intent.params (encoded as floats)
                        let oracle    = intent.plan_name.as_deref().unwrap_or("hazen_P_at_gpm");
                        let loop_pos  = intent.params.get("loop_pos").copied().unwrap_or(0.0) as usize;
                        let r_start   = intent.params.get("range_start").copied().unwrap_or(100.0);
                        let r_end     = intent.params.get("range_end").copied().unwrap_or(1000.0);
                        let step_v    = intent.params.get("step").copied().unwrap_or(10.0);
                        let cond_val  = intent.params.get("cond_val").copied().unwrap_or(65.0);
                        let n_fixed   = intent.params.get("n_fixed").copied().unwrap_or(0.0) as usize;
                        let fixed_args: Vec<f64> = (0..n_fixed)
                            .filter_map(|i| intent.params.get(&format!("f{}", i)).copied())
                            .collect();
                        // Extract string metadata encoded in constraints
                        let get_meta = |prefix: &str| -> String {
                            intent.constraints.iter()
                                .find(|c| c.starts_with(prefix))
                                .and_then(|c| c.splitn(2, '=').nth(1))
                                .unwrap_or("")
                                .to_string()
                        };
                        let cond_op      = get_meta("cond_op");
                        let cond_op_str  = if cond_op.is_empty() { "<" } else { &cond_op };
                        let loop_label   = get_meta("loop_label");
                        let result_label = get_meta("result_label");
                        let loop_unit    = get_meta("loop_unit");
                        let result_unit  = get_meta("result_unit");
                        let title        = get_meta("title");

                        let executor = PlanExecutor::new(plans.as_slice());
                        let executor = match jit_map.as_ref() {
                            Some(map) => executor.with_jit_map(map.clone()),
                            None      => executor,
                        };
                        match executor.execute_loop(
                            oracle, loop_pos, &fixed_args,
                            r_start, r_end, step_v,
                            cond_op_str, cond_val.abs()
                        ) {
                            Ok(lr) => {
                                let human = lr.to_human(
                                    &loop_label, &result_label, &loop_unit, &result_unit,
                                    cond_op_str, cond_val.abs(), &title
                                );
                                let human_escaped = escape_json(&human);
                                // ── Cognitive Memory: persist this experience ──
                                let result_summary = match lr.critical_point {
                                    Some((qv, rv)) => format!("critical@{:.1}{}={:.3}{}",
                                        qv, loop_unit, rv, result_unit),
                                    None => format!("no_critical/{}_iters", lr.results.len()),
                                };
                                crate::cognitive_memory::save_experience(
                                    &crate::cognitive_memory::Experience {
                                        query:          &q,
                                        structure:      "loop",
                                        plan:           &format!("loop:{}", oracle),
                                        oracle,
                                        result_summary: &result_summary,
                                        success:        lr.critical_point.is_some(),
                                        score:          if lr.critical_point.is_some() { 0.85 } else { 0.50 },
                                        elapsed_us:     lr.total_ns / 1000.0,
                                    }
                                );
                                return ("200 OK", format!(
                                    r#"{{"ok":true,"domain":"cognitive_loop","query":"{}","plan":"loop:{}","human":"{}","result":{{"iters":{},"critical":{}}}}}"#,
                                    q, oracle, human_escaped,
                                    lr.results.len(),
                                    lr.critical_point.map(|(q,v)| format!("[{:.1},{:.3}]", q, v))
                                        .unwrap_or("null".to_string())
                                ));
                            }
                            Err(e) => {
                                return ("400 Bad Request", format!(
                                    r#"{{"ok":false,"error":"loop executor: {}"}}"#, e
                                ));
                            }
                        }
                    }
                    // ─────────────────────────────────────────────────────

                    let routed = intent_parser::route_to_plan(&intent, &available);
                    match routed {
                        Some(plan_name) => {
                            // Auto-populate params from intent
                            let params = intent.params.clone();
                            // ── Physics guard ──
                            if let Err(e) = validate_plan_physics(&plan_name, &params) {
                                return ("400 Bad Request", format!(r#"{{\"ok\":false,\"domain\":\"{}\",\"error\":\"{}\"}}"#, domain, e));
                            }
                            // ── AOT fast path ──
                            let aot_result: Option<crate::plan::PlanResult> = AOT_CACHE.get()
                                .and_then(|aot| if aot.has_plan(&plan_name) { aot.execute(&plan_name, &params).ok() } else { None });
                            let exec_result = if let Some(r) = aot_result {
                                Ok(r)
                            } else {
                                let executor = PlanExecutor::new(plans.as_slice());
                                let executor = match jit_map.as_ref() {
                                    Some(map) => executor.with_jit_map(map.clone()),
                                    None      => executor,
                                };
                                executor.execute(&plan_name, params.clone())
                            };
                            match exec_result {
                                Ok(r) => {
                                    let human   = escape_json(&r.to_human());
                                    let result  = r.to_json();
                                    // ── Cognitive Memory ──
                                    let steps = r.steps.len();
                                    crate::cognitive_memory::save_experience(
                                        &crate::cognitive_memory::Experience {
                                            query:          &q,
                                            structure:      "plan",
                                            plan:           &plan_name,
                                            oracle:         &plan_name,
                                            result_summary: &format!("steps={}", steps),
                                            success:        true,
                                            score:          0.80,
                                            elapsed_us:     r.total_ns as f64 / 1000.0,
                                        }
                                    );
                                    // Save session context for "what if" follow-ups
                                    intent_ctx_map().insert(session_key.clone(), IntentContext {
                                        last_plan: plan_name.to_string(),
                                        last_params: params.clone(),
                                        last_domain: domain.clone(),
                                        timestamp: std::time::Instant::now(),
                                    });
                                    // Always include domain so callers can route
                                    // Decision Engine auto-analysis
                                    let decision_steps: Vec<(String, String, f64)> = r.steps.iter()
                                        .map(|s| (s.step.clone(), s.oracle.clone(), s.value)).collect();
                                    let decision_json = decision_analyze(&plan_name, &decision_steps, r.total_ns as u64);
                                    ("200 OK", format!(
                                        r#"{{"ok":true,"domain":"{}","query":"{}","plan":"{}","human":"{}","result":{},"decision":{}}}"#,
                                        domain, q, plan_name, human, result, decision_json
                                    ))
                                }
                                Err(e) => ("400 Bad Request", format!(
                                    r#"{{"ok":false,"domain":"{}","error":"plan: {}"}}"#, domain, e
                                )),
                            }
                        }
                        None => ("200 OK", format!(
                            // Always include domain — critical for Qomni's intent router
                            r#"{{"ok":true,"domain":"{}","query":"{}","plan":null,"note":"no matching plan found"}}"#,
                            domain, q
                        )),
                    }
                }
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"intent: {}"}}"#, e)),
            }
        }

        // ── VM query ──────────────────────────────────────────────
        ("POST", "/query") => {
            let q = extract_json_str(body, "q").unwrap_or_default();
            let mut vm = vm.lock().unwrap();
            match vm.query(&q) {
                Ok(result) => {
                    let escaped = escape_json(&result);
                    ("200 OK", format!(r#"{{"ok":true,"query":"{}","result":"{}"}}"#, q, escaped))
                }
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
            }
        }

        // ── Eval ──────────────────────────────────────────────────
        ("POST", "/eval") => {
            let expr_src = extract_json_str(body, "expr").unwrap_or_default();
            let src = format!("let __result = {}
-> respond(__result)
", expr_src);
            let mut lexer  = Lexer::new(&src);
            let tokens     = lexer.tokenize();
            let mut parser = Parser::new(tokens);
            match parser.parse() {
                Ok(prog) => {
                    let mut vm = vm.lock().unwrap();
                    match vm.run(&prog) {
                        Ok(out) => {
                            let val     = out.join(", ");
                            let escaped = escape_json(&val);
                            ("200 OK", format!(r#"{{"ok":true,"expr":"{}","result":"{}"}}"#, expr_src, escaped))
                        }
                        Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
                    }
                }
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"parse: {}"}}"#, e)),
            }
        }

        // ── Unit Conversion ───────────────────────────────────────
        // POST /convert
        // Body: {"value":100,"from":"ft2","to":"m2"}
        ("POST", "/convert") => {
            let value = match extract_json_float(body, "value") {
                Some(v) => v,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'value'"}"#.into()),
            };
            let from = match extract_json_str(body, "from") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'from'"}"#.into()),
            };
            let to = match extract_json_str(body, "to") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'to'"}"#.into()),
            };
            match crate::units::convert(value, &from, &to) {
                Ok(result) => ("200 OK", format!(
                    r#"{{"ok":true,"value":{value},"from":"{from}","to":"{to}","result":{result:.6}}}"#
                )),
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
            }
        }

        // ── Web Fetch ─────────────────────────────────────────────
        // POST /web/fetch
        // Body: {"url":"https://example.com","selector":"optional CSS selector or keyword"}
        // Returns raw text content (stripped of HTML tags, max 8KB)
        ("POST", "/web/fetch") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'url'"}"#.into()),
            };

            // Security: only allow http/https, no local IPs
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return ("400 Bad Request", r#"{"ok":false,"error":"Only http/https URLs allowed"}"#.into());
            }
            // Block local/private addresses
            let blocked = ["localhost","127.0.0.1","0.0.0.0","::1","10.","192.168.","172.16.","169.254."];
            if blocked.iter().any(|b| url.contains(b)) {
                return ("403 Forbidden", r#"{"ok":false,"error":"Private/local URLs not allowed"}"#.into());
            }

            let selector = extract_json_str(body, "selector");

            match ureq::get(&url)
                .set("User-Agent", "CRYS-L/2.2 (Qomni AI Lab)")
                .timeout(std::time::Duration::from_secs(10))
                .call()
            {
                Ok(resp) => {
                    let raw = resp.into_string().unwrap_or_default();
                    // Strip HTML tags — simple regex-free approach
                    let text = strip_html(&raw);
                    // Apply selector filter (keyword search)
                    let content = if let Some(ref kw) = selector {
                        extract_around_keyword(&text, kw, 2000)
                    } else {
                        text.chars().take(8000).collect()
                    };
                    let escaped = escape_json(&content);
                    ("200 OK", format!(
                        r#"{{"ok":true,"url":"{}","length":{},"content":"{}"}}"#,
                        url, content.len(), escaped
                    ))
                }
                Err(e) => ("502 Bad Gateway", format!(r#"{{"ok":false,"error":"fetch: {}"}}"#, e)),
            }
        }


        // ── Web Security Check ────────────────────────────────────────────
        // POST /web/security-check
        ("POST", "/web/security-check") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return ("400 Bad Request", r#"{"ok":false,"error":"Only http/https"}"#.into());
            }
            let blocked = ["localhost","127.0.0.1","0.0.0.0","::1","10.","192.168.","172.16."];
            if blocked.iter().any(|b| url.contains(b)) {
                return ("403 Forbidden", r#"{"ok":false,"error":"Private URL"}"#.into());
            }
            match ureq::get(&url)
                .set("User-Agent", "CRYS-L Security Scanner/2.2")
                .timeout(std::time::Duration::from_secs(10))
                .call()
            {
                Ok(resp) => {
                    let status = resp.status();
                    let gh = |n: &str| resp.header(n).unwrap_or("").to_string();
                    let csp    = gh("content-security-policy");
                    let hsts   = gh("strict-transport-security");
                    let xframe = gh("x-frame-options");
                    let xcto   = gh("x-content-type-options");
                    let server = gh("server");
                    let referr = gh("referrer-policy");
                    let mut findings: Vec<&str> = Vec::new();
                    let mut score: i32 = 100;
                    if csp.is_empty()    { findings.push("CSP-MISSING");    score -= 30; }
                    if hsts.is_empty()   { findings.push("HSTS-MISSING");   score -= 20; }
                    if xframe.is_empty() { findings.push("XFRAME-MISSING"); score -= 10; }
                    if xcto.is_empty()   { findings.push("XCTO-MISSING");   score -= 10; }
                    if !server.is_empty() { findings.push("SERVER-LEAK");   score -= 10; }
                    let fj = findings.iter().map(|f| format!("\"{}\"", f)).collect::<Vec<_>>().join(",");
                    let clean = |s: &str| s.replace('"', "\\\"");
                    ("200 OK", format!(
                        "{{\"ok\":true,\"status\":{},\"score\":{},\"csp\":\"{}\",\"hsts\":\"{}\",\"xframe\":\"{}\",\"xcto\":\"{}\",\"server\":\"{}\",\"referrer\":\"{}\",\"findings\":[{}]}}",
                        status, score, clean(&csp), clean(&hsts), clean(&xframe), clean(&xcto), clean(&server), clean(&referr), fj
                    ))
                }
                Err(e) => ("502 Bad Gateway", format!("{{\"ok\":false,\"error\":\"{}\"}}", e)),
            }
        }

        // POST /patch/nginx/apply — Hot-apply nginx security header snippet (nginx -s reload, zero downtime)
        // Body: {"id":"csp","directive":"add_header Content-Security-Policy \"...\" always;","domain":"optional"}
        // Requires QOMNI_PATCH_ENABLED=1 env var to be active.
        ("POST", "/patch/nginx/apply") => {
            use std::process::Command;
            use std::env;

            // Kill switch: QOMNI_PATCH_ENABLED must be "1"
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Patch engine disabled. Set QOMNI_PATCH_ENABLED=1 to enable.","hint":"systemctl edit crysl-nfpa --force"}"#.into());
            }

            let patch_id = extract_json_str(body, "id").unwrap_or_else(|| "patch".to_string());
            let directive = match extract_json_str(body, "directive") {
                Some(d) => d,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing directive"}"#.into()),
            };

            // Whitelist allowed directives (prevent arbitrary nginx config injection)
            let allowed_prefixes = [
                "add_header ", "server_tokens ", "more_clear_headers ",
                "proxy_hide_header ", "expires ", "charset ",
            ];
            if !allowed_prefixes.iter().any(|p| directive.trim_start().starts_with(p)) {
                return ("400 Bad Request", r#"{"ok":false,"error":"Directive not in whitelist. Only add_header, server_tokens, etc."}"#.into());
            }

            // Sanitize patch_id for filename
            let safe_id: String = patch_id.chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .take(32)
                .collect();

            // Write conf snippet
            let conf_path = format!("/etc/nginx/conf.d/qomni-patch-{}.conf", safe_id);
            let conf_content = format!(
                "# Qomni Auto-Patch: {} — applied {}\n# Disable: set QOMNI_PATCH_ENABLED=0 or rm {}\nserver_name_in_redirect off;\n# Header patch:\nadd_header X-Qomni-Patched \"{}\" always;\n{}\n",
                safe_id,
                chrono_now_approx(),
                conf_path,
                safe_id,
                directive
            );

            match std::fs::write(&conf_path, &conf_content) {
                Err(e) => return ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"write failed: {}"}}"#, e)),
                Ok(_) => {}
            }

            // nginx -t test
            let test = Command::new("nginx").arg("-t").output();
            match test {
                Ok(out) if out.status.success() => {},
                Ok(out) => {
                    // Config invalid — remove and reject
                    let _ = std::fs::remove_file(&conf_path);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let err_clean: String = stderr.chars()
                        .filter(|c| c.is_ascii_graphic() || *c == ' ')
                        .take(200).collect();
                    return ("400 Bad Request", format!(r#"{{"ok":false,"error":"nginx -t failed: {}"}}"#, err_clean));
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&conf_path);
                    return ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"nginx not found: {}"}}"#, e));
                }
            }

            // nginx -s reload (graceful — zero downtime, keeps existing connections alive)
            let reload = Command::new("nginx").args(&["-s", "reload"]).output();
            match reload {
                Ok(out) if out.status.success() => {
                    ("200 OK", format!(
                        r#"{{"ok":true,"message":"Patch applied + nginx reloaded (zero downtime)","conf":"{}","directive":"{}"}}"#,
                        conf_path, directive.replace('"', "'")
                    ))
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"reload failed: {}"}}"#,
                        stderr.chars().filter(|c| c.is_ascii_graphic()||*c==' ').take(100).collect::<String>()))
                }
                Err(e) => ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
            }
        }

        // POST /patch/nginx/rollback — Remove a previously applied patch
        // Body: {"id":"csp"}
        ("POST", "/patch/nginx/rollback") => {
            use std::process::Command;
            use std::env;

            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Patch engine disabled"}"#.into());
            }

            let patch_id = extract_json_str(body, "id").unwrap_or_default();
            let safe_id: String = patch_id.chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .take(32).collect();
            let conf_path = format!("/etc/nginx/conf.d/qomni-patch-{}.conf", safe_id);

            match std::fs::remove_file(&conf_path) {
                Ok(_) => {
                    let _ = Command::new("nginx").args(&["-s", "reload"]).output();
                    ("200 OK", format!(r#"{{"ok":true,"message":"Patch {} rolled back, nginx reloaded"}}"#, safe_id))
                }
                Err(e) => ("404 Not Found", format!(r#"{{"ok":false,"error":"{}"}}"#, e))
            }
        }

        // POST /system/mode — Master kill switch for audit/patch features
        // Body: {"mode":"safe"} or {"mode":"audit"} — requires auth header
        ("POST", "/system/mode") => {
            use std::env;
            // Check for commander key
            let req_key = extract_json_str(body, "key").unwrap_or_default();
            let sys_key = env::var("QOMNI_COMMANDER_KEY").unwrap_or_else(|_| "qomni-admin-2026".to_string());
            if req_key != sys_key {
                return ("403 Forbidden", r#"{"ok":false,"error":"Invalid commander key"}"#.into());
            }
            let mode = extract_json_str(body, "mode").unwrap_or_default();
            match mode.as_str() {
                "safe" => {
                    // Disable patch engine by removing marker file
                    let _ = std::fs::write("/tmp/qomni-mode", "safe");
                    ("200 OK", r#"{"ok":true,"mode":"safe","message":"Patch engine DISABLED. Audit read-only."}"#.into())
                }
                "audit" => {
                    let _ = std::fs::write("/tmp/qomni-mode", "audit");
                    ("200 OK", r#"{"ok":true,"mode":"audit","message":"Audit mode ACTIVE. Crawler + patch engine enabled."}"#.into())
                }
                _ => ("400 Bad Request", r#"{"ok":false,"error":"mode must be 'safe' or 'audit'"}"#.into())
            }
        }


        // POST /web/crawl — Ojo de Dios: path discovery + form detection + sensitive file probe
        // Body: {"url":"https://target.com","depth":1}
        // Use only on systems you own or have written authorization to test.
        ("POST", "/web/crawl") => {
            use std::collections::{HashSet, VecDeque};

            let base_url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
                return ("400 Bad Request", r#"{"ok":false,"error":"Only http/https"}"#.into());
            }
            // Block private/local addresses
            let blocked = ["localhost","127.0.0.1","0.0.0.0","::1","10.","192.168.","172.16.","169.254."];
            if blocked.iter().any(|b| base_url.contains(b)) {
                return ("403 Forbidden", r#"{"ok":false,"error":"Private URL not allowed"}"#.into());
            }

            // Extract base domain for scope
            let base_domain: String = {
                let after_proto = base_url.trim_start_matches("https://").trim_start_matches("http://");
                after_proto.split('/').next().unwrap_or("").to_string()
            };
            let base_origin = format!("https://{}", base_domain);

            // ── Sensitive paths to probe ─────────────────────────────────────
            let probe_paths: &[&str] = &[
                "/.git/HEAD", "/.git/config", "/.env", "/.env.production",
                "/wp-config.php", "/wp-login.php", "/wp-json/wp/v2/users",
                "/admin", "/admin/login", "/administrator", "/phpmyadmin",
                "/api/v1/", "/api/v2/", "/api/users", "/api/admin",
                "/config.php", "/config.json", "/database.yml",
                "/backup.sql", "/dump.sql", "/backup.zip",
                "/server-status", "/server-info",
                "/robots.txt", "/sitemap.xml", "/.well-known/security.txt",
                "/xmlrpc.php", "/cgi-bin/", "/shell.php", "/cmd.php",
            ];

            // ── Fetch base page ──────────────────────────────────────────────
            let mut discovered_paths: Vec<String> = Vec::new();
            let mut forms: Vec<String> = Vec::new();
            let mut crawl_findings: Vec<String> = Vec::new();

            let base_resp = ureq::get(&base_url)
                .set("User-Agent", "Qomni-Crawler/1.0 (authorized security audit)")
                .timeout(std::time::Duration::from_secs(10))
                .call();

            let base_html = match base_resp {
                Ok(resp) => resp.into_string().unwrap_or_default(),
                Err(e) => {
                    return ("502 Bad Gateway", format!(r#"{{"ok":false,"error":"{}"}}"#, e));
                }
            };

            // ── Extract links from HTML ──────────────────────────────────────
            let mut seen: HashSet<String> = HashSet::new();
            let mut queue: VecDeque<String> = VecDeque::new();

            // Simple href extractor
            let mut pos = 0;
            while let Some(href_start) = base_html[pos..].find("href=\"") {
                let start = pos + href_start + 6;
                let rest = &base_html[start..];
                let end = rest.find('"').unwrap_or(rest.len());
                let href = &rest[..end];

                let full_url = if href.starts_with("http://") || href.starts_with("https://") {
                    if href.contains(&base_domain) { href.to_string() } else { pos = start + end; continue; }
                } else if href.starts_with('/') {
                    format!("{}{}", base_origin, href)
                } else if href.starts_with('#') || href.starts_with("mailto:") || href.starts_with("javascript:") {
                    pos = start + end; continue;
                } else {
                    format!("{}/{}", base_origin, href)
                };

                // Extract just the path
                let path = full_url.trim_start_matches("https://").trim_start_matches("http://")
                    .trim_start_matches(&base_domain)
                    .to_string();
                let path = if path.is_empty() { "/".to_string() } else { path };

                if seen.insert(path.clone()) && seen.len() < 40 {
                    discovered_paths.push(path.clone());
                    queue.push_back(full_url);
                }
                pos = start + end;
            }

            // ── Extract forms ────────────────────────────────────────────────
            pos = 0;
            while let Some(form_start) = base_html[pos..].find("<form") {
                let start = pos + form_start;
                let rest = &base_html[start..];
                let end = rest.find("</form>").unwrap_or(rest.len().min(2000));
                let form_html = &rest[..end];

                // Extract action
                let action = if let Some(a) = form_html.find("action=\"") {
                    let s = a + 8;
                    let e = form_html[s..].find('"').unwrap_or(0);
                    form_html[s..s+e].to_string()
                } else { "/".to_string() };

                // Extract method
                let method = if form_html.to_lowercase().contains("method=\"post\"") { "POST" } else { "GET" };

                // Extract input names
                let mut inputs: Vec<String> = Vec::new();
                let mut ipos = 0;
                while let Some(inp) = form_html[ipos..].find("name=\"") {
                    let is = ipos + inp + 6;
                    let ie = form_html[is..].find('"').unwrap_or(0);
                    let name = &form_html[is..is+ie];
                    // Flag sensitive field names
                    let is_sensitive = ["password","passwd","pass","pwd","token","secret","key","credit","card","cvv"]
                        .iter().any(|s| name.to_lowercase().contains(s));
                    if is_sensitive {
                        inputs.push(format!("{}[SENSITIVE]", name));
                        crawl_findings.push(format!("FORM-SENSITIVE-INPUT: {} field '{}' in form action='{}'", method, name, action));
                    } else {
                        inputs.push(name.to_string());
                    }
                    ipos = is + ie + 1;
                    if ipos >= form_html.len() { break; }
                }

                let inputs_str = inputs.iter().take(6).cloned().collect::<Vec<_>>().join(",");
                let clean_action = action.chars().filter(|c| c.is_ascii_graphic() && *c != '"').take(60).collect::<String>();
                forms.push(format!("{}:{} inputs=[{}]", method, clean_action, inputs_str));
                pos = start + end + 7;
                if pos >= base_html.len() { break; }
            }

            // ── Probe sensitive paths ────────────────────────────────────────
            struct ProbeResult { path: String, status: u16, finding: Option<String> }
            let mut probe_results: Vec<ProbeResult> = Vec::new();

            for path in probe_paths {
                let probe_url = format!("{}{}", base_origin, path);
                match ureq::get(&probe_url)
                    .set("User-Agent", "Qomni-Crawler/1.0 (authorized security audit)")
                    .timeout(std::time::Duration::from_secs(5))
                    .call()
                {
                    Ok(resp) => {
                        let status = resp.status();
                        let finding = if status == 200 {
                            let body_preview = resp.into_string().unwrap_or_default();
                            let body_preview = &body_preview[..body_preview.len().min(300)];
                            let critical = path.contains(".env") || path.contains(".git") ||
                                path.contains("config") || path.contains(".sql") ||
                                path.contains("backup");
                            if critical {
                                crawl_findings.push(format!("CRITICAL-EXPOSURE: {} returned 200 OK — data leak likely", path));
                                Some(format!("EXPOSED: {} bytes preview", body_preview.len()))
                            } else if path.contains("wp-login") || path.contains("admin") {
                                crawl_findings.push(format!("ADMIN-PANEL: {} accessible without auth", path));
                                Some("ADMIN-ACCESSIBLE".to_string())
                            } else {
                                None
                            }
                        } else { None };
                        probe_results.push(ProbeResult { path: path.to_string(), status, finding });
                    }
                    Err(_) => {
                        probe_results.push(ProbeResult { path: path.to_string(), status: 0, finding: None });
                    }
                }
            }

            // ── Serialize results ────────────────────────────────────────────
            let clean = |s: &str| s.chars()
                .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
                .collect::<String>()
                .replace('"', "'");

            let paths_json = discovered_paths.iter()
                .map(|p| format!("\"{}\"", clean(p)))
                .collect::<Vec<_>>().join(",");

            let forms_json = forms.iter()
                .map(|f| format!("\"{}\"", clean(f)))
                .collect::<Vec<_>>().join(",");

            let probe_json = probe_results.iter()
                .filter(|r| r.status > 0)
                .map(|r| {
                    let f = r.finding.as_deref().unwrap_or("");
                    format!("{{\"path\":\"{}\",\"status\":{},\"finding\":\"{}\"}}",
                        clean(&r.path), r.status, clean(f))
                })
                .collect::<Vec<_>>().join(",");

            let crit_json = crawl_findings.iter()
                .map(|f| format!("\"{}\"", clean(f)))
                .collect::<Vec<_>>().join(",");

            ("200 OK", format!(
                "{{\"ok\":true,\"domain\":\"{}\",\"pages_found\":{},\"forms_found\":{},\"paths\":[{}],\"forms\":[{}],\"probes\":[{}],\"critical_findings\":[{}]}}",
                clean(&base_domain),
                discovered_paths.len(),
                forms.len(),
                paths_json,
                forms_json,
                probe_json,
                crit_json
            ))
        }


        // POST /web/probe — Active security probe: SQLi, XSS, LFI, header injection
        // Body: {"url":"https://target.com/search?q=test","params":{"q":"test"},"method":"GET"}
        // ONLY runs on domains listed in /opt/qomni-staging/targets/authorized.txt
        // Requires QOMNI_PATCH_ENABLED=1 (audit mode)
        ("POST", "/web/probe") => {
            use std::process::Command;
            use std::env;

            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Probe engine requires AUDIT mode. Toggle SAFE→AUDIT in panel."}"#.into());
            }

            let target_url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            if !target_url.starts_with("http://") && !target_url.starts_with("https://") {
                return ("400 Bad Request", r#"{"ok":false,"error":"Only http/https"}"#.into());
            }

            // Extract domain for authorization check
            let domain: String = {
                let after_proto = target_url.trim_start_matches("https://").trim_start_matches("http://");
                after_proto.split('/').next().unwrap_or("").to_string()
            };

            // Check authorized domains list
            let authorized_raw = std::fs::read_to_string("/opt/qomni-staging/targets/authorized.txt")
                .unwrap_or_default();
            let is_authorized = authorized_raw.lines().any(|line| {
                let trimmed = line.trim();
                !trimmed.starts_with('#') && !trimmed.is_empty() && {
                    let auth_domain = trimmed.split('|').next().unwrap_or("").trim();
                    domain == auth_domain || domain.ends_with(&format!(".{}", auth_domain))
                }
            });

            if !is_authorized {
                return ("403 Forbidden", format!(
                    r#"{{"ok":false,"error":"Domain '{}' not in authorized list. Add to /opt/qomni-staging/targets/authorized.txt"}}"#,
                    domain.replace('"', "")
                ));
            }

            // ── Probe payloads ────────────────────────────────────────────────
            // SQLi: error-based detection (look for DB error strings in response)
            let sqli_payloads: &[(&str, &str)] = &[
                ("'", "SQLi-basic-quote"),
                ("' OR '1'='1", "SQLi-OR-bypass"),
                ("1 AND SLEEP(0)--", "SQLi-timing-marker"),
                ("' UNION SELECT NULL--", "SQLi-union"),
                ("\"; DROP TABLE users--", "SQLi-destructive-marker"),
            ];
            let sqli_error_signatures = [
                "sql syntax", "mysql_fetch", "ora-00933", "pg_query",
                "sqlstate", "sqlite_", "syntax error", "unclosed quotation",
                "you have an error in your sql", "warning: mysql",
                "microsoft ole db", "odbc drivers error",
            ];

            // XSS: reflection check
            let xss_payloads: &[(&str, &str)] = &[
                ("<script>qomni_xss</script>", "XSS-script-tag"),
                ("\"onmouseover=\"alert(1)\"", "XSS-event-attr"),
                ("javascript:alert(1)", "XSS-js-proto"),
                ("<img src=x onerror=qomni_xss>", "XSS-img-onerror"),
            ];

            // LFI: path traversal
            let lfi_payloads: &[(&str, &str)] = &[
                ("../../../etc/passwd", "LFI-etc-passwd"),
                ("..%2F..%2F..%2Fetc%2Fpasswd", "LFI-encoded"),
                ("/etc/passwd", "LFI-absolute"),
            ];

            // Header injection
            let header_payloads: &[(&str, &str)] = &[
                ("x-forwarded-for: 127.0.0.1", "HEADER-xff-spoof"),
                ("x-original-url: /admin", "HEADER-url-override"),
            ];

            // ── Parse existing params from URL ────────────────────────────────
            let (base_path, query_str) = if let Some(q) = target_url.split_once('?') {
                (q.0.to_string(), q.1.to_string())
            } else {
                (target_url.clone(), String::new())
            };

            // Parse params
            let params: Vec<(String, String)> = query_str.split('&')
                .filter(|s| s.contains('='))
                .map(|kv| {
                    let mut parts = kv.splitn(2, '=');
                    let k = parts.next().unwrap_or("\\").to_string();
                    let v = parts.next().unwrap_or("").to_string();
                    (k, v)
                })
                .filter(|(k,_)| !k.is_empty())
                .collect();

            let mut all_findings: Vec<String> = Vec::new();
            let mut probes_run: u32 = 0;

            // ── Run SQLi probes ───────────────────────────────────────────────
            for (param_key, original_val) in &params {
                for (payload, label) in sqli_payloads {
                    let test_qs: String = params.iter().map(|(k, v)| {
                        if k == param_key {
                            format!("{}={}", k, url_encode(payload))
                        } else {
                            format!("{}={}", k, v)
                        }
                    }).collect::<Vec<_>>().join("&");

                    let test_url = format!("{}?{}", base_path, test_qs);
                    probes_run += 1;

                    let start = std::time::Instant::now();
                    match ureq::get(&test_url)
                        .set("User-Agent", "Qomni-Probe/1.0 (authorized security audit)")
                        .timeout(std::time::Duration::from_secs(6))
                        .call()
                    {
                        Ok(resp) => {
                            let elapsed = start.elapsed().as_millis();
                            let body_text = resp.into_string().unwrap_or_default().to_lowercase();
                            let sql_error = sqli_error_signatures.iter().any(|s| body_text.contains(s));
                            if sql_error {
                                all_findings.push(format!(
                                    "{{\"type\":\"SQLI\",\"severity\":\"CRITICAL\",\"param\":\"{}\",\"payload\":\"{}\",\"label\":\"{}\",\"evidence\":\"DB error in response\",\"ms\":{}}}",
                                    param_key, clean_str(payload), label, elapsed
                                ));
                            } else if elapsed > 4000 && label.contains("timing") {
                                all_findings.push(format!(
                                    "{{\"type\":\"SQLI-TIMING\",\"severity\":\"HIGH\",\"param\":\"{}\",\"label\":\"{}\",\"evidence\":\"Response delay {}ms suggests blind SQLi\",\"ms\":{}}}",
                                    param_key, label, elapsed, elapsed
                                ));
                            }
                        }
                        Err(_) => {}
                    }
                }
            }

            // ── Run XSS reflection probes ─────────────────────────────────────
            for (param_key, _) in &params {
                for (payload, label) in xss_payloads {
                    let test_qs: String = params.iter().map(|(k, v)| {
                        if k == param_key {
                            format!("{}={}", k, url_encode(payload))
                        } else { format!("{}={}", k, v) }
                    }).collect::<Vec<_>>().join("&");
                    let test_url = format!("{}?{}", base_path, test_qs);
                    probes_run += 1;

                    match ureq::get(&test_url)
                        .set("User-Agent", "Qomni-Probe/1.0 (authorized security audit)")
                        .timeout(std::time::Duration::from_secs(5))
                        .call()
                    {
                        Ok(resp) => {
                            let body_text = resp.into_string().unwrap_or_default();
                            // Check if payload is reflected unencoded
                            if body_text.contains("<script>qomni_xss</script>")
                                || body_text.contains("onerror=qomni_xss")
                            {
                                all_findings.push(format!(
                                    "{{\"type\":\"XSS-REFLECTED\",\"severity\":\"HIGH\",\"param\":\"{}\",\"label\":\"{}\",\"evidence\":\"Payload reflected unencoded in response\"}}",
                                    param_key, label
                                ));
                            }
                        }
                        Err(_) => {}
                    }
                }
            }

            // ── Run LFI probes on path params ─────────────────────────────────
            for (param_key, _) in params.iter().filter(|(k,_)| {
                let kl = k.to_lowercase();
                kl.contains("file") || kl.contains("path") || kl.contains("page")
                    || kl.contains("include") || kl.contains("template") || kl.contains("load")
            }) {
                for (payload, label) in lfi_payloads {
                    let test_qs = format!("{}={}", param_key, url_encode(payload));
                    let test_url = format!("{}?{}", base_path, test_qs);
                    probes_run += 1;

                    match ureq::get(&test_url)
                        .set("User-Agent", "Qomni-Probe/1.0 (authorized security audit)")
                        .timeout(std::time::Duration::from_secs(5))
                        .call()
                    {
                        Ok(resp) => {
                            let body_text = resp.into_string().unwrap_or_default();
                            if body_text.contains("root:x:0:0") || body_text.contains("/bin/bash") {
                                all_findings.push(format!(
                                    "{{\"type\":\"LFI-CRITICAL\",\"severity\":\"CRITICAL\",\"param\":\"{}\",\"label\":\"{}\",\"evidence\":\"/etc/passwd content in response\"}}",
                                    param_key, label
                                ));
                            }
                        }
                        Err(_) => {}
                    }
                }
            }

            // ── Run header injection probes ───────────────────────────────────
            for (header, label) in header_payloads {
                let parts: Vec<&str> = header.splitn(2, ": ").collect();
                if parts.len() != 2 { continue; }
                probes_run += 1;
                match ureq::get(&target_url)
                    .set("User-Agent", "Qomni-Probe/1.0")
                    .set(parts[0], parts[1])
                    .timeout(std::time::Duration::from_secs(5))
                    .call()
                {
                    Ok(resp) => {
                        let status = resp.status();
                        // If /admin override worked (200 when base was 403/404)
                        if label.contains("url-override") && status == 200 {
                            all_findings.push(format!(
                                "{{\"type\":\"HEADER-INJECTION\",\"severity\":\"HIGH\",\"label\":\"{}\",\"evidence\":\"X-Original-URL bypass returned 200\"}}",
                                label
                            ));
                        }
                    }
                    Err(_) => {}
                }
            }

            // ── Save results to staging log ───────────────────────────────────
            let log_entry = format!(
                "{{\"timestamp\":{},\"domain\":\"{}\",\"url\":\"{}\",\"probes_run\":{},\"findings_count\":{}}}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                domain.replace('"', ""),
                target_url.chars().filter(|c| c.is_ascii_graphic()).take(100).collect::<String>().replace('"', "'"),
                probes_run,
                all_findings.len()
            );
            let _ = std::fs::write(
                format!("/opt/qomni-staging/logs/probe_{}.json", chrono_now_approx()),
                &log_entry
            );

            let findings_json = all_findings.join(",");
            let verdict = if all_findings.iter().any(|f| f.contains("CRITICAL")) {
                "CRITICAL"
            } else if !all_findings.is_empty() {
                "VULNERABLE"
            } else {
                "CLEAN"
            };

            ("200 OK", format!(
                "{{\"ok\":true,\"domain\":\"{}\",\"probes_run\":{},\"verdict\":\"{}\",\"findings\":[{}]}}",
                domain.replace('"', ""),
                probes_run,
                verdict,
                findings_json
            ))
        }


        // ──────────────────────────────────────────────────────────────────────
        // POST /web/probe/v2 — Evidence-validated probe (no false positives)
        // Body: {"url":"https://target.com/?q=test","mode":"low|medium|aggressive"}
        // ──────────────────────────────────────────────────────────────────────
        ("POST", "/web/probe/v2") => {
            use std::env;
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Requires AUDIT mode"}"#.into());
            }

            let target_url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let mode = extract_json_str(body, "mode").unwrap_or_else(|| "low".to_string());

            let domain: String = target_url
                .trim_start_matches("https://").trim_start_matches("http://")
                .split('/').next().unwrap_or("").to_string();

            let authorized_raw = std::fs::read_to_string("/opt/qomni-staging/targets/authorized.txt")
                .unwrap_or_default();
            if !authorized_raw.lines().any(|l| {
                let d = l.trim().split('|').next().unwrap_or("").trim();
                !d.starts_with('#') && (domain == d || domain.ends_with(&format!(".{}", d)))
            }) {
                return ("403 Forbidden", format!(
                    r#"{{"ok":false,"error":"Domain '{}' not authorized"}}"#, domain
                ));
            }

            // Intensity settings
            let (repeat, delay_sec, max_params) = match mode.as_str() {
                "aggressive" => (3usize, 8u64, 10usize),
                "medium"     => (2,      5,    5),
                _            => (1,      3,    3),   // low (default, safest)
            };

            // Parse URL params
            let (base_path, qs) = target_url.split_once('?')
                .map(|(a,b)| (a.to_string(), b.to_string()))
                .unwrap_or((target_url.clone(), String::new()));

            let params: Vec<(String, String)> = qs.split('&')
                .filter(|s| s.contains('='))
                .map(|kv| {
                    let mut p = kv.splitn(2, '=');
                    (p.next().unwrap_or("\\").to_string(), p.next().unwrap_or("").to_string())
                })
                .filter(|(k, _)| !k.is_empty())
                .take(max_params)
                .collect();

            let mut findings: Vec<String> = Vec::new();

            // ── Helper: fetch with param replaced ─────────────────────────────
            let make_url = |pk: &str, payload: &str| -> String {
                let qs2: String = params.iter().map(|(k, v)| {
                    if k == pk { format!("{}={}", k, url_encode(payload)) }
                    else { format!("{}={}", k, v) }
                }).collect::<Vec<_>>().join("&");
                format!("{}?{}", base_path, qs2)
            };

            // ── XSS: 3-level evidence ─────────────────────────────────────────
            // Level 1: REFLECTION   — reflected but HTML-encoded (not dangerous)
            // Level 2: POTENTIAL    — reflected unencoded in text context
            // Level 3: CONFIRMED    — reflected unencoded in executable context (script/event)
            let xss_marker = "qomni9xss9test"; // safe marker (no JS chars)
            let xss_payloads: &[(&str, &str, &str)] = &[
                ("<script>qomni9xss9test</script>", "script-tag",   "CONFIRMED"),
                ("<img src=x onerror=qomni9xss9test>", "img-event", "CONFIRMED"),
                ("\"onmouseover=\"qomni9xss9test\"",   "event-attr", "POTENTIAL"),
                (xss_marker, "plain-marker",             "REFLECTION"),
            ];

            for (param_key, _) in &params {
                for (payload, label, expected_level) in xss_payloads {
                    let test_url = make_url(param_key, payload);
                    match ureq::get(&test_url)
                        .set("User-Agent", "Qomni-Probe/2.0 (authorized)")
                        .timeout(std::time::Duration::from_secs(delay_sec))
                        .call()
                    {
                        Ok(resp) => {
                            let body_text = resp.into_string().unwrap_or_default();

                            // Determine actual XSS level
                            let level = if body_text.contains("<script>qomni9xss9test</script>")
                                || body_text.contains("onerror=qomni9xss9test")
                            {
                                "CONFIRMED"  // payload reflected unencoded in executable context
                            } else if body_text.contains("qomni9xss9test")
                                && !body_text.contains("&lt;")
                                && !body_text.contains("&#")
                            {
                                "POTENTIAL"  // marker reflected but not in exec context
                            } else if body_text.contains("qomni9xss9test") {
                                "REFLECTION" // reflected but encoded (WAF or proper escaping)
                            } else {
                                continue;    // not reflected at all — not vulnerable
                            };

                            // Only report POTENTIAL or higher
                            if level == "REFLECTION" { continue; }

                            let severity = if level == "CONFIRMED" { "HIGH" } else { "MEDIUM" };
                            findings.push(format!(
                                r#"{{"type":"XSS","level":"{}","severity":"{}","param":"{}","payload_type":"{}","evidence":"Payload reflected unencoded","fix":"sanitize output with htmlspecialchars()"}}"#,
                                level, severity,
                                clean_str(param_key), label
                            ));
                        }
                        Err(_) => {}
                    }
                }
            }

            // ── SQLi: repeat test + variance check ───────────────────────────
            let sqli_error_sigs = [
                "you have an error in your sql syntax",
                "warning: mysql", "pg_query()", "sqlite_",
                "ora-00933", "microsoft ole db", "odbc driver",
                "sqlstate", "unclosed quotation mark",
            ];

            for (param_key, _) in &params {
                // Error-based: quote injection
                let test_url = make_url(param_key, "'");
                match ureq::get(&test_url)
                    .set("User-Agent", "Qomni-Probe/2.0 (authorized)")
                    .timeout(std::time::Duration::from_secs(delay_sec))
                    .call()
                {
                    Ok(resp) => {
                        let body_low = resp.into_string().unwrap_or_default().to_lowercase();
                        if sqli_error_sigs.iter().any(|s| body_low.contains(s)) {
                            findings.push(format!(
                                r#"{{"type":"SQLI","level":"CONFIRMED","severity":"CRITICAL","param":"{}","evidence":"DB error string in response — error-based injection","fix":"Use prepared statements / parameterized queries"}}"#,
                                clean_str(param_key)
                            ));
                        }
                    }
                    Err(_) => {}
                }

                // Timing-based: ONLY if repeat >= 2 (medium/aggressive mode)
                if repeat >= 2 {
                    let baseline_url = make_url(param_key, "1");
                    let sleep_url    = make_url(param_key, "1 AND SLEEP(2)--");

                    // Measure baseline (repeat times)
                    let mut baseline_ms: Vec<u128> = Vec::new();
                    for _ in 0..repeat {
                        let t = std::time::Instant::now();
                        let _ = ureq::get(&baseline_url)
                            .set("User-Agent", "Qomni-Probe/2.0")
                            .timeout(std::time::Duration::from_secs(delay_sec))
                            .call();
                        baseline_ms.push(t.elapsed().as_millis());
                    }
                    let baseline_avg = baseline_ms.iter().sum::<u128>() / baseline_ms.len() as u128;

                    // Measure with sleep payload (repeat times)
                    let mut sleep_ms: Vec<u128> = Vec::new();
                    for _ in 0..repeat {
                        let t = std::time::Instant::now();
                        let _ = ureq::get(&sleep_url)
                            .set("User-Agent", "Qomni-Probe/2.0")
                            .timeout(std::time::Duration::from_secs(delay_sec + 3))
                            .call();
                        sleep_ms.push(t.elapsed().as_millis());
                    }
                    let sleep_avg = sleep_ms.iter().sum::<u128>() / sleep_ms.len() as u128;

                    // Variance check: sleep must be > baseline + 1500ms AND consistent
                    let min_sleep = *sleep_ms.iter().min().unwrap_or(&0);
                    let delay_confirmed = sleep_avg > baseline_avg + 1500
                        && min_sleep > baseline_avg + 800; // all runs show delay

                    if delay_confirmed {
                        findings.push(format!(
                            r#"{{"type":"SQLI-TIMING","level":"CONFIRMED","severity":"HIGH","param":"{}","evidence":"Consistent delay: baseline={}ms sleep_avg={}ms (repeated {} times)","fix":"Use prepared statements"}}"#,
                            clean_str(param_key), baseline_avg, sleep_avg, repeat
                        ));
                    }
                }
            }

            // ── LFI: real pattern validation ──────────────────────────────────
            let lfi_real_patterns = [
                "root:x:0:0",          // /etc/passwd actual format
                "daemon:x:1:1",        // second line of /etc/passwd
                "/bin/bash",           // shell entries
                "nobody:x:",           // common passwd entry
            ];

            for (param_key, _) in params.iter().filter(|(k,_)| {
                let kl = k.to_lowercase();
                ["file","path","page","include","template","load","doc"].iter()
                    .any(|p| kl.contains(p))
            }) {
                for payload in ["../../../etc/passwd", "/etc/passwd", "....//....//etc/passwd"] {
                    let test_url = make_url(param_key, payload);
                    match ureq::get(&test_url)
                        .set("User-Agent", "Qomni-Probe/2.0 (authorized)")
                        .timeout(std::time::Duration::from_secs(delay_sec))
                        .call()
                    {
                        Ok(resp) => {
                            let body_text = resp.into_string().unwrap_or_default();
                            let pattern_matches = lfi_real_patterns.iter()
                                .filter(|p| body_text.contains(*p))
                                .count();
                            // Require 2+ real patterns to avoid WAF fake responses
                            if pattern_matches >= 2 {
                                findings.push(format!(
                                    r#"{{"type":"LFI","level":"CONFIRMED","severity":"CRITICAL","param":"{}","evidence":"{} real /etc/passwd patterns confirmed","fix":"Whitelist allowed file paths, never pass user input to file() directly"}}"#,
                                    clean_str(param_key), pattern_matches
                                ));
                            }
                        }
                        Err(_) => {}
                    }
                }
            }

            // ── Live correlation: check if these params are in recent attack feed ─
            let nginx_log = std::process::Command::new("tail")
                .args(&["-n", "200", "/var/log/nginx/access.log"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();

            let param_names: Vec<&str> = params.iter().map(|(k,_)| k.as_str()).collect();
            let live_attacks: Vec<String> = nginx_log.lines().rev().take(100)
                .filter(|line| {
                    (line.contains(" 400 ") || line.contains(" 403 ") || line.contains(" 429 "))
                    && param_names.iter().any(|p| line.contains(p))
                })
                .take(5)
                .map(|l| {
                    let parts: Vec<&str> = l.splitn(10, ' ').collect();
                    let ip = parts.get(0).unwrap_or(&"?");
                    let status = parts.get(8).unwrap_or(&"?");
                    format!(r#"{{"ip":"{}","status":"{}"}}"#,
                        ip.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='.').collect::<String>(),
                        status.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>())
                })
                .collect();

            let verdict = if findings.iter().any(|f| f.contains("\"CRITICAL\"")) {
                "CRITICAL"
            } else if findings.iter().any(|f| f.contains("CONFIRMED")) {
                "VULNERABLE"
            } else if findings.iter().any(|f| f.contains("POTENTIAL")) {
                "POTENTIAL"
            } else { "CLEAN" };

            // Auto-defense recommendation
            let defense_rules: Vec<String> = findings.iter().filter_map(|f| {
                if f.contains("\"XSS\"") {
                    Some(r#"{"rule":"add_header Content-Security-Policy \"default-src 'self'; script-src 'self';\" always;","target":"nginx"}"#.to_string())
                } else if f.contains("\"SQLI\"") {
                    Some(r#"{"rule":"Implement prepared statements in your query layer — cannot be fixed at nginx level","target":"code"}"#.to_string())
                } else { None }
            }).collect::<std::collections::HashSet<_>>().into_iter().collect();

            ("200 OK", format!(
                r#"{{"ok":true,"domain":"{}","mode":"{}","params_tested":{},"verdict":"{}","findings":[{}],"live_correlation":[{}],"defense":[{}]}}"#,
                domain, mode, params.len(), verdict,
                findings.join(","),
                live_attacks.join(","),
                defense_rules.join(",")
            ))
        }

        // ──────────────────────────────────────────────────────────────────────
        // POST /web/orchestrate — Full pipeline: CRAWL → PROBE → LIVE → DEFENSE
        // Body: {"url":"https://target.com","mode":"low|medium|aggressive"}
        // ──────────────────────────────────────────────────────────────────────
        ("POST", "/web/orchestrate") => {
            use std::env;
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Requires AUDIT mode"}"#.into());
            }

            let base_url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let mode = extract_json_str(body, "mode").unwrap_or_else(|| "low".to_string());

            let domain: String = base_url
                .trim_start_matches("https://").trim_start_matches("http://")
                .split('/').next().unwrap_or("").to_string();

            let authorized_raw = std::fs::read_to_string("/opt/qomni-staging/targets/authorized.txt")
                .unwrap_or_default();
            if !authorized_raw.lines().any(|l| {
                let d = l.trim().split('|').next().unwrap_or("").trim();
                !d.starts_with('#') && (domain == d || domain.ends_with(&format!(".{}", d)))
            }) {
                return ("403 Forbidden", format!(
                    r#"{{"ok":false,"error":"Domain '{}' not authorized"}}"#, domain
                ));
            }

            let base_origin = if base_url.starts_with("https://") {
                format!("https://{}", domain)
            } else {
                format!("http://{}", domain)
            };

            // ── PHASE 1: CRAWL — discover URLs with params ────────────────────
            let crawl_result = ureq::get(&base_url)
                .set("User-Agent", "Qomni-Orchestrator/1.0 (authorized)")
                .timeout(std::time::Duration::from_secs(10))
                .call();

            let html = match crawl_result {
                Ok(r) => r.into_string().unwrap_or_default(),
                Err(e) => return ("502 Bad Gateway", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
            };

            // Extract URLs with query params
            let mut urls_with_params: Vec<String> = Vec::new();
            let mut pos = 0;
            while let Some(href_start) = html[pos..].find("href=\"") {
                let start = pos + href_start + 6;
                let rest = &html[start..];
                let end = rest.find('"').unwrap_or(rest.len());
                let href = &rest[..end];
                if href.contains('?') && href.contains('=') {
                    let full = if href.starts_with("http") {
                        href.to_string()
                    } else if href.starts_with('/') {
                        format!("{}{}", base_origin, href)
                    } else { pos = start + end; continue; };
                    if full.contains(&domain) && !urls_with_params.contains(&full) {
                        urls_with_params.push(full);
                    }
                }
                pos = start + end;
                if urls_with_params.len() >= 5 { break; }
            }
            // Always include base_url if it has params
            if base_url.contains('?') && !urls_with_params.contains(&base_url) {
                urls_with_params.insert(0, base_url.clone());
            }

            // ── PHASE 2: PROBE each URL ───────────────────────────────────────
            let max_probe_urls = match mode.as_str() {
                "aggressive" => 5usize, "medium" => 3, _ => 2
            };

            let mut all_findings: Vec<String> = Vec::new();
            let mut probed_urls: Vec<String> = Vec::new();

            for probe_url in urls_with_params.iter().take(max_probe_urls) {
                // Parse params from this URL
                let qs = probe_url.split_once('?').map(|(_,q)| q).unwrap_or("\\");
                let params: Vec<(String, String)> = qs.split('&')
                    .filter(|s| s.contains('='))
                    .map(|kv| {
                        let mut p = kv.splitn(2, '=');
                        (p.next().unwrap_or("").to_string(), p.next().unwrap_or("").to_string())
                    })
                    .filter(|(k,_)| !k.is_empty())
                    .collect();

                if params.is_empty() { continue; }

                let base_p = probe_url.split_once('?').map(|(b,_)| b).unwrap_or(probe_url);
                let make = |pk: &str, payload: &str| -> String {
                    let qs2 = params.iter().map(|(k,v)| {
                        if k==pk { format!("{}={}", k, url_encode(payload)) }
                        else { format!("{}={}", k, v) }
                    }).collect::<Vec<_>>().join("&");
                    format!("{}?{}", base_p, qs2)
                };

                let timeout_secs = match mode.as_str() { "aggressive"=>8u64, "medium"=>5, _=>3 };
                let sqli_err_sigs = ["you have an error in your sql syntax","warning: mysql",
                    "pg_query()","sqlite_","ora-00933","microsoft ole db","sqlstate","unclosed quotation"];

                for (pk, _) in &params {
                    // XSS check
                    let xss_url = make(pk, "<script>qomni9xss9test</script>");
                    if let Ok(r) = ureq::get(&xss_url).set("User-Agent","Qomni-Orchestrator/1.0")
                        .timeout(std::time::Duration::from_secs(timeout_secs)).call()
                    {
                        let body_t = r.into_string().unwrap_or_default();
                        if body_t.contains("<script>qomni9xss9test</script>") {
                            all_findings.push(format!(
                                r#"{{"url":"{}","type":"XSS","level":"CONFIRMED","severity":"HIGH","param":"{}"}}"#,
                                clean_str(base_p), clean_str(pk)
                            ));
                        }
                    }
                    // SQLi check
                    let sqli_url = make(pk, "'");
                    if let Ok(r) = ureq::get(&sqli_url).set("User-Agent","Qomni-Orchestrator/1.0")
                        .timeout(std::time::Duration::from_secs(timeout_secs)).call()
                    {
                        let body_l = r.into_string().unwrap_or_default().to_lowercase();
                        if sqli_err_sigs.iter().any(|s| body_l.contains(s)) {
                            all_findings.push(format!(
                                r#"{{"url":"{}","type":"SQLI","level":"CONFIRMED","severity":"CRITICAL","param":"{}"}}"#,
                                clean_str(base_p), clean_str(pk)
                            ));
                        }
                    }
                }
                probed_urls.push(clean_str(probe_url));
            }

            // ── PHASE 3: LIVE correlation ─────────────────────────────────────
            let nginx_log = std::process::Command::new("tail")
                .args(&["-n","500","/var/log/nginx/access.log"])
                .output().map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();

            let fb_status = std::process::Command::new("fail2ban-client")
                .arg("status").output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();
            let total_banned: u64 = fb_status.lines()
                .find(|l| l.contains("Currently banned:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|n| n.trim().parse().ok()).unwrap_or(0);

            // Attack rate in last 5 min (300s)
            let recent_attacks = nginx_log.lines().rev()
                .filter(|l| l.contains(" 400 ")||l.contains(" 403 ")||l.contains(" 429 "))
                .take(20).count();

            let entropy = (recent_attacks as f64 / 20.0_f64).min(1.0);

            // ── PHASE 4: DEFENSE recommendations ─────────────────────────────
            let mut defenses: Vec<&str> = Vec::new();
            let findings_str = all_findings.join(",");
            if findings_str.contains("\"XSS\"") {
                defenses.push(r#"{"rule":"add_header Content-Security-Policy \"default-src 'self'; script-src 'self'; object-src 'none';\" always;","id":"csp","target":"nginx","priority":"HIGH"}"#);
            }
            if findings_str.contains("\"SQLI\"") {
                defenses.push(r#"{"rule":"Use PDO/prepared statements in backend code. Rate-limit via nginx: limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;","id":"sqli-mitigation","target":"code+nginx","priority":"CRITICAL"}"#);
            }
            if total_banned > 0 || entropy > 0.3 {
                defenses.push(r#"{"rule":"add_header X-Frame-Options \"SAMEORIGIN\" always; add_header X-Content-Type-Options \"nosniff\" always;","id":"headers","target":"nginx","priority":"MEDIUM"}"#);
            }

            let verdict = if findings_str.contains("\"CRITICAL\"") { "CRITICAL" }
                else if findings_str.contains("CONFIRMED") { "VULNERABLE" }
                else { "CLEAN" };

            let urls_json = probed_urls.iter().map(|u| format!("\"{}\"", u)).collect::<Vec<_>>().join(",");

            ("200 OK", format!(
                r#"{{"ok":true,"domain":"{}","mode":"{}","phase1_urls_found":{},"phase2_urls_probed":{},"phase3_live":{{"banned":{},"recent_attacks":{},"entropy":{:.2}}},"phase4_defenses":[{}],"verdict":"{}","findings":[{}],"probed_urls":[{}]}}"#,
                domain, mode,
                urls_with_params.len(), probed_urls.len(),
                total_banned, recent_attacks, entropy,
                defenses.join(","),
                verdict,
                all_findings.join(","),
                urls_json
            ))
        }


        // ──────────────────────────────────────────────────────────────────────
        // POST /web/rollback — Revert last applied nginx rule (auto-rollback)
        // Body: {"id":"rule-id"} or {} to rollback last rule
        // ──────────────────────────────────────────────────────────────────────
        ("POST", "/web/rollback") => {
            use std::env;
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Requires AUDIT mode"}"#.into());
            }

            // Read incident state to find active rules
            let state_raw = std::fs::read_to_string("/opt/qomni-staging/incident_state.json")
                .unwrap_or_else(|_| r#"{"active_rules":[]}"#.to_string());

            // Find backup config
            let backup_path = "/opt/qomni-staging/nginx_backup.conf";
            let original_conf = "/etc/nginx/conf.d/qomni-headers.conf";

            if !std::path::Path::new(backup_path).exists() {
                return ("404 Not Found", r#"{"ok":false,"error":"No backup available to rollback"}"#.into());
            }

            // Restore backup
            match std::fs::copy(backup_path, original_conf) {
                Ok(_) => {
                    // Reload nginx
                    let reload = std::process::Command::new("nginx")
                        .arg("-s").arg("reload")
                        .output();

                    match reload {
                        Ok(out) if out.status.success() => {
                            // Clear active rules from incident state
                            let updated_state = state_raw.replace("\"active_rules\":[", "\"active_rules\":[")
                                .lines().collect::<Vec<_>>().join("\n");
                            // Write cleared state
                            let _ = std::fs::write(
                                "/opt/qomni-staging/incident_state.json",
                                updated_state.replace("\"active_rules\":[^]]*]", "\"active_rules\":[]")
                            );

                            ("200 OK", r#"{"ok":true,"message":"Nginx config rolled back and reloaded","rules_cleared":true}"#.into())
                        }
                        _ => ("500 Internal Server Error", r#"{"ok":false,"error":"Rollback copied but nginx reload failed"}"#.into())
                    }
                }
                Err(e) => ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"Failed to restore backup: {}"}}"#, e))
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        // GET /web/incident — Get current incident state
        // POST /web/incident — Update incident state
        // ──────────────────────────────────────────────────────────────────────
        ("GET", "/web/incident") => {
            let state = std::fs::read_to_string("/opt/qomni-staging/incident_state.json")
                .unwrap_or_else(|_| r#"{"target":"","threat_level":"NONE","active_rules":[],"blocked_ips":[],"last_update":0,"confidence":0.0}"#.to_string());
            ("200 OK", state)
        }

        ("POST", "/web/incident") => {
            use std::env;
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Requires AUDIT mode"}"#.into());
            }

            let now = chrono_now_approx();
            // Merge incoming body into state (simple append strategy)
            let target   = extract_json_str(body, "target").unwrap_or_default();
            let threat   = extract_json_str(body, "threat_level").unwrap_or_default();
            let rule     = extract_json_str(body, "add_rule");
            let ip       = extract_json_str(body, "add_ip");
            let conf_str = extract_json_str(body, "confidence").unwrap_or_else(|| "0.0".to_string());
            let confidence: f64 = conf_str.parse().unwrap_or(0.0);

            // Read existing
            let existing = std::fs::read_to_string("/opt/qomni-staging/incident_state.json")
                .unwrap_or_else(|_| r#"{"target":"","threat_level":"NONE","active_rules":[],"blocked_ips":[],"last_update":0,"confidence":0.0}"#.to_string());

            // Simple field extraction from existing JSON
            let ex_rules_start = existing.find("\"active_rules\":[").map(|i| i+16).unwrap_or(0);
            let ex_rules_end   = if ex_rules_start>0 { existing[ex_rules_start..].find(']').map(|i|i+ex_rules_start).unwrap_or(ex_rules_start) } else { 0 };
            let mut rules_inner = if ex_rules_start>0 && ex_rules_end>ex_rules_start {
                existing[ex_rules_start..ex_rules_end].to_string()
            } else { String::new() };
            if let Some(r) = &rule {
                if !rules_inner.is_empty() { rules_inner.push(','); }
                rules_inner.push('"');
                rules_inner.push_str(&r.chars().filter(|c| c.is_ascii_alphanumeric()||" -_".contains(*c)).take(80).collect::<String>());
                rules_inner.push('"');
            }

            let ex_ips_start = existing.find("\"blocked_ips\":[").map(|i| i+15).unwrap_or(0);
            let ex_ips_end   = if ex_ips_start>0 { existing[ex_ips_start..].find(']').map(|i|i+ex_ips_start).unwrap_or(ex_ips_start) } else { 0 };
            let mut ips_inner = if ex_ips_start>0 && ex_ips_end>ex_ips_start {
                existing[ex_ips_start..ex_ips_end].to_string()
            } else { String::new() };
            if let Some(i) = &ip {
                let safe_ip = i.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='.').take(20).collect::<String>();
                if !safe_ip.is_empty() {
                    if !ips_inner.is_empty() { ips_inner.push(','); }
                    ips_inner.push('"'); ips_inner.push_str(&safe_ip); ips_inner.push('"');
                }
            }

            let new_state = format!(
                r#"{{"target":"{}","threat_level":"{}","active_rules":[{}],"blocked_ips":[{}],"last_update":{},"confidence":{:.3}}}"#,
                if target.is_empty() {
                    existing.find("\"target\":\"").and_then(|i| {
                        let s = i+10; existing[s..].find('"').map(|e| existing[s..s+e].to_string())
                    }).unwrap_or_default()
                } else { target },
                if threat.is_empty() { "NONE".to_string() } else { threat },
                rules_inner,
                ips_inner,
                now,
                confidence
            );

            let _ = std::fs::write("/opt/qomni-staging/incident_state.json", &new_state);
            ("200 OK", format!(r#"{{"ok":true,"state":{}}}"#, new_state))
        }



        // ──────────────────────────────────────────────────────────────────────
        // POST /defense/block-ip — Block IP via nginx deny + log to incident
        // Body: {"ip":"1.2.3.4","reason":"botnet|sqli|xss|manual"}
        // ──────────────────────────────────────────────────────────────────────
        ("POST", "/defense/block-ip") => {
            use std::env;
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Requires AUDIT mode"}"#.into());
            }

            let ip = match extract_json_str(body, "ip") {
                Some(i) => i,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing ip"}"#.into()),
            };
            let reason = extract_json_str(body, "reason").unwrap_or_else(|| "manual".to_string());

            // Validate IP format (basic)
            let safe_ip: String = ip.chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .take(15).collect();
            if safe_ip.is_empty() || safe_ip.split('.').count() != 4 {
                return ("400 Bad Request", r#"{"ok":false,"error":"Invalid IP format"}"#.into());
            }

            // Append to nginx deny list
            let deny_conf = "/etc/nginx/conf.d/qomni-blocked.conf";
            let existing = std::fs::read_to_string(deny_conf).unwrap_or_default();
            let deny_line = format!("deny {};  # qomni-{} {}\n", safe_ip, reason, chrono_now_approx());

            if existing.contains(&format!("deny {};", safe_ip)) {
                return ("200 OK", format!(r#"{{"ok":true,"message":"IP {} already blocked","new":false}}"#, safe_ip));
            }

            // Backup before first write
            let backup = "/opt/qomni-staging/nginx_blocked_backup.conf";
            if !std::path::Path::new(backup).exists() {
                let _ = std::fs::write(backup, &existing);
            }

            let new_content = format!("{}{}",
                if existing.is_empty() { "# Qomni blocked IPs\n".to_string() } else { existing },
                deny_line
            );

            match std::fs::write(deny_conf, &new_content) {
                Ok(_) => {
                    // Reload nginx
                    let reload = std::process::Command::new("nginx")
                        .args(&["-s", "reload"]).output();

                    // Log to incident state
                    let state_path = "/opt/qomni-staging/incident_state.json";
                    let state = std::fs::read_to_string(state_path)
                        .unwrap_or_else(|_| r#"{"blocked_ips":[]}"#.to_string());
                    let _ = std::fs::write(state_path,
                        state.replacen(r#""blocked_ips":["#, &format!(r#""blocked_ips"["{}","#, safe_ip), 1)
                    );

                    match reload {
                        Ok(o) if o.status.success() => ("200 OK", format!(
                            r#"{{"ok":true,"ip":"{}","reason":"{}","message":"IP blocked and nginx reloaded"}}"#,
                            safe_ip, reason
                        )),
                        _ => ("200 OK", format!(
                            r#"{{"ok":true,"ip":"{}","message":"IP written but nginx reload failed — check manually"}}"#,
                            safe_ip
                        ))
                    }
                }
                Err(e) => ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"{}"}}"#, e))
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        // POST /defense/rate-limit — Apply nginx rate limiting for a zone
        // Body: {"zone":"api|login|general","rate":"30r/m","burst":10}
        // ──────────────────────────────────────────────────────────────────────
        ("POST", "/defense/rate-limit") => {
            use std::env;
            if env::var("QOMNI_PATCH_ENABLED").unwrap_or_default() != "1" {
                return ("403 Forbidden", r#"{"ok":false,"error":"Requires AUDIT mode"}"#.into());
            }

            let zone   = extract_json_str(body, "zone").unwrap_or_else(|| "general".to_string());
            let rate   = extract_json_str(body, "rate").unwrap_or_else(|| "30r/m".to_string());
            let burst_s = extract_json_str(body, "burst").unwrap_or_else(|| "10".to_string());
            let burst: u32 = burst_s.parse().unwrap_or(10);

            // Sanitize
            let safe_zone: String = zone.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='-').take(20).collect();
            let safe_rate: String = rate.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='/').take(10).collect();

            let conf_path = "/etc/nginx/conf.d/qomni-ratelimit.conf";
            let existing = std::fs::read_to_string(conf_path).unwrap_or_default();

            // Backup
            let backup = "/opt/qomni-staging/nginx_rl_backup.conf";
            if !std::path::Path::new(backup).exists() {
                let _ = std::fs::write(backup, &existing);
            }

            let zone_name = format!("qomni_{}", safe_zone);
            let new_block = format!(
                "\n# Qomni rate-limit: {} — applied {}\nlimit_req_zone $binary_remote_addr zone={}:10m rate={};\n",
                safe_zone, chrono_now_approx(), zone_name, safe_rate
            );

            if existing.contains(&zone_name) {
                return ("200 OK", format!(
                    r#"{{"ok":true,"message":"Zone {} already exists","new":false}}"#, zone_name
                ));
            }

            let new_content = format!("{}{}", existing, new_block);
            match std::fs::write(conf_path, &new_content) {
                Ok(_) => {
                    let reload = std::process::Command::new("nginx")
                        .args(&["-s", "reload"]).output();
                    match reload {
                        Ok(o) if o.status.success() => ("200 OK", format!(
                            r#"{{"ok":true,"zone":"{}","rate":"{}","burst":{},"message":"Rate-limit zone created and nginx reloaded"}}"#,
                            zone_name, safe_rate, burst
                        )),
                        _ => ("200 OK", format!(
                            r#"{{"ok":true,"zone":"{}","message":"Written but nginx reload failed"}}"#, zone_name
                        ))
                    }
                }
                Err(e) => ("500 Internal Server Error", format!(r#"{{"ok":false,"error":"{}"}}"#, e))
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        // POST /audit/verify — Re-run security check to validate a fix was applied
        // Body: {"url":"https://target.com"}
        // Returns delta vs previous finding (headers present/missing)
        // ──────────────────────────────────────────────────────────────────────
        ("POST", "/audit/verify") => {
            let target_url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };

            let domain: String = target_url
                .trim_start_matches("https://").trim_start_matches("http://")
                .split('/').next().unwrap_or("").to_string();

            // Re-check security headers
            let check_result = ureq::get(&target_url)
                .set("User-Agent", "Qomni-Verify/1.0")
                .timeout(std::time::Duration::from_secs(8))
                .call();

            match check_result {
                Ok(resp) => {
                    let headers_to_check = [
                        ("Content-Security-Policy", "CSP"),
                        ("Strict-Transport-Security", "HSTS"),
                        ("X-Frame-Options", "X-FRAME"),
                        ("X-Content-Type-Options", "X-CONTENT-TYPE"),
                        ("Permissions-Policy", "PERMISSIONS"),
                    ];

                    let mut present: Vec<String> = Vec::new();
                    let mut missing: Vec<String> = Vec::new();

                    for (header, label) in &headers_to_check {
                        if resp.header(header).is_some() {
                            present.push(format!(r#"{{"header":"{}","status":"PRESENT"}}"#, label));
                        } else {
                            missing.push(format!(r#"{{"header":"{}","status":"MISSING"}}"#, label));
                        }
                    }

                    let score = present.len() * 20; // 5 headers, 20 pts each = 100 max
                    let status = if missing.is_empty() { "PASS" } else if missing.len() <= 2 { "PARTIAL" } else { "FAIL" };

                    ("200 OK", format!(
                        r#"{{"ok":true,"domain":"{}","score":{},"status":"{}","present":[{}],"missing":[{}]}}"#,
                        domain, score, status,
                        present.join(","),
                        missing.join(",")
                    ))
                }
                Err(e) => ("502 Bad Gateway", format!(r#"{{"ok":false,"error":"{}"}}"#, e))
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        // GET /defense/status — Current defense posture summary
        // Returns: active nginx rules, blocked IPs, rate-limit zones, incident
        // ──────────────────────────────────────────────────────────────────────
        ("GET", "/defense/status") => {
            let blocked_conf = std::fs::read_to_string("/etc/nginx/conf.d/qomni-blocked.conf")
                .unwrap_or_default();
            let rl_conf = std::fs::read_to_string("/etc/nginx/conf.d/qomni-ratelimit.conf")
                .unwrap_or_default();
            let headers_conf = std::fs::read_to_string("/etc/nginx/conf.d/qomni-headers.conf")
                .unwrap_or_default();
            let incident = std::fs::read_to_string("/opt/qomni-staging/incident_state.json")
                .unwrap_or_else(|_| r#"{"threat_level":"NONE"}"#.to_string());

            let blocked_count = blocked_conf.lines().filter(|l| l.trim_start().starts_with("deny")).count();
            let rl_zones: Vec<String> = rl_conf.lines()
                .filter(|l| l.contains("limit_req_zone"))
                .map(|l| {
                    let zone = l.split("zone=").nth(1).unwrap_or("").split(':').next().unwrap_or("?");
                    format!("\"{}\"", zone.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='_').take(30).collect::<String>())
                })
                .collect();

            let has_csp   = headers_conf.contains("Content-Security-Policy");
            let has_hsts  = headers_conf.contains("Strict-Transport-Security");
            let has_frame = headers_conf.contains("X-Frame-Options");

            ("200 OK", format!(
                r#"{{"ok":true,"blocked_ips":{},"rate_limit_zones":[{}],"headers":{{"csp":{},"hsts":{},"xframe":{}}},"incident":{}}}"#,
                blocked_count,
                rl_zones.join(","),
                has_csp, has_hsts, has_frame,
                incident
            ))
        }



        // POST /web/live-threats — real-time attack feed from fail2ban + nginx logs
        ("POST", "/web/live-threats") => {
            use std::process::Command;

            // ── fail2ban jails ────────────────────────────────────────────────
            let fb_status = Command::new("fail2ban-client")
                .arg("status")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();

            // Extract jail names
            let jail_names: Vec<&str> = {
                let mut names = Vec::new();
                for line in fb_status.lines() {
                    if line.contains("Jail list:") {
                        if let Some(list) = line.split(':').nth(1) {
                            for n in list.split(',') {
                                let trimmed = n.trim();
                                if !trimmed.is_empty() { names.push(trimmed); }
                            }
                        }
                    }
                }
                names
            };

            // Query each jail for banned count + last IP
            let mut jail_json_parts: Vec<String> = Vec::new();
            let mut total_banned: u64 = 0;
            let mut last_global_ip = String::new();

            for jail in &jail_names {
                let jail_out = Command::new("fail2ban-client")
                    .args(&["status", jail])
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                    .unwrap_or_default();

                let banned: u64 = jail_out.lines()
                    .find(|l| l.contains("Currently banned:"))
                    .and_then(|l| l.split(':').nth(1))
                    .and_then(|n| n.trim().parse().ok())
                    .unwrap_or(0);

                let last_ip: &str = jail_out.lines()
                    .find(|l| l.contains("Banned IP list:"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().split_whitespace().last().unwrap_or("\\"))
                    .unwrap_or("");

                if !last_ip.is_empty() { last_global_ip = last_ip.to_string(); }
                total_banned += banned;

                let jail_clean = jail.replace('"', "");
                let ip_clean = last_ip.replace('"', "");
                jail_json_parts.push(format!(
                    "{{\"name\":\"{}\",\"banned\":{},\"last_ip\":\"{}\"}}",
                    jail_clean, banned, ip_clean
                ));
            }

            let jails_json = jail_json_parts.join(",");

            // ── nginx log tail — parse attacks ────────────────────────────────
            let log_paths = [
                "/var/log/nginx/access.log",
                "/var/log/nginx/qomni.access.log",
                "/var/log/nginx/error.log",
            ];
            let mut log_content = String::new();
            for lp in &log_paths {
                if let Ok(out) = Command::new("tail").args(&["-n", "200", lp]).output() {
                    log_content.push_str(&String::from_utf8_lossy(&out.stdout));
                }
            }

            // Parse lines — extract attacks (4xx/5xx + suspicious paths)
            let suspicious = ["/wp-admin","/xmlrpc","/phpmyadmin","/.env","/.git",
                              "/shell","/cmd","eval(","/cgi-bin","/manager/html",
                              "UNION SELECT","../","%00"];
            let mut attacks: Vec<String> = Vec::new();
            let mut attacks_count: u64 = 0;

            for line in log_content.lines().rev() {
                let is_attack = line.contains(" 400 ") || line.contains(" 401 ")
                    || line.contains(" 403 ") || line.contains(" 404 ")
                    || line.contains(" 429 ") || line.contains(" 500 ")
                    || suspicious.iter().any(|s| line.contains(s));
                if !is_attack { continue; }
                attacks_count += 1;
                if attacks.len() >= 15 { continue; }

                // Parse: IP - - [time] "METHOD PATH proto" STATUS ...
                let parts: Vec<&str> = line.splitn(10, ' ').collect();
                let ip = parts.get(0).unwrap_or(&"?").replace('"', "\\");
                let time_raw = parts.get(3).unwrap_or(&"").trim_start_matches('[');
                // time like: 15/Apr/2026:04:40:54
                let time_display = time_raw.split(':').take(2).collect::<Vec<_>>().join(":");
                let request = parts.get(6).unwrap_or(&"?").replace('"', "");
                let status: u32 = parts.get(8).unwrap_or(&"0").parse().unwrap_or(0);

                let ip_c = ip.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='.'||*c==':').collect::<String>();
                let path_c = request.chars().filter(|c| c.is_ascii_graphic()).take(60).collect::<String>();
                let time_c = time_display.chars().filter(|c| c.is_ascii_alphanumeric()||*c=='/'||*c==':'||*c==' ').take(20).collect::<String>();

                attacks.push(format!(
                    "{{\"ip\":\"{}\",\"time\":\"{}\",\"path\":\"{}\",\"status\":{}}}",
                    ip_c, time_c, path_c, status
                ));
            }

            let attacks_json = attacks.join(",");

            // ── Entropy calculation — ratio of anomalous to total lines ───────
            let total_lines = log_content.lines().count().max(1);
            let entropy_raw = (attacks_count as f64) / (total_lines as f64).min(200.0);
            let entropy = entropy_raw.min(1.0);

            // ── Build response ─────────────────────────────────────────────────
            let last_ip_clean = last_global_ip.replace('"', "");
            ("200 OK", format!(
                "{{\"ok\":true,\"total_banned\":{},\"jails\":[{}],\"attacks\":[{}],\"attacks_per_hour\":{},\"entropy\":{:.3},\"last_blocked_ip\":\"{}\"}}",
                total_banned, jails_json, attacks_json, attacks_count, entropy, last_ip_clean
            ))
        }


        // ── Execution Graph Engine v3 ─────────────────────────────────────
        // POST /graph/execute
        // Supports: {mode:"proactive"}, assert constraints, if/then/else, caching
        ("POST", "/graph/execute") => {
            let nodes = parse_graph_nodes(body);
            if nodes.is_empty() {
                return ("400 Bad Request", r#"{"ok":false,"error":"no nodes in graph"}"#.into());
            }
            let proactive_mode = body.contains("\"proactive\"");

            // Build dependency map
            let mut dep_map: std::collections::HashMap<String, Vec<String>> =
                std::collections::HashMap::new();
            for node in &nodes {
                let mut deps: Vec<String> = Vec::new();
                for (_, val) in &node.params {
                    if let GraphParamVal::Ref(r) = val {
                        if let Some(dep_id) = r.strip_prefix('$')
                            .and_then(|s| s.split('.').next())
                        {
                            let d = dep_id.to_string();
                            if !deps.contains(&d) { deps.push(d); }
                        }
                    }
                }
                for expr in node.asserts.iter().map(|a| a.expr.as_str())
                    .chain(node.if_cond.iter().map(|s| s.as_str()))
                {
                    if let Some(ref_part) = expr.strip_prefix('$') {
                        if let Some(dep_id) = ref_part.split('.').next() {
                            let d = dep_id.to_string();
                            if d != node.id && !deps.contains(&d) { deps.push(d); }
                        }
                    }
                }
                dep_map.insert(node.id.clone(), deps);
            }

            // ── Graph Optimizer ────────────────────────────────────────
            let (live_indices, opt_report) = optimize_graph(&nodes);
            let nodes: Vec<GraphNode> = live_indices.into_iter().map(|i| {
                let n = &nodes[i];
                GraphNode {
                    id: n.id.clone(), plan: n.plan.clone(),
                    params: n.params.iter().map(|(k,v)| (k.clone(), match v {
                        GraphParamVal::Float(f) => GraphParamVal::Float(*f),
                        GraphParamVal::Ref(r) => GraphParamVal::Ref(r.clone()),
                    })).collect(),
                    asserts: n.asserts.iter().map(|a| GraphAssert {
                        expr: a.expr.clone(), fail_msg: a.fail_msg.clone()
                    }).collect(),
                    if_cond: n.if_cond.clone(),
                    else_plan: n.else_plan.clone(),
                }
            }).collect();

            let exec_order = match graph_topo_sort(&nodes, &dep_map) {
                Ok(o)  => o,
                Err(e) => return ("400 Bad Request",
                    format!(r#"{{"ok":false,"error":"graph: {}"}}"#, e)),
            };

            let executor = PlanExecutor::new(plans.as_slice());
            let executor = match jit_map.as_ref() {
                Some(map) => executor.with_jit_map(map.clone()),
                None      => executor,
            };

            let mut scope: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            let mut scope_units: std::collections::HashMap<String, &'static str> =
                std::collections::HashMap::new();
            let mut exec_cache: std::collections::HashMap<u64, Vec<(String,String,f64)>> =
                std::collections::HashMap::new();
            let mut node_results: Vec<serde::NodeResultJson> = Vec::new();
            let mut global_warnings: Vec<String> = Vec::new();

            for node_id in &exec_order {
                let node = match nodes.iter().find(|n| &n.id == node_id) {
                    Some(n) => n,
                    None    => continue,
                };

                // ── Conditional plan selection ──────────────────────────
                let active_plan = if let (Some(cond), Some(else_p)) =
                    (&node.if_cond, &node.else_plan)
                {
                    if eval_graph_cond(cond, &scope) { node.plan.as_str() }
                    else { else_p.as_str() }
                } else {
                    node.plan.as_str()
                };

                // ── Resolve params ──────────────────────────────────────
                let mut resolved: std::collections::HashMap<String, f64> =
                    std::collections::HashMap::new();
                for (k, v) in &node.params {
                    match v {
                        GraphParamVal::Float(f) => { resolved.insert(k.clone(), *f); }
                        GraphParamVal::Ref(r)   => {
                            let key = r.strip_prefix('$').unwrap_or(r.as_str());
                            match scope.get(key) {
                                Some(raw_val) => {
                                    let src_unit = scope_units.get(key).copied().unwrap_or("");
                                    match wire_param_safe(*raw_val, src_unit, k.as_str()) {
                                        Ok((val, Some(note))) => {
                                            resolved.insert(k.clone(), val);
                                            global_warnings.push(format!(
                                                "[{}] auto-converted: {}", node_id, note
                                            ));
                                        }
                                        Ok((val, None)) => { resolved.insert(k.clone(), val); }
                                        Err(dim_err) => {
                                            global_warnings.push(format!(
                                                "[{}] DIMENSION ERROR: {} (param: {})", node_id, dim_err, k
                                            ));
                                            resolved.insert(k.clone(), *raw_val); // passthrough
                                        }
                                    }
                                }
                                None => return ("400 Bad Request", format!(
                                    r#"{{"ok":false,"error":"ref '{}' not found (node {})"}}"#,
                                    r, node_id
                                )),
                            }
                        }
                    }
                }

                // ── Unit mismatch detection (warn mode) ─────────────────
                let unit_mismatches = validate_wiring_units(node, &scope_units);
                for (param, src_ref, src_unit, dst_unit) in unit_mismatches {
                    global_warnings.push(format!(
                        "[{}] unit mismatch: param '{}' wired from '{}' ({}) but expected ({})",
                        node_id, param, src_ref, src_unit, dst_unit
                    ));
                }

                // ── Node cache lookup ───────────────────────────────────
                let cache_key = node_cache_key(active_plan, &resolved);
                let (step_tuples, was_cached) = if let Some(cached) = exec_cache.get(&cache_key) {
                    (cached.clone(), true)
                } else {
                    match executor.execute(active_plan, resolved.clone()) {
                        Ok(result) => {
                            let tuples: Vec<(String,String,f64)> = result.steps.iter()
                                .map(|s| (s.step.clone(), s.oracle.clone(), s.value))
                                .collect();
                            exec_cache.insert(cache_key, tuples.clone());
                            (tuples, false)
                        }
                        Err(e) => return ("500 Internal Server Error", format!(
                            r#"{{"ok":false,"node":"{}","plan":"{}","error":"{}"}}"#,
                            node_id, active_plan, e
                        )),
                    }
                };

                // ── Update scope with step results + units ──────────────
                for (step_name, oracle_name, val) in &step_tuples {
                    let scope_key = format!("{}.{}", node_id, step_name);
                    scope.insert(scope_key.clone(), *val);
                    let unit = oracle_unit(oracle_name);
                    if !unit.is_empty() {
                        scope_units.insert(scope_key, unit);
                    }
                }

                // ── Threshold alerts ────────────────────────────────────
                let warnings = check_thresholds(active_plan, &step_tuples);
                for w in &warnings {
                    global_warnings.push(format!("[{}] {}", node_id, w));
                }

                // ── Constraint Engine ───────────────────────────────────
                let mut violations: Vec<String> = Vec::new();
                for assert in &node.asserts {
                    if eval_graph_cond(&assert.expr, &scope) {
                        violations.push(assert.fail_msg.clone());
                    }
                }
                if !violations.is_empty() {
                    // -- Policy-Based Auto-Fix Engine v5.1
                    let twin_conf = avg_twin_confidence();
                    let autofix_json = if let Some((policy, fix_plan)) =
                        find_policy(active_plan, &step_tuples)
                    {
                        apply_autofix_policy(policy, fix_plan, &resolved, &executor, twin_conf)
                    } else { String::new() };

                    return ("422 Unprocessable Entity", format!(
                        r#"{{"ok":false,"node":"{}","plan":"{}","constraint_violation":"{}","violations":[{}]{}}}"#,
                        node_id, active_plan,
                        violations[0],
                        violations.iter().map(|v| format!(r#""{}""#, v)).collect::<Vec<_>>().join(","),
                        autofix_json
                    ));
                }

                node_results.push(serde::NodeResultJson {
                    id:      node_id.clone(),
                    plan:    active_plan.to_string(),
                    steps:   step_tuples,
                    cached:  was_cached,
                    ok:      true,
                });
            }

            // ── Proactive Mode v2: weighted + conditional correlations ──
            let mut proactive_alerts: Vec<String> = Vec::new();
            if proactive_mode {
                // Run v2 correlations first (weighted + conditional)
                for nr in &node_results {
                    let v2_alerts = run_proactive_v2(&nr.id, &nr.plan, &scope, &executor);
                    proactive_alerts.extend(v2_alerts);
                }
                // Also run legacy correlations
                for nr in &node_results {
                    for &(corr_plan_trigger, corr_targets) in PROACTIVE_CORRELATIONS {
                        if nr.plan.as_str() != corr_plan_trigger { continue; }
                        for &(corr_plan, param_formulas) in corr_targets {
                            let mut corr_params: std::collections::HashMap<String, f64> =
                                std::collections::HashMap::new();
                            for &(param, formula) in param_formulas {
                                let val = resolve_proactive_param(formula, &scope, &nr.id);
                                corr_params.insert(param.to_string(), val);
                            }
                            match executor.execute(corr_plan, corr_params) {
                                Ok(corr_result) => {
                                    let corr_warnings = check_thresholds(
                                        corr_plan, &corr_result.steps.iter()
                                            .map(|s| (s.step.clone(), s.oracle.clone(), s.value))
                                            .collect::<Vec<_>>()
                                    );
                                    let step_summary: Vec<String> = corr_result.steps.iter()
                                        .map(|s| format!(r#"{{"step":"{}","value":{:.4},"unit":"{}"}}"#,
                                            s.step, s.value, oracle_unit(&s.oracle)))
                                        .collect();
                                    let alert_str = format!(
                                        r#"{{"triggered_by":"{}","corr_plan":"{}","results":[{}],"warnings":[{}]}}"#,
                                        nr.id, corr_plan, step_summary.join(","),
                                        corr_warnings.iter()
                                            .map(|w| format!(r#""{}""#, escape_json(w)))
                                            .collect::<Vec<_>>().join(",")
                                    );
                                    proactive_alerts.push(alert_str);
                                }
                                Err(_) => {} // skip failed correlations silently
                            }
                        }
                    }
                }
            }

            // ── Build response ──────────────────────────────────────────
            let nodes_json: Vec<String> = node_results.iter().map(|nr| {
                let steps_json: Vec<String> = nr.steps.iter().map(|(step, oracle, val)| {
                    let unit = oracle_unit(oracle);
                    if unit.is_empty() {
                        format!(r#"{{"step":"{}","oracle":"{}","value":{:.6}}}"#, step, oracle, val)
                    } else {
                        format!(r#"{{"step":"{}","oracle":"{}","value":{:.6},"unit":"{}"}}"#,
                            step, oracle, val, unit)
                    }
                }).collect();
                format!(r#"{{"id":"{}","plan":"{}","cached":{},"steps":[{}]}}"#,
                    nr.id, nr.plan, nr.cached, steps_json.join(","))
            }).collect();

            let cache_hits = node_results.iter().filter(|nr| nr.cached).count();
            let warnings_json: Vec<String> = global_warnings.iter()
                .map(|w| format!(r#""{}""#, escape_json(w))).collect();
            let proactive_json = if proactive_mode {
                format!(",\"proactive_alerts\":[{}]", proactive_alerts.join(","))
            } else { String::new() };

            let response = format!(
                r#"{{"ok":true,"nodes":[{}],"node_count":{},"cache_hits":{},"scope_vars":{},"warnings":[{}],"optimizer":{}{}}}"#,
                nodes_json.join(","),
                node_results.len(),
                cache_hits,
                scope.len(),
                warnings_json.join(","),
                opt_report.to_json(),
                proactive_json
            );

            // ── Save to Graph Memory + Adaptive Learning ─────────────────
            let summary = format!(
                r#"{{"plans":[{}],"nodes":{},"cache_hits":{}}}"#,
                node_results.iter().map(|nr| format!(r#""{}""#, nr.plan))
                    .collect::<Vec<_>>().join(","),
                node_results.len(),
                cache_hits
            );
            save_graph_memory(&summary);
            let exec_hash = memory_hash(&summary);
            record_execution_outcome(exec_hash, "success", 0);

            ("200 OK", response)
        }

        // ── Graph Memory endpoints ────────────────────────────────────────
        ("GET", "/graph/memory") => {
            let entries = load_graph_memory(20);
            ("200 OK", format!(r#"{{"ok":true,"count":{},"entries":[{}]}}"#,
                entries.len(), entries.join(",")))
        }

        ("GET", "/graph/memory/ranked") => {
            let entries = load_graph_memory_ranked(20);
            ("200 OK", format!(r#"{{"ok":true,"count":{},"ranked":true,"entries":[{}]}}"#,
                entries.len(), entries.join(",")))
        }

        ("POST", "/graph/recall") => {
            // Return last 5 entries matching any plan mentioned in query
            let query_lower = body.to_lowercase();
            let all = load_graph_memory(50);
            let matched: Vec<&String> = all.iter()
                .filter(|e| {
                    // simple keyword match against the entry
                    query_lower.split_whitespace().any(|w| e.contains(w))
                })
                .take(5).collect();
            ("200 OK", format!(r#"{{"ok":true,"matched":{},"entries":[{}]}}"#,
                matched.len(), matched.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",")))
        }

        
        // ── Health (extended with watchdog metrics) ───────────────────────
        ("GET", "/health/detailed") => {
            let (_, json) = crate::server::watchdog_assess();
            ("200 OK", json)
        }

        ("GET", "/norms") => {
            ("200 OK", norms_list_json())
        }

        ("GET", "/norms/lookup") => {
            // ?code=NFPA20-4.26.1  (encoded in body as {"code":"..."})
            let code = extract_json_str(body, "code").unwrap_or_default();
            match norm_lookup(&code) {
                Some(e) => ("200 OK", format!(
                    r#"{{"ok":true,"code":"{}","standard":"{}","year":{},"section":"{}","title":"{}","requirement":"{}","threshold":"{}","action":"{}"}}"#,
                    e.code, e.standard, e.year, e.section, e.title, e.requirement, e.threshold, e.action
                )),
                None => ("404 Not Found", format!(r#"{{"ok":false,"error":"norm {} not found"}}"#, code)),
            }
        }

        // -- Digital Twin endpoints ------------------------------------------
        ("POST", "/twin/update") => {
            // Accept simple {"vars":{"k":N}} or full {"variables":{"k":{"value":N,"unit":"U"}}}
            if body.contains("\"vars\"") && !body.contains("\"variables\"") {
                // Direct insert into state map for simple format
                if let Some(vars) = extract_json_obj_float(body, "vars") {
                    let state = twin_state_map();
                    let conf_m = twin_confidence_map();
                    let ts_m = twin_timestamp_map();
                    let now = twin_ts();
                    for (k, v) in vars {
                        state.insert(k.clone(), (v, String::new()));
                        conf_m.insert(k.clone(), 0.9);
                        ts_m.insert(k, now);
                    }
                }
            } else {
                twin_update(body);
            }
            twin_snapshot();
            ("200 OK", twin_state_json())
        }

        ("GET", "/twin/state") => {
            ("200 OK", twin_state_json())
        }

        ("DELETE", "/twin/state") => {
            twin_state_map().clear();
            ("200 OK", r#"{"ok":true,"message":"twin state cleared"}"#.into())
        }

        ("GET", "/twin/history") => {
            let history = twin_history_lock();
            let entries: Vec<String> = history.iter().rev().take(20).map(|(_, s)| s.clone()).collect();
            ("200 OK", format!(r#"{{"ok":true,"count":{},"snapshots":[{}]}}"#,
                entries.len(), entries.join(",")))
        }

        ("POST", "/twin/execute") => {
            // Execute graph using twin state to fill unresolved params
            let twin_vars: std::collections::HashMap<String, (f64, String)> = twin_state_map().iter().map(|e| (e.key().clone(), e.value().clone())).collect();
            // Build augmented body: inject twin vars as param defaults
            // For each node param that references a twin variable $twin.VAR, resolve it
            let mut augmented = body.to_string();
            // Replace $twin.VAR_NAME refs with actual values from twin state
            for (var_name, (val, _unit)) in &twin_vars {
                let placeholder = format!("\"$twin.{}\"", var_name);
                let replacement = format!("{:.6}", val);
                augmented = augmented.replace(&placeholder, &replacement);
            }
            // Re-route to graph execute
            let nodes = parse_graph_nodes(&augmented);
            if nodes.is_empty() {
                return ("422 Unprocessable Entity",
                    r#"{"ok":false,"error":"no resolvable nodes after twin injection"}"#.into());
            }
            // Build a simple single-node response
            let executor = PlanExecutor::new(plans.as_slice());
            let executor = match jit_map.as_ref() {
                Some(map) => executor.with_jit_map(map.clone()),
                None      => executor,
            };
            let mut results: Vec<String> = Vec::new();
            for node in &nodes {
                let mut resolved: std::collections::HashMap<String, f64> =
                    std::collections::HashMap::new();
                for (k, v) in &node.params {
                    match v {
                        GraphParamVal::Float(f) => { resolved.insert(k.clone(), *f); }
                        GraphParamVal::Ref(r) => {
                            let key = r.strip_prefix('$').unwrap_or(r.as_str());
                            if let Some(&(val, _)) = twin_vars.get(key) {
                                resolved.insert(k.clone(), val);
                            }
                        }
                    }
                }
                match executor.execute(&node.plan, resolved) {
                    Ok(res) => {
                        let steps_json: Vec<String> = res.steps.iter().map(|s| {
                            format!(r#"{{"step":"{}","value":{:.6},"unit":"{}"}}"#,
                                s.step, s.value, oracle_unit(&s.oracle))
                        }).collect();
                        results.push(format!(r#"{{"id":"{}","plan":"{}","steps":[{}]}}"#,
                            node.id, node.plan, steps_json.join(",")));
                    }
                    Err(e) => {
                        results.push(format!(r#"{{"id":"{}","error":"{}"}}"#, node.id, e));
                    }
                }
            }
            twin_snapshot();
            ("200 OK", format!(r#"{{"ok":true,"mode":"twin","nodes":[{}],"twin_vars":{}}}"#,
                results.join(","),
                twin_vars.len()))
        }

        // -- Graph Simulation Engine v4 ----------------------------------
        // POST /graph/simulate — Multi-Objective v7.1
        // goals:[{"target":"A.step","weight":0.6,"direction":"minimize"},...]
        // Single goal: {"optimize":"minimize","target":"A.step"} (legacy compat)
        ("POST", "/graph/simulate") => {
            let nodes = parse_graph_nodes(body);
            if nodes.is_empty() {
                return ("400 Bad Request", r#"{"ok":false,"error":"no nodes in graph"}"#.into());
            }
            let mut variants = parse_simulate_variants(body);
            if variants.is_empty() {
                return ("400 Bad Request", r#"{"ok":false,"error":"no variants defined"}"#.into());
            }
            let max_variants = extract_json_float(body, "max_variants")
                .map(|v| v as usize).unwrap_or(16).min(64);
            let budget_exceeded = variants.len() > max_variants;
            if budget_exceeded { variants.truncate(max_variants); }

            // Parse multi-objective goals array OR legacy single goal
            // goals format: [{"target":"A.hp","weight":0.6,"direction":"minimize"},...]
            let goals: Vec<(String, f64, String)> = parse_goals_array(body).unwrap_or_else(|| {
                // Legacy fallback
                let t = extract_json_str(body, "target").unwrap_or_default();
                let d = extract_json_str(body, "optimize").unwrap_or_else(|| "minimize".into());
                if t.is_empty() { Vec::new() } else { vec![(t, 1.0, d)] }
            });

            let executor = PlanExecutor::new(plans.as_slice());
            let executor = match jit_map.as_ref() {
                Some(map) => executor.with_jit_map(map.clone()),
                None      => executor,
            };

            // ── Phase 1: Execute all variants, collect goal values ────────────
            struct VarRecord {
                ok:        bool,
                nodes_json: String,
                goal_vals: Vec<Option<f64>>,  // one per goal
            }

            let mut records: Vec<VarRecord> = Vec::new();

            for variant_overrides in variants.iter() {
                let mut var_results: Vec<String> = Vec::new();
                let mut var_scope: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
                let mut var_ok = true;
                let mut goal_vals: Vec<Option<f64>> = vec![None; goals.len()];

                for node in &nodes {
                    let mut resolved: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
                    for (k, v) in &node.params {
                        match v {
                            GraphParamVal::Float(f) => { resolved.insert(k.clone(), *f); }
                            GraphParamVal::Ref(r) => {
                                let key = r.strip_prefix('$').unwrap_or(r.as_str());
                                if let Some(&val) = var_scope.get(key) { resolved.insert(k.clone(), val); }
                            }
                        }
                    }
                    if let Some(overrides) = variant_overrides.get(node.id.as_str()) {
                        for (ok2, ov) in overrides { resolved.insert(ok2.to_string(), *ov); }
                    }
                    match executor.execute(&node.plan, resolved) {
                        Ok(res) => {
                            let steps_json: Vec<String> = res.steps.iter().map(|s| {
                                let u = oracle_unit(&s.oracle);
                                if u.is_empty() { format!(r#"{{"step":"{}","value":{:.4}}}"#, s.step, s.value) }
                                else { format!(r#"{{"step":"{}","value":{:.4},"unit":"{}"}}"#, s.step, s.value, u) }
                            }).collect();
                            for s in &res.steps {
                                let key = format!("{}.{}", node.id, s.step);
                                var_scope.insert(key.clone(), s.value);
                                for (gi, (gt, _, _)) in goals.iter().enumerate() {
                                    if *gt == key { goal_vals[gi] = Some(s.value); }
                                }
                            }
                            var_results.push(format!(r#"{{"id":"{}","plan":"{}","steps":[{}]}}"#,
                                node.id, node.plan, steps_json.join(",")));
                        }
                        Err(e) => {
                            var_results.push(format!(r#"{{"id":"{}","error":"{}"}}"#, node.id, e));
                            var_ok = false;
                        }
                    }
                }
                records.push(VarRecord { ok: var_ok, nodes_json: var_results.join(","), goal_vals });
            }

            // ── Phase 2: Normalize goal values (min-max per goal) ────────────
            let n_goals = goals.len();
            let mut goal_mins = vec![f64::MAX; n_goals];
            let mut goal_maxs = vec![f64::MIN; n_goals];
            for rec in &records {
                for (gi, gv) in rec.goal_vals.iter().enumerate() {
                    if let Some(v) = gv {
                        if *v < goal_mins[gi] { goal_mins[gi] = *v; }
                        if *v > goal_maxs[gi] { goal_maxs[gi] = *v; }
                    }
                }
            }

            // ── Phase 3: Composite score + Pareto front ───────────────────────
            let total_weight: f64 = goals.iter().map(|(_, w, _)| w).sum();
            let mut composite_scores: Vec<f64> = Vec::new();

            for rec in &records {
                let mut score = 0.0f64;
                for (gi, (_, w, dir)) in goals.iter().enumerate() {
                    if let Some(v) = rec.goal_vals[gi] {
                        let range = goal_maxs[gi] - goal_mins[gi];
                        let norm = if range < 1e-12 { 0.5 } else { (v - goal_mins[gi]) / range };
                        // normalize to [0,1] where 1.0 = best
                        let directed = if dir == "minimize" { 1.0 - norm } else { norm };
                        score += directed * w;
                    }
                }
                composite_scores.push(if total_weight > 0.0 { score / total_weight } else { 0.0 });
            }

            // Pareto dominance: A dominates B if A is >= B in all goals and > B in at least one
            let mut pareto_front: Vec<usize> = Vec::new();
            let n = records.len();
            'outer: for i in 0..n {
                if !records[i].ok { continue; }
                for j in 0..n {
                    if i == j || !records[j].ok { continue; }
                    let mut j_dominates = true;
                    let mut any_better  = false;
                    for (gi, (_, _, dir)) in goals.iter().enumerate() {
                        let vi = records[i].goal_vals[gi].unwrap_or(f64::MAX);
                        let vj = records[j].goal_vals[gi].unwrap_or(f64::MAX);
                        // "better" for j means lower if minimize, higher if maximize
                        let (j_better, j_not_worse) = if dir == "minimize" {
                            (vj < vi, vj <= vi)
                        } else {
                            (vj > vi, vj >= vi)
                        };
                        if !j_not_worse { j_dominates = false; break; }
                        if j_better { any_better = true; }
                    }
                    if j_dominates && any_better { continue 'outer; } // i dominated by j
                }
                pareto_front.push(i);
            }

            // Best variant by composite score
            let best_idx = composite_scores.iter().enumerate()
                .filter(|(i, _)| records[*i].ok)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);

            // ── Phase 4: Build response ───────────────────────────────────────
            let mut variant_results: Vec<String> = Vec::new();
            for (vi, rec) in records.iter().enumerate() {
                let goal_vals_json: Vec<String> = goals.iter().enumerate().map(|(gi, (gt, w, d))| {
                    let v_str = rec.goal_vals[gi].map(|v| format!("{:.4}", v))
                        .unwrap_or_else(|| "null".into());
                    format!(r#"{{"target":"{}","value":{},"weight":{:.2},"direction":"{}","score":{:.4}}}"#,
                        gt, v_str, w, d,
                        rec.goal_vals[gi].map(|_| composite_scores[vi]).unwrap_or(0.0))
                }).collect();
                let on_pareto = pareto_front.contains(&vi);
                variant_results.push(format!(
                    r#"{{"variant":{},"ok":{},"composite_score":{:.4},"on_pareto_front":{},"goals":[{}],"nodes":[{}]}}"#,
                    vi, rec.ok, composite_scores[vi], on_pareto,
                    goal_vals_json.join(","), rec.nodes_json
                ));
            }

            let goals_desc: Vec<String> = goals.iter().map(|(t, w, d)|
                format!(r#"{{"target":"{}","weight":{:.2},"direction":"{}"}}"#, t, w, d)
            ).collect();
            let pareto_json: Vec<String> = pareto_front.iter().map(|i| i.to_string()).collect();

            ("200 OK", format!(
                r#"{{"ok":true,"variants":{},"budget_exceeded":{},"goals":[{}],"best_variant":{},"best_composite_score":{:.4},"pareto_front":[{}],"results":[{}]}}"#,
                variant_results.len(), budget_exceeded,
                goals_desc.join(","),
                best_idx, composite_scores.get(best_idx).copied().unwrap_or(0.0),
                pareto_json.join(","),
                variant_results.join(",")
            ))
        }

        ("GET", "/twin/analyze") => {
            ("200 OK", twin_analyze_json())
        }

        ("POST", "/twin/confidence") => {
            // Batch confidence update: {"variables":{"var_name":{"confidence":0.85},...}}
            let conf_map_json = body;
            let var_start = match conf_map_json.find("\"variables\"") {
                Some(i) => i, None => return ("400 Bad Request", r#"{"ok":false,"error":"no variables"}"#.into()),
            };
            let obj_start = match conf_map_json[var_start..].find('{') {
                Some(i) => var_start + i + 1, None => return ("400 Bad Request", r#"{"ok":false,"error":"parse error"}"#.into()),
            };
            let mut depth = 1usize;
            let mut obj_end = obj_start;
            for (i, b) in conf_map_json[obj_start..].bytes().enumerate() {
                match b { b'{' => depth += 1, b'}' => { depth -= 1; if depth == 0 { obj_end = obj_start+i; break; } } _ => {} }
            }
            let vars_str = &conf_map_json[obj_start..obj_end];
            let conf_lock = twin_confidence_map();
            let mut ts_lock   = twin_timestamp_map();
            let now = twin_ts();
            let mut updated = 0usize;
            let mut s = vars_str;
            loop {
                s = s.trim_start_matches([',', ' ', '\n', '\r', '\t']);
                if s.is_empty() || !s.starts_with('"') { break; }
                let k_end = match s[1..].find('"') { Some(e) => e+1, None => break };
                let var_name = s[1..k_end].to_string();
                let rest = s[k_end+1..].trim_start_matches([' ', ':']);
                let inner_start = match rest.find('{') { Some(i) => i+1, None => break };
                let mut d2 = 1usize;
                let mut inner_end = inner_start;
                for (i, b) in rest[inner_start..].bytes().enumerate() {
                    match b { b'{' => d2 += 1, b'}' => { d2 -= 1; if d2 == 0 { inner_end = inner_start+i; break; } } _ => {} }
                }
                let inner = &rest[inner_start..inner_end];
                if let Some(c) = inner.find("\"confidence\"")
                    .and_then(|p| inner[p+12..].find(':').map(|q| p+12+q+1))
                    .and_then(|p| inner[p..].trim_start_matches(' ').split([',','}']).next())
                    .and_then(|v| v.trim().parse::<f32>().ok())
                {
                    conf_lock.insert(var_name.clone(), c.clamp(0.0, 1.0));
                    ts_lock.insert(var_name, now);
                    updated += 1;
                }
                s = &rest[inner_end+1..];
            }
            ("200 OK", format!(r#"{{"ok":true,"updated":{}}}"#, updated))
        }

        ("POST", "/graph/feedback") => {
            ("200 OK", apply_external_feedback(body))
        }

        ("GET", "/graph/learning") => {
            // Return top 10 graphs by composite score
            let store = learning_store_map();
            let mut ranked: Vec<(u64, f32, u32)> = store.iter()
                .map(|entry| { let (&h, &(sr, cr, ec, _)) = (entry.key(), entry.value()); (h, composite_score(sr, cr, ec), ec) })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let entries: Vec<String> = ranked.iter().take(10).map(|(h, cs, ec)| {
                format!(r#"{{"hash":"{:x}","composite_score":{:.3},"executions":{}}}"#, h, cs, ec)
            }).collect();
            drop(store);
            ("200 OK", format!(r#"{{"ok":true,"count":{},"learned":[{}]}}"#,
                entries.len(), entries.join(",")))
        }

        ("POST", "/graph/consensus") => {
            let executor = crate::plan::PlanExecutor::new(plans.as_slice());
            let executor = match jit_map.as_ref() {
                Some(map) => executor.with_jit_map(map.clone()),
                None      => executor,
            };
            ("200 OK", run_consensus_inner(body, &executor))
        }

        ("POST", "/dkp/ingest") => {
            ("200 OK", dkp_ingest(body))
        }

        ("POST", "/dkp/query") => {
            ("200 OK", dkp_query(body))
        }

        ("GET", "/dkp/stats") => {
            ("200 OK", dkp_stats())
        }

        ("DELETE", "/dkp/purge") => {
            ("200 OK", dkp_purge())
        }

        // POST /dkp/publish — push high-confidence local facts to a mesh peer
        // Body: {"min_confidence":0.7,"target":"http://10.99.0.3:8090","domains":["hydraulic"]}
        ("POST", "/dkp/publish") => {
            ("200 OK", dkp_publish(body))
        }


        ("GET", "/ot/analyze") | ("POST", "/ot/analyze") => {
            ("200 OK", ot_analyze_twin())
        }

        ("POST", "/ot/threat_ingest") => {
            ("200 OK", ot_threat_ingest(body))
        }

        ("GET", "/ot/status") => {
            ("200 OK", ot_status())
        }

        ("POST", "/domain/bootstrap") => {
            ("200 OK", domain_bootstrap(body))
        }

        ("GET", "/domain/catalog") => {
            ("200 OK", domain_catalog())
        }

        ("GET", "/twin/events") => {
            let since = extract_json_float(body, "since").map(|v| v as u64).unwrap_or(0);
            let limit = extract_json_float(body, "limit").map(|v| v as usize).unwrap_or(100);
            ("200 OK", iot_events_json(since, limit))
        }

        ("GET", "/twin/stream") => {
            // Long-poll streaming: return events since ?since=TS
            let since_ts = {
                let q = path.find('?').map(|i| &path[i+1..]).unwrap_or("\\");
                q.split('&').find_map(|kv| {
                    let mut parts = kv.splitn(2, '=');
                    let k = parts.next().unwrap_or(""); let v = parts.next().unwrap_or("");
                    if k == "since" { v.parse::<u64>().ok() } else { None }
                }).unwrap_or(0)
            };
            ("200 OK", iot_events_json(since_ts, 200))
        }

        ("POST", "/nvd/poll") => {
            ("200 OK", nvd_poll_manual())
        }

        ("GET", "/nvd/status") => {
            ("200 OK", nvd_status())
        }

        ("GET", "/dkp/export") | ("POST", "/dkp/export") => {
            ("200 OK", dkp_export(body))
        }

        ("POST", "/dkp/import") => {
            ("200 OK", dkp_import(body))
        }

        ("GET", "/twin/sse") => {
            let since_ts = path.find('?')
                .map(|i| &path[i+1..]).unwrap_or("\\")
                .split('&').find_map(|kv| {
                    let mut p = kv.splitn(2, '=');
                    let k = p.next().unwrap_or(""); let v = p.next().unwrap_or("");
                    if k == "since" { v.parse::<u64>().ok() } else { None }
                }).unwrap_or(twin_ts().saturating_sub(60));
            let (sse_bytes, _) = twin_sse_response(since_ts);
            ("200 SSE", String::from_utf8_lossy(&sse_bytes).into_owned())
        }

        ("GET", "/registry/sync/status") => {
            ("200 OK", registry_sync_status())
        }

        ("POST", "/dkp/crawl") => {
            ("200 OK", dkp_crawl(body))
        }

        ("POST", "/registry/sync/now") => {
            let n = registry_sync_once();
            let now = twin_ts();
            REGISTRY_LAST_SYNC.store(now, Ordering::Relaxed);
            if n > 0 { REGISTRY_FACTS_SYNCED.fetch_add(n as u64, Ordering::Relaxed); }
            ("200 OK", format!(r#"{{"ok":true,"synced":{}}}"#, n))
        }

        ("GET", "/debug/jit") => {
            // Dump JIT fn table info for debugging
            let info = if let Some(ref tbl) = **jit_map {
                let entries: Vec<String> = tbl.iter()
                    .filter(|(name, _)| name.starts_with("vib") || name.starts_with("const") || name.starts_with("sum") || name.starts_with("bearing") || name.starts_with("maint") || name.starts_with("et_crop") || name.starts_with("rect_channel"))
                    .map(|(name, (addr, n))| format!(r#"{{"name":"{}","addr":{},"n_params":{}}}"#, name, addr, n))
                    .collect();
                format!(r#"{{"ok":true,"entries":[{}]}}"#, entries.join(","))
            } else {
                r#"{"ok":false,"error":"no jit map"}"#.into()
            };
            ("200 OK", info)
        }


        // ══════════════════════════════════════════════════════════════
        // CRYS-L Command Center — Recon / Fuzzer / Auth endpoints
        // ══════════════════════════════════════════════════════════════

        // ── RECON: WAF Detection ──────────────────────────────────────
        ("POST", "/recon/waf-detect") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            if !url.starts_with("http") {
                return ("400 Bad Request", r#"{"ok":false,"error":"Only http/https"}"#.into());
            }
            let mut waf_name = "None detected".to_string();
            let mut waf_vendor = "unknown".to_string();
            let mut confidence = 0u32;
            let mut headers_raw = String::new();
            let mut evidence: Vec<String> = Vec::new();

            let test_urls = vec![
                url.clone(),
                format!("{}/?id=1%27+OR+%271%27%3D%271", url.trim_end_matches('/')),
                format!("{}/%3Cscript%3Ealert(1)%3C/script%3E", url.trim_end_matches('/')),
            ];

            for test_url in &test_urls {
                match ureq::get(test_url)
                    .set("User-Agent", "Qomni-WAFDetect/1.0")
                    .timeout(std::time::Duration::from_secs(10))
                    .call()
                {
                    Ok(resp) => {
                        let mut hdrs = String::new();
                        for name in resp.headers_names() {
                            if let Some(val) = resp.header(&name) {
                                hdrs.push_str(&format!("{}: {}\n", name, val));
                                let lname = name.to_lowercase();
                                let lval = val.to_lowercase();
                                if lname == "server" && lval.contains("cloudflare") { waf_name = "Cloudflare".into(); waf_vendor = "cloudflare".into(); confidence = 95; evidence.push(format!("Server: {}", val)); }
                                if lname == "cf-ray" { waf_name = "Cloudflare".into(); waf_vendor = "cloudflare".into(); confidence = 99; evidence.push(format!("CF-Ray: {}", val)); }
                                if lname == "x-sucuri-id" || lval.contains("sucuri") { waf_name = "Sucuri WAF".into(); waf_vendor = "sucuri".into(); confidence = 95; evidence.push(format!("{}: {}", name, val)); }
                                if lname == "server" && lval.contains("awselb") { waf_name = "AWS ELB/WAF".into(); waf_vendor = "aws".into(); confidence = 80; evidence.push(format!("Server: {}", val)); }
                                if lname == "x-amz-cf-id" || lname == "x-amz-cf-pop" { waf_name = "AWS CloudFront".into(); waf_vendor = "aws".into(); confidence = 90; evidence.push(format!("{}: {}", name, val)); }
                                if lname == "server" && lval.contains("akamai") { waf_name = "Akamai".into(); waf_vendor = "akamai".into(); confidence = 90; evidence.push(format!("Server: {}", val)); }
                                if lval.contains("incapsula") { waf_name = "Imperva Incapsula".into(); waf_vendor = "imperva".into(); confidence = 95; evidence.push(format!("{}: {}", name, val)); }
                                if lname == "server" && lval.contains("bigip") { waf_name = "F5 BIG-IP".into(); waf_vendor = "f5".into(); confidence = 90; evidence.push(format!("Server: {}", val)); }
                                if lname == "server" && lval.contains("ddos-guard") { waf_name = "DDoS-Guard".into(); waf_vendor = "ddos-guard".into(); confidence = 95; evidence.push(format!("Server: {}", val)); }
                                if lname == "server" && lval.contains("barracuda") { waf_name = "Barracuda WAF".into(); waf_vendor = "barracuda".into(); confidence = 85; evidence.push(format!("{}: {}", name, val)); }
                                if lname == "server" && lval.contains("stackpath") { waf_name = "StackPath".into(); waf_vendor = "stackpath".into(); confidence = 90; evidence.push(format!("{}: {}", name, val)); }
                                if lname == "server" && lval.contains("fortiweb") { waf_name = "FortiWeb".into(); waf_vendor = "fortinet".into(); confidence = 90; evidence.push(format!("{}: {}", name, val)); }
                            }
                        }
                        if headers_raw.is_empty() { headers_raw = hdrs; }
                    }
                    Err(ureq::Error::Status(_, resp)) => {
                        for name in resp.headers_names() {
                            if let Some(val) = resp.header(&name) {
                                let lname = name.to_lowercase();
                                let lval = val.to_lowercase();
                                if lname == "cf-ray" || (lname == "server" && lval.contains("cloudflare")) { waf_name = "Cloudflare".into(); waf_vendor = "cloudflare".into(); confidence = 99; evidence.push(format!("{}: {}", name, val)); }
                            }
                        }
                    }
                    _ => {}
                }
            }

            let ev_json: Vec<String> = evidence.iter().map(|e| format!("\"{}\"", e.replace('"', "'"))).collect();
            ("200 OK", format!(
                r#"{{"ok":true,"waf":{{"detected":{},"name":"{}","vendor":"{}","confidence":{},"evidence":[{}]}},"headers":"{}"}}"#,
                confidence > 0, waf_name, waf_vendor, confidence,
                ev_json.join(","),
                headers_raw.replace('"', "'").replace('\n', "\\n")
            ))
        }

        // ── RECON: Subdomain Enumeration ──────────────────────────────
        ("POST", "/recon/subdomains") => {
            let domain = match extract_json_str(body, "domain") {
                Some(d) => d,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing domain"}"#.into()),
            };
            let prefixes = ["www","mail","ftp","admin","webmail","smtp","pop","imap","blog","shop",
                           "dev","staging","test","api","app","m","portal","vpn","remote",
                           "ns1","ns2","mx","cdn","assets","static","media",
                           "docs","wiki","help","support","status","monitor",
                           "git","ci","sso","auth","login","panel","cpanel","cloud","backup","db"];
            let mut found = Vec::new();
            for prefix in &prefixes {
                let sub = format!("{}.{}", prefix, domain);
                if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&format!("{}:80", sub)) {
                    let ips: Vec<String> = addrs.map(|a| a.ip().to_string()).collect();
                    if !ips.is_empty() {
                        found.push(format!(r#"{{"subdomain":"{}","ips":[{}]}}"#,
                            sub, ips.iter().map(|ip| format!("\"{}\"", ip)).collect::<Vec<_>>().join(",")));
                    }
                }
            }
            ("200 OK", format!(r#"{{"ok":true,"domain":"{}","found":[{}],"total":{}}}"#,
                domain, found.join(","), found.len()))
        }

        // ── RECON: Port Scan ──────────────────────────────────────────
        ("POST", "/recon/portscan") => {
            let target = match extract_json_str(body, "target") {
                Some(t) => t,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing target"}"#.into()),
            };
            let profile = extract_json_str(body, "profile").unwrap_or("quick".into());
            let ports: Vec<u16> = match profile.as_str() {
                "full" => (1..=1024).collect(),
                "common" => vec![21,22,23,25,53,80,110,135,139,143,443,445,993,995,
                                1433,1521,3306,3389,5432,5900,6379,8080,8443,9090,9200,27017],
                _ => vec![21,22,25,53,80,110,143,443,445,3306,3389,5432,8080,8443],
            };
            let mut open_ports = Vec::new();
            let timeout = std::time::Duration::from_millis(800);
            for port in &ports {
                let addr_str = format!("{}:{}", target, port);
                let connected = if let Ok(addr) = addr_str.parse::<std::net::SocketAddr>() {
                    std::net::TcpStream::connect_timeout(&addr, timeout).is_ok()
                } else if let Ok(mut addrs) = std::net::ToSocketAddrs::to_socket_addrs(&addr_str) {
                    addrs.any(|a| std::net::TcpStream::connect_timeout(&a, timeout).is_ok())
                } else { false };
                if connected {
                    let svc = match port {
                        21=>"ftp",22=>"ssh",23=>"telnet",25=>"smtp",53=>"dns",80=>"http",
                        110=>"pop3",135=>"msrpc",139=>"netbios",143=>"imap",443=>"https",
                        445=>"smb",993=>"imaps",995=>"pop3s",1433=>"mssql",1521=>"oracle",
                        3306=>"mysql",3389=>"rdp",5432=>"postgresql",5900=>"vnc",6379=>"redis",
                        8080=>"http-proxy",8443=>"https-alt",9090=>"prometheus",9200=>"elasticsearch",
                        27017=>"mongodb",_=>"unknown"
                    };
                    open_ports.push(format!(r#"{{"port":{},"service":"{}","state":"open"}}"#, port, svc));
                }
            }
            ("200 OK", format!(r#"{{"ok":true,"target":"{}","profile":"{}","ports":[{}],"total_scanned":{},"open_count":{}}}"#,
                target, profile, open_ports.join(","), ports.len(), open_ports.len()))
        }

        // ── RECON: Technology Detection ───────────────────────────────
        ("POST", "/recon/techdetect") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let mut techs = Vec::new();
            match ureq::get(&url).set("User-Agent","Qomni-TechDetect/1.0").timeout(std::time::Duration::from_secs(15)).call() {
                Ok(resp) => {
                    for name in resp.headers_names() {
                        if let Some(val) = resp.header(&name) {
                            let ln = name.to_lowercase();
                            let lv = val.to_lowercase();
                            if ln == "server" { techs.push(format!(r#"{{"name":"{}","category":"server","confidence":95}}"#, val.replace('"',"'"))); }
                            if ln == "x-powered-by" { techs.push(format!(r#"{{"name":"{}","category":"framework","confidence":90}}"#, val.replace('"',"'"))); }
                            if ln == "x-generator" { techs.push(format!(r#"{{"name":"{}","category":"cms","confidence":85}}"#, val.replace('"',"'"))); }
                            if lv.contains("php") { techs.push(format!(r#"{{"name":"PHP {}","category":"language","confidence":85}}"#, val.replace('"',"'"))); }
                        }
                    }
                    if let Ok(bt) = resp.into_string() {
                        let lb = bt.to_lowercase();
                        if lb.contains("wp-content") { techs.push(r#"{"name":"WordPress","category":"cms","confidence":95}"#.into()); }
                        if lb.contains("drupal") { techs.push(r#"{"name":"Drupal","category":"cms","confidence":85}"#.into()); }
                        if lb.contains("joomla") { techs.push(r#"{"name":"Joomla","category":"cms","confidence":80}"#.into()); }
                        if lb.contains("react") || lb.contains("__next") { techs.push(r#"{"name":"React/Next.js","category":"js-framework","confidence":75}"#.into()); }
                        if lb.contains("vue") { techs.push(r#"{"name":"Vue.js","category":"js-framework","confidence":75}"#.into()); }
                        if lb.contains("angular") { techs.push(r#"{"name":"Angular","category":"js-framework","confidence":75}"#.into()); }
                        if lb.contains("jquery") { techs.push(r#"{"name":"jQuery","category":"js-library","confidence":80}"#.into()); }
                        if lb.contains("bootstrap") { techs.push(r#"{"name":"Bootstrap","category":"css-framework","confidence":80}"#.into()); }
                        if lb.contains("tailwind") { techs.push(r#"{"name":"Tailwind CSS","category":"css-framework","confidence":80}"#.into()); }
                        if lb.contains("gtag") || lb.contains("google-analytics") { techs.push(r#"{"name":"Google Analytics","category":"analytics","confidence":90}"#.into()); }
                        if lb.contains("shopify") { techs.push(r#"{"name":"Shopify","category":"ecommerce","confidence":90}"#.into()); }
                        if lb.contains("woocommerce") { techs.push(r#"{"name":"WooCommerce","category":"ecommerce","confidence":90}"#.into()); }
                    }
                }
                Err(ureq::Error::Status(_, resp)) => {
                    for name in resp.headers_names() {
                        if let Some(val) = resp.header(&name) {
                            if name.to_lowercase() == "server" { techs.push(format!(r#"{{"name":"{}","category":"server","confidence":90}}"#, val.replace('"',"'"))); }
                        }
                    }
                }
                Err(_) => {}
            }
            techs.sort(); techs.dedup();
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","technologies":[{}],"total":{}}}"#, url, techs.join(","), techs.len()))
        }

        // ── RECON: SSL/TLS Analysis ───────────────────────────────────
        ("POST", "/recon/ssl") => {
            let domain = match extract_json_str(body, "domain") {
                Some(d) => d,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing domain"}"#.into()),
            };
            let output = std::process::Command::new("sh")
                .args(&["-c", &format!(
                    "echo | openssl s_client -servername {} -connect {}:443 2>/dev/null | openssl x509 -noout -dates -subject -issuer -serial -ext subjectAltName 2>/dev/null", domain, domain
                )]).output();
            let mut ci = std::collections::HashMap::new();
            if let Ok(out) = output {
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    if line.starts_with("notBefore=") { ci.insert("not_before", line[10..].to_string()); }
                    if line.starts_with("notAfter=") { ci.insert("not_after", line[9..].to_string()); }
                    if line.starts_with("subject=") { ci.insert("subject", line[8..].to_string()); }
                    if line.starts_with("issuer=") { ci.insert("issuer", line[7..].to_string()); }
                    if line.starts_with("serial=") { ci.insert("serial", line[7..].to_string()); }
                    if line.contains("DNS:") { ci.insert("san", line.trim().to_string()); }
                }
            }
            let proto_out = std::process::Command::new("sh")
                .args(&["-c", &format!(
                    "for p in tls1 tls1_1 tls1_2 tls1_3; do r=$(echo | openssl s_client -$p -connect {}:443 2>&1); if echo \"$r\" | grep -q 'Protocol.*TLSv'; then echo \"$p:1\"; else echo \"$p:0\"; fi; done", domain
                )]).output();
            let mut protos = Vec::new();
            if let Ok(out) = proto_out {
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    let p: Vec<&str> = line.splitn(2,':').collect();
                    if p.len()==2 { protos.push(format!(r#"{{"protocol":"{}","enabled":{}}}"#, p[0], p[1]=="1")); }
                }
            }
            let grade = if ci.contains_key("not_after") {
                if protos.iter().any(|p| p.contains("tls1_3") && p.contains("true")) { "A" }
                else if protos.iter().any(|p| p.contains("\"tls1\"") && p.contains("true")) { "C" }
                else { "B" }
            } else { "F" };
            ("200 OK", format!(
                r#"{{"ok":true,"domain":"{}","grade":"{}","subject":"{}","issuer":"{}","not_before":"{}","not_after":"{}","serial":"{}","san":"{}","protocols":[{}]}}"#,
                domain, grade,
                ci.get("subject").cloned().unwrap_or_default().replace('"',"'"),
                ci.get("issuer").cloned().unwrap_or_default().replace('"',"'"),
                ci.get("not_before").cloned().unwrap_or_default(),
                ci.get("not_after").cloned().unwrap_or_default(),
                ci.get("serial").cloned().unwrap_or_default(),
                ci.get("san").cloned().unwrap_or_default().replace('"',"'"),
                protos.join(",")
            ))
        }

        // ── RECON: DNS Analysis ───────────────────────────────────────
        ("POST", "/recon/dns") => {
            let domain = match extract_json_str(body, "domain") {
                Some(d) => d,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing domain"}"#.into()),
            };
            let mut records = Vec::new();
            for rtype in &["A","AAAA","MX","NS","TXT","CNAME","SOA","CAA"] {
                if let Ok(out) = std::process::Command::new("dig").args(&["+short",&domain,rtype]).output() {
                    for line in String::from_utf8_lossy(&out.stdout).lines() {
                        let l = line.trim();
                        if !l.is_empty() { records.push(format!(r#"{{"type":"{}","value":"{}"}}"#, rtype, l.replace('"',"'"))); }
                    }
                }
            }
            let txt_raw = std::process::Command::new("dig").args(&["+short",&domain,"TXT"]).output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
            let has_spf = txt_raw.contains("v=spf1");
            let has_dmarc = std::process::Command::new("dig").args(&["+short",&format!("_dmarc.{}",domain),"TXT"]).output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains("v=DMARC1")).unwrap_or(false);
            let has_dnssec = std::process::Command::new("dig").args(&["+dnssec","+short",&domain]).output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains("RRSIG")).unwrap_or(false);
            let has_caa = records.iter().any(|r| r.contains("\"CAA\""));
            let security = format!("[{},{},{},{}]",
                format!(r#"{{"check":"SPF","present":{},"status":"{}"}}"#, has_spf, if has_spf {"pass"} else {"fail"}),
                format!(r#"{{"check":"DMARC","present":{},"status":"{}"}}"#, has_dmarc, if has_dmarc {"pass"} else {"fail"}),
                format!(r#"{{"check":"DNSSEC","present":{},"status":"{}"}}"#, has_dnssec, if has_dnssec {"pass"} else {"warn"}),
                format!(r#"{{"check":"CAA","present":{},"status":"{}"}}"#, has_caa, if has_caa {"pass"} else {"warn"})
            );
            ("200 OK", format!(r#"{{"ok":true,"domain":"{}","records":[{}],"security":{}}}"#, domain, records.join(","), security))
        }

        // ── RECON: WHOIS ──────────────────────────────────────────────
        ("POST", "/recon/whois") => {
            let domain = match extract_json_str(body, "domain") {
                Some(d) => d,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing domain"}"#.into()),
            };
            let raw = std::process::Command::new("whois").arg(&domain).output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
            let mut info = std::collections::HashMap::new();
            for line in raw.lines() {
                let l = line.trim();
                if l.starts_with("Registrar:") { info.insert("registrar", l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()); }
                if l.starts_with("Creation Date:") { info.insert("created", l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()); }
                if l.starts_with("Registry Expiry Date:") { info.insert("expires", l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()); }
                if l.starts_with("Updated Date:") { info.insert("updated", l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()); }
                if l.starts_with("Registrant Organization:") { info.insert("org", l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()); }
                if l.starts_with("Name Server:") { info.entry("ns").or_insert_with(String::new).push_str(&format!("{},", l.splitn(2,':').nth(1).unwrap_or("").trim())); }
                if l.starts_with("DNSSEC:") { info.insert("dnssec", l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()); }
            }
            ("200 OK", format!(
                r#"{{"ok":true,"domain":"{}","registrar":"{}","created":"{}","expires":"{}","updated":"{}","organization":"{}","nameservers":"{}","dnssec":"{}","raw":"{}"}}"#,
                domain,
                info.get("registrar").cloned().unwrap_or_default().replace('"',"'"),
                info.get("created").cloned().unwrap_or_default(),
                info.get("expires").cloned().unwrap_or_default(),
                info.get("updated").cloned().unwrap_or_default(),
                info.get("org").cloned().unwrap_or_default().replace('"',"'"),
                info.get("ns").cloned().unwrap_or_default().replace('"',"'"),
                info.get("dnssec").cloned().unwrap_or_default(),
                raw.chars().take(2000).collect::<String>().replace('"',"'").replace('\n',"\\n").replace('\r',"")
            ))
        }

        // ── RECON: HTTP Headers ───────────────────────────────────────
        ("POST", "/recon/headers") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let mut headers_list = Vec::new();
            let mut cookies = Vec::new();
            let mut hdr_map = std::collections::HashMap::new();
            match ureq::get(&url).set("User-Agent","Qomni-HeaderScan/1.0").timeout(std::time::Duration::from_secs(15)).call() {
                Ok(resp) => {
                    for name in resp.headers_names() {
                        if let Some(val) = resp.header(&name) {
                            headers_list.push(format!(r#"{{"name":"{}","value":"{}"}}"#, name, val.replace('"',"'").replace('\\',"/")));
                            hdr_map.insert(name.to_lowercase(), val.to_string());
                            if name.to_lowercase() == "set-cookie" {
                                let lv = val.to_lowercase();
                                let cn = val.split('=').next().unwrap_or("?");
                                cookies.push(format!(r#"{{"name":"{}","secure":{},"httponly":{},"samesite":{}}}"#,
                                    cn, lv.contains("secure"), lv.contains("httponly"), lv.contains("samesite")));
                            }
                        }
                    }
                }
                Err(ureq::Error::Status(_, resp)) => {
                    for name in resp.headers_names() {
                        if let Some(val) = resp.header(&name) {
                            headers_list.push(format!(r#"{{"name":"{}","value":"{}"}}"#, name, val.replace('"',"'")));
                            hdr_map.insert(name.to_lowercase(), val.to_string());
                        }
                    }
                }
                Err(_) => {}
            }
            let pos = ["strict-transport-security","content-security-policy","x-frame-options",
                       "x-content-type-options","referrer-policy","permissions-policy",
                       "x-xss-protection","cross-origin-embedder-policy","cross-origin-opener-policy","cross-origin-resource-policy"];
            let neg = ["server","x-powered-by","x-aspnet-version","x-generator"];
            let mut checks = Vec::new();
            let mut pass = 0u32;
            for h in &pos {
                let found = hdr_map.contains_key(*h);
                if found { pass += 1; }
                checks.push(format!(r#"{{"header":"{}","present":{},"type":"positive"}}"#, h, found));
            }
            for h in &neg {
                let found = hdr_map.contains_key(*h);
                if !found { pass += 1; }
                checks.push(format!(r#"{{"header":"{}","present":{},"type":"negative"}}"#, h, found));
            }
            let total = (pos.len() + neg.len()) as u32;
            let score = pass * 100 / total;
            let grade = if score >= 90 {"A"} else if score >= 80 {"B"} else if score >= 60 {"C"} else if score >= 40 {"D"} else {"F"};
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","headers":[{}],"checks":[{}],"cookies":[{}],"score":{},"grade":"{}"}}"#,
                url, headers_list.join(","), checks.join(","), cookies.join(","), score, grade))
        }

        // ── FUZZ: Parameter Fuzzer ────────────────────────────────────
        ("POST", "/fuzz/run") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let wordlist = extract_json_str(body, "wordlist").unwrap_or("common".into());
            let params: Vec<&str> = match wordlist.as_str() {
                "extended" => vec!["id","page","search","q","query","name","user","username","email","pass","password",
                    "token","key","api_key","file","path","url","redirect","next","return",
                    "callback","action","cmd","debug","test","admin","config","db","sort","limit","offset","format",
                    "type","category","status","lang","filter","from","to","date"],
                _ => vec!["id","page","search","q","name","user","email","token","file","url",
                    "redirect","action","cmd","debug","admin","config","sort","limit","type","category","status"],
            };
            let mut results = Vec::new();
            let baseline = ureq::get(&url).set("User-Agent","Qomni-Fuzzer/1.0").timeout(std::time::Duration::from_secs(10)).call()
                .ok().and_then(|r| r.into_string().ok()).map(|s| s.len()).unwrap_or(0);
            for param in &params {
                let tu = format!("{}{}{}=FUZZ", url, if url.contains('?'){"&"}else{"?"}, param);
                match ureq::get(&tu).set("User-Agent","Qomni-Fuzzer/1.0").timeout(std::time::Duration::from_secs(8)).call() {
                    Ok(resp) => {
                        let st = resp.status();
                        let ln = resp.into_string().ok().map(|s| s.len()).unwrap_or(0);
                        if st != 404 && (ln != baseline || st != 200) {
                            results.push(format!(r#"{{"param":"{}","status":{},"length":{},"interesting":true}}"#, param, st, ln));
                        }
                    }
                    Err(ureq::Error::Status(code, resp)) => {
                        let ln = resp.into_string().ok().map(|s| s.len()).unwrap_or(0);
                        results.push(format!(r#"{{"param":"{}","status":{},"length":{},"interesting":true}}"#, param, code, ln));
                    }
                    _ => {}
                }
            }
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","results":[{}],"total_tested":{},"interesting":{}}}"#,
                url, results.join(","), params.len(), results.len()))
        }

        // ── FUZZ: Directory Bruteforce ────────────────────────────────
        ("POST", "/fuzz/dirbust") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let extensions = extract_json_str(body, "extensions").unwrap_or(".php,.bak,.sql".into());
            let dirs: Vec<&str> = vec!["admin","login","wp-admin","backup",".git",".env","config","phpmyadmin",
                "api","robots.txt","sitemap.xml",".htaccess","test","dev","staging","uploads",
                "server-status","xmlrpc.php","readme.html","console","dashboard","panel",
                "tmp","logs","dump","database","package.json","composer.json","Dockerfile",
                ".well-known","security.txt","graphql","swagger","api-docs"];
            let exts: Vec<&str> = extensions.split(',').map(|e| e.trim()).collect();
            let base = url.trim_end_matches('/');
            let mut found = Vec::new();
            for dir in &dirs {
                let tu = format!("{}/{}", base, dir);
                match ureq::get(&tu).set("User-Agent","Qomni-Dirbust/1.0").timeout(std::time::Duration::from_secs(8)).call() {
                    Ok(resp) => {
                        let st = resp.status(); let ln = resp.into_string().ok().map(|s|s.len()).unwrap_or(0);
                        if st != 404 { found.push(format!(r#"{{"path":"{}","status":{},"length":{}}}"#, dir, st, ln)); }
                    }
                    Err(ureq::Error::Status(code, resp)) => {
                        let ln = resp.into_string().ok().map(|s|s.len()).unwrap_or(0);
                        if code != 404 { found.push(format!(r#"{{"path":"{}","status":{},"length":{}}}"#, dir, code, ln)); }
                    }
                    _ => {}
                }
                for ext in &exts {
                    if dir.contains('.') { continue; }
                    let tu = format!("{}/{}{}", base, dir, ext);
                    match ureq::get(&tu).set("User-Agent","Qomni-Dirbust/1.0").timeout(std::time::Duration::from_secs(6)).call() {
                        Ok(resp) => { let st=resp.status(); let ln=resp.into_string().ok().map(|s|s.len()).unwrap_or(0); if st!=404 { found.push(format!(r#"{{"path":"{}{}","status":{},"length":{}}}"#, dir, ext, st, ln)); } }
                        Err(ureq::Error::Status(code, resp)) => { let ln=resp.into_string().ok().map(|s|s.len()).unwrap_or(0); if code!=404 { found.push(format!(r#"{{"path":"{}{}","status":{},"length":{}}}"#, dir, ext, code, ln)); } }
                        _ => {}
                    }
                }
            }
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","found":[{}],"total_tested":{},"found_count":{}}}"#,
                url, found.join(","), dirs.len(), found.len()))
        }

        // ── FUZZ: HTTP Method Testing ─────────────────────────────────
        ("POST", "/fuzz/methods") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let methods = ["GET","POST","PUT","DELETE","PATCH","OPTIONS","HEAD","TRACE"];
            let mut results = Vec::new();
            for method in &methods {
                if let Ok(out) = std::process::Command::new("curl")
                    .args(&["-s","-o","/dev/null","-w","%{http_code}:%{size_download}","-X",method,&url,"--max-time","8"])
                    .output() {
                    let text = String::from_utf8_lossy(&out.stdout);
                    let parts: Vec<&str> = text.splitn(2,':').collect();
                    let status: u32 = parts.get(0).unwrap_or(&"0").parse().unwrap_or(0);
                    let size: u64 = parts.get(1).unwrap_or(&"0").parse().unwrap_or(0);
                    results.push(format!(r#"{{"method":"{}","status":{},"size":{},"allowed":{}}}"#,
                        method, status, size, status!=405 && status!=501 && status!=0));
                }
            }
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","methods":[{}]}}"#, url, results.join(",")))
        }

        // ── AUTH: CORS Test ───────────────────────────────────────────
        ("POST", "/auth/cors-test") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let origins = ["https://evil.com","null","https://attacker.example.com","http://localhost"];
            let mut results = Vec::new();
            for origin in &origins {
                if let Ok(out) = std::process::Command::new("curl")
                    .args(&["-s","-I","-H",&format!("Origin: {}",origin),&url,"--max-time","10"])
                    .output() {
                    let text = String::from_utf8_lossy(&out.stdout).to_string();
                    let acao = text.lines().find(|l| l.to_lowercase().starts_with("access-control-allow-origin"))
                        .map(|l| l.splitn(2,':').nth(1).unwrap_or("").trim().to_string()).unwrap_or_default();
                    let acac = text.lines().any(|l| l.to_lowercase().contains("access-control-allow-credentials") && l.to_lowercase().contains("true"));
                    let reflected = acao == *origin || acao == "*";
                    let vuln = reflected && (acac || acao == "*");
                    results.push(format!(r#"{{"origin":"{}","acao":"{}","acac":{},"reflected":{},"vulnerable":{}}}"#,
                        origin, acao.replace('"',"'"), acac, reflected, vuln));
                }
            }
            let any_v = results.iter().any(|r| r.contains("\"vulnerable\":true"));
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","vulnerable":{},"tests":[{}]}}"#, url, any_v, results.join(",")))
        }

        // ── AUTH: Session Analysis ────────────────────────────────────
        ("POST", "/auth/session-analysis") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let mut cookies = Vec::new();
            let mut issues = Vec::new();
            match ureq::get(&url).set("User-Agent","Qomni-Session/1.0").timeout(std::time::Duration::from_secs(15)).call() {
                Ok(resp) => {
                    for name in resp.headers_names() {
                        if name.to_lowercase() == "set-cookie" {
                            if let Some(val) = resp.header(&name) {
                                let lv = val.to_lowercase();
                                let cn = val.split('=').next().unwrap_or("?");
                                let sec = lv.contains("secure"); let ho = lv.contains("httponly"); let ss = lv.contains("samesite");
                                if !sec { issues.push(format!(r#"{{"cookie":"{}","issue":"Missing Secure flag","severity":"high"}}"#, cn)); }
                                if !ho { issues.push(format!(r#"{{"cookie":"{}","issue":"Missing HttpOnly flag","severity":"high"}}"#, cn)); }
                                if !ss { issues.push(format!(r#"{{"cookie":"{}","issue":"Missing SameSite attribute","severity":"medium"}}"#, cn)); }
                                cookies.push(format!(r#"{{"name":"{}","secure":{},"httponly":{},"samesite":{}}}"#, cn, sec, ho, ss));
                            }
                        }
                    }
                }
                Err(ureq::Error::Status(_, resp)) => {
                    for name in resp.headers_names() {
                        if name.to_lowercase() == "set-cookie" {
                            if let Some(val) = resp.header(&name) {
                                let cn = val.split('=').next().unwrap_or("?");
                                cookies.push(format!(r#"{{"name":"{}","secure":false,"httponly":false,"samesite":false}}"#, cn));
                            }
                        }
                    }
                }
                Err(_) => {}
            }
            if url.starts_with("http://") { issues.push(r#"{"cookie":"*","issue":"Site uses HTTP not HTTPS","severity":"critical"}"#.into()); }
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","cookies":[{}],"issues":[{}],"total_cookies":{},"total_issues":{}}}"#,
                url, cookies.join(","), issues.join(","), cookies.len(), issues.len()))
        }

        // ── AUTH: Endpoint Discovery ──────────────────────────────────
        ("POST", "/auth/discover") => {
            let url = match extract_json_str(body, "url") {
                Some(u) => u,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing url"}"#.into()),
            };
            let base = url.trim_end_matches('/');
            let paths = ["/login","/signin","/auth","/oauth","/sso","/api/login","/api/auth","/api/token",
                "/wp-login.php","/admin/login","/user/login","/accounts/login","/register","/signup",
                "/forgot-password","/reset-password","/2fa","/mfa","/logout","/signout",
                "/token","/api/users/me","/api/profile","/me","/dashboard","/admin","/panel","/console",
                "/.well-known/openid-configuration","/oauth/authorize","/oauth/token","/graphql"];
            let mut found = Vec::new();
            for path in &paths {
                let tu = format!("{}{}", base, path);
                if let Ok(out) = std::process::Command::new("curl")
                    .args(&["-s","-o","/dev/null","-w","%{http_code}",&tu,"--max-time","6","-L"])
                    .output() {
                    let status: u32 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0);
                    if status != 404 && status != 0 {
                        let ptype = if path.contains("login")||path.contains("signin")||path.contains("auth") {"login"}
                            else if path.contains("register")||path.contains("signup") {"register"}
                            else if path.contains("forgot")||path.contains("reset") {"recovery"}
                            else if path.contains("oauth")||path.contains("sso") {"sso"}
                            else if path.contains("api")||path.contains("token") {"api"}
                            else if path.contains("admin")||path.contains("panel")||path.contains("dashboard") {"admin"}
                            else {"other"};
                        found.push(format!(r#"{{"path":"{}","status":{},"type":"{}"}}"#, path, status, ptype));
                    }
                }
            }
            ("200 OK", format!(r#"{{"ok":true,"url":"{}","endpoints":[{}],"total_found":{}}}"#, url, found.join(","), found.len()))
        }


        // ── POST /plan/simulate — Multi-simulation sweep ────────────
        ("POST", "/plan/simulate") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan'"}"#.into()),
            };
            let output_step = extract_json_str(body, "output_step");

            // Parse sweep object
            let sweep_obj = match body.find("\"sweep\"") {
                Some(pos) => {
                    let rest = &body[pos + 7..];
                    let start = match rest.find('{') {
                        Some(s) => s,
                        None => return ("400 Bad Request", r#"{"ok":false,"error":"missing sweep object"}"#.into()),
                    };
                    let rest2 = &rest[start..];
                    // Find matching closing brace
                    let mut depth = 0i32;
                    let mut end = 0;
                    for (i, ch) in rest2.bytes().enumerate() {
                        match ch {
                            b'{' => depth += 1,
                            b'}' => { depth -= 1; if depth == 0 { end = i + 1; break; } }
                            _ => {}
                        }
                    }
                    if end == 0 {
                        return ("400 Bad Request", r#"{"ok":false,"error":"malformed sweep"}"#.into());
                    }
                    rest2[..end].to_string()
                }
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'sweep'"}"#.into()),
            };

            // Parse each sweep parameter
            let aot = match AOT_CACHE.get() {
                Some(a) => a,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"AOT cache not initialized"}"#.into()),
            };
            let tidx = match aot.turbo_index(&plan_name) {
                Some(i) => i,
                None => return ("400 Bad Request", format!(r#"{{"ok":false,"error":"plan '{}' not in turbo table"}}"#, plan_name)),
            };
            let tp = aot.turbo_plan(tidx).unwrap();

            // Build param value lists for each sweep param
            let mut param_value_lists: Vec<(String, Vec<f64>)> = Vec::new();
            // Parse sweep keys
            let sweep_inner = &sweep_obj[1..sweep_obj.len()-1]; // strip outer {}
            // Simple parser: find "key": then value (object with min/max/step OR array)
            let mut spos = 0;
            while spos < sweep_inner.len() {
                let key_start = match sweep_inner[spos..].find('"') {
                    Some(p) => spos + p + 1,
                    None => break,
                };
                let key_end = match sweep_inner[key_start..].find('"') {
                    Some(p) => key_start + p,
                    None => break,
                };
                let key = sweep_inner[key_start..key_end].to_string();

                // Skip to value (after colon)
                let after_key = &sweep_inner[key_end + 1..];
                let colon_pos = match after_key.find(':') {
                    Some(p) => key_end + 1 + p + 1,
                    None => break,
                };
                let val_str = sweep_inner[colon_pos..].trim_start();

                if val_str.starts_with('{') {
                    // Range object: {"min":X,"max":Y,"step":Z}
                    let obj_end = match val_str.find('}') {
                        Some(p) => p + 1,
                        None => break,
                    };
                    let obj = &val_str[..obj_end];
                    let min_v = extract_json_float(obj, "min").unwrap_or(0.0);
                    let max_v = extract_json_float(obj, "max").unwrap_or(0.0);
                    let step_v = extract_json_float(obj, "step").unwrap_or(1.0);
                    let mut values = Vec::new();
                    let mut v = min_v;
                    while v <= max_v + step_v * 0.01 {
                        values.push(v);
                        v += step_v;
                    }
                    param_value_lists.push((key, values));
                    spos = colon_pos + obj_end;
                } else if val_str.starts_with('[') {
                    // Explicit array
                    let arr_end = match val_str.find(']') {
                        Some(p) => p + 1,
                        None => break,
                    };
                    let arr = &val_str[1..arr_end-1];
                    let values: Vec<f64> = arr.split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    param_value_lists.push((key, values));
                    spos = colon_pos + arr_end;
                } else {
                    // Scalar fixed value (e.g., "charset_size": 95)
                    // Find the end of the number (next comma, closing brace, or whitespace)
                    let num_str = val_str.trim_start();
                    let num_end = num_str.find(|c: char| c == ',' || c == '}' || c == '"')
                        .unwrap_or(num_str.len());
                    let num_part = num_str[..num_end].trim();
                    if let Ok(v) = num_part.parse::<f64>() {
                        param_value_lists.push((key, vec![v]));
                    }
                    spos = colon_pos + num_end;
                }
            }

            if param_value_lists.is_empty() {
                return ("400 Bad Request", r#"{"ok":false,"error":"no sweep parameters parsed"}"#.into());
            }

            // Generate cartesian product of all param value lists
            let mut grid: Vec<Vec<(String, f64)>> = vec![Vec::new()];
            for (key, values) in &param_value_lists {
                let mut new_grid = Vec::new();
                for combo in &grid {
                    for &v in values {
                        let mut new_combo = combo.clone();
                        new_combo.push((key.clone(), v));
                        new_grid.push(new_combo);
                    }
                }
                grid = new_grid;
            }

            let n_scenarios = grid.len();
            if n_scenarios > 100_000 {
                return ("400 Bad Request", r#"{"ok":false,"error":"too many scenarios (max 100000)"}"#.into());
            }

            // Build param_sets for batch execution
            let mut param_sets: Vec<Vec<f64>> = Vec::with_capacity(n_scenarios);
            for combo in &grid {
                let mut params_arr = vec![0.0f64; tp.n_params];
                for (key, val) in combo {
                    if let Some(pi) = tp.param_names.iter().position(|p| p == key) {
                        params_arr[pi] = *val;
                    }
                }
                param_sets.push(params_arr);
            }

            // Execute all via batch_execute_turbo
            let t0 = std::time::Instant::now();
            let all_results = aot.batch_execute_turbo(tidx, &param_sets);
            let total_ns = t0.elapsed().as_nanos() as u64;
            let avg_ns = if n_scenarios > 0 { total_ns / n_scenarios as u64 } else { 0 };

            // Find output_step index
            let out_idx = output_step.as_ref().and_then(|name| {
                tp.step_names.iter().position(|s| s == name)
            });

            // Build results JSON
            let mut results_json = Vec::with_capacity(n_scenarios.min(10000));
            for (i, combo) in grid.iter().enumerate().take(10000) {
                let mut params_obj = String::from("{");
                for (j, (key, val)) in combo.iter().enumerate() {
                    if j > 0 { params_obj.push(','); }
                    if *val == val.floor() && val.abs() < 1e12 {
                        params_obj.push_str(&format!("\"{}\":{:.0}", key, val));
                    } else {
                        params_obj.push_str(&format!("\"{}\":{:.4}", key, val));
                    }
                }
                params_obj.push('}');

                let result_val = if let Some(oi) = out_idx {
                    all_results.get(i).and_then(|r| r.get(oi)).copied().unwrap_or(0.0)
                } else {
                    // Return first step result
                    all_results.get(i).and_then(|r| r.first()).copied().unwrap_or(0.0)
                };
                let rv = if result_val == result_val.floor() && result_val.abs() < 1e12 {
                    format!("{:.1}", result_val)
                } else {
                    format!("{:.4}", result_val)
                };

                if let Some(ref step_name) = output_step {
                    results_json.push(format!(r#"{{"params":{},"{}":{}}}"#, params_obj, step_name, rv));
                } else {
                    results_json.push(format!(r#"{{"params":{},"result":{}}}"#, params_obj, rv));
                }
            }

            ("200 OK", format!(
                r#"{{"ok":true,"plan":"{}","scenarios":{},"results":[{}],"total_ns":{},"avg_ns_per_scenario":{}}}"#,
                plan_name, n_scenarios, results_json.join(","), total_ns, avg_ns
            ))
        }

        // ── POST /plan/reactive/create — Create a reactive session ──
        ("POST", "/plan/reactive/create") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan'"}"#.into()),
            };
            let params_map = match extract_json_obj_float(body, "params") {
                Some(p) => p,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'params' object"}"#.into()),
            };

            let aot = match AOT_CACHE.get() {
                Some(a) => a,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"AOT not initialized"}"#.into()),
            };
            let tidx = match aot.turbo_index(&plan_name) {
                Some(i) => i,
                None => return ("400 Bad Request", format!(r#"{{"ok":false,"error":"plan '{}' not in turbo table"}}"#, plan_name)),
            };
            let tp = aot.turbo_plan(tidx).unwrap();

            // Build ordered params array
            let mut params_arr = vec![0.0f64; tp.n_params];
            for (key, val) in &params_map {
                if let Some(pi) = tp.param_names.iter().position(|p| p == key) {
                    params_arr[pi] = *val;
                }
            }

            // Execute plan
            let t0 = std::time::Instant::now();
            let (results_arr, n) = match aot.execute_turbo(tidx, &params_arr) {
                Some(r) => r,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"turbo execution failed"}"#.into()),
            };
            let exec_ns = t0.elapsed().as_nanos() as u64;
            let n = n.min(tp.n_steps).min(32);
            let cached_results: Vec<f64> = results_arr[..n].to_vec();

            // Get step dependencies
            let step_deps = aot.step_dependencies(&plan_name).unwrap_or_default();

            // Generate session ID
            let session_id = format!("rs-{:x}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());

            // Build result JSON
            let mut steps_json = Vec::new();
            for (i, result) in cached_results.iter().enumerate() {
                let name = tp.step_names.get(i).cloned().unwrap_or_else(|| format!("step_{}", i));
                let rv = if *result == result.floor() && result.abs() < 1e12 {
                    format!("{:.2}", result)
                } else {
                    format!("{:.6}", result)
                };
                steps_json.push(format!(r#"{{"step":"{}","value":{}}}"#, name, rv));
            }

            let session = ReactiveSession {
                id: session_id.clone(),
                plan_name: plan_name.clone(),
                plan_idx: tidx,
                params: params_map.clone(),
                cached_results: cached_results.clone(),
                step_deps,
                last_access: std::time::Instant::now(),
            };

            // Store session
            session_insert(session);

            ("200 OK", format!(
                r#"{{"ok":true,"session_id":"{}","plan":"{}","steps":[{}],"exec_ns":{}}}"#,
                session_id, plan_name, steps_json.join(","), exec_ns
            ))
        }

        // ── POST /plan/reactive/update — Update params, recompute dirty steps ──
        ("POST", "/plan/reactive/update") => {
            let session_id = match extract_json_str(body, "session_id") {
                Some(s) => s,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'session_id'"}"#.into()),
            };
            let changed_params = match extract_json_obj_float(body, "params") {
                Some(p) => p,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'params' object"}"#.into()),
            };

            let aot = match AOT_CACHE.get() {
                Some(a) => a,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"AOT not initialized"}"#.into()),
            };

            let mut session_ref = match session_map().get_mut(&session_id) {
                Some(s) => s,
                None => return ("404 Not Found", r#"{"ok":false,"error":"session not found"}"#.into()),
            };
            let session = session_ref.value_mut();
            session.last_access = std::time::Instant::now();

            let tp = match aot.turbo_plan(session.plan_idx) {
                Some(t) => t,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"plan gone from turbo table"}"#.into()),
            };

            // Determine which param indices changed
            let mut changed_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for (key, val) in &changed_params {
                if let Some(pi) = tp.param_names.iter().position(|p| p == key) {
                    let old_val = session.params.get(key).copied().unwrap_or(0.0);
                    if (old_val - val).abs() > 1e-15 {
                        session.params.insert(key.clone(), *val);
                        changed_indices.insert(pi);
                    }
                }
            }

            if changed_indices.is_empty() {
                // Nothing changed, return cached results
                let mut steps_json = Vec::new();
                for (i, result) in session.cached_results.iter().enumerate() {
                    let name = tp.step_names.get(i).cloned().unwrap_or_else(|| format!("step_{}", i));
                    steps_json.push(format!(r#"{{"step":"{}","value":{:.6},"recomputed":false}}"#, name, result));
                }
                return ("200 OK", format!(
                    r#"{{"ok":true,"session_id":"{}","dirty_steps":0,"steps":[{}]}}"#,
                    session_id, steps_json.join(",")
                ));
            }

            // Mark dirty steps: any step that depends (directly or transitively) on changed params
            let n_steps = session.cached_results.len();
            let mut dirty: Vec<bool> = vec![false; n_steps];
            for si in 0..n_steps {
                if si < session.step_deps.len() {
                    for &dep_pi in &session.step_deps[si] {
                        if changed_indices.contains(&dep_pi) {
                            dirty[si] = true;
                            break;
                        }
                    }
                }
            }

            // Also mark steps that depend on dirty steps (transitive through step deps)
            // Simple: re-scan since step_deps already includes transitive param deps
            // But we also need to handle step->step chains. Re-execute the whole plan
            // (it's ~10ns) and only report which steps changed.

            // Build ordered params array from session.params
            let mut params_arr = vec![0.0f64; tp.n_params];
            for (key, val) in &session.params {
                if let Some(pi) = tp.param_names.iter().position(|p| p == key) {
                    params_arr[pi] = *val;
                }
            }

            let t0 = std::time::Instant::now();
            let (new_results_arr, new_n) = match aot.execute_turbo(session.plan_idx, &params_arr) {
                Some(r) => r,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"re-execution failed"}"#.into()),
            };
            let exec_ns = t0.elapsed().as_nanos() as u64;
            let new_n = new_n.min(tp.n_steps).min(32);

            let mut dirty_count = 0;
            let mut steps_json = Vec::new();
            for i in 0..new_n {
                let name = tp.step_names.get(i).cloned().unwrap_or_else(|| format!("step_{}", i));
                let new_val = new_results_arr[i];
                let old_val = session.cached_results.get(i).copied().unwrap_or(0.0);
                let is_dirty = dirty.get(i).copied().unwrap_or(false);
                let changed = (new_val - old_val).abs() > 1e-15;
                let recomputed = is_dirty || changed;
                if recomputed { dirty_count += 1; }
                let rv = if new_val == new_val.floor() && new_val.abs() < 1e12 {
                    format!("{:.2}", new_val)
                } else {
                    format!("{:.6}", new_val)
                };
                steps_json.push(format!(
                    r#"{{"step":"{}","value":{},"recomputed":{}}}"#,
                    name, rv, recomputed
                ));
            }

            // Update cached results
            session.cached_results = new_results_arr[..new_n].to_vec();

            ("200 OK", format!(
                r#"{{"ok":true,"session_id":"{}","dirty_steps":{},"total_steps":{},"exec_ns":{},"steps":[{}]}}"#,
                session_id, dirty_count, new_n, exec_ns, steps_json.join(",")
            ))
        }

        // ── POST /plan/batch_bench — Benchmark batch execution ──────
        ("POST", "/plan/batch_bench") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan'"}"#.into()),
            };
            let params = match extract_json_arr_float(body, "params") {
                Some(p) => p,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'params' array"}"#.into()),
            };

            let aot = match AOT_CACHE.get() {
                Some(a) => a,
                None => return ("500 Internal Server Error", r#"{"ok":false,"error":"AOT not initialized"}"#.into()),
            };
            let tidx = match aot.turbo_index(&plan_name) {
                Some(i) => i,
                None => return ("400 Bad Request", format!(r#"{{"ok":false,"error":"plan '{}' not in turbo table"}}"#, plan_name)),
            };

            // Build 1000 identical param sets
            let n_sets = 1000usize;
            let param_sets: Vec<Vec<f64>> = (0..n_sets).map(|_| params.clone()).collect();

            // Warmup
            let _ = aot.batch_execute_turbo(tidx, &param_sets[..100.min(n_sets)]);

            // Timed run
            let t0 = std::time::Instant::now();
            let results = aot.batch_execute_turbo(tidx, &param_sets);
            let total_ns = t0.elapsed().as_nanos() as u64;
            let avg_ns = total_ns / n_sets as u64;

            let first_result = results.first().map(|r| {
                r.iter().map(|v| {
                    if *v == v.floor() && v.abs() < 1e12 { format!("{:.1}", v) } else { format!("{:.4}", v) }
                }).collect::<Vec<_>>().join(",")
            }).unwrap_or_default();

            ("200 OK", format!(
                r#"{{"ok":true,"plan":"{}","batch_size":{},"total_ns":{},"avg_ns_per_execution":{},"first_result":[{}]}}"#,
                plan_name, n_sets, total_ns, avg_ns, first_result
            ))
        }



        // ── POST /plan/simd_batch — AVX2/AVX-512 plan-level SIMD batch ──
        // Body: {"plan":"plan_pump_sizing","scenarios":[[500,100,0.75],[300,80,0.7],...]}
        // or:   {"plan":"plan_pump_sizing","n":400,"params":[500,100,0.75]}  (N identical)
        // Auto-selects: scalar / AVX2x4 / AVX-512x8 / rayon+AVX2 based on CPU+batch size
        ("POST", "/plan/simd_batch") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan'"}"#.into()),
            };

            // Build param_sets either from "scenarios" array or "n" + "params"
            let param_sets: Vec<Vec<f64>> = if body.contains("\"scenarios\"") {
                // Parse [[p0,p1,...],[p0,p1,...],...] — simplified parser
                let start = match body.find("\"scenarios\"") {
                    Some(p) => p,
                    None    => return ("400 Bad Request", r#"{"ok":false,"error":"bad scenarios"}"#.into()),
                };
                let arr_start = match body[start..].find('[') { Some(p) => start+p+1, None => return ("400 Bad Request", r#"{"ok":false,"error":"bad scenarios"}"#.into()) };
                let mut sets = Vec::new();
                let mut pos = arr_start;
                while pos < body.len() {
                    let inner = body[pos..].trim_start();
                    if inner.starts_with('[') {
                        let end = match inner.find(']') { Some(p) => p+1, None => break };
                        let row_str = &inner[1..end-1];
                        let row: Vec<f64> = row_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                        if !row.is_empty() { sets.push(row); }
                        pos += inner.len() - inner.trim_start_matches('[').len() + end;
                    } else if inner.starts_with(']') { break; }
                    else { pos += 1; }
                }
                sets
            } else {
                let base_params = match extract_json_arr_float(body, "params") {
                    Some(p) => p,
                    None => return ("400 Bad Request", r#"{"ok":false,"error":"missing params or scenarios"}"#.into()),
                };
                let n = extract_json_float(body, "n").map(|v| v as usize).unwrap_or(100).min(100_000);
                (0..n).map(|_| base_params.clone()).collect()
            };

            let n = param_sets.len();
            let n_outputs = 5usize; // default; plan-specific

            // Try parallel batch first (>= 8 scenarios), then single-thread SIMD
            let result = if n >= 8 {
                batch_plan::execute_parallel_batch(&plan_name, &param_sets, n_outputs)
                    .or_else(|| batch_plan::execute_simd_batch(&plan_name, &param_sets, n_outputs))
            } else {
                batch_plan::execute_simd_batch(&plan_name, &param_sets, n_outputs)
            };

            match result {
                Some(r) => {
                    let ns_per = if n > 0 { r.time_ns / n as u64 } else { 0 };
                    let avx_note = if is_x86_feature_detected!("avx512f") {
                        "avx512f+avx2+fma3"
                    } else if is_x86_feature_detected!("avx2") {
                        "avx2+fma3"
                    } else { "scalar" };
                    ("200 OK", format!(
                        r#"{{"ok":true,"plan":"{}","n_scenarios":{},"path":"{}","cpu_features":"{}","total_ns":{},"ns_per_scenario":{},"n_outputs":{}}}"#,
                        plan_name, n, r.path.name(), avx_note,
                        r.time_ns, ns_per, r.n_outputs
                    ))
                }
                None => ("400 Bad Request", format!(
                    r#"{{"ok":false,"error":"no SIMD kernel for plan '{}'","available":["plan_pump_sizing","plan_voltage_drop","plan_planilla","plan_beam_analysis"]}}"#,
                    plan_name
                )),
            }
        }

        // -- POST /simulation/start
        ("POST", "/simulation/start") => {
            let plan_str = extract_json_str(body, "plan").unwrap_or_else(|| "plan_pump_sizing".into());
            let mode_str = extract_json_str(body, "mode").unwrap_or_else(|| "stratified".into());
            use crate::simulation_engine::{SweepMode, SweepSpec};
            let sweep = match mode_str.as_str() {
                "stress"      => SweepSpec::pump_stress(),
                "adversarial" => SweepSpec { mode: SweepMode::Adversarial, ..SweepSpec::pump_stress() },
                _             => SweepSpec::pump_default(),
            };
            let plan_id = PlanId::from_str(&plan_str).unwrap_or(PlanId::PumpSizing);
            let eng = global_engine();
            if eng.is_running() {
                ("200 OK", format!("{{\"ok\":true,\"status\":\"already_running\",\"plan\":\"{}\"}}", plan_str))
            } else {
                eng.start(plan_id, sweep);
                ("200 OK", format!("{{\"ok\":true,\"status\":\"started\",\"plan\":\"{}\",\"mode\":\"{}\",\"block_size\":256,\"arch\":\"l1_block+avx2+physics+pareto\"}}", plan_str, mode_str))
            }
        }

        // -- GET /simulation/status
        ("GET", "/simulation/status") => {
            let eng = global_engine();
            if !eng.is_running() {
                ("200 OK", "{\"ok\":false,\"error\":\"not running\"}".to_string())
            } else {
                let s = eng.get_stats();
                let pareto_json = match &s.pareto_front {
                    Some(pf) => {
                        let sols: Vec<String> = pf.solutions.iter().take(5).map(|sol| {
                            format!("{{\"hp\":{:.3},\"eff\":{:.3},\"cost\":{:.0},\"risk\":{:.3},\"p\":[{:.1},{:.1},{:.3}]}}",
                                sol.hp_req, sol.eff_score, sol.cost_usd, sol.risk_score,
                                sol.params[0], sol.params[1], sol.params[2])
                        }).collect();
                        format!("{{\"size\":{},\"n_dominated\":{},\"top5\":[{}]}}",
                            pf.solutions.len(), pf.n_dominated, sols.join(","))
                    }
                    None => "null".into(),
                };
                let numa_json = match &s.numa {
                    Some(n) => format!("{{\"nodes\":{},\"l1d_kb\":{},\"block_fits_l1\":{},\"avx512\":{},\"avx2\":{},\"advisory\":\"{}\"}}",
                        n.node_count, n.l1d_kb, n.block_fits_l1, n.avx512, n.avx2, n.advisory),
                    None => "null".into(),
                };
                ("200 OK", format!("{{\"ok\":true,\"running\":true,\"ticks\":{},\"scenarios_total\":{},\"per_s\":{},\"valid_frac\":{:.3},\"invalid_total\":{},\"tick_ns\":{},\"kernel\":\"{}\",\"mode\":\"{}\",\"pareto_size\":{},\"pareto\":{},\"numa\":{}}}",
                    s.ticks, s.scenarios_total, s.scenarios_per_s, s.valid_fraction,
                    s.invalid_count, s.last_tick_ns, s.kernel_path, s.sweep_mode,
                    s.pareto_size, pareto_json, numa_json))
            }
        }

        // -- POST /simulation/stop
        ("POST", "/simulation/stop") => {
            global_engine().stop();
            ("200 OK", "{\"ok\":true,\"status\":\"stopped\"}".to_string())
        }

        // ── POST /decision/analyze — Decision Engine analysis ─────────
        ("POST", "/decision/analyze") => {
            let plan_name = match extract_json_str(body, "plan") {
                Some(n) => n,
                None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan' field"}"#.into()),
            };

            // Check if we should execute first or use pre-computed steps
            let execute_flag = body.contains("\"execute\":true") || body.contains("\"execute\": true");

            let (steps_data, total_ns) = if execute_flag {
                // Execute the plan first, then analyze
                let params = match extract_json_obj_float(body, "params") {
                    Some(p) => p,
                    None    => std::collections::HashMap::new(),
                };
                let t0 = std::time::Instant::now();
                let aot_result: Option<crate::plan::PlanResult> = AOT_CACHE.get()
                    .and_then(|aot| if aot.has_plan(&plan_name) { aot.execute(&plan_name, &params).ok() } else { None });
                let exec_result = if let Some(r) = aot_result {
                    Ok(r)
                } else {
                    let executor = PlanExecutor::new(plans.as_slice());
                    let executor = match jit_map.as_ref() {
                        Some(map) => executor.with_jit_map(map.clone()),
                        None      => executor,
                    };
                    executor.execute(&plan_name, params.clone())
                };
                match exec_result {
                    Ok(r) => {
                        let ns = t0.elapsed().as_nanos() as u64;
                        let steps: Vec<(String, String, f64)> = r.steps.iter()
                            .map(|s| (s.step.clone(), s.oracle.clone(), s.value))
                            .collect();
                        (steps, ns)
                    }
                    Err(e) => return ("400 Bad Request", format!(
                        r#"{{"ok":false,"error":"execution failed: {}"}}"#, escape_json(&e)
                    )),
                }
            } else {
                // Pre-computed steps provided in body
                let mut steps: Vec<(String, String, f64)> = Vec::new();
                let steps_str = {
                    let pattern = "\"steps\":";
                    if let Some(idx) = body.find(pattern) {
                        let rest = body[idx + pattern.len()..].trim();
                        if rest.starts_with('[') {
                            let end = rest.find(']').unwrap_or(rest.len());
                            rest[..end+1].to_string()
                        } else { "[]".to_string() }
                    } else { "[]".to_string() }
                };
                let mut pos = 0usize;
                let bytes = steps_str.as_bytes();
                while pos < bytes.len() {
                    if bytes[pos] == b'{' {
                        let obj_end = steps_str[pos..].find('}').map(|i| pos + i + 1).unwrap_or(bytes.len());
                        let obj = &steps_str[pos..obj_end];
                        let step_name = extract_json_str(obj, "step").unwrap_or_default();
                        let oracle = extract_json_str(obj, "oracle").unwrap_or_default();
                        let result = extract_json_float(obj, "result").unwrap_or(0.0);
                        if !step_name.is_empty() {
                            steps.push((step_name, oracle, result));
                        }
                        pos = obj_end;
                    } else {
                        pos += 1;
                    }
                }
                (steps, 0u64)
            };

            let result = decision_analyze(&plan_name, &steps_data, total_ns);
            let enhanced = policy_enhance(&result, &plan_name, body);
            ("200 OK", enhanced)
        }

        // ── GET /decision/rules — List analysis rules for transparency ─
        ("GET", "/decision/rules") => {
            ("200 OK", decision_rules_json())
        }


        // ── Pipeline: compute → optimize → decide → explain ──────────────────
        ("POST", "/pipeline") => {
            handle_pipeline(body, plans.as_slice(), jit_map.as_ref())
        }

                // ── Audit Log ─────────────────────────────────────────────────────────
        ("GET", "/audit/recent") => {
            let n = extract_json_float(body, "n").unwrap_or(20.0) as usize;
            let entries = audit_read_recent(n.max(1).min(100));
            ("200 OK", format!(r#"{{"ok":true,"entries":{}}}"#, entries))
        }

        // ── Digital Twin ──────────────────────────────────────────────────────
        ("GET", "/twin/state") => {
            ("200 OK", format!(r#"{{"ok":true,"twin":{}}}"#, twin_get()))
        }

        // ── Multi-Strategy Pipeline ───────────────────────────────────────────
        ("POST", "/pipeline/multi_strategy") => {
            handle_multi_strategy(body, plans.as_slice(), jit_map.as_ref())
        }

        // ── Cluster Health ────────────────────────────────────────────────────
        ("GET", "/cluster/health") => {
            cluster_health_json()
        }

        // ── SLO / Health Detailed ────────────────────────────────────────────
        ("GET", "/health/detailed") => {
            health_detailed_json()
        }

        // ── Closed-Loop Verify ────────────────────────────────────────────────
        ("POST", "/verify") => {
            verify_prediction(body)
        }

        // ── Learning Stats ────────────────────────────────────────────────────
        ("GET", "/learn/stats") => {
            learn_stats_json()
        }

        // ── Level 7: Autonomous Loop Control ─────────────────────────────────
        ("POST", "/live/start") => {
            if LIVE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
                ("200 OK", r#"{"ok":true,"status":"already_running"}"#.into())
            } else {
                LIVE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
                if let (Some(p), Some(j)) = (LIVE_PLANS.get(), LIVE_JIT.get()) {
                    spawn_autonomous_loop(Arc::clone(p), Arc::clone(j));
                }
                ("200 OK", r#"{"ok":true,"status":"started","message":"Autonomous loop active"}"#.into())
            }
        }
        ("POST", "/live/stop") => {
            LIVE_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
            ("200 OK", r#"{"ok":true,"status":"stopped"}"#.into())
        }
        ("GET", "/live/status") => {
            live_status_json()
        }

        ("GET",  "/agents/list")       => agents_list_json(),
        ("POST", "/agents/analyze")    => agents_analyze(body),
        ("POST", "/agents/debate")     => agents_debate(body),
        // ── v2 Plan API ────────────────────────────────────────────
        ("GET", "/v2/plans") => {
            match std::fs::read_dir("/opt/crysl/plans") {
                Ok(entries) => {
                    let mut names: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .filter_map(|e| {
                            let n = e.file_name().to_string_lossy().into_owned();
                            if n.ends_with(".crysl") { Some(n.trim_end_matches(".crysl").to_string()) } else { None }
                        })
                        .collect();
                    names.sort();
                    let items: Vec<String> = names.iter().map(|n| format!("\"{}\"", n)).collect();
                    ("200 OK", format!("{{\"ok\":true,\"count\":{},\"plans\":[{}]}}", names.len(), items.join(",")))
                }
                Err(e) => ("500 Internal Server Error", format!("{{\"ok\":false,\"error\":\"{}\"}}", e)),
            }
        }

        ("POST", "/v2/plan/run") | ("POST", "/v2/plan/bench") => {
            let is_bench = path == "/v2/plan/bench";
            let plan_name = extract_json_str(&body, "plan").unwrap_or_default();
            if plan_name.is_empty() {
                return ("400 Bad Request", "{\"ok\":false,\"error\":\"missing plan field\"}".into());
            }
            let fpath = format!("/opt/crysl/plans/{}.crysl", plan_name);
            match std::fs::read_to_string(&fpath) {
                Err(_) => ("404 Not Found", format!("{{\"ok\":false,\"error\":\"plan not found: {}\"}}", plan_name)),
                Ok(plan_src) => match parse_plans(&plan_src) {
                    Err(errs) => {
                        let msgs: Vec<String> = errs.iter().map(|e| format!("\"{}\"", e.message)).collect();
                        ("400 Bad Request", format!("{{\"ok\":false,\"errors\":[{}]}}", msgs.join(",")))
                    }
                    Ok(ref plans) if plans.is_empty() => ("400 Bad Request", "{\"ok\":false,\"error\":\"empty file\"}".into()),
                    Ok(plans) => {
                        let plan = &plans[0];
                        // parse params from JSON body
                        let mut args = std::collections::HashMap::<String, f64>::new();
                        if let Some(ps) = body.find("\"params\"") {
                            if let Some(ob) = body[ps..].find('{') {
                                let start = ps + ob;
                                let mut depth = 0i32; let mut end = start;
                                for (i, ch) in body[start..].char_indices() {
                                    match ch { '{' => depth += 1, '}' => { depth -= 1; if depth == 0 { end = start + i + 1; break; } } _ => {} }
                                }
                                for kv in body[start..end].split(',') {
                                    let p: Vec<&str> = kv.splitn(2, ':').collect();
                                    if p.len() == 2 {
                                        let k = p[0].trim().trim_matches(|c| c == '"' || c == '{').trim().to_string();
                                        if let Ok(v) = p[1].trim().trim_matches('}').parse::<f64>() { args.insert(k, v); }
                                    }
                                }
                            }
                        }
                        if is_bench {
                            for _ in 0..200 { let _ = execute_plan(plan, &args); }
                            let t0 = std::time::Instant::now();
                            for _ in 0..2000 { let _ = execute_plan(plan, &args); }
                            let ns = t0.elapsed().as_nanos() as u64;
                            ("200 OK", format!("{{\"ok\":true,\"plan\":\"{}\",\"iters\":2000,\"total_ns\":{},\"per_iter_ns\":{}}}", plan_name, ns, ns/2000))
                        } else {
                            match execute_plan(plan, &args) {
                                Err(e) => ("400 Bad Request", format!("{{\"ok\":false,\"error\":\"{}\"}}", e)),
                                Ok(result) => {
                                    let outs: Vec<String> = result.outputs.iter().map(|(k,v)| format!("\"{}\":{:.6}", k, v)).collect();
                                    ("200 OK", format!("{{\"ok\":true,\"plan\":\"{}\",\"latency_ns\":{},\"outputs\":{{{}}}}}", plan_name, result.latency_ns, outs.join(",")))
                                }
                            }
                        }
                    }
                },
            }
        }


        // POST /compile — CRYS-L source → backend IR
        // body: {"src":"oracle ...", "backend":"llvm"} or {"src":"...", "target":"wasm"}
        ("POST", "/compile") => {
            let src_text = extract_json_str(body, "src").unwrap_or_else(|| body.to_string());
            let backend  = extract_json_str(body, "backend").unwrap_or_default();
            let target   = extract_json_str(body, "target").unwrap_or_default();

            let tokens = Lexer::new(&src_text).tokenize();
            match Parser::new(tokens).parse() {
                Err(e) => ("400 Bad Request",
                    format!(r#"{{"ok":false,"error":"parse: {}"}}"#, e.replace('"', "'"))),
                Ok(prog) => {
                    if backend == "llvm" || backend == "llvm18" {
                        ("200 OK", llvm_backend::handle_compile_llvm(body, &prog))
                    } else if target == "wasm" {
                        ("200 OK", wasm_backend::handle_compile_wasm(body, &prog))
                    } else {
                        ("400 Bad Request",
                            r#"{"ok":false,"error":"specify backend=llvm or target=wasm"}"#.into())
                    }
                }
            }
        }

        // POST /linalg/compute — run CRYS-L program with linalg built-ins via VM
        ("POST", "/linalg/compute") => {
            let src_text = extract_json_str(body, "src").unwrap_or_else(|| body.to_string());
            let tokens = Lexer::new(&src_text).tokenize();
            match Parser::new(tokens).parse() {
                Err(e) => ("400 Bad Request",
                    format!(r#"{{"ok":false,"error":"parse: {}"}}"#, e.replace('"', "'"))),
                Ok(prog) => {
                    let mut interp = vm.lock().unwrap();
                    match interp.run(&prog) {
                        Ok(output) => {
                            let lines: Vec<String> = output.iter()
                                .map(|s| { let q = '"'; format!("{}{}{}" , q, s.replace('"', "'"), q) })
                                .collect();
                            ("200 OK", format!(r#"{{"ok":true,"output":[{}]}}"#, lines.join(",")))
                        }
                        Err(e) => ("400 Bad Request",
                            format!(r#"{{"ok":false,"error":"{}"}}"#, e.replace('"', "'"))),
                    }
                }
            }
        }


        // ── Commander-Level Benchmark Proofs ─────────────────────────────────

        // POST /simulation/jitter_bench — Proof 1: Determinism vs Jitter
        // Body: {"ticks":10000}
        ("POST", "/simulation/jitter_bench") => {
            let ticks = extract_json_int_body(body, "ticks").unwrap_or(10000).clamp(1000, 100000) as usize;
            let result = benchmark_proofs::run_jitter_bench(ticks);
            ("200 OK", benchmark_proofs::jitter_to_json(&result))
        }

        // GET /simulation/simd_density — Proof 2: SIMD Saturation
        ("GET", "/simulation/simd_density") => {
            let result = benchmark_proofs::compute_simd_density(simulation_engine::global_engine());
            ("200 OK", benchmark_proofs::simd_density_to_json(&result))
        }

        // POST /simulation/adversarial — Proof 3: Poison Pill Shield
        // Body: {"ticks":5000}
        ("POST", "/simulation/adversarial") => {
            let ticks = extract_json_int_body(body, "ticks").unwrap_or(5000).clamp(500, 50000) as usize;
            let result = benchmark_proofs::run_adversarial(ticks);
            ("200 OK", benchmark_proofs::adversarial_to_json(&result))
        }

        // GET /benchmark/vs_llm — Proof 4: LLM Speedup Factor
        ("GET", "/benchmark/vs_llm") => {
            let result = benchmark_proofs::compute_llm_factor(simulation_engine::global_engine());
            ("200 OK", benchmark_proofs::llm_factor_to_json(&result))
        }



        // GET /benchmark/all — Run all 4 proofs in sequence (for reports)
        ("GET", "/benchmark/all") | ("POST", "/benchmark/all") => {
            let engine = simulation_engine::global_engine();
            let jitter  = benchmark_proofs::run_jitter_bench(5000);
            let simd    = benchmark_proofs::compute_simd_density(&engine);
            let adv     = benchmark_proofs::run_adversarial(2000);
            let llm     = benchmark_proofs::compute_llm_factor(&engine);
            ("200 OK", format!(
                r#"{{"ok":true,"suite":"commander_benchmark_v1","proofs":{{"jitter":{},"simd":{},"adversarial":{},"vs_llm":{}}}}}"#,
                benchmark_proofs::jitter_to_json(&jitter),
                benchmark_proofs::simd_density_to_json(&simd),
                benchmark_proofs::adversarial_to_json(&adv),
                benchmark_proofs::llm_factor_to_json(&llm)
            ))
        }

        _ => ("404 Not Found", r#"{"error":"not found"}"#.into()),
    }
}

// ── Decision Engine v1.0 ─────────────────────────────────────────────

fn decision_plan_label(plan: &str) -> &'static str {
    match plan {
        "plan_pump_sizing"              => "Fire Pump Analysis",
        "plan_electrical_load"          => "Electrical Load Analysis",
        "plan_voltage_drop"             => "Voltage Drop Analysis",
        "plan_power_factor_correction"  => "Power Factor Correction",
        "plan_nfpa13_demand"            => "NFPA 13 Hydraulic Demand",
        "plan_pipe_losses"              => "Pipe Loss Analysis",
        "plan_pump_selection"           => "Pump Selection",
        "plan_cvss_assessment"          => "CVSS Security Assessment",
        "plan_password_audit"           => "Password Strength Audit",
        "plan_bcrypt_audit"             => "Bcrypt Hash Cracking Audit",
        "plan_crypto_audit"             => "Cryptographic Audit",
        "plan_bmi_assessment"           => "BMI Health Assessment",
        "plan_slope_analysis"           => "Slope Stability Analysis",
        _                               => "Engineering Computation",
    }
}

fn decision_plan_domain(plan: &str) -> &'static str {
    match plan {
        p if p.contains("pump") || p.contains("nfpa") || p.contains("pipe") => "fire_protection",
        p if p.contains("electrical") || p.contains("voltage") || p.contains("power") => "electrical",
        p if p.contains("cvss") || p.contains("password") || p.contains("crypto") || p.contains("bcrypt") => "cybersecurity",
        p if p.contains("bmi") => "health",
        p if p.contains("slope") => "geotechnical",
        _ => "engineering",
    }
}

struct StepAnalysis {
    step:           String,
    value:          f64,
    status:         &'static str,
    title:          String,
    detail:         String,
    recommendation: String,
}

struct DecisionRecommendation {
    priority: u32,
    action:   String,
    detail:   String,
}

fn analyze_step(step: &str, _oracle: &str, value: f64) -> StepAnalysis {
    let step_lower = step.to_lowercase();

    // HP / Pump sizing rules
    if step_lower.contains("hp") || step_lower.contains("pump_hp") || step_lower == "hp_required" {
        if value > 200.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: format!("{:.1} HP exceeds single pump capacity", value),
                detail: format!("{:.1} HP requires parallel pump configuration per NFPA 20.", value),
                recommendation: "Design parallel pump system with 2+ pumps sharing load".into(),
            };
        } else if value > 50.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: format!("{:.1} HP requires VFD starter", value),
                detail: format!("{:.1} HP motor needs variable frequency drive or soft starter per NFPA 20 sec 4.14.", value),
                recommendation: format!("Install VFD rated for {:.0} HP. Select {:.0} HP motor (20% margin).", value, (value * 1.2).ceil()),
            };
        } else if value >= 5.0 {
            let motor = ((value * 1.2) / 5.0).ceil() * 5.0;
            return StepAnalysis {
                step: step.to_string(), value,
                status: "valid",
                title: "Pump sizing adequate".into(),
                detail: format!("{:.1} HP within normal range. Recommend {:.0} HP motor (20% margin).", value, motor),
                recommendation: format!("Select standard {:.0} HP motor with across-the-line starter", motor),
            };
        } else {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "valid",
                title: "Small pump - fractional HP".into(),
                detail: format!("{:.2} HP is a small pump. Standard fractional HP motor.", value),
                recommendation: "Select standard fractional HP motor".into(),
            };
        }
    }

    // Pressure / PSI rules
    if step_lower.contains("psi") || step_lower.contains("pressure") || step_lower.contains("residual") {
        if value < 20.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "Pressure below NFPA minimum".into(),
                detail: format!("{:.1} psi is below 20 psi NFPA minimum. System will not function.", value),
                recommendation: "Add booster pump or redesign system to achieve minimum 20 psi".into(),
            };
        } else if value < 50.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Low pressure - marginal".into(),
                detail: format!("{:.1} psi is within range but low. May not reach remote sprinklers.", value),
                recommendation: "Verify pressure at most remote sprinkler. Consider booster if <50 psi at riser".into(),
            };
        } else {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "valid",
                title: "Pressure adequate".into(),
                detail: format!("{:.1} psi meets NFPA requirements.", value),
                recommendation: "No action needed. Verify annual test per NFPA 25".into(),
            };
        }
    }

    // Shutoff pressure
    if step_lower.contains("shutoff") {
        if value > 140.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Shutoff pressure exceeds 140% rated".into(),
                detail: format!("{:.1} psi shutoff exceeds NFPA 20 limit (max 140% of rated).", value),
                recommendation: "Select pump with flatter curve or add pressure relief valve".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Shutoff pressure within limits".into(),
            detail: format!("{:.1} psi shutoff is within NFPA 20 requirements.", value),
            recommendation: "Verify during acceptance test per NFPA 20 sec 14.2".into(),
        };
    }

    // Flow / GPM rules
    if step_lower.contains("gpm") || step_lower.contains("flow") {
        if value > 2500.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "High flow - loop piping recommended".into(),
                detail: format!("{:.0} GPM exceeds 2500. Single feed may cause excessive friction loss.", value),
                recommendation: "Design loop piping configuration to distribute flow".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Flow rate acceptable".into(),
            detail: format!("{:.0} GPM within normal range.", value),
            recommendation: "Verify flow at hydraulically most remote area".into(),
        };
    }

    // Voltage drop rules
    if step_lower.contains("v_drop") || step_lower.contains("voltage_drop") || step_lower.contains("dv") {
        if value > 5.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "Voltage drop exceeds NEC limit".into(),
                detail: format!("{:.2}V drop exceeds 5V NEC maximum. Equipment may malfunction.", value),
                recommendation: "Increase conductor size or reduce circuit length per NEC 210.19(A)".into(),
            };
        } else if value > 3.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Voltage drop approaching limit".into(),
                detail: format!("{:.2}V drop is above 3V recommended. Monitor during load.", value),
                recommendation: "Consider upsizing conductor one gauge for margin".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Voltage drop acceptable".into(),
            detail: format!("{:.2}V drop within NEC limits.", value),
            recommendation: "No action needed".into(),
        };
    }

    // Cost rules
    if step_lower.contains("cost") {
        if value > 200000.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "Extreme cost - review scope".into(),
                detail: format!("${:.0} exceeds $200k threshold. Verify scope and alternatives.", value),
                recommendation: "Conduct value engineering review. Consider phased implementation".into(),
            };
        } else if value > 50000.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "High cost - alternatives available".into(),
                detail: format!("${:.0} is significant. Review for optimization opportunities.", value),
                recommendation: "Get competitive bids. Consider alternative approaches".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Cost within range".into(),
            detail: format!("${:.0} is within expected range.", value),
            recommendation: "Proceed with standard procurement".into(),
        };
    }

    // Factor of Safety rules
    if step_lower.contains("fos") || step_lower.contains("safety_factor") {
        if value < 1.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "FAILURE - factor of safety below 1.0".into(),
                detail: format!("FoS {:.2} means load exceeds capacity. Immediate redesign required.", value),
                recommendation: "STOP - redesign immediately. Increase member size or reduce load".into(),
            };
        } else if value < 1.5 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Marginal safety factor".into(),
                detail: format!("FoS {:.2} is below recommended 1.5 minimum.", value),
                recommendation: "Increase capacity to achieve FoS >= 1.5".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Safety factor adequate".into(),
            detail: format!("FoS {:.2} meets or exceeds 1.5 requirement.", value),
            recommendation: "Design is adequate. Document in calculations".into(),
        };
    }

    // Power loss rules
    if step_lower.contains("loss") || step_lower.contains("p_loss") {
        if value > 5.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Significant power loss".into(),
                detail: format!("{:.2} units of loss exceeds 5% threshold. Efficiency concern.", value),
                recommendation: "Review conductor sizing and connection resistance".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Power loss acceptable".into(),
            detail: format!("{:.2} units of loss within acceptable range.", value),
            recommendation: "No action needed".into(),
        };
    }

    // Head (ft) rules
    if step_lower.contains("head") && !step_lower.contains("header") {
        if value > 200.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "High head - multi-stage pump may be needed".into(),
                detail: format!("{:.1} ft of head exceeds single-stage pump range.", value),
                recommendation: "Consider multi-stage pump or series booster configuration".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Head within single-stage range".into(),
            detail: format!("{:.1} ft of head is within standard pump range.", value),
            recommendation: "Select pump from manufacturer curves for this duty point".into(),
        };
    }

    // CVSS score rules
    if step_lower.contains("cvss") || step_lower.contains("base_score") {
        if value > 9.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "CRITICAL vulnerability".into(),
                detail: format!("CVSS {:.1} - critical severity. Immediate patching required.", value),
                recommendation: "Patch within 24 hours. Isolate affected systems".into(),
            };
        } else if value > 7.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "HIGH vulnerability".into(),
                detail: format!("CVSS {:.1} - high severity. Prioritize remediation.", value),
                recommendation: "Schedule patching within 7 days. Monitor for exploitation".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Vulnerability score manageable".into(),
            detail: format!("CVSS {:.1} - medium/low severity.", value),
            recommendation: "Include in regular patch cycle".into(),
        };
    }

    // Entropy (password audit)
    if step_lower.contains("entropy") {
        if value < 40.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "Very weak password entropy".into(),
                detail: format!("{:.0} bits entropy - trivially crackable.", value),
                recommendation: "Enforce minimum 12 chars with mixed case + symbols + numbers".into(),
            };
        } else if value < 72.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Weak password entropy".into(),
                detail: format!("{:.0} bits - below NIST SP800-63B recommendation of 72 bits.", value),
                recommendation: "Increase password length or complexity".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Password entropy adequate".into(),
            detail: format!("{:.0} bits entropy meets NIST requirements.", value),
            recommendation: "Password strength is acceptable".into(),
        };
    }

    // Current / Amperage rules
    if step_lower.contains("i_load") || step_lower.contains("current") || step_lower.contains("amp") {
        if value > 100.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "High current - service upgrade may be needed".into(),
                detail: format!("{:.1}A exceeds 100A. Requires NEC 230.79 service protection.", value),
                recommendation: "Verify service entrance capacity. May need panel upgrade".into(),
            };
        } else if value > 30.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Moderate current - verify conductor".into(),
                detail: format!("{:.1}A requires careful conductor sizing per NEC 310.15.", value),
                recommendation: format!("Use minimum #{} AWG copper or verify ampacity tables", if value > 60.0 { "4" } else { "10" }),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Current within normal range".into(),
            detail: format!("{:.1}A is within standard branch circuit capacity.", value),
            recommendation: "Standard wiring acceptable".into(),
        };
    }

    // BMI rules
    if step_lower.contains("bmi") {
        if value > 30.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Obesity - elevated health risk".into(),
                detail: format!("BMI {:.1} indicates obesity (WHO grade I+).", value),
                recommendation: "Medical evaluation recommended. Consider nutrition intervention".into(),
            };
        } else if value > 25.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Overweight".into(),
                detail: format!("BMI {:.1} indicates overweight per WHO guidelines.", value),
                recommendation: "Lifestyle modifications recommended".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "BMI in healthy range".into(),
            detail: format!("BMI {:.1} is within normal range.", value),
            recommendation: "Maintain current health practices".into(),
        };
    }

    // Slope rules
    if step_lower.contains("slope") {
        if value > 30.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "critical",
                title: "LANDSLIDE RISK - steep slope".into(),
                detail: format!("{:.1}% slope exceeds 30% - geotechnical study mandatory per E.050.", value),
                recommendation: "Geotechnical study required. Consider retaining structures".into(),
            };
        } else if value > 15.0 {
            return StepAnalysis {
                step: step.to_string(), value,
                status: "warning",
                title: "Moderate slope - stability analysis needed".into(),
                detail: format!("{:.1}% slope requires stability analysis per SENCICO E.050.", value),
                recommendation: "Perform slope stability analysis before construction".into(),
            };
        }
        return StepAnalysis {
            step: step.to_string(), value,
            status: "valid",
            title: "Gentle slope - stable".into(),
            detail: format!("{:.1}% slope is within stable range.", value),
            recommendation: "Standard construction practices apply".into(),
        };
    }

    // Default: generic valid
    StepAnalysis {
        step: step.to_string(),
        value,
        status: "valid",
        title: format!("{} = {:.4}", step, value),
        detail: format!("Computed value {:.4} for step {}.", value, step),
        recommendation: "Review against project requirements".into(),
    }
}

fn decision_analyze(plan_name: &str, steps: &[(String, String, f64)], total_ns: u64) -> String {
    let label = decision_plan_label(plan_name);
    let domain = decision_plan_domain(plan_name);

    let mut analyses: Vec<StepAnalysis> = Vec::new();
    let mut critical_count = 0u32;
    let mut warning_count = 0u32;

    for (step, oracle, value) in steps {
        let a = analyze_step(step, oracle, *value);
        match a.status {
            "critical" => critical_count += 1,
            "warning"  => warning_count += 1,
            _          => {}
        }
        analyses.push(a);
    }

    let max_possible = (steps.len() as f64) * 3.0;
    let risk_score = if max_possible > 0.0 {
        ((critical_count as f64 * 3.0 + warning_count as f64 * 1.0) / max_possible * 10.0 * 100.0).round() / 100.0
    } else { 0.0 };

    let risk_level = if risk_score >= 7.5 { "CRITICAL" }
        else if risk_score >= 5.0 { "HIGH" }
        else if risk_score >= 2.5 { "MEDIUM" }
        else { "LOW" };

    let overall_status = if critical_count > 0 { "critical" }
        else if warning_count > 0 { "warning" }
        else { "valid" };

    let twin_conf_map = twin_confidence_map();
    let confidence = if twin_conf_map.is_empty() {
        1.0f64
    } else {
        let sum: f64 = twin_conf_map.iter().map(|e| *e.value() as f64).sum();
        let cnt = twin_conf_map.len() as f64;
        ((sum / cnt) * 1000.0).round() / 1000.0
    };

    let twin_vars = twin_state_map().len();
    let threats_active = {
        let store = fact_store_lock();
        store.iter().filter(|f| f.domain == "cyber_ot").count()
    };
    let cves_tracked = NVD_FACTS_INGESTED.load(std::sync::atomic::Ordering::Relaxed);

    let analysis_json: Vec<String> = analyses.iter().map(|a| {
        format!(
            r#"{{"step":"{}","value":{},"status":"{}","title":"{}","detail":"{}","recommendation":"{}"}}"#,
            escape_json(&a.step), a.value, a.status,
            escape_json(&a.title), escape_json(&a.detail), escape_json(&a.recommendation)
        )
    }).collect();

    let mut recommendations: Vec<DecisionRecommendation> = Vec::new();
    let mut priority = 1u32;
    for a in &analyses {
        if a.status == "critical" {
            recommendations.push(DecisionRecommendation {
                priority,
                action: format!("fix_{}", a.step),
                detail: a.recommendation.clone(),
            });
            priority += 1;
        }
    }
    for a in &analyses {
        if a.status == "warning" {
            recommendations.push(DecisionRecommendation {
                priority,
                action: format!("review_{}", a.step),
                detail: a.recommendation.clone(),
            });
            priority += 1;
        }
    }

    let recs_json: Vec<String> = recommendations.iter().map(|r| {
        format!(
            r#"{{"priority":{},"action":"{}","detail":"{}"}}"#,
            r.priority, escape_json(&r.action), escape_json(&r.detail)
        )
    }).collect();

    let steps_json: Vec<String> = steps.iter().map(|(step, oracle, val)| {
        format!(r#"{{"step":"{}","oracle":"{}","result":{:.6}}}"#, step, oracle, if val.is_finite() { *val } else if *val > 0.0 { 1e99_f64 } else { -1e99_f64 })
    }).collect();

    let optimize_suggestion = match plan_name {
        "plan_pump_sizing" => r#"{"available":true,"suggestion":"Sweep Q_gpm 350-650, eff 0.65-0.85 to find minimum HP"}"#,
        "plan_electrical_load" => r#"{"available":true,"suggestion":"Sweep P_w and pf to optimize conductor sizing"}"#,
        "plan_voltage_drop" => r#"{"available":true,"suggestion":"Sweep conductor gauge and length to minimize drop"}"#,
        "plan_pipe_losses" => r#"{"available":true,"suggestion":"Sweep pipe diameter 4-12 inches to find optimal friction loss"}"#,
        "plan_nfpa13_demand" => r#"{"available":true,"suggestion":"Sweep area and K-factor to find critical demand scenario"}"#,
        _ => r#"{"available":false,"suggestion":"No optimization template for this plan"}"#,
    };

    format!(
        r#"{{"ok":true,"plan":"{}","plan_label":"{}","domain":"{}","steps":[{}],"total_ns":{},"decision":{{"status":"{}","risk_score":{},"risk_level":"{}","confidence":{},"analysis":[{}],"recommendations":[{}]}},"context":{{"twin_vars":{},"threats_active":{},"cves_tracked":{},"ot_status":"{}"}},"optimize":{}}}"#,
        plan_name, label, domain,
        steps_json.join(","), total_ns,
        overall_status, risk_score, risk_level, confidence,
        analysis_json.join(","),
        recs_json.join(","),
        twin_vars, threats_active, cves_tracked,
        if threats_active > 0 { "elevated" } else { "nominal" },
        optimize_suggestion
    )
}

fn decision_rules_json() -> String {
    r#"{"ok":true,"engine":"CRYS-L Decision Engine v1.0","rules":[{"category":"hp_pump","thresholds":[{"range":"<5","status":"valid","note":"Small/fractional HP"},{"range":"5-50","status":"valid","note":"Standard, recommend +20% margin"},{"range":"50-200","status":"warning","note":"Needs VFD per NFPA 20 sec 4.14"},{"range":">200","status":"critical","note":"Parallel pumps required"}]},{"category":"pressure_psi","thresholds":[{"range":"<20","status":"critical","note":"Below NFPA minimum"},{"range":"20-50","status":"warning","note":"Low, verify remote sprinklers"},{"range":">50","status":"valid","note":"Adequate"}]},{"category":"shutoff_pressure","thresholds":[{"range":">140% rated","status":"warning","note":"Exceeds NFPA 20 limit"}]},{"category":"flow_gpm","thresholds":[{"range":">2500","status":"warning","note":"Loop piping needed"}]},{"category":"voltage_drop","thresholds":[{"range":">5V","status":"critical","note":"NEC violation"},{"range":">3V","status":"warning","note":"Approaching limit"}]},{"category":"cost","thresholds":[{"range":">200000","status":"critical","note":"Review scope"},{"range":">50000","status":"warning","note":"Significant cost"}]},{"category":"factor_of_safety","thresholds":[{"range":"<1.0","status":"critical","note":"Failure condition"},{"range":"<1.5","status":"warning","note":"Marginal"}]},{"category":"power_loss","thresholds":[{"range":">5%","status":"warning","note":"Efficiency concern"}]},{"category":"head_ft","thresholds":[{"range":">200","status":"warning","note":"Multi-stage pump may be needed"}]},{"category":"cvss_score","thresholds":[{"range":">9.0","status":"critical","note":"Patch within 24h"},{"range":">7.0","status":"warning","note":"Patch within 7 days"}]},{"category":"entropy","thresholds":[{"range":"<40","status":"critical","note":"Trivially crackable"},{"range":"<72","status":"warning","note":"Below NIST recommendation"}]},{"category":"slope","thresholds":[{"range":">30%","status":"critical","note":"Landslide risk"},{"range":">15%","status":"warning","note":"Stability analysis needed"}]}]}"#.to_string()
}



// ── Execution Graph Engine v3 ────────────────────────────────────────
// ── Self-Healing Watchdog Metrics ────────────────────────────────────

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// CRYS_NO_FMA=1 → force VMULSD+VADDSD (SSE2) for cross-arch hash portability
static NO_FMA_MODE: AtomicBool = AtomicBool::new(false);

static REQ_COUNT:     AtomicU64 = AtomicU64::new(0);
static REQ_NS_TOTAL:  AtomicU64 = AtomicU64::new(0);
static ERR_COUNT:     AtomicU64 = AtomicU64::new(0);
static CACHE_HITS:    AtomicU64 = AtomicU64::new(0);

pub fn record_request(elapsed_ns: u64, is_error: bool) {
    REQ_COUNT.fetch_add(1, Ordering::Relaxed);
    REQ_NS_TOTAL.fetch_add(elapsed_ns, Ordering::Relaxed);
    if is_error { ERR_COUNT.fetch_add(1, Ordering::Relaxed); }
}
pub fn record_cache_hit() { CACHE_HITS.fetch_add(1, Ordering::Relaxed); }

pub fn health_metrics() -> (u64, f64, u64, u64, f64) {
    let reqs = REQ_COUNT.load(Ordering::Relaxed);
    let ns   = REQ_NS_TOTAL.load(Ordering::Relaxed);
    let errs = ERR_COUNT.load(Ordering::Relaxed);
    let hits = CACHE_HITS.load(Ordering::Relaxed);
    let avg_ms = if reqs > 0 { (ns / reqs) as f64 / 1_000_000.0 } else { 0.0 };
    let err_pct = if reqs > 0 { errs as f64 / reqs as f64 * 100.0 } else { 0.0 };
    (reqs, avg_ms, errs, hits, err_pct)
}

fn read_mem_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

pub fn watchdog_assess() -> (&'static str, String) {
    let (reqs, avg_ms, errs, hits, err_pct) = health_metrics();
    let rss_mb = read_mem_rss_kb() / 1024;

    let status = if avg_ms > 50.0 || err_pct > 10.0 || rss_mb > 800 {
        "degraded"
    } else if avg_ms > 10.0 || err_pct > 2.0 || rss_mb > 400 {
        "warning"
    } else {
        "healthy"
    };

    let issues: Vec<String> = vec![
        if avg_ms  > 50.0  { Some(format!("high_latency:{:.1}ms", avg_ms)) } else { None },
        if err_pct > 10.0  { Some(format!("error_rate:{:.1}%", err_pct)) } else { None },
        if rss_mb  > 800   { Some(format!("high_memory:{}MB", rss_mb)) } else { None },
    ].into_iter().flatten().collect();

    let json = format!(
        r#"{{"status":"{}","requests":{},"avg_latency_ms":{:.3},"error_rate_pct":{:.2},"cache_hits":{},"mem_rss_mb":{},"issues":[{}]}}"#,
        status, reqs, avg_ms, err_pct, hits, rss_mb,
        issues.iter().map(|i| format!(r#""{}""#, i)).collect::<Vec<_>>().join(",")
    );
    (status, json)
}

fn save_health_log(status: &str, json: &str) {
    let _ = std::fs::create_dir_all("/opt/crysl/memory");
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("/opt/crysl/memory/health_log.ndjson")
    {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs();
        if status != "healthy" {
            let _ = writeln!(f, r#"{{"ts":{},"health":{}}}"#, ts, json);
        }
    }
}

/// Spawn the watchdog background thread.
pub fn start_watchdog() {
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
            let (status, json) = watchdog_assess();
            if status != "healthy" {
                eprintln!("[WATCHDOG] {} — {}", status.to_uppercase(), json);
                save_health_log(status, &json);
            }
        }
    });
}


// ── NormIndex — Engineering Norm Reference Database ───────────────────

struct NormEntry {
    code:        &'static str,  // "NFPA20-4.26.1"
    standard:    &'static str,  // "NFPA 20"
    year:        u16,
    section:     &'static str,  // "§4.26.1"
    title:       &'static str,
    requirement: &'static str,
    threshold:   &'static str,  // "≥ 65 psi"
    action:      &'static str,
}

static NORM_INDEX: &[NormEntry] = &[
    // ── NFPA 20 (Fire Pumps) ─────────────────────────────────────
    NormEntry {
        code: "NFPA20-4.26.1", standard: "NFPA 20", year: 2022,
        section: "§4.26.1", title: "Rated Pump Pressure",
        requirement: "Net positive suction head shall be sufficient at rated flow",
        threshold: "≥ 65 psi residual pressure at system demand",
        action: "Install booster pump or increase supply pressure"
    },
    NormEntry {
        code: "NFPA20-4.14", standard: "NFPA 20", year: 2022,
        section: "§4.14", title: "Motor Sizing",
        requirement: "Driver horsepower shall be sufficient at 150% of rated flow",
        threshold: "> 50 HP requires reduced-voltage starter",
        action: "Use soft-starter or VFD for motors >50HP"
    },
    NormEntry {
        code: "NFPA20-4.14.2", standard: "NFPA 20", year: 2022,
        section: "§4.14.2", title: "Large Motor Protection",
        requirement: "Motors >100HP require additional structural and electrical assessment",
        threshold: "> 100 HP",
        action: "Structural review of pump room + dedicated electrical service"
    },
    NormEntry {
        code: "NFPA20-9.3.1", standard: "NFPA 20", year: 2022,
        section: "§9.3.1", title: "Acceptance Test",
        requirement: "Pump shall deliver 150% of rated flow at ≥65% of rated pressure",
        threshold: "≥ 65% rated pressure at 150% flow",
        action: "Performance test required before system acceptance"
    },
    // ── NFPA 13 (Sprinklers) ─────────────────────────────────────
    NormEntry {
        code: "NFPA13-11.2.3", standard: "NFPA 13", year: 2022,
        section: "§11.2.3", title: "Minimum Residual Pressure",
        requirement: "Minimum operating pressure at the most remote sprinkler",
        threshold: "≥ 7 psi at sprinkler head",
        action: "Increase supply pressure or reduce coverage area per head"
    },
    NormEntry {
        code: "NFPA13-19.3.3", standard: "NFPA 13", year: 2022,
        section: "§19.3.3", title: "Hose Stream Allowance",
        requirement: "Inside hose stream demand to be added to sprinkler demand",
        threshold: "100-500 gpm depending on hazard classification",
        action: "Include hose stream in total demand calculation"
    },
    // ── NEC (National Electrical Code) ───────────────────────────
    NormEntry {
        code: "NEC310.15", standard: "NEC", year: 2023,
        section: "§310.15", title: "Conductor Ampacity",
        requirement: "Conductors shall be sized to carry the load current continuously",
        threshold: "12 AWG: 20A, 10 AWG: 30A, 8 AWG: 50A, 6 AWG: 65A",
        action: "Increase conductor gauge or add parallel conductors"
    },
    NormEntry {
        code: "NEC210.19", standard: "NEC", year: 2023,
        section: "§210.19(A)", title: "Voltage Drop",
        requirement: "Branch circuit conductors sized to avoid excessive voltage drop",
        threshold: "≤ 3% for branch circuit, ≤ 5% total (branch + feeder)",
        action: "Increase conductor cross-section or reduce circuit length"
    },
    NormEntry {
        code: "NEC230.79", standard: "NEC", year: 2023,
        section: "§230.79", title: "Service Disconnecting Means",
        requirement: "Service entrance rated for available fault current",
        threshold: "> 100A requires service disconnect of equivalent rating",
        action: "Install properly rated service entrance equipment"
    },
    // ── CNE Peru (Codigo Nacional de Electricidad) ────────────────
    NormEntry {
        code: "CNE-4.3.1", standard: "CNE Peru", year: 2011,
        section: "§4.3.1", title: "Caída de Tensión Máxima",
        requirement: "La caída de tensión no debe exceder límites establecidos",
        threshold: "≤ 2.5% alimentadores, ≤ 4% circuitos derivados",
        action: "Redimensionar conductor o reducir longitud del circuito"
    },
    // ── AWWA (American Water Works Association) ───────────────────
    NormEntry {
        code: "AWWAM11-4.3", standard: "AWWA M11", year: 2017,
        section: "§4.3", title: "Pipe Flow Velocity",
        requirement: "Flow velocity limits to prevent erosion and water hammer",
        threshold: "≤ 8 fps normal, ≤ 12 fps absolute maximum",
        action: "Increase pipe diameter to reduce velocity below limit"
    },
    // ── NIST / Cybersecurity ─────────────────────────────────────
    NormEntry {
        code: "NIST-800-63B-5.1", standard: "NIST SP800-63B", year: 2020,
        section: "§5.1", title: "Password Strength Requirements",
        requirement: "Memorized secrets shall be at least 8 characters",
        threshold: "≥ 72 bits entropy for high-security systems",
        action: "Use 14+ characters with mixed charset or passphrase"
    },
    NormEntry {
        code: "NIST-CVSSv3", standard: "NIST CVSS v3.1", year: 2019,
        section: "Severity Scale", title: "Vulnerability Severity",
        requirement: "CVSS score determines remediation urgency",
        threshold: "Low:<4, Medium:4-7, High:7-9, Critical:≥9",
        action: "High: patch within 30 days, Critical: patch within 24 hours"
    },
    // ── OMS (World Health Organization) ──────────────────────────
    NormEntry {
        code: "OMS-BMI", standard: "OMS/WHO", year: 2000,
        section: "BMI Classification", title: "Índice de Masa Corporal",
        requirement: "BMI classification for adult nutritional status",
        threshold: "Normal: 18.5-25, Sobrepeso: 25-30, Obesidad: ≥30",
        action: "Sobrepeso: intervención nutricional. Obesidad: valoración médica"
    },
    // ── Peru E.050 (Geotecnia) ────────────────────────────────────
    NormEntry {
        code: "E050-3.2", standard: "RNE E.050", year: 2018,
        section: "§3.2", title: "Estabilidad de Taludes",
        requirement: "Evaluación de estabilidad para taludes naturales y cortes",
        threshold: "Pendiente >15%: análisis requerido. >30%: estudio geotécnico",
        action: "Contratar ingeniero geotécnico certificado (SENCICO)"
    },
    // ── ISO 27001 (Information Security) ─────────────────────────
    NormEntry {
        code: "ISO27001-A.12.6", standard: "ISO 27001", year: 2022,
        section: "§A.12.6.1", title: "Management of Technical Vulnerabilities",
        requirement: "Timely identification and remediation of technical vulnerabilities",
        threshold: "Risk-based patching timeline aligned with CVSS score",
        action: "Implement vulnerability management program per ISO 27001 Annex A"
    },
    // ── IEC 60038 (Standard Voltages) ────────────────────────────
    NormEntry {
        code: "IEC60038-2.2", standard: "IEC 60038", year: 2009,
        section: "§2.2", title: "Standard Voltages",
        requirement: "Voltage tolerance at point of delivery",
        threshold: "±10% of nominal voltage under normal conditions",
        action: "Install voltage regulator or correct upstream impedance"
    },
    // ── ASHRAE (HVAC) ─────────────────────────────────────────────
    NormEntry {
        code: "ASHRAE62.1-6.2", standard: "ASHRAE 62.1", year: 2022,
        section: "§6.2", title: "Ventilation Rate Procedure",
        requirement: "Minimum outdoor air required for acceptable indoor air quality",
        threshold: "15 cfm/person + 0.06 cfm/ft2 floor area",
        action: "Increase outdoor air intake or add air purification"
    },
];

/// Look up a norm entry by code prefix.
fn norm_lookup(code: &str) -> Option<&'static NormEntry> {
    NORM_INDEX.iter().find(|e| e.code == code || code.contains(e.code))
}

/// Enrich a threshold alert message with full norm reference.
fn enrich_alert(norm_code: &str, step: &str, value: f64, threshold: f64, op: &str) -> String {
    let base = format!(
        "{} {:.4} {} {} (threshold: {} {})",
        step, value,
        if op == ">" { "excede" } else { "cae por debajo de" },
        threshold,
        op, threshold
    );
    match norm_lookup(norm_code) {
        Some(e) => format!(
            "[{} {} {}] {} — Req: {} — Acción: {}",
            e.standard, e.year, e.section, base, e.requirement, e.action
        ),
        None => format!("[{}] {}", norm_code, base),
    }
}

/// GET /norms endpoint: list all norm entries
fn norms_list_json() -> String {
    let entries: Vec<String> = NORM_INDEX.iter().map(|e| {
        format!(
            r#"{{"code":"{}","standard":"{}","year":{},"section":"{}","title":"{}","threshold":"{}"}}"#,
            e.code, e.standard, e.year, e.section, e.title, e.threshold
        )
    }).collect();
    format!(r#"{{"count":{},"norms":[{}]}}"#, entries.len(), entries.join(","))
}


// ── Unit Type Validation ──────────────────────────────────────────────

/// Infer expected unit from parameter name using naming conventions.
// -- Dimensional Type System v4 ------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dimension {
    Pressure,       // psi, kPa, bar, atm, mH2O
    Flow,           // gpm, lps, m3/h, m3/s, lpm
    Power,          // hp, kW, W, MW
    Voltage,        // V, kV, mV
    Current,        // A, mA, kA
    ApparentPower,  // kVA, MVA
    ReactivePower,  // kVAR, MVAR
    Length,         // m, ft, km, mm, cm, in
    Area,           // m2, ft2, ha, km2
    Volume,         // m3, ft3, l
    Temperature,    // C, F, K
    Frequency,      // MHz, GHz, kHz, Hz
    Signal,         // dBm, dBW
    DataRate,       // Mbps, Gbps, kbps
    Entropy,        // bits, bytes
    Velocity,       // m/s, ft/s, km/h
    Dimensionless,  // ratio, %, score
    Unknown,
}

fn unit_dimension(unit: &str) -> Dimension {
    match unit {
        "psi" | "kPa" | "bar" | "atm" | "mH2O" => Dimension::Pressure,
        "gpm" | "lps" | "l/s" | "m3/h" | "m3/s" | "lpm" | "gpm/ft2" => Dimension::Flow,
        "hp" | "kW" | "W" | "MW" => Dimension::Power,
        "V" | "kV" | "mV" => Dimension::Voltage,
        "A" | "mA" | "kA" => Dimension::Current,
        "kVA" | "MVA" => Dimension::ApparentPower,
        "kVAR" | "MVAR" => Dimension::ReactivePower,
        "m" | "km" | "ft" | "in" | "mm" | "cm" => Dimension::Length,
        "m2" | "ft2" | "ha" | "km2" => Dimension::Area,
        "m3" | "ft3" | "l" => Dimension::Volume,
        "K" => Dimension::Temperature,
        "MHz" | "GHz" | "kHz" | "Hz" => Dimension::Frequency,
        "dBm" | "dBW" => Dimension::Signal,
        "Mbps" | "Gbps" | "kbps" => Dimension::DataRate,
        "bits" | "bytes" => Dimension::Entropy,
        "m/s" | "ft/s" | "km/h" => Dimension::Velocity,
        "" | "score" | "score/10" => Dimension::Dimensionless,
        _ => Dimension::Unknown,
    }
}

/// Convert value to SI base unit. Returns None for unknown units.
fn to_si(value: f64, unit: &str) -> Option<f64> {
    Some(match unit {
        // Pressure -> Pa
        "psi"  => value * 6894.757,
        "kPa"  => value * 1000.0,
        "bar"  => value * 100_000.0,
        "atm"  => value * 101_325.0,
        "mH2O" => value * 9806.65,
        // Flow -> m3/s
        "gpm"  => value * 6.30902e-5,
        "lps" | "l/s" => value * 0.001,
        "lpm"  => value / 60_000.0,
        "m3/h" => value / 3600.0,
        "m3/s" => value,
        // Power -> W
        "hp"   => value * 745.7,
        "kW"   => value * 1000.0,
        "W"    => value,
        "MW"   => value * 1_000_000.0,
        // Voltage -> V
        "V"    => value,
        "kV"   => value * 1000.0,
        "mV"   => value * 0.001,
        // Current -> A
        "A"    => value,
        "mA"   => value * 0.001,
        "kA"   => value * 1000.0,
        // kVA -> VA
        "kVA"  => value * 1000.0,
        "MVA"  => value * 1_000_000.0,
        "kVAR" => value * 1000.0,
        "MVAR" => value * 1_000_000.0,
        // Length -> m
        "m"    => value,
        "km"   => value * 1000.0,
        "ft"   => value * 0.3048,
        "in"   => value * 0.0254,
        "mm"   => value * 0.001,
        "cm"   => value * 0.01,
        // Area -> m2
        "m2"   => value,
        "ft2"  => value * 0.09290304,
        "ha"   => value * 10_000.0,
        "km2"  => value * 1_000_000.0,
        // Volume -> m3
        "m3"   => value,
        "ft3"  => value * 0.028317,
        "l"    => value * 0.001,
        // Temperature -> K
        "K"    => value,
        // Frequency -> Hz
        "Hz"   => value,
        "kHz"  => value * 1_000.0,
        "MHz"  => value * 1_000_000.0,
        "GHz"  => value * 1_000_000_000.0,
        // DataRate -> bps
        "kbps" => value * 1_000.0,
        "Mbps" => value * 1_000_000.0,
        "Gbps" => value * 1_000_000_000.0,
        // Entropy -> bits
        "bits"  => value,
        "bytes" => value * 8.0,
        // Velocity -> m/s
        "m/s"  => value,
        "ft/s" => value * 0.3048,
        "km/h" => value / 3.6,
        _ => return None,
    })
}

/// Convert from SI base unit to target unit.
fn from_si(si: f64, unit: &str) -> Option<f64> {
    Some(match unit {
        "psi"  => si / 6894.757,
        "kPa"  => si / 1000.0,
        "bar"  => si / 100_000.0,
        "atm"  => si / 101_325.0,
        "mH2O" => si / 9806.65,
        "gpm"  => si / 6.30902e-5,
        "lps" | "l/s" => si / 0.001,
        "lpm"  => si * 60_000.0,
        "m3/h" => si * 3600.0,
        "m3/s" => si,
        "hp"   => si / 745.7,
        "kW"   => si / 1000.0,
        "W"    => si,
        "MW"   => si / 1_000_000.0,
        "V"    => si,
        "kV"   => si / 1000.0,
        "mV"   => si / 0.001,
        "A"    => si,
        "mA"   => si / 0.001,
        "kA"   => si / 1000.0,
        "kVA"  => si / 1000.0,
        "MVA"  => si / 1_000_000.0,
        "kVAR" => si / 1000.0,
        "MVAR" => si / 1_000_000.0,
        "m"    => si,
        "km"   => si / 1000.0,
        "ft"   => si / 0.3048,
        "in"   => si / 0.0254,
        "mm"   => si / 0.001,
        "cm"   => si / 0.01,
        "m2"   => si,
        "ft2"  => si / 0.09290304,
        "ha"   => si / 10_000.0,
        "km2"  => si / 1_000_000.0,
        "m3"   => si,
        "ft3"  => si / 0.028317,
        "l"    => si / 0.001,
        "K"    => si,
        "Hz"   => si,
        "kHz"  => si / 1_000.0,
        "MHz"  => si / 1_000_000.0,
        "GHz"  => si / 1_000_000_000.0,
        "kbps" => si / 1_000.0,
        "Mbps" => si / 1_000_000.0,
        "Gbps" => si / 1_000_000_000.0,
        "bits"  => si,
        "bytes" => si / 8.0,
        "m/s"  => si,
        "ft/s" => si / 0.3048,
        "km/h" => si * 3.6,
        _ => return None,
    })
}

/// Auto-convert value from src_unit to dst_unit.
/// Returns (converted_value, was_converted) or None if dimensions are incompatible.
pub fn convert_unit(value: f64, from: &str, to: &str) -> Option<(f64, bool)> {
    if from == to || from.is_empty() || to.is_empty() { return Some((value, false)); }
    let dim_from = unit_dimension(from);
    let dim_to   = unit_dimension(to);
    // Unknown units: pass through without conversion
    if dim_from == Dimension::Unknown || dim_to == Dimension::Unknown {
        return Some((value, false));
    }
    // Incompatible dimensions: reject
    if dim_from != dim_to { return None; }
    // Same dimension: convert via SI
    let si = to_si(value, from)?;
    let converted = from_si(si, to)?;
    Some((converted, true))
}

/// Validate and auto-convert a wired parameter.
/// Returns Ok((value, conversion_note)) or Err(dimension_mismatch_msg).
pub fn wire_param_safe(
    value: f64,
    src_unit: &str,
    dst_param: &str,
) -> Result<(f64, Option<String>), String> {
    let dst_unit = expected_unit_from_param(dst_param);
    match convert_unit(value, src_unit, dst_unit) {
        Some((v, true)) => Ok((v, Some(format!("{} {} -> {}", value, src_unit, dst_unit)))),
        Some((v, false)) => Ok((v, None)),
        None => Err(format!(
            "dimension mismatch: {} ({:?}) cannot wire to {} ({:?})",
            src_unit, unit_dimension(src_unit), dst_unit, unit_dimension(dst_unit)
        )),
    }
}


fn expected_unit_from_param(param: &str) -> &'static str {
    let p = param.to_lowercase();
    // Flow
    if p.ends_with("_gpm") || p == "q_gpm" || p == "q" || p.contains("flow") { return "gpm"; }
    // Pressure
    if p.ends_with("_psi") || p == "p_psi" || p.contains("pressure") || p.contains("presion") { return "psi"; }
    // Head/loss (ft)
    if p.ends_with("_ft") || p.contains("head") || p.contains("hf") || p.contains("h_loss") { return "ft"; }
    // Power
    if p.ends_with("_hp") || p.contains("pump_hp") { return "hp"; }
    if p.ends_with("_kw") || p.ends_with("_kw)") { return "kW"; }
    if p.ends_with("_w") || p == "p_w" { return "W"; }
    // Electrical
    if p.ends_with("_v") || p == "v" || p.contains("voltage") || p.contains("tension") { return "V"; }
    if p.ends_with("_a") || p == "i" || p.contains("current") || p.contains("corriente") { return "A"; }
    if p.ends_with("_kva") || p.contains("apparent") { return "kVA"; }
    if p.ends_with("_kvar") || p.contains("reactive") { return "kVAR"; }
    // Length/Distance
    if p.ends_with("_m") || p.ends_with("_m2") || p == "l_m" { return "m"; }
    if p.ends_with("_km") { return "km"; }
    // RF/Signal
    if p.ends_with("_dbm") || p.contains("tx_dbm") || p.contains("rx") { return "dBm"; }
    if p.ends_with("_db") || p.contains("path_loss") || p.contains("gain") { return "dB"; }
    if p.ends_with("_mhz") || p.contains("freq") { return "MHz"; }
    if p.ends_with("_mbps") || p.contains("capacity") || p.contains("bandwidth") { return "Mbps"; }
    // Crypto/Security
    if p.ends_with("_bits") || p.contains("entropy") || p.contains("key_bit") { return "bits"; }
    // Score/dimensionless
    if p.contains("pf") || p.contains("eff") || p.contains("factor") { return ""; } // ratio
    // Temperature
    if p.ends_with("_c") && p.len() > 2 { return "°C"; }
    if p.ends_with("_f") && p.len() > 2 { return "°F"; }
    // Area
    if p.ends_with("_ft2") || p.contains("area_ft") { return "ft2"; }
    if p.ends_with("_ha") || p.contains("area_ha") { return "ha"; }
    ""
}

/// Check if two units are physically compatible (same dimension family).
fn units_compatible(src: &str, dst: &str) -> bool {
    if src.is_empty() || dst.is_empty() { return true; } // unknown = accept
    if src == dst { return true; }
    // Compatibility groups (same physical dimension)
    let groups: &[&[&str]] = &[
        &["gpm", "m3/s", "m3/h", "l/s", "gpm/ft2"],      // flow
        &["psi", "kPa", "bar", "ft", "mH2O", "atm"],       // pressure / head
        &["hp", "kW", "W", "MW"],                           // power
        &["V", "kV", "mV"],                                  // voltage
        &["A", "mA", "kA"],                                  // current
        &["kVA", "MVA"],                                     // apparent power
        &["kVAR", "MVAR"],                                   // reactive power
        &["m", "km", "ft", "in", "mm", "cm"],               // length
        &["m2", "ft2", "ha", "km2"],                         // area
        &["m3", "ft3", "l"],                                 // volume
        &["dB", "dBm", "dBW"],                               // RF levels
        &["MHz", "GHz", "kHz"],                              // frequency
        &["bits", "bytes"],                                   // data/entropy
        &["°C", "°F", "K"],                                  // temperature
        &["kg/m2", "lb/ft2"],                                // surface density (BMI etc)
        &["score", "score/10"],                              // dimensionless scores
    ];
    for group in groups {
        let has_src = group.contains(&src);
        let has_dst = group.contains(&dst);
        if has_src && has_dst { return true; }
    }
    false
}

/// Validate wiring in a graph node: check $ref source unit vs param expected unit.
/// Returns list of (param_name, source_ref, src_unit, expected_unit) for mismatches.
fn validate_wiring_units(
    node: &GraphNode,
    scope_units: &std::collections::HashMap<String, &'static str>,
) -> Vec<(String, String, String, String)> {
    let mut mismatches: Vec<(String, String, String, String)> = Vec::new();
    for (param_name, val) in &node.params {
        if let GraphParamVal::Ref(r) = val {
            let ref_key = r.strip_prefix('$').unwrap_or(r.as_str());
            let src_unit = scope_units.get(ref_key).copied().unwrap_or("");
            let dst_unit = expected_unit_from_param(param_name);
            if !src_unit.is_empty() && !dst_unit.is_empty()
                && !units_compatible(src_unit, dst_unit)
            {
                mismatches.push((
                    param_name.clone(),
                    r.clone(),
                    src_unit.to_string(),
                    dst_unit.to_string(),
                ));
            }
        }
    }
    mismatches
}


// ── Graph Optimizer ───────────────────────────────────────────────────

struct OptReport {
    nodes_before:  usize,
    nodes_after:   usize,
    dedup_pairs:   Vec<(String, String)>,  // (original_id, duplicate_id)
    dead_removed:  Vec<String>,
    const_folded:  usize,
}

impl OptReport {
    fn to_json(&self) -> String {
        let dedup_json: Vec<String> = self.dedup_pairs.iter()
            .map(|(a,b)| format!(r#"{{"original":"{}","duplicate":"{}"}}"#, a, b))
            .collect();
        let dead_json: Vec<String> = self.dead_removed.iter()
            .map(|n| format!(r#""{}""#, n)).collect();
        format!(
            r#"{{"nodes_before":{},"nodes_after":{},"dedup_pairs":[{}],"dead_removed":[{}],"const_folded":{}}}"#,
            self.nodes_before, self.nodes_after,
            dedup_json.join(","), dead_json.join(","), self.const_folded
        )
    }
}

/// Optimize the graph before execution:
/// 1. Remove unreachable (dead) nodes — not referenced by any downstream node
/// 2. Mark duplicate nodes (same plan + all-literal params) — second+ will hit cache
/// 3. Count const-foldable nodes (all params are literals)
fn optimize_graph(nodes: &[GraphNode]) -> (Vec<usize>, OptReport) {
    let n = nodes.len();

    // ── Dead node removal ─────────────────────────────────────────
    // A node is "live" if it's the last node OR referenced by another node's $ref or assert
    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in nodes {
        // Collect all $ref targets from params
        for (_, v) in &node.params {
            if let GraphParamVal::Ref(r) = v {
                if let Some(dep) = r.strip_prefix('$').and_then(|s| s.split('.').next()) {
                    referenced.insert(dep.to_string());
                }
            }
        }
        // Collect from if_cond / asserts
        for expr in node.asserts.iter().map(|a| a.expr.as_str())
            .chain(node.if_cond.iter().map(|s| s.as_str()))
        {
            if let Some(dep) = expr.strip_prefix('$').and_then(|s| s.split('.').next()) {
                referenced.insert(dep.to_string());
            }
        }
    }
    // Last node is always live (it's the output)
    let last_id = nodes.last().map(|n| n.id.clone()).unwrap_or_default();
    referenced.insert(last_id);

    let mut dead_removed: Vec<String> = Vec::new();
    let mut live_indices: Vec<usize> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if referenced.contains(&node.id) || i == n - 1 {
            live_indices.push(i);
        } else {
            dead_removed.push(node.id.clone());
        }
    }

    // ── Deduplication detection ────────────────────────────────────
    // Two nodes are duplicates if same plan + all params are identical literals
    let mut dedup_pairs: Vec<(String, String)> = Vec::new();
    let mut seen_sigs: Vec<(String, String)> = Vec::new(); // (plan, params_sig)
    for i in &live_indices {
        let node = &nodes[*i];
        // Build signature from literal params only
        let mut sig_parts: Vec<String> = node.params.iter()
            .filter_map(|(k, v)| {
                if let GraphParamVal::Float(f) = v { Some(format!("{}={:.6}", k, f)) }
                else { None }
            })
            .collect();
        sig_parts.sort();
        let sig = format!("{}|{}", node.plan, sig_parts.join(","));
        if let Some(orig) = seen_sigs.iter().find(|(_, s)| s == &sig).map(|(id,_)| id.clone()) {
            dedup_pairs.push((orig, node.id.clone()));
        } else {
            seen_sigs.push((node.id.clone(), sig));
        }
    }

    // ── Const-foldable count ───────────────────────────────────────
    let const_folded = live_indices.iter().filter(|&&i| {
        nodes[i].params.values().all(|v| matches!(v, GraphParamVal::Float(_)))
    }).count();

    let report = OptReport {
        nodes_before: n,
        nodes_after:  live_indices.len(),
        dedup_pairs,
        dead_removed,
        const_folded,
    };
    (live_indices, report)
}


// ── Graph Engine Core Types & Functions ──────────────────────────────

mod serde {
    pub struct NodeResultJson {
        pub id:     String,
        pub plan:   String,
        pub steps:  Vec<(String, String, f64)>,
        pub cached: bool,
        pub ok:     bool,
    }
}

#[derive(Debug)]
enum GraphParamVal { Float(f64), Ref(String) }

#[derive(Debug)]
struct GraphAssert { expr: String, fail_msg: String }

#[derive(Debug)]
struct GraphNode {
    id:        String,
    plan:      String,
    params:    std::collections::HashMap<String, GraphParamVal>,
    asserts:   Vec<GraphAssert>,
    if_cond:   Option<String>,
    else_plan: Option<String>,
}

/// Evaluate condition: "$node.step op literal"
fn eval_graph_cond(expr: &str, scope: &std::collections::HashMap<String, f64>) -> bool {
    let ops: &[&str] = &["<=", ">=", "!=", "<", ">", "=="];
    for &op in ops {
        if let Some(pos) = expr.find(op) {
            let lhs_str = expr[..pos].trim();
            let rhs_str = expr[pos+op.len()..].trim();
            let resolve = |s: &str| -> f64 {
                if s.starts_with('$') { *scope.get(&s[1..]).unwrap_or(&0.0) }
                else { s.parse::<f64>().unwrap_or(0.0) }
            };
            let lhs = resolve(lhs_str);
            let rhs = resolve(rhs_str);
            return match op {
                "<"  => lhs < rhs,
                "<=" => lhs <= rhs,
                ">"  => lhs > rhs,
                ">=" => lhs >= rhs,
                "==" => (lhs - rhs).abs() < 1e-9,
                "!=" => (lhs - rhs).abs() >= 1e-9,
                _    => false,
            };
        }
    }
    false
}

/// Hash(plan + sorted resolved params) for node-level caching.
fn node_cache_key(plan: &str, params: &std::collections::HashMap<String, f64>) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    plan.hash(&mut h);
    let mut sorted: Vec<_> = params.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted { k.hash(&mut h); v.to_bits().hash(&mut h); }
    h.finish()
}

/// Parse {"graph":[{"id":"A","plan":"p","params":{...},"assert":[...],"if":"...","else":"..."},…]}
fn parse_graph_nodes(body: &str) -> Vec<GraphNode> {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let start = match body.find("\"graph\"") { Some(s) => s, None => return nodes };
    let arr_start = match body[start..].find('[') { Some(i) => start+i+1, None => return nodes };
    let mut pos = arr_start;
    let bytes = body.as_bytes();
    while pos < body.len() {
        while pos < body.len() && matches!(bytes[pos], b' '|b','|b'\n'|b'\r') { pos += 1; }
        if pos >= body.len() || bytes[pos] == b']' { break; }
        if bytes[pos] != b'{' { pos += 1; continue; }
        let obj_start = pos;
        let mut depth = 0usize;
        let mut obj_end = pos;
        for i in pos..body.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => { depth -= 1; if depth == 0 { obj_end = i+1; break; } }
                _ => {}
            }
        }
        if obj_end <= obj_start { break; }
        let obj = &body[obj_start..obj_end];
        pos = obj_end;
        let id   = match extract_json_str(obj, "id")  { Some(v) => v, None => continue };
        let plan = extract_json_str(obj, "then")
            .or_else(|| extract_json_str(obj, "plan")).unwrap_or_default();
        if plan.is_empty() { continue; }
        let params    = parse_graph_params(obj);
        let asserts   = parse_graph_asserts(obj);
        let if_cond   = extract_json_str(obj, "if");
        let else_plan = extract_json_str(obj, "else");
        nodes.push(GraphNode { id, plan, params, asserts, if_cond, else_plan });
    }
    nodes
}

fn parse_graph_params(obj: &str) -> std::collections::HashMap<String, GraphParamVal> {
    let mut map: std::collections::HashMap<String, GraphParamVal> = std::collections::HashMap::new();
    let start = match obj.find("\"params\"") { Some(s) => s, None => return map };
    let inner_start = match obj[start..].find('{') { Some(i) => start+i+1, None => return map };
    let mut depth = 1usize;
    let mut inner_end = inner_start;
    for (i, b) in obj[inner_start..].bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => { depth -= 1; if depth == 0 { inner_end = inner_start+i; break; } }
            _ => {}
        }
    }
    let mut s = &obj[inner_start..inner_end];
    loop {
        s = s.trim_start_matches([',', ' ', '\n', '\r', '\t']);
        if s.is_empty() || !s.starts_with('"') { break; }
        let k_end = match s[1..].find('"') { Some(e) => e+1, None => break };
        let key = s[1..k_end].to_string();
        let after_colon = match s[k_end+1..].find(':') {
            Some(i) => s[k_end+1+i+1..].trim_start_matches(' '),
            None    => break,
        };
        if after_colon.starts_with('"') {
            let inner = &after_colon[1..];
            let v_end = match inner.find('"') { Some(e) => e, None => break };
            map.insert(key, GraphParamVal::Ref(inner[..v_end].to_string()));
            s = &inner[v_end+1..];
        } else {
            let v_end = after_colon.find([',', '}']).unwrap_or(after_colon.len());
            let val: f64 = after_colon[..v_end].trim().parse().unwrap_or(0.0);
            map.insert(key, GraphParamVal::Float(val));
            s = &after_colon[v_end..];
        }
    }
    map
}

fn parse_graph_asserts(obj: &str) -> Vec<GraphAssert> {
    let mut out: Vec<GraphAssert> = Vec::new();
    let start = match obj.find("\"assert\"") { Some(s) => s, None => return out };
    let arr_start = match obj[start..].find('[') { Some(i) => start+i+1, None => return out };
    let arr_end   = match obj[arr_start..].find(']') { Some(i) => arr_start+i, None => return out };
    let arr = &obj[arr_start..arr_end];
    let mut s = arr;
    loop {
        s = s.trim_start_matches([',', ' ', '\n', '\r']);
        if s.is_empty() { break; }
        let expr = extract_json_str(s, "if").or_else(|| extract_json_str(s, "expr")).unwrap_or_default();
        let msg  = extract_json_str(s, "fail").or_else(|| extract_json_str(s, "msg")).unwrap_or_default();
        if !expr.is_empty() { out.push(GraphAssert { expr, fail_msg: msg }); }
        if let Some(i) = s.find('}') { s = &s[i+1..]; } else { break; }
    }
    out
}

/// Kahn's topological sort.
fn graph_topo_sort(
    nodes: &[GraphNode],
    deps:  &std::collections::HashMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    use std::collections::{HashMap, VecDeque};
    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let mut in_deg: HashMap<String, usize> = ids.iter().map(|id| (id.clone(), 0)).collect();
    for id in &ids {
        if let Some(d) = deps.get(id) { *in_deg.entry(id.clone()).or_insert(0) += d.len(); }
    }
    let mut queue: VecDeque<String> = in_deg.iter()
        .filter(|(_, &d)| d == 0).map(|(id, _)| id.clone()).collect();
    let mut order: Vec<String> = Vec::new();
    while let Some(curr) = queue.pop_front() {
        order.push(curr.clone());
        for id in &ids {
            if let Some(d) = deps.get(id) {
                if d.contains(&curr) {
                    let deg = in_deg.entry(id.clone()).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 { queue.push_back(id.clone()); }
                }
            }
        }
    }
    if order.len() != ids.len() { Err("cycle detected in graph".to_string()) }
    else { Ok(order) }
}


// ── Unit System — Oracle output unit registry ─────────────────────────
fn oracle_unit(name: &str) -> &'static str {
    match name {
        // Fire protection
        "nfpa20_pump_hp"          => "hp",
        "nfpa13_demand_gpm"       => "gpm",
        "nfpa13_sprinkler"        => "gpm",
        "nfpa13_demand_flow"      => "gpm",
        "nfpa13_area_density"     => "gpm/ft2",
        "nfpa13_sprinkler_count"  => "heads",
        "nfpa20_shutoff_pressure" => "psi",
        "nfpa20_150pct_flow"      => "gpm",
        "nfpa20_head_pressure"    => "psi",
        "nfpa72_detector_count"   => "units",
        "nfpa101_egress_capacity" => "persons/min",
        "nfpa101_exit_width"      => "in",
        // Hydraulics
        "pipe_friction_loss"      => "psi",
        "residual_pressure"       => "psi",
        "pump_deficit_hp"         => "hp",
        "hazen_williams_hf"       => "ft",
        "hazen_williams_velocity" => "m/s",
        "pipe_velocity"           => "fps",
        "pipe_area"               => "m2",
        "manning_flow"            => "m3/s",
        "hazen_P_at_gpm"          => "psi",
        "hazen_critical_q_gpm"    => "gpm",
        "pipe_network_loss"       => "psi",
        // Electrical
        "voltage_drop"            => "V",
        "voltage_drop_3ph"        => "V",
        "load_current_1ph"        => "A",
        "load_current_3ph"        => "A",
        "apparent_power"          => "kVA",
        "reactive_power"          => "kVAR",
        "power_loss"              => "W",
        "transformer_rating"      => "kVA",
        "capacitor_kvar"          => "kVAR",
        "short_circuit_current"   => "A",
        // Civil / Structural
        "beam_deflection"         => "mm",
        "beam_moment"             => "kN·m",
        "beam_shear"              => "kN",
        "column_buckling"         => "kN",
        "concrete_capacity"       => "kN",
        "slab_moment"             => "kN·m/m",
        "rebar_area"              => "cm2/m",
        // HVAC
        "hvac_cooling_load"       => "kW",
        "hvac_ventilation_flow"   => "m3/h",
        // Solar
        "solar_panel_count"       => "panels",
        "solar_battery_kwh"       => "kWh",
        "solar_peak_hours"        => "h",
        // Finance / Comercio
        "igv"                     => "S/",
        "precio_con_igv"          => "S/",
        "base_imponible"          => "S/",
        "roi"                     => "%",
        "break_even_units"        => "units",
        "markup_price"            => "S/",
        "net_present_value"       => "S/",
        // Health
        "bmi"                     => "kg/m2",
        "drug_dose_mg"            => "mg",
        // Mechanics
        "motor_power"             => "kW",
        "motor_torque"            => "Nm",
        // Geo
        "slope_pct"               => "%",
        "earthwork_volume"        => "m3",
        "transport_cost"          => "S/",
        // Stats
        "mean"                    => "",
        "std_dev"                 => "",
        "sample_size_n"           => "n",
        "regression_slope"        => "",
        // Agro
        "irrigation_volume"       => "m3/day",
        "crop_yield"              => "t/ha",
        // Telecom
        "path_loss_fspl"          => "dB",
        "link_budget"             => "dB",
        "shannon_capacity"        => "Mbps",
        "cell_coverage_km2"       => "km2",
        // Cybersecurity
        "password_entropy"        => "bits",
        "brute_force_years"       => "years",
        "bcrypt_hashrate"         => "h/s",
        "bcrypt_crack_seconds"    => "seconds",
        "dict_crack_seconds"      => "seconds",
        "keyspace"                => "combinations",
        "secs_to_years"           => "years",
        "secs_to_hours"           => "hours",
        "cvss_base_score"         => "score",
        "aes_key_strength"        => "score/10",
        "rsa_key_strength"        => "score/10",
        "network_risk_score"      => "score",
        "ssl_risk"                => "score",
        "scan_time_min"           => "min",
        "patch_priority_score"    => "score",
        _                         => "",
    }
}

// ── DomainCorrelationMap ──────────────────────────────────────────────
// plan_name → [(correlated_plan, [("param", "formula")])]
// Formula: literal float, OR "step_name*factor", OR "step_name"
struct ProactiveAlert {
    source_plan:  String,
    corr_plan:    String,
    result_steps: Vec<(String, f64, &'static str)>, // (step, val, unit)
    warnings:     Vec<String>,
}

static PROACTIVE_CORRELATIONS: &[(&str, &[(&str, &[(&str, &str)])])] = &[
    // When hydraulic demand is computed → also estimate pump electrical load
    ("plan_nfpa13_demand", &[
        ("plan_pump_sizing", &[
            ("Q_gpm",  "flow_gpm"),
            ("P_psi",  "65.0"),
            ("eff",    "0.7"),
        ]),
    ]),
    // When pump is sized → estimate electrical load on that pump motor
    ("plan_pump_sizing", &[
        ("plan_electrical_load", &[
            ("P_w",   "pump_hp*746.0"),
            ("V",     "440.0"),
            ("pf",    "0.85"),
            ("L",     "30.0"),
            ("A",     "10.0"),
        ]),
    ]),
    // Voltage drop → check power factor correction need
    ("plan_voltage_drop", &[
        ("plan_power_factor_correction", &[
            ("P_kw",       "P_loss_w*0.001"),
            ("pf_actual",  "0.75"),
            ("pf_meta",    "0.95"),
        ]),
    ]),
    // Pipe losses → check if booster needed
    ("plan_pipe_losses", &[
        ("plan_pump_selection", &[
            ("P_residual",  "80.0"),
            ("P_required",  "65.0"),
            ("Q_gpm",       "500.0"),
        ]),
    ]),
    // Password audit → cross-check crypto strength
    ("plan_password_audit", &[
        ("plan_crypto_audit", &[
            ("aes_bits",     "256.0"),
            ("rsa_bits",     "2048.0"),
            ("charset_size", "94.0"),
            ("pwd_length",   "12.0"),
        ]),
    ]),
];

// ── Alert Thresholds ──────────────────────────────────────────────────
// (plan, step, op, threshold, norm_ref, recommendation)
static ALERT_THRESHOLDS: &[(&str, &str, &str, f64, &str, &str)] = &[
    ("plan_electrical_load",   "I_load",    ">",  30.0, "NEC 310.15",          "Corriente >30A: verificar calibre conductor AWG"),
    ("plan_electrical_load",   "I_load",    ">", 100.0, "NEC 230.79",          "Corriente >100A: requiere protección de servicio"),
    ("plan_pump_sizing",       "pump_hp",   ">",  50.0, "NFPA 20 §4.14",       "Bomba >50HP: requiere arrancador suave o variador"),
    ("plan_pump_sizing",       "pump_hp",   ">", 100.0, "NFPA 20 §4.14.2",     "Bomba >100HP: revisión estructural del cuarto de bombas"),
    ("plan_pressure",          "residual",  "<",  65.0, "NFPA 20 §4.26.1",     "Presión residual baja: agregar bomba booster"),
    ("plan_pipe_losses",       "velocity",  ">",   8.0, "AWWA M11 §4.3",       "Velocidad >8fps: erosión acelerada. Aumentar diámetro"),
    ("plan_pipe_losses",       "velocity",  ">",  12.0, "AWWA M11 §4.3",       "Velocidad >12fps: CRÍTICO — falla inminente de tubería"),
    ("plan_voltage_drop",      "V_drop",    ">",   5.0, "NEC 210.19(A)",       "Caída de tensión >5%: redimensionar conductor"),
    ("plan_voltage_drop",      "V_drop",    ">",  10.0, "CNE Peru §4.3.1",     "Caída >10%: INCUMPLE norma peruana. Rediseñar circuito"),
    ("plan_cvss_assessment",   "base_score",">",   7.0, "NIST CVSS v3",        "CVSS >7: vulnerabilidad ALTA — parche inmediato"),
    ("plan_cvss_assessment",   "base_score",">",   9.0, "NIST CVSS v3",        "CVSS >9: vulnerabilidad CRÍTICA — acción en <24h"),
    ("plan_password_audit",    "entropy",   "<",  72.0, "NIST SP800-63B §5.1", "Entropía <72 bits: contraseña débil. Mínimo 12 chars+símbolos"),
    ("plan_bmi_assessment",    "bmi",       ">",  25.0, "OMS BMI",             "IMC >25: sobrepeso. Evaluar intervención nutricional"),
    ("plan_bmi_assessment",    "bmi",       ">",  30.0, "OMS BMI",             "IMC >30: obesidad grado I — riesgo cardiovascular elevado"),
    ("plan_slope_analysis",    "slope_pct", ">",  15.0, "E.050 SENCICO §3.2",  "Pendiente >15%: análisis de estabilidad requerido"),
    ("plan_slope_analysis",    "slope_pct", ">",  30.0, "E.050 SENCICO §3.2",  "Pendiente >30%: RIESGO DESLIZAMIENTO — estudio geotécnico"),
];

fn check_thresholds(plan: &str, steps: &[(String, String, f64)]) -> Vec<String> {
    let mut alerts: Vec<String> = Vec::new();
    for &(tplan, tstep, op, threshold, norm, msg) in ALERT_THRESHOLDS {
        if tplan != plan { continue; }
        for (sname, _, sval) in steps {
            if sname.as_str() != tstep { continue; }
            let triggered = match op {
                ">"  => *sval > threshold,
                ">=" => *sval >= threshold,
                "<"  => *sval < threshold,
                "<=" => *sval <= threshold,
                _    => false,
            };
            if triggered {
                alerts.push(format!(
                    "[{}] {} (valor={:.3}{}) — {}",
                    norm, tstep, sval, oracle_unit(tstep), msg
                ));
            }
        }
    }
    alerts
}

fn resolve_proactive_param(
    formula: &str,
    scope: &std::collections::HashMap<String, f64>,
    node_prefix: &str,
) -> f64 {
    // "step_name" → scope["prefix.step_name"]
    // "step_name*factor" → scope["prefix.step_name"] * factor
    // "literal"   → parse as f64
    if let Some(mul_pos) = formula.find('*') {
        let step = formula[..mul_pos].trim();
        let factor: f64 = formula[mul_pos+1..].trim().parse().unwrap_or(1.0);
        let key = format!("{}.{}", node_prefix, step);
        let val = scope.get(&key).copied().unwrap_or(0.0);
        val * factor
    } else if formula.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
        // step reference
        let key = format!("{}.{}", node_prefix, formula);
        scope.get(&key).copied().unwrap_or(0.0)
    } else {
        formula.parse::<f64>().unwrap_or(0.0)
    }
}

// ── Graph Memory ──────────────────────────────────────────────────────
static GRAPH_MEMORY_PATH: &str = "/opt/crysl/memory/graph_log.ndjson";

// -- Digital Twin State Engine v4 ----------------------------------------

static TWIN_STATE: std::sync::OnceLock<DashMap<String, (f64, String)>> =
    std::sync::OnceLock::new();

static TWIN_HISTORY: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, String)>>> =
    std::sync::OnceLock::new();

fn twin_state_map() -> &'static DashMap<String, (f64, String)> {
    TWIN_STATE.get_or_init(|| DashMap::new())
}

fn twin_history_lock() -> std::sync::MutexGuard<'static, Vec<(u64, String)>> {
    TWIN_HISTORY.get_or_init(|| std::sync::Mutex::new(Vec::new()))
        .lock().unwrap_or_else(|p| p.into_inner())
}

fn twin_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs()
}

/// Update one or more twin state variables.
/// Input: {"variables": {"pressure_node_3": {"value": 58.0, "unit": "psi"}, ...}}
fn twin_update(body: &str) {
    let now = twin_ts();
    let ts_lock = twin_timestamp_map();
    let conf_lock = twin_confidence_map();
    let state = twin_state_map();
    // Parse "variables" object
    let var_start = match body.find("\"variables\"") {
        Some(i) => i, None => return,
    };
    let obj_start = match body[var_start..].find('{') {
        Some(i) => var_start + i + 1, None => return,
    };
    let mut depth = 1usize;
    let mut obj_end = obj_start;
    for (i, b) in body[obj_start..].bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => { depth -= 1; if depth == 0 { obj_end = obj_start + i; break; } }
            _ => {}
        }
    }
    let vars_str = &body[obj_start..obj_end];
    // Parse each variable entry: "name": {"value": N, "unit": "U"}
    let mut s = vars_str;
    loop {
        s = s.trim_start_matches([',', ' ', '\n', '\r', '\t']);
        if s.is_empty() || !s.starts_with('"') { break; }
        let k_end = match s[1..].find('"') { Some(e) => e + 1, None => break };
        let var_name = s[1..k_end].to_string();
        let rest = s[k_end+1..].trim_start_matches([' ', ':']);
        // Find the inner object {value: N, unit: "U"}
        let inner_start = match rest.find('{') { Some(i) => i + 1, None => break };
        let mut d2 = 1usize;
        let mut inner_end = inner_start;
        for (i, b) in rest[inner_start..].bytes().enumerate() {
            match b { b'{' => d2 += 1, b'}' => { d2 -= 1; if d2 == 0 { inner_end = inner_start + i; break; } } _ => {} }
        }
        let inner = &rest[inner_start..inner_end];
        let value = inner.find("\"value\"")
            .and_then(|p| inner[p+7..].find(':').map(|q| p+7+q+1))
            .and_then(|p| inner[p..].trim_start_matches(' ').split([',', '}']).next())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        let unit = inner.find("\"unit\"")
            .and_then(|p| inner[p+6..].find('"').map(|q| p+6+q+1))
            .and_then(|p| inner[p..].find('"').map(|q| inner[p..p+q].to_string()))
            .unwrap_or_default();
        let conf = inner.find("\"confidence\"")
            .and_then(|p| inner[p+12..].find(':').map(|q| p+12+q+1))
            .and_then(|p| inner[p..].trim_start_matches(' ').split([',','}']).next())
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        ts_lock.insert(var_name.clone(), now);
        conf_lock.insert(var_name.clone(), conf);
        iot_push_event(&var_name, value, &unit, "twin_update");
        state.insert(var_name, (value, unit));
        // Advance past this entry
        s = &rest[inner_end+1..];
    }
    // Broadcast twin state snapshot to WebSocket subscribers
    let ts = twin_ts();
    let ws_msg = format!(
        r#"{{"type":"twin_update","ts":{}}}"#,
        ts
    );
    ws_broadcast(&ws_msg);
}

fn twin_state_json() -> String {
    let state = twin_state_map();
    let ts = twin_ts();
    let conf_map = twin_confidence_map();
    let ts_map   = twin_timestamp_map();
    let now2     = twin_ts();
    let vars: Vec<String> = state.iter().map(|entry| {
        let (k, (v, u)) = (entry.key(), entry.value());
        let conf = conf_map.get(k).map(|r| *r.value()).unwrap_or(1.0);
        let last = ts_map.get(k).map(|r| *r.value()).unwrap_or(now2);
        let age  = now2.saturating_sub(last);
        format!(r#""{}":{{"value":{:.6},"unit":"{}","confidence":{:.2},"age_secs":{}}}"#,
            k, v, u, conf, age)
    }).collect();
    let history = twin_history_lock();
    format!(
        r#"{{"ok":true,"ts":{},"variable_count":{},"snapshot_count":{},"variables":{{{}}}}}"#,
        ts, state.len(), history.len(), vars.join(",")
    )
}

fn twin_snapshot() {
    let state = twin_state_map();
    let ts = twin_ts();
    let vars: Vec<String> = state.iter().map(|entry| {
        let (k, (v, u)) = (entry.key(), entry.value());
        format!(r#""{}":{{"value":{:.4},"unit":"{}"}}"#, k, v, u)
    }).collect();
    let snap = format!(r#"{{"ts":{},"variables":{{{}}}}}"#, ts, vars.join(","));
    let mut history = twin_history_lock();
    history.push((ts, snap));
    if history.len() > 1000 { history.remove(0); }
}


// -- TwinSignal Confidence Engine v5 ------------------------------------

static TWIN_CONFIDENCE: std::sync::OnceLock<DashMap<String, f32>> =
    std::sync::OnceLock::new();
static TWIN_TIMESTAMP_MAP: std::sync::OnceLock<DashMap<String, u64>> =
    std::sync::OnceLock::new();

fn twin_confidence_map() -> &'static DashMap<String, f32> {
    TWIN_CONFIDENCE.get_or_init(|| DashMap::new())
}

fn twin_timestamp_map() -> &'static DashMap<String, u64> {
    TWIN_TIMESTAMP_MAP.get_or_init(|| DashMap::new())
}

/// Signal health classification
fn signal_health(confidence: f32, age_secs: u64) -> &'static str {
    if confidence < 0.3 { return "unreliable"; }
    if confidence < 0.6 { return "degraded"; }
    if age_secs > 300   { return "stale"; }
    "healthy"
}

/// Analyze twin state: flag stale/unreliable signals, run thresholds
fn twin_analyze_json() -> String {
    let state    = twin_state_map();
    let conf_map = twin_confidence_map();
    let ts_map   = twin_timestamp_map();
    let now      = twin_ts();

    let mut issues: Vec<String> = Vec::new();
    let mut reliable_vars: Vec<String> = Vec::new();

    for entry in state.iter() {
        let (var, (val, unit)) = (entry.key(), entry.value());
        let conf     = conf_map.get(var).map(|r| *r.value()).unwrap_or(1.0);
        let last_ts  = ts_map.get(var).map(|r| *r.value()).unwrap_or(now);
        let age_secs = now.saturating_sub(last_ts);
        let health   = signal_health(conf, age_secs);

        let entry = format!(
            r#"{{"var":"{}","value":{:.4},"unit":"{}","confidence":{:.2},"age_secs":{},"health":"{}"}}"#,
            var, val, unit, conf, age_secs, health
        );

        if health != "healthy" {
            issues.push(format!(
                r#"{{"type":"signal_{}","var":"{}","confidence":{:.2},"age_secs":{}}}"#,
                health, var, conf, age_secs
            ));
        }
        reliable_vars.push(entry);
    }

    // Run threshold scan on reliable variables mapped to oracle names

    let overall = if issues.iter().any(|i| i.contains("unreliable")) {
        "critical"
    } else if !issues.is_empty() {
        "warning"
    } else {
        "healthy"
    };

    format!(
        r#"{{"ok":true,"twin_health":"{}","variable_count":{},"issues":{},"issues_count":{},"signals":[{}]}}"#,
        overall,
        reliable_vars.len(),
        if issues.is_empty() { "[]".to_string() } else { format!("[{}]", issues.join(",")) },
        issues.len(),
        reliable_vars.join(",")
    )
}


// -- Auto-Fix Engine v5 -------------------------------------------------

/// Map: (violated_plan, step_name, comparison) -> remediation_plan
/// When a constraint fires and a matching rule exists, auto-execute the fix plan.
// -- AutoFix Policy Engine v5.1 -----------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum FixMode { Suggest, Confirm, Auto }

struct AutoFixPolicy {
    trigger_plan:   &'static str,
    step_name:      &'static str,
    fix_plan:       &'static str,
    mode:           FixMode,
    min_confidence: f32,   // minimum avg twin confidence to allow auto-fix
    description:    &'static str,
}

static FIX_POLICIES: &[AutoFixPolicy] = &[
    AutoFixPolicy {
        trigger_plan: "plan_pressure", step_name: "residual",
        fix_plan: "plan_pump_selection", mode: FixMode::Auto,
        min_confidence: 0.70,
        description: "Residual pressure below NFPA 20 minimum — auto-select booster pump",
    },
    AutoFixPolicy {
        trigger_plan: "plan_pump_sizing", step_name: "hp_required",
        fix_plan: "plan_motor_sizing", mode: FixMode::Confirm,
        min_confidence: 0.80,
        description: "Pump HP exceeds threshold — human confirmation required for motor change",
    },
    AutoFixPolicy {
        trigger_plan: "plan_voltage_drop", step_name: "V_drop",
        fix_plan: "plan_cable_resize", mode: FixMode::Suggest,
        min_confidence: 0.0,
        description: "Voltage drop exceeded — suggest cable resize (no auto action)",
    },
    AutoFixPolicy {
        trigger_plan: "plan_nfpa13_demand", step_name: "flow_gpm",
        fix_plan: "plan_pump_sizing", mode: FixMode::Auto,
        min_confidence: 0.75,
        description: "Demand flow high — auto-size pump",
    },
    AutoFixPolicy {
        trigger_plan: "plan_pipe_losses", step_name: "friction",
        fix_plan: "plan_pipe_network_3", mode: FixMode::Suggest,
        min_confidence: 0.0,
        description: "Friction loss high — suggest network re-analysis",
    },
];

fn avg_twin_confidence() -> f32 {
    let conf_map = twin_confidence_map();
    if conf_map.is_empty() { return 1.0; } // no twin state = full confidence
    let sum: f32 = conf_map.iter().map(|e| *e.value()).sum();
    sum / conf_map.len() as f32
}

/// Returns (policy, fix_plan, mode) or None
fn find_policy(active_plan: &str, step_tuples: &[(String, String, f64)])
    -> Option<(&'static AutoFixPolicy, &'static str)>
{
    for policy in FIX_POLICIES {
        if active_plan != policy.trigger_plan { continue; }
        if step_tuples.iter().any(|(s, _, _)| s.as_str() == policy.step_name) {
            return Some((policy, policy.fix_plan));
        }
    }
    None
}

fn apply_autofix_policy(
    policy: &AutoFixPolicy,
    fix_plan: &'static str,
    resolved: &std::collections::HashMap<String, f64>,
    executor: &crate::plan::PlanExecutor,
    twin_conf: f32,
) -> String {
    let mode_str = match policy.mode {
        FixMode::Suggest  => "suggest",
        FixMode::Confirm  => "confirm_required",
        FixMode::Auto     => "auto",
    };

    // Confidence gate: block Auto mode if twin confidence too low
    let blocked_by_confidence = policy.mode == FixMode::Auto
        && twin_conf < policy.min_confidence;

    if blocked_by_confidence {
        return format!(
            r#","auto_fix":{{"plan":"{}","mode":"blocked","reason":"twin_confidence_too_low","twin_confidence":{:.2},"required":{:.2},"description":"{}"}}"#,
            fix_plan, twin_conf, policy.min_confidence, policy.description
        );
    }

    match policy.mode {
        FixMode::Suggest => {
            format!(
                r#","auto_fix":{{"plan":"{}","mode":"suggest","description":"{}","note":"Call POST /graph/execute with this plan to apply"}}"#,
                fix_plan, policy.description
            )
        }
        FixMode::Confirm => {
            // Return a one-time confirmation token (timestamp-based)
            let token = format!("{:x}", twin_ts() ^ fix_plan.len() as u64 * 31337);
            format!(
                r#","auto_fix":{{"plan":"{}","mode":"confirm_required","token":"{}","description":"{}","note":"Re-submit with confirm_token to execute"}}"#,
                fix_plan, token, policy.description
            )
        }
        FixMode::Auto => {
            match executor.execute(fix_plan, resolved.clone()) {
                Ok(fix_result) => {
                    let fix_steps: Vec<String> = fix_result.steps.iter().map(|s| {
                        format!(r#"{{"step":"{}","value":{:.4},"unit":"{}"}}"#,
                            s.step, s.value, oracle_unit(&s.oracle))
                    }).collect();
                    format!(
                        r#","auto_fix":{{"plan":"{}","mode":"auto","applied":true,"twin_confidence":{:.2},"description":"{}","steps":[{}]}}"#,
                        fix_plan, twin_conf, policy.description, fix_steps.join(",")
                    )
                }
                Err(e) => format!(
                    r#","auto_fix":{{"plan":"{}","mode":"auto","applied":false,"error":"{}"}}"#,
                    fix_plan, e
                ),
            }
        }
    }
}



// -- Memory Engine v2 ---------------------------------------------------

static MEMORY_SCORES: std::sync::OnceLock<DashMap<u64, (f32, u32)>> =
    std::sync::OnceLock::new();

fn memory_scores_map() -> &'static DashMap<u64, (f32, u32)> {
    MEMORY_SCORES.get_or_init(|| DashMap::new())
}

fn memory_hash(summary: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    summary.hash(&mut h);
    h.finish()
}

fn record_memory_success(summary: &str, score: f32) {
    let hash = memory_hash(summary);
    let scores = memory_scores_map();
    let mut entry = scores.entry(hash).or_insert((0.0, 0));
    // Exponential moving average for score
    entry.0 = entry.0 * 0.8 + score * 0.2;
    entry.1 += 1;
}

fn load_graph_memory_ranked(limit: usize) -> Vec<String> {
    let raw = match std::fs::read_to_string(GRAPH_MEMORY_PATH) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let learn_store = learning_store_map();
    let mut entries: Vec<(f32, u32, String)> = raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let hash = memory_hash(line);
            let (cs, ec) = if let Some(r) = learn_store.get(&hash) { let &(sr, cr, ec, _) = r.value(); if true {
                (composite_score(sr, cr, ec), ec)
            } else { unreachable!() } } else { (0.5, 0) };
            (cs, ec, line.to_string())
        })
        .collect();
    // Sort by composite score desc
    entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    entries.into_iter().take(limit)
        .map(|(score, count, entry)| {
            format!(r#"{{"composite_score":{:.3},"executions":{},"entry":{}}}"#, score, count, entry)
        })
        .collect()
}


// -- Proactive Correction Engine v5 ------------------------------------

/// Weighted correlation with optional trigger condition
struct CorrelationV2 {
    from_plan:  &'static str,
    to_plan:    &'static str,
    weight:     f32,
    trigger:    Option<&'static str>,  // e.g. "$node.step > threshold"
    params:     &'static [(&'static str, &'static str)],
}

static CORRELATIONS_V2: &[CorrelationV2] = &[
    CorrelationV2 {
        from_plan: "plan_nfpa13_demand",
        to_plan:   "plan_pump_sizing",
        weight:    0.9,
        trigger:   Some("$node.flow_gpm > 400"),
        params:    &[("Q_gpm","$node.flow_gpm"),("P_psi","100"),("eff","0.75")],
    },
    CorrelationV2 {
        from_plan: "plan_pump_sizing",
        to_plan:   "plan_voltage_drop",
        weight:    0.8,
        trigger:   Some("$node.hp_required > 30"),
        params:    &[("I","$node.hp_required * 1.5"),("L_m","50"),("A_mm2","16")],
    },
    CorrelationV2 {
        from_plan: "plan_pressure",
        to_plan:   "plan_pump_selection",
        weight:    1.0,
        trigger:   Some("$node.residual < 65"),
        params:    &[("P_residual","$node.residual"),("P_required","65"),("Q_gpm","500")],
    },
    CorrelationV2 {
        from_plan: "plan_voltage_drop",
        to_plan:   "plan_pump_sizing",
        weight:    0.6,
        trigger:   None,  // always correlate
        params:    &[("Q_gpm","200"),("P_psi","80"),("eff","0.70")],
    },
];

fn eval_v2_trigger(trigger: &str, scope: &std::collections::HashMap<String, f64>, node_id: &str) -> bool {
    // Replace $node.step with actual scope key
    let resolved_expr = trigger.replace("$node.", &format!("${}.", node_id));
    eval_graph_cond(&resolved_expr, scope)
}

fn resolve_v2_param(formula: &str, scope: &std::collections::HashMap<String, f64>, node_id: &str) -> f64 {
    // If formula is a $ref, resolve from scope
    if formula.starts_with('$') {
        let key = formula[1..].replace("node.", &format!("{}.", node_id));
        if let Some(&val) = scope.get(&key) { return val; }
    }
    // If formula contains *, do simple a * b computation
    if let Some(pos) = formula.find('*') {
        let lhs_str = formula[..pos].trim();
        let rhs_str = formula[pos+1..].trim();
        let lhs = if lhs_str.starts_with('$') {
            let key = lhs_str[1..].replace("node.", &format!("{}", node_id));
            scope.get(&key).copied().unwrap_or(0.0)
        } else { lhs_str.parse().unwrap_or(0.0) };
        let rhs: f64 = rhs_str.parse().unwrap_or(0.0);
        return lhs * rhs;
    }
    formula.parse().unwrap_or(0.0)
}

/// Run proactive v2 correlations for a given node result.
fn run_proactive_v2(
    nr_id: &str,
    nr_plan: &str,
    scope: &std::collections::HashMap<String, f64>,
    executor: &crate::plan::PlanExecutor,
) -> Vec<String> {
    let mut alerts = Vec::new();
    for corr in CORRELATIONS_V2 {
        if corr.from_plan != nr_plan { continue; }
        // Check trigger condition
        let triggered = corr.trigger
            .map(|t| eval_v2_trigger(t, scope, nr_id))
            .unwrap_or(true);
        if !triggered { continue; }

        let mut corr_params: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for &(param, formula) in corr.params {
            corr_params.insert(param.to_string(), resolve_v2_param(formula, scope, nr_id));
        }
        match executor.execute(corr.to_plan, corr_params) {
            Ok(result) => {
                let step_strs: Vec<String> = result.steps.iter().map(|s| {
                    format!(r#"{{"step":"{}","value":{:.4},"unit":"{}"}}"#,
                        s.step, s.value, oracle_unit(&s.oracle))
                }).collect();
                alerts.push(format!(
                    r#"{{"triggered_by":"{}","corr_plan":"{}","weight":{:.1},"results":[{}]}}"#,
                    nr_id, corr.to_plan, corr.weight, step_strs.join(",")
                ));
            }
            Err(_) => {}
        }
    }
    alerts
}


// -- Adaptive Learning Engine v5.2 --------------------------------------

/// Scoring components stored per graph_hash
/// (success_rate, correctness, execution_count, error_count)
static LEARNING_STORE: std::sync::OnceLock<
    DashMap<u64, (f32, f32, u32, u32)>
> = std::sync::OnceLock::new();

fn learning_store_map() -> &'static DashMap<u64, (f32, f32, u32, u32)> {
    LEARNING_STORE.get_or_init(|| DashMap::new())
}

/// Composite score: success_rate * correctness * reuse_factor
/// reuse_factor = ln(reuse_count + e) / e   [normalizes to ~1.0 at e uses, grows slowly]
fn composite_score(success_rate: f32, correctness: f32, execution_count: u32) -> f32 {
    let reuse_factor = ((execution_count as f32 + std::f32::consts::E).ln()
        / std::f32::consts::E).min(2.0);
    (success_rate * correctness * reuse_factor).min(1.0)
}

/// Record execution outcome for adaptive learning.
/// outcome: "success" | "partial" | "failure"
/// violation_count: how many constraints were violated
pub fn record_execution_outcome(hash: u64, outcome: &str, violation_count: usize) {
    let store = learning_store_map();
    let mut entry = store.entry(hash).or_insert((1.0, 1.0, 0, 0));
    entry.2 += 1; // execution_count

    let success = match outcome {
        "success" => 1.0f32,
        "partial" => 0.5,
        "failure" => 0.0,
        _         => 0.5,
    };
    // Update correctness based on violation_count (fewer violations = more correct)
    let correctness = if violation_count == 0 { 1.0 }
                      else { (1.0 / (violation_count as f32 + 1.0)).max(0.1) };

    // Exponential moving average (alpha = 0.3 for faster adaptation)
    entry.0 = entry.0 * 0.7 + success * 0.3;
    entry.1 = entry.1 * 0.7 + correctness * 0.3;

    if outcome == "failure" { entry.3 += 1; }
}

/// External feedback: POST /graph/feedback
/// {"graph_hash": "HEX", "outcome": "success|failure|partial", "correction": 0.8}
fn apply_external_feedback(body: &str) -> String {
    let hash_hex = match extract_json_str(body, "graph_hash") {
        Some(h) => h, None => return r#"{"ok":false,"error":"missing graph_hash"}"#.into(),
    };
    let hash = u64::from_str_radix(hash_hex.trim_start_matches("0x"), 16)
        .or_else(|_| hash_hex.parse::<u64>())
        .unwrap_or(0);
    if hash == 0 {
        return r#"{"ok":false,"error":"invalid graph_hash"}"#.into();
    }
    let outcome = extract_json_str(body, "outcome").unwrap_or_else(|| "partial".into());
    let correction = extract_json_float(body, "correction").unwrap_or(1.0) as f32;
    let violations = extract_json_float(body, "violation_count")
        .map(|v| v as usize).unwrap_or(0);

    record_execution_outcome(hash, &outcome, violations);

    // Apply correction factor directly
    {
        let store = learning_store_map();
        if let Some(mut entry) = store.get_mut(&hash) {
            entry.1 = (entry.1 * correction).clamp(0.0, 1.0);
        }
    }

    let store = learning_store_map();
    let (score_str, exec_str) = if let Some(r) = store.get(&hash) { let &(sr, cr, ec, _) = r.value(); if true {
        let cs = composite_score(sr, cr, ec);
        (format!("{:.3}", cs), format!("{}", ec))
    } else { unreachable!() } } else { ("0.0".into(), "0".into()) };

    format!(
        r#"{{"ok":true,"hash":"{}","outcome":"{}","composite_score":{},"executions":{}}}"#,
        hash_hex, outcome, score_str, exec_str
    )
}

fn get_learning_stats(hash: u64) -> Option<String> {
    let store = learning_store_map();
    store.get(&hash).map(|r| { let &(sr, cr, ec, errc) = r.value();
        let cs = composite_score(sr, cr, ec);
        format!(
            r#"{{"success_rate":{:.3},"correctness":{:.3},"composite_score":{:.3},"executions":{},"errors":{}}}"#,
            sr, cr, cs, ec, errc
        )
    })
}


// -- Multi-Agent Consensus Engine v5.3 ----------------------------------

struct AgentDef {
    domain:   &'static str,
    plans:    &'static [&'static str],
    weight:   f32,
    verdict_fn: fn(steps: &[(String, String, f64)]) -> (&'static str, f32, String),
}

fn hydraulic_verdict(steps: &[(String, String, f64)]) -> (&'static str, f32, String) {
    // Thresholds: query DKP first, fallback to NFPA 20 defaults
    let min_residual = dkp_threshold("hydraulic", "min_residual_pressure_psi", 65.0);
    let max_flow     = dkp_threshold("hydraulic", "max_flow_gpm", 1000.0);
    let max_friction = dkp_threshold("hydraulic", "max_friction_loss_psi", 50.0);

    let mut issues: Vec<String> = Vec::new();
    let mut confidence = 1.0f32;
    for (step, _, val) in steps {
        if step.contains("residual") && *val < min_residual {
            issues.push(format!("residual_pressure={:.1}psi<{:.0}psi(NFPA20)", val, min_residual));
            confidence -= 0.3;
        }
        if step.contains("flow_gpm") && *val > max_flow {
            issues.push(format!("flow={:.0}gpm>{:.0}gpm(check_pump)", val, max_flow));
            confidence -= 0.1;
        }
        if step.contains("friction") && *val > max_friction {
            issues.push(format!("friction_loss={:.1}>{:.0}psi(high)", val, max_friction));
            confidence -= 0.15;
        }
    }
    let verdict = if confidence < 0.4 { "critical" }
                  else if confidence < 0.7 { "warning" }
                  else { "ok" };
    let key = if issues.is_empty() {
        "All hydraulic parameters within NFPA limits".into()
    } else { issues.join("; ") };
    (verdict, confidence.max(0.1), key)
}

fn electrical_verdict(steps: &[(String, String, f64)]) -> (&'static str, f32, String) {
    // Thresholds: DKP-first, fallback to NEC/NFPA 20 defaults
    let max_v_drop    = dkp_threshold("electrical", "max_voltage_drop_v", 5.0);
    let hp_starter    = dkp_threshold("electrical", "hp_starter_threshold", 50.0);
    let hp_special    = dkp_threshold("electrical", "hp_special_protection", 100.0);

    let mut issues: Vec<String> = Vec::new();
    let mut confidence = 1.0f32;
    for (step, _, val) in steps {
        if step.contains("V_drop") && *val > max_v_drop {
            issues.push(format!("V_drop={:.2}V>{:.0}V(NEC_limit)", val, max_v_drop));
            confidence -= 0.25;
        }
        if step.contains("hp_required") && *val > hp_special {
            issues.push(format!("hp={:.1}>{:.0}HP(NFPA20_special_protection)", val, hp_special));
            confidence -= 0.3;
        } else if step.contains("hp_required") && *val > hp_starter {
            issues.push(format!("hp={:.1}>{:.0}HP(requires_starter)", val, hp_starter));
            confidence -= 0.2;
        }
    }
    let verdict = if confidence < 0.5 { "critical" }
                  else if confidence < 0.75 { "warning" }
                  else { "ok" };
    let key = if issues.is_empty() {
        "Electrical parameters within NEC limits".into()
    } else { issues.join("; ") };
    (verdict, confidence.max(0.1), key)
}

fn cost_verdict(steps: &[(String, String, f64)]) -> (&'static str, f32, String) {
    // Simplified cost model: hp * $800/hp for pump, V_drop affects cable cost
    let mut total_cost_usd = 0.0f64;
    let mut cost_items: Vec<String> = Vec::new();
    for (step, _, val) in steps {
        if step.contains("hp_required") {
            let cost = val * 800.0;
            total_cost_usd += cost;
            cost_items.push(format!("pump=${:.0}", cost));
        }
        if step.contains("flow_gpm") {
            let cost = val * 2.5; // pipe material estimate
            total_cost_usd += cost;
            cost_items.push(format!("piping=${:.0}", cost));
        }
    }
    let max_budget = dkp_threshold("cost", "max_project_budget_usd", 50_000.0);
    let verdict = if total_cost_usd > max_budget { "warning" }
                  else { "ok" };
    let key = if cost_items.is_empty() {
        "Cost: insufficient data".into()
    } else {
        format!("total_est=${:.0} ({})", total_cost_usd, cost_items.join(", "))
    };
    (verdict, 0.75, key) // cost confidence always slightly lower (estimates)
}

static AGENTS: &[AgentDef] = &[
    AgentDef {
        domain: "hydraulic",
        plans:  &["plan_pressure","plan_pump_sizing","plan_pipe_losses","plan_nfpa13_demand",
                  "plan_pipe_network_3","plan_pump_selection"],
        weight: 0.40,
        verdict_fn: hydraulic_verdict,
    },
    AgentDef {
        domain: "electrical",
        plans:  &["plan_voltage_drop","plan_pump_sizing"],
        weight: 0.35,
        verdict_fn: electrical_verdict,
    },
    AgentDef {
        domain: "cost",
        plans:  &["plan_pump_sizing","plan_nfpa13_demand","plan_pipe_losses"],
        weight: 0.25,
        verdict_fn: cost_verdict,
    },
];

// -- Context Engine v5.4 ------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum SystemType {
    Hospital,    // Life safety: hydraulic > electrical > cost
    Industrial,  // Operations: electrical > hydraulic > cost
    Commercial,  // Balance: hydraulic = electrical > cost
    Residential, // Cost matters: hydraulic > cost > electrical
    Critical,    // Maximum safety: hydraulic = electrical, cost irrelevant
    General,     // Default balanced weights
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RiskLevel { Critical, High, Medium, Low }

struct Context {
    system_type: SystemType,
    risk_level:  RiskLevel,
}

fn parse_context(body: &str) -> Context {
    let sys = match extract_json_str(body, "system_type").as_deref() {
        Some("hospital")    => SystemType::Hospital,
        Some("industrial")  => SystemType::Industrial,
        Some("commercial")  => SystemType::Commercial,
        Some("residential") => SystemType::Residential,
        Some("critical")    => SystemType::Critical,
        _                   => SystemType::General,
    };
    let risk = match extract_json_str(body, "risk_level").as_deref() {
        Some("critical") => RiskLevel::Critical,
        Some("high")     => RiskLevel::High,
        Some("low")      => RiskLevel::Low,
        _                => RiskLevel::Medium,
    };
    Context { system_type: sys, risk_level: risk }
}

/// Compute dynamic agent weights based on context.
/// Returns (hydraulic_w, electrical_w, cost_w)
fn compute_weights(ctx: &Context) -> (f32, f32, f32) {
    let base = match ctx.system_type {
        SystemType::Hospital    => (0.50, 0.40, 0.10),
        SystemType::Industrial  => (0.35, 0.45, 0.20),
        SystemType::Commercial  => (0.40, 0.38, 0.22),
        SystemType::Residential => (0.40, 0.25, 0.35),
        SystemType::Critical    => (0.45, 0.45, 0.10),
        SystemType::General     => (0.40, 0.35, 0.25),
    };
    // Risk level multiplier: higher risk → amplify safety-critical weights
    let risk_factor = match ctx.risk_level {
        RiskLevel::Critical => 1.20,
        RiskLevel::High     => 1.10,
        RiskLevel::Medium   => 1.00,
        RiskLevel::Low      => 0.90,
    };
    // Re-normalize after applying risk factor to hydraulic + electrical
    let h = base.0 * risk_factor;
    let e = base.1 * risk_factor;
    let c = base.2 * (1.0 / risk_factor); // cost less important at higher risk
    let total = h + e + c;
    (h/total, e/total, c/total)
}

/// Generate human-readable explanation for an agent's verdict.
fn explain_verdict(domain: &str, verdict: &str, key_finding: &str, confidence: f32) -> String {
    match verdict {
        "ok" => match domain {
            "hydraulic"  => format!("Hydraulic: all parameters within NFPA tolerance ({:.0}% confidence)", confidence*100.0),
            "electrical" => format!("Electrical: NEC compliance verified ({:.0}% confidence)", confidence*100.0),
            "cost"       => format!("Cost: within acceptable budget range ({:.0}% confidence)", confidence*100.0),
            _            => format!("{}: compliant ({:.0}%)", domain, confidence*100.0),
        },
        "warning" => format!("{} WARNING: {} — manual review suggested", domain, key_finding),
        "critical" => format!("{} CRITICAL: {} — immediate action required", domain, key_finding),
        _ => format!("{}: {}", domain, key_finding),
    }
}


fn run_consensus_inner(
    body: &str,
    executor: &crate::plan::PlanExecutor,
) -> String {
    let nodes = parse_graph_nodes(body);
    if nodes.is_empty() {
        return r#"{"ok":false,"error":"no nodes in graph"}"#.into();
    }

    // Parse context for dynamic weighting
    let ctx = parse_context(body);
    let (w_hyd, w_elec, w_cost) = compute_weights(&ctx);
    let dynamic_weights = [w_hyd, w_elec, w_cost];

    let ctx_type_str = match ctx.system_type {
        SystemType::Hospital    => "hospital",
        SystemType::Industrial  => "industrial",
        SystemType::Commercial  => "commercial",
        SystemType::Residential => "residential",
        SystemType::Critical    => "critical",
        SystemType::General     => "general",
    };
    let risk_str = match ctx.risk_level {
        RiskLevel::Critical => "critical",
        RiskLevel::High     => "high",
        RiskLevel::Medium   => "medium",
        RiskLevel::Low      => "low",
    };

    let mut debate: Vec<String> = Vec::new();
    let mut consensus_score = 0.0f32;
    let mut total_weight = 0.0f32;
    let mut all_findings: Vec<String> = Vec::new();
    let mut conf_vector: Vec<String> = Vec::new();
    let mut why_list: Vec<String> = Vec::new();
    let mut has_veto = false;      // any critical finding vetoes proceed
    let mut min_conf = 1.0f32;

    for (i, agent) in AGENTS.iter().enumerate() {
        let effective_weight = if i < 3 { dynamic_weights[i] } else { agent.weight };

        let mut agent_steps: Vec<(String, String, f64)> = Vec::new();
        for node in &nodes {
            if !agent.plans.contains(&node.plan.as_str()) { continue; }
            let mut resolved: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            for (k, v) in &node.params {
                if let GraphParamVal::Float(f) = v { resolved.insert(k.clone(), *f); }
            }
            if let Ok(res) = executor.execute(&node.plan, resolved) {
                for s in &res.steps {
                    agent_steps.push((s.step.clone(), s.oracle.clone(), s.value));
                }
            }
        }

        let (verdict, confidence, key_finding) = (agent.verdict_fn)(&agent_steps);

        // VETO: any critical finding overrides consensus
        if verdict == "critical" { has_veto = true; }
        if confidence < min_conf { min_conf = confidence; }

        let vote = match verdict { "ok" => 1.0, "warning" => 0.5, _ => 0.0 };
        consensus_score += vote * effective_weight * confidence;
        total_weight    += effective_weight;

        if verdict != "ok" {
            all_findings.push(format!("[{}] {}", agent.domain, key_finding));
        }

        conf_vector.push(format!(r#""{}":{:.2}"#, agent.domain, confidence));
        why_list.push(format!(r#""{}""#,
            escape_json(&explain_verdict(agent.domain, verdict, &key_finding, confidence))));

        debate.push(format!(
            r#"{{"agent":"{}","domain":"{}","verdict":"{}","confidence":{:.2},"weight":{:.2},"key_finding":"{}"}}"#,
            agent.domain, agent.domain, verdict, confidence, effective_weight,
            escape_json(&key_finding)
        ));
    }

    let normalized = if total_weight > 0.0 { consensus_score / total_weight } else { 0.5 };

    // Veto gate: critical finding → halt regardless of score
    let consensus_verdict = if has_veto || min_conf < 0.3 {
        "halt_review_required"
    } else if normalized > 0.75 && min_conf >= 0.6 {
        "proceed"
    } else if normalized > 0.5 {
        "proceed_with_caution"
    } else {
        "halt_review_required"
    };

    let recommendations: Vec<String> = all_findings.iter()
        .map(|f| format!(r#""{}""#, escape_json(f))).collect();

    format!(
        r#"{{"ok":true,"consensus":"{}","confidence":{:.3},"min_confidence":{:.2},"confidence_vector":{{{}}},
"context":{{"system_type":"{}","risk_level":"{}","weights":{{"hydraulic":{:.2},"electrical":{:.2},"cost":{:.2}}}}},
"agents":{},"veto_triggered":{},"debate":[{}],"why":[{}],"recommendations":[{}]}}"#,
        consensus_verdict, normalized, min_conf,
        conf_vector.join(","),
        ctx_type_str, risk_str, w_hyd, w_elec, w_cost,
        debate.len(), has_veto,
        debate.join(","), why_list.join(","), recommendations.join(",")
    )
}


// -- DKP Local Knowledge Store v6.0 -------------------------------------
// Distributed Knowledge Protocol: local layer only (P2P deferred)
// Facts are typed, timestamped, confidence-scored, and domain-tagged.

#[derive(Debug, Clone)]
/// Authority levels for DKP norm conflict resolution:
///   3 = LocalRegulation  (e.g. MINEM Peru, SUNAT) — always wins
///   2 = International    (e.g. ISO, NFPA, WHO, FAO-56)
///   1 = Advisory         (e.g. OSHA advisory, FAO guidance, best practice)
///   0 = Unspecified      (default, treated as Advisory)
struct KnowledgeFact {
    id:              u64,     // hash(domain+key+value_str)
    domain:          String,  // e.g. "hydraulic", "electrical", "nfpa20"
    key:             String,  // e.g. "min_residual_pressure_psi"
    value_f:         f64,     // numeric value (0.0 if non-numeric)
    value_s:         String,  // string representation
    unit:            String,  // e.g. "psi", "hp", "V"
    confidence:      f32,     // 0.0–1.0
    source:          String,  // "agent:hydraulic", "user:feedback", "oracle:plan_pressure"
    created_ts:      u64,     // unix seconds
    ttl_secs:        u64,     // 0 = permanent; set for realtime data
    authority_level: u8,      // 0-3: higher authority wins on key conflicts
    mode:            String,  // "snapshot" | "realtime" — realtime gets short TTL
}

fn authority_label(level: u8) -> &'static str {
    match level {
        3 => "LocalRegulation",
        2 => "International",
        1 => "Advisory",
        _ => "Unspecified",
    }
}

static FACT_STORE: std::sync::OnceLock<
    std::sync::Mutex<Vec<KnowledgeFact>>
> = std::sync::OnceLock::new();

fn fact_store_lock()
    -> std::sync::MutexGuard<'static, Vec<KnowledgeFact>>
{
    FACT_STORE.get_or_init(|| std::sync::Mutex::new(Vec::new()))
        .lock().unwrap_or_else(|p| p.into_inner())
}

fn fact_id(domain: &str, key: &str, value_s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in domain.bytes().chain(b"|".iter().copied())
        .chain(key.bytes()).chain(b"|".iter().copied())
        .chain(value_s.bytes())
    {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// POST /dkp/ingest
/// {"domain":"hydraulic","key":"min_residual_psi","value":65.0,"unit":"psi",
///  "confidence":0.95,"source":"nfpa20","ttl_secs":0}
fn dkp_ingest(body: &str) -> String {
    let domain = match extract_json_str(body, "domain") {
        Some(d) => d, None => return r#"{"ok":false,"error":"missing domain"}"#.into(),
    };
    let key = match extract_json_str(body, "key") {
        Some(k) => k, None => return r#"{"ok":false,"error":"missing key"}"#.into(),
    };
    let value_f = extract_json_float(body, "value_f")
        .or_else(|| extract_json_float(body, "value"))
        .unwrap_or(0.0);
    let value_s = extract_json_str(body, "value_s")
        .or_else(|| extract_json_str(body, "value_str"))
        .unwrap_or_else(|| format!("{:.6}", value_f));
    let unit       = extract_json_str(body, "unit").unwrap_or_default();
    let confidence = extract_json_float(body, "confidence").unwrap_or(1.0) as f32;
    let source     = extract_json_str(body, "source").unwrap_or_else(|| "unknown".into());
    let ttl_secs   = extract_json_float(body, "ttl_secs").map(|v| v as u64).unwrap_or(0);

    let authority_level = extract_json_float(body, "authority_level")
        .map(|v| v as u8).unwrap_or(0);
    let mode = extract_json_str(body, "mode").unwrap_or_else(|| "snapshot".into());

    // Auto-set TTL for realtime data if not specified
    let ttl_secs = if ttl_secs == 0 && mode == "realtime" {
        3600 // realtime facts expire after 1 hour by default
    } else { ttl_secs };

    let id = fact_id(&domain, &key, &value_s);
    let ts = twin_ts();

    let fact = KnowledgeFact {
        id, domain: domain.clone(), key: key.clone(),
        value_f, value_s: value_s.clone(), unit: unit.clone(),
        confidence, source: source.clone(),
        created_ts: ts, ttl_secs,
        authority_level, mode: mode.clone(),
    };

    {
        let mut store = fact_store_lock();
        // Authority-aware upsert:
        // A higher-authority fact is NEVER overwritten by a lower-authority one.
        // Same authority: always update (newest wins).
        let existing_authority = store.iter()
            .find(|f| f.domain == domain && f.key == key)
            .map(|f| f.authority_level)
            .unwrap_or(0);

        if authority_level >= existing_authority {
            store.retain(|f| !(f.domain == domain && f.key == key));
            store.push(fact);
        } else {
            // Lower-authority fact rejected — log but don't store
            eprintln!("[DKP] Rejected ingest: {}:{} authority={} ({}) < existing={} ({}). Higher-authority norm wins.",
                domain, key,
                authority_level, authority_label(authority_level),
                existing_authority, authority_label(existing_authority));
            return format!(
                r#"{{"ok":false,"reason":"authority_conflict","existing_authority":{},"existing_label":"{}","submitted_authority":{},"submitted_label":"{}","hint":"Increase authority_level to override local regulation"}}"#,
                existing_authority, authority_label(existing_authority),
                authority_level, authority_label(authority_level)
            );
        }
    }
    save_dkp_store();

    format!(
        r#"{{"ok":true,"id":"{:x}","domain":"{}","key":"{}","value_s":"{}","unit":"{}","confidence":{:.2},"source":"{}","authority_level":{},"authority_label":"{}","mode":"{}","ttl_secs":{}}}"#,
        id, escape_json(&domain), escape_json(&key),
        escape_json(&value_s), escape_json(&unit),
        confidence, escape_json(&source),
        authority_level, authority_label(authority_level),
        escape_json(&mode), ttl_secs
    )
}

/// POST /dkp/query
/// {"domain":"hydraulic","key":"min_residual_psi"}          → exact match
/// {"domain":"hydraulic"}                                    → all in domain
/// {"key":"residual"}                                        → key contains
/// {"min_confidence":0.8}                                    → confidence filter
/// {"limit":10}                                              → max results
fn dkp_query(body: &str) -> String {
    let filter_domain = extract_json_str(body, "domain");
    let filter_key    = extract_json_str(body, "key");
    let min_conf      = extract_json_float(body, "min_confidence").map(|v| v as f32).unwrap_or(0.0);
    let limit         = extract_json_float(body, "limit").map(|v| v as usize).unwrap_or(50);

    let now = twin_ts();
    let store = fact_store_lock();

    let mut results: Vec<String> = store.iter()
        .filter(|f| {
            // TTL check
            if f.ttl_secs > 0 && now.saturating_sub(f.created_ts) > f.ttl_secs { return false; }
            // domain filter (contains, case-insensitive)
            if let Some(ref d) = filter_domain {
                if !f.domain.to_lowercase().contains(&d.to_lowercase()) { return false; }
            }
            // key filter (contains)
            if let Some(ref k) = filter_key {
                if !f.key.to_lowercase().contains(&k.to_lowercase()) { return false; }
            }
            // confidence floor
            if f.confidence < min_conf { return false; }
            true
        })
        .take(limit)
        .map(|f| format!(
            r#"{{"id":"{:x}","domain":"{}","key":"{}","value":{},"value_s":"{}","unit":"{}","confidence":{:.2},"source":"{}","authority_level":{},"authority_label":"{}","mode":"{}","age_secs":{}}}"#,
            f.id, escape_json(&f.domain), escape_json(&f.key),
            f.value_f, escape_json(&f.value_s), escape_json(&f.unit),
            f.confidence, escape_json(&f.source),
            f.authority_level, authority_label(f.authority_level),
            escape_json(&f.mode),
            now.saturating_sub(f.created_ts)
        ))
        .collect();

    format!(
        r#"{{"ok":true,"count":{},"facts":[{}]}}"#,
        results.len(), results.join(",")
    )
}

/// GET /dkp/stats
fn dkp_stats() -> String {
    let now = twin_ts();
    let store = fact_store_lock();
    let total = store.len();
    let expired = store.iter().filter(|f| {
        f.ttl_secs > 0 && now.saturating_sub(f.created_ts) > f.ttl_secs
    }).count();
    let domains: std::collections::HashSet<&str> = store.iter().map(|f| f.domain.as_str()).collect();
    let avg_conf: f32 = if total == 0 { 0.0 } else {
        store.iter().map(|f| f.confidence).sum::<f32>() / total as f32
    };
    let domain_list: Vec<String> = domains.iter().map(|d| format!(r#""{}""#, d)).collect();
    format!(
        r#"{{"ok":true,"total_facts":{},"expired":{},"active":{},"domains":[{}],"avg_confidence":{:.3}}}"#,
        total, expired, total - expired, domain_list.join(","), avg_conf
    )
}

/// DELETE /dkp/purge — remove expired TTL facts
fn dkp_purge() -> String {
    let now = twin_ts();
    let mut store = fact_store_lock();
    let before = store.len();
    store.retain(|f| f.ttl_secs == 0 || now.saturating_sub(f.created_ts) <= f.ttl_secs);
    let removed = before - store.len();
    format!(r#"{{"ok":true,"removed":{},"remaining":{}}}"#, removed, store.len())
}



// ── POST /dkp/publish — push high-confidence facts to mesh peer ─────────────
fn dkp_publish(body: &str) -> String {
    let min_conf: f32 = extract_json_str(body, "min_confidence")
        .and_then(|s| s.parse().ok()).unwrap_or(0.70);
    let target = extract_json_str(body, "target")
        .unwrap_or_else(|| "http://10.99.0.3:8090".into());
    let domain_filter = extract_json_str(body, "domain").unwrap_or_default();

    let facts_json: Vec<String> = {
        let store = fact_store_lock();
        store.iter()
            .filter(|f| {
                f.confidence >= min_conf
                    && (domain_filter.is_empty() || f.domain == domain_filter)
                    && f.ttl_secs == 0
            })
            .map(|f| format!(
                concat!(r#"{{"domain":"{}","key":"{}","value":{},"#,
                        r#""value_s":"{}","unit":"{}","confidence":{:.3},"#,
                        r#""source":"mesh:server5","authority":{}}}"#),
                f.domain, f.key, f.value_f, f.value_s, f.unit,
                f.confidence, f.authority_level
            ))
            .collect()
    };

    let n = facts_json.len();
    if n == 0 {
        return format!(
            r#"{{"ok":true,"published":0,"target":"{}","note":"no facts meet threshold {:.2}"}}"#,
            target, min_conf
        );
    }

    let mut ok_n = 0usize;
    let mut err_n = 0usize;
    for fact_body in &facts_json {
        let url = format!("{}/dkp/ingest", target.trim_end_matches('/'));
        match ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(3))
            .send_string(fact_body)
        {
            Ok(r) if r.status() < 300 => ok_n += 1,
            _ => err_n += 1,
        }
    }
    format!(
        r#"{{"ok":true,"candidates":{},"published":{},"errors":{},"target":"{}","min_conf":{:.2}}}"#,
        n, ok_n, err_n, target, min_conf
    )
}

const DKP_STORE_PATH: &str = "/opt/crysl/data/dkp_store.ndjson";

/// Lookup a numeric threshold from DKP, fallback to default.
fn dkp_threshold(domain: &str, key: &str, default: f64) -> f64 {
    let store = fact_store_lock();
    store.iter()
        .find(|f| f.domain == domain && f.key == key)
        .map(|f| f.value_f)
        .unwrap_or(default)
}

fn save_dkp_store() {
    let store = fact_store_lock();
    let lines: Vec<String> = store.iter().map(|f| {
        format!(
            r#"{{"id":"{:x}","domain":"{}","key":"{}","value_f":{},"value_s":"{}","unit":"{}","confidence":{:.4},"source":"{}","created_ts":{},"ttl_secs":{},"authority_level":{},"mode":"{}"}}"#,
            f.id, f.domain, f.key, f.value_f, f.value_s, f.unit,
            f.confidence, f.source, f.created_ts, f.ttl_secs,
            f.authority_level, f.mode
        )
    }).collect();
    let _ = std::fs::create_dir_all("/opt/crysl/data");
    let _ = std::fs::write(DKP_STORE_PATH, lines.join("\n"));
}
fn load_dkp_store() {
    let raw = match std::fs::read_to_string(DKP_STORE_PATH) {
        Ok(s) => s, Err(_) => return,
    };
    let mut store = fact_store_lock();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let id_hex  = extract_json_str(line, "id").unwrap_or_default();
        let id      = u64::from_str_radix(&id_hex, 16).unwrap_or(0);
        let domain  = extract_json_str(line, "domain").unwrap_or_default();
        let key     = extract_json_str(line, "key").unwrap_or_default();
        let value_f = extract_json_float(line, "value_f").unwrap_or(0.0);
        let value_s = extract_json_str(line, "value_s").unwrap_or_else(|| format!("{}", value_f));
        let unit    = extract_json_str(line, "unit").unwrap_or_default();
        let confidence = extract_json_float(line, "confidence").unwrap_or(1.0) as f32;
        let source  = extract_json_str(line, "source").unwrap_or_else(|| "persisted".into());
        let created_ts = extract_json_float(line, "created_ts").map(|v| v as u64).unwrap_or(0);
        let ttl_secs   = extract_json_float(line, "ttl_secs").map(|v| v as u64).unwrap_or(0);
        if id == 0 || domain.is_empty() || key.is_empty() { continue; }
        store.retain(|f| f.id != id);
        let authority_level = extract_json_float(line, "authority_level")
            .map(|v| v as u8).unwrap_or(0);
        let mode = extract_json_str(line, "mode").unwrap_or_else(|| "snapshot".into());
        store.push(KnowledgeFact {
            id, domain, key, value_f, value_s, unit,
            confidence, source, created_ts, ttl_secs,
            authority_level, mode,
        });
    }
}


// -- OT Guardian v6.2 ---------------------------------------------------
// Air-Gap Semántico: physics cannot lie.
// If sensors say X but physics implies Y ≠ X → signal = compromised.

#[derive(Debug, Clone, Copy, PartialEq)]
enum OtSeverity { Compromised, Suspicious, Degraded }

struct OtRule {
    name:        &'static str,
    description: &'static str,
    severity:    OtSeverity,
    /// Returns true if anomaly detected. Receives the full twin state snapshot.
    check_fn:    fn(vars: &std::collections::HashMap<String, f64>) -> bool,
}

/// Rules encode physics incoherence — the "Air-Gap Semántico"
static OT_RULES: &[OtRule] = &[
    OtRule {
        name: "pressure_power_incoherence",
        description: "Low residual pressure with high motor power — possible pipe burst or bypass valve open (Stuxnet pattern)",
        severity: OtSeverity::Compromised,
        check_fn: |vars| {
            let pressure = vars.get("residual_pressure").or_else(|| vars.get("residual")).copied().unwrap_or(f64::MAX);
            let power    = vars.get("motor_power_hp").or_else(|| vars.get("hp_required")).copied().unwrap_or(0.0);
            pressure < 30.0 && power > 40.0
        },
    },
    OtRule {
        name: "flow_pressure_incoherence",
        description: "High flow demand with near-zero pressure — sensor spoofing or catastrophic pipe failure",
        severity: OtSeverity::Compromised,
        check_fn: |vars| {
            let flow     = vars.get("flow_gpm").copied().unwrap_or(0.0);
            let pressure = vars.get("residual_pressure").or_else(|| vars.get("residual")).copied().unwrap_or(f64::MAX);
            flow > 500.0 && pressure < 10.0
        },
    },
    OtRule {
        name: "voltage_drop_spike",
        description: "Voltage drop exceeds 2× NEC limit — possible unauthorized load injection",
        severity: OtSeverity::Suspicious,
        check_fn: |vars| {
            let v_drop = vars.get("V_drop").copied().unwrap_or(0.0);
            v_drop > 10.0
        },
    },
    OtRule {
        name: "pump_hp_anomaly",
        description: "Motor HP far above design spec — unexpected load added to circuit",
        severity: OtSeverity::Suspicious,
        check_fn: |vars| {
            let hp = vars.get("hp_required").copied().unwrap_or(0.0);
            hp > 200.0
        },
    },
    OtRule {
        name: "friction_loss_spike",
        description: "Friction loss 3× normal — possible valve partially closed or blockage",
        severity: OtSeverity::Degraded,
        check_fn: |vars| {
            let friction = vars.get("friction_loss").or_else(|| vars.get("friction")).copied().unwrap_or(0.0);
            friction > 150.0
        },
    },
];

fn ot_analyze_twin() -> String {
    // Collect current twin state as f64 map
    let twin = twin_state_map();
    let vars: std::collections::HashMap<String, f64> = twin.iter()
        .map(|entry| (entry.key().clone(), entry.value().0))
        .collect();

    let mut findings: Vec<String> = Vec::new();
    let mut max_severity = "nominal";

    for rule in OT_RULES {
        if (rule.check_fn)(&vars) {
            let sev_str = match rule.severity {
                OtSeverity::Compromised => { max_severity = "compromised"; "compromised" },
                OtSeverity::Suspicious  => {
                    if max_severity != "compromised" { max_severity = "suspicious"; }
                    "suspicious"
                },
                OtSeverity::Degraded => {
                    if max_severity == "nominal" { max_severity = "degraded"; }
                    "degraded"
                },
            };
            findings.push(format!(
                r#"{{"rule":"{}","severity":"{}","description":"{}"}}"#,
                rule.name, sev_str, rule.description
            ));
        }
    }

    // Load active cyber threats from DKP
    let threats = {
        let store = fact_store_lock();
        store.iter()
            .filter(|f| f.domain == "cyber_ot")
            .map(|f| format!(
                r#"{{"key":"{}","value_s":"{}","confidence":{:.2},"source":"{}"}}"#,
                escape_json(&f.key), escape_json(&f.value_s), f.confidence, escape_json(&f.source)
            ))
            .collect::<Vec<_>>()
    };

    format!(
        r#"{{"ok":true,"ot_status":"{}","findings_count":{},"threats_loaded":{},"findings":[{}],"active_threats":[{}]}}"#,
        max_severity,
        findings.len(),
        threats.len(),
        findings.join(","),
        threats.join(",")
    )
}

/// POST /ot/threat_ingest
/// {"source":"cisa","type":"ics_advisory","affected_hw":"plc_s7_1200",
///  "attack_vector":"unauthorized_write","protocol":"modbus","cvss":8.5,"ttl_secs":604800}
fn ot_threat_ingest(body: &str) -> String {
    let source      = extract_json_str(body, "source").unwrap_or_else(|| "unknown".into());
    let adv_type    = extract_json_str(body, "type").unwrap_or_else(|| "advisory".into());
    let affected_hw = extract_json_str(body, "affected_hw").unwrap_or_else(|| "unknown".into());
    let vector      = extract_json_str(body, "attack_vector").unwrap_or_else(|| "unknown".into());
    let protocol    = extract_json_str(body, "protocol").unwrap_or_else(|| "unknown".into());
    let cvss        = extract_json_float(body, "cvss").unwrap_or(5.0);
    let ttl_secs    = extract_json_float(body, "ttl_secs").map(|v| v as u64).unwrap_or(604_800); // 7 days

    // Confidence derived from CVSS: scale 0-10 → 0.5-1.0
    let confidence  = (0.5 + cvss / 20.0) as f32;

    // Key: type:hw:vector — unique per threat signature
    let key = format!("{}:{}:{}", adv_type, affected_hw, vector);
    let value_s = format!("protocol={} cvss={:.1}", protocol, cvss);

    // Ingest as a DKP fact in domain=cyber_ot
    let ingest_body = format!(
        r#"{{"domain":"cyber_ot","key":"{}","value":{},"value_str":"{}","unit":"cvss","confidence":{},"source":"{}","ttl_secs":{}}}"#,
        escape_json(&key), cvss, escape_json(&value_s),
        confidence, escape_json(&source), ttl_secs
    );

    let result = dkp_ingest(&ingest_body);
    format!(
        r#"{{"ok":true,"threat_ingested":true,"key":"{}","cvss":{:.1},"confidence":{:.2},"ttl_days":{:.1},"dkp_result":{}}}"#,
        escape_json(&key), cvss, confidence, ttl_secs as f64 / 86400.0, result
    )
}

/// GET /ot/status — quick OT health check without full analysis
fn ot_status() -> String {
    let var_count = twin_state_map().len();

    let threat_count = {
        let store = fact_store_lock();
        store.iter().filter(|f| f.domain == "cyber_ot").count()
    };

    let rule_count = OT_RULES.len();
    format!(
        r#"{{"ok":true,"ot_rules":{},"twin_variables":{},"active_threats":{},"endpoints":["/ot/analyze","/ot/threat_ingest","/ot/status"]}}"#,
        rule_count, var_count, threat_count
    )
}


// -- Multi-Domain Knowledge Packs v7.2 ----------------------------------

struct DomainFact {
    key:        &'static str,
    value:      f64,
    unit:       &'static str,
    confidence: f32,
    source:     &'static str,
}

struct DomainPack {
    name:        &'static str,
    description: &'static str,
    facts:       &'static [DomainFact],
    ot_rules:    &'static [&'static str],  // names of applicable OT rules
}

static DOMAIN_PACKS: &[DomainPack] = &[
    DomainPack {
        name: "health",
        description: "Clinics & Hospitals: WHO/OMS biosafety, medical gases, autoclave, HVAC",
        ot_rules: &["pressure_power_incoherence","flow_pressure_incoherence","voltage_drop_spike"],
        facts: &[
            DomainFact { key:"autoclave_min_temp_c",       value:134.0, unit:"°C",   confidence:0.99, source:"who_biosafety_2020" },
            DomainFact { key:"autoclave_min_pressure_bar", value:2.0,   unit:"bar",  confidence:0.99, source:"who_biosafety_2020" },
            DomainFact { key:"autoclave_hold_time_min",    value:3.0,   unit:"min",  confidence:0.99, source:"who_biosafety_2020" },
            DomainFact { key:"surgery_airflow_ach",        value:20.0,  unit:"ACH",  confidence:0.98, source:"ashrae_170_2021" },
            DomainFact { key:"icu_airflow_ach",            value:12.0,  unit:"ACH",  confidence:0.98, source:"ashrae_170_2021" },
            DomainFact { key:"o2_supply_pressure_psi",     value:55.0,  unit:"psi",  confidence:0.97, source:"nfpa_99_2021" },
            DomainFact { key:"n2o_supply_pressure_psi",    value:55.0,  unit:"psi",  confidence:0.97, source:"nfpa_99_2021" },
            DomainFact { key:"medical_vacuum_inhg",        value:19.0,  unit:"inHg", confidence:0.97, source:"nfpa_99_2021" },
            DomainFact { key:"max_voltage_drop_v",         value:3.0,   unit:"V",    confidence:0.95, source:"nec_517_hospital" },
            DomainFact { key:"min_residual_pressure_psi",  value:80.0,  unit:"psi",  confidence:0.98, source:"nfpa_20_hospital" },
            DomainFact { key:"emergency_power_transfer_s", value:10.0,  unit:"s",    confidence:0.99, source:"nfpa_99_2021" },
        ],
    },
    DomainPack {
        name: "agro",
        description: "Agriculture/Irrigation: evapotranspiration, drip systems, soil chemistry",
        ot_rules: &["flow_pressure_incoherence","friction_loss_spike"],
        facts: &[
            DomainFact { key:"eto_ref_mm_day",              value:5.0,   unit:"mm/d", confidence:0.90, source:"fao_56_penman" },
            DomainFact { key:"drip_emitter_flow_lph",       value:2.0,   unit:"L/h",  confidence:0.92, source:"irrigation_association" },
            DomainFact { key:"drip_operating_pressure_psi", value:12.0,  unit:"psi",  confidence:0.93, source:"irrigation_association" },
            DomainFact { key:"soil_fc_pct",                 value:35.0,  unit:"%",    confidence:0.85, source:"fao_soil_guide" },
            DomainFact { key:"irrigation_efficiency",       value:0.90,  unit:"ratio",confidence:0.90, source:"fao_56_penman" },
            DomainFact { key:"npk_n_ratio_pct",             value:15.0,  unit:"%",    confidence:0.88, source:"inia_peru_2023" },
            DomainFact { key:"npk_p_ratio_pct",             value:8.0,   unit:"%",    confidence:0.88, source:"inia_peru_2023" },
            DomainFact { key:"max_friction_loss_psi",       value:5.0,   unit:"psi",  confidence:0.90, source:"irrigation_association" },
            DomainFact { key:"pump_max_flow_gpm",           value:200.0, unit:"gpm",  confidence:0.88, source:"irrigation_design_std" },
        ],
    },
    DomainPack {
        name: "mining",
        description: "Mining/Energy: structural FoS, motor efficiency, IEEE/NEC power, slope stability",
        ot_rules: &["pressure_power_incoherence","voltage_drop_spike","pump_hp_anomaly"],
        facts: &[
            DomainFact { key:"structural_fos_min",          value:1.5,   unit:"ratio",confidence:0.99, source:"aci_318_2019" },
            DomainFact { key:"slope_fos_min",               value:1.3,   unit:"ratio",confidence:0.99, source:"osha_1926_653" },
            DomainFact { key:"motor_efficiency_min",        value:0.90,  unit:"ratio",confidence:0.95, source:"ieee_841_2009" },
            DomainFact { key:"max_voltage_drop_v",          value:7.5,   unit:"V",    confidence:0.95, source:"nec_215_2020" },
            DomainFact { key:"hp_special_protection",       value:200.0, unit:"hp",   confidence:0.97, source:"nfpa_79_2021" },
            DomainFact { key:"vibration_alarm_mm_s",        value:7.1,   unit:"mm/s", confidence:0.93, source:"iso_10816_3" },
            DomainFact { key:"vibration_trip_mm_s",         value:11.2,  unit:"mm/s", confidence:0.97, source:"iso_10816_3" },
            DomainFact { key:"max_project_budget_usd",      value:500_000.0, unit:"USD", confidence:0.70, source:"internal_estimate" },
            DomainFact { key:"min_residual_pressure_psi",   value:45.0,  unit:"psi",  confidence:0.95, source:"mine_safety_regulations" },
        ],
    },
    DomainPack {
        name: "logistics",
        description: "Logistics/ERP: route optimization, vehicle capacity, financial thresholds",
        ot_rules: &["voltage_drop_spike"],
        facts: &[
            DomainFact { key:"max_vehicle_load_kg",         value:30_000.0, unit:"kg", confidence:0.95, source:"peru_reglamento_transporte" },
            DomainFact { key:"route_efficiency_target",     value:0.85,  unit:"ratio",confidence:0.80, source:"logistics_kpi_std" },
            DomainFact { key:"fuel_consumption_l_100km",    value:35.0,  unit:"L/100km",confidence:0.85, source:"vehiculo_pesado_std" },
            DomainFact { key:"max_transaction_usd_alert",   value:10_000.0, unit:"USD",confidence:0.90, source:"sunat_reg_2023" },
            DomainFact { key:"warehouse_temp_max_c",        value:25.0,  unit:"°C",   confidence:0.88, source:"indecopi_food_storage" },
            DomainFact { key:"max_project_budget_usd",      value:100_000.0, unit:"USD",confidence:0.70, source:"internal_estimate" },
        ],
    },
];

fn domain_bootstrap(body: &str) -> String {
    let domain_name = match extract_json_str(body, "domain") {
        Some(d) => d,
        None => return r#"{"ok":false,"error":"missing domain"}"#.into(),
    };
    let pack = match DOMAIN_PACKS.iter().find(|p| p.name == domain_name.as_str()) {
        Some(p) => p,
        None => {
            let names: Vec<&str> = DOMAIN_PACKS.iter().map(|p| p.name).collect();
            return format!(r#"{{"ok":false,"error":"unknown domain","available":{:?}}}"#, names);
        }
    };
    let ts = twin_ts();
    let mut ingested = 0usize;
    {
        let mut store = fact_store_lock();
        for f in pack.facts {
            store.retain(|existing| !(existing.domain == domain_name && existing.key == f.key));
            store.push(KnowledgeFact {
                id:         fact_id(&domain_name, f.key, f.unit),
                domain:     domain_name.clone(),
                key:        f.key.to_string(),
                value_f:    f.value,
                value_s:    format!("{:.4}", f.value),
                unit:       f.unit.to_string(),
                confidence: f.confidence,
                source:     f.source.to_string(),
                created_ts: ts,
                ttl_secs:   0,
        authority_level: 0,
        mode: "snapshot".to_string(),
    });
            ingested += 1;
        }
    }
    save_dkp_store();
    format!(
        r#"{{"ok":true,"domain":"{}","description":"{}","facts_ingested":{},"ot_rules":{:?}}}"#,
        domain_name, pack.description, ingested, pack.ot_rules
    )
}

fn domain_catalog() -> String {
    let entries: Vec<String> = DOMAIN_PACKS.iter().map(|p| {
        format!(r#"{{"name":"{}","description":"{}","facts":{},"ot_rules":{}}}"#,
            p.name, p.description, p.facts.len(), p.ot_rules.len())
    }).collect();
    format!(r#"{{"ok":true,"domains":[{}]}}"#, entries.join(","))
}

// -- IoT Event Buffer v7.2 ----------------------------------------------

#[derive(Debug, Clone)]
struct IotEvent {
    ts:       u64,
    var_name: String,
    value:    f64,
    unit:     String,
    source:   String,
}

static IOT_EVENT_BUF: std::sync::OnceLock<
    std::sync::Mutex<std::collections::VecDeque<IotEvent>>
> = std::sync::OnceLock::new();

fn iot_buf_lock()
    -> std::sync::MutexGuard<'static, std::collections::VecDeque<IotEvent>>
{
    IOT_EVENT_BUF.get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::with_capacity(1000)))
        .lock().unwrap_or_else(|p| p.into_inner())
}

/// Called from twin_update to push events into the ring buffer
fn iot_push_event(var_name: &str, value: f64, unit: &str, source: &str) {
    let ts = twin_ts();
    let ev = IotEvent { ts, var_name: var_name.to_string(), value, unit: unit.to_string(), source: source.to_string() };
    let mut buf = iot_buf_lock();
    if buf.len() >= 1000 { buf.pop_front(); }
    buf.push_back(ev);
}

fn iot_events_json(since_ts: u64, limit: usize) -> String {
    let buf = iot_buf_lock();
    let events: Vec<String> = buf.iter()
        .filter(|e| e.ts >= since_ts)
        .rev().take(limit)
        .map(|e| format!(
            r#"{{"ts":{},"var":"{}","value":{:.4},"unit":"{}","source":"{}"}}"#,
            e.ts, escape_json(&e.var_name), e.value, escape_json(&e.unit), escape_json(&e.source)
        ))
        .collect();
    let oldest_ts = buf.front().map(|e| e.ts).unwrap_or(0);
    format!(
        r#"{{"ok":true,"count":{},"oldest_ts":{},"events":[{}]}}"#,
        events.len(), oldest_ts, events.join(",")
    )
}


// -- NVD/CISA Auto-Polling v7.3 ----------------------------------------
// Background thread: polls NVD every POLL_INTERVAL_SECS for ICS/OT CVEs
// Uses curl subprocess (no reqwest dependency needed)

const NVD_POLL_INTERVAL_SECS: u64 = 3600; // 1 hour

static NVD_LAST_POLL_TS: AtomicU64 = AtomicU64::new(0);
static NVD_FACTS_INGESTED: AtomicU64 = AtomicU64::new(0);



/// Parse CVEs from NVD JSON response and ingest into DKP.
/// NVD API v2: {"vulnerabilities":[{"cve":{"id":"CVE-...","metrics":{"cvssMetricV31":[{"cvssData":{"baseScore":8.5}}]},"descriptions":[{"lang":"en","value":"..."}],"configurations":[...]}}]}
fn parse_and_ingest_nvd(json: &str) -> usize {
    let mut count = 0usize;
    let mut pos = 0;
    // Find each CVE id
    while let Some(cve_start) = json[pos..].find("\"CVE-").map(|i| pos + i) {
        let id_start = cve_start + 1;
        let id_end = json[id_start..].find('"').map(|i| id_start + i).unwrap_or(id_start + 20);
        let cve_id = &json[id_start..id_end];
        pos = id_end;

        // Find CVSS score near this CVE entry (within next 2000 chars)
        let chunk = &json[pos..std::cmp::min(pos + 2000, json.len())];
        let cvss = chunk.find("\"baseScore\"")
            .and_then(|p| chunk[p+11..].find(':').map(|q| p+11+q+1))
            .and_then(|p| chunk[p..].trim_start_matches(' ').split([',', '}']).next())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(0.0);

        // Only ingest CVEs with CVSS >= 7.0 (high/critical)
        if cvss < 7.0 { continue; }

        // Extract description (first 120 chars)
        let desc = chunk.find("\"value\"")
            .and_then(|p| chunk[p+7..].find('"').map(|q| p+7+q+1))
            .and_then(|p| chunk[p..].find('"').map(|q| chunk[p..p+q].to_string()))
            .unwrap_or_else(|| "ICS vulnerability".to_string());
        let desc_short: String = desc.chars().take(120).collect();

        // Ingest as cyber_ot fact, TTL 30 days
        let key = format!("nvd:{}", cve_id);
        let value_s = format!("cvss={:.1} {}", cvss, desc_short);
        let confidence = ((0.5 + cvss / 20.0) as f32).clamp(0.0, 1.0);
        let ts = twin_ts();

        {
            let mut store = fact_store_lock();
            store.retain(|f| !(f.domain == "cyber_ot" && f.key == key));
            store.push(KnowledgeFact {
                id:         fact_id("cyber_ot", &key, &value_s),
                domain:     "cyber_ot".to_string(),
                key:        key.clone(),
                value_f:    cvss,
                value_s:    value_s.clone(),
                unit:       "cvss".to_string(),
                confidence,
                source:     "nvd.nist.gov".to_string(),
                created_ts: ts,
                ttl_secs:   2_592_000, // 30 days,
        authority_level: 0,
        mode: "snapshot".to_string(),
    });
        }
        count += 1;
    }
    if count > 0 { save_dkp_store(); }
    count
}

/// POST /nvd/poll — manual NVD fetch (also called by background thread)
/// Queries NVD for recent ICS/OT CVEs (Siemens, Schneider, ABB, Modbus, Profinet)
fn nvd_poll_manual() -> String {
    let now = twin_ts();
    // ICS-relevant keyword search via NVD API v2
    // Using keywordSearch for common OT/ICS vendors and protocols
    let queries = &[
        "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=SCADA&resultsPerPage=20",
        "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=PLC+Siemens&resultsPerPage=10",
        "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=Modbus&resultsPerPage=10",
    ];
    let mut total_ingested = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for url in queries {
        // Use curl with timeout (5s connect, 15s max)
        let output = std::process::Command::new("curl")
            .args(&["-s", "--connect-timeout", "5", "--max-time", "15",
                    "-H", "User-Agent: CRYS-L/7.3 OT-Guardian",
                    "-H", "Accept: application/json",
                    url])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let json = String::from_utf8_lossy(&out.stdout);
                let n = parse_and_ingest_nvd(&json);
                total_ingested += n;
            }
            Ok(out) => {
                errors.push(format!("curl_exit_{}", out.status.code().unwrap_or(-1)));
            }
            Err(e) => {
                errors.push(format!("spawn_error:{}", e));
            }
        }
    }

    NVD_LAST_POLL_TS.store(now, Ordering::Relaxed);
    NVD_FACTS_INGESTED.fetch_add(total_ingested as u64, Ordering::Relaxed);

    let err_json = if errors.is_empty() { "[]".into() }
        else { format!("[{}]", errors.iter().map(|e| format!(r#""{}""#, e)).collect::<Vec<_>>().join(",")) };

    format!(
        r#"{{"ok":true,"ts":{},"cves_ingested":{},"queries":{},"errors":{}}}"#,
        now, total_ingested, queries.len(), err_json
    )
}

fn dkp_crawl(body: &str) -> String {
    let url = extract_json_str(body, "url");
    let pdf = extract_json_str(body, "pdf");
    let domain = extract_json_str(body, "domain").unwrap_or_else(|| "auto".into());
    let authority = extract_json_float(body, "authority").map(|v| v as u8);

    let (source_arg, source_val) = if let Some(u) = url {
        ("--url", u)
    } else if let Some(p) = pdf {
        ("--pdf", p)
    } else {
        return r#"{"ok":false,"error":"missing url or pdf field"}"#.into();
    };

    // Validate: no shell injection — only allow safe URL/path chars
    let safe_re = |s: &str| s.chars().all(|c| c.is_alphanumeric() || "/:._-?=&%+#@".contains(c));
    if !safe_re(&source_val) {
        return r#"{"ok":false,"error":"invalid characters in source"}"#.into();
    }

    let crawler = "/opt/crysl/dkp_crawler.py";
    let mut cmd = std::process::Command::new("python3");
    cmd.arg(crawler).arg(source_arg).arg(&source_val);
    if domain != "auto" { cmd.arg("--domain").arg(&domain); }
    if let Some(auth) = authority { cmd.arg("--authority").arg(auth.to_string()); }

    let before = {
        let store = fact_store_lock();
        store.len()
    };

    let output = cmd.output();
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let after = {
                let store = fact_store_lock();
                store.len()
            };
            let new_facts = (after as i64) - (before as i64);
            // Parse ingested/skipped from log output
            let ingested = stdout.lines()
                .find(|l| l.contains("ingested="))
                .and_then(|l| {
                    let start = l.find("ingested=")? + 9;
                    let end = l[start..].find(|c: char| !c.is_ascii_digit()).map(|i| start+i).unwrap_or(l.len());
                    l[start..end].parse::<usize>().ok()
                }).unwrap_or(0);
            format!(
                r#"{{"ok":true,"source":"{}","domain":"{}","ingested":{},"total_facts":{},"log":{}}}"#,
                &source_val, domain, ingested, after,
                serde_json::json!(stdout.lines().filter(|l| !l.is_empty()).collect::<Vec<_>>())
            )
        }
        Err(e) => format!(r#"{{"ok":false,"error":"{}"}}"#, e),
    }
}

fn nvd_status() -> String {
    let last = NVD_LAST_POLL_TS.load(Ordering::Relaxed);
    let total = NVD_FACTS_INGESTED.load(Ordering::Relaxed) as usize;
    let now = twin_ts();
    let age = now.saturating_sub(last);
    let next = if last == 0 { 0 } else { NVD_POLL_INTERVAL_SECS.saturating_sub(age) };
    format!(
        r#"{{"ok":true,"last_poll_ts":{},"age_secs":{},"next_poll_secs":{},"total_cves_ingested":{},"poll_interval_secs":{}}}"#,
        last, age, next, total, NVD_POLL_INTERVAL_SECS
    )
}

/// Spawn NVD background polling thread — called once at server startup
fn spawn_nvd_poller() {
    std::thread::spawn(|| {
        // Initial delay: 30 seconds after startup
        std::thread::sleep(std::time::Duration::from_secs(30));
        loop {
            let _result = nvd_poll_manual();
            std::thread::sleep(std::time::Duration::from_secs(NVD_POLL_INTERVAL_SECS));
        }
    });
}


// -- P2P DKP: Signed Knowledge Exchange v8.2 ----------------------------
// FNV-64 HMAC: sign(facts_json + "|" + ts + "|" + secret) → trust_token
// Not cryptographically strong, but sufficient for local mesh trust.

fn p2p_sign(payload: &str, secret: &str, ts: u64) -> u64 {
    let combined = format!("{}|{}|{}", payload, ts, secret);
    let mut h: u64 = 14695981039346656037;
    for b in combined.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn p2p_secret() -> String {
    std::env::var("CRYSL_P2P_SECRET")
        .or_else(|_| std::env::var("QOMNI_MESH_SECRET"))
        .unwrap_or_else(|_| "crysl-default-p2p-secret-2026".to_string())
}

/// GET /dkp/export — export all facts as signed bundle
/// {"token":"HEX","ts":N,"facts":[...]}
fn dkp_export(body: &str) -> String {
    let domain_filter = extract_json_str(body, "domain");
    let min_conf = extract_json_float(body, "min_confidence").map(|v| v as f32).unwrap_or(0.0);
    let ts = twin_ts();
    let secret = p2p_secret();

    let store = fact_store_lock();
    let facts: Vec<String> = store.iter()
        .filter(|f| {
            if f.confidence < min_conf { return false; }
            if let Some(ref d) = domain_filter { if &f.domain != d { return false; } }
            true
        })
        .map(|f| format!(
            r#"{{"domain":"{}","key":"{}","value":{},"value_str":"{}","unit":"{}","confidence":{:.4},"source":"{}","ttl_secs":{}}}"#,
            escape_json(&f.domain), escape_json(&f.key), f.value_f,
            escape_json(&f.value_s), escape_json(&f.unit), f.confidence,
            escape_json(&f.source), f.ttl_secs
        ))
        .collect();
    drop(store);

    let facts_json = format!("[{}]", facts.join(","));
    let token = p2p_sign(&facts_json, &secret, ts);

    format!(
        r#"{{"ok":true,"ts":{},"count":{},"token":"{:x}","facts":{}}}"#,
        ts, facts.len(), token, facts_json
    )
}

/// POST /dkp/import — receive signed bundle from peer, validate, ingest
/// {"token":"HEX","ts":N,"facts":[...],"peer":"instance_id"}
fn dkp_import(body: &str) -> String {
    let peer = extract_json_str(body, "peer").unwrap_or_else(|| "unknown".into());
    let ts = extract_json_float(body, "ts").map(|v| v as u64).unwrap_or(0);
    let token_hex = match extract_json_str(body, "token") {
        Some(t) => t,
        None => return r#"{"ok":false,"error":"missing token"}"#.into(),
    };
    let token_recv = u64::from_str_radix(token_hex.trim_start_matches("0x"), 16)
        .unwrap_or(0);

    // Extract facts array substring for signature verification
    let facts_start = match body.find("\"facts\"") {
        Some(i) => i, None => return r#"{"ok":false,"error":"missing facts"}"#.into(),
    };
    let arr_start = match body[facts_start..].find('[').map(|i| facts_start + i) {
        Some(i) => i, None => return r#"{"ok":false,"error":"malformed facts"}"#.into(),
    };
    let mut depth = 0i32;
    let mut arr_end = arr_start;
    for (i, b) in body[arr_start..].bytes().enumerate() {
        match b { b'[' => depth += 1, b']' => { depth -= 1; if depth == 0 { arr_end = arr_start + i + 1; break; } } _ => {} }
    }
    let facts_json = &body[arr_start..arr_end];

    // Verify signature (allow ±300s clock drift)
    let secret = p2p_secret();
    let now = twin_ts();
    let valid_ts = now.saturating_sub(ts) < 300 || ts.saturating_sub(now) < 300;
    let expected = p2p_sign(facts_json, &secret, ts);

    if !valid_ts {
        return format!(r#"{{"ok":false,"error":"timestamp_expired","age_secs":{}}}"#,
            now.saturating_sub(ts));
    }
    if token_recv != expected {
        return r#"{"ok":false,"error":"invalid_signature"}"#.into();
    }

    // Parse and ingest facts from the array
    let ingest_body = format!(r#"{{"domain":"__import__","key":"__batch__"}}"#); // placeholder
    let mut count = 0usize;
    let mut pos = 1usize; // skip '['
    let bytes = facts_json.as_bytes();

    while pos < facts_json.len() {
        while pos < facts_json.len() && matches!(bytes[pos], b' '|b','|b'\n'|b'\r') { pos += 1; }
        if pos >= facts_json.len() || bytes[pos] == b']' { break; }
        if bytes[pos] != b'{' { pos += 1; continue; }
        let obj_start = pos;
        let mut d = 0usize;
        let mut obj_end = pos;
        for i in pos..facts_json.len() {
            match bytes[i] {
                b'{' => d += 1,
                b'}' => { d -= 1; if d == 0 { obj_end = i + 1; break; } }
                _ => {}
            }
        }
        let obj = &facts_json[obj_start..obj_end];
        pos = obj_end;

        let domain  = extract_json_str(obj, "domain").unwrap_or_default();
        let key     = extract_json_str(obj, "key").unwrap_or_default();
        let value_f = extract_json_float(obj, "value").unwrap_or(0.0);
        let value_s = extract_json_str(obj, "value_str").unwrap_or_else(|| format!("{:.6}", value_f));
        let unit    = extract_json_str(obj, "unit").unwrap_or_default();
        let confidence = extract_json_float(obj, "confidence").unwrap_or(0.8) as f32;
        let source  = format!("peer:{}", peer);
        let ttl     = extract_json_float(obj, "ttl_secs").map(|v| v as u64).unwrap_or(86400);

        if domain.is_empty() || key.is_empty() { continue; }

        let fact_ts = twin_ts();
        {
            let mut store = fact_store_lock();
            store.retain(|f| !(f.domain == domain && f.key == key));
            store.push(KnowledgeFact {
                id: fact_id(&domain, &key, &value_s),
                domain, key, value_f, value_s, unit,
                confidence, source, created_ts: fact_ts, ttl_secs: ttl,
        authority_level: 0,
        mode: "snapshot".to_string(),
    });
        }
        count += 1;
    }
    if count > 0 { save_dkp_store(); }

    format!(
        r#"{{"ok":true,"imported":{},"peer":"{}","ts":{}}}"#,
        count, escape_json(&peer), ts
    )
}

// -- SSE Streaming v8.2 -------------------------------------------------

/// GET /twin/sse — Server-Sent Events stream (stays open ~30s, sends IoT events)
fn twin_sse_response(since_ts: u64) -> (Vec<u8>, bool) {
    // Build SSE body: collect next 10 events since since_ts, format as SSE
    let buf = iot_buf_lock();
    let events: Vec<String> = buf.iter()
        .filter(|e| e.ts >= since_ts)
        .map(|e| format!(
            "data: {{\"ts\":{},\"var\":\"{}\",\"value\":{:.4},\"unit\":\"{}\"}}\n\n",
            e.ts, e.var_name, e.value, e.unit
        ))
        .collect();
    drop(buf);

    let heartbeat = format!("data: {{\"ts\":{},\"type\":\"heartbeat\"}}\n\n", twin_ts());
    let body = if events.is_empty() { heartbeat } else { events.join("") + &heartbeat };
    (body.into_bytes(), true)
}


// -- Registry Auto-Sync v8.3 ------------------------------------------------
// Background thread: sync new facts from CRYSL-REGISTRY (port 9002) every 60s

const REGISTRY_SYNC_INTERVAL: u64 = 60;
const REGISTRY_URL: &str = "http://127.0.0.1:9002";

static REGISTRY_LAST_SYNC: AtomicU64 = AtomicU64::new(0);
static REGISTRY_FACTS_SYNCED: AtomicU64 = AtomicU64::new(0);



/// Pull facts from registry since last_sync timestamp, ingest into local DKP
fn registry_sync_once() -> usize {
    let last = REGISTRY_LAST_SYNC.load(Ordering::Relaxed);
    let url = format!("{}/dkp/sync?since={}", REGISTRY_URL, last);

    let output = std::process::Command::new("curl")
        .args(&["-s", "--connect-timeout", "3", "--max-time", "10",
                "-H", "Accept: application/json", &url])
        .output();

    let json = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        _ => return 0,
    };

    // Parse facts array from response: {"facts":[...]} or {"results":[...]}
    let arr_key = if json.contains("\"facts\"") { "\"facts\"" } else { "\"results\"" };
    let facts_start = match json.find(arr_key) { Some(i) => i, None => return 0 };
    let arr_start = match json[facts_start..].find('[').map(|i| facts_start + i) {
        Some(i) => i, None => return 0,
    };
    let mut depth = 0i32;
    let mut arr_end = arr_start;
    for (i, b) in json[arr_start..].bytes().enumerate() {
        match b { b'[' => depth += 1, b']' => { depth -= 1; if depth == 0 { arr_end = arr_start + i + 1; break; } } _ => {} }
    }
    let facts_json = &json[arr_start..arr_end];

    let mut count = 0usize;
    let mut pos = 1usize;
    let bytes = facts_json.as_bytes();
    while pos < facts_json.len() {
        while pos < facts_json.len() && matches!(bytes[pos], b' '|b','|b'\n'|b'\r') { pos += 1; }
        if pos >= facts_json.len() || bytes[pos] == b']' { break; }
        if bytes[pos] != b'{' { pos += 1; continue; }
        let obj_start = pos;
        let mut d = 0usize;
        let mut obj_end = pos;
        for i in pos..facts_json.len() {
            match bytes[i] { b'{' => d += 1, b'}' => { d -= 1; if d == 0 { obj_end = i + 1; break; } } _ => {} }
        }
        let obj = &facts_json[obj_start..obj_end];
        pos = obj_end;

        let domain  = extract_json_str(obj, "domain").unwrap_or_default();
        let key     = extract_json_str(obj, "key").unwrap_or_default();
        let value_f = extract_json_float(obj, "value").unwrap_or(0.0);
        let value_s = extract_json_str(obj, "value_str")
            .unwrap_or_else(|| format!("{:.6}", value_f));
        let unit       = extract_json_str(obj, "unit").unwrap_or_default();
        let confidence = extract_json_float(obj, "confidence").unwrap_or(0.8) as f32;
        let ttl        = extract_json_float(obj, "ttl_secs").map(|v| v as u64).unwrap_or(86400);

        if domain.is_empty() || key.is_empty() { continue; }
        // Only accept facts with confidence >= 0.7
        if confidence < 0.7 { continue; }

        let ts = twin_ts();
        {
            let mut store = fact_store_lock();
            store.retain(|f| !(f.domain == domain && f.key == key));
            store.push(KnowledgeFact {
                id: fact_id(&domain, &key, &value_s),
                domain, key, value_f, value_s, unit,
                confidence,
                source: "registry:sync".to_string(),
                created_ts: ts,
                ttl_secs: ttl,
        authority_level: 0,
        mode: "snapshot".to_string(),
    });
        }
        count += 1;
    }
    if count > 0 { save_dkp_store(); }
    count
}

fn registry_sync_status() -> String {
    let last = REGISTRY_LAST_SYNC.load(Ordering::Relaxed);
    let total = REGISTRY_FACTS_SYNCED.load(Ordering::Relaxed) as usize;
    let now = twin_ts();
    let age = now.saturating_sub(last);
    format!(
        r#"{{"ok":true,"last_sync_ts":{},"age_secs":{},"total_synced":{},"interval_secs":{},"registry_url":"{}"}}"#,
        last, age, total, REGISTRY_SYNC_INTERVAL, REGISTRY_URL
    )
}

/// Spawn registry sync background thread — call once at startup
fn spawn_registry_syncer() {
    std::thread::spawn(|| {
        // Initial delay: 10 seconds
        std::thread::sleep(std::time::Duration::from_secs(10));
        loop {
            let n = registry_sync_once();
            let now = twin_ts();
            REGISTRY_LAST_SYNC.store(now, Ordering::Relaxed);
            if n > 0 { REGISTRY_FACTS_SYNCED.fetch_add(n as u64, Ordering::Relaxed); }
            std::thread::sleep(std::time::Duration::from_secs(REGISTRY_SYNC_INTERVAL));
        }
    });
}


fn ensure_memory_dir() {
    let _ = std::fs::create_dir_all("/opt/crysl/memory");
    load_dkp_store();
}

fn save_graph_memory_v2(graph_summary: &str, success: bool) {
    let score = if success { 1.0f32 } else { 0.0f32 };
    record_memory_success(graph_summary, score);
    save_graph_memory(graph_summary);
}

fn save_graph_memory(graph_summary: &str) {
    ensure_memory_dir();
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true).open(GRAPH_MEMORY_PATH)
    {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs();
        let _ = writeln!(f, r#"{{"ts":{},"graph":{}}}"#, ts, graph_summary);
    }
}

fn load_graph_memory(limit: usize) -> Vec<String> {
    ensure_memory_dir();
    match std::fs::read_to_string(GRAPH_MEMORY_PATH) {
        Ok(content) => {
            let lines: Vec<String> = content.lines()
                .rev().take(limit).map(|l| l.to_string()).collect();
            lines
        }
        Err(_) => Vec::new(),
    }
}


// ── HTML stripper (no regex dependency) ───────────────────────────────

// -- Simulation variant parser -------------------------------------------

/// Parse variants array from /graph/simulate body.
/// Format: "variants": [{"node_id": {"param": value}}, ...]
/// Returns: Vec<HashMap<node_id, Vec<(param, value)>>>

/// Parse multi-objective goals array from body.
/// Format: "goals":[{"target":"A.hp","weight":0.6,"direction":"minimize"},...]
fn parse_goals_array(body: &str) -> Option<Vec<(String, f64, String)>> {
    let start = body.find("\"goals\"")?;
    let arr_start = body[start..].find('[').map(|i| start + i + 1)?;
    let mut depth = 1i32;
    let mut arr_end = arr_start;
    for (i, b) in body[arr_start..].bytes().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => { depth -= 1; if depth == 0 { arr_end = arr_start + i; break; } }
            _ => {}
        }
    }
    let arr = &body[arr_start..arr_end];
    let mut goals = Vec::new();
    let mut pos = 0usize;
    let bytes = arr.as_bytes();
    while pos < arr.len() {
        while pos < arr.len() && matches!(bytes[pos], b' '|b','|b'\n'|b'\r') { pos += 1; }
        if pos >= arr.len() || bytes[pos] == b']' { break; }
        if bytes[pos] != b'{' { pos += 1; continue; }
        let obj_start = pos;
        let mut d = 0usize;
        let mut obj_end = pos;
        for i in pos..arr.len() {
            match bytes[i] {
                b'{' => d += 1,
                b'}' => { d -= 1; if d == 0 { obj_end = i + 1; break; } }
                _ => {}
            }
        }
        if obj_end <= obj_start { break; }
        let obj = &arr[obj_start..obj_end];
        let target    = extract_json_str(obj, "target").unwrap_or_default();
        let weight    = extract_json_float(obj, "weight").unwrap_or(1.0);
        let direction = extract_json_str(obj, "direction").unwrap_or_else(|| "minimize".into());
        if !target.is_empty() {
            goals.push((target, weight, direction));
        }
        pos = obj_end;
    }
    if goals.is_empty() { None } else { Some(goals) }
}

fn parse_simulate_variants(
    body: &str
) -> Vec<std::collections::HashMap<String, Vec<(String, f64)>>> {
    let mut result = Vec::new();
    let start = match body.find("\"variants\"") { Some(s) => s, None => return result };
    let arr_start = match body[start..].find('[') { Some(i) => start+i+1, None => return result };
    let mut pos = arr_start;
    let bytes = body.as_bytes();
    while pos < body.len() {
        while pos < body.len() && matches!(bytes[pos], b' '|b','|b'\n'|b'\r') { pos += 1; }
        if pos >= body.len() || bytes[pos] == b']' { break; }
        if bytes[pos] != b'{' { pos += 1; continue; }
        let obj_start = pos;
        let mut depth = 0usize;
        let mut obj_end = pos;
        for i in pos..body.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => { depth -= 1; if depth == 0 { obj_end = i+1; break; } }
                _ => {}
            }
        }
        if obj_end <= obj_start { break; }
        let obj = &body[obj_start..obj_end];
        pos = obj_end;
        // Parse each node_id -> {param: value} in this variant object
        let mut variant_map: std::collections::HashMap<String, Vec<(String, f64)>> =
            std::collections::HashMap::new();
        let mut s = &obj[1..obj.len()-1]; // strip outer {}
        loop {
            s = s.trim_start_matches([',', ' ', '\n', '\r', '\t']);
            if s.is_empty() || !s.starts_with('"') { break; }
            let k_end = match s[1..].find('"') { Some(e) => e+1, None => break };
            let node_id = s[1..k_end].to_string();
            let rest = s[k_end+1..].trim_start_matches([' ', ':']);
            // Inner object {param: value, ...}
            let inner_start = match rest.find('{') { Some(i) => i+1, None => break };
            let mut d2 = 1usize;
            let mut inner_end = inner_start;
            for (i, b) in rest[inner_start..].bytes().enumerate() {
                match b { b'{' => d2 += 1, b'}' => { d2 -= 1; if d2 == 0 { inner_end = inner_start+i; break; } } _ => {} }
            }
            let inner = &rest[inner_start..inner_end];
            let mut params: Vec<(String, f64)> = Vec::new();
            let mut ps = inner;
            loop {
                ps = ps.trim_start_matches([',', ' ', '\n', '\r', '\t']);
                if ps.is_empty() || !ps.starts_with('"') { break; }
                let pk_end = match ps[1..].find('"') { Some(e) => e+1, None => break };
                let param_name = ps[1..pk_end].to_string();
                let after = ps[pk_end+1..].trim_start_matches([' ', ':']);
                let v_end = after.find([',', '}']).unwrap_or(after.len());
                if let Ok(val) = after[..v_end].trim().parse::<f64>() {
                    params.push((param_name, val));
                }
                ps = &after[v_end..];
            }
            variant_map.insert(node_id, params);
            s = &rest[inner_end+1..];
        }
        result.push(variant_map);
    }
    result
}


fn strip_html(html: &str) -> String {
    // Work char-by-char to avoid UTF-8 boundary panics
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut buf = String::new(); // accumulate chars for script detection
    let mut in_script = false;

    for ch in html.chars() {
        buf.push(ch);
        // Keep buf bounded — drain first char (not first byte)
        if buf.chars().count() > 16 {
            let n = buf.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            buf.drain(..n);
        }
        let lb = buf.to_lowercase();

        if !in_script && (lb.ends_with("<script") || lb.ends_with("<style")) {
            in_script = true;
        }
        if in_script {
            if lb.ends_with("</script>") || lb.ends_with("</style>") {
                in_script = false;
                buf.clear();
            }
            continue;
        }
        if ch == '<' { in_tag = true; continue; }
        if ch == '>' { in_tag = false; continue; }
        if !in_tag {
            if ch == '\n' || ch == '\r' || ch == '\t' { out.push(' '); }
            else { out.push(ch); }
        }
    }

    // Collapse whitespace
    let mut prev_space = false;
    let mut clean = String::with_capacity(out.len());
    for c in out.chars() {
        if c == ' ' {
            if !prev_space { clean.push(' '); }
            prev_space = true;
        } else {
            clean.push(c);
            prev_space = false;
        }
    }
    clean.trim().to_string()
}

/// Extract text around a keyword occurrence, ±context_chars on each side.
fn extract_around_keyword(text: &str, keyword: &str, context_chars: usize) -> String {
    let lower = text.to_lowercase();
    let kw_lower = keyword.to_lowercase();
    if let Some(pos) = lower.find(&kw_lower) {
        // Walk back from pos to a valid char boundary
        let start = {
            let mut s = pos.saturating_sub(context_chars / 2);
            while s > 0 && !text.is_char_boundary(s) { s -= 1; }
            s
        };
        let raw_end = (pos + kw_lower.len() + context_chars / 2).min(text.len());
        let end = {
            let mut e = raw_end;
            while e < text.len() && !text.is_char_boundary(e) { e += 1; }
            e.min(text.len())
        };
        text[start..end].to_string()
    } else {
        text.chars().take(context_chars).collect()
    }
}

// ── JSON helpers ──────────────────────────────────────────────────────

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Extract a numeric (float) value: "key":123.45
fn extract_json_float(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let start   = json.find(&pattern)? + pattern.len();
    let rest    = json[start..].trim();
    let end     = rest.find(|c: char| c == ',' || c == '}' || c == ' ')
        .unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Extract a string value: "key":"value"


fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
            | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

fn clean_str(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric() || " -_'.+=".contains(*c)).take(60).collect()
}

fn chrono_now_approx() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("unix:{}", secs)
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let start   = json.find(&pattern)? + pattern.len();
    let rest    = json[start..].trim();
    if rest.starts_with('"') {
        let chars: Vec<char> = rest[1..].chars().collect();
        let mut result = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            match chars[i] {
                '"' => break,
                '\\' if i + 1 < chars.len() => {
                    match chars[i + 1] {
                        'n'  => { result.push('\n'); i += 2; }
                        't'  => { result.push('\t'); i += 2; }
                        'r'  => { result.push('\r'); i += 2; }
                        '\\' => { result.push('\\'); i += 2; }
                        '"'  => { result.push('"');  i += 2; }
                        '/'  => { result.push('/');  i += 2; }
                        c    => { result.push('\\'); result.push(c); i += 2; }
                    }
                }
                c => { result.push(c); i += 1; }
            }
        }
        Some(result)
    } else {
        None
    }
}

/// Extract an array of f64: "key":[1.0, 2.0, 3.0]
fn extract_json_arr_float(json: &str, key: &str) -> Option<Vec<f64>> {
    let pattern = format!("\"{}\":", key);
    let start   = json.find(&pattern)? + pattern.len();
    let rest    = json[start..].trim();
    if !rest.starts_with('[') { return None; }
    let end = rest.find(']')? + 1;
    let inner = &rest[1..end-1];
    let vals: Vec<f64> = inner.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Some(vals)
}

/// Extract a flat object of string->f64 pairs: "key":{"a":1.0,"b":2.5}
fn extract_json_obj_float(json: &str, key: &str) -> Option<std::collections::HashMap<String, f64>> {
    let pattern = format!("\"{}\":", key);
    let start   = json.find(&pattern)? + pattern.len();
    let rest    = json[start..].trim();
    if !rest.starts_with('{') { return None; }
    let end = rest.find('}')? + 1;
    let obj = &rest[1..end-1]; // strip { }

    let mut map = std::collections::HashMap::new();
    // Parse "name":value pairs
    let mut s = obj;
    loop {
        s = s.trim_start_matches([',', ' ', '\n', '\r']);
        if s.is_empty() || !s.starts_with('"') { break; }
        let k_end  = s[1..].find('"')? + 1;
        let k_name = s[1..k_end].to_string();
        let after_colon = s[k_end+1..].trim_start();
        if !after_colon.starts_with(':') { break; }
        let val_str = after_colon[1..].trim_start();
        // Read until comma or }
        let val_end = val_str.find([',', '}']).unwrap_or(val_str.len());
        let val: f64 = val_str[..val_end].trim().parse().unwrap_or(0.0);
        map.insert(k_name, val);
        s = &val_str[val_end..];
    }
    Some(map)
}

// ═══════════════════════════════════════════════════════════════════════
// EXPLANATION ENGINE — per-step human-readable explanations
// ═══════════════════════════════════════════════════════════════════════

fn explain_step(
    step: &str,
    oracle: &str,
    value: f64,
    inputs: &std::collections::HashMap<String, f64>,
    prev_steps: &[plan::StepResult],
) -> String {
    let get = |k: &str| -> f64 {
        inputs.get(k).copied()
            .or_else(|| prev_steps.iter().find(|s| s.step == k).map(|s| s.value))
            .unwrap_or(0.0)
    };
    match oracle {
        "hp_oracle" | "pump_hp_oracle" | "hydraulic_hp" | "hp_required" => {
            let q = get("Q_gpm"); let p = get("P_psi"); let eff = get("eff");
            format!("{} = Q\u{00d7}P/(3960\u{00d7}eff) = {:.0}\u{00d7}{:.0}/(3960\u{00d7}{:.2}) = {:.2} HP",
                step, q, p, eff, value)
        }
        "shutoff_pressure" | "shutoff_p" => {
            let p = get("P_psi");
            format!("{} = P\u{00d7}1.4 = {:.0}\u{00d7}1.4 = {:.1} PSI (NFPA 20 churn limit)", step, p, value)
        }
        "flow_at_150pct" | "flow_150pct" => {
            let q = get("Q_gpm");
            format!("{} = Q\u{00d7}1.5 = {:.0}\u{00d7}1.5 = {:.0} GPM (150% overrun test point)", step, q, value)
        }
        "password_entropy" => {
            let cs = get("charset_size"); let len = get("length");
            format!("{} = len\u{00d7}log2(charset) = {:.0}\u{00d7}log2({:.0}) = {:.2} bits",
                step, len, cs, value)
        }
        "brute_force_years" | "crack_years" => {
            format!("{} = 2^entropy / rate / 31536000 = {:.2e} years to exhaust keyspace", step, value)
        }
        "bcrypt_hashrate" | "effective_rate" => {
            let cost = get("cost_factor"); let base = get("base_gpu_rate");
            format!("{} = base/2^(cost-5) = {:.0}/2^{:.0} = {:.0} hash/s (RTX 4090 at bcrypt cost {})",
                step, base, (cost - 5.0).max(0.0), value, cost as i64)
        }
        "igv_oracle" | "igv_amount" | "igv18" => {
            let base = get("base_imponible");
            format!("{} = base\u{00d7}18% = {:.2}\u{00d7}0.18 = {:.2} PEN (IGV Peru)", step, base, value)
        }
        "total_oracle" | "total" => {
            let base = get("base_imponible");
            let igv = inputs.get("igv_amount").copied()
                .or_else(|| prev_steps.iter().find(|s| s.step == "igv_amount").map(|s| s.value))
                .unwrap_or(0.0);
            format!("{} = base + IGV = {:.2} + {:.2} = {:.2} PEN", step, base, igv, value)
        }
        "current_oracle" | "amps_oracle" | "I_load" => {
            let p = get("P_w"); let v = get("V"); let pf = get("pf");
            format!("{} = P/(V\u{00d7}pf) = {:.0}/({:.0}\u{00d7}{:.2}) = {:.2} A", step, p, v, pf, value)
        }
        "bcrypt_crack_seconds" | "brute_seconds" => {
            let rate = get("effective_rate");
            let ks = get("keyspace_total").max(1.0);
            format!("{} = keyspace / rate = {:.2e} / {:.0}/s = {:.2}s", step, ks, rate, value)
        }
        "dict_crack_seconds" | "dict_seconds" => {
            let rate = get("effective_rate");
            let ds = get("dict_size");
            format!("{} = dict_size / rate = {:.0} / {:.0}/s = {:.2}s", step, ds, rate, value)
        }
        "secs_to_years" | "brute_years" => {
            format!("{} = seconds / 31,536,000 = {:.2} years", step, value)
        }
        "secs_to_hours" | "dict_hours" => {
            format!("{} = seconds / 3600 = {:.2} hours", step, value)
        }
        "keyspace" | "keyspace_total" => {
            let cs = get("charset_size"); let len = get("length");
            format!("{} = charset^length = {:.0}^{:.0} = {:.2e} combinations", step, cs, len, value)
        }
        "essalud_oracle" | "essalud" => {
            let s = get("sueldo");
            format!("{} = sueldo\u{00d7}9% = {:.2}\u{00d7}0.09 = {:.2} PEN (EsSalud employer)", step, s, value)
        }
        "aes_key_strength" | "strength_ratio" => {
            let ent = inputs.get("entropy")
                .copied()
                .or_else(|| prev_steps.iter().find(|s| s.step == "entropy").map(|s| s.value))
                .unwrap_or(0.0);
            format!("{} = entropy/256\u{00d7}10 = {:.1}/256\u{00d7}10 = {:.2}/10 (AES-256=10)", step, ent, value)
        }
        _ => {
            let input_str: Vec<String> = inputs.iter().take(4)
                .map(|(k, v)| format!("{}={:.2}", k, v)).collect();
            format!("{} via {} = {:.4}", step, oracle, value)
        }
    }
}

fn steps_with_explanations(
    steps: &[plan::StepResult],
    inputs: &std::collections::HashMap<String, f64>,
) -> String {
    let parts: Vec<String> = steps.iter().map(|s| {
        let exp = explain_step(&s.step, &s.oracle, s.value, inputs, steps);
        let exp_escaped = exp.replace('\\', "\\\\").replace('"', "\\\"");
        format!(
            r#"{{"step":"{}","oracle":"{}","result":{:.6},"latency_ns":{:.1},"explanation":"{}"}}"#,
            s.step, s.oracle, if s.value.is_finite() { s.value } else if s.value > 0.0 { 1e99_f64 } else { -1e99_f64 }, s.latency_ns, exp_escaped
        )
    }).collect();
    format!("[{}]", parts.join(","))
}

// ═══════════════════════════════════════════════════════════════════════
// PIPELINE ENGINE — chains compute → optimize → decide → explain
// ═══════════════════════════════════════════════════════════════════════

fn parse_pipeline_stages(body: &str) -> Vec<String> {
    if let Some(start) = body.find("\"stages\"") {
        let after = &body[start + 8..];
        if let Some(arr_start) = after.find('[') {
            let arr_portion = &after[arr_start + 1..];
            if let Some(arr_end) = arr_portion.find(']') {
                return arr_portion[..arr_end]
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    vec!["compute".to_string(), "decide".to_string(), "explain".to_string()]
}

fn extract_nested_str(body: &str, outer_key: &str, inner_key: &str) -> Option<String> {
    let key = format!("\"{}\"", outer_key);
    let start = body.find(&key)?;
    let after_outer = &body[start + key.len()..];
    let obj_start = after_outer.find('{')?;
    let obj_body = &after_outer[obj_start..];
    let mut depth = 0usize;
    let mut end_pos = 0;
    for (i, c) in obj_body.char_indices() {
        if c == '{' { depth += 1; }
        if c == '}' { depth -= 1; if depth == 0 { end_pos = i + 1; break; } }
    }
    extract_json_str(&obj_body[..end_pos], inner_key)
}

fn run_optimize_sweep(
    body: &str,
    plan_name: &str,
    target: &str,
    minimize: bool,
    base_params: &std::collections::HashMap<String, f64>,
    plans: &[PlanDecl],
    jit_map: &Option<plan::JitFnMap>,
) -> (usize, f64, f64, Option<std::collections::HashMap<String, f64>>) {
    // Extract the "optimize":{...} object from body
    let opt_key = "\"optimize\"";
    let opt_start = match body.find(opt_key) {
        Some(p) => p, None => return (0, 0.0, 0.0, None),
    };
    let after_opt = &body[opt_start + opt_key.len()..];
    let obj_start = match after_opt.find('{') {
        Some(p) => p, None => return (0, 0.0, 0.0, None),
    };
    let obj_body = &after_opt[obj_start..];
    let mut depth = 0usize; let mut end_pos = obj_body.len();
    for (i, c) in obj_body.char_indices() {
        if c == '{' { depth += 1; }
        if c == '}' { depth -= 1; if depth == 0 { end_pos = i + 1; break; } }
    }
    let opt_obj = &obj_body[..end_pos];

    // Extract "sweep":{...} from opt_obj
    let sweep_key = "\"sweep\"";
    let sweep_start = match opt_obj.find(sweep_key) {
        Some(p) => p, None => return (0, 0.0, 0.0, None),
    };
    let after_sweep = &opt_obj[sweep_start + sweep_key.len()..];
    let sw_start = match after_sweep.find('{') {
        Some(p) => p, None => return (0, 0.0, 0.0, None),
    };
    let sw_body = &after_sweep[sw_start..];
    let mut sw_depth = 0usize; let mut sw_end = sw_body.len();
    for (i, c) in sw_body.char_indices() {
        if c == '{' { sw_depth += 1; }
        if c == '}' { sw_depth -= 1; if sw_depth == 0 { sw_end = i + 1; break; } }
    }
    let sweep_body = &sw_body[..sw_end];

    // Parse sweep params: {"Q_gpm":{"min":400,"max":800,"step":50},"P_psi":80}
    let mut param_lists: Vec<(String, Vec<f64>)> = Vec::new();
    let mut pos = 1usize;
    while pos < sweep_body.len().saturating_sub(1) {
        let quote_pos = match sweep_body[pos..].find('"') {
            Some(p) => pos + p, None => break,
        };
        let key_start = quote_pos + 1;
        let key_end = match sweep_body[key_start..].find('"') {
            Some(p) => key_start + p, None => break,
        };
        let key = sweep_body[key_start..key_end].to_string();
        let colon_pos = match sweep_body[key_end + 1..].find(':') {
            Some(p) => key_end + 1 + p + 1, None => break,
        };
        pos = colon_pos;
        while pos < sweep_body.len() && matches!(sweep_body.as_bytes()[pos], b' ' | b'\n' | b'\r' | b'\t') {
            pos += 1;
        }
        if pos >= sweep_body.len() { break; }

        if sweep_body.as_bytes()[pos] == b'{' {
            let op = &sweep_body[pos..];
            let mut d = 0usize; let mut e = 0;
            for (i, c) in op.char_indices() {
                if c == '{' { d += 1; } if c == '}' { d -= 1; if d == 0 { e = i + 1; break; } }
            }
            let rng = &op[..e];
            let min_v = extract_json_float(rng, "min").unwrap_or(0.0);
            let max_v = extract_json_float(rng, "max").unwrap_or(1.0);
            let step_v = extract_json_float(rng, "step").unwrap_or(1.0);
            let mut vals = Vec::new();
            let mut v = min_v;
            while v <= max_v + 1e-9 { vals.push(v); v += step_v; }
            param_lists.push((key, vals));
            pos += e;
        } else {
            let num_start = pos;
            while pos < sweep_body.len() && !matches!(sweep_body.as_bytes()[pos], b',' | b'}') { pos += 1; }
            if let Ok(v) = sweep_body[num_start..pos].trim().parse::<f64>() {
                param_lists.push((key, vec![v]));
            }
        }
        if pos < sweep_body.len() && sweep_body.as_bytes()[pos] == b',' { pos += 1; }
    }

    // Check that at least one param has multiple values
    if param_lists.iter().all(|(_, v)| v.len() <= 1) { return (0, 0.0, 0.0, None); }

    // Cartesian product
    let mut combos: Vec<std::collections::HashMap<String, f64>> = vec![std::collections::HashMap::new()];
    for (key, vals) in &param_lists {
        let mut nc = Vec::new();
        for combo in &combos {
            for &v in vals {
                let mut c = combo.clone(); c.insert(key.clone(), v); nc.push(c);
            }
        }
        combos = nc;
    }
    let scenarios = combos.len();

    let executor = {
        let ex = plan::PlanExecutor::new(plans);
        match jit_map { Some(map) => ex.with_jit_map(map.clone()), None => ex }
    };

    let t_opt = std::time::Instant::now();
    let mut best_val: f64 = if minimize { f64::MAX } else { f64::MIN };
    let mut best_params: Option<std::collections::HashMap<String, f64>> = None;

    for combo in combos {
        let mut run_params = base_params.clone();
        run_params.extend(combo);
        if let Ok(r) = executor.execute(plan_name, run_params.clone()) {
            let tv = r.steps.iter().find(|s| s.step == target)
                .or_else(|| r.steps.last()).map(|s| s.value).unwrap_or(0.0);
            let better = if minimize { tv < best_val } else { tv > best_val };
            if better { best_val = tv; best_params = Some(run_params); }
        }
    }
    let opt_ns = t_opt.elapsed().as_nanos() as f64;
    (scenarios, opt_ns, best_val, best_params)
}

fn build_pipeline_summary(
    plan_name: &str,
    stages: &[String],
    compute_result: &Option<plan::PlanResult>,
    stage_outputs: &[String],
) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(cr) = compute_result {
        if let Some(last) = cr.steps.last() {
            parts.push(format!("{}: {:.2}", last.step.replace('_', " "), last.value));
        }
    }
    for out in stage_outputs {
        if out.contains(r#""stage":"optimize""#) {
            if let Some(p) = out.find(r#""best_value":"#) {
                let after = &out[p + 13..];
                let end = after.find(|c: char| c == ',' || c == '}').unwrap_or(after.len());
                if let Ok(bv) = after[..end].trim().parse::<f64>() {
                    parts.push(format!("optimized to {:.4}", bv));
                }
            }
            if let Some(p) = out.find(r#""improvement_pct":"#) {
                let after = &out[p + 18..];
                let end = after.find(|c: char| c == ',' || c == '}').unwrap_or(after.len());
                if let Ok(pct) = after[..end].trim().parse::<f64>() {
                    parts.push(format!("{:.1}% improvement", pct));
                }
            }
        }
        if out.contains(r#""stage":"decide"\\"#) {
            if let Some(p) = out.find(r#""status":""#) {
                let after = &out[p + 10..];
                let end = after.find('"').unwrap_or(after.len());
                parts.push(format!("status: {}", &after[..end]));
            }
        }
    }
    if parts.is_empty() { format!("Pipeline complete for {}", plan_name) }
    else { parts.join(". ") }
}

fn handle_pipeline(
    body: &str,
    plans: &[PlanDecl],
    jit_map: &Option<plan::JitFnMap>,
) -> (&'static str, String) {
    let plan_name = match extract_json_str(body, "plan") {
        Some(n) => n,
        None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan'"}"#.into()),
    };
    let base_params = extract_json_obj_float(body, "params").unwrap_or_default();
    let stages = parse_pipeline_stages(body);

    let t0 = std::time::Instant::now();
    let mut stage_outputs: Vec<String> = Vec::new();
    let mut compute_result: Option<plan::PlanResult> = None;
    let mut active_params = base_params.clone();

    let make_executor = || {
        let ex = plan::PlanExecutor::new(plans);
        match jit_map { Some(map) => ex.with_jit_map(map.clone()), None => ex }
    };

    for stage in &stages {
        match stage.as_str() {
            "compute" => {
                match make_executor().execute(&plan_name, active_params.clone()) {
                    Ok(r) => {
                        stage_outputs.push(format!(
                            r#"{{"stage":"compute","ns":{:.0},"result":{}}}"#,
                            r.total_ns, r.to_json()
                        ));
                        compute_result = Some(r);
                    }
                    Err(e) => return ("400 Bad Request",
                        format!(r#"{{"ok":false,"stage":"compute","error":"{}"}}"#, e)),
                }
            }
            "decide" => {
                match compute_result.as_ref() {
                    Some(cr) => {
                        let steps_data: Vec<(String, String, f64)> = cr.steps.iter()
                            .map(|s| (s.step.clone(), s.oracle.clone(), s.value)).collect();
                        let dec = decision_analyze(&plan_name, &steps_data, cr.total_ns as u64);
                        stage_outputs.push(format!(r#"{{"stage":"decide","result":{}}}"#, dec));
                    }
                    None => stage_outputs.push(r#"{"stage":"decide","error":"run compute first"}"#.into()),
                }
            }
            "explain" => {
                match compute_result.as_ref() {
                    Some(cr) => {
                        let explained = steps_with_explanations(&cr.steps, &active_params);
                        stage_outputs.push(format!(r#"{{"stage":"explain","steps":{}}}"#, explained));
                    }
                    None => stage_outputs.push(r#"{"stage":"explain","error":"run compute first"}"#.into()),
                }
            }
            "optimize" => {
                let orig_val = compute_result.as_ref()
                    .and_then(|cr| {
                        let target_hint = extract_nested_str(body, "optimize", "target")
                            .or_else(|| cr.steps.last().map(|s| s.step.clone()))
                            .unwrap_or_default();
                        cr.steps.iter().find(|s| s.step == target_hint).map(|s| s.value)
                    }).unwrap_or(0.0);

                let target = extract_nested_str(body, "optimize", "target")
                    .or_else(|| compute_result.as_ref()
                        .and_then(|cr| cr.steps.last().map(|s| s.step.clone())))
                    .unwrap_or_else(|| "result".to_string());
                let minimize = body.contains(r#""minimize":true"#) || body.contains(r#""minimize": true"#);

                let (scenarios, opt_ns, best_val, best_params_opt) =
                    run_optimize_sweep(body, &plan_name, &target, minimize, &active_params, plans, jit_map);

                if let Some(best_p) = best_params_opt {
                    let improvement = if orig_val.abs() > 1e-9 {
                        (best_val - orig_val) / orig_val.abs() * 100.0
                    } else { 0.0 };
                    let direction = if minimize { "reduced" } else { "increased" };

                    let changed: Vec<String> = best_p.iter()
                        .filter(|(k, v)| active_params.get(*k).map(|old| (old - *v).abs() > 1e-9).unwrap_or(true))
                        .map(|(k, v)| format!(r#""{}":{:.4}"#, k, v)).collect();

                    active_params = best_p.clone();
                    // Re-run with optimal params so downstream stages use best result
                    if let Ok(r) = make_executor().execute(&plan_name, active_params.clone()) {
                        compute_result = Some(r);
                    }

                    stage_outputs.push(format!(
                        r#"{{"stage":"optimize","target":"{}","minimize":{},"scenarios":{},"best_value":{:.4},"original_value":{:.4},"improvement_pct":{:.1},"direction":"{}","changed_params":{{{}}},"ns":{:.0}}}"#,
                        target, minimize, scenarios, best_val, orig_val, improvement.abs(),
                        direction, changed.join(","), opt_ns
                    ));
                } else {
                    stage_outputs.push(format!(
                        r#"{{"stage":"optimize","error":"no sweep params in optimize config","target":"{}"}}"#,
                        target
                    ));
                }
            }
            _ => {
                stage_outputs.push(format!(r#"{{"stage":"{}","error":"unknown stage"}}"#, stage));
            }
        }
    }

    let total_ns = t0.elapsed().as_nanos();
    let summary = build_pipeline_summary(&plan_name, &stages, &compute_result, &stage_outputs);

    ("200 OK", format!(
        r#"{{"ok":true,"plan":"{}","stages":[{}],"summary":"{}","total_ns":{}}}"#,
        plan_name,
        stage_outputs.join(","),
        summary.replace('"', "'"),
        total_ns
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 5: POLICY ENGINE — domain-aware action governance
// ═══════════════════════════════════════════════════════════════════════

fn policy_gate(risk: f64, confidence: f64, domain: &str) -> &'static str {
    if confidence < 0.65 { return "BLOCKED"; }
    let max_auto = match domain {
        "fire_protection"           => 2.5,
        "cybersecurity" | "security"=> 2.0,
        "electrical"                => 3.0,
        "finance"                   => 3.5,
        _                           => 4.5,
    };
    if risk > 7.5      { "CONFIRM" }
    else if risk > max_auto { "SUGGEST" }
    else               { "AUTO" }
}

fn policy_reason(mode: &str, risk: f64, conf: f64, domain: &str) -> String {
    match mode {
        "BLOCKED" => format!(
            "Confidence {:.0}% below 65% minimum. Insufficient data for safe action in {} domain.",
            conf * 100.0, domain),
        "CONFIRM" => format!(
            "Risk score {:.1} exceeds critical threshold. Human approval required before execution.",
            risk),
        "SUGGEST" => format!(
            "Risk {:.1} in {} domain. Recommendation provided — human review advised.",
            risk, domain),
        _ => format!(
            "Low risk ({:.1}), confidence {:.0}%. Safe to execute automatically.",
            risk, conf * 100.0),
    }
}

fn policy_enhance(decision_json: &str, plan_name: &str, params_body: &str) -> String {
    // Extract risk_score from decision.risk_score
    let risk = {
        let dec_pos = decision_json.find(r#""decision":{"#);
        dec_pos.and_then(|p| {
            let after = &decision_json[p..];
            extract_json_float(after, "risk_score")
        }).unwrap_or(5.0)
    };

    let domain = extract_json_str(decision_json, "domain")
        .unwrap_or_else(|| "general".into());

    // Confidence: calibrate from step count and risk
    let confidence = if risk < 3.0 { 0.90 } else if risk < 6.0 { 0.80 } else { 0.70 };

    let mode = policy_gate(risk, confidence, &domain);
    let reason = policy_reason(mode, risk, confidence, &domain);

    // Hash params for audit trail
    let mut h = 0u64;
    for b in params_body.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
    audit_write(plan_name, mode, risk, confidence, h);

    // Build evidence and tradeoffs based on status
    let evidence = if domain == "fire_protection" {
        r#"["NFPA 20 Section 6.4.1 (rated flow)", "NFPA 20 Section 6.4.2 (shutoff)", "NFPA 20 Section 6.4.3 (overrun)"]"#
    } else if domain == "electrical" {
        r#"["NEC Article 310 (conductor sizing)", "NEC Article 240 (overcurrent protection)"]"#
    } else if domain == "cybersecurity" || domain == "security" {
        r#"["NIST SP 800-63B (password entropy)", "OWASP Authentication Guidelines"]"#
    } else {
        r#"["Calculated from plan formulas"]"#
    };

    let actions_json = match mode {
        "AUTO"    => r#"["generate_report","export_pdf","apply_config"]"#,
        "SUGGEST" => r#"["generate_report","export_pdf"]"#,
        "CONFIRM" => r#"["generate_report","request_approval"]"#,
        _         => r#"["collect_more_data"]"#,
    };

    // Inject policy block before final closing brace
    let policy_block = format!(
        r#","policy":{{"mode":"{}","confidence":{:.2},"risk_score":{:.2},"reason":"{}","evidence":{},"allowed_actions":{}}}"#,
        mode, confidence, risk,
        reason.replace('"', "'"),
        evidence,
        actions_json
    );

    if decision_json.ends_with('}') {
        format!("{}{}}}", &decision_json[..decision_json.len()-1], policy_block)
    } else {
        format!("{}{}", decision_json, policy_block)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 5: AUDIT LOG — immutable append-only decision trail
// ═══════════════════════════════════════════════════════════════════════

fn audit_write(plan: &str, mode: &str, risk: f64, confidence: f64, params_hash: u64) {
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs();
    let entry = format!(
        r#"{{"ts":{},"plan":"{}","mode":"{}","risk":{:.3},"confidence":{:.3},"params_hash":"{:016x}"}}"#,
        ts, plan, mode, risk, confidence, params_hash
    );
    let _ = std::fs::create_dir_all("/opt/crysl/data");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("/opt/crysl/data/audit.jsonl") {
        let _ = writeln!(f, "{}", entry);
    }
}

fn audit_read_recent(n: usize) -> String {
    use std::io::{BufRead, BufReader};
    let file = match std::fs::File::open("/opt/crysl/data/audit.jsonl") {
        Ok(f) => f,
        Err(_) => return "[]".into(),
    };
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .collect();
    let total = lines.len();
    let start = if total > n { total - n } else { 0 };
    let entries: Vec<&str> = lines[start..].iter().map(|s| s.as_str()).collect();
    format!("[{}]", entries.join(","))
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 5: DIGITAL TWIN — real-time state management
// ═══════════════════════════════════════════════════════════════════════



fn twin_get() -> String {
    let state = twin_state_map();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs();
    let pairs: Vec<String> = state.iter()
        .map(|e| format!(r#""{}":{:.4}"#, e.key(), e.value().0))
        .collect();
    let avg_conf: f32 = {
        let cm = twin_confidence_map();
        if cm.is_empty() { 1.0 } else {
            cm.iter().map(|e| *e.value()).sum::<f32>() / cm.len() as f32
        }
    };
    format!(r#"{{"vars":{{{}}},"confidence":{:.3},"ts":{}}}"#,
        pairs.join(","), avg_conf, ts)
}

fn twin_update_returning(body: &str) -> String {
    // Convert simple {"vars":{"k":v}} to twin_update's {"variables":{"k":{"value":v,"unit":""}}}
    if let Some(vars) = extract_json_obj_float(body, "vars") {
        let var_entries: Vec<String> = vars.iter()
            .map(|(k, v)| format!(r#""{}":{{"value":{:.4},"unit":""}}"#, k, v)).collect();
        let converted = format!(r#"{{"variables":{{{}}}}}"#, var_entries.join(","));
        twin_update(&converted);
    }
    twin_get()
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 6: MULTI-STRATEGY OPTIMIZER — parallel strategies + scoring
// ═══════════════════════════════════════════════════════════════════════
//
// Request format:
// {
//   "plan": "plan_pump_sizing",
//   "params": {"Q_gpm": 500, "P_psi": 80, "eff": 0.7},
//   "strategies": [
//     {"name":"min_hp","target":"hp_required","minimize":true,
//      "sweep":{"Q_gpm":{"min":300,"max":700,"step":50}}},
//     {"name":"max_safety","target":"shutoff_p","minimize":false,
//      "sweep":{"P_psi":{"min":60,"max":150,"step":10}}}
//   ],
//   "score_weights": {"efficiency": 0.5, "safety": 0.3, "cost": 0.2}
// }

fn parse_strategy_block<'a>(body: &'a str, idx: usize) -> Option<(String, String, bool, String)> {
    // Find the idx-th object in the "strategies" array
    let arr_key = "\"strategies\"";
    let arr_pos = body.find(arr_key)?;
    let after_arr = &body[arr_pos + arr_key.len()..];
    let arr_start = after_arr.find('[')?;
    let arr_body = &after_arr[arr_start + 1..];

    // Count through objects
    let mut depth = 0i32;
    let mut obj_count = 0usize;
    let mut obj_start = None;
    let mut obj_end = None;

    for (i, c) in arr_body.char_indices() {
        match c {
            '{' => {
                if depth == 0 {
                    if obj_count == idx { obj_start = Some(i); }
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if obj_count == idx {
                        obj_end = Some(i + 1);
                        break;
                    }
                    obj_count += 1;
                }
            }
            _ => {}
        }
    }

    let (start, end) = (obj_start?, obj_end?);
    let strat_body = &arr_body[start..end];

    let name    = extract_json_str(strat_body, "name").unwrap_or_else(|| format!("strategy_{}", idx));
    let target  = extract_json_str(strat_body, "target").unwrap_or_else(|| "result".into());
    let minimize = strat_body.contains("\"minimize\":true") || strat_body.contains("\"minimize\": true");

    // Extract sweep sub-object from strategy body
    let sweep_key = "\"sweep\"";
    let sweep_body = if let Some(sp) = strat_body.find(sweep_key) {
        let after = &strat_body[sp + sweep_key.len()..];
        if let Some(os) = after.find('{') {
            let sb = &after[os..];
            let mut d = 0usize; let mut e = sb.len();
            for (i, c) in sb.char_indices() {
                if c == '{' { d += 1; } if c == '}' { d -= 1; if d == 0 { e = i + 1; break; } }
            }
            sb[..e].to_string()
        } else { "{}".into() }
    } else { "{}".into() };

    Some((name, target, minimize, sweep_body))
}

fn count_strategies(body: &str) -> usize {
    let arr_key = "\"strategies\"";
    let arr_pos = match body.find(arr_key) { Some(p) => p, None => return 0 };
    let after_arr = &body[arr_pos + arr_key.len()..];
    let arr_start = match after_arr.find('[') { Some(p) => p, None => return 0 };
    let arr_body = &after_arr[arr_start + 1..];

    let mut depth = 0i32;
    let mut count = 0usize;
    for c in arr_body.chars() {
        match c {
            '[' => depth += 1,
            '{' => { if depth == 0 { count += 1; } depth += 1; }
            '}' | ']' => { depth -= 1; if depth < 0 { break; } }
            _ => {}
        }
    }
    count
}

fn score_strategy(value: f64, target: &str, minimize: bool) -> f64 {
    // Normalize to [0,1] score. Higher = better.
    // For "minimize" strategies: lower value = higher score
    // We cap at reasonable ranges per domain
    let raw = match target {
        t if t.contains("hp") => { let cap = 50.0f64; if minimize { 1.0 - (value / cap).min(1.0) } else { (value / cap).min(1.0) } }
        t if t.contains("psi") || t.contains("pressure") => { let cap = 200.0f64; if minimize { 1.0 - (value / cap).min(1.0) } else { (value / cap).min(1.0) } }
        t if t.contains("year") || t.contains("years") => { let cap = 1e10f64; if minimize { 1.0 - (value / cap).log10().abs() / 10.0 } else { (value / cap).log10().abs() / 10.0 } }
        t if t.contains("entropy") || t.contains("bits") => { let cap = 256.0f64; (value / cap).min(1.0) }
        _ => { if minimize { 1.0 / (1.0 + value.abs()) } else { value.tanh() } }
    };
    raw.clamp(0.0, 1.0)
}

fn handle_multi_strategy(
    body: &str,
    plans: &[PlanDecl],
    jit_map: &Option<plan::JitFnMap>,
) -> (&'static str, String) {
    let plan_name = match extract_json_str(body, "plan") {
        Some(n) => n,
        None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'plan'"}"#.into()),
    };
    let base_params = extract_json_obj_float(body, "params").unwrap_or_default();
    let n_strategies = count_strategies(body);

    if n_strategies == 0 {
        return ("400 Bad Request", r#"{"ok":false,"error":"no strategies found"}"#.into());
    }

    let executor = {
        let ex = plan::PlanExecutor::new(plans);
        match jit_map { Some(map) => ex.with_jit_map(map.clone()), None => ex }
    };

    let t0 = std::time::Instant::now();
    let mut strategy_results: Vec<String> = Vec::new();
    let mut best_score = f64::MIN;
    let mut best_strategy = String::new();
    let mut best_params_out = base_params.clone();

    for i in 0..n_strategies {
        if let Some((name, target, minimize, sweep_body)) = parse_strategy_block(body, i) {
            // Build a synthetic optimize body using the sweep_body
            let fake_body = format!(
                r#"{{"plan":"{}","params":{{}},"optimize":{{"target":"{}","minimize":{},"sweep":{}}}}}"#,
                plan_name, target, minimize, sweep_body
            );
            let (scenarios, opt_ns, best_val, best_p) =
                run_optimize_sweep(&fake_body, &plan_name, &target, minimize, &base_params, plans, jit_map);

            let score = score_strategy(best_val, &target, minimize);

            if score > best_score {
                best_score = score;
                best_strategy = name.clone();
                if let Some(ref p) = best_p {
                    best_params_out = p.clone();
                }
            }

            let changed: Vec<String> = best_p.as_ref().map(|p| {
                p.iter()
                    .filter(|(k, v)| base_params.get(*k).map(|old| (old - *v).abs() > 1e-9).unwrap_or(true))
                    .map(|(k, v)| format!(r#""{}":{:.4}"#, k, v)).collect()
            }).unwrap_or_default();

            strategy_results.push(format!(
                r#"{{"name":"{}","target":"{}","minimize":{},"scenarios":{},"best_value":{:.4},"score":{:.3},"changed_params":{{{}}},"ns":{:.0}}}"#,
                name, target, minimize, scenarios, best_val, score,
                changed.join(","), opt_ns
            ));
        }
    }

    // Execute with best params and get explanation
    let final_result_json = if let Ok(r) = executor.execute(&plan_name, best_params_out.clone()) {
        let explained = steps_with_explanations(&r.steps, &best_params_out);
        format!(r#"{{"steps":{},"total_ns":{:.0}}}"#, explained, r.total_ns)
    } else {
        "{}".into()
    };

    // Policy check on best result
    let policy_mode = policy_gate(3.0 - best_score * 3.0, 0.85, "general");
    let total_ns = t0.elapsed().as_nanos();

    ("200 OK", format!(
        r#"{{"ok":true,"plan":"{}","winner":"{}","winner_score":{:.3},"strategies":[{}],"best_result":{},"policy":{{"mode":"{}","confidence":0.85}},"total_ns":{}}}"#,
        plan_name, best_strategy, best_score,
        strategy_results.join(","),
        final_result_json, policy_mode, total_ns
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 8: CLUSTER HEALTH — multi-node status
// ═══════════════════════════════════════════════════════════════════════

fn cluster_health_json() -> (&'static str, String) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs();

    // Count audit entries for activity metric
    let audit_count = std::fs::read_to_string("/opt/crysl/data/audit.jsonl")
        .map(|s| s.lines().count())
        .unwrap_or(0);

    let twin = twin_get();
    let twin_conf: f32 = {
        let cm = twin_confidence_map();
        if cm.is_empty() { 1.0 } else {
            cm.iter().map(|e| *e.value()).sum::<f32>() / cm.len() as f32
        }
    };

    ("200 OK", format!(
        r#"{{"ok":true,"cluster":{{"nodes":[{{"id":"server5","role":"primary","status":"healthy","compute":"CRYS-L v3.1","features":207,"ts":{}}}],"consensus":"single-node","policy_version":"v1.0"}},"twin_confidence":{:.3},"twin":{},"audit":{{"total_decisions":{}}},"slo":{{"compute_target_ns":1000,"simulate_target_ms":1,"concurrent_error_rate":0.0}}}}"#,
        ts, twin_conf, twin, audit_count
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 6: SLO METRICS — request timing ring buffer
// ═══════════════════════════════════════════════════════════════════════

static METRICS_RING: std::sync::OnceLock<std::sync::Mutex<std::collections::VecDeque<(u64, bool)>>> =
    std::sync::OnceLock::new();
static REQUEST_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERROR_COUNT:   std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn record_metric(latency_ns: u64, ok: bool) {
    let ring = METRICS_RING.get_or_init(|| {
        std::sync::Mutex::new(std::collections::VecDeque::with_capacity(1001))
    });
    if let Ok(mut r) = ring.lock() {
        if r.len() >= 1000 { r.pop_front(); }
        r.push_back((latency_ns, ok));
    }
    REQUEST_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !ok { ERROR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
}

fn slo_stats() -> (f64, f64, f64, f64, u64, u64) {
    // returns (p50_us, p95_us, p99_us, error_rate, total_reqs, total_errors)
    let total = REQUEST_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let errors = ERROR_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let ring = METRICS_RING.get_or_init(|| {
        std::sync::Mutex::new(std::collections::VecDeque::with_capacity(1001))
    });
    let r = ring.lock().unwrap_or_else(|p| p.into_inner());
    if r.is_empty() { return (0.0, 0.0, 0.0, 0.0, total, errors); }
    let mut lats: Vec<f64> = r.iter().map(|(l, _)| *l as f64 / 1000.0).collect();
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = lats.len();
    let err_count = r.iter().filter(|(_, ok)| !ok).count();
    (
        lats[n / 2],
        lats[(n * 95 / 100).min(n - 1)],
        lats[(n * 99 / 100).min(n - 1)],
        err_count as f64 / n as f64,
        total, errors
    )
}

fn health_detailed_json() -> (&'static str, String) {
    let (p50, p95, p99, err_rate, total, errors) = slo_stats();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs();

    // SLO evaluation
    let compute_ok  = p99 < 1000.0;   // 1ms p99
    let error_ok    = err_rate < 0.01; // <1% errors
    let live_active = LIVE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);

    let audit_count = std::fs::read_to_string("/opt/crysl/data/audit.jsonl")
        .map(|s| s.lines().count()).unwrap_or(0);
    let live_actions = std::fs::read_to_string("/opt/crysl/data/live_actions.jsonl")
        .map(|s| s.lines().count()).unwrap_or(0);

    ("200 OK", format!(
        r#"{{"ok":true,"ts":{},"slo":{{"compute_p50_us":{:.1},"compute_p95_us":{:.1},"compute_p99_us":{:.1},"error_rate":{:.4},"compute_slo_ok":{},"error_slo_ok":{}}},"requests":{{"total":{},"errors":{},"last_1000_sampled":true}},"watchdog":{{"status":"healthy","autonomous_loop":{},"live_actions":{}}},"audit":{{"total_decisions":{}}}}}"#,
        ts, p50, p95, p99, err_rate,
        compute_ok, error_ok,
        total, errors,
        live_active, live_actions,
        audit_count
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 6: SMART WATCHDOG V2 — degredation detection + alerts
// ═══════════════════════════════════════════════════════════════════════

fn check_slo_health() -> Vec<String> {
    let (_, _, p99, err_rate, _, _) = slo_stats();
    let mut alerts = Vec::new();
    if p99 > 50000.0 { // >50ms p99
        alerts.push(format!("p99 latency degraded: {:.0}µs (target: <1000µs)", p99));
    }
    if err_rate > 0.05 { // >5% errors
        alerts.push(format!("error rate elevated: {:.1}% (target: <1%)", err_rate * 100.0));
    }
    alerts
}

fn write_alert(alerts: &[String]) {
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs();
    let msgs: Vec<String> = alerts.iter().map(|a| format!("\"{}\"", a.replace('"', "'"))).collect();
    let entry = format!(r#"{{"ts":{},"type":"slo_alert","alerts":[{}]}}"#, ts, msgs.join(","));
    let _ = std::fs::create_dir_all("/opt/crysl/data");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("/opt/crysl/data/alerts.jsonl") {
        let _ = writeln!(f, "{}", entry);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 6: CLOSED-LOOP VERIFY — compare prediction to twin state
// ═══════════════════════════════════════════════════════════════════════

fn verify_prediction(body: &str) -> (&'static str, String) {
    // Body: {"step":"hp_required","expected":14.43,"tolerance_pct":5.0}
    let step = match extract_json_str(body, "step") {
        Some(s) => s,
        None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'step'"}"#.into()),
    };
    let expected = match extract_json_float(body, "expected") {
        Some(v) => v,
        None => return ("400 Bad Request", r#"{"ok":false,"error":"missing 'expected'"}"#.into()),
    };
    let tolerance = extract_json_float(body, "tolerance_pct").unwrap_or(5.0) / 100.0;

    // Look up step in twin state
    let state = twin_state_map();
    let actual = state.get(&step).map(|e| e.value().0);

    if let Some(obs) = actual {
        let delta_pct = (obs - expected).abs() / expected.abs().max(1e-9) * 100.0;
        let verified = delta_pct <= tolerance * 100.0;
        let confidence_adj = if verified { 0.05 } else { -0.10 };

        // Update twin confidence
        let conf_m = twin_confidence_map();
        let cur = conf_m.get(&step).map(|e| *e.value()).unwrap_or(0.9f32);
        conf_m.insert(step.clone(), (cur + confidence_adj as f32).clamp(0.1, 1.0));

        ("200 OK", format!(
            r#"{{"ok":true,"step":"{}","expected":{:.4},"actual":{:.4},"delta_pct":{:.2},"verified":{},"confidence_updated":{:.3}}}"#,
            step, expected, obs, delta_pct, verified,
            (cur + confidence_adj as f32).clamp(0.1, 1.0)
        ))
    } else {
        ("200 OK", format!(
            r#"{{"ok":true,"step":"{}","expected":{:.4},"actual":null,"verified":false,"reason":"step not found in twin state"}}"#,
            step, expected
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 6: LEARNING ENGINE — audit log pattern analysis
// ═══════════════════════════════════════════════════════════════════════

fn learn_stats_json() -> (&'static str, String) {
    use std::io::{BufRead, BufReader};
    use std::collections::HashMap;

    let file = match std::fs::File::open("/opt/crysl/data/audit.jsonl") {
        Ok(f) => f,
        Err(_) => return ("200 OK", r#"{"ok":true,"stats":[],"total":0,"message":"no audit data yet"}"#.into()),
    };

    let mut domain_risk: HashMap<String, (f64, usize)> = HashMap::new();
    let mut mode_counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;

    for line in BufReader::new(file).lines().filter_map(|l| l.ok()) {
        let plan = extract_json_str(&line, "plan").unwrap_or_else(|| "unknown".into());
        let mode = extract_json_str(&line, "mode").unwrap_or_else(|| "unknown".into());
        let risk = extract_json_float(&line, "risk").unwrap_or(0.0);

        let domain = if plan.contains("pump") || plan.contains("fire") || plan.contains("nfpa") {
            "fire_protection"
        } else if plan.contains("electric") || plan.contains("voltage") {
            "electrical"
        } else if plan.contains("password") || plan.contains("crypto") || plan.contains("bcrypt") {
            "cybersecurity"
        } else if plan.contains("factura") || plan.contains("planilla") || plan.contains("peru") {
            "finance"
        } else {
            "general"
        };

        let e = domain_risk.entry(domain.to_string()).or_insert((0.0, 0));
        e.0 += risk; e.1 += 1;
        *mode_counts.entry(mode).or_insert(0) += 1;
        total += 1;
    }

    let domain_stats: Vec<String> = domain_risk.iter().map(|(d, (risk_sum, count))| {
        let avg = if *count > 0 { risk_sum / *count as f64 } else { 0.0 };
        let threshold = match d.as_str() {
            "fire_protection" => 2.5,
            "cybersecurity"   => 2.0,
            "electrical"      => 3.0,
            "finance"         => 3.5,
            _                 => 4.5,
        };
        format!(
            r#"{{"domain":"{}","avg_risk":{:.2},"count":{},"threshold":{:.1},"within_threshold":{}}}"#,
            d, avg, count, threshold, avg <= threshold
        )
    }).collect();

    let mode_stats: Vec<String> = mode_counts.iter()
        .map(|(m, c)| format!(r#"{{\"mode\":\"{}\",\"count\":{}}}"#, m, c)).collect();

    let live_actions = std::fs::read_to_string("/opt/crysl/data/live_actions.jsonl")
        .map(|s| s.lines().count()).unwrap_or(0);

    ("200 OK", format!(
        r#"{{"ok":true,"total_decisions":{},"live_actions":{},"by_domain":[{}],"by_mode":[{}]}}"#,
        total, live_actions,
        domain_stats.join(","),
        mode_counts.iter().map(|(m,c)| format!(r#"{{"mode":"{}","count":{}}}"#, m, c)).collect::<Vec<_>>().join(",")
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 7: AUTONOMOUS LOOP — observe → decide → act → verify → learn
// ═══════════════════════════════════════════════════════════════════════

static LIVE_PLANS: std::sync::OnceLock<std::sync::Arc<Vec<PlanDecl>>> = std::sync::OnceLock::new();
static LIVE_JIT:   std::sync::OnceLock<std::sync::Arc<std::option::Option<plan::JitFnMap>>> = std::sync::OnceLock::new();
static LIVE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LIVE_ACTION_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LIVE_LAST_ACTION: std::sync::OnceLock<std::sync::Mutex<String>> = std::sync::OnceLock::new();

fn live_last_action() -> String {
    LIVE_LAST_ACTION.get_or_init(|| std::sync::Mutex::new("none".into()))
        .lock().unwrap_or_else(|p| p.into_inner()).clone()
}

fn live_status_json() -> (&'static str, String) {
    let active = LIVE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
    let actions = LIVE_ACTION_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let last = live_last_action();
    let live_actions_count = std::fs::read_to_string("/opt/crysl/data/live_actions.jsonl")
        .map(|s| s.lines().count()).unwrap_or(0);
    ("200 OK", format!(
        r#"{{"ok":true,"active":{},"action_count":{},"live_log_entries":{},"last_action":{}}}"#,
        active, actions, live_actions_count,
        if last == "none" { "null".to_string() } else { format!("\"{}\"", last.replace('"', "'")) }
    ))
}

fn spawn_autonomous_loop(
    plans: std::sync::Arc<Vec<PlanDecl>>,
    jit_map: std::sync::Arc<std::option::Option<plan::JitFnMap>>,
) {
    std::thread::Builder::new()
        .name("qomni-live".into())
        .spawn(move || {
            eprintln!("[LIVE] Autonomous loop started (poll: 10s)");
            let mut cycle = 0u64;

            while LIVE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
                cycle += 1;
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default().as_secs();

                // ── OBSERVE: read twin state ──────────────────────────────────
                let state = twin_state_map();
                let mut findings: Vec<String> = Vec::new();
                let mut severity = "ok";

                for entry in state.iter() {
                    let (key, (val, _)) = (entry.key(), entry.value());
                    let finding = match key.as_str() {
                        k if k.contains("pressure") && *val < 50.0 => {
                            severity = "critical";
                            Some(format!("{} critically low ({:.1})", k, val))
                        }
                        k if k.contains("pressure") && *val > 200.0 => {
                            severity = "critical";
                            Some(format!("{} critically high ({:.1})", k, val))
                        }
                        k if k.contains("efficiency") && *val < 0.55 => {
                            severity = "warning";
                            Some(format!("{} degraded ({:.2})", k, val))
                        }
                        k if k.contains("flow") && *val < 100.0 => {
                            severity = "warning";
                            Some(format!("{} low flow ({:.1} gpm)", k, val))
                        }
                        _ => None,
                    };
                    if let Some(f) = finding { findings.push(f); }
                }

                // ── OBSERVE: check SLO metrics ────────────────────────────────
                let slo_alerts = check_slo_health();
                if !slo_alerts.is_empty() {
                    if severity == "ok" { severity = "warning"; }
                    findings.extend(slo_alerts);
                }

                // ── DECIDE: only act if there are findings ────────────────────
                if !findings.is_empty() {
                    // ── ACT: determine action based on severity ───────────────
                    let action_type = match severity {
                        "critical" => "emergency_alert",
                        "warning"  => "advisory",
                        _          => "log",
                    };

                    let findings_json: Vec<String> = findings.iter()
                        .map(|f| format!("\"{}\"", f.replace('"', "'"))).collect();

                    let action_entry = format!(
                        r#"{{"ts":{},"cycle":{},"severity":"{}","action":"{}","findings":[{}],"twin_vars":{},"auto":true}}"#,
                        ts, cycle, severity, action_type, findings_json.join(","),
                        state.len()
                    );

                    // ── VERIFY / LOG ──────────────────────────────────────────
                    use std::io::Write;
                    let _ = std::fs::create_dir_all("/opt/crysl/data");
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true).append(true)
                        .open("/opt/crysl/data/live_actions.jsonl") {
                        let _ = writeln!(f, "{}", action_entry);
                    }

                    // Update last action summary
                    let last = LIVE_LAST_ACTION.get_or_init(|| std::sync::Mutex::new(String::new()));
                    if let Ok(mut l) = last.lock() {
                        *l = format!("cycle:{} severity:{} findings:{}", cycle, severity, findings.len());
                    }
                    LIVE_ACTION_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[LIVE] cycle={} severity={} findings={}", cycle, severity, findings.len());

                    // ── LEARN: write to audit for pattern tracking ────────────
                    audit_write("autonomous_loop", action_type, match severity { "critical" => 8.0, "warning" => 4.0, _ => 1.0 }, 0.9, cycle);
                }

                std::thread::sleep(std::time::Duration::from_secs(10));
            }
            eprintln!("[LIVE] Autonomous loop stopped after {} cycles", cycle);
        }).ok();
}



// ═══════════════════════════════════════════════════════════════
// LEVEL 8 — MULTI-AGENT ENGINE (new-style: session-based, context-weights)
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
enum Verdict8 { Proceed, Caution, Halt }

#[derive(Clone, Debug)]
struct AgentResult8 {
    agent: String,
    verdict: Verdict8,
    confidence: f64,
    score: f64,
    notes: Vec<String>,
}

fn v8_value(v: &Verdict8) -> f64 {
    match v { Verdict8::Proceed => 1.0, Verdict8::Caution => 0.0, Verdict8::Halt => -1.0 }
}
fn v8_str(v: &Verdict8) -> &'static str {
    match v { Verdict8::Proceed => "Proceed", Verdict8::Caution => "Caution", Verdict8::Halt => "Halt" }
}

fn agent_session_get(s: &std::collections::HashMap<String,f64>, k: &str) -> f64 {
    *s.get(k).unwrap_or(&0.0)
}

fn agent_hydraulic(s: &std::collections::HashMap<String,f64>) -> AgentResult8 {
    let hp   = agent_session_get(s, "hp");
    let flow = agent_session_get(s, "flow_gpm");
    let psi  = agent_session_get(s, "head_psi");
    let mut notes = Vec::new();
    let verdict = if hp > 75.0 || flow > 1500.0 || psi > 200.0 {
        notes.push(format!("High demand: hp={:.1} flow={:.0}gpm psi={:.0}", hp, flow, psi));
        Verdict8::Halt
    } else if hp > 30.0 || flow > 750.0 || psi > 130.0 {
        notes.push(format!("Elevated: hp={:.1} flow={:.0}gpm psi={:.0}", hp, flow, psi));
        Verdict8::Caution
    } else {
        notes.push(format!("OK: hp={:.1} flow={:.0}gpm psi={:.0}", hp, flow, psi));
        Verdict8::Proceed
    };
    let score = (1.0 - (hp/100.0).min(1.0))*0.6 + (1.0 - (flow/2000.0).min(1.0))*0.4;
    AgentResult8 { agent:"Hydraulic".into(), verdict, confidence:0.85, score, notes }
}

fn agent_electrical(s: &std::collections::HashMap<String,f64>) -> AgentResult8 {
    let kw   = agent_session_get(s, "kw");
    let hp   = agent_session_get(s, "hp");
    let load = if kw > 0.0 { kw } else { hp * 0.7457 };
    let mut notes = Vec::new();
    let verdict = if load > 150.0 {
        notes.push(format!("Critical load: {:.1}kW", load)); Verdict8::Halt
    } else if load > 56.0 {
        notes.push(format!("High load: {:.1}kW — 3-phase", load)); Verdict8::Caution
    } else {
        notes.push(format!("OK: {:.1}kW", load)); Verdict8::Proceed
    };
    let score = 1.0 - (load/200.0).min(1.0);
    AgentResult8 { agent:"Electrical".into(), verdict, confidence:0.90, score, notes }
}

fn agent_cost(s: &std::collections::HashMap<String,f64>) -> AgentResult8 {
    let hp   = agent_session_get(s, "hp");
    let flow = agent_session_get(s, "flow_gpm");
    let cost = hp * 120.0 + flow * 2.0;
    let mut notes = Vec::new();
    let verdict = if cost > 15000.0 {
        notes.push(format!("Budget alert: ${:.0}", cost)); Verdict8::Halt
    } else if cost > 5000.0 {
        notes.push(format!("Cost elevated: ${:.0}", cost)); Verdict8::Caution
    } else {
        notes.push(format!("OK: ${:.0}", cost)); Verdict8::Proceed
    };
    let score = 1.0 - (cost/20000.0).min(1.0);
    AgentResult8 { agent:"Cost".into(), verdict, confidence:0.75, score, notes }
}

fn agent_cyber(s: &std::collections::HashMap<String,f64>) -> AgentResult8 {
    let threat  = agent_session_get(s, "threat_score");
    let anomaly = agent_session_get(s, "anomaly_score");
    let risk = threat.max(anomaly);
    let mut notes = Vec::new();
    let verdict = if risk > 0.7 {
        notes.push(format!("HIGH threat: {:.2}", risk)); Verdict8::Halt
    } else if risk > 0.35 {
        notes.push(format!("Elevated risk: {:.2}", risk)); Verdict8::Caution
    } else {
        notes.push("Threat: normal".into()); Verdict8::Proceed
    };
    let score = 1.0 - risk.min(1.0);
    AgentResult8 { agent:"Cyber".into(), verdict, confidence:0.80, score, notes }
}

fn context_weights8(ctx: &str) -> std::collections::HashMap<String,f64> {
    let mut w = std::collections::HashMap::new();
    match ctx {
        "hospital" => {
            w.insert("Hydraulic".into(), 0.45);
            w.insert("Electrical".into(), 0.35);
            w.insert("Cost".into(), 0.10);
            w.insert("Cyber".into(), 0.10);
        }
        "datacenter" => {
            w.insert("Hydraulic".into(), 0.20);
            w.insert("Electrical".into(), 0.35);
            w.insert("Cost".into(), 0.15);
            w.insert("Cyber".into(), 0.30);
        }
        "industrial" => {
            w.insert("Hydraulic".into(), 0.35);
            w.insert("Electrical".into(), 0.40);
            w.insert("Cost".into(), 0.20);
            w.insert("Cyber".into(), 0.05);
        }
        "commercial" => {
            w.insert("Hydraulic".into(), 0.30);
            w.insert("Electrical".into(), 0.30);
            w.insert("Cost".into(), 0.30);
            w.insert("Cyber".into(), 0.10);
        }
        _ => {
            w.insert("Hydraulic".into(), 0.40);
            w.insert("Electrical".into(), 0.30);
            w.insert("Cost".into(), 0.20);
            w.insert("Cyber".into(), 0.10);
        }
    }
    w
}

fn consensus8(results: &[AgentResult8], weights: &std::collections::HashMap<String,f64>) -> (Verdict8, f64, f64) {
    let mut score = 0.0f64;
    let mut total_weight = 0.0f64;
    for r in results {
        let w = weights.get(&r.agent).copied().unwrap_or(0.1);
        score += v8_value(&r.verdict) * w * r.confidence;
        total_weight += w;
    }
    let fs = if total_weight > 0.0 { score / total_weight } else { 0.0 };
    let verdict = if fs > 0.4 { Verdict8::Proceed } else if fs < -0.4 { Verdict8::Halt } else { Verdict8::Caution };
    (verdict, fs.abs().min(1.0), fs)
}

fn ar8_to_json(r: &AgentResult8) -> String {
    let notes: Vec<String> = r.notes.iter().map(|n| format!("\"{}\"", n.replace('"',"'"))).collect();
    format!(r#"{{"agent":"{}","verdict":"{}","confidence":{:.3},"score":{:.3},"notes":[{}]}}"#,
        r.agent, v8_str(&r.verdict), r.confidence, r.score, notes.join(","))
}

fn parse_session8(body: &str) -> (std::collections::HashMap<String,f64>, String) {
    let mut ctx = "default".to_string();
    // Extract context string
    if let Some(ci) = body.find("\"context\"") {
        let after = &body[ci+9..];
        if let Some(start) = after.find('"') {
            let s2 = &after[start+1..];
            if let Some(end) = s2.find('"') {
                ctx = s2[..end].to_string();
            }
        }
    }
    let mut session = std::collections::HashMap::new();
    if let Some(vars) = extract_json_obj_float(body, "session") {
        session = vars;
    }
    // Enrich from twin
    let twin = twin_state_map();
    for e in twin.iter() {
        if !session.contains_key(e.key()) {
            session.insert(e.key().clone(), e.value().0);
        }
    }
    (session, ctx)
}

fn agents_list_json() -> (&'static str, String) {
    ("200 OK", r#"{"ok":true,"agents":[
  {"name":"Hydraulic","domain":"fire_protection","weight_default":0.40,"evaluates":["hp","flow_gpm","head_psi"]},
  {"name":"Electrical","domain":"electrical","weight_default":0.30,"evaluates":["kw","hp"]},
  {"name":"Cost","domain":"economics","weight_default":0.20,"evaluates":["hp","flow_gpm"]},
  {"name":"Cyber","domain":"cybersecurity","weight_default":0.10,"evaluates":["threat_score","anomaly_score"]}
],"contexts":["default","hospital","datacenter","industrial","commercial"]}"#.into())
}

fn agents_analyze(body: &str) -> (&'static str, String) {
    let (session, ctx) = parse_session8(body);
    let weights = context_weights8(&ctx);
    let results = vec![
        agent_hydraulic(&session),
        agent_electrical(&session),
        agent_cost(&session),
        agent_cyber(&session),
    ];
    let (verdict, conf, score) = consensus8(&results, &weights);
    let risk = match verdict { Verdict8::Halt => 4.0, Verdict8::Caution => 2.0, Verdict8::Proceed => 0.5 };
    let mode = policy_gate(risk, conf, "general");
    let reason = policy_reason(mode, risk, conf, "general");
    audit_write("multi_agent", mode, risk, conf, 0);
    let agents_json: Vec<String> = results.iter().map(ar8_to_json).collect();
    let weights_json: Vec<String> = weights.iter().map(|(k,v)| format!("\"{}\":{:.2}",k,v)).collect();
    let json = format!(
        r#"{{"ok":true,"context":"{}","consensus":{{"verdict":"{}","confidence":{:.3},"score":{:.3}}},"agents":[{}],"policy":{{"mode":"{}","reason":"{}"}},"weights":{{{}}}}}"#,
        ctx, v8_str(&verdict), conf, score, agents_json.join(","), mode, reason, weights_json.join(",")
    );
    ("200 OK", json)
}

// ═══════════════════════════════════════════════════════════════
// LEVEL 9 — DEBATE + MASS SIMULATION ENGINE
// ═══════════════════════════════════════════════════════════════

fn sim_scenarios(flow_base: f64, n: usize) -> Vec<std::collections::HashMap<String,f64>> {
    let mut out = Vec::new();
    let effs  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85];
    let psis  = [70.0, 85.0, 100.0, 115.0, 130.0];
    let flows = [flow_base*0.80, flow_base*0.90, flow_base, flow_base*1.10, flow_base*1.20];
    'outer: for &eff in &effs {
        for &psi in &psis {
            for &flow in &flows {
                let hp = (flow * psi) / (3960.0 * eff);
                let mut p = std::collections::HashMap::new();
                p.insert("flow_gpm".into(), flow);
                p.insert("head_psi".into(), psi);
                p.insert("eff".into(), eff);
                p.insert("hp".into(), hp);
                p.insert("kw".into(), hp * 0.7457);
                out.push(p);
                if out.len() >= n { break 'outer; }
            }
        }
    }
    out
}

fn score8(ar: &[AgentResult8], w: &std::collections::HashMap<String,f64>) -> f64 {
    let mut s = 0.0; let mut tw = 0.0;
    for r in ar {
        let wt = w.get(&r.agent).copied().unwrap_or(0.1);
        s += r.score * wt * r.confidence;
        tw += wt;
    }
    if tw > 0.0 { s / tw } else { 0.0 }
}

fn update_weights8(w: &mut std::collections::HashMap<String,f64>, ar: &[AgentResult8]) {
    for r in ar {
        if let Some(wt) = w.get_mut(&r.agent) {
            *wt = (*wt + (r.score - 0.5) * 0.02).clamp(0.05, 0.70);
        }
    }
    let sum: f64 = w.values().sum();
    if sum > 0.0 { for v in w.values_mut() { *v /= sum; } }
}

fn extract_json_int_body(body: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{}\":", key);
    let start = body.find(&pat)? + pat.len();
    let rest = body[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn agents_debate(body: &str) -> (&'static str, String) {
    let (session, ctx) = parse_session8(body);
    let iterations = extract_json_int_body(body, "iterations").unwrap_or(3).clamp(1,5) as usize;
    let flow_base = session.get("flow_gpm").copied().unwrap_or(500.0);
    let scenarios_n = 150usize; // 150 scenarios per iteration

    let mut weights = context_weights8(&ctx);
    let mut best_params: Option<std::collections::HashMap<String,f64>> = None;
    let mut best_score  = -999.0f64;
    let mut best_ar: Vec<AgentResult8> = Vec::new();
    let mut total_eval  = 0usize;
    let mut iter_log: Vec<String> = Vec::new();

    for iter_i in 0..iterations {
        // Generate scenarios around current best or base
        let ref_flow = best_params.as_ref()
            .and_then(|p| p.get("flow_gpm").copied().map(|f| f))
            .unwrap_or(flow_base);
        let scenarios = sim_scenarios(ref_flow, scenarios_n);
        total_eval += scenarios.len();

        let mut candidates: Vec<(std::collections::HashMap<String,f64>, f64, Vec<AgentResult8>)> = Vec::new();
        for sc in scenarios {
            let mut enriched = session.clone();
            for (k,v) in &sc { enriched.insert(k.clone(), *v); }
            let ar = vec![
                agent_hydraulic(&enriched),
                agent_electrical(&enriched),
                agent_cost(&enriched),
                agent_cyber(&enriched),
            ];
            let s = score8(&ar, &weights);
            candidates.push((sc, s, ar));
        }
        candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((top_p, top_s, top_ar)) = candidates.first() {
            if *top_s > best_score {
                best_score = *top_s;
                best_params = Some(top_p.clone());
                best_ar = top_ar.clone();
            }
            update_weights8(&mut weights, top_ar);
            let top3: Vec<String> = candidates.iter().take(3).map(|(p,sc,_)|
                format!(r#"{{"flow":{:.1},"hp":{:.2},"psi":{:.1},"score":{:.3}}}"#,
                    p.get("flow_gpm").copied().unwrap_or(0.0),
                    p.get("hp").copied().unwrap_or(0.0),
                    p.get("head_psi").copied().unwrap_or(0.0), sc)
            ).collect();
            iter_log.push(format!(r#"{{"iter":{},"candidates":{},"top_score":{:.3},"top3":[{}]}}"#,
                iter_i+1, candidates.len(), top_s, top3.join(",")));
        }
    }

    // Final consensus on best
    let final_ar = if !best_ar.is_empty() {
        best_ar.clone()
    } else {
        vec![agent_hydraulic(&session), agent_electrical(&session), agent_cost(&session), agent_cyber(&session)]
    };
    let (verdict, conf, score) = consensus8(&final_ar, &weights);
    let risk = match verdict { Verdict8::Halt => 4.0, Verdict8::Caution => 2.0, Verdict8::Proceed => 0.5 };
    let mode = policy_gate(risk, conf, "general");
    let reason = policy_reason(mode, risk, conf, "general");
    audit_write("debate", mode, risk, conf, 0);

    let best_json = if let Some(ref bp) = best_params {
        let kv: Vec<String> = bp.iter().map(|(k,v)| format!("\"{}\":{:.3}",k,v)).collect();
        format!("{{{}}}", kv.join(","))
    } else { "{}".into() };
    let agents_json: Vec<String> = final_ar.iter().map(ar8_to_json).collect();
    let weights_json: Vec<String> = weights.iter().map(|(k,v)| format!("\"{}\":{:.3}",k,v)).collect();
    let json = format!(
        r#"{{"ok":true,"context":"{}","iterations_run":{},"total_evaluated":{},"best":{},"consensus":{{"verdict":"{}","confidence":{:.3},"score":{:.3}}},"agents":[{}],"policy":{{"mode":"{}","reason":"{}"}},"final_weights":{{{}}},"debate":[{}]}}"#,
        ctx, iterations, total_eval, best_json,
        v8_str(&verdict), conf, score, agents_json.join(","),
        mode, reason, weights_json.join(","),
        iter_log.join(",")
    );
    ("200 OK", json)
}

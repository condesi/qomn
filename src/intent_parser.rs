// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v2.0 — Intent Parser Interface
// Percy Rojas M. · Qomni AI Lab · 2026
//
// Converts natural language queries into IntentAST structs.
// The LLM backend is pluggable — use any OpenAI-compatible API.
//
// No direct API calls here. The caller provides a LlmBackend trait impl.
// This allows:
//   - MockBackend (keyword routing, production)
//   - Mock backend for tests
//   - Qomni Engine's own LLM router (Server5)
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use serde_json;

// ═══════════════════════════════════════════════════════════════════════
// ORACLE REGISTRY — Cognitive Compiler Meta-Planner
// Each oracle self-describes its variables, units, keywords, and loop support.
// The registry is the ONLY place to add new computation domains.
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OracleKind {
    Pressure,    // hydraulic, pneumatic
    Flow,        // volumetric flow
    Voltage,     // electrical
    Financial,   // economics, interest
    Structural,  // beams, columns
    Thermal,     // temperature, heat
    Generic,
}

#[derive(Debug, Clone)]
pub struct OracleMeta {
    pub name:          &'static str,
    pub kind:          OracleKind,
    pub loop_pos:      usize,              // which arg is the loop variable (0-based)
    pub loop_var:      &'static str,       // e.g. "Q_gpm"
    pub loop_label:    &'static str,       // display name: "Q"
    pub result_label:  &'static str,       // display name: "P_final"
    pub loop_unit:     &'static str,       // e.g. "gpm"
    pub result_unit:   &'static str,       // e.g. "psi"
    pub keywords:      &'static [&'static str],
    pub unit_tokens:   &'static [&'static str],
    pub var_tokens:    &'static [&'static str],
}

impl OracleMeta {
    /// Score this oracle against a query. Higher = better match.
    pub fn score(&self, query: &str) -> u32 {
        let mut s = 0u32;
        for k in self.keywords    { if query.contains(k) { s += 2; } }
        for u in self.unit_tokens { if query.contains(u) { s += 3; } }
        for v in self.var_tokens  { if query.contains(v) { s += 1; } }
        s
    }
}

/// Static registry — add new oracles here, nothing else changes.
static ORACLE_REGISTRY: &[OracleMeta] = &[
    OracleMeta {
        name: "hazen_P_at_gpm",
        kind: OracleKind::Pressure,
        loop_pos:     0,
        loop_var:     "Q_gpm",
        loop_label:   "Q",
        result_label: "P_final",
        loop_unit:    "gpm",
        result_unit:  "psi",
        keywords:   &["caudal", "flujo", "presion", "tuberia", "tubería",
                      "red hidraulica", "red hidráulica", "red contra incendio",
                      "perdida de carga", "pérdida de carga"],
        unit_tokens: &["gpm", "psi", "l/s", "m3/s"],
        var_tokens:  &["q=", "caudal=", "q_gpm"],
    },
    OracleMeta {
        name: "hazen_hf_gpm_inch",
        kind: OracleKind::Flow,
        loop_pos:     0,
        loop_var:     "Q_gpm",
        loop_label:   "Q",
        result_label: "h_loss",
        loop_unit:    "gpm",
        result_unit:  "m",
        keywords:   &["perdida de carga", "pérdida de carga", "friccion", "fricción", "hazen"],
        unit_tokens: &["gpm", "m/s", "l/s"],
        var_tokens:  &["hazen", "hf", "h_loss"],
    },
    OracleMeta {
        name: "voltage_drop",
        kind: OracleKind::Voltage,
        loop_pos:     0,
        loop_var:     "I",
        loop_label:   "I",
        result_label: "Vdrop",
        loop_unit:    "A",
        result_unit:  "V",
        keywords:   &["corriente", "amperio", "ampere", "caida de tension",
                      "caída de tensión", "voltage drop", "conductor", "cable electrico"],
        unit_tokens: &["amperio", "ampere", " a,", "voltio", "volt"],
        var_tokens:  &["corriente", "i=", "intensidad"],
    },
    OracleMeta {
        name: "load_current_1ph",
        kind: OracleKind::Voltage,
        loop_pos:     0,
        loop_var:     "P_w",
        loop_label:   "P",
        result_label: "I",
        loop_unit:    "W",
        result_unit:  "A",
        keywords:   &["potencia", "consumo", "carga electrica", "carga eléctrica",
                      "corriente de carga", "monofasico", "monofásico"],
        unit_tokens: &["watt", "kw", "vatios", "va", "kva"],
        var_tokens:  &["potencia", "p=", "carga"],
    },
    OracleMeta {
        name: "nfpa20_pump_hp",
        kind: OracleKind::Flow,
        loop_pos:     0,
        loop_var:     "Q_gpm",
        loop_label:   "Q",
        result_label: "HP",
        loop_unit:    "gpm",
        result_unit:  "HP",
        keywords:   &["bomba", "pump", "hp requerido", "potencia bomba",
                      "bomba contra incendio", "nfpa 20"],
        unit_tokens: &["hp", "gpm", "horsepower"],
        var_tokens:  &["bomba", "pump", "hp"],
    },
    OracleMeta {
        name: "darcy_head_loss",
        kind: OracleKind::Pressure,
        loop_pos:     1,  // f is fixed, L varies... or use velocity as loop
        loop_var:     "v",
        loop_label:   "v",
        result_label: "h_f",
        loop_unit:    "m/s",
        result_unit:  "m",
        keywords:   &["darcy", "weisbach", "perdida friccion", "velocidad tuberia"],
        unit_tokens: &["m/s", "f=", "darcy"],
        var_tokens:  &["darcy", "f=", "velocidad"],
    },
    OracleMeta {
        name: "nfpa13_sprinkler",
        kind: OracleKind::Flow,
        loop_pos:     1,   // K fixed, P varies
        loop_var:     "P",
        loop_label:   "P",
        result_label: "Q_head",
        loop_unit:    "psi",
        result_unit:  "gpm",
        keywords:   &["rociador", "sprinkler", "k-factor", "k factor", "caudal rociador"],
        unit_tokens: &["psi", "gpm", "k="],
        var_tokens:  &["rociador", "k-factor", "k="],
    },
];

/// Select the best matching oracle for a query using weighted scoring.
/// Returns the oracle with the highest score (min score 1 to avoid random match).
pub fn match_oracle<'a>(query: &str, kind_filter: Option<OracleKind>) -> Option<&'a OracleMeta> {
    ORACLE_REGISTRY.iter()
        .filter(|o| kind_filter.map(|k| o.kind == k).unwrap_or(true))
        .map(|o| (o, o.score(query)))
        .filter(|(_, s)| *s > 0)
        .max_by_key(|(_, s)| *s)
        .map(|(o, _)| o)
}

/// Extract fixed args for a loop oracle from the query.
/// Returns (fixed_args_vec, loop_label, result_label, loop_unit, result_unit, title).
pub fn extract_loop_args(oracle: &OracleMeta, query: &str)
    -> (Vec<f64>, &'static str, &'static str, &'static str, &'static str, String)
{
    let fixed = match oracle.name {
        "hazen_P_at_gpm" => {
            let c = extract_number_after_pub(query, "c=").unwrap_or(120.0);
            let d = extract_number_before_pub(query, " pulgadas")
                .or_else(|| extract_number_before_pub(query, "\""))
                .or_else(|| extract_number_before_pub(query, "mm").map(|v| v / 25.4))
                .unwrap_or(6.0);
            let l = extract_number_before_pub(query, " m,")
                .or_else(|| extract_number_before_pub(query, " metros"))
                .or_else(|| extract_number_after_pub(query, "metros"))
                .unwrap_or(300.0);
            let p = extract_number_after_pub(query, "presion disponible")
                .or_else(|| extract_number_after_pub(query, "presion inicial"))
                .unwrap_or(100.0);
            vec![c, d, l, p]
        },
        "hazen_hf_gpm_inch" => {
            let c = extract_number_after_pub(query, "c=").unwrap_or(120.0);
            let d = extract_number_before_pub(query, " pulgadas").unwrap_or(6.0);
            let l = extract_number_before_pub(query, " metros").unwrap_or(100.0);
            vec![c, d, l]
        },
        "voltage_drop" => {
            let l   = extract_number_before_pub(query, " m,")
                .or_else(|| extract_number_after_pub(query, "metros")).unwrap_or(100.0);
            let rho = 0.0000000172_f64; // copper resistivity
            let a   = extract_number_after_pub(query, "mm2")
                .or_else(|| extract_number_after_pub(query, "seccion")).unwrap_or(4.0)
                / 1_000_000.0;
            vec![l, rho, a]
        },
        "load_current_1ph" => {
            let v  = extract_number_after_pub(query, "voltios")
                .or_else(|| extract_number_after_pub(query, "v=")).unwrap_or(220.0);
            let pf = extract_number_after_pub(query, "fp=")
                .or_else(|| extract_number_after_pub(query, "fp ")).unwrap_or(0.9);
            vec![v, pf]
        },
        "nfpa20_pump_hp" => {
            let p   = extract_number_after_pub(query, "psi").unwrap_or(100.0);
            let eff = extract_number_after_pub(query, "eficiencia").unwrap_or(0.7);
            vec![p, eff]
        },
        "darcy_head_loss" => {
            let f = extract_number_after_pub(query, "f=").unwrap_or(0.02);
            let l = extract_number_before_pub(query, " metros").unwrap_or(100.0);
            let d = extract_number_before_pub(query, " m,").unwrap_or(0.1);
            vec![f, l, d]
        },
        "nfpa13_sprinkler" => {
            let k = extract_number_after_pub(query, "k=").unwrap_or(5.6);
            vec![k]
        },
        _ => vec![],
    };
    let title = format!("{} Sweep", match oracle.kind {
        OracleKind::Pressure   => "Hydraulic Pressure",
        OracleKind::Flow       => "Flow Rate",
        OracleKind::Voltage    => "Electrical",
        OracleKind::Financial  => "Financial",
        OracleKind::Structural => "Structural",
        OracleKind::Thermal    => "Thermal",
        OracleKind::Generic    => "Parametric",
    });
    (fixed, oracle.loop_label, oracle.result_label, oracle.loop_unit, oracle.result_unit, title)
}

// Public wrappers so extract_loop_args can call the private helpers
pub fn extract_number_after_pub(text: &str, after: &str) -> Option<f64> {
    extract_number_after(text, after)
}
pub fn extract_number_before_pub(text: &str, before: &str) -> Option<f64> {
    extract_number_before(text, before)
}

// ── Domain types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Domain {
    FireProtection,
    Electrical,
    Civil,
    Mechanical,
    WebSearch,
    General,
    CognitiveLoop,   // Cognitive Compiler: loop/simulation/sweep
    Unknown(String),
}

impl Domain {
    pub fn from_str(s: &str) -> Self {
        match s {
            "fire_protection" | "incendios" | "nfpa" => Domain::FireProtection,
            "electrical"      | "electrico"            => Domain::Electrical,
            "civil"           | "estructural"          => Domain::Civil,
            "mechanical"      | "mecanica"             => Domain::Mechanical,
            "web_search"      | "web" | "search"       => Domain::WebSearch,
            "general"                                  => Domain::General,
            "cognitive_loop"  | "loop" | "simulation"  => Domain::CognitiveLoop,
            other => Domain::Unknown(other.to_string()),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            Domain::FireProtection => "fire_protection",
            Domain::Electrical     => "electrical",
            Domain::Civil          => "civil",
            Domain::Mechanical     => "mechanical",
            Domain::WebSearch      => "web_search",
            Domain::General        => "general",
            Domain::CognitiveLoop  => "cognitive_loop",
            Domain::Unknown(s)     => s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Standard {
    Nfpa13, Nfpa20, Nfpa72, Nfpa101,
    Iec60038, Iec60228,
    Unknown(String),
}

impl Standard {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "NFPA13" | "NFPA 13" => Standard::Nfpa13,
            "NFPA20" | "NFPA 20" => Standard::Nfpa20,
            "NFPA72" | "NFPA 72" => Standard::Nfpa72,
            "NFPA101"| "NFPA 101"=> Standard::Nfpa101,
            "IEC60038"| "IEC 60038" => Standard::Iec60038,
            "IEC60228"| "IEC 60228" => Standard::Iec60228,
            other => Standard::Unknown(other.to_string()),
        }
    }
}

/// Structured output of the intent parser.
/// This is what you get after `parse_intent(query)`.
#[derive(Debug, Clone)]
pub struct IntentAST {
    pub domain:      Domain,
    pub standard:    Option<Standard>,
    pub plan_name:   Option<String>,           // which plan_decl to execute
    pub params:      HashMap<String, f64>,     // extracted numeric params
    pub units:       HashMap<String, String>,  // param name -> unit string
    pub constraints: Vec<String>,              // e.g. "P >= 7.0"
    pub raw_query:   String,                   // original user text
}

impl IntentAST {
    pub fn unknown(query: &str) -> Self {
        IntentAST {
            domain:      Domain::Unknown("unknown".into()),
            standard:    None,
            plan_name:   None,
            params:      HashMap::new(),
            units:       HashMap::new(),
            constraints: vec![],
            raw_query:   query.to_string(),
        }
    }
}

// ── LLM Backend trait ─────────────────────────────────────────────────

/// Any LLM that can take a system prompt + user message and return JSON.
pub trait LlmBackend: Send + Sync {
    /// Send a chat completion request. Returns the assistant's text response.
    fn chat(&self, system: &str, user: &str) -> Result<String, String>;
}

// ── System prompt (universal — not NFPA specific) ─────────────────────

/// The system prompt instructs the LLM to extract a structured IntentAST
/// from a natural language query in any domain.
pub const SYSTEM_PROMPT: &str = r#"
You are the CRYS-L Intent Parser for Qomni Engine.
Your job is to detect the USER INTENT from any natural language query and extract a structured JSON.

Rules:
1. "domain" must be one of: fire_protection, electrical, civil, mechanical, web_search, general
2. Use "web_search" when the user asks to search the internet, look up information, navigate to a site,
   find out about a person/topic online, or any query implying live/external data retrieval.
   Examples that are web_search: "busca en internet", "buscame información sobre", "qué encuentras
   sobre", "navega y busca", "investiga online", "search for", "find information about", "lookup".
3. For web_search: set "plan_name" to the CLEANED search query (the thing to search, not the instruction).
   Example: "qomni buscame en internet percy rojas" → plan_name: "percy rojas"
4. For engineering domains: "plan_name" is the snake_case name of the best matching plan,
   "params" contains numeric values, "units" contains physical units.
5. "constraints" are conditions the result must satisfy.
6. If none of the above match, use domain: "general".

Output ONLY valid JSON. No explanation. No markdown.

Schema:
{
  "domain": "...",
  "standard": "...",
  "plan_name": "...",
  "params": { "param_name": value, ... },
  "units": { "param_name": "unit_string", ... },
  "constraints": [ "...", ... ]
}
"#;

// ── Intent Parser ─────────────────────────────────────────────────────

pub struct IntentParser {
    backend: Box<dyn LlmBackend>,
}

impl IntentParser {
    pub fn new(backend: Box<dyn LlmBackend>) -> Self {
        Self { backend }
    }

    /// Parse a natural language query into an IntentAST.
    /// The LLM call is delegated to the backend.
    pub fn parse(&self, query: &str) -> Result<IntentAST, String> {
        // Get LLM response
        let response = self.backend.chat(SYSTEM_PROMPT, query)
            .map_err(|e| format!("LLM backend error: {}", e))?;

        // Parse JSON response
        self.parse_json_response(&response, query)

    }
}

// ── MockBackend — Keyword Routing (no LLM call) ──────────────────────
// This backend parses queries locally using keyword/number extraction.
// It is the production backend for CRYS-L on Server5.

pub struct MockBackend;

impl LlmBackend for MockBackend {
    fn chat(&self, _system: &str, query: &str) -> Result<String, String> {
        let lower = query.to_lowercase();

        // ── Cognitive loop / sweep / simulation ──────────────────────────
        // Detect: "varía X de A a B" | "aumenta X cuando Y supera Z"
        let has_loop = lower.contains("varia ") || lower.contains("varía ")
            || lower.contains("simula") || lower.contains("barre ")
            || lower.contains("rango") || lower.contains("iterar")
            || lower.contains("aumenta ") || lower.contains("disminuye ")
            || lower.contains("desde ") && lower.contains(" hasta ")
            || lower.contains("sweep") || lower.contains("variacion")
            || lower.contains("variación") || lower.contains("paso ");

        let has_cond = lower.contains("cuando ") || lower.contains("si ")
            || lower.contains("supera") || lower.contains("excede")
            || lower.contains("sobrepasa") || lower.contains("mayor que")
            || lower.contains("menor que") || lower.contains("cae por debajo")
            || lower.contains("supere") || lower.contains("excedan");

        let is_sweep = has_loop && has_cond;

        if is_sweep {
            // Determine which oracle to loop over
            let oracle = if lower.contains("tension") || lower.contains("caida")
                || lower.contains("volt") || lower.contains("corriente") || lower.contains(" a ")
            {
                "voltage_drop"
            } else if lower.contains("presion") || lower.contains("psi") || lower.contains("caudal")
                || lower.contains("gpm") || lower.contains("flujo")
            {
                "hazen_P_at_gpm"
            } else if lower.contains("calor") || lower.contains("temperatura") {
                "hvac_cooling_load"
            } else {
                "voltage_drop"
            };

            // Condition operator
            let cond_op = if lower.contains("supera") || lower.contains("excede")
                || lower.contains("mayor que") || lower.contains("sobrepasa")
                || lower.contains("supere") { ">"
            } else if lower.contains("cae por debajo") || lower.contains("menor que")
                || lower.contains("baje de") || lower.contains("caiga") { "<"
            } else { ">" };

            // Extract range start/end
            let r_start = extract_number_after(&lower, "desde ")
                .or_else(|| extract_number_after(&lower, "de "))
                .or_else(|| extract_number_before(&lower, " a "))
                .unwrap_or(10.0);
            let r_end = extract_number_after(&lower, "hasta ")
                .or_else(|| extract_number_after(&lower, "a "))
                .unwrap_or(100.0);
            let step_v = extract_number_after(&lower, "pasos ")
                .or_else(|| extract_number_after(&lower, "paso "))
                .or_else(|| extract_number_after(&lower, "incremento "))
                .unwrap_or(10.0);

            // Condition threshold
            let cond_val = extract_number_after(&lower, "supera ")
                .or_else(|| extract_number_after(&lower, "excede "))
                .or_else(|| extract_number_after(&lower, "debajo de "))
                .or_else(|| extract_number_after(&lower, "menor que "))
                .or_else(|| extract_number_after(&lower, "mayor que "))
                .unwrap_or(if lower.contains("psi") { 65.0 } else { 5.0 });

            // Fixed oracle params (those not being swept)
            let fixed_json = if oracle == "voltage_drop" {
                let l_m = extract_number_before(&lower, " m.")
                    .or_else(|| extract_number_before(&lower, " metros"))
                    .or_else(|| extract_number_after(&lower, "linea"))
                    .unwrap_or(100.0);
                let a_mm2 = extract_number_before(&lower, " mm2")
                    .or_else(|| extract_number_after(&lower, "mm2"))
                    .unwrap_or(4.0);
                let rho: f64 = 1.72e-8;
                let a_m2: f64 = a_mm2 / 1_000_000.0;
                format!(r#"[{:.1},{:e},{:e}]"#, l_m, rho, a_m2)
            } else if oracle == "hazen_P_at_gpm" {
                let c = extract_number_after(&lower, "c=").unwrap_or(120.0);
                let d = extract_number_before(&lower, " pulgadas")
                    .or_else(|| extract_number_before(&lower, "\""))
                    .unwrap_or(6.0);
                let l = extract_number_before(&lower, " m,")
                    .or_else(|| extract_number_before(&lower, " metros"))
                    .unwrap_or(100.0);
                let p0 = extract_number_before(&lower, " psi")
                    .or_else(|| extract_number_after(&lower, "psi"))
                    .unwrap_or(100.0);
                format!(r#"[{:.1},{:.1},{:.1},{:.1}]"#, c, d, l, p0)
            } else {
                "[]".to_string()
            };

            let loop_pos: usize = 0;
            let n_fixed = if oracle == "voltage_drop" { 3 } else { 4 };
            return Ok(format!(
                r#"{{"domain":"cognitive_loop","oracle":"{}","cond_op":"{}","cond_val":{:.2},"range_start":{:.1},"range_end":{:.1},"step":{:.1},"loop_pos":{},"n_fixed":{},"fixed_args":{}}}"#,
                oracle, cond_op, cond_val, r_start, r_end, step_v, loop_pos, n_fixed, fixed_json
            ));
        }

        // ── FIRE PROTECTION — NFPA 13/20/72/101 ─────────────────────────
        let is_fire = ["rociador", "sprinkler", "incendio", "nfpa", "bomba contra incendio",
                       "bomba", "pump",
                       "detector", "rociadores", "sistema contra incendio", "sistema de incendio",
                       "hidrante", "extintor", "alarm", "egress", "evacuacion"]
            .iter().any(|w| lower.contains(w));

        if is_fire {
            // pipe network 3 segments
            let is_net3_in_fire = lower.contains("red hidraulica") || lower.contains("red hidráulica")
                || lower.contains("red de tuberias") || lower.contains("red de tubería")
                || lower.contains("3 tramos") || lower.contains("tres tramos")
                || (lower.contains("tramo") && lower.contains("serie"));

            if is_net3_in_fire && !is_sweep {
                let q_gpm = extract_number_before(&lower, " gpm")
                    .or_else(|| extract_number_after(&lower, "gpm"))
                    .or_else(|| extract_number_before(&lower, " l/s").map(|v| v * 15.8503))
                    .unwrap_or(500.0);
                let p0 = extract_number_before(&lower, " psi")
                    .or_else(|| extract_number_after(&lower, "psi"))
                    .unwrap_or(100.0);
                let d_in = extract_number_before(&lower, " pulgadas")
                    .or_else(|| extract_number_before(&lower, "\""))
                    .or_else(|| extract_number_before(&lower, "mm").map(|v| v / 25.4))
                    .unwrap_or(6.0);
                let c_coef = extract_number_after(&lower, "c=").unwrap_or(120.0);
                let l_seg = extract_number_before(&lower, " m,")
                    .or_else(|| extract_number_before(&lower, " metros"))
                    .unwrap_or(100.0);
                return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "Hazen-Williams NFPA-13",
  "plan_name": "plan_pipe_network_3",
  "params": {{"Q_gpm": {:.1}, "P0_psi": {:.1}, "C": {:.1}, "D_in": {:.1}, "L_m": {:.1}}},
  "units": {{"Q_gpm": "gpm", "P0_psi": "psi", "C": "adim", "D_in": "in", "L_m": "m/segment"}},
  "constraints": ["P3_psi >= 65.0"]
}}"#, q_gpm, p0, c_coef, d_in, l_seg));
            }

            // hazen sweep
            let is_hazen_sweep = is_sweep || lower.contains("presion critica")
                || lower.contains("caudal critico en red")
                || (lower.contains("hazen") && (lower.contains("psi") || lower.contains("caudal")));
            if is_hazen_sweep && !is_net3_in_fire {
                let p_avail = extract_number_before(&lower, " psi")
                    .or_else(|| extract_number_after(&lower, "psi"))
                    .unwrap_or(100.0);
                let p_min = 65.0_f64;
                let c_coef = extract_number_after(&lower, "c=").unwrap_or(120.0);
                let d_in = extract_number_before(&lower, " pulgadas")
                    .or_else(|| extract_number_before(&lower, "\""))
                    .unwrap_or(6.0);
                let l_total = extract_number_before(&lower, " m,")
                    .or_else(|| extract_number_before(&lower, " metros"))
                    .unwrap_or(100.0);
                return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "Hazen-Williams NFPA-13",
  "plan_name": "plan_hazen_sweep",
  "params": {{"P_avail_psi": {:.1}, "P_min_psi": {:.1}, "C": {:.1}, "D_in": {:.1}, "L_total_m": {:.1}}},
  "units": {{"P_avail_psi": "psi", "P_min_psi": "psi", "C": "adim", "D_in": "in", "L_total_m": "m"}},
  "constraints": ["Q_crit_gpm > 0"]
}}"#, p_avail, p_min, c_coef, d_in, l_total));
            }

            // pump sizing
            if lower.contains("bomba") || lower.contains("pump") {
                let q = extract_number_before(&lower, " gpm")
                    .or_else(|| extract_number_after(&lower, "gpm"))
                    .or_else(|| extract_number_before(&lower, " lpm"))
                    .unwrap_or(500.0);
                let p = extract_number_before(&lower, " psi")
                    .or_else(|| extract_number_after(&lower, "psi"))
                    .unwrap_or(100.0);
                let eff = extract_number_after(&lower, "eficiencia").unwrap_or(0.70);
                return Ok(format!(r#"{{
  "domain": "fire_protection",
  "standard": "NFPA20",
  "plan_name": "plan_pump_sizing",
  "params": {{"Q_gpm": {:.1}, "P_psi": {:.1}, "eff": {:.2}}},
  "units": {{"Q_gpm": "gpm", "P_psi": "psi", "eff": "dimensionless"}},
  "constraints": []
}}"#, q, p, eff));
            }

            // ── Demand / Density based design (NFPA 13 §11) ────────────────
            // Query mentions "densidad" → route to plan_nfpa13_demand
            let has_density = lower.contains("densidad") || lower.contains("density")
                || lower.contains("gpm/ft") || lower.contains("mm/min");
            if has_density {
                let density = extract_number_after(&lower, "densidad")
                    .or_else(|| extract_number_after(&lower, "density"))
                    .unwrap_or(0.1);
                // NFPA 13 §11.2.3.1.4 density/area curve (design area ft²)
                let design_area_ft2: f64 = if density <= 0.10 { 5000.0 }
                    else if density <= 0.15 { 3000.0 }
                    else if density <= 0.20 { 2000.0 }
                    else if density <= 0.25 { 1750.0 }
                    else { 1500.0 };
                // Hose stream per hazard class (ordinary → 250 gpm)
                let hose_stream = 250.0_f64;
                return Ok(format!(r#"{{
  "domain": "fire_protection",
  "standard": "NFPA13",
  "plan_name": "plan_nfpa13_demand",
  "params": {{"area_ft2": {:.1}, "density": {:.3}, "hose_stream": {:.1}}},
  "units": {{"area_ft2": "ft2", "density": "gpm/ft2", "hose_stream": "gpm"}},
  "constraints": ["flow_gpm > 0"]
}}"#, design_area_ft2, density, hose_stream));
            }

            // sprinkler (default) — K-factor based
            let k = extract_number_after(&lower, "k=").unwrap_or(5.6);
            let area = extract_number_after(&lower, "m2").unwrap_or(
                extract_number_after(&lower, "ft2").unwrap_or(1000.0));
            let p_avail = extract_number_after(&lower, "psi").unwrap_or(60.0);
            let hose_str = 250.0_f64;
            return Ok(format!(r#"{{
  "domain": "fire_protection",
  "standard": "NFPA13",
  "plan_name": "plan_sprinkler_system",
  "params": {{"K": {:.1}, "area_ft2": {:.1}, "P_avail": {:.1}, "hose_stream": {:.1}}},
  "units": {{"K": "adim", "area_ft2": "ft2", "P_avail": "psi", "hose_stream": "gpm"}},
  "constraints": ["Q_crit_gpm > 0"]
}}"#, k, area, p_avail, hose_str));
        }

        // ── AMBIGUITY GATE — must be before is_network3 ──────────────────
        let has_number = lower.chars().any(|c| c.is_ascii_digit());
        let is_vague = !has_number && (
            lower.contains("deberia") || lower.contains("debería") ||
            lower.contains("normalmente") || lower.contains("tipicamente") ||
            lower.contains("típicamente") || lower.contains("generalmente") ||
            lower.contains("que recomiendas") || lower.contains("cuanto suele") ||
            lower.contains("cuanto es normal") || lower.contains("que valor") ||
            lower.contains("que se usa") || lower.contains("en general") ||
            lower.contains("que presion") || lower.contains("cuanto caudal")
        );
        if is_vague {
            return Ok(format!(
                r#"{{"ok":true,"domain":"general","query":"{}","plan":null,"note":"ambiguous query — routed to LLM"}}"#,
                lower
            ));
        }

        // ── HYDRAULIC NETWORK — series 3-segment pipe ────────────────────
        let is_network3 = (lower.contains("red hidraulica") || lower.contains("red hidráulica")
                           || lower.contains("red contra incendio") || lower.contains("red de tuberias")
                           || (lower.contains("nodo") && (lower.contains("tuberia") || lower.contains("tubería")))
                           || (lower.contains("tramo") && lower.contains("serie"))
                           || lower.contains("3 tramos") || lower.contains("tres tramos"))
            && !is_sweep;

        if is_network3 {
            let q_gpm = extract_number_before(&lower, " gpm")
                .or_else(|| extract_number_after(&lower, "gpm"))
                .or_else(|| extract_number_before(&lower, " l/s").map(|v| v * 15.8503))
                .unwrap_or(500.0);
            let p0 = extract_number_before(&lower, " psi")
                .or_else(|| extract_number_after(&lower, "psi"))
                .unwrap_or(100.0);
            let d_in = extract_number_before(&lower, " pulgadas")
                .or_else(|| extract_number_before(&lower, "\""))
                .or_else(|| extract_number_before(&lower, "mm").map(|v| v / 25.4))
                .unwrap_or(6.0);
            let c_coef = extract_number_after(&lower, "c=").unwrap_or(120.0);
            let l_seg = extract_number_before(&lower, " m,")
                .or_else(|| extract_number_before(&lower, " metros"))
                .unwrap_or(100.0);
            return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "Hazen-Williams NFPA-13",
  "plan_name": "plan_pipe_network_3",
  "params": {{"Q_gpm": {:.1}, "P0_psi": {:.1}, "C": {:.1}, "D_in": {:.1}, "L_m": {:.1}}},
  "units": {{"Q_gpm": "gpm", "P0_psi": "psi", "C": "adim", "D_in": "in", "L_m": "m/segment"}},
  "constraints": ["P3_psi >= 65.0"]
}}"#, q_gpm, p0, c_coef, d_in, l_seg));
        }

        // ── PUMP SIZING — standalone HP query ────────────────────────────
        let is_pump_hp = (lower.contains("bomba") || lower.contains("pump"))
            && (lower.contains(" hp") || lower.contains("caballos de fuerza")
                || lower.contains("potencia bomba") || lower.contains("cuanto hp")
                || lower.contains("hp necesita") || lower.contains("hp requiere")
                || lower.contains("hp para"));
        if is_pump_hp {
            let q_gpm = extract_number_before(&lower, " gpm")
                .or_else(|| extract_number_after(&lower, "gpm"))
                .or_else(|| extract_number_before(&lower, " lpm").map(|v| v * 0.264172))
                .unwrap_or(500.0);
            let p_psi = extract_number_before(&lower, " psi")
                .or_else(|| extract_number_after(&lower, "psi"))
                .unwrap_or(100.0);
            let eff = extract_number_after(&lower, "eficiencia").unwrap_or(0.70);
            return Ok(format!(r#"{{
  "domain": "fire_protection",
  "standard": "NFPA20",
  "plan_name": "plan_pump_sizing",
  "params": {{"Q_gpm": {:.1}, "P_psi": {:.1}, "eff": {:.2}}},
  "units": {{"Q_gpm": "gpm", "P_psi": "psi", "eff": "dimensionless"}},
  "constraints": []
}}"#, q_gpm, p_psi, eff));
        }

        // ── HYDRAULICS / FLUIDOS ─────────────────────────────────────────
        let is_hydraulic = ["caudal", "flujo", "tuberia", "tubería", "manning", "hazen",
                            "williams", "darcy", "velocidad fluido", "gasto",
                            "conduccion", "conducción", "acueducto", "canal", "diametro tuberia",
                            "presion tuberia", "pérdida de carga", "perdida de carga"]
            .iter().any(|w| lower.contains(w));

        if is_hydraulic {
            // critical flow
            if lower.contains("caudal critico") || lower.contains("caudal crítico")
                || lower.contains("caudal minimo") || lower.contains("caudal mínimo")
                || (lower.contains("critico") && lower.contains("caudal"))
            {
                let p_avail = extract_number_before(&lower, " psi")
                    .or_else(|| extract_number_after(&lower, "psi"))
                    .unwrap_or(100.0);
                let c_coef = extract_number_after(&lower, "c=").unwrap_or(120.0);
                let d_in = extract_number_before(&lower, " pulgadas")
                    .or_else(|| extract_number_before(&lower, "\""))
                    .or_else(|| extract_number_before(&lower, "mm").map(|v| v / 25.4))
                    .unwrap_or(6.0);
                let l_total = extract_number_before(&lower, " m,")
                    .or_else(|| extract_number_before(&lower, " metros"))
                    .unwrap_or(100.0);
                return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "Hazen-Williams NFPA-13",
  "plan_name": "plan_hazen_critical_q",
  "params": {{"P_avail_psi": {:.1}, "P_min_psi": 65.0, "C": {:.1}, "D_in": {:.1}, "L_total_m": {:.1}}},
  "units": {{"P_avail_psi": "psi", "P_min_psi": "psi", "C": "adim", "D_in": "in", "L_total_m": "m"}},
  "constraints": ["Q_crit_gpm > 0"]
}}"#, p_avail, c_coef, d_in, l_total));
            }

            let d = extract_number_before(&lower, "mm").map(|v| v / 1000.0)
                .or_else(|| extract_number_after(&lower, "mm").map(|v| v / 1000.0))
                .or_else(|| extract_number_before(&lower, " pulgadas").map(|v| v * 0.0254))
                .or_else(|| extract_number_after(&lower, "pulgadas").map(|v| v * 0.0254))
                .or_else(|| extract_number_before(&lower, "\"").map(|v| v * 0.0254))
                .unwrap_or(0.1);
            let l = extract_number_before(&lower, " m,")
                .or_else(|| extract_number_before(&lower, " metros"))
                .or_else(|| extract_number_after(&lower, "metros"))
                .or_else(|| extract_number_before(&lower, " m "))
                .unwrap_or(100.0);
            let q = extract_number_before(&lower, " l/s").map(|v| v / 1000.0)
                .or_else(|| extract_number_after(&lower, "l/s").map(|v| v / 1000.0))
                .or_else(|| extract_number_before(&lower, " m3/s"))
                .or_else(|| extract_number_before(&lower, " gpm").map(|v| v * 0.0000630902))
                .or_else(|| extract_number_after(&lower, "gpm").map(|v| v * 0.0000630902))
                .unwrap_or(0.05);

            if lower.contains("hazen") || lower.contains("williams") {
                let c_coef = extract_number_after(&lower, "c=").unwrap_or(
                    extract_number_after(&lower, "c =").unwrap_or(120.0));
                return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "Hazen-Williams",
  "plan_name": "plan_pipe_hazen",
  "params": {{"Q": {:.4}, "C": {:.1}, "D": {:.4}, "L": {:.1}}},
  "units": {{"Q": "m3/s", "C": "adim", "D": "m", "L": "m"}},
  "constraints": ["velocity <= 3.0"]
}}"#, q, c_coef, d, l));
            }
            let n = extract_number_after(&lower, "n=").unwrap_or(0.013);
            let s = extract_number_after(&lower, "s=").unwrap_or(0.001);
            return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "Manning",
  "plan_name": "plan_pipe_manning",
  "params": {{"n": {:.4}, "D": {:.4}, "S": {:.5}}},
  "units": {{"n": "s/m^1/3", "D": "m", "S": "m/m"}},
  "constraints": ["velocity <= 3.0"]
}}"#, n, d, s));
        }

        // ── HVAC / CLIMATIZACIÓN ─────────────────────────────────────────
        let is_hvac = ["climatizacion", "climatización", "aire acondicionado", "hvac",
                       "btu", "toneladas de refrigeracion", "toneladas refrigeracion",
                       "carga termica", "carga térmica", "ventilacion", "ventilación",
                       "refrigeracion", "refrigeración", "enfriamiento", "calefaccion",
                       "calefacción", "frigoria"]
            .iter().any(|w| lower.contains(w));
        if is_hvac {
            let area = extract_number_before(&lower, " m2")
                .or_else(|| extract_number_after(&lower, "m2"))
                .or_else(|| extract_number_before(&lower, " metros cuadrados"))
                .unwrap_or(100.0);
            let h = extract_number_after(&lower, "altura").unwrap_or(3.0);
            let ach = extract_number_after(&lower, "renovaciones").unwrap_or(8.0);
            let dt = extract_number_after(&lower, "delta").unwrap_or(
                extract_number_before(&lower, " grados").unwrap_or(10.0));
            let occ = extract_number_after(&lower, "personas").unwrap_or(5.0);
            return Ok(format!(r#"{{
  "domain": "hvac",
  "standard": "ASHRAE",
  "plan_name": "plan_hvac_cooling",
  "params": {{"area_m2": {:.1}, "ceiling_h": {:.1}, "ach": {:.1}, "delta_t": {:.1}, "occupants": {:.1}}},
  "units": {{"area_m2": "m2", "ceiling_h": "m", "ach": "1/h", "delta_t": "°C", "occupants": "pax"}},
  "constraints": []
}}"#, area, h, ach, dt, occ));
        }

        // ── SOLAR FOTOVOLTAICO ────────────────────────────────────────────
        let is_solar = ["solar", "fotovoltaico", "fotovoltaica", "paneles solares",
                        "panel solar", "kwp", "horas pico", "hsp ",
                        "sistema solar", "energia solar", "generacion solar", "payback solar"]
            .iter().any(|w| lower.contains(w));
        if is_solar {
            let kwh_day = extract_number_after(&lower, "kwh").unwrap_or(
                extract_number_before(&lower, " kwh").unwrap_or(20.0));
            let hsp = extract_number_after(&lower, "hsp").unwrap_or(5.0);
            let panel_wp = extract_number_before(&lower, " wp").unwrap_or(400.0);
            let eff = extract_number_after(&lower, "eficiencia").unwrap_or(0.80);
            let cost = extract_number_before(&lower, " usd").unwrap_or(5000.0);
            let tariff = extract_number_after(&lower, "tarifa").unwrap_or(0.18);
            return Ok(format!(r#"{{
  "domain": "solar",
  "standard": "IEC61215",
  "plan_name": "plan_solar_fv",
  "params": {{"kwh_daily": {:.1}, "hsp": {:.1}, "panel_wp": {:.0}, "eff": {:.2}, "cost_usd": {:.0}, "tariff": {:.3}}},
  "units": {{"kwh_daily": "kWh/day", "hsp": "h", "panel_wp": "W", "eff": "adim", "cost_usd": "USD", "tariff": "USD/kWh"}},
  "constraints": []
}}"#, kwh_day, hsp, panel_wp, eff, cost, tariff));
        }

        // ── PUMP SIZING — standalone HP query ────────────────────────────
        // (already handled above as is_pump_hp, kept here for clarity)

        // ── ELECTRICAL ───────────────────────────────────────────────────
        let is_electrical = ["tension", "caida", "volt", "amperio", "corriente", "potencia",
                             "transformador", "conductor", "cable", "cortocircuito", "tablero",
                             "factor de potencia", "reactiva", "kva", "kvar", "kwh",
                             "instalacion electrica", "electrico", "circuito"]
            .iter().any(|w| lower.contains(w));

        if is_electrical && !lower.contains("rpm") && !(lower.contains("torque") && lower.contains("motor")) {
            if lower.contains("factor de potencia") || lower.contains("condensador") || lower.contains("kvar") {
                let p = extract_number_after(&lower, "kw").unwrap_or(100.0);
                let pf_actual = extract_number_after(&lower, "pf=").unwrap_or(0.75);
                return Ok(format!(r#"{{
  "domain": "electrical",
  "standard": "IEC60038",
  "plan_name": "plan_power_factor_correction",
  "params": {{"P_kw": {:.1}, "pf_actual": {:.2}, "pf_meta": 0.95}},
  "units": {{"P_kw": "kW", "pf_actual": "adim", "pf_meta": "adim"}},
  "constraints": ["pf_meta >= 0.95"]
}}"#, p, pf_actual));
            }
            if lower.contains("transformador") {
                let p = extract_number_after(&lower, "kva").unwrap_or(
                    extract_number_after(&lower, "kw").unwrap_or(100.0));
                return Ok(format!(r#"{{
  "domain": "electrical",
  "standard": "IEC60038",
  "plan_name": "plan_transformer",
  "params": {{"P_kw": {:.1}, "pf": 0.90, "eff": 0.98}},
  "units": {{"P_kw": "kW", "pf": "adim", "eff": "adim"}},
  "constraints": []
}}"#, p));
            }
            if lower.contains("trifasico") || lower.contains("trifásico") || lower.contains("3 fases") {
                let p = extract_number_after(&lower, "kw").unwrap_or(50.0);
                let v = extract_number_after(&lower, "v ").unwrap_or(380.0);
                let l = extract_number_after(&lower, "metros").unwrap_or(50.0);
                return Ok(format!(r#"{{
  "domain": "electrical",
  "standard": "IEC60038",
  "plan_name": "plan_electrical_3ph",
  "params": {{"P_kw": {:.1}, "V": {:.1}, "pf": 0.85, "L": {:.1}, "A": 35.0}},
  "units": {{"P_kw": "kW", "V": "V", "pf": "adim", "L": "m", "A": "mm2"}},
  "constraints": ["voltage_drop_pct <= 5.0"]
}}"#, p, v, l));
            }
            // direct voltage drop
            if (lower.contains("caida") || lower.contains("caída"))
                && lower.contains("tension")
                && !lower.contains("potencia") && !lower.contains("kw") && !lower.contains("carga")
            {
                let i_a = extract_number_before(&lower, " a ")
                    .or_else(|| extract_number_before(&lower, " a,"))
                    .or_else(|| extract_number_before(&lower, " ampere"))
                    .or_else(|| extract_number_before(&lower, " amperio"))
                    .unwrap_or(10.0);
                let l_m = extract_number_before(&lower, " m.")
                    .or_else(|| extract_number_before(&lower, " m,"))
                    .or_else(|| extract_number_before(&lower, " metros"))
                    .or_else(|| extract_number_after(&lower, "metros"))
                    .or_else(|| extract_number_before(&lower, " m "))
                    .unwrap_or(100.0);
                let a_mm2 = extract_number_before(&lower, " mm2")
                    .or_else(|| extract_number_after(&lower, "mm2"))
                    .unwrap_or(4.0);
                return Ok(format!(r#"{{
  "domain": "electrical",
  "standard": "IEC60038",
  "plan_name": "plan_voltage_drop",
  "params": {{"I": {:.1}, "L_m": {:.1}, "A_mm2": {:.1}}},
  "units": {{"I": "A", "L_m": "m", "A_mm2": "mm2"}},
  "constraints": ["voltage_drop_pct <= 3.0"]
}}"#, i_a, l_m, a_mm2));
            }
            // single-phase (default)
            let p = extract_number_after(&lower, "w ").unwrap_or(
                extract_number_after(&lower, "kw").map(|v| v * 1000.0).unwrap_or(5000.0));
            let v = extract_number_after(&lower, "v ").unwrap_or(220.0);
            let l = extract_number_after(&lower, "metros").unwrap_or(
                extract_number_after(&lower, "m ").unwrap_or(50.0));
            return Ok(format!(r#"{{
  "domain": "electrical",
  "standard": "IEC60038",
  "plan_name": "plan_electrical_load",
  "params": {{"P_w": {:.1}, "V": {:.1}, "pf": 0.90, "L": {:.1}, "A": 35.0}},
  "units": {{"P_w": "W", "V": "V", "pf": "adim", "L": "m", "A": "mm2"}},
  "constraints": ["voltage_drop_pct <= 5.0"]
}}"#, p, v, l));
        }

        // ── COMERCIO / NEGOCIOS ───────────────────────────────────────────
        let is_comercio = ["punto de equilibrio", "punto equilibrio", "break even",
                           "margen", "markup", "precio de venta", "roi ",
                           "retorno inversion", "retorno de inversion", "cac ", " ltv ",
                           "precio sugerido", "cuanto cobrar"]
            .iter().any(|w| lower.contains(w));
        if is_comercio {
            if lower.contains("break even") || (lower.contains("punto") && lower.contains("equilibrio")) {
                let fixed = extract_number_after(&lower, "fijos").unwrap_or(10000.0);
                let price = extract_number_after(&lower, "precio").unwrap_or(100.0);
                let var_cost = extract_number_after(&lower, "variable").unwrap_or(60.0);
                return Ok(format!(r#"{{
  "domain": "business",
  "standard": "contabilidad",
  "plan_name": "plan_break_even",
  "params": {{"fixed_cost": {:.1}, "price": {:.1}, "var_cost": {:.1}}},
  "units": {{"fixed_cost": "S/", "price": "S/", "var_cost": "S/"}},
  "constraints": []
}}"#, fixed, price, var_cost));
            }
            let cost = extract_number_after(&lower, "costo").unwrap_or(100.0);
            let markup = extract_number_after(&lower, "markup").unwrap_or(
                extract_number_before(&lower, "%").unwrap_or(30.0));
            return Ok(format!(r#"{{
  "domain": "business",
  "standard": "contabilidad",
  "plan_name": "plan_pricing",
  "params": {{"cost": {:.2}, "markup_pct": {:.1}}},
  "units": {{"cost": "S/", "markup_pct": "%"}},
  "constraints": []
}}"#, cost, markup));
        }

        // ── FINANZAS — Préstamo, interés compuesto ────────────────────────
        let is_finanzas = ["prestamo", "préstamo", "cuota mensual", "amortizacion",
                           "amortización", "tasa de interes", "tasa de interés",
                           "interes compuesto", "interés compuesto", "vpn ", " tir ",
                           "valor presente", "hipoteca"]
            .iter().any(|w| lower.contains(w));
        if is_finanzas {
            let principal = extract_number_before(&lower, " soles").unwrap_or(
                extract_number_before(&lower, " usd").unwrap_or(100000.0));
            let rate = extract_number_before(&lower, "% anual").unwrap_or(
                extract_number_after(&lower, "tasa").unwrap_or(12.0));
            let years = extract_number_before(&lower, " años").unwrap_or(5.0);
            if lower.contains("interes compuesto") || lower.contains("interés compuesto") {
                return Ok(format!(r#"{{
  "domain": "finance",
  "standard": "IFRS",
  "plan_name": "plan_compound_interest",
  "params": {{"P": {:.2}, "r_pct": {:.2}, "years": {:.1}}},
  "units": {{"P": "S/", "r_pct": "%", "years": "años"}},
  "constraints": []
}}"#, principal, rate, years));
            }
            return Ok(format!(r#"{{
  "domain": "finance",
  "standard": "IFRS",
  "plan_name": "plan_loan_amortization",
  "params": {{"P": {:.2}, "r_annual_pct": {:.2}, "years": {:.1}}},
  "units": {{"P": "S/", "r_annual_pct": "%", "years": "años"}},
  "constraints": []
}}"#, principal, rate, years));
        }

        // ── SALUD / MEDICINA ──────────────────────────────────────────────
        let is_salud = ["imc ", "indice de masa", "masa corporal", "bmi ",
                        "dosis ", "medicamento", "farmaco", "fármaco",
                        "calorias", "calorías", "metabolismo basal", "peso ideal",
                        "egfr ", "creatinina", "filtrado glomerular"]
            .iter().any(|w| lower.contains(w));
        if is_salud {
            let weight = extract_number_before(&lower, " kg").unwrap_or(70.0);
            let height = extract_number_before(&lower, " m ").unwrap_or(
                extract_number_before(&lower, " cm").map(|v| v / 100.0).unwrap_or(1.70));
            let age = extract_number_before(&lower, " años").unwrap_or(35.0);
            if lower.contains("dosis") || lower.contains("medicamento") || lower.contains("farmaco") || lower.contains("fármaco") {
                let dose = extract_number_after(&lower, "mg/kg").unwrap_or(5.0);
                let freq = extract_number_after(&lower, "veces").unwrap_or(3.0);
                return Ok(format!(r#"{{
  "domain": "health",
  "standard": "OMS",
  "plan_name": "plan_drug_dosing",
  "params": {{"weight_kg": {:.1}, "dose_mg_per_kg": {:.2}, "frequency_per_day": {:.1}}},
  "units": {{"weight_kg": "kg", "dose_mg_per_kg": "mg/kg", "frequency_per_day": "1/day"}},
  "constraints": []
}}"#, weight, dose, freq));
            }
            return Ok(format!(r#"{{
  "domain": "health",
  "standard": "OMS",
  "plan_name": "plan_bmi_assessment",
  "params": {{"weight_kg": {:.1}, "height_m": {:.2}, "age": {:.0}}},
  "units": {{"weight_kg": "kg", "height_m": "m", "age": "años"}},
  "constraints": []
}}"#, weight, height, age));
        }

        // ── MECÁNICA / MOTORES ────────────────────────────────────────────
        let is_mecanica = ["torque", "rpm ", "potencia motor", "motor electrico",
                           "motor eléctrico", "transmision", "transmisión", "engranaje",
                           "reductora", "fajas", "consumo combustible"]
            .iter().any(|w| lower.contains(w));
        if is_mecanica && !is_pump_hp {
            let power = extract_number_before(&lower, " kw").unwrap_or(
                extract_number_after(&lower, "kw").unwrap_or(10.0));
            let rpm = extract_number_before(&lower, " rpm").unwrap_or(
                extract_number_after(&lower, "rpm").unwrap_or(1450.0));
            let ratio = extract_number_after(&lower, "relacion").unwrap_or(4.0);
            let eff = extract_number_after(&lower, "eficiencia").unwrap_or(0.95);
            return Ok(format!(r#"{{
  "domain": "mechanical",
  "standard": "ISO",
  "plan_name": "plan_motor_drive",
  "params": {{"power_kw": {:.2}, "rpm": {:.0}, "ratio": {:.2}, "eff": {:.2}}},
  "units": {{"power_kw": "kW", "rpm": "rpm", "ratio": "adim", "eff": "adim"}},
  "constraints": []
}}"#, power, rpm, ratio, eff));
        }

        // ── TOPOGRAFÍA / GEOTECNIA ────────────────────────────────────────
        let is_topo = ["pendiente", "talud", "corte y relleno", "volumen de tierra",
                       "movimiento de tierra", "capacidad portante", "curvas de nivel",
                       "topografico", "nivelacion", "nivelación", "cota", "desnivel",
                       "terraplen", "terraplén"]
            .iter().any(|w| lower.contains(w));
        if is_topo {
            if lower.contains("corte") || lower.contains("relleno") || lower.contains("volumen") {
                let area = extract_number_before(&lower, " m2").unwrap_or(1000.0);
                let cut_d = extract_number_after(&lower, "corte").unwrap_or(2.0);
                let fill_d = extract_number_after(&lower, "relleno").unwrap_or(1.5);
                let comp = extract_number_after(&lower, "compactacion").unwrap_or(0.85);
                return Ok(format!(r#"{{
  "domain": "topography",
  "standard": "MTC",
  "plan_name": "plan_earthwork",
  "params": {{"area_m2": {:.1}, "cut_depth": {:.2}, "fill_depth": {:.2}, "compaction": {:.2}}},
  "units": {{"area_m2": "m2", "cut_depth": "m", "fill_depth": "m", "compaction": "adim"}},
  "constraints": []
}}"#, area, cut_d, fill_d, comp));
            }
            let delta_h = extract_number_after(&lower, "desnivel").unwrap_or(5.0);
            let dist = extract_number_after(&lower, "distancia").unwrap_or(100.0);
            return Ok(format!(r#"{{
  "domain": "topography",
  "standard": "MTC",
  "plan_name": "plan_slope_analysis",
  "params": {{"delta_h": {:.2}, "distance": {:.2}}},
  "units": {{"delta_h": "m", "distance": "m"}},
  "constraints": []
}}"#, delta_h, dist));
        }

        // ── LOGÍSTICA / TRANSPORTE ────────────────────────────────────────
        let is_logistica = ["flete", "costo de transporte", "costo transporte",
                            "costo envio", "costo de envio", "distribucion", "distribución",
                            "ruta de reparto", "flota vehicular", "tiempo de entrega",
                            "cuanto cuesta enviar", "cuanto cuesta el flete", "km a recorrer"]
            .iter().any(|w| lower.contains(w));
        if is_logistica {
            let dist = extract_number_before(&lower, " km").unwrap_or(100.0);
            let cost_km = extract_number_after(&lower, "s/km").unwrap_or(3.5);
            let trips = extract_number_after(&lower, "viajes").unwrap_or(1.0);
            let units = extract_number_before(&lower, " unidades").unwrap_or(100.0);
            return Ok(format!(r#"{{
  "domain": "logistics",
  "standard": "ISO",
  "plan_name": "plan_transport",
  "params": {{"distance_km": {:.1}, "cost_per_km": {:.2}, "n_trips": {:.1}, "units_per_trip": {:.0}}},
  "units": {{"distance_km": "km", "cost_per_km": "S//km", "n_trips": "viajes", "units_per_trip": "unid"}},
  "constraints": []
}}"#, dist, cost_km, trips, units));
        }

        // ── ESTADÍSTICA / DATOS ───────────────────────────────────────────
        let is_estadistica = ["estadistica", "estadística", "desviacion estandar",
                              "desviación estándar", "varianza", "tamaño de muestra",
                              "tamano de muestra", "correlacion", "correlación", "percentil",
                              "coeficiente de variacion", "regresion lineal"]
            .iter().any(|w| lower.contains(w));
        if is_estadistica {
            if lower.contains("tamaño") || lower.contains("tamano") {
                let conf = extract_number_before(&lower, "% confianza").unwrap_or(95.0);
                let margin = extract_number_after(&lower, "margen").unwrap_or(5.0);
                return Ok(format!(r#"{{
  "domain": "statistics",
  "standard": "ISO",
  "plan_name": "plan_sample_size",
  "params": {{"confidence_pct": {:.1}, "margin_error_pct": {:.1}}},
  "units": {{"confidence_pct": "%", "margin_error_pct": "%"}},
  "constraints": []
}}"#, conf, margin));
            }
            let n = extract_number_after(&lower, "muestra").unwrap_or(100.0);
            let sum_x = n * 50.0;
            let sum_x2 = sum_x * 55.0;
            return Ok(format!(r#"{{
  "domain": "statistics",
  "standard": "ISO",
  "plan_name": "plan_statistics",
  "params": {{"n": {:.0}, "sum_x": {:.2}, "sum_x2": {:.2}}},
  "units": {{"n": "obs", "sum_x": "Σx", "sum_x2": "Σx2"}},
  "constraints": []
}}"#, n, sum_x, sum_x2));
        }

        // ── AGRO / AGRICULTURA ────────────────────────────────────────────
        let is_agro = ["riego", "evapotranspiracion", "evapotranspiración", "eto ",
                       " kc ", "cultivo", "hectareas", "hectáreas", "cosecha",
                       "rendimiento cultivo", "fertilizante", "agua de riego", "sistema de riego"]
            .iter().any(|w| lower.contains(w));
        if is_agro {
            let area = extract_number_before(&lower, " ha").unwrap_or(1.0);
            let eto = extract_number_after(&lower, "eto").unwrap_or(5.0);
            let kc = extract_number_after(&lower, "kc").unwrap_or(0.85);
            let eff = extract_number_after(&lower, "eficiencia").unwrap_or(0.85);
            return Ok(format!(r#"{{
  "domain": "agriculture",
  "standard": "FAO56",
  "plan_name": "plan_irrigation",
  "params": {{"area_ha": {:.2}, "eto_mm": {:.2}, "kc": {:.2}, "eff": {:.2}}},
  "units": {{"area_ha": "ha", "eto_mm": "mm/day", "kc": "adim", "eff": "adim"}},
  "constraints": []
}}"#, area, eto, kc, eff));
        }

        // ── TELECOMUNICACIONES ────────────────────────────────────────────
        // ── CIBERSEGURIDAD ───────────────────────────────────────────────
        let is_cybersec = ["contrasena", "contrase", "password", "entropia",
                           "cvss", "vulnerabilidad", "vulnerability", "parche",
                           "puertos abiertos", "escaneo de red", "network scan",
                           "certificado ssl", "ssl cert", "fuerza bruta", "brute force",
                           "cifrado aes", "cifrado rsa", "clave criptografica", "crypto",
                           "riesgo de red", "ciberseguridad", "ciberataque",
                           "cyberseguridad", "cybersecurity", "audita ", "auditoria",
                           "audit ", "escaneo seguridad", "analiza seguridad", "seguridad web",
                           "pentest", "hacking", "intrusion", "firewall", "ddos"]
            .iter().any(|w| lower.contains(w));
        if is_cybersec {
            let charset = extract_number_after(&lower, "caracteres").unwrap_or(72.0);
            let length = extract_number_after(&lower, "longitud").unwrap_or(
                extract_number_after(&lower, "length").unwrap_or(12.0));
            let speed = 1_000_000_000.0f64;

            if lower.contains("contrasena") || lower.contains("contrase")
                || lower.contains("password") || lower.contains("fuerza bruta")
                || lower.contains("entropia")
            {
                return Ok(format!(
                    "{{\"domain\":\"cybersecurity\",\"standard\":\"NIST-SP800-63\",\"plan_name\":\"plan_password_audit\",\
                    \"params\":{{\"charset_size\":{:.1},\"length\":{:.1},\"attempts_per_sec\":{:.0}}},\
                    \"units\":{{\"charset_size\":\"chars\",\"length\":\"chars\",\"attempts_per_sec\":\"1/s\"}},\
                    \"constraints\":[]}}",
                    charset, length, speed));
            }
            if lower.contains("cvss") || lower.contains("vulnerabilidad") || lower.contains("parche") {
                return Ok(
                    "{\"domain\":\"cybersecurity\",\"standard\":\"CVSSv3\",\"plan_name\":\"plan_cvss_assessment\",\
                    \"params\":{\"av\":0.85,\"ac\":0.77,\"pr\":0.68,\"ui\":0.85,\"scope\":6.42,\"c\":0.56,\"i\":0.56,\"a\":0.56},\
                    \"units\":{},\"constraints\":[]}".to_string());
            }
            if lower.contains("cifrado") || lower.contains("crypto") || lower.contains("aes") || lower.contains("rsa") {
                let aes_bits = extract_number_before(&lower, " bits").unwrap_or(256.0);
                return Ok(format!(
                    "{{\"domain\":\"cybersecurity\",\"standard\":\"NIST-FIPS\",\"plan_name\":\"plan_crypto_audit\",\
                    \"params\":{{\"aes_bits\":{:.1},\"rsa_bits\":4096.0,\"charset_size\":95.0,\"pwd_length\":16.0}},\
                    \"units\":{{\"aes_bits\":\"bits\"}},\"constraints\":[]}}",
                    aes_bits));
            }
            let ports = extract_number_after(&lower, "puertos").unwrap_or(
                extract_number_after(&lower, "ports").unwrap_or(25.0));
            let unpatched = extract_number_after(&lower, "vulnerabilidades").unwrap_or(5.0);
            return Ok(format!(
                "{{\"domain\":\"cybersecurity\",\"standard\":\"ISO-27001\",\"plan_name\":\"plan_network_security\",\
                \"params\":{{\"open_ports\":{:.1},\"unpatched\":{:.1},\"services\":5.0,\"days_ssl\":90.0}},\
                \"units\":{{\"open_ports\":\"ports\",\"unpatched\":\"CVEs\"}},\"constraints\":[]}}",
                ports, unpatched));
        }


        let is_telecom = ["trayectoria libre", "ghz ", " ghz", "frecuencia mhz", "frecuencia ghz", "perdida de señal",
                          "enlace de radio", "radioenlace", "cobertura celular",
                          "ancho de banda", "fspl", "path loss", "link budget",
                          "shannon", "capacidad canal", "dbm", "antena yagi"]
            .iter().any(|w| lower.contains(w));
        if is_telecom {
            let tx = extract_number_before(&lower, " dbm").unwrap_or(23.0);
            let freq = extract_number_before(&lower, " mhz").unwrap_or(
                extract_number_before(&lower, " ghz").map(|v| v * 1000.0).unwrap_or(900.0));
            let dist = extract_number_before(&lower, " km").unwrap_or(10.0);
            let bw = extract_number_after(&lower, "ancho de banda").unwrap_or(20.0);
            return Ok(format!(r#"{{
  "domain": "telecom",
  "standard": "ITU-R",
  "plan_name": "plan_telecom_link",
  "params": {{"tx_dbm": {:.1}, "freq_mhz": {:.1}, "distance_km": {:.2}, "bw_mhz": {:.1}}},
  "units": {{"tx_dbm": "dBm", "freq_mhz": "MHz", "distance_km": "km", "bw_mhz": "MHz"}},
  "constraints": []
}}"#, tx, freq, dist, bw));
        }

        // ── CIVIL / ESTRUCTURAL ───────────────────────────────────────────
        let is_civil = ["columna estructural", "viga", "losa", "cimentacion", "cimentación",
                        "zapata", "concreto", "hormigon", "armado", "acero", "esfuerzo",
                        "momento flector", "cortante", "deflexion", "deflexión", "pandeo",
                        "capacidad portante", "suelo arcilloso"]
            .iter().any(|w| lower.contains(w));
        if is_civil {
            if (lower.contains("columna") && !lower.contains("agua")) || lower.contains("pandeo") {
                let p = extract_number_after(&lower, "kn").unwrap_or(500.0);
                let l = extract_number_after(&lower, "m ").unwrap_or(3.0);
                let b = extract_number_after(&lower, "cm").unwrap_or(30.0);
                return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "ACI318",
  "plan_name": "plan_column_design",
  "params": {{"P_kn": {:.1}, "L_m": {:.1}, "b_cm": {:.1}, "h_cm": {:.1}}},
  "units": {{"P_kn": "kN", "L_m": "m", "b_cm": "cm", "h_cm": "cm"}},
  "constraints": ["P_kn <= Pcr_euler"]
}}"#, p, l, b, b));
            }
            if lower.contains("zapata") || lower.contains("cimentacion") {
                let q = extract_number_after(&lower, "kg/m2").unwrap_or(15.0);
                let b = extract_number_after(&lower, "m ").unwrap_or(1.5);
                return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "ACI318",
  "plan_name": "plan_footing",
  "params": {{"q_allow": {:.1}, "B": {:.1}, "L": {:.1}}},
  "units": {{"q_allow": "t/m2", "B": "m", "L": "m"}},
  "constraints": []
}}"#, q, b, b));
            }
            let p = extract_number_after(&lower, "kn").unwrap_or(50.0);
            let l = extract_number_after(&lower, "m ").unwrap_or(5.0);
            return Ok(format!(r#"{{
  "domain": "civil",
  "standard": "ACI318",
  "plan_name": "plan_beam_analysis",
  "params": {{"P_kn": {:.1}, "L_m": {:.1}, "E_gpa": 200000.0, "I_cm4": 5000.0}},
  "units": {{"P_kn": "kN", "L_m": "m", "E_gpa": "MPa", "I_cm4": "cm4"}},
  "constraints": ["delta <= L/360"]
}}"#, p, l));
        }

        // ── PERU LEGAL / LABORAL ──────────────────────────────────────────
        let is_legal = ["cts", "vacaciones", "gratificacion", "gratificación",
                        "liquidacion laboral", "liquidación laboral", "beneficios sociales",
                        "remuneracion minima", "rmv", "uit ", "sunafil", "multa laboral",
                        "trabajador despedido", "horas extras", "horas extra"]
            .iter().any(|w| lower.contains(w));
        if is_legal {
            if lower.contains("multa") || lower.contains("sunafil") || lower.contains("infraccion") {
                return Ok(r#"{
  "domain": "general",
  "standard": "SUNAFIL-Peru",
  "plan_name": "plan_multa_sunafil",
  "params": {},
  "units": {},
  "constraints": []
}"#.into());
            }
            let sueldo = extract_number_after(&lower, "s/").unwrap_or(
                extract_number_after(&lower, "sueldo").unwrap_or(3000.0));
            let meses = extract_number_after(&lower, "meses").unwrap_or(12.0);
            let dias_vac = extract_number_after(&lower, "dias").unwrap_or(30.0);
            return Ok(format!(r#"{{
  "domain": "general",
  "standard": "MTyPE-Peru",
  "plan_name": "plan_liquidacion_laboral",
  "params": {{"sueldo": {:.2}, "meses_trabajo": {:.1}, "dias_vacac": {:.1}}},
  "units": {{"sueldo": "PEN", "meses_trabajo": "meses", "dias_vacac": "dias"}},
  "constraints": []
}}"#, sueldo, meses, dias_vac));
        }

        // ── ACCOUNTING PERU ───────────────────────────────────────────────
        let is_accounting = ["igv", "factura", "comprobante", "sunat", "impuesto",
                             "planilla", "sueldo", "salario", "remuneracion",
                             "depreciacion", "depreciación", "activo fijo",
                             "ratio financiero", "liquidez", "detraccion", "detracción",
                             "essalud", "afp", "onp", "precio con igv", "base imponible"]
            .iter().any(|w| lower.contains(w));
        if is_accounting {
            if lower.contains("planilla") || lower.contains("sueldo") || lower.contains("salario") {
                let sueldo = extract_number_after(&lower, "s/").unwrap_or(
                    extract_number_after(&lower, "soles").unwrap_or(3000.0));
                return Ok(format!(r#"{{
  "domain": "general",
  "standard": "SUNAT-Peru",
  "plan_name": "plan_planilla",
  "params": {{"sueldo": {:.2}}},
  "units": {{"sueldo": "PEN"}},
  "constraints": ["sueldo >= 1025.0"]
}}"#, sueldo));
            }
            if lower.contains("ratio") || lower.contains("liquidez") {
                let ac = extract_number_after(&lower, "activo").unwrap_or(100000.0);
                let pc = extract_number_after(&lower, "pasivo").unwrap_or(50000.0);
                let ut = extract_number_after(&lower, "utilidad").unwrap_or(20000.0);
                let vt = extract_number_after(&lower, "ventas").unwrap_or(150000.0);
                let pa = extract_number_after(&lower, "patrimonio").unwrap_or(80000.0);
                return Ok(format!(r#"{{
  "domain": "general",
  "standard": "SUNAT-Peru",
  "plan_name": "plan_ratios_financieros",
  "params": {{"activo_c": {:.2}, "pasivo_c": {:.2}, "utilidad": {:.2}, "ventas": {:.2}, "patrimonio": {:.2}}},
  "units": {{"activo_c": "PEN", "pasivo_c": "PEN", "utilidad": "PEN", "ventas": "PEN", "patrimonio": "PEN"}},
  "constraints": ["liquidez >= 1.0"]
}}"#, ac, pc, ut, vt, pa));
            }
            if lower.contains("depreciacion") || lower.contains("depreciación") || lower.contains("activo fijo") {
                let costo = extract_number_after(&lower, "costo").unwrap_or(10000.0);
                let vida = extract_number_after(&lower, "años").unwrap_or(5.0);
                return Ok(format!(r#"{{
  "domain": "general",
  "standard": "SUNAT-Peru",
  "plan_name": "plan_depreciacion",
  "params": {{"costo": {:.2}, "vida_util": {:.1}, "valor_residual": 0.0}},
  "units": {{"costo": "PEN", "vida_util": "años", "valor_residual": "PEN"}},
  "constraints": []
}}"#, costo, vida));
            }
            let monto = extract_number_after(&lower, "s/").unwrap_or(1000.0);
            return Ok(format!(r#"{{
  "domain": "general",
  "standard": "SUNAT-Peru",
  "plan_name": "plan_factura_peru",
  "params": {{"base_imponible": {:.2}}},
  "units": {{"base_imponible": "PEN"}},
  "constraints": []
}}"#, monto));
        }

        // Unknown — route to general LLM
        Ok(format!(
            r#"{{"ok":true,"domain":"general","query":"{}","plan":null,"note":"no matching plan found"}}"#,
            lower
        ))
    }
}

// ── parse_json_response — JSON string → IntentAST ─────────────────────

impl IntentParser {
    fn parse_json_response(&self, json: &str, query: &str) -> Result<IntentAST, String> {
        // Check for cognitive_loop domain
        if json.contains("\"cognitive_loop\"") || json.contains("cognitive_loop") {
            let oracle = extract_json_str(json, "oracle").unwrap_or_default();
            let cond_op = extract_json_str(json, "cond_op").unwrap_or(">=".into());
            let cond_val = extract_json_f64(json, "cond_val").unwrap_or(65.0);
            let range_start = extract_json_f64(json, "range_start").unwrap_or(100.0);
            let range_end = extract_json_f64(json, "range_end").unwrap_or(1000.0);
            let step_v = extract_json_f64(json, "step").unwrap_or(10.0);
            let loop_pos = extract_json_f64(json, "loop_pos").unwrap_or(0.0) as usize;
            let n_fixed = extract_json_f64(json, "n_fixed").unwrap_or(0.0) as usize;
            let fixed_args: Vec<f64> = parse_f64_array(json, "fixed_args");
            let mut params = HashMap::new();
            params.insert("oracle".to_string(), 0.0);
            params.insert("cond_val".to_string(), cond_val);
            params.insert("range_start".to_string(), range_start);
            params.insert("range_end".to_string(), range_end);
            params.insert("step".to_string(), step_v);
            params.insert("loop_pos".to_string(), loop_pos as f64);
            params.insert("n_fixed".to_string(), n_fixed as f64);
            for (i, v) in fixed_args.iter().enumerate() {
                params.insert(format!("f{}", i), *v);
            }
            let mut units = HashMap::new();
            units.insert("oracle_name".to_string(), oracle.clone());
            units.insert("cond_op".to_string(), cond_op.clone());
            let constraints = vec![format!("cond_op={}", cond_op)];
            return Ok(IntentAST {
                domain:      Domain::CognitiveLoop,
                standard:    None,
                plan_name:   Some(oracle.clone()),
                params,
                units,
                constraints,
                raw_query:   query.to_string(),
            });
        }

        // Check for ambiguous/general (plan: null)
        if json.contains("\"plan\":null") || json.contains("plan_name\": null") {
            let domain_str = extract_json_str(json, "domain").unwrap_or("general".into());
            return Ok(IntentAST {
                domain:      Domain::from_str(&domain_str),
                standard:    None,
                plan_name:   None,
                params:      HashMap::new(),
                units:       HashMap::new(),
                constraints: vec![],
                raw_query:   query.to_string(),
            });
        }

        // Normal plan
        let domain_str  = extract_json_str(json, "domain").unwrap_or("unknown".into());
        let standard_s  = extract_json_str(json, "standard");
        let plan_name   = extract_json_str(json, "plan_name");
        let params      = extract_json_obj_f64(json, "params");
        let units       = extract_json_obj_str(json, "units");
        let constraints = extract_json_arr_str(json, "constraints");

        Ok(IntentAST {
            domain:      Domain::from_str(&domain_str),
            standard:    standard_s.map(|s| Standard::from_str(&s)),
            plan_name,
            params,
            units,
            constraints,
            raw_query:   query.to_string(),
        })
    }
}

// ── route_to_plan — Match IntentAST plan_name to available plans ──────

pub fn route_to_plan<'a>(intent: &IntentAST, available: &'a [String]) -> Option<&'a str> {
    let wanted = intent.plan_name.as_deref()?;
    // Direct match
    if let Some(p) = available.iter().find(|p| p.as_str() == wanted) {
        return Some(p.as_str());
    }
    // Prefix match (e.g. "loop:voltage_drop" won't match but "voltage_drop" might)
    let base = wanted.trim_start_matches("loop:");
    available.iter().find(|p| p.as_str() == base).map(|p| p.as_str())
}


// ── Minimal JSON helpers (no serde dependency) ────────────────────────

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    // skip whitespace and colon
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    } else if rest.starts_with("null") {
        None
    } else {
        None
    }
}

fn extract_json_obj_f64(json: &str, key: &str) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let pattern = format!("\"{}\"", key);
    let start = match json.find(&pattern) { Some(s) => s, None => return map };
    let rest = &json[start + pattern.len()..];
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    if !rest.starts_with('{') { return map; }
    let end = rest.find('}').unwrap_or(rest.len());
    let inner = &rest[1..end];
    for part in inner.split(',') {
        let mut kv = part.splitn(2, ':');
        let k = kv.next().map(|s| s.trim().trim_matches('"')).unwrap_or("");
        let v = kv.next().map(|s| s.trim()).unwrap_or("0");
        if let Ok(f) = v.parse::<f64>() {
            map.insert(k.to_string(), f);
        }
    }
    map
}

fn extract_json_obj_str(json: &str, key: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let pattern = format!("\"{}\"", key);
    let start = match json.find(&pattern) { Some(s) => s, None => return map };
    let rest = &json[start + pattern.len()..];
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    if !rest.starts_with('{') { return map; }
    let end = rest.find('}').unwrap_or(rest.len());
    let inner = &rest[1..end];
    for part in inner.split(',') {
        let mut kv = part.splitn(2, ':');
        let k = kv.next().map(|s| s.trim().trim_matches('"')).unwrap_or("");
        let v = kv.next()
            .map(|s| s.trim().trim_matches('"'))
            .unwrap_or("");
        if !k.is_empty() {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map
}

fn extract_json_arr_str(json: &str, key: &str) -> Vec<String> {
    let mut arr = vec![];
    let pattern = format!("\"{}\"", key);
    let start = match json.find(&pattern) { Some(s) => s, None => return arr };
    let rest = &json[start + pattern.len()..];
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    if !rest.starts_with('[') { return arr; }
    let end = rest.find(']').unwrap_or(rest.len());
    let inner = &rest[1..end];
    for part in inner.split(',') {
        let s = part.trim().trim_matches('"').to_string();
        if !s.is_empty() { arr.push(s); }
    }
    arr
}

/// Extract a numeric value for a JSON key (not inside an object)
fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    let end = rest.find(|c: char| c == ',' || c == '}' || c == '\n').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Parse a JSON array of floats: "[1.0,2.0,3.0]"
fn parse_f64_array(json: &str, key: &str) -> Vec<f64> {
    let pattern = format!("\"{}\"", key);
    let start = match json.find(&pattern) { Some(s) => s, None => return vec![] };
    let rest = &json[start + pattern.len()..];
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    if !rest.starts_with('[') { return vec![]; }
    let end = rest.find(']').unwrap_or(rest.len());
    let inner = &rest[1..end];
    inner.split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

fn extract_number_after(text: &str, after: &str) -> Option<f64> {
    let idx = text.find(after)?;
    let rest = text[idx + after.len()..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_number_before(text: &str, before: &str) -> Option<f64> {
    let idx = text.find(before)?;
    let prefix = text[..idx].trim_end();
    let start = prefix.rfind(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .map(|i| i + 1).unwrap_or(0);
    prefix[start..].parse().ok()
}

// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v2.0 — Intent Parser Interface
// Percy Rojas M. · Qomni AI Lab · 2026
//
// Converts natural language queries into IntentAST structs.
// The LLM backend is pluggable — use any OpenAI-compatible API.
//
// No direct API calls here. The caller provides a LlmBackend trait impl.
// This allows:
//   - OpenAI-compatible API via reqwest (production)
//   - Mock backend for tests
//   - Qomni Engine's own LLM router (Server5)
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

// ── Domain types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Domain {
    FireProtection,
    Electrical,
    Civil,
    Mechanical,
    Unknown(String),
}

impl Domain {
    pub fn from_str(s: &str) -> Self {
        match s {
            "fire_protection" | "incendios" | "nfpa" => Domain::FireProtection,
            "electrical"      | "electrico"            => Domain::Electrical,
            "civil"           | "estructural"          => Domain::Civil,
            "mechanical"      | "mecanica"             => Domain::Mechanical,
            other => Domain::Unknown(other.to_string()),
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            Domain::FireProtection => "fire_protection",
            Domain::Electrical     => "electrical",
            Domain::Civil          => "civil",
            Domain::Mechanical     => "mechanical",
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
/// from a natural language query in any engineering domain.
pub const SYSTEM_PROMPT: &str = r#"
You are the CRYS-L Intent Parser for Qomni Engine.
Given a natural language engineering query, extract a structured JSON IntentAST.

Rules:
1. "domain" must be one of: fire_protection, electrical, civil, mechanical
2. "plan_name" is the snake_case name of the best matching plan
3. "params" contains numeric values extracted from the query
4. "units" contains the physical unit for each param
5. "constraints" are conditions the result must satisfy
6. If domain is ambiguous, set domain to the closest match and add a constraint

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

    fn parse_json_response(&self, json: &str, raw_query: &str) -> Result<IntentAST, String> {
        // Minimal JSON parser — extract fields without serde dependency
        let domain_str   = extract_json_str(json, "domain").unwrap_or("unknown".into());
        let standard_str = extract_json_str(json, "standard");
        let plan_name    = extract_json_str(json, "plan_name");
        let params       = extract_json_obj_f64(json, "params");
        let units        = extract_json_obj_str(json, "units");
        let constraints  = extract_json_arr_str(json, "constraints");

        Ok(IntentAST {
            domain:      Domain::from_str(&domain_str),
            standard:    standard_str.map(|s| Standard::from_str(&s)),
            plan_name,
            params,
            units,
            constraints,
            raw_query:   raw_query.to_string(),
        })
    }
}

// ── Plan Router ───────────────────────────────────────────────────────

/// Given an IntentAST, select the best plan from the available plans.
/// Falls back to heuristics when the LLM did not specify a plan_name.
pub fn route_to_plan(intent: &IntentAST, available_plans: &[String]) -> Option<String> {
    // 1. LLM suggested a plan — verify it exists
    if let Some(ref suggested) = intent.plan_name {
        if available_plans.contains(suggested) {
            return Some(suggested.clone());
        }
    }
    // 2. Heuristic: domain + standard -> plan
    let candidate = match (&intent.domain, &intent.standard) {
        (Domain::FireProtection, Some(Standard::Nfpa13)) => "plan_sistema_incendios",
        (Domain::FireProtection, Some(Standard::Nfpa20)) => "plan_bomba_incendios",
        (Domain::FireProtection, Some(Standard::Nfpa72)) => "plan_alarma_incendios",
        (Domain::FireProtection, _)                       => "plan_sistema_incendios",
        (Domain::Electrical, _)                           => "plan_instalacion_electrica",
        _                                                  => return None,
    };
    if available_plans.contains(&candidate.to_string()) {
        Some(candidate.to_string())
    } else {
        None
    }
}

// ── Mock backend (for testing without API key) ────────────────────────

/// A mock LLM backend that returns canned responses based on keyword matching.
/// Used in tests and CI — no network required.
pub struct MockBackend;

impl LlmBackend for MockBackend {
    fn chat(&self, _system: &str, user: &str) -> Result<String, String> {
        let lower = user.to_lowercase();

        if lower.contains("rociador") || lower.contains("sprinkler") || lower.contains("incendio") {
            // Extract K factor and area if present
            let k = extract_number_after(&lower, "k=").unwrap_or(5.6);
            let area = extract_number_after(&lower, "m2").unwrap_or(
                extract_number_after(&lower, "metros").unwrap_or(1000.0)
            );
            return Ok(format!(r#"{{
  "domain": "fire_protection",
  "standard": "NFPA13",
  "plan_name": "plan_sistema_incendios",
  "params": {{"area": {:.1}, "K": {:.1}, "P_disponible": 60.0}},
  "units": {{"area": "m2", "K": "gpm/psi^0.5", "P_disponible": "psi"}},
  "constraints": ["P_disponible >= 7.0"]
}}"#, area, k));
        }

        if lower.contains("tension") || lower.contains("caida") || lower.contains("volt") {
            let i = extract_number_after(&lower, "a ").unwrap_or(100.0);
            let l = extract_number_after(&lower, "m ").unwrap_or(50.0);
            return Ok(format!(r#"{{
  "domain": "electrical",
  "standard": "IEC60038",
  "plan_name": "plan_instalacion_electrica",
  "params": {{"I": {:.1}, "L": {:.1}, "rho": 0.0000172, "A": 35.0}},
  "units": {{"I": "A", "L": "m", "rho": "ohm_m", "A": "mm2"}},
  "constraints": ["caida_tension <= 0.05"]
}}"#, i, l));
        }

        // Unknown domain
        Ok(r#"{"domain": "unknown", "standard": null, "plan_name": null, "params": {}, "units": {}, "constraints": []}"#.into())
    }
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

fn extract_number_after(text: &str, after: &str) -> Option<f64> {
    let idx = text.find(after)?;
    let rest = text[idx + after.len()..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

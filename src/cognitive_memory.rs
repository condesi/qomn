//! Cognitive Memory — persists execution experiences across sessions.
//!
//! After each successful plan or loop execution, call `save_experience()`.
//! Records are stored as append-only JSON lines in /opt/crysl/data/experiences.jsonl.
//! Use `load_recent(n)` to retrieve the last n experiences for recall/context.

use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const MEMORY_DIR:  &str = "/opt/crysl/data";
const MEMORY_FILE: &str = "/opt/crysl/data/experiences.jsonl";
const MAX_QUERY_LEN: usize = 150;

/// A single saved execution experience.
pub struct Experience<'a> {
    pub query:          &'a str,    // original user query (truncated to 150)
    pub structure:      &'a str,    // "loop" | "plan" | "single"
    pub plan:           &'a str,    // plan name or "loop:<oracle>"
    pub oracle:         &'a str,    // oracle function name
    pub result_summary: &'a str,    // "critical@20.0A=17.20V" | "no_critical" | "steps=4"
    pub success:        bool,       // did execution produce a meaningful result?
    pub score:          f32,        // 0.0–1.0 confidence/quality score
    pub elapsed_us:     f64,        // JIT execution time in microseconds
}

/// Current unix timestamp in seconds.
pub fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append one experience to the JSONL memory file.
/// Silently skips on any I/O error — memory is best-effort, never blocking.
pub fn save_experience(exp: &Experience<'_>) {
    let _ = create_dir_all(MEMORY_DIR);
    let q = exp.query.chars().take(MAX_QUERY_LEN).collect::<String>()
        .replace('"', "'").replace('\n', " ").replace('\r', "");
    let line = format!(
        r#"{{"ts":{ts},"query":"{q}","structure":"{st}","plan":"{pl}","oracle":"{or}","result":"{rs}","success":{ok},"score":{sc:.2},"elapsed_us":{el:.1}}}"#,
        ts = now_ts(),
        q  = q,
        st = exp.structure,
        pl = exp.plan,
        or = exp.oracle,
        rs = exp.result_summary.replace('"', "'"),
        ok = exp.success,
        sc = exp.score,
        el = exp.elapsed_us,
    );
    if let Ok(mut f) = OpenOptions::new()
        .create(true).append(true)
        .open(MEMORY_FILE)
    {
        let _ = writeln!(f, "{}", line);
    }
}

/// Load the last `n` experiences as raw JSON strings (newest first).
pub fn load_recent(n: usize) -> Vec<String> {
    use std::io::{BufRead, BufReader};
    let f = match std::fs::File::open(MEMORY_FILE) {
        Ok(f) => f,
        Err(_) => return vec![],
    };
    BufReader::new(f)
        .lines()
        .filter_map(|l| l.ok())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(n)
        .collect()
}

/// Count total saved experiences.
pub fn count() -> usize {
    use std::io::{BufRead, BufReader};
    let f = match std::fs::File::open(MEMORY_FILE) {
        Ok(f) => f,
        Err(_) => return 0,
    };
    BufReader::new(f).lines().filter(|l| l.is_ok()).count()
}

/// Return a one-line context hint for the LLM if similar past experiences exist.
/// Useful for injecting "I've solved this kind of query before" context.
pub fn recall_hint(query: &str) -> Option<String> {
    let lower = query.to_lowercase();
    let recent = load_recent(50);
    // Simple keyword overlap: count shared tokens
    let query_tokens: Vec<&str> = lower.split_whitespace().collect();
    let best = recent.iter().filter_map(|line| {
        let line_lower = line.to_lowercase();
        let overlap = query_tokens.iter()
            .filter(|t| t.len() > 3 && line_lower.contains(**t))
            .count();
        if overlap >= 3 { Some((overlap, line.clone())) } else { None }
    }).max_by_key(|(score, _)| *score);

    best.map(|(_, line)| {
        // Extract summary fields for a concise hint
        let result = line.split(r#""result":""#).nth(1)
            .and_then(|s| s.split('"').next())
            .unwrap_or("?");
        let plan = line.split(r#""plan":""#).nth(1)
            .and_then(|s| s.split('"').next())
            .unwrap_or("?");
        format!("[CogMem] Similar query solved via {} → {}", plan, result)
    })
}

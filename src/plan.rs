// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v2.1 — Plan Execution Engine
// Percy Rojas M. · Qomni AI Lab · 2026
//
// Executes plan_decl DAGs:
//   plan name(params) { step s1: oracle1(args) ... }
//
// Independent steps (no data-dependencies within a group) run concurrently
// via std::thread::scope. Groups execute serially in topological order.
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::ast::{PlanDecl, PlanStep, Expr};

// ── Node cache ─────────────────────────────────────────────────────────
// Key: (oracle_name, args_as_ordered_bits) → value
// Uses a fast integer hash of the f64 args to avoid floating-point equality issues.

#[derive(Default)]
pub struct OracleCache {
    store: HashMap<(String, u64), f64>,
    hits:  u64,
    misses: u64,
}

impl OracleCache {
    pub fn new() -> Self { Self::default() }

    fn args_key(args: &[f64]) -> u64 {
        // Combine bits of each argument with a simple polynomial hash
        args.iter().fold(0u64, |acc, &f| {
            acc.wrapping_mul(6364136223846793005)
               .wrapping_add(f.to_bits())
        })
    }

    pub fn get(&mut self, oracle: &str, args: &[f64]) -> Option<f64> {
        let k = (oracle.to_string(), Self::args_key(args));
        if let Some(&v) = self.store.get(&k) {
            self.hits += 1;
            Some(v)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, oracle: &str, args: &[f64], value: f64) {
        self.store.insert((oracle.to_string(), Self::args_key(args)), value);
    }

    pub fn stats(&self) -> (u64, u64, usize) { (self.hits, self.misses, self.store.len()) }
}

// JIT types — re-declared here to avoid circular dep with jit module
/// `(fn_addr_as_usize, n_params)` map — same layout as jit::JitFnTable
pub type JitFnMap = std::collections::HashMap<String, (usize, usize)>;

type OracleJitFn = unsafe extern "C" fn(*const f64, usize) -> f64;

#[inline]
unsafe fn call_fn_ptr(fn_addr: usize, args: &[f64]) -> f64 {
    let f: OracleJitFn = std::mem::transmute(fn_addr);
    f(args.as_ptr(), args.len())
}

// ── PlanResult ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StepResult {
    pub step:       String,
    pub oracle:     String,
    pub value:      f64,
    pub latency_ns: f64,
}

#[derive(Debug)]
pub struct PlanResult {
    pub plan_name:     String,
    pub inputs:        HashMap<String, f64>,
    pub steps:         Vec<StepResult>,
    pub total_ns:      f64,
    pub parallel_groups_count: usize,
    pub cache_hits:    u64,
}

impl PlanResult {
    /// Print a human-readable table
    pub fn display(&self) {
        println!("\nPlan: {}", self.plan_name);
        println!("{:-<60}", "");
        println!("  Inputs:");
        for (k, v) in &self.inputs {
            println!("    {:<20} = {:.4}", k, v);
        }
        println!("  Steps:");
        println!("  {:<20} {:<25} {:>12} {:>10}", "step", "oracle", "result", "ns");
        println!("  {:-<70}", "");
        for s in &self.steps {
            println!("  {:<20} {:<25} {:>12.4} {:>10.1}", s.step, s.oracle, s.value, s.latency_ns);
        }
        println!("  {:-<70}", "");
        println!("  Total compute: {:.1} ns  ({} parallel groups)", self.total_ns, self.parallel_groups_count);
    }

    /// JSON output
    pub fn to_json(&self) -> String {
        let steps_json: Vec<String> = self.steps.iter().map(|s| {
            // Canonicalize: -0.0 → +0.0 (same logical value, same hash)
            // Clamp: inf/-inf → bounded sentinel (never NaN in output)
            let v = if s.value == 0.0 { 0.0_f64 }  // -0.0 → +0.0
                    else if s.value.is_finite() { s.value }
                    else if s.value > 0.0 { 1e99_f64 }
                    else { -1e99_f64 };
            format!(
                r#"    {{"step":"{}", "oracle":"{}", "result":{:.6}, "latency_ns":{:.1}}}"#,
                s.step, s.oracle, v, s.latency_ns
            )
        }).collect();
        format!(
            "{{\n  \"plan\": \"{}\",\n  \"steps\": [\n{}\n  ],\n  \"total_ns\": {:.1},\n  \"cache_hits\": {}\n}}",
            self.plan_name,
            steps_json.join(",\n"),
            self.total_ns,
            self.cache_hits
        )
    }

    /// Human-readable narrative (Markdown). Used by Qomni to inject context.
    pub fn to_human(&self) -> String {
        let mut lines = vec![];
        lines.push(format!("## Engineering Computation — {}", self.plan_name));
        lines.push(String::new());
        if !self.inputs.is_empty() {
            lines.push("**Inputs:**".into());
            for (k, v) in &self.inputs {
                lines.push(format!("  - {}: {:.4} {}", k, v, unit_hint(k)));
            }
            lines.push(String::new());
        }
        lines.push("**Results:**".into());
        for s in &self.steps {
            let label = step_label(&s.step, &s.oracle);
            let unit  = result_unit(&s.oracle);
            lines.push(format!("  - **{}**: {:.4} {}", label, s.value, unit));
        }
        lines.push(String::new());
        lines.push(format!(
            "*Computed in {:.1} µs · {} parallel groups · {} cache hits*",
            self.total_ns / 1000.0,
            self.parallel_groups_count,
            self.cache_hits
        ));
        lines.join("\n")
    }
}

// ── LoopResult — Cognitive Compiler output ─────────────────────────────────

#[derive(Debug)]
pub struct LoopResult {
    pub oracle:         String,
    pub results:        Vec<(f64, f64)>,   // (loop_var_value, oracle_result)
    pub critical_point: Option<(f64, f64)>,
    pub total_ns:       f64,
}

impl LoopResult {
    /// Format as markdown with critical point highlighted + sample table
    pub fn to_human(
        &self,
        loop_label: &str,
        result_label: &str,
        loop_unit: &str,
        result_unit: &str,
        cond_op: &str,
        cond_val: f64,
        plan_label: &str,
    ) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("## Cognitive Simulation — {}", plan_label));
        lines.push(String::new());

        // Critical point
        // Auto-scale precision: use more decimals for small values
        let fmt_v = |v: f64| -> String {
            if v.abs() >= 10.0       { format!("{:.2}", v) }
            else if v.abs() >= 1.0   { format!("{:.3}", v) }
            else if v.abs() >= 0.01  { format!("{:.4}", v) }
            else                     { format!("{:.6}", v) }
        };
        if let Some((q_crit, v_crit)) = self.critical_point {
            lines.push(format!(
                "**⚠ Critical point**: {} = **{:.1} {}** → {} = {} {} ({} {} {})",
                loop_label, q_crit, loop_unit,
                result_label, fmt_v(v_crit), result_unit,
                cond_op, fmt_v(cond_val), result_unit
            ));
        } else {
            lines.push(format!(
                "**✓ Condition {} {:.1} {} never triggered** in range [{:.0}–{:.0} {}]",
                cond_op, cond_val, result_unit,
                self.results.first().map(|r| r.0).unwrap_or(0.0),
                self.results.last().map(|r| r.0).unwrap_or(0.0),
                loop_unit
            ));
        }
        lines.push(String::new());

        // Sample table: every ~10% of range + critical vicinity
        lines.push(format!("| {} ({}) | {} ({}) |", loop_label, loop_unit, result_label, result_unit));
        lines.push("|---:|---:|".to_string());
        let n = self.results.len();
        let stride = (n / 10).max(1);
        let mut shown: Vec<usize> = (0..n).step_by(stride).collect();
        if !shown.contains(&(n - 1)) { shown.push(n - 1); }

        // Also show 2 rows around critical point
        if let Some((q_crit, _)) = self.critical_point {
            for (i, &(q, _)) in self.results.iter().enumerate() {
                if (q - q_crit).abs() < self.results[1].0 - self.results[0].0 + 0.01 {
                    if i > 0 && !shown.contains(&(i-1)) { shown.push(i-1); }
                    if !shown.contains(&i) { shown.push(i); }
                    break;
                }
            }
        }
        shown.sort_unstable();
        shown.dedup();
        for i in shown {
            let (q, v) = self.results[i];
            let marker = if self.critical_point.map(|(cq,_)| (cq - q).abs() < 0.01).unwrap_or(false) { " ◀ critical" } else { "" };
            lines.push(format!("| {:.0} | {}{} |", q, fmt_v(v), marker));
        }
        lines.push(String::new());
        lines.push(format!(
            "*{} iterations · computed in {:.1} µs*",
            n, self.total_ns / 1000.0
        ));
        lines.join("\n")
    }
}

fn unit_hint(param: &str) -> &'static str {
    match param {
        "area_ft2"    => "ft²",  "area" => "m²",
        "K"           => "gpm/psi^0.5",
        "P_avail" | "P_psi" | "rated_P" => "psi",
        "hose_stream" | "Q_gpm" | "rated_Q" => "gpm",
        "eff"         => "(0-1)",
        "N_persons"   => "persons",
        "door_width_in" => "in",
        _             => "",
    }
}

fn step_label(step: &str, oracle: &str) -> String {
    match oracle {
        "nfpa13_area_density"    => "Design density (NFPA 13)".into(),
        "nfpa13_sprinkler"       => "Flow per head Q=K√P".into(),
        "nfpa13_sprinkler_count" => "Sprinkler heads required".into(),
        "nfpa13_demand_flow"     => "Total demand flow (incl. hose)".into(),
        "nfpa20_pump_hp"         => "Fire pump HP required".into(),
        "nfpa72_detector_count"  => "Smoke detectors required".into(),
        "nfpa20_shutoff_pressure"=> "Max shutoff pressure (140% rated)".into(),
        "nfpa20_150pct_flow"     => "Flow at 150% rated".into(),
        "nfpa20_head_pressure"   => "Pump head".into(),
        "nfpa101_egress_capacity"=> "Egress time".into(),
        "nfpa101_exit_width"     => "Required exit width".into(),
        _ => step.replace('_', " "),
    }
}

fn result_unit(oracle: &str) -> &'static str {
    match oracle {
        "nfpa13_area_density"    => "gpm/ft²",
        "nfpa13_sprinkler"       => "gpm",
        "nfpa13_sprinkler_count" => "heads",
        "nfpa13_demand_flow"     => "gpm",
        "nfpa20_pump_hp"         => "HP",
        "nfpa72_detector_count"  => "detectors",
        "nfpa20_shutoff_pressure"=> "psi",
        "nfpa20_150pct_flow"     => "gpm",
        "nfpa20_head_pressure"   => "psi",
        "nfpa101_egress_capacity"=> "min",
        "nfpa101_exit_width"     => "in",
        _ => "",
    }
}

// ── Oracle dispatch (free function, shareable across threads) ──────────

fn call_oracle_static(
    jit:      &Arc<Option<JitFnMap>>,
    fallback: &Arc<HashMap<String, fn(&[f64]) -> f64>>,
    name:     &str,
    args:     &[f64],
) -> Result<f64, String> {
    // 1. Try JIT fn table first (~3 ns)
    if let Some(ref tbl) = **jit {
        if let Some(&(fn_addr, n_params)) = tbl.get(name) {
            if args.len() == n_params || n_params == 0 {
                let result = unsafe { call_fn_ptr(fn_addr, args) };
                return Ok(result);
            }
        }
    }
    // 2. Fall back to built-in oracle table
    if let Some(f) = fallback.get(name) {
        return Ok(f(args));
    }
    Err(format!("oracle '{}' not found in JIT table or builtins", name))
}

/// Cache-aware wrapper: checks cache first, then dispatches.
/// Returns (value, was_cached).
fn call_oracle_cached(
    jit:      &Arc<Option<JitFnMap>>,
    fallback: &Arc<HashMap<String, fn(&[f64]) -> f64>>,
    cache:    &mut OracleCache,
    name:     &str,
    args:     &[f64],
) -> Result<(f64, bool), String> {
    if let Some(cached) = cache.get(name, args) {
        return Ok((cached, true));
    }
    let value = call_oracle_static(jit, fallback, name, args)?;
    cache.insert(name, args, value);
    Ok((value, false))
}

// ── Argument resolution (free functions, no self) ─────────────────────

fn resolve_args(args: &[Expr], scope: &HashMap<String, f64>) -> Result<Vec<f64>, String> {
    args.iter().map(|e| eval_expr(e, scope)).collect()
}

fn eval_expr(expr: &Expr, scope: &HashMap<String, f64>) -> Result<f64, String> {
    use crate::ast::Expr::*;
    use crate::ast::BinaryOp;
    match expr {
        Float(f) => Ok(*f),
        Int(i)   => Ok(*i as f64),
        Ident(name) => scope.get(name.as_str())
            .copied()
            .ok_or_else(|| format!("variable '{}' not in scope", name)),
        Binary(op, lhs, rhs) => {
            let l = eval_expr(lhs, scope)?;
            let r = eval_expr(rhs, scope)?;
            Ok(match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => if r == 0.0 { 0.0 } else { l / r },
                BinaryOp::Pow => l.powf(r),
                _ => return Err(format!("unsupported op in plan step: {:?}", op)),
            })
        }
        _ => Err(format!("complex expression in plan step args — use a let binding")),
    }
}

// ── PlanExecutor ──────────────────────────────────────────────────────

pub struct PlanExecutor<'a> {
    /// JIT fn table for fast oracle dispatch (~3 ns/call).
    jit_table: Arc<Option<JitFnMap>>,
    /// Fallback: oracle name -> fn pointer for non-JIT contexts
    fallback:  Arc<HashMap<String, fn(&[f64]) -> f64>>,
    /// Reference to the compiled plans in the program
    plans:     &'a [PlanDecl],
}

impl<'a> PlanExecutor<'a> {
    pub fn new(plans: &'a [PlanDecl]) -> Self {
        Self {
            jit_table: Arc::new(None),
            fallback:  Arc::new(Self::builtin_oracles()),
            plans,
        }
    }

    /// Supply the JIT fn-address map (binary crate extracts this from JitEngine::fn_table()).
    pub fn with_jit_map(mut self, map: JitFnMap) -> Self {
        self.jit_table = Arc::new(Some(map));
        self
    }

    /// Execute a named plan with parallel DAG dispatch.
    ///
    /// Independent steps (no data dependencies within a group) run concurrently
    /// via `std::thread::scope`. Groups serialize in topological order.
    pub fn execute(&self, plan_name: &str, params: HashMap<String, f64>)
        -> Result<PlanResult, String>
    {
        let plan = self.plans.iter()
            .find(|p| p.name == plan_name)
            .ok_or_else(|| format!("plan '{}' not found", plan_name))?;

        let groups = parallel_groups(plan);
        let n_groups = groups.len();

        let mut scope: HashMap<String, f64> = params.clone();
        let mut ordered_results: Vec<(usize, StepResult)> = Vec::with_capacity(plan.steps.len());
        let mut total_ns = 0.0f64;
        let mut cache = OracleCache::new();
        let mut cache_hits_total = 0u64;

        for group in &groups {
            if group.len() == 1 {
                // ── Single step — no threading overhead ──────────────
                let idx  = group[0];
                let step = &plan.steps[idx];
                let args = resolve_args(&step.args, &scope)?;
                let t0   = Instant::now();
                let (value, was_cached) = call_oracle_cached(
                    &self.jit_table, &self.fallback, &mut cache, &step.oracle, &args
                )?;
                let ns = if was_cached { 0.0 } else { t0.elapsed().as_nanos() as f64 };
                if was_cached { cache_hits_total += 1; }
                scope.insert(step.name.clone(), value);
                total_ns += ns;
                ordered_results.push((idx, StepResult {
                    step:       step.name.clone(),
                    oracle:     step.oracle.clone(),
                    value,
                    latency_ns: ns,
                }));
            } else {
                // ── Multiple independent steps — parallel dispatch ────
                // Resolve all args before spawning (reads snapshot of current scope).
                let step_data: Vec<(usize, String, String, Vec<f64>)> = group.iter()
                    .map(|&idx| {
                        let step = &plan.steps[idx];
                        let args = resolve_args(&step.args, &scope)?;
                        Ok((idx, step.name.clone(), step.oracle.clone(), args))
                    })
                    .collect::<Result<_, String>>()?;

                // Fan-out: run each step in its own thread
                let group_results: Vec<Result<(usize, StepResult), String>> =
                    std::thread::scope(|s| {
                        let handles: Vec<_> = step_data.into_iter().map(|(idx, name, oracle, args)| {
                            let jit  = Arc::clone(&self.jit_table);
                            let fall = Arc::clone(&self.fallback);
                            s.spawn(move || {
                                let t0    = Instant::now();
                                let value = call_oracle_static(&jit, &fall, &oracle, &args)?;
                                let ns    = t0.elapsed().as_nanos() as f64;
                                Ok((idx, StepResult { step: name, oracle, value, latency_ns: ns }))
                            })
                        }).collect();
                        handles.into_iter()
                            .map(|h| h.join().expect("oracle thread panicked"))
                            .collect()
                    });

                // Fan-in: merge results back into scope
                for res in group_results {
                    let (idx, sr) = res?;
                    scope.insert(sr.step.clone(), sr.value);
                    total_ns += sr.latency_ns;
                    ordered_results.push((idx, sr));
                }
            }
        }

        // Sort by original step declaration order for stable, deterministic output
        ordered_results.sort_by_key(|(idx, _)| *idx);
        let step_results = ordered_results.into_iter().map(|(_, sr)| sr).collect();

        Ok(PlanResult {
            plan_name:             plan_name.to_string(),
            inputs:                params,
            steps:                 step_results,
            total_ns,
            parallel_groups_count: n_groups,
            cache_hits:            cache_hits_total,
        })
    }

    // ── Cognitive Compiler: Loop / Simulation Executor ────────────
    /// Execute a single oracle in a loop over a range of values.
    /// oracle: name of the JIT-compiled or builtin oracle
    /// loop_pos: which argument position is the loop variable (0-based)
    /// fixed_args: all other args (in oracle's param order, excluding loop_pos)
    /// range_start/end/step: loop range
    /// cond_op/cond_val: condition to detect critical point ("<", "<=", ">", ">=")
    pub fn execute_loop(
        &self,
        oracle:      &str,
        loop_pos:    usize,
        fixed_args:  &[f64],
        range_start: f64,
        range_end:   f64,
        step_size:   f64,
        cond_op:     &str,
        cond_val:    f64,
    ) -> Result<LoopResult, String> {
        let t_start = std::time::Instant::now();
        let mut results: Vec<(f64, f64)> = Vec::new();
        let mut critical: Option<(f64, f64)> = None;

        let mut q = range_start;
        let n_iters = ((range_end - range_start) / step_size).ceil() as usize + 1;
        results.reserve(n_iters);

        while q <= range_end + step_size * 0.01 {
            // Build arg slice: insert loop var at loop_pos
            let n_total = fixed_args.len() + 1;
            let mut args: Vec<f64> = Vec::with_capacity(n_total);
            let mut fi = 0usize;
            for i in 0..n_total {
                if i == loop_pos {
                    args.push(q);
                } else if fi < fixed_args.len() {
                    args.push(fixed_args[fi]);
                    fi += 1;
                }
            }

            let val = call_oracle_static(&self.jit_table, &self.fallback, oracle, &args)
                .map_err(|e| format!("loop oracle '{}': {}", oracle, e))?;

            let triggered = match cond_op {
                "<"  => val < cond_val,
                "<=" => val <= cond_val,
                ">"  => val > cond_val,
                ">=" => val >= cond_val,
                _    => false,
            };
            if triggered && critical.is_none() {
                critical = Some((q, val));
            }

            results.push((q, val));
            q += step_size;
        }

        let total_ns = t_start.elapsed().as_nanos() as f64;
        Ok(LoopResult {
            oracle: oracle.to_string(),
            results,
            critical_point: critical,
            total_ns,
        })
    }

    // ── Built-in oracle table (pure Rust, no JIT required) ────────
    // Used as fallback when JIT is not available (debug builds) or
    // when the JIT table does not contain the oracle.

    pub fn builtin_oracles() -> HashMap<String, fn(&[f64]) -> f64> {
        let mut m: HashMap<String, fn(&[f64]) -> f64> = HashMap::new();

        // ── Fire protection (NFPA) — Spanish naming (sistema_incendios.crys) ──
        m.insert("nfpa13_sprinkler".into(),  |a| { let (k, p) = (a[0], a[1]); k * p.sqrt() });
        m.insert("nfpa13_area_densidad".into(), |_| 0.15);
        m.insert("nfpa13_area_density".into(), |_| 0.15);
        m.insert("nfpa13_demanda".into(), |a| {
            let (q, k, h) = (a[0], a[1], a[2]);
            if k == 0.0 { return 0.0; }
            (q / k).powi(2) + h
        });
        m.insert("nfpa13_count".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            a[0] / a[1]
        });
        m.insert("nfpa13_sprinkler_count".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            (a[0] / a[1]).ceil()  // NFPA 13: always round UP — 7.69 heads → 8
        });
        m.insert("nfpa13_demand_flow".into(), |a| a[0] * a[1] + a[2]);
        m.insert("nfpa20_bomba_hp".into(), |a| {
            if a[2] == 0.0 { return 0.0; }
            a[0] * a[1] / (3960.0 * a[2])
        });
        m.insert("nfpa20_pump_hp".into(), |a| {
            if a[2] == 0.0 { return 0.0; }
            a[0] * a[1] / (3960.0 * a[2])
        });
        m.insert("nfpa20_head_pressure".into(), |a| a[0] * 0.433);
        m.insert("nfpa20_presion".into(), |a| a[0] * 0.433);
        m.insert("nfpa20_shutoff_pressure".into(), |a| a[0] * 1.4);
        m.insert("nfpa20_150pct_flow".into(), |a| a[0] * 1.5);
        m.insert("nfpa72_detectores".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            (a[0] / (a[1] * a[1])).ceil()
        });
        m.insert("nfpa72_detector_count".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            (a[0] / (a[1] * a[1])).ceil()  // NFPA 72: fractional detectors round UP
        });
        m.insert("nfpa72_beam_length".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            a[0] / a[1]
        });
        m.insert("nfpa72_cobertura".into(), |a| a[0] * a[0]);
        m.insert("nfpa101_egress_capacity".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            a[0] / a[1]
        });
        m.insert("nfpa101_travel_distance".into(), |a| {
            if a[0] == 0.0 { return 0.0; }
            a[1] / a[0]
        });
        m.insert("nfpa101_exit_width".into(), |a| {
            if a[1] == 0.0 { return 0.0; }
            a[0] / a[1]
        });

        // ── Electrical (IEC 60038) ─────────────────────────────────
        m.insert("corriente_cc".into(), |a| { if a[1] == 0.0 { 0.0 } else { a[0] / a[1] } });
        m.insert("caida_tension".into(), |a| {
            if a[3] == 0.0 { return 0.0; }
            2.0 * a[0] * a[1] * a[2] / a[3]
        });
        m.insert("corriente_carga".into(), |a| {
            let d = a[1] * 1.732 * a[2];
            if d == 0.0 { 0.0 } else { (a[0] * 1000.0) / d }
        });

        // ── General math helpers ───────────────────────────────────
        m.insert("sqrt".into(),   |a| a[0].sqrt());
        m.insert("square".into(), |a| a[0] * a[0]);
        m.insert("abs".into(),    |a| a[0].abs());
        m.insert("min2".into(),   |a| a[0].min(a[1]));
        m.insert("max2".into(),   |a| a[0].max(a[1]));
        m.insert("clamp".into(),  |a| a[0].max(a[1]).min(a[2]));
        m.insert("log2".into(),  |a| a[0].log2());
        m.insert("ln".into(),    |a| a[0].ln());
        m.insert("exp2".into(),  |a| (2.0_f64).powf(a[0]));
        m.insert("pow2".into(),  |a| (2.0_f64).powf(a[0]));
        m.insert("log10".into(), |a| a[0].log10());
        m.insert("pow".into(),   |a| a[0].powf(a[1]));
        m.insert("math_ceil".into(),  |a| a[0].ceil());
        m.insert("math_floor".into(), |a| a[0].floor());
        m.insert("math_round".into(), |a| a[0].round());

        // ── Full fire system (NFPA 13/14/20) US units ─────────────────────
        // Hazen-Williams friction loss: h_f(PSI) = 4.73*L*Q^1.852/(2.31*C^1.852*D^4.87)
        // Q in GPM, D in inches, L in feet
        m.insert("hw_friction_psi".into(), |a| {
            let (q, c, d, l) = (a[0], a[1], a[2], a[3]);
            if d == 0.0 || c == 0.0 { return f64::INFINITY; }
            4.73 * l * q.powf(1.852) / (2.31 * c.powf(1.852) * d.powf(4.87))
        });
        m.insert("elevation_psi".into(), |a| a[0] / 2.31);
        m.insert("fittings_psi".into(), |a| a[0] * 0.25);  // NFPA 13 simplified: 25% of friction
        m.insert("total_system_psi".into(), |a| a[0] + a[1] + a[2] + a[3]);
        m.insert("pump_hp_from_system".into(), |a| {
            if a[2] == 0.0 { return 0.0; }
            a[0] * a[1] / (3960.0 * a[2])
        });

        // ── Cybersecurity oracles (password, crypto) ───────────────
        // password_entropy(charset_size, length) → bits
        m.insert("password_entropy".into(), |a| {
            let charset = a[0];
            let length  = a[1];
            if charset <= 0.0 || length <= 0.0 { return 0.0; }
            length * charset.log2()
        });
        // brute_force_years(entropy_bits, attempts_per_sec) → years
        m.insert("brute_force_years".into(), |a| {
            let bits = a[0];
            let rate = a[1];
            if bits <= 0.0 || rate <= 0.0 { return 0.0; }
            // For very high entropy (>1024 bits), cap to avoid infinity
            if bits > 1024.0 { return 1.0e300; }
            let combos = (2.0_f64).powf(bits);
            let seconds = combos / rate;
            seconds / 31536000.0
        });
        // aes_key_strength(key_bits) → score 0-10+
        m.insert("aes_key_strength".into(), |a| {
            a[0] / 256.0 * 10.0
        });
        // bcrypt_hashrate(cost_factor, base_gpu_rate) -> effective h/s
        // base_gpu_rate = hashrate at cost=5 (baseline). Each +1 cost = 2x slower.
        m.insert("bcrypt_hashrate".into(), |a| {
            let cost = a[0];
            let base_rate = a[1]; // h/s at cost 5
            if cost < 4.0 || base_rate <= 0.0 { return 0.0; }
            // cost 5 = baseline, each +1 halves the rate
            base_rate / (2.0_f64).powf(cost - 5.0)
        });
        // bcrypt_crack_seconds(keyspace, effective_hashrate) -> seconds
        m.insert("bcrypt_crack_seconds".into(), |a| {
            let keyspace = a[0];
            let rate = a[1];
            if rate <= 0.0 { return f64::INFINITY; }
            keyspace / rate
        });
        // dict_crack_seconds(dict_size, effective_hashrate) -> seconds
        m.insert("dict_crack_seconds".into(), |a| {
            let dict_size = a[0];
            let rate = a[1];
            if rate <= 0.0 { return f64::INFINITY; }
            dict_size / rate
        });
        // keyspace(charset_size, length) -> total combinations
        m.insert("keyspace".into(), |a| {
            let charset = a[0];
            let length = a[1];
            if charset <= 0.0 || length <= 0.0 { return 0.0; }
            if length > 40.0 { return f64::INFINITY; }
            charset.powf(length)
        });
        // seconds_to_human_unit(seconds) -> value in best unit (returns the value, unit determined by plan)
        m.insert("secs_to_years".into(), |a| {
            a[0] / 31536000.0
        });
        m.insert("secs_to_hours".into(), |a| {
            a[0] / 3600.0
        });

        m
    }
}

// ── Plan Dependency Analysis ───────────────────────────────────────────

/// Returns groups of steps that can run in parallel (Kahn's topological sort).
/// Each inner Vec is an independent group; groups must execute in order.
pub fn parallel_groups(plan: &PlanDecl) -> Vec<Vec<usize>> {
    let n = plan.steps.len();
    let step_names: Vec<&str> = plan.steps.iter().map(|s| s.name.as_str()).collect();

    let mut groups:    Vec<Vec<usize>> = vec![];
    let mut scheduled: Vec<bool>       = vec![false; n];

    while scheduled.iter().any(|s| !s) {
        let mut group = vec![];
        for i in 0..n {
            if scheduled[i] { continue; }
            let step = &plan.steps[i];
            let deps_satisfied = step_deps(step, &step_names).iter().all(|dep| {
                let dep_idx = step_names.iter().position(|&s| s == *dep);
                dep_idx.map_or(true, |idx| scheduled[idx])
            });
            if deps_satisfied { group.push(i); }
        }
        if group.is_empty() { break; } // cycle guard
        for &idx in &group { scheduled[idx] = true; }
        groups.push(group);
    }
    groups
}

fn step_deps<'a>(step: &'a PlanStep, all_steps: &[&str]) -> Vec<&'a str> {
    let mut deps = vec![];
    for arg in &step.args { collect_idents(arg, all_steps, &mut deps); }
    deps
}

fn collect_idents<'a>(expr: &'a Expr, all_steps: &[&str], out: &mut Vec<&'a str>) {
    use crate::ast::Expr::*;
    match expr {
        Ident(name) => { if all_steps.contains(&name.as_str()) { out.push(name.as_str()); } }
        Binary(_, l, r) => { collect_idents(l, all_steps, out); collect_idents(r, all_steps, out); }
        Unary(_, e) => collect_idents(e, all_steps, out),
        _ => {}
    }
}

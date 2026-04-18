// ═══════════════════════════════════════════════════════════════════════
// QOMN v2.0 — Plan Execution Engine
// Percy Rojas M. · Qomni AI Lab · 2026
//
// Executes plan_decl DAGs:
//   plan name(params) { step s1: oracle1(args) ... }
//
// Steps run in declaration order. Output of step N is available
// as a variable in subsequent steps by the step name.
// Independent steps (no data deps) can be flagged for parallel dispatch.
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::time::Instant;

use crate::ast::{PlanDecl, PlanStep, Expr};

// JIT types — re-declared here to avoid circular dep with jit module
// (jit is only in the binary crate; plan is in both lib and bin)
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
        println!("  Total compute: {:.1} ns", self.total_ns);
    }

    /// JSON output
    pub fn to_json(&self) -> String {
        let steps_json: Vec<String> = self.steps.iter().map(|s| {
            format!(
                r#"    {{"step":"{}", "oracle":"{}", "result":{:.6}, "latency_ns":{:.1}}}"#,
                s.step, s.oracle, s.value, s.latency_ns
            )
        }).collect();
        format!(
            "{{\n  \"plan\": \"{}\",\n  \"steps\": [\n{}\n  ],\n  \"total_ns\": {:.1}\n}}",
            self.plan_name,
            steps_json.join(",\n"),
            self.total_ns
        )
    }
}

// ── PlanExecutor ──────────────────────────────────────────────────────

pub struct PlanExecutor<'a> {
    /// JIT fn table for fast oracle dispatch (~3 ns/call).
    /// Populated by the binary crate via `with_jit_map`.
    jit_table: Option<JitFnMap>,
    /// Fallback: oracle name -> fn pointer for non-JIT contexts
    fallback:  HashMap<String, fn(&[f64]) -> f64>,
    /// Reference to the compiled plans in the program
    plans:     &'a [PlanDecl],
}

impl<'a> PlanExecutor<'a> {
    pub fn new(plans: &'a [PlanDecl]) -> Self {
        Self {
            jit_table: None,
            fallback:  Self::builtin_oracles(),
            plans,
        }
    }

    /// Supply the JIT fn-address map (binary crate extracts this from JitEngine::fn_table()).
    pub fn with_jit_map(mut self, map: JitFnMap) -> Self {
        self.jit_table = Some(map);
        self
    }

    /// Execute a named plan with the given parameter values.
    ///
    /// `params` is a map from parameter name -> float value.
    pub fn execute(&self, plan_name: &str, params: HashMap<String, f64>)
        -> Result<PlanResult, String>
    {
        let plan = self.plans.iter()
            .find(|p| p.name == plan_name)
            .ok_or_else(|| format!("plan '{}' not found", plan_name))?;

        let mut scope: HashMap<String, f64> = params.clone();
        let mut step_results = vec![];
        let mut total_ns = 0.0;

        for step in &plan.steps {
            // Resolve arguments from scope
            let args = self.resolve_args(&step.args, &scope)?;

            // Dispatch oracle
            let t0 = Instant::now();
            let value = self.call_oracle(&step.oracle, &args)?;
            let ns = t0.elapsed().as_nanos() as f64;

            // Store result in scope under step name
            scope.insert(step.name.clone(), value);
            total_ns += ns;

            step_results.push(StepResult {
                step:       step.name.clone(),
                oracle:     step.oracle.clone(),
                value,
                latency_ns: ns,
            });
        }

        Ok(PlanResult {
            plan_name: plan_name.to_string(),
            inputs:    params,
            steps:     step_results,
            total_ns,
        })
    }

    // ── Oracle dispatch ───────────────────────────────────────────

    fn call_oracle(&self, name: &str, args: &[f64]) -> Result<f64, String> {
        // 1. Try JIT fn table first (~3 ns)
        if let Some(ref tbl) = self.jit_table {
            if let Some(&(fn_addr, n_params)) = tbl.get(name) {
                if args.len() == n_params || n_params == 0 {
                    let result = unsafe { call_fn_ptr(fn_addr, args) };
                    return Ok(result);
                }
            }
        }
        // 2. Fall back to built-in oracle table
        if let Some(f) = self.fallback.get(name) {
            return Ok(f(args));
        }
        Err(format!("oracle '{}' not found in JIT table or builtins", name))
    }

    // ── Argument resolution ──────────────────────────────────────

    fn resolve_args(&self, args: &[Expr], scope: &HashMap<String, f64>)
        -> Result<Vec<f64>, String>
    {
        args.iter().map(|e| self.eval_expr(e, scope)).collect()
    }

    fn eval_expr(&self, expr: &Expr, scope: &HashMap<String, f64>) -> Result<f64, String> {
        use crate::ast::Expr::*;
        use crate::ast::BinaryOp;
        match expr {
            Float(f) => Ok(*f),
            Int(i)   => Ok(*i as f64),
            Ident(name) => scope.get(name.as_str())
                .copied()
                .ok_or_else(|| format!("variable '{}' not in scope", name)),
            Binary(op, lhs, rhs) => {
                let l = self.eval_expr(lhs, scope)?;
                let r = self.eval_expr(rhs, scope)?;
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

    // ── Built-in oracle table (pure Rust, no JIT required) ────────
    // These mirror the JIT-compiled oracle bodies exactly.
    // Used as fallback when JIT is not available (e.g., debug builds).

    fn builtin_oracles() -> HashMap<String, fn(&[f64]) -> f64> {
        let mut m: HashMap<String, fn(&[f64]) -> f64> = HashMap::new();

        // ── Fire protection (NFPA) ────────────────────────────────
        m.insert("nfpa13_sprinkler".into(),  |a| {
            let (k, p) = (a[0], a[1]);
            k * p.sqrt()
        });
        m.insert("nfpa13_demanda".into(), |a| {
            // P_demand = (Q/K)^2 + head_loss
            let (q, k, h) = (a[0], a[1], a[2]);
            if k == 0.0 { return 0.0; }
            (q / k).powi(2) + h
        });
        m.insert("nfpa20_bomba_hp".into(), |a| {
            let (q, p, eff) = (a[0], a[1], a[2]);
            if eff == 0.0 { return 0.0; }
            q * p / (3960.0 * eff)
        });
        m.insert("nfpa20_presion".into(), |a| {
            a[0] * 0.433
        });
        m.insert("nfpa72_cobertura".into(), |a| {
            a[0] * a[0]
        });
        m.insert("nfpa72_detectores".into(), |a| {
            let (area, spacing) = (a[0], a[1]);
            if spacing == 0.0 { return 0.0; }
            (area / (spacing * spacing)).ceil()
        });

        // ── Electrical (IEC 60038) ─────────────────────────────────
        m.insert("corriente_cc".into(), |a| {
            let (v, z) = (a[0], a[1]);
            if z == 0.0 { return 0.0; }
            v / z
        });
        m.insert("caida_tension".into(), |a| {
            let (i, l, rho, cross) = (a[0], a[1], a[2], a[3]);
            if cross == 0.0 { return 0.0; }
            2.0 * i * l * rho / cross
        });
        m.insert("corriente_carga".into(), |a| {
            let (p_kw, v, fp) = (a[0], a[1], a[2]);
            let denom = v * 1.732 * fp;
            if denom == 0.0 { return 0.0; }
            (p_kw * 1000.0) / denom
        });

        // ── General math helpers ───────────────────────────────────
        m.insert("sqrt".into(),  |a| a[0].sqrt());
        m.insert("square".into(), |a| a[0] * a[0]);
        m.insert("abs".into(),   |a| a[0].abs());
        m.insert("min2".into(),  |a| a[0].min(a[1]));
        m.insert("max2".into(),  |a| a[0].max(a[1]));
        m.insert("clamp".into(), |a| a[0].max(a[1]).min(a[2]));

        m
    }
}

// ── Plan Dependency Analysis ───────────────────────────────────────────

/// Returns a list of steps that can run in parallel (no intra-step data deps).
/// Two steps are independent if neither uses the output of the other.
pub fn parallel_groups(plan: &PlanDecl) -> Vec<Vec<usize>> {
    let n = plan.steps.len();
    let step_names: Vec<&str> = plan.steps.iter().map(|s| s.name.as_str()).collect();

    let mut groups: Vec<Vec<usize>> = vec![];
    let mut scheduled = vec![false; n];

    // Kahn's algorithm: group steps with no unscheduled deps
    while scheduled.iter().any(|s| !s) {
        let mut group = vec![];
        for i in 0..n {
            if scheduled[i] { continue; }
            let step = &plan.steps[i];
            // Check if all args reference only params or already-scheduled steps
            let deps_satisfied = step_deps(step, &step_names).iter().all(|dep| {
                // dep is either a param name (always available) or a step name
                let dep_idx = step_names.iter().position(|&s| s == *dep);
                dep_idx.map_or(true, |idx| scheduled[idx])
            });
            if deps_satisfied {
                group.push(i);
            }
        }
        if group.is_empty() { break; } // cycle guard
        for &idx in &group { scheduled[idx] = true; }
        groups.push(group);
    }
    groups
}

/// Extract step names that a plan step's args depend on.
fn step_deps<'a>(step: &'a PlanStep, all_steps: &[&str]) -> Vec<&'a str> {
    let mut deps = vec![];
    for arg in &step.args {
        collect_idents(arg, all_steps, &mut deps);
    }
    deps
}

fn collect_idents<'a>(expr: &'a Expr, all_steps: &[&str], out: &mut Vec<&'a str>) {
    use crate::ast::Expr::*;
    match expr {
        Ident(name) => {
            if all_steps.contains(&name.as_str()) {
                out.push(name.as_str());
            }
        }
        Binary(_, l, r) => {
            collect_idents(l, all_steps, out);
            collect_idents(r, all_steps, out);
        }
        Unary(_, e) => collect_idents(e, all_steps, out),
        _ => {}
    }
}

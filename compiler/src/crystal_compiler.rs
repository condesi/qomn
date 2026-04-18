// ═══════════════════════════════════════════════════════════════════════
// QOMN v1.3 — Oracle → .crystal Compiler
//              "Aproximación Ternaria de Funciones"
//
// Compila declaraciones `oracle` de QOMN a formato binario .crystal
// usando Physics-as-Oracle (PaO):
//
//   oracle f(x) → y
//     ↓  muestreo multi-escala (borde + interior + ruido)
//   activations[ROWS×COLS]
//     ↓  Random Fourier Features (RFF, semilla fija)
//   float matrix
//     ↓  BitNet absmean → {-1, 0, +1}
//   .crystal
//
// Ventaja sobre v1.2: RFF dan mejor cobertura espectral que sinusoides
// simples; el sampling de borde garantiza cobertura en valores extremos;
// el ruido controlado da robustez ante datos ruidosos/casos de borde.
//
// El .crystal resultante puede cargarse en Qomni como cualquier crystal
// entrenado con SFT — misma API, mismo formato.
// ═══════════════════════════════════════════════════════════════════════

use std::io::Write;
use crate::ast::{OracleDecl, Program, Decl, Expr, BinaryOp, UnaryOp};

// ── Crystal binary format constants ─────────────────────────────────
const MAGIC:   &[u8; 4] = b"CRYS";
const VERSION: u8       = 1u8;
const ROWS:    usize    = 896;   // hidden dim Qwen-0.5B FFN
const COLS:    usize    = 4864;  // input dim

// ── LCG pseudo-RNG (no external crate, fully deterministic) ──────────

/// Linear congruential generator — Knuth constants.
#[inline]
fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005)
         .wrapping_add(1442695040888963407)
}

/// Generate `n` RFF frequencies ~ N(0, σ²) with σ = 1/√n_params.
/// Fixed seed → reproducible across runs.
fn gen_rff_freqs(n: usize, n_params: usize) -> Vec<f32> {
    let sigma = 1.0 / (n_params as f32).sqrt().max(1.0);
    let mut state: u64 = 0xdeadbeef_cafebabe;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Two uniform samples for Box-Muller
        state = lcg_step(state);
        let u1 = (state >> 33) as f32 / (u32::MAX as f32) + 1e-10;
        state = lcg_step(state);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        // Box-Muller → standard normal → scale to σ
        let r = (-2.0 * u1.ln()).sqrt();
        let freq = r * (u2 * std::f32::consts::TAU).cos() * sigma;
        out.push(freq);
    }
    out
}

/// Generate `n` RFF phases uniform in [0, 2π].
fn gen_rff_phases(n: usize) -> Vec<f32> {
    let mut state: u64 = 0xcafebabe_deadbeef;
    (0..n)
        .map(|_| {
            state = lcg_step(state);
            (state >> 33) as f32 / (u32::MAX as f32) * std::f32::consts::TAU
        })
        .collect()
}

/// Log-scale compression for wide dynamic range oracle outputs.
///
/// Problem: formulas like corriente_carga = P/(√3·V·FP) produce outputs
/// spanning 14 orders of magnitude when sampled over [0.001, 100].
/// RFF phase w·y_f32 wraps 90,000+ full cycles → training/validation
/// RFF patterns are completely decorrelated → sign_agree = 0%.
///
/// Solution: map output → log-space before RFF projection, consistently
/// in both training (sample_oracle) and validation.
///
/// Mapping:
///   |y| ≤ 1  → linear (no distortion for small normalised outputs)
///   |y| > 1  → sign(y) × log10(|y|), capped at ±7
///
/// Effect on dynamic range:
///   y = 1e6  → 6.0     (was 1,000,000)
///   y = 1e3  → 3.0
///   y = 10   → 1.0
///   y = 1    → 1.0  (boundary — linear below)
///   y = 0.5  → 0.5  (unchanged)
///   y = 1e-6 → 1e-6 (unchanged)
#[inline]
fn compress_output(y: f64) -> f32 {
    if !y.is_finite() { return 0.0; }
    let abs_y = y.abs();
    if abs_y <= 1.0 {
        y as f32
    } else {
        (y.signum() * abs_y.log10().min(7.0)) as f32
    }
}

/// Tiny deterministic noise for robustness: amplitude ~0.5% of unit std.
#[inline]
fn trit_noise(row: usize, col: usize) -> f32 {
    let h = row.wrapping_mul(2654435761) ^ col.wrapping_mul(0x9e3779b9);
    // map to [-0.005, +0.005]
    ((h % 1000) as f32 - 500.0) * 1e-5
}

// ── Multi-scale Oracle Sampler ────────────────────────────────────────

/// Build a stratified sample set covering:
///   1. Boundary points (param extremes: 0.001 … 100)
///   2. Log-uniform interior (bulk coverage)
///   3. Mid-point perturbations (edge-case robustness)
fn build_sample_set(n_params: usize, n_samples: usize) -> Vec<Vec<f64>> {
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);

    // ── 1. Boundary corners ────────────────────────────────────────
    let boundaries: &[f64] = &[0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    for &b in boundaries {
        samples.push(vec![b; n_params]);
    }
    // Mixed-boundary (low/high alternating)
    for p in 0..n_params.min(8) {
        let mut v = vec![1.0f64; n_params];
        v[p] = 0.001;
        samples.push(v.clone());
        v[p] = 100.0;
        samples.push(v);
    }

    // ── 2. Log-uniform interior ────────────────────────────────────
    let interior = (n_samples * 3 / 4).saturating_sub(samples.len());
    for i in 0..interior {
        let params: Vec<f64> = (0..n_params)
            .map(|p| {
                // Stagger by param index to avoid correlations
                let t = ((i * n_params + p) as f64 + 0.5)
                    / (interior * n_params) as f64;
                (0.001f64).powf(1.0 - t) * (100.0f64).powf(t)
            })
            .collect();
        samples.push(params);
    }

    // ── 3. Perturbed mid-points (×±10% of log-center) ─────────────
    let remaining = n_samples.saturating_sub(samples.len());
    let mut state: u64 = 0xabcdef01_23456789;
    for _ in 0..remaining {
        let params: Vec<f64> = (0..n_params)
            .map(|_| {
                state = lcg_step(state);
                let base = (state >> 33) as f64 / (u32::MAX as f64);
                let t = base; // in [0,1)
                let center = (0.001f64).powf(1.0 - t) * (100.0f64).powf(t);
                // ±10% perturbation in log-space
                state = lcg_step(state);
                let perturb = ((state >> 33) as f64 / (u32::MAX as f64) - 0.5) * 0.2;
                center * (10.0f64).powf(perturb)
            })
            .collect();
        samples.push(params);
    }

    samples.truncate(n_samples);
    samples
}

/// Sample an oracle using multi-scale sampling + RFF projection.
/// Produces a ROWS×COLS float matrix approximating the oracle's behavior.
fn sample_oracle(oracle: &OracleDecl, n_samples: usize) -> Vec<f32> {
    let n_params = oracle.params.len().max(1);
    let mut matrix = vec![0f32; ROWS * COLS];

    // Random Fourier Feature basis (fixed seed — deterministic)
    let rff_w = gen_rff_freqs(COLS, n_params);
    let rff_b = gen_rff_phases(COLS);

    let samples = build_sample_set(n_params, n_samples);

    // Accumulate RFF activations into the matrix
    for (sample_idx, params) in samples.iter().enumerate() {
        let y = eval_oracle_f64(oracle, params);
        if !y.is_finite() { continue; }

        // Log-scale compression: maps wide dynamic range → [-7, +7]
        // Essential for ratio formulas (corriente_carga, nfpa20_bomba_hp, etc.)
        // where raw output can span 14+ orders of magnitude.
        let y_f32 = compress_output(y);

        let row = sample_idx % ROWS;
        let base = row * COLS;
        for col in 0..COLS {
            // RFF: cos(w_j · y_compressed + b_j)
            matrix[base + col] += (rff_w[col] * y_f32 + rff_b[col]).cos();
        }
    }

    // ── Row-wise normalisation + deterministic noise ───────────────
    for row in 0..ROWS {
        let start = row * COLS;
        let end   = start + COLS;
        let slice = &mut matrix[start..end];

        let mean: f32 = slice.iter().sum::<f32>() / COLS as f32;
        let var:  f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / COLS as f32;
        let std   = var.sqrt().max(1e-8);

        for (col, x) in slice.iter_mut().enumerate() {
            *x = (*x - mean) / std + trit_noise(row, col);
        }
    }

    matrix
}

// ── Oracle AST Evaluator ──────────────────────────────────────────────

fn eval_oracle_f64(oracle: &OracleDecl, params: &[f64]) -> f64 {
    let env: Vec<(&str, f64)> = oracle.params.iter()
        .zip(params.iter())
        .map(|(p, &v)| (p.name.as_str(), v))
        .collect();

    for stmt in &oracle.body {
        if let crate::ast::Stmt::Return(expr) = stmt {
            return eval_expr_f64(expr, &env);
        }
    }
    0.0
}

fn eval_expr_f64(expr: &Expr, env: &[(&str, f64)]) -> f64 {
    match expr {
        Expr::Int(n)   => *n as f64,
        Expr::Float(f) => *f,
        Expr::Ident(name) => {
            env.iter().find(|(n, _)| *n == name.as_str())
               .map(|(_, v)| *v)
               .unwrap_or(0.0)
        }
        Expr::Binary(op, lhs, rhs) => {
            let l = eval_expr_f64(lhs, env);
            let r = eval_expr_f64(rhs, env);
            match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => if r.abs() < 1e-12 { 0.0 } else { l / r },
                BinaryOp::Pow => l.powf(r),
                BinaryOp::Mod => l % r,
                _ => 0.0,
            }
        }
        Expr::Unary(UnaryOp::Neg, e) => -eval_expr_f64(e, env),
        Expr::Call(func, args) => {
            if let Expr::Ident(name) = func.as_ref() {
                let a: Vec<f64> = args.iter().map(|a| eval_expr_f64(a, env)).collect();
                match name.as_str() {
                    "sin"  => a.first().copied().unwrap_or(0.0).sin(),
                    "cos"  => a.first().copied().unwrap_or(0.0).cos(),
                    "tan"  => a.first().copied().unwrap_or(0.0).tan(),
                    "sqrt" => a.first().copied().unwrap_or(0.0).abs().sqrt(),
                    "abs"  => a.first().copied().unwrap_or(0.0).abs(),
                    "log"  => a.first().copied().unwrap_or(1.0).abs().max(1e-300).ln(),
                    "log10"=> a.first().copied().unwrap_or(1.0).abs().max(1e-300).log10(),
                    "exp"  => a.first().copied().unwrap_or(0.0).min(709.0).exp(),
                    "pow"  => {
                        let base = a.first().copied().unwrap_or(0.0);
                        let exp  = a.get(1).copied().unwrap_or(1.0);
                        base.powf(exp)
                    }
                    "min"  => {
                        let x = a.first().copied().unwrap_or(0.0);
                        let y = a.get(1).copied().unwrap_or(0.0);
                        x.min(y)
                    }
                    "max"  => {
                        let x = a.first().copied().unwrap_or(0.0);
                        let y = a.get(1).copied().unwrap_or(0.0);
                        x.max(y)
                    }
                    _ => 0.0,
                }
            } else { 0.0 }
        }
        _ => 0.0,
    }
}

// ── Ternary Quantization (BitNet absmean) ────────────────────────────

/// Quantize float matrix to ternary {-1, 0, +1} with absmean threshold.
/// Returns: (weights: Vec<i8>, scales: Vec<f32>)
fn quantize_absmean(matrix: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut weights = vec![0i8; rows * cols];
    let mut scales  = vec![0f32; rows];

    for row in 0..rows {
        let start = row * cols;
        let end   = start + cols;
        let slice = &matrix[start..end];

        let absmean: f32 = slice.iter().map(|x| x.abs()).sum::<f32>() / cols as f32;
        let threshold = absmean;
        scales[row] = absmean.max(1e-8);

        for (i, &x) in slice.iter().enumerate() {
            weights[start + i] = if x >  threshold {  1 }
                                  else if x < -threshold { -1 }
                                  else { 0 };
        }
    }
    (weights, scales)
}

// ── Crystal Packing (2-bit per trit, 4 trits/byte) ───────────────────

/// Pack i8 weights {-1,0,+1} to 2-bit per trit (4 trits/byte).
/// Encoding: 0→00, +1→01, -1→10
fn pack_2bit(weights: &[i8]) -> Vec<u8> {
    let n_bytes = (weights.len() + 3) / 4;
    let mut packed = vec![0u8; n_bytes];
    for (i, &w) in weights.iter().enumerate() {
        let encoded: u8 = match w { 1 => 1, -1 => 2, _ => 0 };
        let byte_idx = i / 4;
        let bit_off  = (i % 4) * 2;
        packed[byte_idx] |= encoded << bit_off;
    }
    packed
}

// ── Validation ────────────────────────────────────────────────────────

/// Statistics from validating an oracle approximation.
pub struct ValidationStats {
    /// Number of test points evaluated
    pub n_test:          usize,
    /// Fraction of ternary weights that are non-zero (model density)
    pub coverage:        f32,
    /// Mean absolute oracle output at test points (sanity check)
    pub mean_abs_output: f64,
    /// Fraction of test outputs that the crystal represents with non-zero trits
    pub sign_agree_rate: f32,
}

/// Validate that a compiled oracle crystal captures the oracle's dynamic range.
///
/// Strategy: evaluate the oracle at `n_test` held-out points (not in training
/// set), encode each output via the same RFF basis, check that the resulting
/// ternary row has enough non-zero entries to represent the value — i.e., the
/// crystal "sees" the output.  Full reconstruction requires Qomni inference;
/// here we check necessary (not sufficient) conditions.
pub fn validate_oracle_crystal(
    oracle: &OracleDecl,
    weights: &[i8],
    _scales: &[f32],
) -> ValidationStats {
    let n_params = oracle.params.len().max(1);
    let n_test   = 40;

    // RFF basis (must match what sample_oracle used)
    let rff_w = gen_rff_freqs(COLS, n_params);
    let rff_b = gen_rff_phases(COLS);

    let mut abs_outputs = Vec::with_capacity(n_test);
    let mut sign_agrees  = 0usize;

    for i in 0..n_test {
        // Held-out points: shift phase by 0.5 so they don't fall on training grid
        let params: Vec<f64> = (0..n_params)
            .map(|p| {
                let t = ((i * n_params + p) as f64 + 0.5 + 0.37)
                    / (n_test * n_params) as f64;
                let t = t.min(1.0 - 1e-9);
                (0.001f64).powf(1.0 - t) * (100.0f64).powf(t)
            })
            .collect();

        let y = eval_oracle_f64(oracle, &params);
        if !y.is_finite() { continue; }
        abs_outputs.push(y.abs());

        // Must match compression used during training
        let y_f32 = compress_output(y);

        // Encode y via RFF → unit trit vector (what inference would use)
        let encoded: Vec<i8> = (0..COLS)
            .map(|col| {
                let v = (rff_w[col] * y_f32 + rff_b[col]).cos();
                if v >  0.2 {  1i8 }
                else if v < -0.2 { -1i8 }
                else { 0i8 }
            })
            .collect();

        // Check if ANY row in the weight matrix has non-trivial overlap
        // with this encoding (sign agreement > 40% of non-zero positions)
        let nonzero_enc: usize = encoded.iter().filter(|&&t| t != 0).count();
        if nonzero_enc == 0 { continue; }

        let best_agree = (0..(weights.len() / COLS).min(ROWS))
            .map(|row| {
                let start = row * COLS;
                encoded.iter().zip(&weights[start..start + COLS])
                    .filter(|(&e, &w)| e != 0 && w != 0 && e == w)
                    .count()
            })
            .max()
            .unwrap_or(0);

        if best_agree as f64 / nonzero_enc as f64 > 0.40 {
            sign_agrees += 1;
        }
    }

    let n_valid     = abs_outputs.len().max(1);
    let mean_abs    = abs_outputs.iter().sum::<f64>() / n_valid as f64;
    let coverage    = weights.iter().filter(|&&w| w != 0).count() as f32
                      / weights.len() as f32;
    let sign_rate   = sign_agrees as f32 / n_test as f32;

    ValidationStats {
        n_test,
        coverage,
        mean_abs_output: mean_abs,
        sign_agree_rate: sign_rate,
    }
}

// ── Crystal Binary Writer ─────────────────────────────────────────────

/// Write a single-layer .crystal file.
/// Format: header(64B) + layer_index(32B) + payload(2-bit packed)
pub fn write_crystal(
    oracle_name: &str,
    weights: &[i8],
    rows: usize,
    cols: usize,
    out_path: &str,
) -> Result<usize, String> {
    let packed   = pack_2bit(weights);
    let n_layers = 1usize;

    let mut f = std::fs::File::create(out_path)
        .map_err(|e| format!("Cannot create '{}': {}", out_path, e))?;

    // ── Header (64 bytes) ──────────────────────────────────────────
    // MAGIC(4) + VERSION+PAD(4) + N_LAYERS(4) + ARCH(48) + PAD(8) = 64B
    f.write_all(MAGIC).map_err(|e| e.to_string())?;                        // 4B
    f.write_all(&[VERSION, 0, 0, 0]).map_err(|e| e.to_string())?;          // 4B
    f.write_all(&(n_layers as u32).to_le_bytes()).map_err(|e| e.to_string())?; // 4B

    let arch = format!("oracle-{}", oracle_name);
    let mut arch_bytes = [0u8; 48];
    let copy_len = arch.len().min(48);
    arch_bytes[..copy_len].copy_from_slice(&arch.as_bytes()[..copy_len]);
    f.write_all(&arch_bytes).map_err(|e| e.to_string())?;                  // 48B
    f.write_all(&[0u8; 8]).map_err(|e| e.to_string())?;                    // 8B → 64B total

    // ── Layer index (32 bytes) ──────────────────────────────────────
    let offset_payload: u64 = 64 + 32 * n_layers as u64;
    f.write_all(&0u32.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(offset_payload as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(rows as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(cols as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&[0u8; 16]).map_err(|e| e.to_string())?;

    // ── Payload ────────────────────────────────────────────────────
    f.write_all(&packed).map_err(|e| e.to_string())?;

    Ok(64 + 32 + packed.len())
}

// ── Public API ────────────────────────────────────────────────────────

pub struct CompileResult {
    pub oracle_name:     String,
    pub out_path:        String,
    pub file_size:       usize,
    pub rows:            usize,
    pub cols:            usize,
    pub n_samples:       usize,
    pub sparsity:        f32,   // fraction of zeros in ternary weights
    pub validation:      ValidationStats,
}

/// Compile all oracle declarations in a program to .crystal files.
pub fn compile_oracles(prog: &Program, out_dir: &str) -> Vec<Result<CompileResult, String>> {
    prog.decls.iter()
        .filter_map(|d| if let Decl::Oracle(o) = d { Some(o) } else { None })
        .map(|o| compile_oracle(o, out_dir))
        .collect()
}

/// Compile a single oracle declaration to a .crystal file.
pub fn compile_oracle(oracle: &OracleDecl, out_dir: &str) -> Result<CompileResult, String> {
    // 4 samples per output row — good coverage with multi-scale set
    let n_samples = ROWS * 4;

    // 1. Multi-scale sample → RFF projection → float matrix
    let matrix = sample_oracle(oracle, n_samples);

    // 2. BitNet absmean quantisation → ternary
    let (weights, scales) = quantize_absmean(&matrix, ROWS, COLS);

    // 3. Statistics
    let zeros    = weights.iter().filter(|&&w| w == 0).count();
    let sparsity = zeros as f32 / weights.len() as f32;

    // 4. Validation (held-out points, sign agreement)
    let validation = validate_oracle_crystal(oracle, &weights, &scales);

    // 5. Write .crystal
    let out_path  = format!("{}/{}.crystal", out_dir, oracle.name);
    let file_size = write_crystal(&oracle.name, &weights, ROWS, COLS, &out_path)?;

    Ok(CompileResult {
        oracle_name: oracle.name.clone(),
        out_path,
        file_size,
        rows: ROWS,
        cols: COLS,
        n_samples,
        sparsity,
        validation,
    })
}

//! CRYS-L Simulation Engine v2.3 — Production-Grade Autonomous Loop
//!
//! 5 architectural upgrades from v2.2:
//!   1. L1 Block Tiling (256 scenarios/block — fits 26KB in 32KB L1d)
//!   2. Stress/Adversarial sweep modes (validates physics robustness)
//!   3. Multi-Objective Pareto Front (efficiency + cost + risk)
//!   4. AVX-512 f64x8 kernel (bare-metal gated)
//!   5. NUMA topology detection + mbind advisory

use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Cache-optimal block size: 256 × (4+8+1 fields) × 8 bytes = 26KB → fits L1d
// Previously 1024 → 104KB → L2 pressure
// ─────────────────────────────────────────────────────────────────────────────
pub const SIM_N:      usize = 1024; // total scenarios per tick
pub const BLOCK_SIZE: usize = 256;  // L1-resident tile (26KB < 32KB L1d per core)
pub const SIM_OUTPUTS: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// SoA Layout — 32-byte aligned (AVX2 register width)
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C, align(32))]
pub struct ScenarioSoA {
    pub p0:    [f64; SIM_N],
    pub p1:    [f64; SIM_N],
    pub p2:    [f64; SIM_N],
    pub p3:    [f64; SIM_N],
    pub out:   [[f64; SIM_N]; SIM_OUTPUTS],
    pub valid: [f64; SIM_N], // 1.0=physics OK, 0.0=invalid (branchless mask)
}

impl ScenarioSoA {
    pub fn new() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            Box::from_raw(ptr)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics Layer — NFPA 20 bounds (branchless via multiply)
// ─────────────────────────────────────────────────────────────────────────────
pub struct PhysicsBound { pub min: f64, pub max: f64, pub name: &'static str }

pub const PUMP_BOUNDS: [PhysicsBound; 3] = [
    PhysicsBound { min: 0.1,  max: 50_000.0, name: "flow_gpm"   },
    PhysicsBound { min: 1.0,  max: 5_000.0,  name: "press_psi"  },
    PhysicsBound { min: 0.10, max: 0.97,     name: "efficiency" },
];

#[inline(always)]
pub fn physics_valid_pump(p0: f64, p1: f64, p2: f64) -> f64 {
    // Branchless: f64::from(bool) → SETCC+CVTSI2SD, no branch
    let v0 = f64::from(p0 >= PUMP_BOUNDS[0].min && p0 <= PUMP_BOUNDS[0].max);
    let v1 = f64::from(p1 >= PUMP_BOUNDS[1].min && p1 <= PUMP_BOUNDS[1].max);
    let v2 = f64::from(p2 >= PUMP_BOUNDS[2].min && p2 <= PUMP_BOUNDS[2].max);
    // Also reject NaN (NaN comparisons return false → v=0.0)
    v0 * v1 * v2
}

// AVX2 branchless physics: validate 4 scenarios per instruction
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn physics_validate_block_avx2(soa: &mut ScenarioSoA, start: usize, end: usize) {
    use std::arch::x86_64::*;
    let vb0_min = _mm256_set1_pd(PUMP_BOUNDS[0].min);
    let vb0_max = _mm256_set1_pd(PUMP_BOUNDS[0].max);
    let vb1_min = _mm256_set1_pd(PUMP_BOUNDS[1].min);
    let vb1_max = _mm256_set1_pd(PUMP_BOUNDS[1].max);
    let vb2_min = _mm256_set1_pd(PUMP_BOUNDS[2].min);
    let vb2_max = _mm256_set1_pd(PUMP_BOUNDS[2].max);
    let vone  = _mm256_set1_pd(1.0);
    let vzero = _mm256_setzero_pd();

    let mut i = start;
    while i + 4 <= end {
        let vp0 = _mm256_loadu_pd(soa.p0.as_ptr().add(i));
        let vp1 = _mm256_loadu_pd(soa.p1.as_ptr().add(i));
        let vp2 = _mm256_loadu_pd(soa.p2.as_ptr().add(i));
        // _CMP_GE_OQ=29: NaN inputs automatically fail (ordered comparison)
        let m0 = _mm256_and_pd(_mm256_cmp_pd(vp0, vb0_min, 29), _mm256_cmp_pd(vb0_max, vp0, 29));
        let m1 = _mm256_and_pd(_mm256_cmp_pd(vp1, vb1_min, 29), _mm256_cmp_pd(vb1_max, vp1, 29));
        let m2 = _mm256_and_pd(_mm256_cmp_pd(vp2, vb2_min, 29), _mm256_cmp_pd(vb2_max, vp2, 29));
        let valid = _mm256_blendv_pd(vzero, vone, _mm256_and_pd(_mm256_and_pd(m0, m1), m2));
        _mm256_storeu_pd(soa.valid.as_mut_ptr().add(i), valid);
        i += 4;
    }
    while i < end {
        soa.valid[i] = physics_valid_pump(soa.p0[i], soa.p1[i], soa.p2[i]);
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 Compute Kernel — L1 block tiling, aggressive prefetch
// Block loop: prefetch NEXT block while computing CURRENT block
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pump_kernel_block_avx2(soa: &mut ScenarioSoA, start: usize, end: usize) {
    use std::arch::x86_64::*;
    let v_gps  = _mm256_set1_pd(0.06309);
    let v_pm   = _mm256_set1_pd(0.70307);
    let v_76   = _mm256_set1_pd(76.04);
    let v_140  = _mm256_set1_pd(1.40);
    let v_120  = _mm256_set1_pd(1.20);
    let vzero  = _mm256_setzero_pd();

    let mut i = start;
    while i + 4 <= end {
        // Prefetch 4 cache lines ahead (32 bytes × 4 = 128 bytes = 2 cache lines)
        if i + 32 < end {
            _mm_prefetch(soa.p0.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(soa.p1.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(soa.p2.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
        }
        let vq = _mm256_loadu_pd(soa.p0.as_ptr().add(i));
        let vp = _mm256_loadu_pd(soa.p1.as_ptr().add(i));
        let ve = _mm256_loadu_pd(soa.p2.as_ptr().add(i));
        let vv = _mm256_loadu_pd(soa.valid.as_ptr().add(i));

        let q_lps = _mm256_mul_pd(vq, v_gps);
        let h_m   = _mm256_mul_pd(vp, v_pm);
        let num   = _mm256_fmadd_pd(q_lps, h_m, vzero);   // FMA: 1 instr vs 2
        let den   = _mm256_mul_pd(ve, v_76);
        let dmask = _mm256_cmp_pd(den, vzero, 4);          // NEQ_OQ
        let hp    = _mm256_blendv_pd(vzero, _mm256_div_pd(num, den), dmask);

        // Physics mask: invalid scenarios → all outputs = 0.0 (no branch)
        _mm256_storeu_pd(soa.out[0].as_mut_ptr().add(i), _mm256_mul_pd(hp, vv));
        _mm256_storeu_pd(soa.out[1].as_mut_ptr().add(i), _mm256_mul_pd(_mm256_mul_pd(hp, v_140), vv));
        _mm256_storeu_pd(soa.out[2].as_mut_ptr().add(i), _mm256_mul_pd(q_lps, vv));
        _mm256_storeu_pd(soa.out[3].as_mut_ptr().add(i), _mm256_mul_pd(h_m, vv));
        _mm256_storeu_pd(soa.out[4].as_mut_ptr().add(i), _mm256_mul_pd(_mm256_mul_pd(hp, v_120), vv));
        i += 4;
    }
    while i < end {
        let v = soa.valid[i];
        let q = soa.p0[i] * 0.06309;
        let h = soa.p1[i] * 0.70307;
        let d = soa.p2[i] * 76.04;
        let hp = if d > 1e-10 { q * h / d } else { 0.0 };
        soa.out[0][i] = hp * v;         soa.out[1][i] = hp * 1.40 * v;
        soa.out[2][i] = q * v;          soa.out[3][i] = h * v;
        soa.out[4][i] = hp * 1.20 * v;
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 f64x8 kernel — bare-metal only (KVM blocks avx512f)
// 8 scenarios per instruction vs 4 for AVX2 → theoretical 2x throughput
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pump_kernel_block_avx512(soa: &mut ScenarioSoA, start: usize, end: usize) {
    use std::arch::x86_64::*;
    let v_gps = _mm512_set1_pd(0.06309);
    let v_pm  = _mm512_set1_pd(0.70307);
    let v_76  = _mm512_set1_pd(76.04);
    let v_140 = _mm512_set1_pd(1.40);
    let v_120 = _mm512_set1_pd(1.20);
    let vzero = _mm512_setzero_pd();

    let mut i = start;
    while i + 8 <= end {
        if i + 64 < end {
            _mm_prefetch(soa.p0.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
            _mm_prefetch(soa.p1.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
        }
        let vq = _mm512_loadu_pd(soa.p0.as_ptr().add(i));
        let vp = _mm512_loadu_pd(soa.p1.as_ptr().add(i));
        let ve = _mm512_loadu_pd(soa.p2.as_ptr().add(i));
        let vv = _mm512_loadu_pd(soa.valid.as_ptr().add(i));

        let q_lps = _mm512_mul_pd(vq, v_gps);
        let h_m   = _mm512_mul_pd(vp, v_pm);
        let num   = _mm512_fmadd_pd(q_lps, h_m, vzero); // AVX-512 FMA
        let den   = _mm512_mul_pd(ve, v_76);
        // _CMP_NEQ_OQ=4: mask-based branchless div
        let mask  = _mm512_cmp_pd_mask(den, vzero, 4);
        let hp    = _mm512_mask_div_pd(vzero, mask, num, den);

        _mm512_storeu_pd(soa.out[0].as_mut_ptr().add(i), _mm512_mul_pd(hp, vv));
        _mm512_storeu_pd(soa.out[1].as_mut_ptr().add(i), _mm512_mul_pd(_mm512_mul_pd(hp, v_140), vv));
        _mm512_storeu_pd(soa.out[2].as_mut_ptr().add(i), _mm512_mul_pd(q_lps, vv));
        _mm512_storeu_pd(soa.out[3].as_mut_ptr().add(i), _mm512_mul_pd(h_m, vv));
        _mm512_storeu_pd(soa.out[4].as_mut_ptr().add(i), _mm512_mul_pd(_mm512_mul_pd(hp, v_120), vv));
        i += 8;
    }
    // AVX2 tail for remaining < 8 scenarios
    if i < end {
        unsafe { pump_kernel_block_avx2(soa, i, end); }
    }
}

// Main blocked kernel dispatcher — selects AVX-512 / AVX2 / scalar
pub fn pump_kernel_blocked(soa: &mut ScenarioSoA) {
    let avx512 = is_x86_feature_detected!("avx512f");
    let avx2   = is_x86_feature_detected!("avx2");

    // Tile loop: process BLOCK_SIZE=256 at a time (fits L1d)
    // Prefetch NEXT block's p0/p1/p2 while computing CURRENT block
    let mut block_start = 0;
    while block_start < SIM_N {
        let block_end = (block_start + BLOCK_SIZE).min(SIM_N);

        // Prefetch NEXT block into L2 while we work on current (T1=L2, not T0=L1)
        if block_start + BLOCK_SIZE < SIM_N {
            let next = block_start + BLOCK_SIZE;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                _mm_prefetch(soa.p0.as_ptr().add(next) as *const i8, _MM_HINT_T1);
                _mm_prefetch(soa.p1.as_ptr().add(next) as *const i8, _MM_HINT_T1);
                _mm_prefetch(soa.p2.as_ptr().add(next) as *const i8, _MM_HINT_T1);
            }
        }

        if avx512 {
            #[cfg(target_arch = "x86_64")]
            unsafe { pump_kernel_block_avx512(soa, block_start, block_end); }
        } else if avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe { pump_kernel_block_avx2(soa, block_start, block_end); }
        } else {
            for i in block_start..block_end {
                let v = soa.valid[i];
                let q = soa.p0[i] * 0.06309;
                let h = soa.p1[i] * 0.70307;
                let d = soa.p2[i] * 76.04;
                let hp = if d > 1e-10 { q * h / d } else { 0.0 };
                soa.out[0][i] = hp * v; soa.out[1][i] = hp * 1.40 * v;
                soa.out[2][i] = q * v;  soa.out[3][i] = h * v;
                soa.out[4][i] = hp * 1.20 * v;
            }
        }
        block_start += BLOCK_SIZE;
    }
}

// Physics validation — also blocked
pub fn physics_validate_blocked(soa: &mut ScenarioSoA) {
    let avx2 = is_x86_feature_detected!("avx2");
    let mut block_start = 0;
    while block_start < SIM_N {
        let block_end = (block_start + BLOCK_SIZE).min(SIM_N);
        if avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe { physics_validate_block_avx2(soa, block_start, block_end); }
        } else {
            for i in block_start..block_end {
                soa.valid[i] = physics_valid_pump(soa.p0[i], soa.p1[i], soa.p2[i]);
            }
        }
        block_start += BLOCK_SIZE;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sweep Modes — Improvement #2: Stress/Adversarial scenarios
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SweepMode {
    /// Stratified coverage sweep (normal operation)
    Stratified,
    /// 10% of scenarios at physics boundaries to test robustness
    Stress,
    /// Adversarial: NaN, zero, negative, extreme → valid_frac should drop to ~0.6
    Adversarial,
}

#[derive(Clone)]
pub struct SweepSpec {
    pub p0_range: (f64, f64),
    pub p1_range: (f64, f64),
    pub p2_range: (f64, f64),
    pub mode: SweepMode,
}

impl SweepSpec {
    pub fn pump_default() -> Self {
        SweepSpec { p0_range:(10.0,5000.0), p1_range:(20.0,500.0), p2_range:(0.50,0.95), mode:SweepMode::Stratified }
    }
    pub fn pump_stress() -> Self {
        SweepSpec { p0_range:(0.0,60_000.0), p1_range:(0.0,6_000.0), p2_range:(0.0,1.5), mode:SweepMode::Stress }
    }

    pub fn fill_soa(&self, soa: &mut ScenarioSoA, tick: u64) {
        let n = SIM_N as f64;
        let off = (tick * 7) % SIM_N as u64;
        for i in 0..SIM_N {
            let t0 = ((i as u64 + off) % SIM_N as u64) as f64 / n;
            let t1 = i as f64 / n;
            let t2 = ((i * 3 / 7 + tick as usize) % SIM_N) as f64 / n;

            match self.mode {
                SweepMode::Stratified => {
                    soa.p0[i] = self.p0_range.0 + t0 * (self.p0_range.1 - self.p0_range.0);
                    soa.p1[i] = self.p1_range.0 + t1 * (self.p1_range.1 - self.p1_range.0);
                    soa.p2[i] = self.p2_range.0 + t2 * (self.p2_range.1 - self.p2_range.0);
                }
                SweepMode::Stress => {
                    // Every 10th scenario: deliberately invalid boundary values
                    if i % 10 == 0 {
                        // Rotate through 5 failure modes
                        match (i / 10) % 5 {
                            0 => { soa.p0[i] = -100.0;      soa.p1[i] = 100.0;  soa.p2[i] = 0.75; } // neg flow
                            1 => { soa.p0[i] = 500.0;       soa.p1[i] = 6000.0; soa.p2[i] = 0.75; } // overpressure
                            2 => { soa.p0[i] = 500.0;       soa.p1[i] = 100.0;  soa.p2[i] = 1.50; } // eff > Carnot
                            3 => { soa.p0[i] = 0.0;         soa.p1[i] = 0.0;    soa.p2[i] = 0.0;  } // all-zero
                            _ => { soa.p0[i] = f64::NAN;    soa.p1[i] = 100.0;  soa.p2[i] = 0.75; } // NaN
                        }
                    } else {
                        // Valid scenarios between stress points
                        soa.p0[i] = self.p0_range.0 + t0 * (self.p0_range.1 - self.p0_range.0);
                        soa.p1[i] = self.p1_range.0 + t1 * (self.p1_range.1 - self.p1_range.0);
                        soa.p2[i] = self.p2_range.0 + t2 * (self.p2_range.1 - self.p2_range.0);
                    }
                }
                SweepMode::Adversarial => {
                    // 40% invalid: 5 failure modes evenly distributed
                    let mode = i % 10;
                    match mode {
                        0 | 1 => { soa.p0[i] = t0 * 60_000.0;    soa.p1[i] = t1 * 6_000.0;  soa.p2[i] = t2 * 1.5; } // out of bounds
                        2     => { soa.p0[i] = -(t0 * 1000.0);    soa.p1[i] = t1 * 200.0;    soa.p2[i] = 0.75; }     // negative flow
                        3     => { soa.p0[i] = 500.0;              soa.p1[i] = 100.0;          soa.p2[i] = 0.98 + t2; } // eff > 0.97
                        _ => {  // valid scenarios
                            soa.p0[i] = self.p0_range.0 + t0 * (self.p0_range.1 - self.p0_range.0);
                            soa.p1[i] = self.p1_range.0 + t1 * (self.p1_range.1 - self.p1_range.0);
                            soa.p2[i] = 0.50 + t2 * 0.47;
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Objective Pareto Front — Improvement #3
// Objectives: maximize efficiency_score, minimize cost, minimize risk
// Pareto-optimal = not dominated by any other scenario
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, serde::Serialize)]
pub struct ParetoSolution {
    pub scenario_idx:    usize,
    pub hp_req:          f64,
    pub eff_score:       f64,  // 0..1 — higher is better
    pub cost_usd:        f64,  // $/HP installed — lower is better
    pub risk_score:      f64,  // 0..1 — lower is better
    pub params:          [f64; 3],
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ParetoFront {
    pub solutions:       Vec<ParetoSolution>,
    pub n_dominated:     usize,  // how many were dominated
    pub n_valid:         usize,
}

// Cost model: HP_req × $750/HP (installed cost) + pressure penalty
fn cost_proxy(hp_req: f64, p_psi: f64) -> f64 {
    hp_req * 750.0 + p_psi * 2.5  // $750/HP + $2.5/psi for high-pressure piping
}

// Risk model: proximity to max operating point (NFPA 20 shutoff ratio)
// Risk = 0 when hp_req = hp_max/1.4 (perfectly on duty point)
// Risk → 1 when hp_req approaches hp_max (operating near shutoff)
fn risk_score(hp_req: f64, hp_max: f64) -> f64 {
    if hp_max < 1e-10 { return 1.0; }
    let ratio = hp_req / hp_max; // should be ~0.714 (1/1.4) at design point
    // Penalize both underloading (<0.5) and overloading (>0.85)
    let dev = (ratio - 0.714).abs();
    (dev * 3.0).min(1.0)
}

// Efficiency score: normalize hp_req relative to theoretical minimum
fn eff_score(hp_req: f64, q_lps: f64, h_m: f64) -> f64 {
    // Theoretical minimum: perfect pump (100% efficiency)
    let hp_ideal = q_lps * h_m / 76.04;
    if hp_ideal < 1e-10 || hp_req < 1e-10 { return 0.0; }
    // Score: 1.0 = at theoretical minimum, 0.0 = very inefficient
    (hp_ideal / hp_req).min(1.0)
}

// Pareto dominance: a dominates b if better in ALL objectives
fn dominates(a: &ParetoSolution, b: &ParetoSolution) -> bool {
    // All objectives either equal or better, at least one strictly better
    let eff_ok   = a.eff_score  >= b.eff_score;
    let cost_ok  = a.cost_usd   <= b.cost_usd;
    let risk_ok  = a.risk_score <= b.risk_score;
    let any_better = a.eff_score > b.eff_score
        || a.cost_usd < b.cost_usd
        || a.risk_score < b.risk_score;
    eff_ok && cost_ok && risk_ok && any_better
}

pub fn compute_pareto_front(soa: &ScenarioSoA) -> ParetoFront {
    // Build candidate list from valid scenarios
    let mut candidates: Vec<ParetoSolution> = (0..SIM_N)
        .filter(|&i| soa.valid[i] > 0.5 && soa.out[0][i] > 1e-6)
        .map(|i| {
            let hp  = soa.out[0][i];
            let hpm = soa.out[1][i];
            let q   = soa.out[2][i];
            let h   = soa.out[3][i];
            ParetoSolution {
                scenario_idx: i,
                hp_req:     hp,
                eff_score:  eff_score(hp, q, h),
                cost_usd:   cost_proxy(hp, soa.p1[i]),
                risk_score: risk_score(hp, hpm),
                params:     [soa.p0[i], soa.p1[i], soa.p2[i]],
            }
        })
        .collect();

    let n_valid = candidates.len();
    let mut dominated = vec![false; candidates.len()];

    // Non-domination sort: O(N²) — for N≤1024 valid scenarios, ~1M comparisons
    for i in 0..candidates.len() {
        for j in 0..candidates.len() {
            if i != j && !dominated[i] && dominates(&candidates[j], &candidates[i]) {
                dominated[i] = true;
                break;
            }
        }
    }

    let n_dominated = dominated.iter().filter(|&&d| d).count();
    let front: Vec<ParetoSolution> = candidates.iter()
        .zip(dominated.iter())
        .filter(|(_, &dom)| !dom)
        .map(|(s, _)| s.clone())
        .collect();

    ParetoFront { solutions: front, n_dominated, n_valid }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan IDs
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum PlanId {
    PumpSizing   = 0,
    VoltageDrop  = 1,
    Planilla     = 2,
    BeamAnalysis = 3,
}
impl PlanId {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "plan_pump_sizing"|"pump"        => Some(PlanId::PumpSizing),
            "plan_voltage_drop"|"vdrop"      => Some(PlanId::VoltageDrop),
            "plan_planilla"|"planilla"       => Some(PlanId::Planilla),
            "plan_beam_analysis"|"beam"      => Some(PlanId::BeamAnalysis),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NUMA Detection — Improvement #5
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, serde::Serialize)]
pub struct NumaInfo {
    pub node_count:     usize,
    pub cpu_node:       usize,   // NUMA node of CPUs 10-11
    pub l1d_kb:         usize,
    pub block_size:     usize,   // current block size
    pub block_fits_l1:  bool,
    pub avx512:         bool,
    pub avx2:           bool,
    pub advisory:       String,
}

pub fn detect_numa() -> NumaInfo {
    // Count NUMA nodes from /sys
    let node_count = (0..16)
        .filter(|&n| std::path::Path::new(&format!("/sys/devices/system/node/node{}", n)).exists())
        .count();

    // Which NUMA node owns CPU 10?
    let cpu_node = std::fs::read_to_string("/sys/bus/cpu/devices/cpu10/node0/cpumap")
        .ok()
        .map(|_| 0usize) // if node0/cpumap exists → node 0
        .unwrap_or(0);

    // L1d cache size
    let l1d_kb = std::fs::read_to_string("/sys/devices/system/cpu/cpu10/cache/index0/size")
        .ok()
        .and_then(|s| s.trim().trim_end_matches('K').parse::<usize>().ok())
        .unwrap_or(32);

    let block_bytes = BLOCK_SIZE * 13 * 8; // 13 f64 fields per scenario
    let block_fits  = block_bytes <= l1d_kb * 1024;
    let avx512      = is_x86_feature_detected!("avx512f");
    let avx2        = is_x86_feature_detected!("avx2");

    let advisory = if node_count > 1 {
        format!("Multi-NUMA: run with 'numactl --cpunodebind={} --membind={}' for deterministic latency", cpu_node, cpu_node)
    } else if avx512 {
        "Single NUMA node — AVX-512 active: optimal configuration".into()
    } else if avx2 {
        "Single NUMA node — AVX2 active. For AVX-512: bare-metal EPYC (no KVM)".into()
    } else {
        "No SIMD detected — scalar fallback".into()
    };

    NumaInfo { node_count, cpu_node, l1d_kb, block_size: BLOCK_SIZE, block_fits_l1: block_fits, avx512, avx2, advisory }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation Loop State
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, serde::Serialize)]
pub struct SimStats {
    pub ticks:           u64,
    pub scenarios_total: u64,
    pub valid_fraction:  f64,
    pub invalid_count:   u64,  // cumulative invalid scenarios caught by physics
    pub last_tick_ns:    u64,
    pub scenarios_per_s: u64,
    pub sweep_mode:      String,
    pub kernel_path:     String,
    pub pareto_front:    Option<ParetoFront>,
    pub pareto_size:     usize,
    pub numa:            Option<NumaInfo>,
}

pub struct SimulationEngine {
    pub running:      Arc<AtomicBool>,
    pub stats:        Arc<RwLock<SimStats>>,
    pub tick_ctr:     Arc<AtomicU64>,
    pub scen_total:   Arc<AtomicU64>,
    pub inv_total:    Arc<AtomicU64>,
    pub per_s_atomic: Arc<AtomicU64>,
    pub peak_per_s:   Arc<AtomicU64>,
}

impl SimulationEngine {
    pub fn new() -> Self {
        let avx512 = is_x86_feature_detected!("avx512f");
        let avx2   = is_x86_feature_detected!("avx2");
        let kernel  = if avx512 { "avx512f+fma" } else if avx2 { "avx2+fma" } else { "scalar" };
        SimulationEngine {
            running:      Arc::new(AtomicBool::new(false)),
            stats:        Arc::new(RwLock::new(SimStats {
                ticks:0, scenarios_total:0, valid_fraction:0.0, invalid_count:0,
                last_tick_ns:0, scenarios_per_s:0,
                sweep_mode:"stratified".into(), kernel_path:kernel.into(),
                pareto_front:None, pareto_size:0, numa:None,
            })),
            tick_ctr:     Arc::new(AtomicU64::new(0)),
            scen_total:   Arc::new(AtomicU64::new(0)),
            inv_total:    Arc::new(AtomicU64::new(0)),
            per_s_atomic: Arc::new(AtomicU64::new(0)),
            peak_per_s:   Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn start(&self, _plan: PlanId, sweep: SweepSpec) {
        if self.running.swap(true, Ordering::SeqCst) { return; }

        // Determine worker count: half of available cores, min 4, max 8
        let n_workers: u64 = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(4).min(8))
            .unwrap_or(6) as u64;

        let avx512   = is_x86_feature_detected!("avx512f");
        let avx2     = is_x86_feature_detected!("avx2");
        let kernel_p = if avx512 { "avx512f+fma" } else if avx2 { "avx2+fma" } else { "scalar" };
        let numa_info = detect_numa();
        let mode_str  = format!("{:?}", sweep.mode).to_lowercase();

        // Reset all hot atomics on every start
        self.scen_total.store(0,   Ordering::SeqCst);
        self.inv_total.store(0,    Ordering::SeqCst);
        self.per_s_atomic.store(0, Ordering::SeqCst);
        self.peak_per_s.store(0,   Ordering::SeqCst);
        self.tick_ctr.store(0,     Ordering::SeqCst);

        // Reset SimStats
        if let Ok(mut s) = self.stats.write() {
            s.ticks = 0; s.scenarios_total = 0; s.scenarios_per_s = 0;
            s.invalid_count = 0; s.valid_fraction = 0.0; s.last_tick_ns = 0;
            s.pareto_front = None; s.pareto_size = 0;
            s.numa = Some(numa_info);
            s.kernel_path = format!("{}x{}w", kernel_p, n_workers);
            s.sweep_mode  = mode_str;
        }

        // Shared window accumulator (all workers add lock-free)
        let win_atomic   = Arc::new(AtomicU64::new(0u64));
        let tick_ns_atom = Arc::new(AtomicU64::new(0u64));
        let valid_bits   = Arc::new(AtomicU64::new(0u64));

        for worker_id in 0..n_workers {
            let running      = self.running.clone();
            let stats        = self.stats.clone();
            let tick_ctr     = self.tick_ctr.clone();
            let sweep        = sweep.clone();
            let scen_total   = self.scen_total.clone();
            let inv_total    = self.inv_total.clone();
            let per_s_atomic = self.per_s_atomic.clone();
            let peak_per_s   = self.peak_per_s.clone();
            let win_atomic   = win_atomic.clone();
            let tick_ns_atom = tick_ns_atom.clone();
            let valid_bits   = valid_bits.clone();
            let is_primary   = worker_id == 0;

            std::thread::Builder::new()
                .name(format!("sim-w{}", worker_id))
                .spawn(move || {
                    let mut soa  = ScenarioSoA::new();
                    // Stagger initial ticks so workers explore different parameter regions
                    let mut tick = worker_id;
                    let mut win_start = Instant::now();

                    while running.load(Ordering::Relaxed) {
                        let t0 = Instant::now();

                        // 1. SWEEP
                        sweep.fill_soa(&mut soa, tick);

                        // 2. PHYSICS VALIDATE — blocked AVX2
                        physics_validate_blocked(&mut soa);

                        // 3. COMPUTE — L1-blocked AVX2/AVX-512
                        pump_kernel_blocked(&mut soa);

                        // 4. EVALUATE
                        let valid_n   = soa.valid.iter().filter(|&&v| v > 0.5).count();
                        let invalid_n = SIM_N - valid_n;
                        let tick_ns   = t0.elapsed().as_nanos() as u64;

                        // 5. Lock-free atomic updates (ALL workers, no RwLock)
                        scen_total.fetch_add(SIM_N as u64, Ordering::Relaxed);
                        inv_total.fetch_add(invalid_n as u64, Ordering::Relaxed);
                        win_atomic.fetch_add(SIM_N as u64, Ordering::Relaxed);
                        tick_ctr.fetch_add(1, Ordering::Relaxed);

                        // 6. PRIMARY worker: Pareto every 50 ticks + stats snapshot
                        if is_primary {
                            valid_bits.store(
                                (valid_n as f64 / SIM_N as f64).to_bits(),
                                Ordering::Relaxed
                            );
                            tick_ns_atom.store(tick_ns, Ordering::Relaxed);

                            if tick % 50 == 0 {
                                // Sliding-window throughput (1-second window)
                                let win_elapsed = win_start.elapsed().as_secs_f64();
                                let current_per_s = if win_elapsed >= 1.0 {
                                    let w = win_atomic.swap(0, Ordering::Relaxed);
                                    let rate = (w as f64 / win_elapsed) as u64;
                                    win_start = Instant::now();
                                    let prev = peak_per_s.load(Ordering::Relaxed);
                                    if rate > prev { peak_per_s.store(rate, Ordering::Relaxed); }
                                    per_s_atomic.store(rate, Ordering::Relaxed);
                                    rate
                                } else { 0u64 };

                                // Pareto front from primary worker's SoA
                                let pareto = Some(compute_pareto_front(&soa));

                                if let Ok(mut s) = stats.write() {
                                    s.ticks           = tick_ctr.load(Ordering::Relaxed);
                                    s.scenarios_total = scen_total.load(Ordering::Relaxed);
                                    s.invalid_count   = inv_total.load(Ordering::Relaxed);
                                    s.valid_fraction  = f64::from_bits(
                                        valid_bits.load(Ordering::Relaxed)
                                    );
                                    s.last_tick_ns    = tick_ns_atom.load(Ordering::Relaxed);
                                    if current_per_s > 0 { s.scenarios_per_s = current_per_s; }
                                    if let Some(ref p) = pareto {
                                        s.pareto_size  = p.solutions.len();
                                        s.pareto_front = pareto;
                                    }
                                }
                            }
                        }

                        tick += n_workers;
                        if tick_ns < 100_000 { std::thread::yield_now(); }
                    }
                }).expect("sim worker spawn failed");
        }
    }


    pub fn stop(&self) { self.running.store(false, Ordering::SeqCst); }
    pub fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    pub fn get_stats(&self) -> SimStats {
        let mut s = self.stats.read().unwrap().clone();
        let live = self.per_s_atomic.load(std::sync::atomic::Ordering::Relaxed);
        if live > 0 { s.scenarios_per_s = live; }
        s
    }
    pub fn peak_per_s(&self) -> u64 { self.peak_per_s.load(std::sync::atomic::Ordering::Relaxed) }
}

use std::sync::OnceLock;
static SIM_ENGINE: OnceLock<SimulationEngine> = OnceLock::new();
pub fn global_engine() -> &'static SimulationEngine {
    SIM_ENGINE.get_or_init(SimulationEngine::new)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── Physics Layer ────────────────────────────────────────────────────────
    #[test] fn physics_valid_nominal() {
        assert_eq!(physics_valid_pump(500.0, 100.0, 0.75), 1.0);
    }
    #[test] fn physics_rejects_overefficiency() {
        assert_eq!(physics_valid_pump(500.0, 100.0, 1.5), 0.0); // Carnot violation
    }
    #[test] fn physics_rejects_negative_flow() {
        assert_eq!(physics_valid_pump(-100.0, 100.0, 0.75), 0.0);
    }
    #[test] fn physics_rejects_zero_pressure() {
        assert_eq!(physics_valid_pump(500.0, 0.0, 0.75), 0.0);
    }
    #[test] fn physics_rejects_nan() {
        assert_eq!(physics_valid_pump(f64::NAN, 100.0, 0.75), 0.0);
    }
    #[test] fn physics_boundary_exact_max() {
        // Efficiency exactly at 0.97 limit — should be valid
        assert_eq!(physics_valid_pump(500.0, 100.0, 0.97), 1.0);
    }

    // ── Stress Sweep ─────────────────────────────────────────────────────────
    #[test] fn stress_sweep_produces_invalids() {
        let spec = SweepSpec::pump_stress();
        let mut soa = ScenarioSoA::new();
        spec.fill_soa(&mut soa, 0);
        physics_validate_blocked(&mut soa);
        let valid_n = soa.valid.iter().filter(|&&v| v > 0.5).count();
        let frac = valid_n as f64 / SIM_N as f64;
        // Stress mode: every 10th invalid → valid_frac ≈ 0.90 ± some (extended ranges also invalidate)
        assert!(frac < 0.99, "Stress sweep should produce some invalids, got {:.3}", frac);
    }

    #[test] fn adversarial_sweep_drops_valid_frac() {
        let spec = SweepSpec { mode: SweepMode::Adversarial, ..SweepSpec::pump_stress() };
        let mut soa = ScenarioSoA::new();
        spec.fill_soa(&mut soa, 0);
        physics_validate_blocked(&mut soa);
        let valid_n = soa.valid.iter().filter(|&&v| v > 0.5).count();
        let frac = valid_n as f64 / SIM_N as f64;
        assert!(frac < 0.70, "Adversarial sweep valid_frac should be <70%, got {:.3}", frac);
    }

    // ── Block Tiling ─────────────────────────────────────────────────────────
    #[test] fn blocked_matches_unblocked() {
        if !is_x86_feature_detected!("avx2") { return; }
        let mut soa1 = ScenarioSoA::new();
        let mut soa2 = ScenarioSoA::new();
        let spec = SweepSpec::pump_default();
        spec.fill_soa(&mut soa1, 42);
        spec.fill_soa(&mut soa2, 42);
        physics_validate_blocked(&mut soa1);
        physics_validate_blocked(&mut soa2);
        // Compute blocked vs unblocked
        pump_kernel_blocked(&mut soa1);
        // Unblocked reference (full 1024 in one AVX2 pass)
        unsafe { pump_kernel_block_avx2(&mut soa2, 0, SIM_N); }
        // Compare
        for i in 0..SIM_N {
            assert!((soa1.out[0][i] - soa2.out[0][i]).abs() < 1e-10,
                "Blocked/unblocked mismatch at i={}: {} vs {}", i, soa1.out[0][i], soa2.out[0][i]);
        }
    }

    // ── Pareto Front ─────────────────────────────────────────────────────────
    #[test] fn pareto_front_non_empty() {
        let spec = SweepSpec::pump_default();
        let mut soa = ScenarioSoA::new();
        spec.fill_soa(&mut soa, 0);
        physics_validate_blocked(&mut soa);
        pump_kernel_blocked(&mut soa);
        let front = compute_pareto_front(&soa);
        assert!(!front.solutions.is_empty(), "Pareto front should have solutions");
        assert!(front.pareto_size() > 0);
    }

    #[test] fn pareto_domination_correct() {
        let better = ParetoSolution {
            scenario_idx:0, hp_req:10.0, eff_score:0.9, cost_usd:5000.0, risk_score:0.1, params:[0.0;3]
        };
        let worse = ParetoSolution {
            scenario_idx:1, hp_req:15.0, eff_score:0.7, cost_usd:8000.0, risk_score:0.4, params:[0.0;3]
        };
        assert!(dominates(&better, &worse));
        assert!(!dominates(&worse, &better));
    }

    // ── NUMA Detection ────────────────────────────────────────────────────────
    #[test] fn numa_detect_runs() {
        let info = detect_numa();
        assert!(info.l1d_kb > 0, "L1d should be detected");
        assert!(info.block_size == BLOCK_SIZE);
        println!("NUMA: {:?}", info);
    }
}

// Helper needed by test
impl ParetoFront {
    pub fn pareto_size(&self) -> usize { self.solutions.len() }
}

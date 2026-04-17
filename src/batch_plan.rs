// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v1.9 — Plan-Level SIMD Batch Kernels
//
// Unlike batch_oracle.rs (which vectorizes individual oracle functions),
// this module vectorizes ACROSS complete PLAN SCENARIOS:
//   f64x4 path (AVX2):    4 full plans in parallel per SIMD instruction
//   f64x8 path (AVX-512): 8 full plans in parallel (bare-metal EPYC only)
//
// Key distinction from oracle-level SIMD:
//   Oracle SIMD:  K*sqrt(P) for 8 K/P pairs → single-oracle vectorization
//   Plan SIMD:    pump_sizing(Q1,P1,eff1), pump_sizing(Q2,P2,eff2),
//                 pump_sizing(Q3,P3,eff3), pump_sizing(Q4,P4,eff4)
//                 all 4 full plan results in ~12 ns
//
// Plan kernels implemented:
//   plan_pump_sizing(Q_gpm, P_psi, eff)             — NFPA 20
//   plan_voltage_drop(I, L_m, A_mm2)                — IEC 60364
//   plan_planilla(sueldo, meses, dias_vacac)         — DL 728 Peru
//   plan_beam_analysis(P_kn, L_m, E_gpa, b_cm, h_cm)— AISC 360
//
// AVX-512 bare-metal paths:
//   8 scenarios simultaneously using __m512d (512-bit double vectors)
//   Activated by is_x86_feature_detected!("avx512f")
//   Contabo KVM: hypervisor masks AVX-512 -> uses avx2 path
//   Bare-metal EPYC Zen3+: full AVX-512F -> 8x throughput
//
// Benchmark projection (AMD EPYC, FMA3+AVX2, 3.0 GHz):
//   plan_pump_sizing:   scalar 23 ns -> AVX2x4 ~7 ns/call (3.3x)
//   plan_voltage_drop:  scalar 10 ns -> AVX2x4 ~3.5 ns/call (2.9x)
//   plan_planilla:      scalar 10 ns -> AVX2x4 ~2.5 ns/call (4x)
//   AVX-512x8: 2x over AVX2x4 on bare-metal EPYC
//
// Multiverso 400-scenario sweep with rayon+AVX2:
//   Current (scalar sequential): ~9 us
//   AVX2x4 + rayon 12 cores:     ~300 ns  (~30x)
//   AVX-512x8 + rayon (bare):    ~150 ns  (~60x)
// ═══════════════════════════════════════════════════════════════════════

use std::time::Instant;
use rayon::prelude::*;

// ── Result types ──────────────────────────────────────────────────────

pub struct SimdBatchResult {
    pub values:      Vec<f64>,
    pub n_scenarios: usize,
    pub n_outputs:   usize,
    pub time_ns:     u64,
    pub path:        SimdPath,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdPath {
    Scalar,
    Avx2x4,
    Avx512x8,
    RayonAvx2,
}

impl SimdPath {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar    => "scalar",
            Self::Avx2x4    => "avx2x4",
            Self::Avx512x8  => "avx512x8",
            Self::RayonAvx2 => "rayon+avx2",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// PLAN: pump_sizing(Q_gpm, P_psi, eff)
// Outputs: [HP_req, HP_max, Q_lps, H_m, HP_shutoff]  5 values
// ══════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn pump_sizing_scalar(q_gpm: f64, p_psi: f64, eff: f64) -> [f64; 5] {
    let q_lps      = q_gpm * 0.063_09;
    let h_m        = p_psi * 0.703_07;
    let den        = eff * 76.04;
    let hp_req     = if den > 0.0 { q_lps * h_m / den } else { 0.0 };
    let hp_max     = hp_req * 1.40;
    let hp_shutoff = hp_req * 1.20;
    [hp_req, hp_max, q_lps, h_m, hp_shutoff]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pump_sizing_avx2x4(q_gpm: [f64;4], p_psi: [f64;4], eff: [f64;4]) -> [f64;20] {
    use std::arch::x86_64::*;
    let vq    = _mm256_loadu_pd(q_gpm.as_ptr());
    let vp    = _mm256_loadu_pd(p_psi.as_ptr());
    let ve    = _mm256_loadu_pd(eff.as_ptr());
    let vzero = _mm256_setzero_pd();
    let c_lps = _mm256_set1_pd(0.063_09);
    let c_hm  = _mm256_set1_pd(0.703_07);
    let c76   = _mm256_set1_pd(76.04);
    let c140  = _mm256_set1_pd(1.40);
    let c120  = _mm256_set1_pd(1.20);
    let q_lps  = _mm256_mul_pd(vq, c_lps);
    let h_m    = _mm256_mul_pd(vp, c_hm);
    let num    = _mm256_mul_pd(q_lps, h_m);
    let den    = _mm256_mul_pd(ve, c76);
    let mask   = _mm256_cmp_pd(den, vzero, 4);
    let hp_req = _mm256_blendv_pd(vzero, _mm256_div_pd(num, den), mask);
    let hp_max     = _mm256_mul_pd(hp_req, c140);
    let hp_shutoff = _mm256_mul_pd(hp_req, c120);
    let mut r_req=[0.0f64;4]; let mut r_max=[0.0f64;4]; let mut r_qlps=[0.0f64;4];
    let mut r_hm=[0.0f64;4];  let mut r_shut=[0.0f64;4];
    _mm256_storeu_pd(r_req.as_mut_ptr(),  hp_req);
    _mm256_storeu_pd(r_max.as_mut_ptr(),  hp_max);
    _mm256_storeu_pd(r_qlps.as_mut_ptr(), q_lps);
    _mm256_storeu_pd(r_hm.as_mut_ptr(),   h_m);
    _mm256_storeu_pd(r_shut.as_mut_ptr(), hp_shutoff);
    let mut out = [0.0f64;20];
    for s in 0..4 {
        out[s*5+0]=r_req[s]; out[s*5+1]=r_max[s]; out[s*5+2]=r_qlps[s];
        out[s*5+3]=r_hm[s];  out[s*5+4]=r_shut[s];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn pump_sizing_avx512x8(q_gpm:[f64;8],p_psi:[f64;8],eff:[f64;8]) -> [f64;40] {
    use std::arch::x86_64::*;
    let vq  = _mm512_loadu_pd(q_gpm.as_ptr());
    let vp  = _mm512_loadu_pd(p_psi.as_ptr());
    let ve  = _mm512_loadu_pd(eff.as_ptr());
    let vzero = _mm512_setzero_pd();
    let c_lps = _mm512_set1_pd(0.063_09);
    let c_hm  = _mm512_set1_pd(0.703_07);
    let c76   = _mm512_set1_pd(76.04);
    let c140  = _mm512_set1_pd(1.40);
    let c120  = _mm512_set1_pd(1.20);
    let q_lps  = _mm512_mul_pd(vq, c_lps);
    let h_m    = _mm512_mul_pd(vp, c_hm);
    let num    = _mm512_fmadd_pd(q_lps, h_m, vzero);
    let den    = _mm512_mul_pd(ve, c76);
    let mask   = _mm512_cmp_pd_mask(den, vzero, _CMP_NEQ_OQ);
    let hp_req = _mm512_mask_div_pd(vzero, mask, num, den);
    let hp_max     = _mm512_mul_pd(hp_req, c140);
    let hp_shutoff = _mm512_mul_pd(hp_req, c120);
    let mut r_req=[0.0f64;8]; let mut r_max=[0.0f64;8]; let mut r_qlps=[0.0f64;8];
    let mut r_hm=[0.0f64;8];  let mut r_shut=[0.0f64;8];
    _mm512_storeu_pd(r_req.as_mut_ptr(),  hp_req);
    _mm512_storeu_pd(r_max.as_mut_ptr(),  hp_max);
    _mm512_storeu_pd(r_qlps.as_mut_ptr(), q_lps);
    _mm512_storeu_pd(r_hm.as_mut_ptr(),   h_m);
    _mm512_storeu_pd(r_shut.as_mut_ptr(), hp_shutoff);
    let mut out = [0.0f64;40];
    for s in 0..8 {
        out[s*5+0]=r_req[s]; out[s*5+1]=r_max[s]; out[s*5+2]=r_qlps[s];
        out[s*5+3]=r_hm[s];  out[s*5+4]=r_shut[s];
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
// PLAN: voltage_drop(I, L_m, A_mm2)
// rho = 0.0172 Ohm.mm2/m (copper, constant folded)
// Outputs: [V_drop, R_line, P_loss, drop_pct]  4 values
// ══════════════════════════════════════════════════════════════════════

const RHO_CU: f64 = 0.0172;

#[inline(always)]
pub fn voltage_drop_scalar(i: f64, l_m: f64, a_mm2: f64) -> [f64; 4] {
    let r_line   = if a_mm2 > 0.0 { RHO_CU * l_m / a_mm2 } else { 0.0 };
    let v_drop   = 2.0 * i * r_line;
    let p_loss   = i * i * r_line;
    let drop_pct = v_drop / 220.0 * 100.0;
    [v_drop, r_line, p_loss, drop_pct]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn voltage_drop_avx2x4(i:[f64;4], l_m:[f64;4], a_mm2:[f64;4]) -> [f64;16] {
    use std::arch::x86_64::*;
    let vi    = _mm256_loadu_pd(i.as_ptr());
    let vl    = _mm256_loadu_pd(l_m.as_ptr());
    let va    = _mm256_loadu_pd(a_mm2.as_ptr());
    let vzero = _mm256_setzero_pd();
    let vrho  = _mm256_set1_pd(RHO_CU);
    let v2    = _mm256_set1_pd(2.0);
    let v100r = _mm256_set1_pd(100.0 / 220.0);
    let rho_l   = _mm256_mul_pd(vrho, vl);
    let mask_a  = _mm256_cmp_pd(va, vzero, 4);
    let r_line  = _mm256_blendv_pd(vzero, _mm256_div_pd(rho_l, va), mask_a);
    let ir      = _mm256_mul_pd(vi, r_line);
    let v_drop  = _mm256_mul_pd(v2, ir);
    let p_loss  = _mm256_mul_pd(vi, ir);
    let dpct    = _mm256_mul_pd(v_drop, v100r);
    let mut r0=[0.0f64;4]; let mut r1=[0.0f64;4];
    let mut r2=[0.0f64;4]; let mut r3=[0.0f64;4];
    _mm256_storeu_pd(r0.as_mut_ptr(), v_drop);
    _mm256_storeu_pd(r1.as_mut_ptr(), r_line);
    _mm256_storeu_pd(r2.as_mut_ptr(), p_loss);
    _mm256_storeu_pd(r3.as_mut_ptr(), dpct);
    let mut out=[0.0f64;16];
    for s in 0..4 { out[s*4+0]=r0[s]; out[s*4+1]=r1[s]; out[s*4+2]=r2[s]; out[s*4+3]=r3[s]; }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn voltage_drop_avx512x8(i:[f64;8],l_m:[f64;8],a_mm2:[f64;8]) -> [f64;32] {
    use std::arch::x86_64::*;
    let vi    = _mm512_loadu_pd(i.as_ptr());
    let vl    = _mm512_loadu_pd(l_m.as_ptr());
    let va    = _mm512_loadu_pd(a_mm2.as_ptr());
    let vzero = _mm512_setzero_pd();
    let vrho  = _mm512_set1_pd(RHO_CU);
    let v2    = _mm512_set1_pd(2.0);
    let v100r = _mm512_set1_pd(100.0 / 220.0);
    let rho_l  = _mm512_mul_pd(vrho, vl);
    let mask_a = _mm512_cmp_pd_mask(va, vzero, _CMP_NEQ_OQ);
    let r_line = _mm512_mask_div_pd(vzero, mask_a, rho_l, va);
    let ir     = _mm512_mul_pd(vi, r_line);
    let v_drop = _mm512_mul_pd(v2, ir);
    let p_loss = _mm512_mul_pd(vi, ir);
    let dpct   = _mm512_mul_pd(v_drop, v100r);
    let mut r0=[0.0f64;8]; let mut r1=[0.0f64;8];
    let mut r2=[0.0f64;8]; let mut r3=[0.0f64;8];
    _mm512_storeu_pd(r0.as_mut_ptr(), v_drop);
    _mm512_storeu_pd(r1.as_mut_ptr(), r_line);
    _mm512_storeu_pd(r2.as_mut_ptr(), p_loss);
    _mm512_storeu_pd(r3.as_mut_ptr(), dpct);
    let mut out=[0.0f64;32];
    for s in 0..8 { out[s*4+0]=r0[s]; out[s*4+1]=r1[s]; out[s*4+2]=r2[s]; out[s*4+3]=r3[s]; }
    out
}

// ══════════════════════════════════════════════════════════════════════
// PLAN: planilla(sueldo, meses, dias_vacac)
// All linear arithmetic — optimal for SIMD (no divisions)
// Outputs: [neto, cts_total, costo_total, vacaciones, essalud]  5 values
// ══════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn planilla_scalar(sueldo: f64, meses: f64, dias_vacac: f64) -> [f64; 5] {
    let neto         = sueldo * 0.87;
    let essalud      = sueldo * 0.09;
    let cts_mensual  = sueldo * 0.0833;
    let cts_total    = cts_mensual * meses;
    let grat_mensual = sueldo * 0.1667;
    let vacaciones   = sueldo * dias_vacac * (1.0 / 30.0);
    let costo_total  = sueldo + essalud + cts_mensual + grat_mensual;
    [neto, cts_total, costo_total, vacaciones, essalud]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn planilla_avx2x4(sueldo:[f64;4],meses:[f64;4],dias:[f64;4]) -> [f64;20] {
    use std::arch::x86_64::*;
    let vs    = _mm256_loadu_pd(sueldo.as_ptr());
    let vm    = _mm256_loadu_pd(meses.as_ptr());
    let vd    = _mm256_loadu_pd(dias.as_ptr());
    let c87   = _mm256_set1_pd(0.87);
    let c09   = _mm256_set1_pd(0.09);
    let c0833 = _mm256_set1_pd(0.0833);
    let c1667 = _mm256_set1_pd(0.1667);
    let cvac  = _mm256_set1_pd(1.0 / 30.0);
    let neto         = _mm256_mul_pd(vs, c87);
    let essalud      = _mm256_mul_pd(vs, c09);
    let cts_mensual  = _mm256_mul_pd(vs, c0833);
    let cts_total    = _mm256_mul_pd(cts_mensual, vm);
    let grat_mensual = _mm256_mul_pd(vs, c1667);
    // vacaciones = sueldo * dias * (1/30) — FMA: fma(sueldo, dias, 0) * cvac
    let sd      = _mm256_mul_pd(vs, vd);
    let vacaciones = _mm256_mul_pd(sd, cvac);
    let t1   = _mm256_add_pd(vs, essalud);
    let t2   = _mm256_add_pd(t1, cts_mensual);
    let costo = _mm256_add_pd(t2, grat_mensual);
    let mut r0=[0.0f64;4]; let mut r1=[0.0f64;4]; let mut r2=[0.0f64;4];
    let mut r3=[0.0f64;4]; let mut r4=[0.0f64;4];
    _mm256_storeu_pd(r0.as_mut_ptr(), neto);
    _mm256_storeu_pd(r1.as_mut_ptr(), cts_total);
    _mm256_storeu_pd(r2.as_mut_ptr(), costo);
    _mm256_storeu_pd(r3.as_mut_ptr(), vacaciones);
    _mm256_storeu_pd(r4.as_mut_ptr(), essalud);
    let mut out=[0.0f64;20];
    for s in 0..4 {
        out[s*5+0]=r0[s]; out[s*5+1]=r1[s]; out[s*5+2]=r2[s];
        out[s*5+3]=r3[s]; out[s*5+4]=r4[s];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn planilla_avx512x8(sueldo:[f64;8],meses:[f64;8],dias:[f64;8]) -> [f64;40] {
    use std::arch::x86_64::*;
    let vs    = _mm512_loadu_pd(sueldo.as_ptr());
    let vm    = _mm512_loadu_pd(meses.as_ptr());
    let vd    = _mm512_loadu_pd(dias.as_ptr());
    let vzero = _mm512_setzero_pd();
    let c87   = _mm512_set1_pd(0.87);
    let c09   = _mm512_set1_pd(0.09);
    let c0833 = _mm512_set1_pd(0.0833);
    let c1667 = _mm512_set1_pd(0.1667);
    let cvac  = _mm512_set1_pd(1.0 / 30.0);
    let neto         = _mm512_mul_pd(vs, c87);
    let essalud      = _mm512_mul_pd(vs, c09);
    let cts_mensual  = _mm512_mul_pd(vs, c0833);
    let cts_total    = _mm512_mul_pd(cts_mensual, vm);
    let grat_mensual = _mm512_mul_pd(vs, c1667);
    let sd         = _mm512_fmadd_pd(vs, vd, vzero);
    let vacaciones = _mm512_mul_pd(sd, cvac);
    let t1  = _mm512_add_pd(vs, essalud);
    let t2  = _mm512_add_pd(t1, cts_mensual);
    let costo = _mm512_add_pd(t2, grat_mensual);
    let mut r0=[0.0f64;8]; let mut r1=[0.0f64;8]; let mut r2=[0.0f64;8];
    let mut r3=[0.0f64;8]; let mut r4=[0.0f64;8];
    _mm512_storeu_pd(r0.as_mut_ptr(), neto);
    _mm512_storeu_pd(r1.as_mut_ptr(), cts_total);
    _mm512_storeu_pd(r2.as_mut_ptr(), costo);
    _mm512_storeu_pd(r3.as_mut_ptr(), vacaciones);
    _mm512_storeu_pd(r4.as_mut_ptr(), essalud);
    let mut out=[0.0f64;40];
    for s in 0..8 {
        out[s*5+0]=r0[s]; out[s*5+1]=r1[s]; out[s*5+2]=r2[s];
        out[s*5+3]=r3[s]; out[s*5+4]=r4[s];
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
// PLAN: beam_analysis(P_kn, L_m, E_gpa, b_cm, h_cm)
// Outputs: [Mmax_kNm, deflection_mm, I_cm4]  3 values
// ══════════════════════════════════════════════════════════════════════

#[inline(always)]
pub fn beam_analysis_scalar(p_kn:f64, l_m:f64, e_gpa:f64, b_cm:f64, h_cm:f64) -> [f64;3] {
    let i_cm4 = b_cm * h_cm * h_cm * h_cm / 12.0;
    let mmax  = p_kn * l_m / 4.0;
    let defl  = if e_gpa > 0.0 && i_cm4 > 0.0 {
        (p_kn * 1000.0 * l_m * l_m * l_m) / (48.0 * e_gpa * 1e9 * i_cm4 * 1e-8) * 1000.0
    } else { 0.0 };
    [mmax, defl, i_cm4]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn beam_analysis_avx2x4(
    p_kn:[f64;4],l_m:[f64;4],e_gpa:[f64;4],b_cm:[f64;4],h_cm:[f64;4],
) -> [f64;12] {
    use std::arch::x86_64::*;
    let vp    = _mm256_loadu_pd(p_kn.as_ptr());
    let vl    = _mm256_loadu_pd(l_m.as_ptr());
    let ve    = _mm256_loadu_pd(e_gpa.as_ptr());
    let vb    = _mm256_loadu_pd(b_cm.as_ptr());
    let vh    = _mm256_loadu_pd(h_cm.as_ptr());
    let vzero = _mm256_setzero_pd();
    let c12   = _mm256_set1_pd(12.0);
    let c4r   = _mm256_set1_pd(0.25);
    // c_defl = 1000 * 1000 / 48 / (1e9 * 1e-8) = 1e6 / (48 * 10) = 1e6/480
    let c_defl = _mm256_set1_pd(1_000.0 * 1_000.0 / 48.0 / (1e9 * 1e-8));
    let h2    = _mm256_mul_pd(vh, vh);
    let h3    = _mm256_mul_pd(h2, vh);
    let bh3   = _mm256_mul_pd(vb, h3);
    let i_cm4 = _mm256_div_pd(bh3, c12);
    let mmax  = _mm256_mul_pd(_mm256_mul_pd(vp, vl), c4r);
    let l2    = _mm256_mul_pd(vl, vl);
    let l3    = _mm256_mul_pd(l2, vl);
    let pl3   = _mm256_mul_pd(vp, l3);
    let pl3c  = _mm256_mul_pd(pl3, c_defl);
    let ei    = _mm256_mul_pd(ve, i_cm4);
    let mask  = _mm256_cmp_pd(ei, vzero, 4);
    let defl  = _mm256_blendv_pd(vzero, _mm256_div_pd(pl3c, ei), mask);
    let mut r0=[0.0f64;4]; let mut r1=[0.0f64;4]; let mut r2=[0.0f64;4];
    _mm256_storeu_pd(r0.as_mut_ptr(), mmax);
    _mm256_storeu_pd(r1.as_mut_ptr(), defl);
    _mm256_storeu_pd(r2.as_mut_ptr(), i_cm4);
    let mut out=[0.0f64;12];
    for s in 0..4 { out[s*3+0]=r0[s]; out[s*3+1]=r1[s]; out[s*3+2]=r2[s]; }
    out
}

// ══════════════════════════════════════════════════════════════════════
// Universal dispatcher + Rayon parallel batch
// ══════════════════════════════════════════════════════════════════════

pub fn execute_simd_batch(
    plan_name:  &str,
    param_sets: &[Vec<f64>],
    n_outputs:  usize,
) -> Option<SimdBatchResult> {
    let n = param_sets.len();
    if n == 0 { return None; }
    let t0 = Instant::now();
    let avx512 = is_x86_feature_detected!("avx512f");
    let avx2   = is_x86_feature_detected!("avx2");

    let (values, path) = match plan_name {
        "plan_pump_sizing" | "pump_sizing" if param_sets[0].len() >= 3 => {
            let mut out = Vec::with_capacity(n * 5);
            let mut i = 0;
            #[cfg(target_arch = "x86_64")]
            if avx512 && n >= 8 {
                while i + 8 <= n {
                    let q:[f64;8]=std::array::from_fn(|j| param_sets[i+j][0]);
                    let p:[f64;8]=std::array::from_fn(|j| param_sets[i+j][1]);
                    let e:[f64;8]=std::array::from_fn(|j| param_sets[i+j][2]);
                    out.extend_from_slice(&unsafe{pump_sizing_avx512x8(q,p,e)});
                    i += 8;
                }
            }
            #[cfg(target_arch = "x86_64")]
            if avx2 { while i+4<=n { let q:[f64;4]=std::array::from_fn(|j| param_sets[i+j][0]); let p:[f64;4]=std::array::from_fn(|j| param_sets[i+j][1]); let e:[f64;4]=std::array::from_fn(|j| param_sets[i+j][2]); out.extend_from_slice(&unsafe{pump_sizing_avx2x4(q,p,e)}); i+=4; } }
            while i<n { out.extend_from_slice(&pump_sizing_scalar(param_sets[i][0],param_sets[i][1],param_sets[i][2])); i+=1; }
            let path = if avx512&&n>=8{SimdPath::Avx512x8} else if avx2&&n>=4{SimdPath::Avx2x4} else{SimdPath::Scalar};
            (out, path)
        }
        "plan_voltage_drop" | "voltage_drop" if param_sets[0].len() >= 3 => {
            let mut out = Vec::with_capacity(n * 4);
            let mut i = 0;
            #[cfg(target_arch = "x86_64")]
            if avx512 && n >= 8 { while i+8<=n { let iv:[f64;8]=std::array::from_fn(|j| param_sets[i+j][0]); let lv:[f64;8]=std::array::from_fn(|j| param_sets[i+j][1]); let av:[f64;8]=std::array::from_fn(|j| param_sets[i+j][2]); out.extend_from_slice(&unsafe{voltage_drop_avx512x8(iv,lv,av)}); i+=8; } }
            #[cfg(target_arch = "x86_64")]
            if avx2 { while i+4<=n { let iv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][0]); let lv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][1]); let av:[f64;4]=std::array::from_fn(|j| param_sets[i+j][2]); out.extend_from_slice(&unsafe{voltage_drop_avx2x4(iv,lv,av)}); i+=4; } }
            while i<n { out.extend_from_slice(&voltage_drop_scalar(param_sets[i][0],param_sets[i][1],param_sets[i][2])); i+=1; }
            let path = if avx512&&n>=8{SimdPath::Avx512x8} else if avx2&&n>=4{SimdPath::Avx2x4} else{SimdPath::Scalar};
            (out, path)
        }
        "plan_planilla" | "planilla" | "plan_planilla_dl728" if param_sets[0].len() >= 3 => {
            let mut out = Vec::with_capacity(n * 5);
            let mut i = 0;
            #[cfg(target_arch = "x86_64")]
            if avx512 && n >= 8 { while i+8<=n { let sv:[f64;8]=std::array::from_fn(|j| param_sets[i+j][0]); let mv:[f64;8]=std::array::from_fn(|j| param_sets[i+j][1]); let dv:[f64;8]=std::array::from_fn(|j| param_sets[i+j][2]); out.extend_from_slice(&unsafe{planilla_avx512x8(sv,mv,dv)}); i+=8; } }
            #[cfg(target_arch = "x86_64")]
            if avx2 { while i+4<=n { let sv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][0]); let mv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][1]); let dv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][2]); out.extend_from_slice(&unsafe{planilla_avx2x4(sv,mv,dv)}); i+=4; } }
            while i<n { out.extend_from_slice(&planilla_scalar(param_sets[i][0],param_sets[i][1],param_sets[i][2])); i+=1; }
            let path = if avx512&&n>=8{SimdPath::Avx512x8} else if avx2&&n>=4{SimdPath::Avx2x4} else{SimdPath::Scalar};
            (out, path)
        }
        "plan_beam_analysis" | "beam_analysis" if param_sets[0].len() >= 5 => {
            let mut out = Vec::with_capacity(n * 3);
            let mut i = 0;
            #[cfg(target_arch = "x86_64")]
            if avx2 { while i+4<=n { let pv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][0]); let lv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][1]); let ev:[f64;4]=std::array::from_fn(|j| param_sets[i+j][2]); let bv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][3]); let hv:[f64;4]=std::array::from_fn(|j| param_sets[i+j][4]); out.extend_from_slice(&unsafe{beam_analysis_avx2x4(pv,lv,ev,bv,hv)}); i+=4; } }
            while i<n { out.extend_from_slice(&beam_analysis_scalar(param_sets[i][0],param_sets[i][1],param_sets[i][2],param_sets[i][3],param_sets[i][4])); i+=1; }
            let path = if avx2&&n>=4{SimdPath::Avx2x4} else{SimdPath::Scalar};
            (out, path)
        }
        _ => return None,
    };

    Some(SimdBatchResult {
        n_scenarios: n, n_outputs,
        values, time_ns: t0.elapsed().as_nanos() as u64, path,
    })
}

/// Rayon parallel batch: splits across all EPYC cores, each thread runs SIMD kernels.
/// Kahn group concept: scenarios have zero inter-dependencies -> trivially parallel.
pub fn execute_parallel_batch(
    plan_name:  &str,
    param_sets: &[Vec<f64>],
    n_outputs:  usize,
) -> Option<SimdBatchResult> {
    let n = param_sets.len();
    if n < 8 { return None; }
    let t0 = Instant::now();
    let n_threads = rayon::current_num_threads();
    let chunk = ((n / n_threads) / 4 * 4).max(4);
    let plan  = plan_name.to_string();
    let results: Vec<Option<Vec<f64>>> = param_sets
        .chunks(chunk)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|c| execute_simd_batch(&plan, c, n_outputs).map(|r| r.values))
        .collect();
    let mut all = Vec::with_capacity(n * n_outputs);
    for r in results {
        all.extend(r?);
    }
    Some(SimdBatchResult {
        n_scenarios: n, n_outputs,
        values: all,
        time_ns: t0.elapsed().as_nanos() as u64,
        path: SimdPath::RayonAvx2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pump_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }
        let q=[500.0,300.0,750.0,1000.0f64]; let p=[100.0,80.0,120.0,150.0f64]; let e=[0.75,0.70,0.80,0.65f64];
        let simd = unsafe { pump_sizing_avx2x4(q,p,e) };
        for s in 0..4 { let sc=pump_sizing_scalar(q[s],p[s],e[s]); for f in 0..5 { assert!((simd[s*5+f]-sc[f]).abs()<1e-8,"s{s}f{f}"); } }
    }
    #[test]
    fn vdrop_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }
        let i=[25.0,10.0,50.0,100.0f64]; let l=[50.0,30.0,80.0,120.0f64]; let a=[10.0,6.0,25.0,50.0f64];
        let simd = unsafe { voltage_drop_avx2x4(i,l,a) };
        for s in 0..4 { let sc=voltage_drop_scalar(i[s],l[s],a[s]); for f in 0..4 { assert!((simd[s*4+f]-sc[f]).abs()<1e-8,"s{s}f{f}"); } }
    }
    #[test]
    fn planilla_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }
        let sv=[5000.0,3000.0,8000.0,1200.0f64]; let mv=[12.0,6.0,12.0,3.0f64]; let dv=[15.0,7.0,30.0,0.0f64];
        let simd = unsafe { planilla_avx2x4(sv,mv,dv) };
        for s in 0..4 { let sc=planilla_scalar(sv[s],mv[s],dv[s]); for f in 0..5 { assert!((simd[s*5+f]-sc[f]).abs()<1e-6,"s{s}f{f}"); } }
    }
}

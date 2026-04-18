// ═══════════════════════════════════════════════════════════════════════
// QOMN v2.0 — Physical Unit Algebra & NFPA Validation
// Percy Rojas M. · Qomni AI Lab · 2026
// ═══════════════════════════════════════════════════════════════════════

use crate::ast::UnitExpr;

// ── NFPA/IEC Physical Range Table ─────────────────────────────────────
/// Known physical parameter ranges for NFPA and IEC standards.
/// Used by the type checker to emit warnings when values exceed limits.
#[derive(Debug, Clone)]
pub struct UnitRange {
    pub unit:     &'static str,
    pub lo:       f64,
    pub hi:       f64,
    pub standard: &'static str,
    pub note:     &'static str,
}

pub static NFPA_RANGES: &[UnitRange] = &[
    UnitRange { unit: "gpm",         lo: 0.0,   hi: 500.0,   standard: "NFPA 13 §7.3",    note: "Sprinkler flow rate" },
    UnitRange { unit: "psi",         lo: 0.0,   hi: 175.0,   standard: "NFPA 13 §7.1",    note: "Sprinkler pressure" },
    UnitRange { unit: "ft",          lo: 0.0,   hi: 500.0,   standard: "NFPA 13 §7.2",    note: "Hydraulic head" },
    UnitRange { unit: "hp",          lo: 0.0,   hi: 1000.0,  standard: "NFPA 20 §4.3",    note: "Fire pump horsepower" },
    UnitRange { unit: "m2",          lo: 0.0,   hi: 50000.0, standard: "NFPA 13/72",      note: "Coverage area" },
    UnitRange { unit: "ft2",         lo: 0.0,   hi: 500000.0,standard: "NFPA 13 §8.4",    note: "Coverage area (imperial)" },
    UnitRange { unit: "A",           lo: 0.0,   hi: 10000.0, standard: "IEC 60038",        note: "Electric current" },
    UnitRange { unit: "V",           lo: 0.0,   hi: 1000.0,  standard: "IEC 60038",        note: "Voltage (LV)" },
    UnitRange { unit: "ohm",         lo: 0.0,   hi: 1e9,     standard: "IEC",              note: "Impedance" },
    UnitRange { unit: "mm2",         lo: 0.5,   hi: 1000.0,  standard: "IEC 60228",        note: "Cable cross section" },
    UnitRange { unit: "kW",          lo: 0.0,   hi: 10000.0, standard: "IEC",              note: "Active power" },
    UnitRange { unit: "kVA",         lo: 0.0,   hi: 10000.0, standard: "IEC",              note: "Apparent power" },
    // K-factor range per NFPA 13 Table 6.2.3.1
    // K expressed as gpm/psi^0.5
    UnitRange { unit: "gpm/psi^0.5", lo: 1.4,   hi: 22.4,   standard: "NFPA 13 Table 6.2.3.1", note: "Sprinkler K-factor" },
    // Efficiency fraction
    UnitRange { unit: "fraction",    lo: 0.0,   hi: 1.0,    standard: "general",           note: "Dimensionless fraction" },
];

/// Look up the NFPA range for a unit string.
pub fn nfpa_range(unit: &str) -> Option<&'static UnitRange> {
    NFPA_RANGES.iter().find(|r| r.unit == unit)
}

/// Validate a value against the NFPA range for its unit.
/// Returns `Err(warning_message)` if out of range.
pub fn validate_nfpa(unit: &str, value: f64, param_name: &str) -> Result<(), String> {
    if let Some(r) = nfpa_range(unit) {
        if value < r.lo || value > r.hi {
            return Err(format!(
                "NFPA range warning: {} = {:.3} {} is outside [{}, {}] {} ({})",
                param_name, value, unit, r.lo, r.hi, r.standard, r.note
            ));
        }
    }
    Ok(())
}

// ── Unit Compatibility Rules ───────────────────────────────────────────

/// Returns true if two unit expressions are dimensionally compatible
/// (same base or one is dimensionless).
pub fn units_compatible(a: &UnitExpr, b: &UnitExpr) -> bool {
    match (a, b) {
        (UnitExpr::Dimensionless, _) | (_, UnitExpr::Dimensionless) => true,
        (UnitExpr::Base(x), UnitExpr::Base(y)) => x == y,
        _ => unit_str(a) == unit_str(b),
    }
}

/// Normalize a UnitExpr to a canonical string for comparison.
pub fn unit_str(u: &UnitExpr) -> String {
    u.display()
}

/// Infer the result unit of a binary operation.
/// This implements basic unit algebra for +, -, *, /, ^
#[derive(Debug, Clone, PartialEq)]
pub enum BinOpUnit {
    Add, Sub, Mul, Div, Pow(f64),
}

pub fn infer_op_unit(op: BinOpUnit, lhs: &UnitExpr, rhs: &UnitExpr) -> Result<UnitExpr, String> {
    match op {
        // Addition/subtraction: units must match, result has same unit
        BinOpUnit::Add | BinOpUnit::Sub => {
            if units_compatible(lhs, rhs) {
                Ok(lhs.clone())
            } else {
                Err(format!(
                    "unit mismatch in +/-: {} vs {}",
                    unit_str(lhs), unit_str(rhs)
                ))
            }
        }
        // Multiplication: combine units
        BinOpUnit::Mul => {
            match (lhs, rhs) {
                (UnitExpr::Dimensionless, x) | (x, UnitExpr::Dimensionless) => Ok(x.clone()),
                (l, r) => Ok(UnitExpr::Mul(Box::new(l.clone()), Box::new(r.clone()))),
            }
        }
        // Division: rational unit algebra
        BinOpUnit::Div => {
            match (lhs, rhs) {
                (_, UnitExpr::Dimensionless) => Ok(lhs.clone()),
                (UnitExpr::Dimensionless, r) => {
                    Ok(UnitExpr::Pow(Box::new(r.clone()), -1.0))
                }
                // Special: gpm / psi^0.5 * psi^0.5 = gpm (K-factor * sqrt(P) cancellation)
                (l, r) if unit_str(l) == unit_str(r) => Ok(UnitExpr::Dimensionless),
                (l, r) => Ok(UnitExpr::Div(Box::new(l.clone()), Box::new(r.clone()))),
            }
        }
        // Power: multiply exponent (only valid if rhs is dimensionless)
        BinOpUnit::Pow(exp) => {
            match lhs {
                UnitExpr::Dimensionless => Ok(UnitExpr::Dimensionless),
                UnitExpr::Base(s) => {
                    if (exp - 1.0).abs() < 1e-9 {
                        Ok(UnitExpr::Base(s.clone()))
                    } else {
                        Ok(UnitExpr::Pow(Box::new(UnitExpr::Base(s.clone())), exp))
                    }
                }
                other => Ok(UnitExpr::Pow(Box::new(other.clone()), exp)),
            }
        }
    }
}

// ── Unit Inference for NFPA oracles ───────────────────────────────────

/// Known NFPA oracle return units — used to auto-annotate oracle calls.
pub struct OracleUnitSignature {
    pub name:        &'static str,
    pub param_units: &'static [&'static str],
    pub return_unit: &'static str,
    pub standard:    &'static str,
}

pub static ORACLE_SIGNATURES: &[OracleUnitSignature] = &[
    OracleUnitSignature {
        name: "nfpa13_sprinkler",
        param_units: &["gpm/psi^0.5", "psi"],
        return_unit: "gpm",
        standard: "NFPA 13",
    },
    OracleUnitSignature {
        name: "nfpa13_demanda",
        param_units: &["gpm", "", "ft"],
        return_unit: "psi",
        standard: "NFPA 13",
    },
    OracleUnitSignature {
        name: "nfpa20_bomba_hp",
        param_units: &["gpm", "psi", "fraction"],
        return_unit: "hp",
        standard: "NFPA 20",
    },
    OracleUnitSignature {
        name: "nfpa20_presion",
        param_units: &["gpm"],
        return_unit: "psi",
        standard: "NFPA 20",
    },
    OracleUnitSignature {
        name: "nfpa72_cobertura",
        param_units: &["m2"],
        return_unit: "m2",
        standard: "NFPA 72",
    },
    OracleUnitSignature {
        name: "nfpa72_detectores",
        param_units: &["m2", "m2"],
        return_unit: "",
        standard: "NFPA 72",
    },
    OracleUnitSignature {
        name: "corriente_cc",
        param_units: &["V", "ohm"],
        return_unit: "A",
        standard: "IEC 60038",
    },
    OracleUnitSignature {
        name: "caida_tension",
        param_units: &["A", "m", "ohm_m", "mm2"],
        return_unit: "V",
        standard: "IEC 60228",
    },
    OracleUnitSignature {
        name: "corriente_carga",
        param_units: &["kW", "V", "fraction"],
        return_unit: "A",
        standard: "IEC 60038",
    },
];

pub fn lookup_oracle_sig(name: &str) -> Option<&'static OracleUnitSignature> {
    ORACLE_SIGNATURES.iter().find(|s| s.name == name)
}

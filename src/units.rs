// ═══════════════════════════════════════════════════════════════════════
// CRYS-L v2.0 — Physical Unit Algebra & NFPA Validation
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

// ── Unit Conversion Engine (v2.2) ──────────────────────────────────────

/// One conversion entry: from_unit × factor = to_unit
struct Conv {
    from:   &'static str,
    to:     &'static str,
    factor: f64,
    offset: f64,   // for °C↔°F; 0.0 for all others
}

/// Full conversion table — bidirectional where possible.
/// Factor convention: result = value * factor + offset
static CONV_TABLE: &[Conv] = &[
    // Area
    Conv { from: "ft2",  to: "m2",   factor: 0.092903,   offset: 0.0 },
    Conv { from: "m2",   to: "ft2",  factor: 10.76391,   offset: 0.0 },
    Conv { from: "in2",  to: "cm2",  factor: 6.4516,     offset: 0.0 },
    Conv { from: "cm2",  to: "in2",  factor: 0.155,      offset: 0.0 },
    // Pressure
    Conv { from: "psi",  to: "bar",  factor: 0.0689476,  offset: 0.0 },
    Conv { from: "bar",  to: "psi",  factor: 14.5038,    offset: 0.0 },
    Conv { from: "psi",  to: "kPa",  factor: 6.89476,    offset: 0.0 },
    Conv { from: "kPa",  to: "psi",  factor: 0.145038,   offset: 0.0 },
    Conv { from: "bar",  to: "kPa",  factor: 100.0,      offset: 0.0 },
    Conv { from: "kPa",  to: "bar",  factor: 0.01,       offset: 0.0 },
    Conv { from: "psi",  to: "Pa",   factor: 6894.76,    offset: 0.0 },
    Conv { from: "Pa",   to: "psi",  factor: 0.000145038,offset: 0.0 },
    // Flow
    Conv { from: "gpm",  to: "Ls",   factor: 0.0630902,  offset: 0.0 },   // L/s
    Conv { from: "Ls",   to: "gpm",  factor: 15.8503,    offset: 0.0 },
    Conv { from: "gpm",  to: "m3h",  factor: 0.227124,   offset: 0.0 },   // m³/h
    Conv { from: "m3h",  to: "gpm",  factor: 4.40287,    offset: 0.0 },
    Conv { from: "Ls",   to: "m3h",  factor: 3.6,        offset: 0.0 },
    Conv { from: "m3h",  to: "Ls",   factor: 0.277778,   offset: 0.0 },
    // Length
    Conv { from: "ft",   to: "m",    factor: 0.3048,     offset: 0.0 },
    Conv { from: "m",    to: "ft",   factor: 3.28084,    offset: 0.0 },
    Conv { from: "in",   to: "cm",   factor: 2.54,       offset: 0.0 },
    Conv { from: "cm",   to: "in",   factor: 0.393701,   offset: 0.0 },
    Conv { from: "in",   to: "mm",   factor: 25.4,       offset: 0.0 },
    Conv { from: "mm",   to: "in",   factor: 0.039370,   offset: 0.0 },
    // Power
    Conv { from: "HP",   to: "kW",   factor: 0.7457,     offset: 0.0 },
    Conv { from: "kW",   to: "HP",   factor: 1.34102,    offset: 0.0 },
    Conv { from: "HP",   to: "W",    factor: 745.7,      offset: 0.0 },
    Conv { from: "W",    to: "HP",   factor: 0.00134102, offset: 0.0 },
    // Temperature
    Conv { from: "C",    to: "F",    factor: 1.8,        offset: 32.0 },
    Conv { from: "F",    to: "C",    factor: 0.555556,   offset: -17.7778 },
    Conv { from: "C",    to: "K",    factor: 1.0,        offset: 273.15 },
    Conv { from: "K",    to: "C",    factor: 1.0,        offset: -273.15 },
    // Mass / density
    Conv { from: "lb",   to: "kg",   factor: 0.453592,   offset: 0.0 },
    Conv { from: "kg",   to: "lb",   factor: 2.20462,    offset: 0.0 },
    Conv { from: "lbft3",to: "kgm3", factor: 16.0185,    offset: 0.0 },
    Conv { from: "kgm3", to: "lbft3",factor: 0.062428,   offset: 0.0 },
    // Velocity
    Conv { from: "fps",  to: "ms",   factor: 0.3048,     offset: 0.0 },   // ft/s → m/s
    Conv { from: "ms",   to: "fps",  factor: 3.28084,    offset: 0.0 },
    // Electrical
    Conv { from: "kW",   to: "kVA",  factor: 1.0,        offset: 0.0 },   // PF=1 approx
    Conv { from: "kVA",  to: "kW",   factor: 1.0,        offset: 0.0 },
];

/// Convert a numeric value from one physical unit to another.
/// Returns `Err` if the conversion pair is unknown.
///
/// # Examples
/// ```
/// use crysl_lib::units::convert;
/// assert!((convert(100.0, "ft2", "m2").unwrap() - 9.29).abs() < 0.01);
/// assert!((convert(65.0,  "psi", "bar").unwrap() - 4.48).abs() < 0.01);
/// assert!((convert(100.0, "gpm", "Ls").unwrap()  - 6.31).abs() < 0.01);
/// ```
pub fn convert(value: f64, from: &str, to: &str) -> Result<f64, String> {
    if from == to { return Ok(value); }

    // Direct lookup
    if let Some(c) = CONV_TABLE.iter().find(|c| c.from == from && c.to == to) {
        return Ok(value * c.factor + c.offset);
    }

    // Two-step via SI base (psi → kPa → bar)
    // Find a path: from → X → to
    for mid in &["m", "m2", "Pa", "kPa", "kW", "Ls", "K", "kg", "ms"] {
        let a = CONV_TABLE.iter().find(|c| c.from == from && c.to == *mid);
        let b = CONV_TABLE.iter().find(|c| c.from == *mid && c.to == to);
        if let (Some(a), Some(b)) = (a, b) {
            let intermediate = value * a.factor + a.offset;
            return Ok(intermediate * b.factor + b.offset);
        }
    }

    Err(format!(
        "CRYS-L: no unit conversion for '{}' → '{}'. \
         Supported: ft2↔m2, psi↔bar/kPa, gpm↔Ls/m3h, ft↔m, HP↔kW, C↔F↔K, lb↔kg",
        from, to
    ))
}

/// Auto-detect if a value was given in metric and convert to imperial for NFPA.
/// NFPA 13 plans expect ft2 and psi. Returns (converted_value, used_unit).
pub fn to_nfpa_imperial(value: f64, unit: &str) -> (f64, &'static str) {
    match unit {
        "m2"  => (value * 10.76391, "ft2"),
        "m"   => (value * 3.28084,  "ft"),
        "bar" => (value * 14.5038,  "psi"),
        "kPa" => (value * 0.145038, "psi"),
        "Ls"  => (value * 15.8503,  "gpm"),
        "m3h" => (value * 4.40287,  "gpm"),
        "kW"  => (value * 1.34102,  "HP"),
        _     => (value,            "unchanged"),
    }
}

/// Float formatting helpers for wasm.
///
/// Rust's core float-to-decimal formatting has had wasm-facing panics in some
/// toolchain/browser combinations (see `dragon.rs` panics). To avoid that class
/// of issues, these helpers do **not** use `format!` on floats.
///
/// They:
/// - Handle `NaN`/`±Inf` explicitly.
/// - For finite values, scale + round into an `i64`, then format integers.

#[inline]
pub fn fmt_f32_fixed(v: f32, decimals: usize) -> String {
    fmt_f64_fixed(v as f64, decimals)
}

#[inline]
pub fn fmt_f32_signed_fixed(v: f32, decimals: usize) -> String {
    fmt_f64_signed_fixed(v as f64, decimals)
}

#[inline]
pub fn fmt_f64_fixed(v: f64, decimals: usize) -> String {
    fmt_f64_fixed_inner(v, decimals, false)
}

#[inline]
pub fn fmt_f64_signed_fixed(v: f64, decimals: usize) -> String {
    fmt_f64_fixed_inner(v, decimals, true)
}

fn fmt_f64_fixed_inner(v: f64, decimals: usize, force_sign: bool) -> String {
    if !v.is_finite() {
        return if v.is_nan() {
            "NaN".to_string()
        } else if v.is_sign_positive() {
            "Inf".to_string()
        } else {
            "-Inf".to_string()
        };
    }

    // Clamp decimals to something reasonable to avoid huge powers.
    let decimals = decimals.min(9);

    // Compute 10^decimals as both f64 (for scaling) and i64 (for splitting).
    let scale_i64 = 10_i64.checked_pow(decimals as u32).unwrap_or(1_i64);
    let scale_f = scale_i64 as f64;

    // Scale + round into an integer.
    let scaled = (v * scale_f).round();
    if !scaled.is_finite() {
        // Extremely large values can overflow the scale.
        return if v.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }

    // Keep within i64 range. If it doesn't fit, degrade gracefully.
    if scaled.abs() > (i64::MAX as f64) {
        return if v.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }

    let scaled_i = scaled as i64;

    // Preserve a negative sign for -0.0 if the caller forces a sign.
    let negative = scaled_i < 0 || (scaled_i == 0 && v.is_sign_negative());

    let abs_i = scaled_i.abs();
    let int_part = abs_i / scale_i64;
    let frac_part = abs_i % scale_i64;

    let mut out = String::new();

    if negative {
        out.push('-');
    } else if force_sign {
        out.push('+');
    }

    out.push_str(&int_part.to_string());

    if decimals > 0 {
        out.push('.');
        let frac_str = frac_part.to_string();
        // Left-pad with zeros.
        for _ in 0..decimals.saturating_sub(frac_str.len()) {
            out.push('0');
        }
        out.push_str(&frac_str);
    }

    out
}

//! Pure-`std` special mathematical functions (`ln_gamma`, `inc_gamma_q`,
//! `erfc_std`, `chi2_sf`) and deterministic `SplitMix64` PRNG.

use std::f64::consts::PI;

/// Natural logarithm of the Gamma function `ln(Gamma(a))` for finite `a > 0`.
pub fn ln_gamma(a: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 {
        return f64::NAN;
    }
    let mut x = a;
    let mut shift_prod = 1.0_f64;
    while x < 15.0 {
        shift_prod *= x;
        x += 1.0;
    }
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    let series = inv_x
        * (1.0 / 12.0
            + inv_x2
                * (-1.0 / 360.0
                    + inv_x2
                        * (1.0 / 1260.0 + inv_x2 * (-1.0 / 1680.0 + inv_x2 * (1.0 / 1188.0)))));
    let ln_gamma_x = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + series;
    ln_gamma_x - shift_prod.ln()
}

/// Regularized upper incomplete gamma function `Q(a, x)` for `a > 0`, `x >= 0`.
pub fn inc_gamma_q(a: f64, x: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }

    let log_prefactor = -x + a * x.ln() - ln_gamma(a);
    if log_prefactor < -745.0 && x >= a + 1.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..50_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() <= sum.abs() * 1e-15 {
                let p = sum * log_prefactor.exp();
                return (1.0 - p).clamp(0.0, 1.0);
            }
        }
        let p = sum * log_prefactor.exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = if b.abs() < TINY { 1.0 / TINY } else { 1.0 / b };
        let mut h = d;
        for i in 1..=50_000 {
            let fi = i as f64;
            let an = -fi * (fi - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= 1e-15 {
                break;
            }
        }
        if log_prefactor < -745.0 {
            0.0
        } else {
            (h * log_prefactor.exp()).clamp(0.0, 1.0)
        }
    }
}

/// Complementary error function `erfc(x)`.
pub fn erfc_std(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        if x > 28.0 {
            0.0
        } else {
            inc_gamma_q(0.5, x * x)
        }
    } else if x < -28.0 {
        2.0
    } else {
        2.0 - inc_gamma_q(0.5, x * x)
    }
}

/// Upper-tail survival function `P(chi^2_df >= stat)` for `df >= 1`.
pub fn chi2_sf(df: usize, stat: f64) -> f64 {
    if df == 0 {
        return 1.0;
    }
    if stat.is_nan() {
        return f64::NAN;
    }
    if stat <= 0.0 {
        return 1.0;
    }
    if !stat.is_finite() {
        return 0.0;
    }
    inc_gamma_q((df as f64) * 0.5, stat * 0.5)
}

/// Deterministic `SplitMix64` PRNG.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let u = (bits as f64 + 0.5) / ((1u64 << 53) as f64);
        u.clamp(1e-15, 1.0 - 1e-15)
    }

    pub fn next_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        let u1 = self.next_unit_f64().max(1e-12);
        let u2 = self.next_unit_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn math_special_functions_match_reference_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-14);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-13);
        assert!((erfc_std(0.0) - 1.0).abs() < 1e-14);
        assert!((erfc_std(1.959_963_984_540_054 * FRAC_1_SQRT_2) - 0.05).abs() < 1e-10);
        assert!((chi2_sf(2, 5.991_464_547_107_979) - 0.05).abs() < 1e-10);
    }
}

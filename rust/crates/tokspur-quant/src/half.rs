//! Half-precision and bfloat16 conversions used by the block encodings.

/// IEEE binary16 bits of `value`, rounded to nearest with ties to even.
///
/// This is the conversion the reference quantisers store their block scales with.
pub fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        // Infinity keeps a zero payload. A NaN keeps its top payload bits and stays quiet.
        let payload = if mantissa == 0 {
            0
        } else {
            0x0200 | (mantissa >> 13) as u16
        };
        return sign | 0x7c00 | payload;
    }

    // Biased binary16 exponent of the same value.
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        // Below the smallest normal binary16. Anything under half the smallest subnormal, every single-precision subnormal included, rounds to zero.
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let kept = significand >> shift;
        let dropped = significand & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let rounded = if dropped > halfway || (dropped == halfway && kept & 1 == 1) {
            kept + 1
        } else {
            kept
        };
        return sign | rounded as u16;
    }

    let kept = ((half_exponent as u32) << 10) | (mantissa >> 13);
    let dropped = mantissa & 0x1fff;
    // A carry out of the mantissa moves into the exponent, up to infinity, as it should.
    let rounded = if dropped > 0x1000 || (dropped == 0x1000 && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    };
    sign | rounded as u16
}

/// The single-precision value of IEEE binary16 `bits`. Exact.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0,
        (0, _) => {
            // Subnormal: renormalise so the leading one becomes the implicit bit.
            let shift = mantissa.leading_zeros() - 21;
            let normalised = (mantissa << shift) & 0x03ff;
            ((113 - shift) << 23) | (normalised << 13)
        }
        (0x1f, _) => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// The single-precision value of bfloat16 `bits`. Exact: bfloat16 is the top half of binary32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The bfloat16 bits of `value` when it is exactly representable, `None` otherwise.
pub fn f32_to_bf16_exact(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    (bits & 0xffff == 0).then_some((bits >> 16) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_survives_a_round_trip() {
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            if value.is_nan() {
                assert!(f16_to_f32(f16_bits(value)).is_nan());
            } else {
                assert_eq!(f16_bits(value), bits, "half {bits:#06x}");
            }
        }
    }

    #[test]
    fn known_values_convert() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(-0.0), 0x8000);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits(1.0e-10), 0x0000);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    }

    #[test]
    fn ties_round_to_even() {
        // 1 + 2^-11 lies exactly between 0x3c00 and 0x3c01.
        assert_eq!(f16_bits(1.0 + 2.0f32.powi(-11)), 0x3c00);
        // 1 + 3 * 2^-11 lies exactly between 0x3c01 and 0x3c02.
        assert_eq!(f16_bits(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
        // Just above the first tie rounds up.
        assert_eq!(f16_bits(1.0 + 2.0f32.powi(-11) + 2.0f32.powi(-23)), 0x3c01);
        // The largest finite half plus half a step overflows to infinity.
        assert_eq!(f16_bits(65520.0), 0x7c00);
        assert_eq!(f16_bits(65519.996), 0x7bff);
        // Half the smallest subnormal ties to zero, anything above it rounds up.
        assert_eq!(f16_bits(2.0f32.powi(-25)), 0x0000);
        assert_eq!(f16_bits(2.0f32.powi(-25) * 1.000_000_1), 0x0001);
        // Between the largest subnormal and the smallest normal.
        assert_eq!(f16_bits(2.0f32.powi(-14) - 2.0f32.powi(-25)), 0x0400);
    }

    #[test]
    fn bfloat16_is_the_top_half_of_binary32() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
        assert_eq!(f32_to_bf16_exact(1.0), Some(0x3f80));
        assert_eq!(f32_to_bf16_exact(0.1), None);
    }
}

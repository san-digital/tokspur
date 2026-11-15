// Copyright (c) 2023-2026 The ggml authors
// Adapted from llama.cpp 6050b467 ggml/src/ggml-quants.c (best_index_int8, quantize_row_iq4_nl_impl as quantize_iq4_nl calls it without an importance matrix, dequantize_row_iq4_nl) and ggml/src/ggml-common.h (kvalues_iq4nl) under the MIT licence in native/third_party/llama-cpp/LICENSE.

//! `IQ4_NL`: blocks of 32 weights, one half-precision scale and 4-bit indices into a fixed non-linear table.

use crate::half::{f16_bits, f16_to_f32};

/// Weights per block.
pub const BLOCK_VALUES: usize = 32;
/// A half scale, then 16 bytes of index nibbles.
pub const BLOCK_BYTES: usize = 18;

/// The non-linear level table (`kvalues_iq4nl`).
pub const LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Scale candidates tried on each side of the direct fit.
const TRIES: i32 = 7;
/// Blocks whose largest magnitude is below this encode as zero (`GROUP_MAX_EPS`).
const SMALLEST_MAGNITUDE: f32 = 1.0e-15;

/// Index of the table level nearest to `value`.
fn nearest_level(value: f32) -> usize {
    if value <= f32::from(LEVELS[0]) {
        return 0;
    }
    if value >= f32::from(LEVELS[15]) {
        return 15;
    }
    let mut lower = 0usize;
    let mut upper = 15usize;
    while upper - lower > 1 {
        let middle = (lower + upper) / 2;
        if value < f32::from(LEVELS[middle]) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    if value - f32::from(LEVELS[upper - 1]) < f32::from(LEVELS[upper]) - value {
        upper - 1
    } else {
        upper
    }
}

/// Weighted projections of the block onto the levels chosen at `inverse` scale.
fn projections(block: &[f32], inverse: f32) -> (f32, f32) {
    let mut sum_qx = 0.0f32;
    let mut sum_q2 = 0.0f32;
    for &value in block {
        let level = f32::from(LEVELS[nearest_level(inverse * value)]);
        let weight = value * value;
        sum_qx += weight * level * value;
        sum_q2 += weight * level * level;
    }
    (sum_qx, sum_q2)
}

/// Quantises whole blocks. `values.len()` is a multiple of 32 and `out` holds 18 bytes per block.
pub fn quantize_iq4_nl(values: &[f32], out: &mut [u8]) {
    for (block, encoded) in values
        .as_chunks::<BLOCK_VALUES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_BYTES>().0.iter_mut())
    {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &value in block {
            let magnitude = value.abs();
            if magnitude > amax {
                amax = magnitude;
                max = value;
            }
        }

        let mut scale = 0.0f32;
        if amax >= SMALLEST_MAGNITUDE {
            // Direct fit: the largest weight lands on the end of the table.
            let (sum_qx, sum_q2) = projections(block, 1.0 / (-max / f32::from(LEVELS[0])));
            scale = if sum_q2 > 0.0 { sum_qx / sum_q2 } else { 0.0 };
            let mut best = scale * sum_qx;
            for attempt in -TRIES..=TRIES {
                let inverse = (attempt as f32 + f32::from(LEVELS[0])) / max;
                let (sum_qx, sum_q2) = projections(block, inverse);
                if sum_q2 > 0.0 && sum_qx * sum_qx > best * sum_q2 {
                    scale = sum_qx / sum_q2;
                    best = scale * sum_qx;
                }
            }
        }

        let scale_bits = f16_bits(scale);
        encoded[..2].copy_from_slice(&scale_bits.to_le_bytes());
        // The reference divides by the unrounded scale here; the decoder uses the stored half.
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        for (index, byte) in encoded[2..].iter_mut().enumerate() {
            let first = nearest_level(inverse * block[index]) as u8;
            let second = nearest_level(inverse * block[index + 16]) as u8;
            *byte = first | (second << 4);
        }
    }
}

/// Decodes whole blocks.
pub fn dequantize_iq4_nl(bytes: &[u8], out: &mut [f32]) {
    for (encoded, block) in bytes
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_VALUES>().0.iter_mut())
    {
        let scale = f16_to_f32(u16::from_le_bytes([encoded[0], encoded[1]]));
        for (index, &byte) in encoded[2..].iter().enumerate() {
            block[index] = scale * f32::from(LEVELS[usize::from(byte & 0x0f)]);
            block[index + 16] = scale * f32::from(LEVELS[usize::from(byte >> 4)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{relative_error, sample};

    #[test]
    fn nearest_level_picks_the_closest_table_entry() {
        assert_eq!(nearest_level(-500.0), 0);
        assert_eq!(nearest_level(500.0), 15);
        assert_eq!(nearest_level(0.0), 8);
        assert_eq!(nearest_level(-4.0), 8);
        assert_eq!(nearest_level(-5.0), 7);
        assert_eq!(nearest_level(7.0), 9);
        // Exactly between two levels the reference takes the upper one.
        assert_eq!(nearest_level(-115.5), 1);
        assert_eq!(nearest_level(-116.0), 0);
        assert_eq!(nearest_level(101.0), 15);
    }

    #[test]
    fn zero_block_has_a_zero_scale_and_the_level_nearest_zero() {
        let mut encoded = [0xffu8; BLOCK_BYTES];
        quantize_iq4_nl(&[0.0; 32], &mut encoded);
        assert_eq!(encoded[..2], [0, 0]);
        assert!(encoded[2..].iter().all(|&byte| byte == 0x88));
        let mut decoded = [1.0f32; 32];
        dequantize_iq4_nl(&encoded, &mut decoded);
        assert_eq!(decoded, [0.0; 32]);
    }

    #[test]
    fn round_trip_is_close() {
        let values = sample(32 * 200, 23, 0.02);
        let mut encoded = vec![0u8; 200 * BLOCK_BYTES];
        quantize_iq4_nl(&values, &mut encoded);
        let mut decoded = vec![0.0f32; values.len()];
        dequantize_iq4_nl(&encoded, &mut decoded);
        let error = relative_error(&values, &decoded);
        assert!(error < 0.12, "IQ4_NL relative error {error}");
    }

    #[test]
    fn a_block_of_table_multiples_decodes_exactly() {
        let mut values = [0.0f32; 32];
        for (index, value) in values.iter_mut().enumerate() {
            *value = f32::from(LEVELS[index % 16]) * 0.5;
        }
        let mut encoded = [0u8; BLOCK_BYTES];
        quantize_iq4_nl(&values, &mut encoded);
        let mut decoded = [0.0f32; 32];
        dequantize_iq4_nl(&encoded, &mut decoded);
        assert_eq!(decoded, values);
    }
}

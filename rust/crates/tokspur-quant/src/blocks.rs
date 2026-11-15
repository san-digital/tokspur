// Copyright (c) 2023-2026 The ggml authors
// Adapted from llama.cpp 6050b467 ggml/src/ggml-quants.c (quantize_row_q8_0_ref, quantize_row_q5_1_ref, quantize_row_q5_1_impl with make_qkx3_quants and the matching dequantisers) under the MIT licence in native/third_party/llama-cpp/LICENSE.

//! `Q8_0` and `Q5_1`: independent blocks of 32 weights.

use crate::half::{f16_bits, f16_to_f32};

/// Weights per block for both encodings.
pub const BLOCK_VALUES: usize = 32;
/// `Q8_0` block: a half scale, then 32 signed bytes.
pub const Q8_0_BYTES: usize = 34;
/// `Q5_1` block: half scale, half minimum, 32 high bits, 16 bytes of low nibbles.
pub const Q5_1_BYTES: usize = 24;

/// Quantises whole `Q8_0` blocks. `values.len()` is a multiple of 32 and `out` holds 34 bytes per block.
pub fn quantize_q8_0(values: &[f32], out: &mut [u8]) {
    for (block, encoded) in values
        .as_chunks::<BLOCK_VALUES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q8_0_BYTES>().0.iter_mut())
    {
        let amax = block.iter().fold(0.0f32, |amax, v| amax.max(v.abs()));
        let scale = amax / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        encoded[..2].copy_from_slice(&f16_bits(scale).to_le_bytes());
        for (value, byte) in block.iter().zip(&mut encoded[2..]) {
            // `roundf`: halves away from zero. The cast saturates like the reference's int8 store.
            *byte = ((value * inverse).round() as i8) as u8;
        }
    }
}

/// Decodes whole `Q8_0` blocks.
pub fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) {
    for (encoded, block) in bytes
        .as_chunks::<Q8_0_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_VALUES>().0.iter_mut())
    {
        let scale = f16_to_f32(u16::from_le_bytes([encoded[0], encoded[1]]));
        for (byte, value) in encoded[2..].iter().zip(block) {
            *value = f32::from(*byte as i8) * scale;
        }
    }
}

/// Quantises whole `Q5_1` blocks. `values.len()` is a multiple of 32 and `out` holds 24 bytes per block.
pub fn quantize_q5_1(values: &[f32], out: &mut [u8]) {
    for (block, encoded) in values
        .as_chunks::<BLOCK_VALUES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q5_1_BYTES>().0.iter_mut())
    {
        let mut min = f32::MAX;
        let mut max = -f32::MAX;
        for &value in block {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
        let scale = (max - min) / 31.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        encoded[..2].copy_from_slice(&f16_bits(scale).to_le_bytes());
        encoded[2..4].copy_from_slice(&f16_bits(min).to_le_bytes());

        let level = |value: f32| ((value - min) * inverse + 0.5) as u8;
        let mut high = 0u32;
        for index in 0..BLOCK_VALUES / 2 {
            let first = level(block[index]);
            let second = level(block[index + BLOCK_VALUES / 2]);
            encoded[8 + index] = (first & 0x0f) | ((second & 0x0f) << 4);
            high |= u32::from((first & 0x10) >> 4) << index;
            high |= u32::from((second & 0x10) >> 4) << (index + BLOCK_VALUES / 2);
        }
        encoded[4..8].copy_from_slice(&high.to_le_bytes());
    }
}

/// Search settings of the reference's fitted `Q5_1` path.
const SEARCH_START: f32 = -0.9;
const SEARCH_STEP: f32 = 0.05;
const SEARCH_STEPS: i32 = 36;
const TOP_LEVEL: i32 = 31;

/// Round to nearest, ties to even, as the reference's `nearest_int` does for small magnitudes.
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    ((shifted.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Least-squares scale and (non-positive) minimum for one block under `weights`.
fn fit_block(values: &[f32], weights: &[f32; BLOCK_VALUES]) -> (f32, f32) {
    let mut min = values[0];
    let mut max = values[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * values[0];
    for (&value, &weight) in values.iter().zip(weights).skip(1) {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
        sum_w += weight;
        sum_x += weight * value;
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        return (0.0, min);
    }

    let level_at = |inverse: f32, min: f32, value: f32| {
        let scaled = inverse * (value - min);
        if scaled > 0.0 {
            nearest_int(scaled.min(TOP_LEVEL as f32))
        } else {
            0
        }
    };
    let error_of = |scale: f32, min: f32, levels: &[i32; BLOCK_VALUES]| {
        let mut error = 0.0f32;
        for ((&value, &weight), &level) in values.iter().zip(weights).zip(levels) {
            let diff = scale * level as f32 + min - value;
            error += weight * (diff * diff);
        }
        error
    };

    let mut levels = [0i32; BLOCK_VALUES];
    let mut inverse = TOP_LEVEL as f32 / (max - min);
    let mut scale = 1.0 / inverse;
    for (level, &value) in levels.iter_mut().zip(values) {
        *level = level_at(inverse, min, value);
    }
    let mut best_error = error_of(scale, min, &levels);

    for step in 0..=SEARCH_STEPS {
        inverse = (SEARCH_START + SEARCH_STEP * step as f32 + TOP_LEVEL as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for ((level, &value), &weight) in levels.iter_mut().zip(values).zip(weights) {
            *level = level_at(inverse, min, value);
            let chosen = *level as f32;
            sum_l += weight * chosen;
            sum_l2 += weight * chosen * chosen;
            sum_xl += weight * chosen * value;
        }
        let determinant = sum_w * sum_l2 - sum_l * sum_l;
        if determinant > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let error = error_of(this_scale, this_min, &levels);
            if error < best_error {
                best_error = error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, min)
}

/// Quantises whole `Q5_1` blocks, fitting each block's scale and minimum by least squares.
///
/// This is the reference's fitted path with every weight given equal importance: a weight's error counts by `sqrt(row mean square + weight squared)`. `row_values` is the row length the mean square is taken over. Levels are then chosen against the stored half-precision scale and minimum, so each weight takes the nearest level the decoder can produce.
pub fn quantize_q5_1_fitted(values: &[f32], row_values: usize, out: &mut [u8]) {
    let row_bytes = row_values / BLOCK_VALUES * Q5_1_BYTES;
    for (row, encoded_row) in values
        .chunks_exact(row_values)
        .zip(out.chunks_exact_mut(row_bytes))
    {
        let mut sum_x2 = 0.0f32;
        for &value in row {
            sum_x2 += value * value;
        }
        let mean_square = sum_x2 / row_values as f32;
        for (block, encoded) in row
            .as_chunks::<BLOCK_VALUES>()
            .0
            .iter()
            .zip(encoded_row.as_chunks_mut::<Q5_1_BYTES>().0.iter_mut())
        {
            let mut weights = [0.0f32; BLOCK_VALUES];
            for (weight, &value) in weights.iter_mut().zip(block) {
                *weight = (mean_square + value * value).sqrt();
            }
            let (scale, min) = fit_block(block, &weights);
            let scale_bits = f16_bits(scale);
            let min_bits = f16_bits(min);
            encoded[..2].copy_from_slice(&scale_bits.to_le_bytes());
            encoded[2..4].copy_from_slice(&min_bits.to_le_bytes());

            let stored_scale = f16_to_f32(scale_bits);
            let stored_min = f16_to_f32(min_bits);
            let level = |value: f32| {
                if stored_scale > 0.0 {
                    nearest_int(((value - stored_min) / stored_scale).clamp(0.0, TOP_LEVEL as f32))
                        as u8
                } else {
                    0
                }
            };
            let mut high = 0u32;
            for index in 0..BLOCK_VALUES / 2 {
                let first = level(block[index]);
                let second = level(block[index + BLOCK_VALUES / 2]);
                encoded[8 + index] = (first & 0x0f) | ((second & 0x0f) << 4);
                high |= u32::from((first & 0x10) >> 4) << index;
                high |= u32::from((second & 0x10) >> 4) << (index + BLOCK_VALUES / 2);
            }
            encoded[4..8].copy_from_slice(&high.to_le_bytes());
        }
    }
}

/// Decodes whole `Q5_1` blocks.
pub fn dequantize_q5_1(bytes: &[u8], out: &mut [f32]) {
    for (encoded, block) in bytes
        .as_chunks::<Q5_1_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_VALUES>().0.iter_mut())
    {
        let scale = f16_to_f32(u16::from_le_bytes([encoded[0], encoded[1]]));
        let min = f16_to_f32(u16::from_le_bytes([encoded[2], encoded[3]]));
        let high = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
        for index in 0..BLOCK_VALUES / 2 {
            let nibbles = encoded[8 + index];
            let first = (nibbles & 0x0f) | ((((high >> index) & 1) as u8) << 4);
            let second = (nibbles >> 4) | ((((high >> (index + 16)) & 1) as u8) << 4);
            block[index] = f32::from(first) * scale + min;
            block[index + 16] = f32::from(second) * scale + min;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Vec<f32> {
        (0..64).map(|i| (i as f32 - 20.0) * 0.037).collect()
    }

    #[test]
    fn q8_0_matches_a_hand_computed_block() {
        let mut values = [0.0f32; 32];
        values[0] = 1.0;
        values[1] = -0.4;
        values[2] = 0.25;
        values[3] = 1.6 / 127.0;
        let mut out = [0u8; Q8_0_BYTES];
        quantize_q8_0(&values, &mut out);
        // scale = 1/127; 1.0 -> 127, -0.4 -> -50.8, 0.25 -> 31.75, 1.6/127 -> 1.6.
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), f16_bits(1.0 / 127.0));
        assert_eq!(out[2] as i8, 127);
        assert_eq!(out[3] as i8, -51);
        assert_eq!(out[4] as i8, 32);
        assert_eq!(out[5] as i8, 2);
        assert!(out[6..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn q8_0_zero_block_encodes_as_zero() {
        let mut out = [0xffu8; Q8_0_BYTES];
        quantize_q8_0(&[0.0; 32], &mut out);
        assert_eq!(out, [0u8; Q8_0_BYTES]);
    }

    #[test]
    fn q8_0_round_trip_stays_within_half_a_step() {
        let values = ramp();
        let mut encoded = vec![0u8; 2 * Q8_0_BYTES];
        quantize_q8_0(&values, &mut encoded);
        let mut decoded = vec![0.0f32; 64];
        dequantize_q8_0(&encoded, &mut decoded);
        for (block, restored) in values.chunks(32).zip(decoded.chunks(32)) {
            let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            for (a, b) in block.iter().zip(restored) {
                // Half a step, plus the half-precision rounding of the scale.
                assert!(
                    (a - b).abs() <= amax / 127.0 * 0.5 + amax * 0.0005,
                    "{a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn q5_1_matches_a_hand_computed_block() {
        // Levels 0..=31 exactly: min 0, scale 1.
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut out = [0u8; Q5_1_BYTES];
        quantize_q5_1(&values, &mut out);
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), f16_bits(1.0));
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), 0);
        // Values 16..=31 carry the fifth bit: they are the second half of the block.
        assert_eq!(
            u32::from_le_bytes([out[4], out[5], out[6], out[7]]),
            0xffff_0000
        );
        for index in 0..16u8 {
            assert_eq!(out[8 + usize::from(index)], index | (index << 4));
        }
        let mut decoded = [0.0f32; 32];
        dequantize_q5_1(&out, &mut decoded);
        assert_eq!(decoded.to_vec(), values);
    }

    #[test]
    fn q5_1_round_trip_stays_within_half_a_step() {
        let values = ramp();
        let mut encoded = vec![0u8; 2 * Q5_1_BYTES];
        quantize_q5_1(&values, &mut encoded);
        let mut decoded = vec![0.0f32; 64];
        dequantize_q5_1(&encoded, &mut decoded);
        for (block, restored) in values.chunks(32).zip(decoded.chunks(32)) {
            let min = block.iter().copied().fold(f32::MAX, f32::min);
            let max = block.iter().copied().fold(-f32::MAX, f32::max);
            for (a, b) in block.iter().zip(restored) {
                // Half a step, plus the half-precision rounding of scale and minimum.
                assert!((a - b).abs() <= (max - min) / 31.0 * 0.55, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn fitted_q5_1_is_closer_than_the_plain_reference() {
        let values = crate::testing::sample(640 * 64, 29, 0.02);
        let mut plain = vec![0u8; values.len() / 32 * Q5_1_BYTES];
        let mut fitted = plain.clone();
        quantize_q5_1(&values, &mut plain);
        quantize_q5_1_fitted(&values, 640, &mut fitted);
        let mut from_plain = vec![0.0f32; values.len()];
        let mut from_fitted = vec![0.0f32; values.len()];
        dequantize_q5_1(&plain, &mut from_plain);
        dequantize_q5_1(&fitted, &mut from_fitted);
        let plain_error = crate::testing::relative_error(&values, &from_plain);
        let fitted_error = crate::testing::relative_error(&values, &from_fitted);
        assert!(
            fitted_error < plain_error * 0.97,
            "{fitted_error} vs {plain_error}"
        );
        assert!(fitted_error < 0.05, "{fitted_error}");
    }

    #[test]
    fn fitted_q5_1_handles_constant_and_zero_blocks() {
        let mut values = vec![0.0f32; 64];
        values[32..].fill(-0.25);
        let mut encoded = vec![0u8; 2 * Q5_1_BYTES];
        quantize_q5_1_fitted(&values, 64, &mut encoded);
        let mut decoded = vec![1.0f32; 64];
        dequantize_q5_1(&encoded, &mut decoded);
        assert_eq!(decoded, values);
    }

    #[test]
    fn q5_1_constant_block_keeps_its_value_in_the_minimum() {
        let mut out = [0u8; Q5_1_BYTES];
        quantize_q5_1(&[0.5; 32], &mut out);
        let mut decoded = [0.0f32; 32];
        dequantize_q5_1(&out, &mut decoded);
        assert_eq!(decoded, [0.5; 32]);
    }
}

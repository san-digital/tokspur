// Copyright (c) 2023-2026 The ggml authors
// Adapted from llama.cpp 6050b467 ggml/src/ggml-quants.c (nearest_int, make_qkx2_quants, make_qkx3_quants, make_qp_quants, get_scale_min_k4, quantize_row_q4_K_ref, quantize_row_q5_K_ref, quantize_row_q4_K_impl and quantize_row_q5_K_impl at uniform importance, and the matching dequantisers) under the MIT licence in native/third_party/llama-cpp/LICENSE.

//! `Q4_K` and `Q5_K`: super-blocks of 256 weights in eight sub-blocks of 32.
//!
//! Each sub-block has a 6-bit scale and a 6-bit minimum relative to two half-precision super-block factors. Two fits are offered, neither needing calibration data: the reference path, and the reference's weighted path with every weight given equal importance, which searches more finely and fits the eight scales and minimums jointly.

use crate::half::{f16_bits, f16_to_f32};

/// Weights per super-block.
pub const SUPER_BLOCK_VALUES: usize = 256;
/// `Q4_K` super-block: two halves, 12 bytes of packed scales, 128 bytes of nibbles.
pub const Q4_K_BYTES: usize = 144;
/// `Q5_K` super-block: as `Q4_K` plus 32 bytes of fifth bits ahead of the nibbles.
pub const Q5_K_BYTES: usize = 176;

const SUB_BLOCK: usize = 32;
const SUB_BLOCKS: usize = SUPER_BLOCK_VALUES / SUB_BLOCK;

/// Search settings that differ between the two encodings.
struct Search {
    /// Largest quantisation level.
    levels: i32,
    /// First candidate offset from `levels` and the step between candidates.
    start: f32,
    step: f32,
    /// Candidates tried after the direct fit.
    steps: i32,
}

const Q4_K_SEARCH: Search = Search {
    levels: 15,
    start: -1.0,
    step: 0.1,
    steps: 20,
};
const Q5_K_SEARCH: Search = Search {
    levels: 31,
    start: -0.5,
    step: 0.1,
    steps: 15,
};

/// The reference's branch-free round to nearest, ties to even, for magnitudes up to 2^22.
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    ((shifted.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Fits `scale * level - offset` to one sub-block, minimising the weighted squared error.
///
/// Returns the scale and writes the chosen levels and the (non-negative) offset.
fn fit_sub_block(
    search: &Search,
    values: &[f32],
    weights: &[f32; SUB_BLOCK],
    levels: &mut [u8],
    offset: &mut f32,
) -> f32 {
    let top = search.levels;
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
    if max == min {
        levels.fill(0);
        *offset = -min;
        return 0.0;
    }

    let mut inverse = top as f32 / (max - min);
    let mut scale = 1.0 / inverse;
    let mut best_error = 0.0f32;
    for ((&value, &weight), level) in values.iter().zip(weights).zip(levels.iter_mut()) {
        let chosen = nearest_int(inverse * (value - min)).clamp(0, top);
        *level = chosen as u8;
        let diff = scale * chosen as f32 + min - value;
        best_error += weight * (diff * diff);
    }

    let mut trial = [0u8; SUB_BLOCK];
    for step in 0..=search.steps {
        inverse = (search.start + search.step * step as f32 + top as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for ((&value, &weight), level) in values.iter().zip(weights).zip(trial.iter_mut()) {
            let chosen = nearest_int(inverse * (value - min)).clamp(0, top);
            *level = chosen as u8;
            let chosen = chosen as f32;
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
            let mut error = 0.0f32;
            for ((&value, &weight), &level) in values.iter().zip(weights).zip(&trial) {
                let diff = this_scale * f32::from(level) + this_min - value;
                error += weight * (diff * diff);
            }
            if error < best_error {
                levels.copy_from_slice(&trial);
                best_error = error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *offset = -min;
    scale
}

/// Search settings of the weighted fit.
const WEIGHTED_START: f32 = -0.9;
const WEIGHTED_STEP: f32 = 0.05;
const WEIGHTED_STEPS: i32 = 36;
/// Scales or minimums that are all below this are stored as zero (`GROUP_MAX_EPS`).
const SMALLEST_FACTOR: f32 = 1.0e-15;
/// Largest stored scale or minimum.
const TOP_FACTOR: i32 = 63;

/// As [`fit_sub_block`] with the weighted routines' search: finer steps, and levels kept in range as the running minimum moves.
fn fit_sub_block_weighted(
    top: i32,
    values: &[f32],
    weights: &[f32; SUB_BLOCK],
    levels: &mut [u8],
    offset: &mut f32,
) -> f32 {
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
        levels.fill(0);
        *offset = -min;
        return 0.0;
    }

    let mut inverse = top as f32 / (max - min);
    let mut scale = 1.0 / inverse;
    let mut best_error = 0.0f32;
    for ((&value, &weight), level) in values.iter().zip(weights).zip(levels.iter_mut()) {
        let chosen = nearest_int(inverse * (value - min)).clamp(0, top);
        *level = chosen as u8;
        let diff = scale * chosen as f32 + min - value;
        best_error += weight * (diff * diff);
    }

    let mut trial = [0u8; SUB_BLOCK];
    for step in 0..=WEIGHTED_STEPS {
        inverse = (WEIGHTED_START + WEIGHTED_STEP * step as f32 + top as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for ((&value, &weight), level) in values.iter().zip(weights).zip(trial.iter_mut()) {
            let scaled = inverse * (value - min);
            let chosen = if scaled > 0.0 {
                nearest_int(scaled.min(top as f32))
            } else {
                0
            };
            *level = chosen as u8;
            let chosen = chosen as f32;
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
            let mut error = 0.0f32;
            for ((&value, &weight), &level) in values.iter().zip(weights).zip(&trial) {
                let diff = this_scale * f32::from(level) + this_min - value;
                error += weight * (diff * diff);
            }
            if error < best_error {
                levels.copy_from_slice(&trial);
                best_error = error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *offset = -min;
    scale
}

/// Stores eight non-negative factors as 6-bit multiples of one shared step, minimising the weighted squared error. Returns the step.
fn fit_shared_step(
    factors: &[f32; SUB_BLOCKS],
    weights: &[f32; SUB_BLOCKS],
    levels: &mut [u8; SUB_BLOCKS],
) -> f32 {
    let mut max = 0.0f32;
    for &factor in factors {
        if factor > max {
            max = factor;
        }
    }
    if max < SMALLEST_FACTOR {
        levels.fill(0);
        return 0.0;
    }
    let mut inverse = TOP_FACTOR as f32 / max;
    for (level, &factor) in levels.iter_mut().zip(factors) {
        *level = nearest_int(inverse * factor) as u8;
    }
    let scale = 1.0 / inverse;
    let mut best_error = 0.0f32;
    for ((&factor, &weight), &level) in factors.iter().zip(weights).zip(levels.iter()) {
        let diff = factor - scale * f32::from(level);
        best_error += weight * diff * diff;
    }
    for step in (-4i32..=4).filter(|&step| step != 0) {
        let candidate = (0.1 * step as f32 + TOP_FACTOR as f32) / max;
        let candidate_scale = 1.0 / candidate;
        let mut error = 0.0f32;
        for (&factor, &weight) in factors.iter().zip(weights) {
            let level = nearest_int(candidate * factor).min(TOP_FACTOR);
            let diff = factor - candidate_scale * level as f32;
            error += weight * diff * diff;
        }
        if error < best_error {
            best_error = error;
            inverse = candidate;
        }
    }

    let mut sum_lx = 0.0f32;
    let mut sum_l2 = 0.0f32;
    for ((level, &factor), &weight) in levels.iter_mut().zip(factors).zip(weights) {
        let chosen = nearest_int(inverse * factor).min(TOP_FACTOR);
        *level = chosen as u8;
        let chosen = chosen as f32;
        sum_lx += weight * factor * chosen;
        sum_l2 += weight * chosen * chosen;
    }
    // Move one factor at a time to a neighbouring level while that improves the joint fit.
    for _ in 0..5 {
        let mut changed = false;
        for ((level, &factor), &weight) in levels.iter_mut().zip(factors).zip(weights) {
            let current = f32::from(*level);
            let mut rest_lx = sum_lx - weight * factor * current;
            let mut rest_l2 = sum_l2 - weight * current * current;
            if rest_lx > 0.0 && rest_l2 > 0.0 {
                let moved = nearest_int(factor * rest_l2 / rest_lx).min(TOP_FACTOR);
                if moved != i32::from(*level) {
                    rest_lx += weight * factor * moved as f32;
                    rest_l2 += weight * moved as f32 * moved as f32;
                    if rest_lx * rest_lx * sum_l2 > sum_lx * sum_lx * rest_l2 {
                        *level = moved as u8;
                        sum_lx = rest_lx;
                        sum_l2 = rest_l2;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 }
}

/// Reads sub-block `index`'s 6-bit scale and minimum from the 12 packed bytes.
fn scale_and_min(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

/// Which fit chooses a super-block's scales and minimums.
#[derive(Clone, Copy)]
enum Path {
    Reference,
    Weighted,
}

/// Fits all eight sub-blocks by the reference path and packs their scales. Returns the fitted levels.
///
/// `header` is the 16 bytes shared by both encodings: the two half-precision factors and the 12 bytes of packed scales.
fn fit_super_block(search: &Search, values: &[f32], header: &mut [u8]) -> [u8; SUPER_BLOCK_VALUES] {
    let mut levels = [0u8; SUPER_BLOCK_VALUES];
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut offsets = [0.0f32; SUB_BLOCKS];
    let mut max_scale = 0.0f32;
    let mut max_offset = 0.0f32;

    for (index, (sub, sub_levels)) in values
        .as_chunks::<SUB_BLOCK>()
        .0
        .iter()
        .zip(levels.as_chunks_mut::<SUB_BLOCK>().0.iter_mut())
        .enumerate()
    {
        let mut sum_x2 = 0.0f32;
        for &value in sub {
            sum_x2 += value * value;
        }
        let average = (sum_x2 / SUB_BLOCK as f32).sqrt();
        let mut weights = [0.0f32; SUB_BLOCK];
        for (weight, &value) in weights.iter_mut().zip(sub) {
            *weight = average + value.abs();
        }
        scales[index] = fit_sub_block(search, sub, &weights, sub_levels, &mut offsets[index]);
        if scales[index] > max_scale {
            max_scale = scales[index];
        }
        if offsets[index] > max_offset {
            max_offset = offsets[index];
        }
    }

    let inverse_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inverse_offset = if max_offset > 0.0 {
        63.0 / max_offset
    } else {
        0.0
    };
    let packed = &mut header[4..16];
    packed.fill(0);
    for index in 0..SUB_BLOCKS {
        // The reference stores these through `uint8_t` before clamping.
        let scale = (nearest_int(inverse_scale * scales[index]) as u8).min(63);
        let offset = (nearest_int(inverse_offset * offsets[index]) as u8).min(63);
        if index < 4 {
            packed[index] = scale;
            packed[index + 4] = offset;
        } else {
            packed[index + 4] = (scale & 0x0f) | ((offset & 0x0f) << 4);
            packed[index - 4] |= (scale >> 4) << 6;
            packed[index] |= (offset >> 4) << 6;
        }
    }
    let scale_bits = f16_bits(max_scale / 63.0);
    let offset_bits = f16_bits(max_offset / 63.0);
    header[..2].copy_from_slice(&scale_bits.to_le_bytes());
    header[2..4].copy_from_slice(&offset_bits.to_le_bytes());
    levels
}

/// Fits all eight sub-blocks by the weighted path at uniform importance and packs their scales. Returns the fitted levels.
fn fit_super_block_weighted(
    search: &Search,
    values: &[f32],
    header: &mut [u8],
) -> [u8; SUPER_BLOCK_VALUES] {
    let mut sum_x2 = 0.0f32;
    for &value in values {
        sum_x2 += value * value;
    }
    let spread = 2.0 * sum_x2 / SUPER_BLOCK_VALUES as f32;

    let mut levels = [0u8; SUPER_BLOCK_VALUES];
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut offsets = [0.0f32; SUB_BLOCKS];
    let mut sums = [0.0f32; SUB_BLOCKS];
    for (index, (sub, sub_levels)) in values
        .as_chunks::<SUB_BLOCK>()
        .0
        .iter()
        .zip(levels.as_chunks_mut::<SUB_BLOCK>().0.iter_mut())
        .enumerate()
    {
        // With an importance of one a weight's error counts by this much.
        let mut weights = [0.0f32; SUB_BLOCK];
        for (weight, &value) in weights.iter_mut().zip(sub) {
            *weight = (spread + value * value).sqrt();
        }
        let mut sum_w = 0.0f32;
        for &weight in &weights {
            sum_w += weight;
        }
        sums[index] = sum_w;
        scales[index] = fit_sub_block_weighted(
            search.levels,
            sub,
            &weights,
            sub_levels,
            &mut offsets[index],
        );
    }

    let mut scale_levels = [0u8; SUB_BLOCKS];
    let mut offset_levels = [0u8; SUB_BLOCKS];
    let scale_step = fit_shared_step(&scales, &sums, &mut scale_levels);
    let offset_step = fit_shared_step(&offsets, &sums, &mut offset_levels);
    let packed = &mut header[4..16];
    packed.fill(0);
    for index in 0..SUB_BLOCKS {
        let scale = scale_levels[index].min(63);
        let offset = offset_levels[index].min(63);
        if index < 4 {
            packed[index] = scale;
            packed[index + 4] = offset;
        } else {
            packed[index + 4] = (scale & 0x0f) | ((offset & 0x0f) << 4);
            packed[index - 4] |= (scale >> 4) << 6;
            packed[index] |= (offset >> 4) << 6;
        }
    }
    header[..2].copy_from_slice(&f16_bits(scale_step).to_le_bytes());
    header[2..4].copy_from_slice(&f16_bits(offset_step).to_le_bytes());
    levels
}

/// Fits a super-block, then chooses every level against the stored half-precision factors.
fn encode_super_block(
    path: Path,
    search: &Search,
    values: &[f32],
    header: &mut [u8],
) -> [u8; SUPER_BLOCK_VALUES] {
    let mut levels = match path {
        Path::Reference => fit_super_block(search, values, header),
        Path::Weighted => fit_super_block_weighted(search, values, header),
    };
    let stored_scale = f16_to_f32(u16::from_le_bytes([header[0], header[1]]));
    let stored_offset = f16_to_f32(u16::from_le_bytes([header[2], header[3]]));
    for (index, (sub, sub_levels)) in values
        .as_chunks::<SUB_BLOCK>()
        .0
        .iter()
        .zip(levels.as_chunks_mut::<SUB_BLOCK>().0.iter_mut())
        .enumerate()
    {
        let (scale, offset) = scale_and_min(index, &header[4..16]);
        let step = stored_scale * f32::from(scale);
        if step == 0.0 {
            continue;
        }
        let shift = stored_offset * f32::from(offset);
        for (&value, level) in sub.iter().zip(sub_levels.iter_mut()) {
            *level = nearest_int((value + shift) / step).clamp(0, search.levels) as u8;
        }
    }
    levels
}

/// Quantises whole `Q4_K` super-blocks by the reference path. `values.len()` is a multiple of 256 and `out` holds 144 bytes per super-block.
pub fn quantize_q4_k(values: &[f32], out: &mut [u8]) {
    quantize_q4_k_by(Path::Reference, values, out);
}

/// As [`quantize_q4_k`], by the weighted path at uniform importance.
pub fn quantize_q4_k_fitted(values: &[f32], out: &mut [u8]) {
    quantize_q4_k_by(Path::Weighted, values, out);
}

fn quantize_q4_k_by(path: Path, values: &[f32], out: &mut [u8]) {
    for (block, encoded) in values
        .as_chunks::<SUPER_BLOCK_VALUES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q4_K_BYTES>().0.iter_mut())
    {
        let (header, nibbles) = encoded.split_at_mut(16);
        let levels = encode_super_block(path, &Q4_K_SEARCH, block, header);
        for (pair, packed) in levels
            .as_chunks::<64>()
            .0
            .iter()
            .zip(nibbles.as_chunks_mut::<32>().0.iter_mut())
        {
            for (index, byte) in packed.iter_mut().enumerate() {
                *byte = pair[index] | (pair[index + 32] << 4);
            }
        }
    }
}

/// Quantises whole `Q5_K` super-blocks by the reference path. `values.len()` is a multiple of 256 and `out` holds 176 bytes per super-block.
pub fn quantize_q5_k(values: &[f32], out: &mut [u8]) {
    quantize_q5_k_by(Path::Reference, values, out);
}

/// As [`quantize_q5_k`], by the weighted path at uniform importance.
pub fn quantize_q5_k_fitted(values: &[f32], out: &mut [u8]) {
    quantize_q5_k_by(Path::Weighted, values, out);
}

fn quantize_q5_k_by(path: Path, values: &[f32], out: &mut [u8]) {
    for (block, encoded) in values
        .as_chunks::<SUPER_BLOCK_VALUES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q5_K_BYTES>().0.iter_mut())
    {
        let (header, rest) = encoded.split_at_mut(16);
        let (high, nibbles) = rest.split_at_mut(32);
        let levels = encode_super_block(path, &Q5_K_SEARCH, block, header);
        high.fill(0);
        for (group, (pair, packed)) in levels
            .as_chunks::<64>()
            .0
            .iter()
            .zip(nibbles.as_chunks_mut::<32>().0.iter_mut())
            .enumerate()
        {
            let first_bit = 1u8 << (2 * group);
            let second_bit = 2u8 << (2 * group);
            for (index, byte) in packed.iter_mut().enumerate() {
                let mut first = pair[index];
                if first > 15 {
                    first -= 16;
                    high[index] |= first_bit;
                }
                let mut second = pair[index + 32];
                if second > 15 {
                    second -= 16;
                    high[index] |= second_bit;
                }
                *byte = first | (second << 4);
            }
        }
    }
}

/// Decodes whole `Q4_K` super-blocks.
pub fn dequantize_q4_k(bytes: &[u8], out: &mut [f32]) {
    for (encoded, block) in bytes
        .as_chunks::<Q4_K_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<SUPER_BLOCK_VALUES>().0.iter_mut())
    {
        let scale = f16_to_f32(u16::from_le_bytes([encoded[0], encoded[1]]));
        let offset = f16_to_f32(u16::from_le_bytes([encoded[2], encoded[3]]));
        let packed = &encoded[4..16];
        for (group, (nibbles, pair)) in encoded[16..]
            .as_chunks::<32>()
            .0
            .iter()
            .zip(block.as_chunks_mut::<64>().0.iter_mut())
            .enumerate()
        {
            let (scale_1, offset_1) = scale_and_min(2 * group, packed);
            let (scale_2, offset_2) = scale_and_min(2 * group + 1, packed);
            let (step_1, shift_1) = (scale * f32::from(scale_1), offset * f32::from(offset_1));
            let (step_2, shift_2) = (scale * f32::from(scale_2), offset * f32::from(offset_2));
            for (index, &byte) in nibbles.iter().enumerate() {
                pair[index] = step_1 * f32::from(byte & 0x0f) - shift_1;
                pair[index + 32] = step_2 * f32::from(byte >> 4) - shift_2;
            }
        }
    }
}

/// Decodes whole `Q5_K` super-blocks.
pub fn dequantize_q5_k(bytes: &[u8], out: &mut [f32]) {
    for (encoded, block) in bytes
        .as_chunks::<Q5_K_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<SUPER_BLOCK_VALUES>().0.iter_mut())
    {
        let scale = f16_to_f32(u16::from_le_bytes([encoded[0], encoded[1]]));
        let offset = f16_to_f32(u16::from_le_bytes([encoded[2], encoded[3]]));
        let packed = &encoded[4..16];
        let high = &encoded[16..48];
        for (group, (nibbles, pair)) in encoded[48..]
            .as_chunks::<32>()
            .0
            .iter()
            .zip(block.as_chunks_mut::<64>().0.iter_mut())
            .enumerate()
        {
            let (scale_1, offset_1) = scale_and_min(2 * group, packed);
            let (scale_2, offset_2) = scale_and_min(2 * group + 1, packed);
            let (step_1, shift_1) = (scale * f32::from(scale_1), offset * f32::from(offset_1));
            let (step_2, shift_2) = (scale * f32::from(scale_2), offset * f32::from(offset_2));
            let first_bit = 1u8 << (2 * group);
            let second_bit = 2u8 << (2 * group);
            for (index, &byte) in nibbles.iter().enumerate() {
                let first = (byte & 0x0f) + if high[index] & first_bit != 0 { 16 } else { 0 };
                let second = (byte >> 4) + if high[index] & second_bit != 0 { 16 } else { 0 };
                pair[index] = step_1 * f32::from(first) - shift_1;
                pair[index + 32] = step_2 * f32::from(second) - shift_2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{relative_error, sample};

    #[test]
    fn nearest_int_rounds_ties_to_even() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(14.6), 15);
        assert_eq!(nearest_int(-3.2), -3);
    }

    #[test]
    fn packed_scales_round_trip() {
        let mut header = [0u8; 16];
        let values = sample(256, 7, 0.02);
        encode_super_block(Path::Reference, &Q4_K_SEARCH, &values, &mut header);
        for index in 0..SUB_BLOCKS {
            let (scale, offset) = scale_and_min(index, &header[4..16]);
            assert!(scale <= 63 && offset <= 63);
        }
        // The largest scale and the largest offset always quantise to 63.
        let scales: Vec<u8> = (0..SUB_BLOCKS)
            .map(|i| scale_and_min(i, &header[4..16]).0)
            .collect();
        let offsets: Vec<u8> = (0..SUB_BLOCKS)
            .map(|i| scale_and_min(i, &header[4..16]).1)
            .collect();
        assert_eq!(scales.iter().max(), Some(&63));
        assert_eq!(offsets.iter().max(), Some(&63));
    }

    #[test]
    fn q4_k_round_trip_is_close() {
        let values = sample(256 * 16, 11, 0.03);
        let mut encoded = vec![0u8; 16 * Q4_K_BYTES];
        quantize_q4_k(&values, &mut encoded);
        let mut decoded = vec![0.0f32; values.len()];
        dequantize_q4_k(&encoded, &mut decoded);
        let error = relative_error(&values, &decoded);
        assert!(error < 0.09, "Q4_K relative error {error}");
    }

    #[test]
    fn q5_k_round_trip_is_closer_than_q4_k() {
        let values = sample(256 * 16, 13, 0.03);
        let mut q4 = vec![0u8; 16 * Q4_K_BYTES];
        let mut q5 = vec![0u8; 16 * Q5_K_BYTES];
        quantize_q4_k(&values, &mut q4);
        quantize_q5_k(&values, &mut q5);
        let mut from_q4 = vec![0.0f32; values.len()];
        let mut from_q5 = vec![0.0f32; values.len()];
        dequantize_q4_k(&q4, &mut from_q4);
        dequantize_q5_k(&q5, &mut from_q5);
        let error_q4 = relative_error(&values, &from_q4);
        let error_q5 = relative_error(&values, &from_q5);
        assert!(error_q5 < 0.045, "Q5_K relative error {error_q5}");
        assert!(error_q5 < error_q4 * 0.65, "{error_q5} vs {error_q4}");
    }

    #[test]
    fn zero_and_constant_super_blocks_decode_exactly_or_closely() {
        let mut encoded = [0xffu8; Q4_K_BYTES];
        quantize_q4_k(&[0.0; 256], &mut encoded);
        assert_eq!(encoded, [0u8; Q4_K_BYTES]);

        let mut encoded = [0u8; Q5_K_BYTES];
        quantize_q5_k(&[-0.25; 256], &mut encoded);
        let mut decoded = [0.0f32; 256];
        dequantize_q5_k(&encoded, &mut decoded);
        for value in decoded {
            assert!((value + 0.25).abs() < 1.0e-3, "{value}");
        }
    }

    #[test]
    fn the_weighted_fit_is_closer_than_the_reference() {
        let values = sample(256 * 64, 31, 0.02);
        let error =
            |quantize: fn(&[f32], &mut [u8]), dequantize: fn(&[u8], &mut [f32]), bytes: usize| {
                let mut encoded = vec![0u8; 64 * bytes];
                quantize(&values, &mut encoded);
                let mut decoded = vec![0.0f32; values.len()];
                dequantize(&encoded, &mut decoded);
                relative_error(&values, &decoded)
            };
        let q4_reference = error(quantize_q4_k, dequantize_q4_k, Q4_K_BYTES);
        let q4_fitted = error(quantize_q4_k_fitted, dequantize_q4_k, Q4_K_BYTES);
        let q5_reference = error(quantize_q5_k, dequantize_q5_k, Q5_K_BYTES);
        let q5_fitted = error(quantize_q5_k_fitted, dequantize_q5_k, Q5_K_BYTES);
        assert!(q4_fitted < q4_reference, "{q4_fitted} vs {q4_reference}");
        assert!(q5_fitted < q5_reference, "{q5_fitted} vs {q5_reference}");
    }

    #[test]
    fn the_weighted_fit_handles_zero_and_constant_super_blocks() {
        let mut encoded = [0xffu8; Q4_K_BYTES];
        quantize_q4_k_fitted(&[0.0; 256], &mut encoded);
        assert_eq!(encoded, [0u8; Q4_K_BYTES]);

        let mut encoded = [0u8; Q5_K_BYTES];
        quantize_q5_k_fitted(&[-0.25; 256], &mut encoded);
        let mut decoded = [0.0f32; 256];
        dequantize_q5_k(&encoded, &mut decoded);
        for value in decoded {
            assert!((value + 0.25).abs() < 1.0e-3, "{value}");
        }
    }

    #[test]
    fn shared_steps_reproduce_exact_multiples() {
        let factors = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 31.5];
        let mut levels = [0u8; SUB_BLOCKS];
        let step = fit_shared_step(&factors, &[1.0; SUB_BLOCKS], &mut levels);
        assert_eq!(step, 0.5);
        assert_eq!(levels, [0, 1, 2, 3, 4, 5, 6, 63]);
        assert_eq!(
            fit_shared_step(&[0.0; SUB_BLOCKS], &[1.0; SUB_BLOCKS], &mut levels),
            0.0
        );
        assert_eq!(levels, [0; SUB_BLOCKS]);
    }

    #[test]
    fn output_is_deterministic() {
        let values = sample(256 * 4, 17, 0.05);
        let mut first = vec![0u8; 4 * Q4_K_BYTES];
        let mut second = vec![0u8; 4 * Q4_K_BYTES];
        quantize_q4_k(&values, &mut first);
        quantize_q4_k(&values, &mut second);
        assert_eq!(first, second);
    }
}

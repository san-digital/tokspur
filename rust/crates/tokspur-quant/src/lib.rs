//! Conversion of the original Flash-Next checkpoint into the quantised GGUF layout the runtime serves.
//!
//! The block encodings follow the reference routines in llama.cpp's `ggml/src/ggml-quants.c`; each module carries its notice. They use strict IEEE single-precision arithmetic in the reference's operation order and never a fused multiply-add, so the output is the same function of the input on every platform. Nothing here needs calibration data.

#![forbid(unsafe_code)]

pub mod blocks;
pub mod convert;
pub mod gguf;
pub mod half;
pub mod iq4nl;
pub mod kquant;
pub mod layout;
pub mod model;
pub mod safetensors;
pub mod tokenizer;

use std::fmt;

/// A tensor encoding the converter can write, named as GGUF names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Encoding {
    F32,
    Bf16,
    Q8_0,
    Q5_1,
    Q4K,
    Q5K,
    Iq4Nl,
}

impl Encoding {
    /// Every encoding, in GGUF type order.
    pub const ALL: [Self; 7] = [
        Self::F32,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q4K,
        Self::Q5K,
        Self::Iq4Nl,
        Self::Bf16,
    ];

    /// The GGUF tensor type id.
    pub const fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Iq4Nl => 20,
            Self::Bf16 => 30,
        }
    }

    /// The encoding for a GGUF tensor type id, if the converter writes it.
    pub fn from_ggml_type(id: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|encoding| encoding.ggml_type() == id)
    }

    /// The GGUF spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q5_1 => "Q5_1",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Iq4Nl => "IQ4_NL",
        }
    }

    /// The encoding spelled `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|encoding| encoding.name() == name)
    }

    /// Weights per independently encoded block.
    pub const fn block_values(self) -> usize {
        match self {
            Self::F32 | Self::Bf16 => 1,
            Self::Q8_0 | Self::Q5_1 | Self::Iq4Nl => 32,
            Self::Q4K | Self::Q5K => kquant::SUPER_BLOCK_VALUES,
        }
    }

    /// Bytes per encoded block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::Q8_0 => blocks::Q8_0_BYTES,
            Self::Q5_1 => blocks::Q5_1_BYTES,
            Self::Q4K => kquant::Q4_K_BYTES,
            Self::Q5K => kquant::Q5_K_BYTES,
            Self::Iq4Nl => iq4nl::BLOCK_BYTES,
        }
    }

    /// Encoded size of `values` weights, which must fill whole blocks.
    pub fn encoded_len(self, values: u64) -> Result<u64, QuantError> {
        let block = self.block_values() as u64;
        if !values.is_multiple_of(block) {
            return Err(QuantError::Shape(
                "tensor does not fill whole encoding blocks",
            ));
        }
        (values / block)
            .checked_mul(self.block_bytes() as u64)
            .ok_or(QuantError::Shape("encoded tensor size overflows"))
    }
}

/// Why an encode or decode was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantError {
    /// Lengths do not describe whole blocks, or the output does not match the input.
    Shape(&'static str),
    /// A weight is not finite, or cannot be stored exactly where exactness is required.
    Value(&'static str),
}

impl fmt::Display for QuantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) | Self::Value(reason) => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for QuantError {}

fn check_lengths(encoding: Encoding, values: usize, bytes: usize) -> Result<(), QuantError> {
    if encoding.encoded_len(values as u64)? != bytes as u64 {
        return Err(QuantError::Shape(
            "encoded buffer does not match the weight count",
        ));
    }
    Ok(())
}

/// Encodes `values` into `out`, which must be exactly [`Encoding::encoded_len`] bytes.
///
/// Every weight must be finite. `BF16` output additionally requires each weight to be exactly representable, which holds for weights read from a bfloat16 checkpoint.
pub fn quantize(encoding: Encoding, values: &[f32], out: &mut [u8]) -> Result<(), QuantError> {
    check_lengths(encoding, values.len(), out.len())?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(QuantError::Value("weight is not finite"));
    }
    match encoding {
        Encoding::F32 => {
            for (value, bytes) in values.iter().zip(out.as_chunks_mut::<4>().0.iter_mut()) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
        }
        Encoding::Bf16 => {
            for (value, bytes) in values.iter().zip(out.as_chunks_mut::<2>().0.iter_mut()) {
                let bits = half::f32_to_bf16_exact(*value)
                    .ok_or(QuantError::Value("weight is not exact in bfloat16"))?;
                bytes.copy_from_slice(&bits.to_le_bytes());
            }
        }
        Encoding::Q8_0 => blocks::quantize_q8_0(values, out),
        Encoding::Q5_1 => blocks::quantize_q5_1(values, out),
        Encoding::Q4K => kquant::quantize_q4_k(values, out),
        Encoding::Q5K => kquant::quantize_q5_k(values, out),
        Encoding::Iq4Nl => iq4nl::quantize_iq4_nl(values, out),
    }
    Ok(())
}

/// How an encoding's block parameters are chosen where the converter offers a choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fit {
    /// The reference routine for the encoding.
    Reference,
    /// The reference's weighted fit with every weight given equal importance: a least-squares search for each block's scale and minimum. `Q5_1`, `Q4_K` and `Q5_K` have one; it lands closer to the original weights than their plain reference.
    LeastSquares,
}

/// Encodes rows of `row_values` weights, as [`quantize`] does, with a choice of fit.
pub fn quantize_rows(
    encoding: Encoding,
    fit: Fit,
    row_values: usize,
    values: &[f32],
    out: &mut [u8],
) -> Result<(), QuantError> {
    if fit == Fit::Reference {
        return quantize(encoding, values, out);
    }
    if !matches!(encoding, Encoding::Q5_1 | Encoding::Q4K | Encoding::Q5K) {
        return Err(QuantError::Shape("this encoding has no least-squares fit"));
    }
    if row_values == 0
        || !row_values.is_multiple_of(encoding.block_values())
        || !values.len().is_multiple_of(row_values)
    {
        return Err(QuantError::Shape("rows do not fill whole encoding blocks"));
    }
    check_lengths(encoding, values.len(), out.len())?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(QuantError::Value("weight is not finite"));
    }
    match encoding {
        Encoding::Q4K => kquant::quantize_q4_k_fitted(values, out),
        Encoding::Q5K => kquant::quantize_q5_k_fitted(values, out),
        _ => blocks::quantize_q5_1_fitted(values, row_values, out),
    }
    Ok(())
}

/// Decodes `bytes` into `out`, the inverse length relation of [`quantize`].
pub fn dequantize(encoding: Encoding, bytes: &[u8], out: &mut [f32]) -> Result<(), QuantError> {
    check_lengths(encoding, out.len(), bytes.len())?;
    match encoding {
        Encoding::F32 => {
            for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
                *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        Encoding::Bf16 => {
            for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
                *value = half::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        Encoding::Q8_0 => blocks::dequantize_q8_0(bytes, out),
        Encoding::Q5_1 => blocks::dequantize_q5_1(bytes, out),
        Encoding::Q4K => kquant::dequantize_q4_k(bytes, out),
        Encoding::Q5K => kquant::dequantize_q5_k(bytes, out),
        Encoding::Iq4Nl => iq4nl::dequantize_iq4_nl(bytes, out),
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testing {
    /// Deterministic bell-shaped test weights (sum of four uniform draws).
    pub(crate) fn sample(count: usize, seed: u64, spread: f32) -> Vec<f32> {
        let mut state = seed | 1;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..count)
            .map(|_| (uniform() + uniform() + uniform() + uniform() - 2.0) * spread)
            .collect()
    }

    /// Root of summed squared error over summed squared weights.
    pub(crate) fn relative_error(original: &[f32], restored: &[f32]) -> f64 {
        let mut error = 0.0f64;
        let mut energy = 0.0f64;
        for (a, b) in original.iter().zip(restored) {
            error += f64::from(a - b) * f64::from(a - b);
            energy += f64::from(*a) * f64::from(*a);
        }
        (error / energy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::sample;

    #[test]
    fn encodings_report_the_gguf_ids_and_sizes() {
        for encoding in Encoding::ALL {
            assert_eq!(
                Encoding::from_ggml_type(encoding.ggml_type()),
                Some(encoding)
            );
            assert_eq!(Encoding::from_name(encoding.name()), Some(encoding));
        }
        assert_eq!(Encoding::Q4K.encoded_len(2560 * 640), Ok(921_600));
        assert_eq!(Encoding::Q5_1.encoded_len(640), Ok(480));
        assert_eq!(Encoding::Iq4Nl.encoded_len(160), Ok(90));
        assert!(Encoding::Q4K.encoded_len(640).is_err());
        assert_eq!(Encoding::from_ggml_type(1), None);
    }

    #[test]
    fn every_encoding_round_trips_through_the_dispatchers() {
        let values = sample(512, 3, 0.04);
        for encoding in Encoding::ALL {
            if encoding == Encoding::Bf16 {
                continue;
            }
            let Ok(len) = encoding.encoded_len(values.len() as u64) else {
                panic!("512 weights fill whole blocks of {}", encoding.name());
            };
            let mut encoded = vec![0u8; len as usize];
            assert_eq!(quantize(encoding, &values, &mut encoded), Ok(()));
            let mut decoded = vec![0.0f32; values.len()];
            assert_eq!(dequantize(encoding, &encoded, &mut decoded), Ok(()));
            let worst = values
                .iter()
                .zip(&decoded)
                .fold(0.0f32, |worst, (a, b)| worst.max((a - b).abs()));
            assert!(worst < 0.03, "{} worst error {worst}", encoding.name());
        }
    }

    #[test]
    fn bfloat16_output_must_be_exact() {
        let exact = [1.0f32, -2.5, 0.0];
        let mut out = [0u8; 6];
        assert_eq!(quantize(Encoding::Bf16, &exact, &mut out), Ok(()));
        let mut decoded = [0.0f32; 3];
        assert_eq!(dequantize(Encoding::Bf16, &out, &mut decoded), Ok(()));
        assert_eq!(decoded, exact);
        assert_eq!(
            quantize(Encoding::Bf16, &[0.1, 0.0, 0.0], &mut out),
            Err(QuantError::Value("weight is not exact in bfloat16"))
        );
    }

    #[test]
    fn bad_lengths_and_non_finite_weights_are_refused() {
        let mut out = [0u8; 34];
        assert!(quantize(Encoding::Q8_0, &[0.0; 31], &mut out).is_err());
        assert!(quantize(Encoding::Q8_0, &[0.0; 32], &mut out[..33]).is_err());
        let mut values = [0.0f32; 32];
        values[5] = f32::NAN;
        assert_eq!(
            quantize(Encoding::Q8_0, &values, &mut out),
            Err(QuantError::Value("weight is not finite"))
        );
    }
}

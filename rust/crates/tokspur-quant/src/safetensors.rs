//! Reader for the original checkpoint: safetensors shards read with positioned reads.
//!
//! A shard is an 8-byte little-endian header length, a JSON header mapping tensor names to `dtype`, `shape` and `data_offsets`, then the tensor bytes. Tensors are never mapped or loaded whole; callers read the element ranges they need.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use tokspur_json::{Json, JsonError, JsonLimits, parse_json_limited};

use crate::half::bf16_to_f32;

/// Largest JSON header accepted. The checkpoint's largest is under a megabyte.
const HEADER_LIMIT: u64 = 64 * 1024 * 1024;

/// Element types the converter reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    I64,
}

impl Dtype {
    /// Bytes per element.
    pub const fn width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

/// Where one tensor lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Index into [`Checkpoint::shards`].
    pub shard: usize,
    /// Absolute byte offset of the first element in the shard file.
    pub offset: u64,
    /// Element count, the product of `shape`.
    pub elements: u64,
}

/// Why the checkpoint could not be read.
#[derive(Debug)]
pub enum CheckpointError {
    Io { path: PathBuf, source: io::Error },
    Json { path: PathBuf, source: JsonError },
    Format { path: PathBuf, reason: &'static str },
    Missing(String),
    Range(&'static str),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Json { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Format { path, reason } => write!(formatter, "{}: {reason}", path.display()),
            Self::Missing(name) => write!(formatter, "checkpoint has no tensor {name}"),
            Self::Range(reason) => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Format { .. } | Self::Missing(_) | Self::Range(_) => None,
        }
    }
}

/// The opened shards of one checkpoint and the tensors they hold.
#[derive(Debug)]
pub struct Checkpoint {
    shards: Vec<(PathBuf, File)>,
    tensors: BTreeMap<String, TensorInfo>,
}

fn format(path: &Path, reason: &'static str) -> CheckpointError {
    CheckpointError::Format {
        path: path.to_owned(),
        reason,
    }
}

fn unsigned(path: &Path, value: &Json) -> Result<u64, CheckpointError> {
    value
        .unsigned_integer()
        .map_err(|_| format(path, "safetensors header has a non-integer field"))
}

impl Checkpoint {
    /// Opens `paths` and indexes their tensors. A tensor name may appear in one shard only.
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<Self, CheckpointError> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensors = BTreeMap::new();
        for (index, path) in paths.iter().enumerate() {
            let path = path.as_ref();
            let io = |source| CheckpointError::Io {
                path: path.to_owned(),
                source,
            };
            let file = File::open(path).map_err(io)?;
            let size = file.metadata().map_err(io)?.len();
            let mut length = [0u8; 8];
            file.read_exact_at(&mut length, 0).map_err(io)?;
            let header_length = u64::from_le_bytes(length);
            if header_length == 0 || header_length > HEADER_LIMIT || header_length > size - 8 {
                return Err(format(path, "safetensors header length is out of range"));
            }
            let mut header = vec![0u8; header_length as usize];
            file.read_exact_at(&mut header, 8).map_err(io)?;
            let limits = JsonLimits::new(header.len(), 4_000_000, 8);
            let json =
                parse_json_limited(&header, limits).map_err(|source| CheckpointError::Json {
                    path: path.to_owned(),
                    source,
                })?;
            let data_start = 8 + header_length;
            let entries = json
                .object()
                .map_err(|_| format(path, "safetensors header is not an object"))?;
            for (name, entry) in entries {
                if name.as_slice() == b"__metadata__" {
                    continue;
                }
                let name = String::from_utf8(name.clone())
                    .map_err(|_| format(path, "tensor name is not UTF-8"))?;
                let field = |key: &str| {
                    entry
                        .find(key)
                        .ok()
                        .flatten()
                        .ok_or_else(|| format(path, "safetensors tensor entry lacks a field"))
                };
                let dtype = match field("dtype")?.string() {
                    Ok(b"BF16") => Dtype::Bf16,
                    Ok(b"F32") => Dtype::F32,
                    Ok(b"I64") => Dtype::I64,
                    _ => return Err(format(path, "unsupported safetensors dtype")),
                };
                let shape = field("shape")?
                    .array()
                    .map_err(|_| format(path, "safetensors shape is not an array"))?
                    .iter()
                    .map(|dimension| unsigned(path, dimension))
                    .collect::<Result<Vec<u64>, _>>()?;
                let offsets = field("data_offsets")?
                    .array()
                    .map_err(|_| format(path, "safetensors offsets are not an array"))?;
                let [begin, end] = offsets.as_slice() else {
                    return Err(format(path, "safetensors offsets are not a pair"));
                };
                let (begin, end) = (unsigned(path, begin)?, unsigned(path, end)?);
                let elements = shape
                    .iter()
                    .try_fold(1u64, |total, &dimension| total.checked_mul(dimension))
                    .ok_or_else(|| format(path, "safetensors shape overflows"))?;
                let bytes = elements
                    .checked_mul(dtype.width())
                    .ok_or_else(|| format(path, "safetensors tensor size overflows"))?;
                if end < begin || end - begin != bytes || end > size - data_start {
                    return Err(format(path, "safetensors tensor extent is inconsistent"));
                }
                let info = TensorInfo {
                    dtype,
                    shape,
                    shard: index,
                    offset: data_start + begin,
                    elements,
                };
                if tensors.insert(name, info).is_some() {
                    return Err(format(path, "tensor appears in more than one shard"));
                }
            }
            shards.push((path.to_owned(), file));
        }
        Ok(Self { shards, tensors })
    }

    /// Every tensor by name.
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, CheckpointError> {
        self.tensors
            .get(name)
            .ok_or_else(|| CheckpointError::Missing(name.to_owned()))
    }

    /// The shard paths, in the order they were opened.
    pub fn shards(&self) -> impl Iterator<Item = &Path> {
        self.shards.iter().map(|(path, _)| path.as_path())
    }

    fn read(&self, info: &TensorInfo, first: u64, bytes: &mut [u8]) -> Result<(), CheckpointError> {
        let width = info.dtype.width();
        let count = bytes.len() as u64 / width;
        if first
            .checked_add(count)
            .is_none_or(|end| end > info.elements)
        {
            return Err(CheckpointError::Range("read past the end of a tensor"));
        }
        let (path, file) = &self.shards[info.shard];
        file.read_exact_at(bytes, info.offset + first * width)
            .map_err(|source| CheckpointError::Io {
                path: path.clone(),
                source,
            })
    }

    /// Reads `out.len()` elements of a floating-point tensor starting at element `first`.
    ///
    /// `scratch` is reused between calls to hold the raw bytes.
    pub fn read_f32(
        &self,
        info: &TensorInfo,
        first: u64,
        out: &mut [f32],
        scratch: &mut Vec<u8>,
    ) -> Result<(), CheckpointError> {
        let width = info.dtype.width() as usize;
        scratch.resize(out.len() * width, 0);
        self.read(info, first, scratch)?;
        match info.dtype {
            Dtype::Bf16 => {
                for (value, chunk) in out.iter_mut().zip(scratch.as_chunks::<2>().0.iter()) {
                    *value = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            Dtype::F32 => {
                for (value, chunk) in out.iter_mut().zip(scratch.as_chunks::<4>().0.iter()) {
                    *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            Dtype::I64 => return Err(CheckpointError::Range("integer tensor read as floats")),
        }
        Ok(())
    }

    /// Reads a whole 64-bit integer tensor.
    pub fn read_i64(&self, name: &str) -> Result<Vec<i64>, CheckpointError> {
        let info = self.tensor(name)?;
        if info.dtype != Dtype::I64 {
            return Err(CheckpointError::Range(
                "tensor does not hold 64-bit integers",
            ));
        }
        let mut bytes = vec![0u8; info.elements as usize * 8];
        self.read(info, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn shard(directory: &Path, name: &str, header: &str, data: &[u8]) -> PathBuf {
        let path = directory.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(data).unwrap();
        path
    }

    fn scratch_directory(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("tokspur-quant-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn reads_ranges_of_bf16_f32_and_i64_tensors() {
        let directory = scratch_directory("read");
        let mut data = Vec::new();
        for bits in [0x3f80u16, 0xc000, 0x3f00, 0x0000] {
            data.extend_from_slice(&bits.to_le_bytes());
        }
        data.extend_from_slice(&7i64.to_le_bytes());
        data.extend_from_slice(&(-9i64).to_le_bytes());
        data.extend_from_slice(&1.5f32.to_le_bytes());
        let header = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"n":{"dtype":"I64","shape":[2],"data_offsets":[8,24]},"f":{"dtype":"F32","shape":[1],"data_offsets":[24,28]}}"#;
        let path = shard(&directory, "one.safetensors", header, &data);
        let checkpoint = Checkpoint::open(&[&path]).unwrap();
        assert_eq!(checkpoint.tensors().len(), 3);

        let a = checkpoint.tensor("a").unwrap();
        assert_eq!((a.shape.as_slice(), a.elements), ([2u64, 2].as_slice(), 4));
        let mut scratch = Vec::new();
        let mut values = [0.0f32; 3];
        checkpoint
            .read_f32(a, 1, &mut values, &mut scratch)
            .unwrap();
        assert_eq!(values, [-2.0, 0.5, 0.0]);
        assert!(
            checkpoint
                .read_f32(a, 2, &mut values, &mut scratch)
                .is_err()
        );

        let mut one = [0.0f32; 1];
        checkpoint
            .read_f32(checkpoint.tensor("f").unwrap(), 0, &mut one, &mut scratch)
            .unwrap();
        assert_eq!(one, [1.5]);
        assert_eq!(checkpoint.read_i64("n").unwrap(), vec![7, -9]);
        assert!(checkpoint.read_i64("a").is_err());
        assert!(checkpoint.tensor("missing").is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn refuses_inconsistent_headers() {
        let directory = scratch_directory("refuse");
        let wrong_extent = r#"{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,4]}}"#;
        let path = shard(&directory, "extent.safetensors", wrong_extent, &[0; 8]);
        assert!(Checkpoint::open(&[&path]).is_err());

        let past_end = r#"{"a":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}"#;
        let path = shard(&directory, "end.safetensors", past_end, &[0; 8]);
        assert!(Checkpoint::open(&[&path]).is_err());

        let unknown = r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
        let path = shard(&directory, "dtype.safetensors", unknown, &[0; 8]);
        assert!(Checkpoint::open(&[&path]).is_err());

        let ok = r#"{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#;
        let first = shard(&directory, "first.safetensors", ok, &[0; 8]);
        let second = shard(&directory, "second.safetensors", ok, &[0; 8]);
        assert!(Checkpoint::open(&[&first]).is_ok());
        assert!(Checkpoint::open(&[&first, &second]).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}

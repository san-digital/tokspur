//! Plans the output files and writes them: every tensor read from the checkpoint, transformed, encoded and written at its planned offset.
//!
//! The output is a pure function of the checkpoint and the layout. Worker count and chunking change only how fast it is produced.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use tokspur_text::sha256::{Sha256, hex};

use crate::gguf::{self, Record, Value};
use crate::layout::{Recipe, RowRun, TargetTensor, ValueOp};
use crate::safetensors::{Checkpoint, CheckpointError, TensorInfo};
use crate::{QuantError, dequantize, quantize_rows};

/// Weights converted per unit of work: 32 MiB of single precision.
const CHUNK_VALUES: u64 = 8 * 1024 * 1024;

/// A tensor with its place in an output file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub tensor: TargetTensor,
    /// Offset from the start of the file's data section.
    pub offset: u64,
    pub bytes: u64,
}

/// One output file.
#[derive(Debug, Clone, PartialEq)]
pub struct FilePlan {
    pub name: String,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<PlannedTensor>,
}

impl FilePlan {
    /// The file's header bytes.
    pub fn header(&self) -> Vec<u8> {
        let records: Vec<Record> = self
            .tensors
            .iter()
            .map(|planned| Record {
                name: planned.tensor.name.clone(),
                dimensions: planned.tensor.dimensions.clone(),
                kind: planned.tensor.encoding.ggml_type(),
                offset: planned.offset,
            })
            .collect();
        gguf::header(&self.metadata, &records)
    }

    /// Bytes of tensor data, each tensor padded to the alignment.
    pub fn data_bytes(&self) -> u64 {
        self.tensors
            .last()
            .map_or(0, |last| gguf::aligned(last.offset + last.bytes))
    }
}

/// Why a conversion failed.
#[derive(Debug)]
pub enum ConvertError {
    Checkpoint(CheckpointError),
    Quant(QuantError),
    Io { path: PathBuf, source: io::Error },
    Plan(&'static str),
    Cancelled,
}

impl fmt::Display for ConvertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Quant(error) => error.fmt(formatter),
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Plan(reason) => formatter.write_str(reason),
            Self::Cancelled => formatter.write_str("conversion stopped after an earlier error"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<CheckpointError> for ConvertError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

impl From<QuantError> for ConvertError {
    fn from(error: QuantError) -> Self {
        Self::Quant(error)
    }
}

fn place(tensors: &[TargetTensor]) -> Result<Vec<PlannedTensor>, ConvertError> {
    let mut offset = 0u64;
    let mut planned = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let bytes = tensor.encoding.encoded_len(tensor.values())?;
        // Rows must fill whole blocks so that no block spans two rows.
        tensor.encoding.encoded_len(tensor.row_values())?;
        planned.push(PlannedTensor {
            tensor: tensor.clone(),
            offset,
            bytes,
        });
        offset = gguf::aligned(offset + bytes);
    }
    Ok(planned)
}

/// Plans a single-file model, as the next-token head is stored.
pub fn plan_single(
    name: &str,
    metadata: Vec<(String, Value)>,
    tensors: &[TargetTensor],
) -> Result<FilePlan, ConvertError> {
    Ok(FilePlan {
        name: name.to_owned(),
        metadata,
        tensors: place(tensors)?,
    })
}

/// Plans a split model: a first file holding only the metadata, then files of tensors.
///
/// A new file starts when the next tensor would take the current one past `file_bytes`, the rule `llama-gguf-split --split-max-size` applies. `name` receives the file's one-based number and the file count.
pub fn plan_split(
    name: &dyn Fn(usize, usize) -> String,
    metadata: Vec<(String, Value)>,
    tensors: &[TargetTensor],
    file_bytes: u64,
) -> Result<Vec<FilePlan>, ConvertError> {
    let mut groups: Vec<Vec<TargetTensor>> = Vec::new();
    let mut used = 0u64;
    for tensor in tensors {
        let bytes = tensor.encoding.encoded_len(tensor.values())?;
        if groups.is_empty() || used + bytes > file_bytes {
            groups.push(Vec::new());
            used = 0;
        }
        if let Some(group) = groups.last_mut() {
            group.push(tensor.clone());
        }
        used += gguf::aligned(bytes);
    }
    let count = groups.len() + 1;
    let total = i32::try_from(tensors.len())
        .map_err(|_| ConvertError::Plan("too many tensors for a split model"))?;
    let split = |number: usize| {
        [
            ("split.no".to_owned(), Value::U16(number as u16)),
            ("split.tensors.count".to_owned(), Value::I32(total)),
            ("split.count".to_owned(), Value::U16(count as u16)),
        ]
    };
    if count > usize::from(u16::MAX) {
        return Err(ConvertError::Plan("too many files for a split model"));
    }
    let mut first = metadata;
    first.extend(split(0));
    let mut plans = vec![FilePlan {
        name: name(1, count),
        metadata: first,
        tensors: Vec::new(),
    }];
    for (index, group) in groups.iter().enumerate() {
        plans.push(FilePlan {
            name: name(index + 2, count),
            metadata: split(index + 1).to_vec(),
            tensors: place(group)?,
        });
    }
    Ok(plans)
}

/// Buffers one worker reuses between units.
#[derive(Default)]
struct Scratch {
    bytes: Vec<u8>,
    spare: Vec<f32>,
}

/// Fills `out` with `rows` consecutive target rows starting at `first_row`, before any value change.
fn read_rows(
    checkpoint: &Checkpoint,
    tensor: &TargetTensor,
    first_row: u64,
    out: &mut [f32],
    scratch: &mut Scratch,
) -> Result<(), ConvertError> {
    let width = tensor.row_values() as usize;
    match &tensor.recipe {
        Recipe::Rows(runs) => {
            let mut skip = first_row;
            let mut filled = 0usize;
            for RowRun {
                source,
                first_row: run_first,
                rows,
            } in runs
            {
                if filled == out.len() {
                    break;
                }
                if skip >= *rows {
                    skip -= rows;
                    continue;
                }
                let take = ((rows - skip) as usize).min((out.len() - filled) / width);
                let info = checkpoint.tensor(source)?;
                checkpoint.read_f32(
                    info,
                    (run_first + skip) * width as u64,
                    &mut out[filled..filled + take * width],
                    &mut scratch.bytes,
                )?;
                filled += take * width;
                skip = 0;
            }
            if filled != out.len() {
                return Err(ConvertError::Plan(
                    "tensor recipe is shorter than the tensor",
                ));
            }
        }
        Recipe::Wide { left, right } => {
            let half = width / 2;
            let rows = out.len() / width;
            scratch.spare.resize(rows * half, 0.0);
            for (source, side) in [(left, 0usize), (right, half)] {
                let info: &TensorInfo = checkpoint.tensor(source)?;
                checkpoint.read_f32(
                    info,
                    first_row * half as u64,
                    &mut scratch.spare,
                    &mut scratch.bytes,
                )?;
                for (row, part) in out
                    .chunks_exact_mut(width)
                    .zip(scratch.spare.chunks_exact(half))
                {
                    row[side..side + half].copy_from_slice(part);
                }
            }
        }
        Recipe::Columns { source, columns } => {
            let info = checkpoint.tensor(source)?;
            scratch.spare.resize(out.len(), 0.0);
            checkpoint.read_f32(
                info,
                first_row * width as u64,
                &mut scratch.spare,
                &mut scratch.bytes,
            )?;
            for (row, original) in out
                .chunks_exact_mut(width)
                .zip(scratch.spare.chunks_exact(width))
            {
                for (value, &column) in row.iter_mut().zip(columns) {
                    *value = original[column as usize];
                }
            }
        }
    }
    Ok(())
}

/// `-exp(value)`, correctly rounded to single precision.
///
/// The exponential is taken in double precision, which is accurate to within one of its own steps on every platform. That settles the single-precision rounding unless the result sits almost exactly between two singles; such a value is refused rather than left to the platform's library.
fn negative_exponential(value: f32) -> Result<f32, ConvertError> {
    let exact = f64::from(value).exp();
    let rounded = exact as f32;
    let neighbour = if f64::from(rounded) < exact {
        rounded.next_up()
    } else {
        rounded.next_down()
    };
    let midpoint = (f64::from(rounded) + f64::from(neighbour)) / 2.0;
    if (exact - midpoint).abs() <= exact * 8.0 * f64::EPSILON {
        return Err(ConvertError::Plan(
            "a decay constant sits on a single-precision rounding boundary",
        ));
    }
    Ok(-rounded)
}

/// Reads target rows as the weights that get encoded.
pub fn read_target_rows(
    checkpoint: &Checkpoint,
    tensor: &TargetTensor,
    first_row: u64,
    out: &mut [f32],
) -> Result<(), ConvertError> {
    let mut scratch = Scratch::default();
    read_values(checkpoint, tensor, first_row, out, &mut scratch)
}

fn read_values(
    checkpoint: &Checkpoint,
    tensor: &TargetTensor,
    first_row: u64,
    out: &mut [f32],
    scratch: &mut Scratch,
) -> Result<(), ConvertError> {
    read_rows(checkpoint, tensor, first_row, out, scratch)?;
    match tensor.op {
        ValueOp::Keep => {}
        ValueOp::PlusOne => {
            for value in out.iter_mut() {
                *value += 1.0;
            }
        }
        ValueOp::NegativeExponential => {
            for value in out.iter_mut() {
                *value = negative_exponential(*value)?;
            }
        }
    }
    Ok(())
}

/// How far one tensor's encoding is from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorReport {
    pub name: String,
    pub encoding: &'static str,
    pub bytes: u64,
    pub sha256: String,
    /// Root of summed squared error over summed squared weights.
    pub relative_error: f64,
}

/// One written file.
#[derive(Debug, Clone, PartialEq)]
pub struct FileReport {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub tensors: Vec<TensorReport>,
}

struct Unit {
    file: usize,
    tensor: usize,
    first_row: u64,
    rows: u64,
}

#[derive(Default, Clone, Copy)]
struct Deviation {
    squared_error: f64,
    energy: f64,
}

fn units(plans: &[FilePlan]) -> Vec<Unit> {
    let mut units = Vec::new();
    for (file, plan) in plans.iter().enumerate() {
        for (tensor, planned) in plan.tensors.iter().enumerate() {
            let rows = planned.tensor.rows();
            let step = (CHUNK_VALUES / planned.tensor.row_values()).max(1);
            let mut first_row = 0;
            while first_row < rows {
                let take = step.min(rows - first_row);
                units.push(Unit {
                    file,
                    tensor,
                    first_row,
                    rows: take,
                });
                first_row += take;
            }
        }
    }
    units
}

fn hash_range(file: &File, path: &Path, start: u64, length: u64) -> Result<String, ConvertError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    let mut done = 0u64;
    while done < length {
        let take = buffer.len().min((length - done) as usize);
        file.read_exact_at(&mut buffer[..take], start + done)
            .map_err(|source| ConvertError::Io {
                path: path.to_owned(),
                source,
            })?;
        hasher.update(&buffer[..take]);
        done += take as u64;
    }
    Ok(hex(&hasher.finalize()))
}

/// The SHA-256 of a whole file.
pub fn hash_file(path: &Path) -> Result<String, ConvertError> {
    let io = |source| ConvertError::Io {
        path: path.to_owned(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(io)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Writes `plans` into `directory` using `workers` threads and returns what was written.
///
/// Files are written under a temporary name and renamed once complete. An existing output file is never replaced. `progress` receives the bytes of tensor data written so far and the total, from time to time.
pub fn write_files(
    checkpoint: &Checkpoint,
    plans: &[FilePlan],
    directory: &Path,
    workers: usize,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<Vec<FileReport>, ConvertError> {
    let io = |path: &Path| {
        let path = path.to_owned();
        move |source| ConvertError::Io { path, source }
    };
    fs::create_dir_all(directory).map_err(io(directory))?;
    let mut files = Vec::with_capacity(plans.len());
    let mut headers = Vec::with_capacity(plans.len());
    for plan in plans {
        let finished = directory.join(&plan.name);
        if finished.exists() {
            return Err(ConvertError::Io {
                path: finished,
                source: io::Error::new(io::ErrorKind::AlreadyExists, "output file already exists"),
            });
        }
        let partial = directory.join(format!("{}.part", plan.name));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&partial)
            .map_err(io(&partial))?;
        let header = plan.header();
        file.set_len(header.len() as u64 + plan.data_bytes())
            .map_err(io(&partial))?;
        file.write_all_at(&header, 0).map_err(io(&partial))?;
        headers.push(header.len() as u64);
        files.push((partial, finished, file));
    }

    let units = units(plans);
    let total: u64 = plans.iter().map(FilePlan::data_bytes).sum();
    let next = AtomicUsize::new(0);
    let written = AtomicU64::new(0);
    let failed = AtomicBool::new(false);
    let deviations: Vec<Vec<Mutex<Deviation>>> = plans
        .iter()
        .map(|plan| {
            plan.tensors
                .iter()
                .map(|_| Mutex::new(Deviation::default()))
                .collect()
        })
        .collect();

    let work = || -> Result<(), ConvertError> {
        let mut scratch = Scratch::default();
        let mut values = Vec::new();
        let mut encoded = Vec::new();
        let mut decoded = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(unit) = units.get(index) else {
                return Ok(());
            };
            if failed.load(Ordering::Relaxed) {
                return Err(ConvertError::Cancelled);
            }
            let planned = &plans[unit.file].tensors[unit.tensor];
            let tensor = &planned.tensor;
            let width = tensor.row_values();
            let row_bytes = tensor.encoding.encoded_len(width)?;
            values.resize((unit.rows * width) as usize, 0.0f32);
            read_values(
                checkpoint,
                tensor,
                unit.first_row,
                &mut values,
                &mut scratch,
            )?;
            encoded.resize((unit.rows * row_bytes) as usize, 0u8);
            quantize_rows(
                tensor.encoding,
                tensor.fit,
                width as usize,
                &values,
                &mut encoded,
            )?;

            decoded.resize(values.len(), 0.0f32);
            dequantize(tensor.encoding, &encoded, &mut decoded)?;
            let mut deviation = Deviation::default();
            for (original, restored) in values.iter().zip(&decoded) {
                let difference = f64::from(*original) - f64::from(*restored);
                deviation.squared_error += difference * difference;
                deviation.energy += f64::from(*original) * f64::from(*original);
            }
            if let Ok(mut total) = deviations[unit.file][unit.tensor].lock() {
                total.squared_error += deviation.squared_error;
                total.energy += deviation.energy;
            }

            let (partial, _, file) = &files[unit.file];
            let offset = headers[unit.file] + planned.offset + unit.first_row * row_bytes;
            file.write_all_at(&encoded, offset).map_err(io(partial))?;
            let done = written.fetch_add(encoded.len() as u64, Ordering::Relaxed);
            // Report roughly every 4 GiB.
            if (done >> 32) != ((done + encoded.len() as u64) >> 32) {
                progress(done + encoded.len() as u64, total);
            }
        }
    };
    let outcome: Result<(), ConvertError> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let result = work();
                    if result.is_err() {
                        failed.store(true, Ordering::Relaxed);
                    }
                    result
                })
            })
            .collect();
        let mut outcome = Ok(());
        for handle in handles {
            let result = handle
                .join()
                .unwrap_or(Err(ConvertError::Plan("a conversion worker panicked")));
            // Keep the first real failure rather than a worker that merely stopped.
            if matches!(outcome, Ok(()) | Err(ConvertError::Cancelled)) && result.is_err() {
                outcome = result;
            }
        }
        outcome
    });
    if let Err(error) = outcome {
        for (partial, _, _) in &files {
            // Best effort: the partial file is useless either way.
            let _ignored = fs::remove_file(partial);
        }
        return Err(error);
    }

    let mut reports = Vec::with_capacity(plans.len());
    for (index, plan) in plans.iter().enumerate() {
        let (partial, finished, file) = &files[index];
        file.sync_all().map_err(io(partial))?;
        let mut tensors = Vec::with_capacity(plan.tensors.len());
        for (planned, deviation) in plan.tensors.iter().zip(&deviations[index]) {
            let deviation = deviation.lock().map(|d| *d).unwrap_or_default();
            tensors.push(TensorReport {
                name: planned.tensor.name.clone(),
                encoding: planned.tensor.encoding.name(),
                bytes: planned.bytes,
                sha256: hash_range(
                    file,
                    partial,
                    headers[index] + planned.offset,
                    planned.bytes,
                )?,
                relative_error: if deviation.energy > 0.0 {
                    (deviation.squared_error / deviation.energy).sqrt()
                } else {
                    0.0
                },
            });
        }
        let bytes = headers[index] + plan.data_bytes();
        let sha256 = hash_file(partial)?;
        fs::hard_link(partial, finished).map_err(io(finished))?;
        fs::remove_file(partial).map_err(io(partial))?;
        reports.push(FileReport {
            name: plan.name.clone(),
            bytes,
            sha256,
            tensors,
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_exponential_is_correctly_rounded() {
        assert_eq!(negative_exponential(0.0).ok(), Some(-1.0));
        // exp(2.328125) = 10.258688448..., between the singles 10.25868797 and 10.25868893 and closer to the first.
        assert_eq!(negative_exponential(2.328_125).ok(), Some(-10.258_688));
        assert_eq!(
            negative_exponential(2.328_125).map(f32::to_bits).ok(),
            Some((-10.258_687_973_022_461_f64 as f32).to_bits())
        );
        assert_eq!(negative_exponential(-3.5).ok(), Some(-0.030_197_383));
    }
}

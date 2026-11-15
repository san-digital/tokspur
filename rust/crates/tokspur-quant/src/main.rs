//! `tokspur-quant`: converts the original Flash-Next checkpoint into quantised GGUF files.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tokspur_gguf::{Gguf, Metadata, Tensor};
use tokspur_json::parse_json;
use tokspur_quant::convert::{FilePlan, FileReport, plan_single, plan_split, write_files};
use tokspur_quant::gguf::{self, Record, Value};
use tokspur_quant::layout::{Layout, SourceIndex, layout};
use tokspur_quant::layout::{Recipe, TargetTensor};
use tokspur_quant::model::{
    Config, NextTokenLabels, NgramHash, Provenance, Text, model_metadata, next_token_metadata,
};
use tokspur_quant::safetensors::{Checkpoint, Dtype};
use tokspur_quant::tokenizer::vocabulary;
use tokspur_quant::{Encoding, Fit, dequantize, quantize, quantize_rows};
use tokspur_text::sha256::{Sha256, hex};

const USAGE: &str = "\
usage: tokspur-quant <command> [options]

commands:
  plan --originals <dir>                 Show the files a conversion would write
      [--only model|next-token|base]     Plan only the model, next-token head or CPU base file
      [--headers-to <dir>]               Also write each file's header, without tensor data
  convert --originals <dir> --out <dir>  Convert the original checkpoint to GGUF files
      [--only model|next-token|base]     Write only the model, next-token head or CPU base file
      [--jobs <n>]                       Worker threads (default: all cores but two)
      [--file-bytes <n>]                 Largest model file (default: 50000000000)
      [--name <text>]                    Model name recorded in the files
      [--repo-url <url>]                 Where the converted files are published
      [--source-repo <owner/name>]       Repository the checkpoint came from
      [--source-revision <commit>]       Its revision
      [--chat-template <file>]           Chat template (default: the checkpoint's own)
      [--report <file>]                  Write per-tensor hashes and errors as JSON
      [--index <json> --partial]         Convert only the tensors whose originals are present,
                                         laid out from a full index of the checkpoint
  extract-base --from <gguf>... --out <file>  Extract CPU base GGUF from converted shards
      [--chat-template <file>]           Override tokenizer.chat_template
      [--repo-url <url>]                 Set or override general.repo_url
      [--name <text>]                    Override general.name and general.base_model.0.name
  rewrite-header --from <gguf> --out <file>  Rewrite GGUF metadata without re-quantising
      [--chat-template <file>]           Override tokenizer.chat_template
      [--repo-url <url>]                 Set or override general.repo_url
      [--name <text>]                    Override general.name and general.base_model.0.name
  diff --a <gguf>... --b <gguf>...       Compare two sets of GGUF files tensor by tensor
  encode <encoding> <in.f32> <out.bin> [row]  Encode a raw little-endian f32 file;
                                         a row length selects the least-squares fit
  error <encoding> <in.f32> <encoded>    Report how far an encoding is from the original weights
";

const DEFAULT_FILE_BYTES: u64 = 50_000_000_000;
const DEFAULT_SOURCE_REPO: &str = "Qwen/Qwen3.8-Flash-Next";
const MODEL_FILE_STEM: &str = "Qwen3.8-Flash-Next-Q4_K";
const BASE_FILE: &str = "Qwen3.8-Flash-Next-base.gguf";
const BASE_TENSORS: [&str; 2] = ["per_layer_token_embd.weight", "token_embd.weight"];
const NEXT_TOKEN_FILE: &str = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

/// Options as `--name value` pairs, in order. Repeated names keep every value.
struct Options {
    names: Vec<String>,
    values: Vec<(String, String)>,
}

impl Options {
    /// Parses `--name value...`: every word up to the next `--name` is a value of that name.
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut names = Vec::new();
        let mut values = Vec::new();
        let mut current: Option<String> = None;
        for argument in arguments {
            if let Some(name) = argument.strip_prefix("--") {
                names.push(name.to_owned());
                current = Some(name.to_owned());
            } else if let Some(name) = &current {
                values.push((name.clone(), argument.clone()));
            } else {
                return Err(format!("unexpected argument {argument}"));
            }
        }
        Ok(Self { names, values })
    }

    /// Whether `--name` was given, with or without a value.
    fn flag(&self, name: &str) -> bool {
        self.names.iter().any(|given| given == name)
    }

    fn all(&self, name: &str) -> Vec<&str> {
        self.values
            .iter()
            .filter(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    fn one(&self, name: &str) -> Result<Option<&str>, String> {
        match self.all(name).as_slice() {
            [] => Ok(None),
            [value] => Ok(Some(value)),
            _ => Err(format!("--{name} takes one value")),
        }
    }

    fn required(&self, name: &str) -> Result<&str, String> {
        self.one(name)?
            .ok_or_else(|| format!("--{name} is required"))
    }

    fn only_known(&self, known: &[&str]) -> Result<(), String> {
        match self
            .names
            .iter()
            .find(|name| !known.contains(&name.as_str()))
        {
            Some(name) => Err(format!("unknown option --{name}")),
            None => Ok(()),
        }
    }
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn say(line: &str) -> Result<(), String> {
    writeln!(io::stdout(), "{line}").map_err(|error| error.to_string())
}

/// Everything read from the original checkpoint's folder.
struct Originals {
    checkpoint: Checkpoint,
    config: Config,
    text: Text,
    layout: Layout,
    ngram: NgramHash,
}

/// The id of the token a tokenizer setting names, given as a string or as `{"content": ...}`.
fn token_id(settings: &tokspur_json::Json, key: &str, tokens: &[Vec<u8>]) -> Result<u32, String> {
    let missing = || format!("tokenizer_config.json does not name its {key}");
    let value = settings
        .find(key)
        .map_err(|error| error.to_string())?
        .ok_or_else(missing)?;
    let content = match value.string() {
        Ok(content) => content,
        Err(_) => value
            .find("content")
            .ok()
            .flatten()
            .and_then(|content| content.string().ok())
            .ok_or_else(missing)?,
    };
    tokens
        .iter()
        .position(|token| token == content)
        .map(|id| id as u32)
        .ok_or_else(|| format!("{key} is not in the vocabulary"))
}

/// Reads a full tensor index: `{"name": {"dtype": "BF16", "shape": [..]}, ...}`.
fn read_index(path: &Path) -> Result<SourceIndex, String> {
    let bytes = read(path)?;
    let limits = tokspur_json::JsonLimits::new(bytes.len(), 1_000_000, 8);
    let json = tokspur_json::parse_json_limited(&bytes, limits)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let malformed = || format!("{} is not a tensor index", path.display());
    let mut index = SourceIndex::new();
    for (name, entry) in json.object().map_err(|_| malformed())? {
        let dtype = match entry
            .find("dtype")
            .ok()
            .flatten()
            .and_then(|d| d.string().ok())
        {
            Some(b"BF16") => Dtype::Bf16,
            Some(b"F32") => Dtype::F32,
            Some(b"I64") => Dtype::I64,
            _ => return Err(malformed()),
        };
        let shape = entry
            .find("shape")
            .ok()
            .flatten()
            .and_then(|shape| shape.array().ok())
            .ok_or_else(malformed)?
            .iter()
            .map(|dimension| dimension.unsigned_integer().map_err(|_| malformed()))
            .collect::<Result<Vec<u64>, String>>()?;
        index.insert(
            String::from_utf8(name.clone()).map_err(|_| malformed())?,
            (dtype, shape),
        );
    }
    Ok(index)
}

fn open_originals(
    directory: &Path,
    chat_template: Option<&str>,
    index: Option<&str>,
) -> Result<Originals, String> {
    let mut shards: Vec<PathBuf> = fs::read_dir(directory)
        .map_err(|error| format!("cannot list {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!(
            "{} holds no safetensors files",
            directory.display()
        ));
    }
    let checkpoint = Checkpoint::open(&shards).map_err(|error| error.to_string())?;
    let config = Config::parse(&read(&directory.join("config.json"))?)
        .map_err(|error| format!("config.json: {error}"))?;

    let present: SourceIndex = checkpoint
        .tensors()
        .iter()
        .map(|(name, info)| (name.clone(), (info.dtype, info.shape.clone())))
        .collect();
    let sources = match index {
        None => present,
        Some(path) => {
            let full = read_index(Path::new(path))?;
            // The index must describe the files that are here.
            if let Some((name, _)) = present
                .iter()
                .find(|(name, entry)| full.get(*name).is_some_and(|listed| listed != *entry))
            {
                return Err(format!("{name} differs from the tensor index"));
            }
            full
        }
    };
    let layout = layout(&config.geometry, &sources).map_err(|error| error.to_string())?;

    let vocabulary = vocabulary(
        &read(&directory.join("tokenizer.json"))?,
        config.geometry.vocabulary as usize,
    )
    .map_err(|error| format!("tokenizer.json: {error}"))?;
    let settings = parse_json(&read(&directory.join("tokenizer_config.json"))?)
        .map_err(|error| format!("tokenizer_config.json: {error}"))?;
    let sampling = parse_json(&read(&directory.join("generation_config.json"))?)
        .map_err(|error| format!("generation_config.json: {error}"))?;
    let number = |key: &str| -> Result<f64, String> {
        sampling
            .find(key)
            .ok()
            .flatten()
            .and_then(|value| value.number_lexeme().ok())
            .and_then(|lexeme| lexeme.parse::<f64>().ok())
            .ok_or_else(|| format!("generation_config.json lacks {key}"))
    };
    let template = match chat_template {
        Some(path) => read(Path::new(path))?,
        None => read(&directory.join("chat_template.jinja"))?,
    };
    let text = Text {
        end_token: token_id(&settings, "eos_token", &vocabulary.tokens)?,
        padding_token: token_id(&settings, "pad_token", &vocabulary.tokens)?,
        add_beginning_token: settings
            .find("add_bos_token")
            .ok()
            .flatten()
            .and_then(|value| value.boolean().ok())
            .unwrap_or(false),
        top_k: number("top_k")? as i32,
        top_p: number("top_p")? as f32,
        temperature: number("temperature")? as f32,
        vocabulary,
        chat_template: template,
    };

    let ngram_layer = config.geometry.ngram_layers.first().copied().unwrap_or(0);
    let constant = |name: &str| -> Result<Vec<u64>, String> {
        let full = format!("model.language_model.layers.{ngram_layer}.ple.ple_embedding.{name}");
        if index.is_some() && !checkpoint.tensors().contains_key(&full) {
            // A partial checkpoint may lack the file that holds the hash constants.
            return Ok(Vec::new());
        }
        checkpoint
            .read_i64(&full)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|value| u64::try_from(value).map_err(|_| format!("{full} holds a negative value")))
            .collect()
    };
    let row_width = layout
        .target
        .iter()
        .find(|tensor| tensor.name == "per_layer_token_embd.weight")
        .map(|tensor| tensor.row_values() as u32)
        .ok_or("layout has no n-gram table")?;
    let ngram = NgramHash {
        multipliers: constant("layer_multipliers")?,
        head_offsets: constant("ngram_heads_offsets")?,
        head_sizes: constant("ngram_heads_vocab_sizes")?,
        row_width,
    };
    Ok(Originals {
        checkpoint,
        config,
        text,
        layout,
        ngram,
    })
}

/// The tensors of `all` whose originals are in the checkpoint.
fn available(checkpoint: &Checkpoint, all: &[TargetTensor]) -> Vec<TargetTensor> {
    let here = |name: &String| checkpoint.tensors().contains_key(name);
    all.iter()
        .filter(|tensor| match &tensor.recipe {
            Recipe::Rows(runs) => runs.iter().all(|run| here(&run.source)),
            Recipe::Wide { left, right } => here(left) && here(right),
            Recipe::Columns { source, .. } => here(source),
        })
        .cloned()
        .collect()
}

fn plans(originals: &Originals, options: &Options) -> Result<Vec<FilePlan>, String> {
    let (target, next_token) = if options.flag("partial") {
        if options.one("index")?.is_none() {
            return Err("--partial needs --index".to_owned());
        }
        (
            available(&originals.checkpoint, &originals.layout.target),
            available(&originals.checkpoint, &originals.layout.next_token),
        )
    } else {
        (
            originals.layout.target.clone(),
            originals.layout.next_token.clone(),
        )
    };
    let only = options.one("only")?;
    if !matches!(only, None | Some("model" | "next-token" | "base")) {
        return Err("--only takes model, next-token or base".to_owned());
    }
    let file_bytes = match options.one("file-bytes")? {
        Some(text) => text
            .parse::<u64>()
            .ok()
            .filter(|&bytes| bytes > 0)
            .ok_or("--file-bytes takes a positive number")?,
        None => DEFAULT_FILE_BYTES,
    };
    let provenance = Provenance {
        name: options
            .one("name")?
            .unwrap_or("Qwen3.8 Flash Next")
            .to_owned(),
        quantized_by: "tokspur".to_owned(),
        repo_url: options.one("repo-url")?.map(str::to_owned),
        source_repo: options
            .one("source-repo")?
            .unwrap_or(DEFAULT_SOURCE_REPO)
            .to_owned(),
        source_revision: options
            .one("source-revision")?
            .unwrap_or("unknown")
            .to_owned(),
        converter: format!("tokspur-quant {}", env!("CARGO_PKG_VERSION")),
    };
    let mut plans = Vec::new();
    if only == Some("base") {
        let base_tensors: Vec<TargetTensor> = target
            .iter()
            .filter(|tensor| BASE_TENSORS.contains(&tensor.name.as_str()))
            .cloned()
            .collect();
        if base_tensors.len() != BASE_TENSORS.len() {
            return Err("checkpoint lacks required CPU base embedding tensors".to_owned());
        }
        let metadata = model_metadata(
            &originals.config,
            &originals.text,
            &originals.ngram,
            &provenance,
        );
        plans.push(
            plan_single(BASE_FILE, metadata, &base_tensors).map_err(|error| error.to_string())?,
        );
        return Ok(plans);
    }
    if only != Some("next-token") {
        let metadata = model_metadata(
            &originals.config,
            &originals.text,
            &originals.ngram,
            &provenance,
        );
        let name = |number: usize, count: usize| {
            format!("{MODEL_FILE_STEM}-{number:05}-of-{count:05}.gguf")
        };
        plans.extend(
            plan_split(&name, metadata, &target, file_bytes).map_err(|error| error.to_string())?,
        );
    }
    if only != Some("model") && !next_token.is_empty() {
        let metadata = next_token_metadata(
            &originals.config,
            &originals.text,
            &NextTokenLabels::default(),
        );
        plans.push(
            plan_single(NEXT_TOKEN_FILE, metadata, &next_token)
                .map_err(|error| error.to_string())?,
        );
    }
    Ok(plans)
}

fn describe(plans: &[FilePlan]) -> Result<(), String> {
    for plan in plans {
        let header = plan.header().len() as u64;
        let mut by_encoding: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
        for planned in &plan.tensors {
            let entry = by_encoding
                .entry(planned.tensor.encoding.name())
                .or_default();
            entry.0 += 1;
            entry.1 += planned.bytes;
        }
        say(&format!(
            "{} bytes={} header={} tensors={}",
            plan.name,
            header + plan.data_bytes(),
            header,
            plan.tensors.len()
        ))?;
        for (encoding, (count, bytes)) in by_encoding {
            say(&format!("  {encoding:7} tensors={count:5} bytes={bytes}"))?;
        }
    }
    Ok(())
}

fn plan(arguments: &[String]) -> Result<(), String> {
    let options = Options::parse(arguments)?;
    options.only_known(&[
        "originals",
        "only",
        "file-bytes",
        "name",
        "repo-url",
        "source-repo",
        "source-revision",
        "chat-template",
        "index",
        "partial",
        "headers-to",
    ])?;
    let originals = open_originals(
        Path::new(options.required("originals")?),
        options.one("chat-template")?,
        options.one("index")?,
    )?;
    let plans = plans(&originals, &options)?;
    if let Some(directory) = options.one("headers-to")? {
        let directory = Path::new(directory);
        fs::create_dir_all(directory)
            .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        for plan in &plans {
            let path = directory.join(format!("{}.header", plan.name));
            fs::write(&path, plan.header())
                .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        }
    }
    describe(&plans)
}

fn report_json(reports: &[FileReport]) -> String {
    let mut out = String::from("{\n  \"files\": [\n");
    for (index, file) in reports.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"bytes\": {}, \"sha256\": \"{}\", \"tensors\": [\n",
            file.name, file.bytes, file.sha256
        ));
        for (position, tensor) in file.tensors.iter().enumerate() {
            out.push_str(&format!(
                "      {{\"name\": \"{}\", \"encoding\": \"{}\", \"bytes\": {}, \"sha256\": \"{}\", \"relative_error\": {:.6e}}}{}\n",
                tensor.name,
                tensor.encoding,
                tensor.bytes,
                tensor.sha256,
                tensor.relative_error,
                if position + 1 == file.tensors.len() { "" } else { "," }
            ));
        }
        out.push_str(if index + 1 == reports.len() {
            "    ]}\n"
        } else {
            "    ]},\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

fn convert(arguments: &[String]) -> Result<(), String> {
    let options = Options::parse(arguments)?;
    options.only_known(&[
        "originals",
        "out",
        "only",
        "jobs",
        "file-bytes",
        "name",
        "repo-url",
        "source-repo",
        "source-revision",
        "chat-template",
        "report",
        "index",
        "partial",
    ])?;
    let directory = Path::new(options.required("out")?);
    let workers = match options.one("jobs")? {
        Some(text) => text
            .parse::<usize>()
            .ok()
            .filter(|&jobs| jobs > 0)
            .ok_or("--jobs takes a positive number")?,
        None => std::thread::available_parallelism()
            .map_or(1, |cores| cores.get().saturating_sub(2).max(1)),
    };
    let originals = open_originals(
        Path::new(options.required("originals")?),
        options.one("chat-template")?,
        options.one("index")?,
    )?;
    let plans = plans(&originals, &options)?;
    describe(&plans)?;

    let started = Instant::now();
    let progress = |done: u64, total: u64| {
        // Progress is a courtesy: a closed stdout must not stop the conversion.
        let _ignored = writeln!(
            io::stdout(),
            "written {:.1} of {:.1} GB after {:.0} s",
            done as f64 / 1e9,
            total as f64 / 1e9,
            started.elapsed().as_secs_f64()
        );
    };
    let reports = write_files(&originals.checkpoint, &plans, directory, workers, &progress)
        .map_err(|error| error.to_string())?;
    for file in &reports {
        say(&format!("{}  {}  {}", file.sha256, file.bytes, file.name))?;
    }
    say(&format!("done in {:.0} s", started.elapsed().as_secs_f64()))?;
    if let Some(path) = options.one("report")? {
        fs::write(path, report_json(&reports))
            .map_err(|error| format!("cannot write {path}: {error}"))?;
    }
    Ok(())
}

/// The tensors of a set of GGUF files, by name.
fn tensor_index<'a>(
    files: &'a [Gguf<'static>],
) -> BTreeMap<&'a [u8], (&'a Gguf<'static>, &'a Tensor)> {
    let mut index = BTreeMap::new();
    for file in files {
        for tensor in file.tensors() {
            index.insert(tensor.name.as_slice(), (file, tensor));
        }
    }
    index
}

fn open_set(paths: &[&str]) -> Result<Vec<Gguf<'static>>, String> {
    paths
        .iter()
        .map(|path| Gguf::from_path(path).map_err(|error| format!("{path}: {error}")))
        .collect()
}

#[derive(Default)]
struct Tally {
    same: u64,
    same_bytes: u64,
    different: u64,
    different_bytes: u64,
    squared_difference: f64,
    energy: f64,
}

fn diff(arguments: &[String]) -> Result<(), String> {
    let options = Options::parse(arguments)?;
    options.only_known(&["a", "b"])?;
    let (first, second) = (open_set(&options.all("a"))?, open_set(&options.all("b"))?);
    if first.is_empty() || second.is_empty() {
        return Err("diff takes --a <gguf>... --b <gguf>...".to_owned());
    }
    let (left, right) = (tensor_index(&first), tensor_index(&second));
    let text = |name: &[u8]| String::from_utf8_lossy(name).into_owned();
    for name in left.keys().filter(|name| !right.contains_key(*name)) {
        say(&format!("only in a: {}", text(name)))?;
    }
    for name in right.keys().filter(|name| !left.contains_key(*name)) {
        say(&format!("only in b: {}", text(name)))?;
    }

    const STEP_VALUES: u64 = 1 << 22;
    let mut tallies: BTreeMap<&str, Tally> = BTreeMap::new();
    let (mut a_bytes, mut b_bytes) = (Vec::new(), Vec::new());
    let (mut a_values, mut b_values) = (Vec::new(), Vec::new());
    for (name, (a_file, a)) in &left {
        let Some((b_file, b)) = right.get(name) else {
            continue;
        };
        if a.dimensions != b.dimensions || a.r#type != b.r#type {
            say(&format!(
                "shape or encoding differs: {} a={:?}/{} b={:?}/{}",
                text(name),
                a.dimensions,
                a.r#type,
                b.dimensions,
                b.r#type
            ))?;
            continue;
        }
        let encoding = Encoding::from_ggml_type(a.r#type)
            .ok_or_else(|| format!("{} uses an encoding this tool does not read", text(name)))?;
        let block = encoding.block_values() as u64;
        let step_bytes = STEP_VALUES / block * encoding.block_bytes() as u64;
        let (mut squared_difference, mut energy, mut identical) = (0.0f64, 0.0f64, true);
        let mut offset = 0u64;
        while offset < a.bytes {
            let length = step_bytes.min(a.bytes - offset);
            a_file
                .tensor_range_bytes(a, offset, length, &mut a_bytes)
                .map_err(|error| error.to_string())?;
            b_file
                .tensor_range_bytes(b, offset, length, &mut b_bytes)
                .map_err(|error| error.to_string())?;
            if a_bytes != b_bytes {
                identical = false;
                let values = (length / encoding.block_bytes() as u64 * block) as usize;
                a_values.resize(values, 0.0f32);
                b_values.resize(values, 0.0f32);
                dequantize(encoding, &a_bytes, &mut a_values).map_err(|e| e.to_string())?;
                dequantize(encoding, &b_bytes, &mut b_values).map_err(|e| e.to_string())?;
                for (x, y) in a_values.iter().zip(&b_values) {
                    let difference = f64::from(*x) - f64::from(*y);
                    squared_difference += difference * difference;
                    energy += f64::from(*x) * f64::from(*x);
                }
            }
            offset += length;
        }
        let tally = tallies.entry(encoding.name()).or_default();
        if identical {
            tally.same += 1;
            tally.same_bytes += a.bytes;
        } else {
            tally.different += 1;
            tally.different_bytes += a.bytes;
            tally.squared_difference += squared_difference;
            tally.energy += energy;
            say(&format!(
                "differs: {} {} relative_difference={:.6}",
                text(name),
                encoding.name(),
                if energy > 0.0 {
                    (squared_difference / energy).sqrt()
                } else {
                    0.0
                }
            ))?;
        }
    }
    for (encoding, tally) in &tallies {
        say(&format!(
            "{encoding:7} identical={} ({} bytes) different={} ({} bytes) relative_difference={:.6}",
            tally.same,
            tally.same_bytes,
            tally.different,
            tally.different_bytes,
            if tally.energy > 0.0 {
                (tally.squared_difference / tally.energy).sqrt()
            } else {
                0.0
            }
        ))?;
    }

    // Tokenizer records, wherever each set keeps them.
    for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"] {
        let find = |files: &[Gguf<'static>]| {
            files.iter().find_map(|file| {
                file.string_array(key)
                    .ok()
                    .map(|v| v.iter().map(|s| s.to_vec()).collect::<Vec<_>>())
            })
        };
        if let (Some(a), Some(b)) = (find(&first), find(&second)) {
            let mismatches = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            say(&format!(
                "{key}: a={} b={} mismatches={}",
                a.len(),
                b.len(),
                mismatches + a.len().abs_diff(b.len())
            ))?;
        }
    }
    let key = "tokenizer.ggml.token_type";
    let kinds =
        |files: &[Gguf<'static>]| files.iter().find_map(|file| file.unsigned_array(key).ok());
    if let (Some(a), Some(b)) = (kinds(&first), kinds(&second)) {
        let mismatches = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        say(&format!(
            "{key}: a={} b={} mismatches={}",
            a.len(),
            b.len(),
            mismatches + a.len().abs_diff(b.len())
        ))?;
    }
    Ok(())
}

fn read_f32(input: &str) -> Result<Vec<f32>, String> {
    let bytes = read(Path::new(input))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{input} is not a whole number of f32 values"));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn encode(arguments: &[String]) -> Result<(), String> {
    let (encoding, input, output, row) = match arguments {
        [encoding, input, output] => (encoding, input, output, None),
        [encoding, input, output, row] => (encoding, input, output, Some(row)),
        _ => {
            return Err("encode takes an encoding, an input file and an output file".to_owned());
        }
    };
    let encoding =
        Encoding::from_name(encoding).ok_or_else(|| format!("unknown encoding {encoding}"))?;
    let values = read_f32(input)?;
    let length = encoding
        .encoded_len(values.len() as u64)
        .map_err(|error| error.to_string())?;
    let mut encoded = vec![0u8; length as usize];
    match row {
        None => quantize(encoding, &values, &mut encoded),
        Some(row) => {
            let row = row
                .parse::<usize>()
                .map_err(|_| format!("row length {row} is not a number"))?;
            quantize_rows(encoding, Fit::LeastSquares, row, &values, &mut encoded)
        }
    }
    .map_err(|error| error.to_string())?;
    fs::write(output, encoded).map_err(|error| format!("cannot write {output}: {error}"))
}

fn error(arguments: &[String]) -> Result<(), String> {
    let [encoding, input, encoded] = arguments else {
        return Err(
            "error takes an encoding, the original f32 file and the encoded file".to_owned(),
        );
    };
    let encoding =
        Encoding::from_name(encoding).ok_or_else(|| format!("unknown encoding {encoding}"))?;
    let values = read_f32(input)?;
    let bytes = read(Path::new(encoded))?;
    let mut decoded = vec![0.0f32; values.len()];
    dequantize(encoding, &bytes, &mut decoded).map_err(|error| error.to_string())?;
    let mut squared_error = 0.0f64;
    let mut energy = 0.0f64;
    let mut worst = 0.0f64;
    for (original, restored) in values.iter().zip(&decoded) {
        let difference = f64::from(*original) - f64::from(*restored);
        squared_error += difference * difference;
        energy += f64::from(*original) * f64::from(*original);
        worst = worst.max(difference.abs());
    }
    let relative = if energy > 0.0 {
        (squared_error / energy).sqrt()
    } else {
        0.0
    };
    say(&format!(
        "{} weights={} relative_rms_error={relative:.6} worst_abs_error={worst:.6e} rms_weight={:.6e}",
        encoding.name(),
        values.len(),
        (energy / values.len().max(1) as f64).sqrt()
    ))
}

fn read_header_prefix(path: &str, len: u64) -> Result<Vec<u8>, String> {
    let length = usize::try_from(len).map_err(|_| format!("{path}: header exceeds usize"))?;
    let file = File::open(path).map_err(|error| format!("cannot open {path}: {error}"))?;
    let mut buf = vec![0u8; length];
    file.read_exact_at(&mut buf, 0)
        .map_err(|error| format!("cannot read header of {path}: {error}"))?;
    Ok(buf)
}

fn decode_gguf_metadata(
    gguf: &Gguf<'_>,
    header_bytes: &[u8],
    drop_split: bool,
) -> Result<Vec<(String, Value)>, String> {
    let mut items: Vec<(&[u8], &Metadata)> = gguf
        .metadata()
        .iter()
        .map(|(k, v)| (k.as_slice(), v))
        .collect();
    items.sort_by_key(|(_, meta)| meta.offset);
    let mut out = Vec::with_capacity(items.len());
    for (key_bytes, meta) in items {
        let key =
            std::str::from_utf8(key_bytes).map_err(|_| "non-UTF-8 GGUF metadata key".to_owned())?;
        if drop_split && matches!(key, "split.no" | "split.tensors.count" | "split.count") {
            continue;
        }
        let raw = header_bytes
            .get(meta.offset as usize..meta.end as usize)
            .ok_or_else(|| format!("metadata value span out of bounds for {key}"))?;
        let val = match meta.r#type {
            2 => Value::U16(u16::from_le_bytes(
                raw.try_into()
                    .map_err(|_| format!("invalid u16 for {key}"))?,
            )),
            4 => Value::U32(u32::from_le_bytes(
                raw.try_into()
                    .map_err(|_| format!("invalid u32 for {key}"))?,
            )),
            5 => Value::I32(i32::from_le_bytes(
                raw.try_into()
                    .map_err(|_| format!("invalid i32 for {key}"))?,
            )),
            6 => Value::F32(f32::from_bits(u32::from_le_bytes(
                raw.try_into()
                    .map_err(|_| format!("invalid f32 for {key}"))?,
            ))),
            7 => Value::Bool(
                raw.first()
                    .copied()
                    .ok_or_else(|| format!("invalid bool for {key}"))?
                    != 0,
            ),
            8 => Value::Text(
                gguf.string_value(key_bytes)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
            ),
            9 if meta.element_type == 5 => {
                let body = raw
                    .get(12..)
                    .ok_or_else(|| format!("truncated i32 array for {key}"))?;
                let values: Vec<i32> = body
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| i32::from_le_bytes(*chunk))
                    .collect();
                if values.len() as u64 != meta.elements {
                    return Err(format!("i32 array count mismatch for {key}"));
                }
                Value::I32s(values)
            }
            9 if meta.element_type == 10 => Value::U64s(
                gguf.unsigned_array(key_bytes)
                    .map_err(|error| error.to_string())?,
            ),
            9 if meta.element_type == 8 => Value::Texts(
                gguf.string_array(key_bytes)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect(),
            ),
            other => return Err(format!("unsupported metadata type {other} for {key}")),
        };
        out.push((key.to_owned(), val));
    }
    Ok(out)
}

fn apply_metadata_overrides(
    metadata: &mut Vec<(String, Value)>,
    options: &Options,
) -> Result<(), String> {
    if let Some(template_path) = options.one("chat-template")? {
        let bytes = read(Path::new(template_path))?;
        let slot = metadata
            .iter_mut()
            .find(|(key, _)| key == "tokenizer.chat_template")
            .ok_or_else(|| "source GGUF lacks tokenizer.chat_template".to_owned())?;
        slot.1 = Value::Text(bytes);
    }
    if let Some(name) = options.one("name")? {
        for (key, value) in metadata.iter_mut() {
            if key == "general.name" || key == "general.base_model.0.name" {
                *value = Value::text(name);
            }
        }
    }
    if let Some(repo_url) = options.one("repo-url")? {
        if let Some((_, value)) = metadata.iter_mut().find(|(k, _)| k == "general.repo_url") {
            *value = Value::text(repo_url);
        } else {
            let pos = metadata
                .iter()
                .position(|(k, _)| k == "general.quantization_version")
                .unwrap_or(metadata.len());
            metadata.insert(pos, ("general.repo_url".to_owned(), Value::text(repo_url)));
        }
    }
    Ok(())
}

fn write_gguf_streaming(
    output: &Path,
    metadata: &[(String, Value)],
    sources: &[(&Gguf<'_>, &Tensor)],
) -> Result<(u64, String), String> {
    if output.exists() {
        return Err(format!("{}: output file already exists", output.display()));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut records = Vec::with_capacity(sources.len());
    let mut rel_offset = 0u64;
    for (_, tensor) in sources {
        let name = std::str::from_utf8(&tensor.name)
            .map_err(|_| "non-UTF-8 tensor name".to_owned())?
            .to_owned();
        records.push(Record {
            name,
            dimensions: tensor.dimensions.clone(),
            kind: tensor.r#type,
            offset: rel_offset,
        });
        rel_offset = gguf::aligned(rel_offset + tensor.bytes);
    }
    let hdr = gguf::header(metadata, &records);
    let data_bytes = if sources.is_empty() { 0 } else { rel_offset };
    let total_bytes = hdr.len() as u64 + data_bytes;

    let file_name = output
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "invalid output file name".to_owned())?;
    let partial = parent.join(format!(".{file_name}.{}.part", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|error| format!("cannot create {}: {error}", partial.display()))?;

    let write_res = (|| -> Result<String, String> {
        let mut hasher = Sha256::new();
        file.write_all_at(&hdr, 0)
            .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
        hasher.update(&hdr);
        let mut written = hdr.len() as u64;
        let mut chunk = Vec::new();

        for ((src_gguf, tensor), rec) in sources.iter().zip(&records) {
            let start = hdr.len() as u64 + rec.offset;
            if start > written {
                let pad = vec![0u8; (start - written) as usize];
                file.write_all_at(&pad, written)
                    .map_err(|error| format!("cannot pad {}: {error}", partial.display()))?;
                hasher.update(&pad);
                written = start;
            }
            let mut sub = 0u64;
            while sub < tensor.bytes {
                let take = (8 << 20).min(tensor.bytes - sub);
                src_gguf
                    .tensor_range_bytes(tensor, sub, take, &mut chunk)
                    .map_err(|error| error.to_string())?;
                file.write_all_at(&chunk, written)
                    .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
                hasher.update(&chunk);
                written += take;
                sub += take;
            }
        }
        if total_bytes > written {
            let pad = vec![0u8; (total_bytes - written) as usize];
            file.write_all_at(&pad, written)
                .map_err(|error| format!("cannot pad {}: {error}", partial.display()))?;
            hasher.update(&pad);
        }
        file.sync_all()
            .map_err(|error| format!("cannot sync {}: {error}", partial.display()))?;
        fs::hard_link(&partial, output)
            .map_err(|error| format!("cannot publish {}: {error}", output.display()))?;
        Ok(hex(&hasher.finalize()))
    })();

    drop(file);
    let _ignored = fs::remove_file(&partial);
    let digest = write_res?;
    Ok((total_bytes, digest))
}

fn extract_base(arguments: &[String]) -> Result<(), String> {
    let options = Options::parse(arguments)?;
    options.only_known(&["from", "out", "chat-template", "repo-url", "name"])?;
    let from_paths = options.all("from");
    if from_paths.is_empty() {
        return Err("--from requires at least one GGUF file".to_owned());
    }
    let out_path = Path::new(options.required("out")?);
    let shards = open_set(&from_paths)?;

    let mut meta_shard: Option<(&str, &Gguf<'static>)> = None;
    for (path, shard) in from_paths.iter().copied().zip(&shards) {
        if shard.find("qwen4exp.block_count").is_some() {
            if meta_shard.is_some() {
                return Err("multiple --from GGUF files carry model metadata".to_owned());
            }
            meta_shard = Some((path, shard));
        }
    }
    let (meta_path, meta_gguf) =
        meta_shard.ok_or_else(|| "--from GGUF set lacks qwen4exp model metadata".to_owned())?;
    let header_bytes = read_header_prefix(meta_path, meta_gguf.data_offset())?;
    let mut metadata = decode_gguf_metadata(meta_gguf, &header_bytes, true)?;
    apply_metadata_overrides(&mut metadata, &options)?;

    let index = tensor_index(&shards);
    let mut selected = Vec::with_capacity(BASE_TENSORS.len());
    for name in BASE_TENSORS {
        let entry = index
            .get(name.as_bytes())
            .copied()
            .ok_or_else(|| format!("--from GGUF set lacks {name}"))?;
        selected.push(entry);
    }

    let started = Instant::now();
    let (bytes, sha256) = write_gguf_streaming(out_path, &metadata, &selected)?;
    let short = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(BASE_FILE);
    say(&format!("{sha256}  {bytes}  {short}"))?;
    say(&format!("done in {:.1} s", started.elapsed().as_secs_f64()))
}

fn rewrite_header(arguments: &[String]) -> Result<(), String> {
    let options = Options::parse(arguments)?;
    options.only_known(&["from", "out", "chat-template", "repo-url", "name"])?;
    let from_path = options.required("from")?;
    let out_path = Path::new(options.required("out")?);
    let mut shards = open_set(&[from_path])?;
    let gguf = shards
        .pop()
        .ok_or_else(|| "missing --from GGUF".to_owned())?;
    let header_bytes = read_header_prefix(from_path, gguf.data_offset())?;
    let mut metadata = decode_gguf_metadata(&gguf, &header_bytes, false)?;
    apply_metadata_overrides(&mut metadata, &options)?;
    let selected: Vec<(&Gguf<'_>, &Tensor)> = gguf.tensors().iter().map(|t| (&gguf, t)).collect();
    let (bytes, sha256) = write_gguf_streaming(out_path, &metadata, &selected)?;
    let short = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.gguf");
    say(&format!("{sha256}  {bytes}  {short}"))
}

fn run(arguments: &[String]) -> Result<(), String> {
    match arguments.split_first() {
        Some((command, rest)) if command == "plan" => plan(rest),
        Some((command, rest)) if command == "convert" => convert(rest),
        Some((command, rest)) if command == "extract-base" => extract_base(rest),
        Some((command, rest)) if command == "rewrite-header" => rewrite_header(rest),
        Some((command, rest)) if command == "diff" => diff(rest),
        Some((command, rest)) if command == "encode" => encode(rest),
        Some((command, rest)) if command == "error" => error(rest),
        _ => Err(USAGE.to_owned()),
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            // A closed stderr leaves nothing to report to.
            let _ignored = writeln!(io::stderr(), "{}", message.trim_end());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(1);

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tokspur-quant-main-test-{label}-{}-{}",
            std::process::id(),
            TEST_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ignored = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_raw_gguf(
        path: &Path,
        metadata: &[(String, Value)],
        tensors: &[(&str, Vec<u64>, u32, Vec<u8>)],
    ) {
        let mut records = Vec::new();
        let mut rel = 0u64;
        for (name, dims, kind, bytes) in tensors {
            records.push(Record {
                name: (*name).to_owned(),
                dimensions: dims.clone(),
                kind: *kind,
                offset: rel,
            });
            rel = gguf::aligned(rel + bytes.len() as u64);
        }
        let hdr = gguf::header(metadata, &records);
        let data_bytes = if tensors.is_empty() { 0 } else { rel };
        let mut image = vec![0u8; hdr.len() + data_bytes as usize];
        image[..hdr.len()].copy_from_slice(&hdr);
        for ((_, _, _, bytes), rec) in tensors.iter().zip(&records) {
            let start = hdr.len() + rec.offset as usize;
            image[start..start + bytes.len()].copy_from_slice(bytes);
        }
        fs::write(path, image).unwrap();
    }

    #[test]
    fn extract_base_is_byte_identical_to_direct_base_and_rewrite_header_updates_metadata() {
        let dir = temp_dir("extract-base");
        let base_meta = vec![
            ("general.architecture".to_owned(), Value::text("qwen4exp")),
            ("general.name".to_owned(), Value::text("Qwen3.8 Flash Next")),
            (
                "general.base_model.0.name".to_owned(),
                Value::text("Qwen3.8 Flash Next"),
            ),
            ("general.quantization_version".to_owned(), Value::U32(2)),
            (
                "tokspur.source.revision".to_owned(),
                Value::text("34567a4712bc9766c4449e2e98e4468bfa24d915"),
            ),
            ("qwen4exp.block_count".to_owned(), Value::U32(48)),
            (
                "qwen4exp.attention.layer_norm_rms_epsilon".to_owned(),
                Value::F32(1e-6),
            ),
            (
                "tokenizer.ggml.add_bos_token".to_owned(),
                Value::Bool(false),
            ),
            (
                "tokenizer.ggml.token_type".to_owned(),
                Value::I32s(vec![1, 3]),
            ),
            (
                "qwen4exp.ngram_vocab_sizes".to_owned(),
                Value::U64s(vec![3_000_003, 3_000_017]),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                Value::Texts(vec![b"hello".to_vec(), b"<|endoftext|>".to_vec()]),
            ),
            (
                "tokenizer.chat_template".to_owned(),
                Value::Text(b"{{ messages[0].content }}".to_vec()),
            ),
        ];

        let mut shard0_meta = vec![
            ("split.no".to_owned(), Value::U16(0)),
            ("split.tensors.count".to_owned(), Value::I32(3)),
            ("split.count".to_owned(), Value::U16(2)),
        ];
        shard0_meta.extend(base_meta.clone());

        let shard1_meta = vec![
            ("split.no".to_owned(), Value::U16(1)),
            ("split.tensors.count".to_owned(), Value::I32(3)),
            ("split.count".to_owned(), Value::U16(2)),
        ];

        // Non-32-byte-aligned tensor sizes exercise inter-tensor and trailing zero padding.
        let ple_bytes: Vec<u8> = (0..54u8).collect();
        let tok_bytes: Vec<u8> = (100..168u8).collect();
        let gpu_bytes: Vec<u8> = (0..68u8).collect();

        let shard0_path = dir.join("shard-00001-of-00002.gguf");
        let shard1_path = dir.join("shard-00002-of-00002.gguf");
        let direct_base_path = dir.join("direct-base.gguf");
        let extracted_base_path = dir.join("extracted-base.gguf");
        let rewritten_base_path = dir.join("rewritten-base.gguf");
        let new_template_path = dir.join("new-template.jinja");
        fs::write(&new_template_path, b"custom-jinja-template\n").unwrap();

        write_raw_gguf(&shard0_path, &shard0_meta, &[]);
        write_raw_gguf(
            &shard1_path,
            &shard1_meta,
            &[
                ("blk.0.attn_q.weight", vec![32, 2], 8, gpu_bytes),
                (
                    "per_layer_token_embd.weight",
                    vec![32, 3],
                    20,
                    ple_bytes.clone(),
                ),
                ("token_embd.weight", vec![32, 2], 8, tok_bytes.clone()),
            ],
        );
        write_raw_gguf(
            &direct_base_path,
            &base_meta,
            &[
                (
                    "per_layer_token_embd.weight",
                    vec![32, 3],
                    20,
                    ple_bytes.clone(),
                ),
                ("token_embd.weight", vec![32, 2], 8, tok_bytes.clone()),
            ],
        );

        let args: Vec<String> = [
            "extract-base",
            "--from",
            shard0_path.to_str().unwrap(),
            shard1_path.to_str().unwrap(),
            "--out",
            extracted_base_path.to_str().unwrap(),
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        run(&args).unwrap();
        assert_eq!(
            fs::read(&extracted_base_path).unwrap(),
            fs::read(&direct_base_path).unwrap()
        );

        let rw_args: Vec<String> = [
            "rewrite-header",
            "--from",
            extracted_base_path.to_str().unwrap(),
            "--out",
            rewritten_base_path.to_str().unwrap(),
            "--chat-template",
            new_template_path.to_str().unwrap(),
            "--repo-url",
            "https://huggingface.co/tokspur/Qwen3.8-Flash-Next-GGUF",
            "--name",
            "Tokspur Qwen3.8 Flash Next",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        run(&rw_args).unwrap();

        let reopened = Gguf::from_path(&rewritten_base_path).unwrap();
        assert_eq!(
            reopened.string_value("tokenizer.chat_template").unwrap(),
            b"custom-jinja-template\n"
        );
        assert_eq!(
            reopened.string_value("general.repo_url").unwrap(),
            b"https://huggingface.co/tokspur/Qwen3.8-Flash-Next-GGUF"
        );
        assert_eq!(
            reopened.string_value("general.name").unwrap(),
            b"Tokspur Qwen3.8 Flash Next"
        );
        assert_eq!(
            reopened.string_value("general.base_model.0.name").unwrap(),
            b"Tokspur Qwen3.8 Flash Next"
        );
        let mut buf = Vec::new();
        reopened
            .tensor_range_bytes(&reopened.tensors()[0], 0, ple_bytes.len() as u64, &mut buf)
            .unwrap();
        assert_eq!(buf, ple_bytes);
        reopened
            .tensor_range_bytes(&reopened.tensors()[1], 0, tok_bytes.len() as u64, &mut buf)
            .unwrap();
        assert_eq!(buf, tok_bytes);

        let _ignored = fs::remove_dir_all(&dir);
    }
}

//! The checkpoint's `config.json`, and the GGUF metadata the converted files carry.

use std::fmt;

use tokspur_json::{Json, JsonError, parse_json};

use crate::gguf::Value;
use crate::layout::Geometry;
use crate::tokenizer::Vocabulary;

/// GGUF architecture name of the model.
pub const ARCHITECTURE: &str = "qwen4exp";

/// What `config.json` says about the text model.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub geometry: Geometry,
    pub context_length: u32,
    pub experts_per_token: u32,
    pub full_attention_interval: u32,
    pub rope_base: f32,
    pub rope_sections: Vec<i32>,
    pub rope_dimensions: u32,
    pub norm_epsilon: f32,
    pub indexer_top_k: u32,
    pub indexer_compress_ratio: i32,
    pub ngram_size: u32,
    pub ngram_heads: u32,
    pub beginning_token: u32,
    /// The token the n-gram hash restarts on: the configuration's end of text.
    pub ngram_reset_token: u32,
    pub image_token: u32,
}

/// Why `config.json` could not be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    Json(JsonError),
    Format(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => error.fmt(formatter),
            Self::Format(reason) => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<JsonError> for ConfigError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

fn field<'a>(value: &'a Json, key: &str) -> Result<&'a Json, ConfigError> {
    value
        .find(key)?
        .filter(|found| !found.is_null())
        .ok_or(ConfigError::Format("config.json lacks a required field"))
}

fn unsigned(value: &Json, key: &str) -> Result<u64, ConfigError> {
    Ok(field(value, key)?.unsigned_integer()?)
}

fn small(value: &Json, key: &str) -> Result<u32, ConfigError> {
    u32::try_from(unsigned(value, key)?)
        .map_err(|_| ConfigError::Format("config.json value is out of range"))
}

fn real(value: &Json, key: &str) -> Result<f64, ConfigError> {
    field(value, key)?
        .number_lexeme()?
        .parse::<f64>()
        .map_err(|_| ConfigError::Format("config.json number is malformed"))
}

impl Config {
    /// Reads the text model's configuration.
    pub fn parse(config_json: &[u8]) -> Result<Self, ConfigError> {
        let json = parse_json(config_json)?;
        let text = field(&json, "text_config")?;
        let layers = unsigned(text, "num_hidden_layers")? as usize;
        let mut linear_layers = Vec::with_capacity(layers);
        for kind in field(text, "layer_types")?.array()? {
            linear_layers.push(match kind.string()? {
                b"linear_attention" => true,
                b"full_attention" => false,
                _ => return Err(ConfigError::Format("unknown layer type")),
            });
        }
        if linear_layers.len() != layers {
            return Err(ConfigError::Format("layer types do not cover every layer"));
        }
        // The configuration numbers these layers from one.
        let mut ngram_layers = Vec::new();
        for layer in field(text, "ple_layer_ids")?.array()? {
            let layer = layer.unsigned_integer()?;
            if layer == 0 {
                return Err(ConfigError::Format("n-gram layer ids start at one"));
            }
            ngram_layers.push(layer as usize - 1);
        }

        let rope = field(text, "rope_parameters")?;
        let mut rope_sections = Vec::new();
        for section in field(rope, "mrope_section")?.array()? {
            rope_sections.push(
                i32::try_from(section.unsigned_integer()?)
                    .map_err(|_| ConfigError::Format("rope section is out of range"))?,
            );
        }
        // GGUF always stores four sections.
        rope_sections.resize(4, 0);
        let head_width = unsigned(text, "head_dim")?;
        let rope_dimensions = (head_width as f64 * real(rope, "partial_rotary_factor")?) as u32;

        let geometry = Geometry {
            hidden: unsigned(text, "hidden_size")?,
            vocabulary: unsigned(text, "vocab_size")?,
            linear_layers,
            experts: unsigned(text, "num_experts")?,
            expert_width: unsigned(text, "moe_intermediate_size")?,
            shared_expert_width: unsigned(text, "shared_expert_intermediate_size")?,
            streams: unsigned(text, "hc_count")?,
            stream_rank: unsigned(text, "hc_lowrank")?,
            attention_heads: unsigned(text, "num_attention_heads")?,
            attention_key_value_heads: unsigned(text, "num_key_value_heads")?,
            attention_head_width: head_width,
            indexer_heads: unsigned(text, "indexer_n_heads")?,
            indexer_key_value_heads: unsigned(text, "indexer_kv_heads")?,
            indexer_head_width: unsigned(text, "indexer_head_dim")?,
            linear_key_heads: unsigned(text, "linear_num_key_heads")?,
            linear_value_heads: unsigned(text, "linear_num_value_heads")?,
            linear_key_width: unsigned(text, "linear_key_head_dim")?,
            linear_value_width: unsigned(text, "linear_value_head_dim")?,
            convolution_kernel: unsigned(text, "linear_conv_kernel_dim")?,
            ngram_layers,
            ngram_parts: unsigned(text, "split_ngram_parts")?,
            ngram_convolution_kernel: unsigned(text, "ple_conv_kernel_size")?,
            next_token_layers: unsigned(text, "mtp_num_hidden_layers")?,
        };
        Ok(Self {
            geometry,
            context_length: small(text, "max_position_embeddings")?,
            experts_per_token: small(text, "num_experts_per_tok")?,
            full_attention_interval: small(text, "full_attention_interval")?,
            rope_base: real(rope, "rope_theta")? as f32,
            rope_sections,
            rope_dimensions,
            norm_epsilon: real(text, "rms_norm_eps")? as f32,
            indexer_top_k: small(text, "indexer_budget")?,
            indexer_compress_ratio: small(text, "indexer_compress_ratio")? as i32,
            ngram_size: small(text, "ngram_size")?,
            ngram_heads: small(text, "heads_per_ngram")?,
            beginning_token: small(text, "bos_token_id")?,
            ngram_reset_token: small(text, "eos_token_id")?,
            image_token: small(&json, "image_token_id")?,
        })
    }

    /// The compression ratio of each trunk layer: the indexer's on full-attention layers.
    fn compress_ratios(&self) -> Vec<i32> {
        self.geometry
            .linear_layers
            .iter()
            .map(|&linear| {
                if linear {
                    0
                } else {
                    self.indexer_compress_ratio
                }
            })
            .collect()
    }
}

/// Text and sampling facts read from the checkpoint's other files.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub vocabulary: Vocabulary,
    pub chat_template: Vec<u8>,
    pub end_token: u32,
    pub padding_token: u32,
    pub add_beginning_token: bool,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
}

/// The n-gram hash constants, stored as metadata because they need all 64 bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHash {
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_sizes: Vec<u64>,
    pub row_width: u32,
}

/// Where the converted files came from, recorded in their metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub name: String,
    pub quantized_by: String,
    pub repo_url: Option<String>,
    pub source_repo: String,
    pub source_revision: String,
    pub converter: String,
}

fn entry(key: &str, value: Value) -> (String, Value) {
    (key.to_owned(), value)
}

fn architecture(key: &str, value: Value) -> (String, Value) {
    (format!("{ARCHITECTURE}.{key}"), value)
}

/// Shape records shared by the model and the next-token head, in the converter's order.
fn shape_records(config: &Config, blocks: u32, next_token: bool) -> Vec<(String, Value)> {
    let geometry = &config.geometry;
    let as_u32 = |value: u64| Value::U32(value as u32);
    let mut records = vec![
        architecture("block_count", Value::U32(blocks)),
        architecture("context_length", Value::U32(config.context_length)),
        architecture("embedding_length", as_u32(geometry.hidden)),
        architecture("attention.head_count", as_u32(geometry.attention_heads)),
        architecture(
            "attention.head_count_kv",
            as_u32(geometry.attention_key_value_heads),
        ),
        architecture(
            "rope.dimension_sections",
            Value::I32s(config.rope_sections.clone()),
        ),
        architecture("rope.freq_base", Value::F32(config.rope_base)),
        architecture(
            "attention.layer_norm_rms_epsilon",
            Value::F32(config.norm_epsilon),
        ),
        architecture("expert_count", as_u32(geometry.experts)),
        architecture("expert_used_count", Value::U32(config.experts_per_token)),
        architecture(
            "attention.key_length",
            as_u32(geometry.attention_head_width),
        ),
        architecture(
            "attention.value_length",
            as_u32(geometry.attention_head_width),
        ),
        architecture("expert_feed_forward_length", as_u32(geometry.expert_width)),
        architecture(
            "expert_shared_feed_forward_length",
            as_u32(geometry.shared_expert_width),
        ),
    ];
    if next_token {
        records.push(architecture(
            "nextn_predict_layers",
            as_u32(geometry.next_token_layers),
        ));
    }
    records.extend([
        architecture("ssm.conv_kernel", as_u32(geometry.convolution_kernel)),
        architecture("ssm.state_size", as_u32(geometry.linear_key_width)),
        architecture("ssm.group_count", as_u32(geometry.linear_key_heads)),
        architecture("ssm.time_step_rank", as_u32(geometry.linear_value_heads)),
        architecture(
            "ssm.inner_size",
            as_u32(geometry.linear_value_heads * geometry.linear_value_width),
        ),
        architecture(
            "full_attention_interval",
            Value::U32(config.full_attention_interval),
        ),
        architecture("rope.dimension_count", Value::U32(config.rope_dimensions)),
        architecture("hyper_connection.count", as_u32(geometry.streams)),
        architecture("hyper_connection.low_rank", as_u32(geometry.stream_rank)),
        architecture(
            "attention.indexer.head_count",
            as_u32(geometry.indexer_heads),
        ),
        architecture(
            "attention.indexer.key_length",
            as_u32(geometry.indexer_head_width),
        ),
        architecture("attention.indexer.top_k", Value::U32(config.indexer_top_k)),
    ]);
    records
}

fn tokenizer_records(config: &Config, text: &Text) -> Vec<(String, Value)> {
    vec![
        entry("tokenizer.ggml.model", Value::text("gpt2")),
        entry("tokenizer.ggml.pre", Value::text("qwen35")),
        entry(
            "tokenizer.ggml.tokens",
            Value::Texts(text.vocabulary.tokens.clone()),
        ),
        entry(
            "tokenizer.ggml.token_type",
            Value::I32s(text.vocabulary.kinds.clone()),
        ),
        entry(
            "tokenizer.ggml.merges",
            Value::Texts(text.vocabulary.merges.clone()),
        ),
        entry("tokenizer.ggml.eos_token_id", Value::U32(text.end_token)),
        entry(
            "tokenizer.ggml.padding_token_id",
            Value::U32(text.padding_token),
        ),
        entry(
            "tokenizer.ggml.bos_token_id",
            Value::U32(config.beginning_token),
        ),
        entry(
            "tokenizer.ggml.add_bos_token",
            Value::Bool(text.add_beginning_token),
        ),
        entry(
            "tokenizer.chat_template",
            Value::Text(text.chat_template.clone()),
        ),
    ]
}

fn sampling_records(text: &Text) -> [(String, Value); 3] {
    [
        entry("general.sampling.top_k", Value::I32(text.top_k)),
        entry("general.sampling.top_p", Value::F32(text.top_p)),
        entry("general.sampling.temp", Value::F32(text.temperature)),
    ]
}

/// Metadata of the model's first file. Split records are added by the file planner.
pub fn model_metadata(
    config: &Config,
    text: &Text,
    ngram: &NgramHash,
    provenance: &Provenance,
) -> Vec<(String, Value)> {
    let geometry = &config.geometry;
    let mut records = vec![
        entry("general.architecture", Value::text(ARCHITECTURE)),
        entry("general.type", Value::text("model")),
    ];
    records.extend(sampling_records(text));
    records.push(entry("general.name", Value::text(&provenance.name)));
    records.extend(shape_records(
        config,
        geometry.linear_layers.len() as u32,
        false,
    ));
    records.extend([
        architecture(
            "attention.compress_ratios",
            Value::I32s(config.compress_ratios()),
        ),
        architecture(
            "ple.layers",
            Value::I32s(geometry.ngram_layers.iter().map(|&l| l as i32).collect()),
        ),
        architecture("ple.ngram_size", Value::U32(config.ngram_size)),
        architecture("ple.heads_per_ngram", Value::U32(config.ngram_heads)),
        architecture(
            "ple.conv_kernel",
            Value::U32(geometry.ngram_convolution_kernel as u32),
        ),
        architecture("ple.eos_token_id", Value::U32(config.ngram_reset_token)),
        architecture(
            "embedding_length_per_layer_input",
            Value::U32(ngram.row_width),
        ),
        architecture(
            "ple.layer_multipliers",
            Value::U64s(ngram.multipliers.clone()),
        ),
        architecture("ple.head_offsets", Value::U64s(ngram.head_offsets.clone())),
        architecture(
            "ple.head_vocab_sizes",
            Value::U64s(ngram.head_sizes.clone()),
        ),
    ]);
    records.extend(tokenizer_records(config, text));
    records.push(architecture(
        "ple.image_token_id",
        Value::U32(config.image_token),
    ));

    let source_url = format!("https://huggingface.co/{}", provenance.source_repo);
    records.extend([
        entry(
            "general.quantized_by",
            Value::text(&provenance.quantized_by),
        ),
        entry("general.license", Value::text("other")),
        entry("general.license.name", Value::text("qwen-community-1.0")),
        entry(
            "general.license.link",
            Value::text(&format!(
                "{source_url}/blob/{}/LICENSE",
                provenance.source_revision
            )),
        ),
        entry("general.base_model.count", Value::U32(1)),
        entry("general.base_model.0.name", Value::text(&provenance.name)),
        entry("general.base_model.0.repo_url", Value::text(&source_url)),
        entry(
            "tokspur.source.revision",
            Value::text(&provenance.source_revision),
        ),
        entry("tokspur.converter", Value::text(&provenance.converter)),
    ]);
    if let Some(url) = &provenance.repo_url {
        records.push(entry("general.repo_url", Value::text(url)));
    }
    records.extend([
        entry("general.quantization_version", Value::U32(2)),
        // llama.cpp's "mostly Q4_K, medium" file type.
        entry("general.file_type", Value::U32(15)),
    ]);
    records
}

/// Labels of the next-token file. The defaults are the ones the runtime's pinned file carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextTokenLabels {
    pub name: String,
    pub size_label: String,
    pub license_link: String,
    pub tags: Vec<String>,
}

impl Default for NextTokenLabels {
    fn default() -> Self {
        Self {
            name: "Ckpt_Q38".to_owned(),
            size_label: "512x95M".to_owned(),
            license_link: "LICENSE".to_owned(),
            tags: vec!["image-text-to-text".to_owned()],
        }
    }
}

/// Metadata of the next-token (MTP) file, which shares the model's embeddings and output head.
pub fn next_token_metadata(
    config: &Config,
    text: &Text,
    labels: &NextTokenLabels,
) -> Vec<(String, Value)> {
    let geometry = &config.geometry;
    let blocks = geometry.linear_layers.len() as u64 + geometry.next_token_layers;
    let mut records = vec![
        entry("general.architecture", Value::text(ARCHITECTURE)),
        architecture("nextn_shared_target_tensors", Value::Bool(true)),
        entry("general.type", Value::text("model")),
    ];
    records.extend(sampling_records(text));
    records.extend([
        entry("general.name", Value::text(&labels.name)),
        entry("general.size_label", Value::text(&labels.size_label)),
        entry("general.license", Value::text("other")),
        entry("general.license.name", Value::text("qwen-community-1.0")),
        entry("general.license.link", Value::text(&labels.license_link)),
        entry(
            "general.tags",
            Value::Texts(labels.tags.iter().map(|t| t.as_bytes().to_vec()).collect()),
        ),
    ]);
    records.extend(shape_records(config, blocks as u32, true));
    // The runtime's pinned file declares no compression for the head's own block.
    let mut ratios = config.compress_ratios();
    ratios.resize(blocks as usize, 0);
    records.push(architecture(
        "attention.compress_ratios",
        Value::I32s(ratios),
    ));
    records.extend(tokenizer_records(config, text));
    records.extend([
        entry("general.quantization_version", Value::U32(2)),
        // llama.cpp's "mostly Q8_0" file type.
        entry("general.file_type", Value::U32(7)),
    ]);
    records
}

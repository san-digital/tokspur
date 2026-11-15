//! The layout against the real checkpoint's tensor index and the directory the runtime loads.
//!
//! `flash-next-sources.json` lists the name, element type and shape of every text-model tensor in `Qwen/Qwen3.8-Flash-Next`. `flash-next-directory.json` lists the tensor directory, data offset and size of the four model files and the next-token file the runtime pins today. A conversion must reproduce that directory exactly: same names, order, shapes, encodings and offsets.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use tokspur_json::{Json, JsonLimits, parse_json_limited};
use tokspur_quant::Encoding;
use tokspur_quant::convert::{FilePlan, plan_single, plan_split};
use tokspur_quant::gguf::Value;
use tokspur_quant::layout::{Layout, Recipe, SourceIndex, layout};
use tokspur_quant::model::Config;
use tokspur_quant::safetensors::Dtype;

const SOURCES: &str = include_str!("fixtures/flash-next-sources.json");
const DIRECTORY: &str = include_str!("fixtures/flash-next-directory.json");

/// The fields of the checkpoint's configuration the converter reads.
const CONFIG: &str = r#"{
    "image_token_id": 248056,
    "text_config": {
        "bos_token_id": 248044, "eos_token_id": 248044, "full_attention_interval": 4,
        "hc_count": 4, "hc_lowrank": 320, "head_dim": 256, "heads_per_ngram": 8,
        "hidden_size": 2560, "indexer_budget": 2048, "indexer_compress_ratio": 4,
        "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
        "layer_types": [LAYERS],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 48, "linear_value_head_dim": 128,
        "max_position_embeddings": 262144, "moe_intermediate_size": 640,
        "mtp_num_hidden_layers": 1, "ngram_size": 3, "num_attention_heads": 24,
        "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 48,
        "num_key_value_heads": 2, "ple_conv_kernel_size": 4, "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000},
        "shared_expert_intermediate_size": 640, "split_ngram_parts": 128, "vocab_size": 248320
    }
}"#;

fn config() -> Config {
    let layers: Vec<&str> = (0..48)
        .map(|layer| {
            if layer % 4 == 3 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            }
        })
        .collect();
    Config::parse(CONFIG.replace("LAYERS", &layers.join(", ")).as_bytes()).unwrap()
}

fn parse(text: &str) -> Json {
    parse_json_limited(text.as_bytes(), JsonLimits::new(text.len(), 1_000_000, 8)).unwrap()
}

fn sources() -> SourceIndex {
    parse(SOURCES)
        .object()
        .unwrap()
        .iter()
        .map(|(name, entry)| {
            let dtype = match entry.find("dtype").unwrap().unwrap().string().unwrap() {
                b"BF16" => Dtype::Bf16,
                b"I64" => Dtype::I64,
                other => panic!("unexpected dtype {other:?}"),
            };
            let shape = entry
                .find("shape")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .iter()
                .map(|dimension| dimension.unsigned_integer().unwrap())
                .collect();
            (String::from_utf8(name.clone()).unwrap(), (dtype, shape))
        })
        .collect()
}

struct Expected {
    data_offset: u64,
    size: u64,
    /// Name, dimensions, GGUF type and offset of each tensor.
    tensors: Vec<(String, Vec<u64>, u32, u64)>,
}

fn directory() -> Vec<Expected> {
    let json = parse(DIRECTORY);
    let number = |value: &Json| value.unsigned_integer().unwrap();
    json.find("files")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()
        .iter()
        .map(|file| Expected {
            data_offset: number(file.find("data_offset").unwrap().unwrap()),
            size: number(file.find("size").unwrap().unwrap()),
            tensors: file
                .find("tensors")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .iter()
                .map(|tensor| {
                    let fields = tensor.array().unwrap();
                    (
                        String::from_utf8(fields[0].string().unwrap().to_vec()).unwrap(),
                        fields[1].array().unwrap().iter().map(number).collect(),
                        number(&fields[2]) as u32,
                        number(&fields[3]),
                    )
                })
                .collect(),
        })
        .collect()
}

fn real_layout() -> Layout {
    layout(&config().geometry, &sources()).unwrap()
}

fn assert_matches(plan: &FilePlan, expected: &Expected) {
    assert_eq!(plan.tensors.len(), expected.tensors.len(), "{}", plan.name);
    for (planned, (name, dimensions, kind, offset)) in plan.tensors.iter().zip(&expected.tensors) {
        assert_eq!(&planned.tensor.name, name);
        assert_eq!(&planned.tensor.dimensions, dimensions, "{name}");
        assert_eq!(planned.tensor.encoding.ggml_type(), *kind, "{name}");
        assert_eq!(planned.offset, *offset, "{name}");
    }
}

#[test]
fn the_configuration_parses_to_the_runtime_geometry() {
    let config = config();
    assert_eq!(config.geometry.linear_layers.len(), 48);
    assert_eq!(config.geometry.ngram_layers, vec![1]);
    assert_eq!(config.rope_sections, vec![11, 11, 10, 0]);
    assert_eq!(config.rope_dimensions, 64);
    assert_eq!(config.rope_base, 1.0e7);
    assert_eq!(f64::from(config.norm_epsilon), 9.999_999_974_752_427e-7);
    assert_eq!(config.context_length, 262_144);
}

#[test]
fn the_model_splits_into_the_files_the_runtime_pins() {
    let expected = directory();
    let layout = real_layout();
    assert_eq!(layout.target.len(), 1224);
    let name = |number: usize, count: usize| format!("model-{number}-of-{count}.gguf");
    let plans = plan_split(&name, Vec::new(), &layout.target, 50_000_000_000).unwrap();
    assert_eq!(plans.len(), 4);
    assert_eq!(plans[3].name, "model-4-of-4.gguf");
    assert!(plans[0].tensors.is_empty());
    assert_eq!(
        plans[0].metadata,
        vec![
            ("split.no".to_owned(), Value::U16(0)),
            ("split.tensors.count".to_owned(), Value::I32(1224)),
            ("split.count".to_owned(), Value::U16(4)),
        ]
    );
    for (plan, expected) in plans.iter().zip(&expected).skip(1) {
        assert_matches(plan, expected);
        // These files carry only the split records, so the whole layout must agree.
        let header = plan.header().len() as u64;
        assert_eq!(header, expected.data_offset, "{}", plan.name);
        assert_eq!(header + plan.data_bytes(), expected.size, "{}", plan.name);
    }
}

#[test]
fn the_next_token_head_has_the_pinned_directory() {
    let expected = &directory()[4];
    let layout = real_layout();
    assert_eq!(layout.next_token.len(), 32);
    let plan = plan_single("head.gguf", Vec::new(), &layout.next_token).unwrap();
    assert_matches(&plan, expected);
    assert_eq!(expected.data_offset + plan.data_bytes(), expected.size);
}

#[test]
fn every_recipe_covers_its_tensor_exactly_once() {
    let sources = sources();
    let layout = real_layout();
    let elements = |name: &str| sources[name].1.iter().product::<u64>();
    let mut used = BTreeSet::new();
    for tensor in layout.target.iter().chain(&layout.next_token) {
        let width = tensor.row_values();
        match &tensor.recipe {
            Recipe::Rows(runs) => {
                assert_eq!(runs.iter().map(|run| run.rows).sum::<u64>(), tensor.rows());
                let mut rows = BTreeSet::new();
                for run in runs {
                    assert!((run.first_row + run.rows) * width <= elements(&run.source));
                    for row in run.first_row..run.first_row + run.rows.min(4096) {
                        assert!(rows.insert((run.source.as_str(), row)), "{}", tensor.name);
                    }
                    used.insert(run.source.clone());
                }
            }
            Recipe::Wide { left, right } => {
                assert_eq!(elements(left) + elements(right), tensor.values());
                used.extend([left.clone(), right.clone()]);
            }
            Recipe::Columns { source, columns } => {
                assert_eq!(elements(source), tensor.values());
                let unique: BTreeSet<u32> = columns.iter().copied().collect();
                assert_eq!(unique.len() as u64, width, "{}", tensor.name);
                assert_eq!(unique.last().copied(), Some(width as u32 - 1));
                used.insert(source.clone());
            }
        }
    }
    // Everything in the checkpoint's text model is used, except the three hash constants that become metadata.
    let unused: Vec<&String> = sources
        .keys()
        .filter(|name| !used.contains(*name))
        .collect();
    assert_eq!(unused.len(), 3, "{unused:?}");
    assert!(unused.iter().all(|name| sources[*name].0 == Dtype::I64));
}

#[test]
fn the_encoding_mix_is_the_runtime_profile() {
    let layout = real_layout();
    let count = |encoding: Encoding| {
        layout
            .target
            .iter()
            .filter(|tensor| tensor.encoding == encoding)
            .count()
    };
    assert_eq!(count(Encoding::Q4K), 94);
    assert_eq!(count(Encoding::Q5K), 2);
    assert_eq!(count(Encoding::Q5_1), 43);
    assert_eq!(count(Encoding::Iq4Nl), 1);
    assert_eq!(count(Encoding::Bf16), 24);
    assert!(layout.next_token.iter().all(|tensor| matches!(
        tensor.encoding,
        Encoding::Q8_0 | Encoding::F32 | Encoding::Bf16
    )));
}

#[test]
fn a_checkpoint_missing_a_tensor_or_with_another_shape_is_refused() {
    let geometry = config().geometry;
    let mut missing = sources();
    missing.remove("lm_head.weight");
    assert!(layout(&geometry, &missing).is_err());
    let mut reshaped = sources();
    reshaped.insert(
        "model.language_model.layers.7.self_attn.k_proj.weight".to_owned(),
        (Dtype::Bf16, vec![256, 2560]),
    );
    assert!(layout(&geometry, &reshaped).is_err());
}

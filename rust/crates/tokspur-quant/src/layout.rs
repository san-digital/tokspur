//! The Flash-Next tensor layout: which original tensors make each GGUF tensor, and how.
//!
//! The mapping follows llama.cpp's converter for this architecture (`conversion/qwen4exp.py` and the Qwen bases it builds on) with the unmerged tensor names the runtime loads. Every target tensor is a sequence of rows of its innermost dimension, assembled from rows of original tensors.

use std::collections::BTreeMap;
use std::fmt;

use crate::safetensors::Dtype;
use crate::{Encoding, Fit};

/// Shape facts read from the checkpoint's `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub hidden: u64,
    pub vocabulary: u64,
    /// `true` for a linear-attention layer, `false` for a full-attention layer.
    pub linear_layers: Vec<bool>,
    pub experts: u64,
    pub expert_width: u64,
    pub shared_expert_width: u64,
    pub streams: u64,
    pub stream_rank: u64,
    pub attention_heads: u64,
    pub attention_key_value_heads: u64,
    pub attention_head_width: u64,
    pub indexer_heads: u64,
    pub indexer_key_value_heads: u64,
    pub indexer_head_width: u64,
    pub linear_key_heads: u64,
    pub linear_value_heads: u64,
    pub linear_key_width: u64,
    pub linear_value_width: u64,
    pub convolution_kernel: u64,
    /// Zero-based layers that carry the n-gram embedding.
    pub ngram_layers: Vec<usize>,
    pub ngram_parts: u64,
    pub ngram_convolution_kernel: u64,
    pub next_token_layers: u64,
}

impl Geometry {
    fn stream_width(&self) -> u64 {
        self.hidden * self.streams
    }

    fn linear_key_rows(&self) -> u64 {
        self.linear_key_heads * self.linear_key_width
    }

    fn linear_value_rows(&self) -> u64 {
        self.linear_value_heads * self.linear_value_width
    }

    fn values_per_key(&self) -> u64 {
        self.linear_value_heads / self.linear_key_heads
    }
}

/// A run of consecutive rows of one original tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowRun {
    pub source: String,
    pub first_row: u64,
    pub rows: u64,
}

/// How a target tensor's rows are assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recipe {
    /// The target's rows are these runs, in order. Source rows have the target's row length.
    Rows(Vec<RowRun>),
    /// Each target row is a row of `left` followed by the same row of `right`.
    Wide { left: String, right: String },
    /// Each target row is a source row with reordered columns: column `c` takes source column `columns[c]`.
    Columns { source: String, columns: Vec<u32> },
}

/// A change applied to every weight after it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueOp {
    Keep,
    /// The checkpoint stores these norm gains centred on zero.
    PlusOne,
    /// The checkpoint stores the logarithm of the negated decay.
    NegativeExponential,
}

/// One tensor of the converted model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetTensor {
    pub name: String,
    /// GGUF dimension order: innermost first.
    pub dimensions: Vec<u64>,
    pub encoding: Encoding,
    pub fit: Fit,
    pub recipe: Recipe,
    pub op: ValueOp,
}

impl TargetTensor {
    /// Weights per row.
    pub fn row_values(&self) -> u64 {
        self.dimensions[0]
    }

    /// Rows in the tensor.
    pub fn rows(&self) -> u64 {
        self.dimensions[1..].iter().product()
    }

    /// Weights in the tensor.
    pub fn values(&self) -> u64 {
        self.dimensions.iter().product()
    }
}

/// The two files' tensors, each in file order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// The model itself, without the next-token head.
    pub target: Vec<TargetTensor>,
    /// The next-token (MTP) head, which shares the target's embeddings and output head.
    pub next_token: Vec<TargetTensor>,
}

/// Why the checkpoint does not fit the Flash-Next layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    Missing(String),
    Shape(String),
    Geometry(&'static str),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(name) => write!(formatter, "checkpoint has no tensor {name}"),
            Self::Shape(name) => write!(formatter, "unexpected shape or type for {name}"),
            Self::Geometry(reason) => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Name, element type and shape of every original tensor.
pub type SourceIndex = BTreeMap<String, (Dtype, Vec<u64>)>;

/// Expert tensors the pinned runtime geometry stores at a wider encoding than the rest.
///
/// The runtime checks every tensor's encoding and the model's total size, so the mix is part of its profile: the gate and up projections of this layer are `Q5_K`, not `Q4_K`.
const WIDE_GATE_UP_LAYERS: [usize; 1] = [2];
/// Layers whose expert down projection is `Q8_0`, not `Q5_1`.
const WIDE_DOWN_LAYERS: [usize; 5] = [2, 4, 30, 46, 47];

struct Builder<'a> {
    geometry: &'a Geometry,
    sources: &'a SourceIndex,
    tensors: Vec<TargetTensor>,
}

impl Builder<'_> {
    /// Checks that `source` is a bfloat16 tensor of `shape` and returns its name.
    fn source(&self, source: String, shape: &[u64]) -> Result<String, LayoutError> {
        match self.sources.get(&source) {
            None => Err(LayoutError::Missing(source)),
            Some((Dtype::Bf16, found)) if found == shape => Ok(source),
            Some(_) => Err(LayoutError::Shape(source)),
        }
    }

    fn push(
        &mut self,
        name: String,
        dimensions: &[u64],
        encoding: Encoding,
        recipe: Recipe,
        op: ValueOp,
    ) {
        self.tensors.push(TargetTensor {
            name,
            dimensions: dimensions.to_vec(),
            encoding,
            fit: Fit::Reference,
            recipe,
            op,
        });
    }

    /// A tensor copied row for row from one source of `shape` (outermost first).
    fn direct(
        &mut self,
        name: String,
        source: String,
        shape: &[u64],
        encoding: Encoding,
        op: ValueOp,
    ) -> Result<(), LayoutError> {
        let source = self.source(source, shape)?;
        // Singleton dimensions are dropped, as the converter squeezes them.
        let mut dimensions: Vec<u64> = shape.iter().copied().filter(|&d| d != 1).collect();
        dimensions.reverse();
        let rows = dimensions[1..].iter().product();
        let recipe = Recipe::Rows(vec![RowRun {
            source,
            first_row: 0,
            rows,
        }]);
        self.push(name, &dimensions, encoding, recipe, op);
        Ok(())
    }

    /// Row runs that reorder value heads from grouped by key head to tiled across key heads.
    ///
    /// Rows before `first` keep their place. After it the checkpoint holds, for each key head, its value heads together; the runtime wants each value position across all key heads together.
    fn tiled_value_rows(&self, source: &str, first: u64, head_rows: u64) -> Vec<RowRun> {
        let geometry = self.geometry;
        let mut runs = Vec::new();
        if first > 0 {
            runs.push(RowRun {
                source: source.to_owned(),
                first_row: 0,
                rows: first,
            });
        }
        for value in 0..geometry.values_per_key() {
            for key in 0..geometry.linear_key_heads {
                runs.push(RowRun {
                    source: source.to_owned(),
                    first_row: first + (key * geometry.values_per_key() + value) * head_rows,
                    rows: head_rows,
                });
            }
        }
        runs
    }

    /// The same reordering as a column permutation.
    fn tiled_value_columns(&self, head_columns: u64) -> Vec<u32> {
        let geometry = self.geometry;
        let mut columns = Vec::new();
        for value in 0..geometry.values_per_key() {
            for key in 0..geometry.linear_key_heads {
                let first = (key * geometry.values_per_key() + value) * head_columns;
                columns.extend((first..first + head_columns).map(|column| column as u32));
            }
        }
        columns
    }

    fn hyper_connection(
        &mut self,
        target: &str,
        source: &str,
        inject: Option<Encoding>,
    ) -> Result<(), LayoutError> {
        let (wide, rank, streams) = (
            self.geometry.stream_width(),
            self.geometry.stream_rank,
            self.geometry.streams,
        );
        self.direct(
            format!("{target}_down.weight"),
            format!("{source}.input_mix_weight_down.weight"),
            &[rank, wide],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;
        if let Some(encoding) = inject {
            self.direct(
                format!("{target}_inject.weight"),
                format!("{source}.block_inject_weight.weight"),
                &[streams, wide],
                encoding,
                ValueOp::Keep,
            )?;
        }
        self.direct(
            format!("{target}_norm.weight"),
            format!("{source}.hc_norm.weight"),
            &[wide],
            Encoding::F32,
            ValueOp::PlusOne,
        )?;
        self.direct(
            format!("{target}_up.weight"),
            format!("{source}.input_mix_weight_up.weight"),
            &[wide, rank],
            Encoding::Q8_0,
            ValueOp::Keep,
        )
    }

    fn feed_forward(
        &mut self,
        block: &str,
        source: &str,
        gate_up: (Encoding, Fit),
        down: (Encoding, Fit),
    ) -> Result<(), LayoutError> {
        let (hidden, experts, width, shared) = (
            self.geometry.hidden,
            self.geometry.experts,
            self.geometry.expert_width,
            self.geometry.shared_expert_width,
        );
        self.direct(
            format!("{block}ffn_down_exps.weight"),
            format!("{source}mlp.experts.down_proj"),
            &[experts, hidden, width],
            down.0,
            ValueOp::Keep,
        )?;
        if let Some(tensor) = self.tensors.last_mut() {
            tensor.fit = down.1;
        }
        self.direct(
            format!("{block}ffn_down_shexp.weight"),
            format!("{source}mlp.shared_expert.down_proj.weight"),
            &[hidden, shared],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;

        // The checkpoint stacks each expert's gate rows on top of its up rows.
        let stacked = self.source(
            format!("{source}mlp.experts.gate_up_proj"),
            &[experts, 2 * width, hidden],
        )?;
        let half = |offset: u64| {
            Recipe::Rows(
                (0..experts)
                    .map(|expert| RowRun {
                        source: stacked.clone(),
                        first_row: expert * 2 * width + offset,
                        rows: width,
                    })
                    .collect(),
            )
        };
        let dimensions = [hidden, width, experts];
        self.push(
            format!("{block}ffn_gate_exps.weight"),
            &dimensions,
            gate_up.0,
            half(0),
            ValueOp::Keep,
        );
        if let Some(tensor) = self.tensors.last_mut() {
            tensor.fit = gate_up.1;
        }
        self.direct(
            format!("{block}ffn_gate_inp.weight"),
            format!("{source}mlp.gate.weight"),
            &[experts, hidden],
            Encoding::F32,
            ValueOp::Keep,
        )?;
        self.direct(
            format!("{block}ffn_gate_inp_shexp.weight"),
            format!("{source}mlp.shared_expert_gate.weight"),
            &[1, hidden],
            Encoding::F32,
            ValueOp::Keep,
        )?;
        self.direct(
            format!("{block}ffn_gate_shexp.weight"),
            format!("{source}mlp.shared_expert.gate_proj.weight"),
            &[shared, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;
        self.push(
            format!("{block}ffn_up_exps.weight"),
            &dimensions,
            gate_up.0,
            half(width),
            ValueOp::Keep,
        );
        if let Some(tensor) = self.tensors.last_mut() {
            tensor.fit = gate_up.1;
        }
        self.direct(
            format!("{block}ffn_up_shexp.weight"),
            format!("{source}mlp.shared_expert.up_proj.weight"),
            &[shared, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )
    }

    fn full_attention(&mut self, block: &str, source: &str) -> Result<(), LayoutError> {
        let geometry = self.geometry;
        let hidden = geometry.hidden;
        let head = geometry.attention_head_width;
        let key_value = geometry.attention_key_value_heads * head;
        let output = geometry.attention_heads * head;
        self.direct(
            format!("{block}attn_k.weight"),
            format!("{source}self_attn.k_proj.weight"),
            &[key_value, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;
        self.direct(
            format!("{block}attn_k_norm.weight"),
            format!("{source}self_attn.k_norm.weight"),
            &[head],
            Encoding::F32,
            ValueOp::PlusOne,
        )?;
        self.direct(
            format!("{block}attn_output.weight"),
            format!("{source}self_attn.o_proj.weight"),
            &[hidden, output],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;
        self.direct(
            format!("{block}attn_q.weight"),
            format!("{source}self_attn.q_proj.weight"),
            &[2 * output, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;
        self.direct(
            format!("{block}attn_q_norm.weight"),
            format!("{source}self_attn.q_norm.weight"),
            &[head],
            Encoding::F32,
            ValueOp::PlusOne,
        )?;
        self.direct(
            format!("{block}attn_v.weight"),
            format!("{source}self_attn.v_proj.weight"),
            &[key_value, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )
    }

    fn indexer(&mut self, block: &str, source: &str) -> Result<(), LayoutError> {
        let geometry = self.geometry;
        let hidden = geometry.hidden;
        let width = geometry.indexer_head_width;
        let query = geometry.indexer_heads * width;
        let key = geometry.indexer_key_value_heads * width;
        // One projection feeds both; its first rows are the query's.
        let joint = self.source(
            format!("{source}self_attn.indexer.index_qk_proj.weight"),
            &[query + key, hidden],
        )?;
        let rows = |first_row, rows| {
            Recipe::Rows(vec![RowRun {
                source: joint.clone(),
                first_row,
                rows,
            }])
        };
        self.direct(
            format!("{block}indexer.k_norm.weight"),
            format!("{source}self_attn.indexer.k_layernorm.weight"),
            &[width],
            Encoding::F32,
            ValueOp::PlusOne,
        )?;
        self.push(
            format!("{block}indexer.k_proj.weight"),
            &[hidden, key],
            Encoding::Bf16,
            rows(query, key),
            ValueOp::Keep,
        );
        self.direct(
            format!("{block}indexer.q_norm.weight"),
            format!("{source}self_attn.indexer.q_layernorm.weight"),
            &[width],
            Encoding::F32,
            ValueOp::PlusOne,
        )?;
        self.push(
            format!("{block}indexer.q_proj.weight"),
            &[hidden, query],
            Encoding::Bf16,
            rows(0, query),
            ValueOp::Keep,
        );
        Ok(())
    }

    fn linear_attention_head(&mut self, block: &str, source: &str) -> Result<(), LayoutError> {
        let geometry = self.geometry;
        let hidden = geometry.hidden;
        let keys = geometry.linear_key_rows();
        let values = geometry.linear_value_rows();
        let width = geometry.linear_value_width;
        let gate = self.source(
            format!("{source}linear_attn.in_proj_z.weight"),
            &[values, hidden],
        )?;
        let recipe = Recipe::Rows(self.tiled_value_rows(&gate, 0, width));
        self.push(
            format!("{block}attn_gate.weight"),
            &[hidden, values],
            Encoding::Q8_0,
            recipe,
            ValueOp::Keep,
        );
        let joint = self.source(
            format!("{source}linear_attn.in_proj_qkv.weight"),
            &[2 * keys + values, hidden],
        )?;
        let recipe = Recipe::Rows(self.tiled_value_rows(&joint, 2 * keys, width));
        self.push(
            format!("{block}attn_qkv.weight"),
            &[hidden, 2 * keys + values],
            Encoding::Q8_0,
            recipe,
            ValueOp::Keep,
        );
        Ok(())
    }

    fn linear_attention_state(&mut self, block: &str, source: &str) -> Result<(), LayoutError> {
        let geometry = self.geometry;
        let hidden = geometry.hidden;
        let heads = geometry.linear_value_heads;
        let keys = geometry.linear_key_rows();
        let values = geometry.linear_value_rows();
        let width = geometry.linear_value_width;
        let kernel = geometry.convolution_kernel;

        let decay = self.source(format!("{source}linear_attn.A_log"), &[heads])?;
        let columns = self.tiled_value_columns(1);
        self.push(
            format!("{block}ssm_a"),
            &[heads],
            Encoding::F32,
            Recipe::Columns {
                source: decay,
                columns: columns.clone(),
            },
            ValueOp::NegativeExponential,
        );
        for (target, name) in [("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b")] {
            let projection = self.source(
                format!("{source}linear_attn.{name}.weight"),
                &[heads, hidden],
            )?;
            let recipe = Recipe::Rows(self.tiled_value_rows(&projection, 0, 1));
            self.push(
                format!("{block}{target}.weight"),
                &[hidden, heads],
                Encoding::F32,
                recipe,
                ValueOp::Keep,
            );
        }
        let convolution = self.source(
            format!("{source}linear_attn.conv1d.weight"),
            &[2 * keys + values, 1, kernel],
        )?;
        let recipe = Recipe::Rows(self.tiled_value_rows(&convolution, 2 * keys, width));
        self.push(
            format!("{block}ssm_conv1d.weight"),
            &[kernel, 2 * keys + values],
            Encoding::F32,
            recipe,
            ValueOp::Keep,
        );
        let step = self.source(format!("{source}linear_attn.dt_bias"), &[heads])?;
        self.push(
            format!("{block}ssm_dt.bias"),
            &[heads],
            Encoding::F32,
            Recipe::Columns {
                source: step,
                columns,
            },
            ValueOp::Keep,
        );
        // The one norm gain the checkpoint does not centre on zero.
        self.direct(
            format!("{block}ssm_norm.weight"),
            format!("{source}linear_attn.norm.weight"),
            &[geometry.linear_key_width],
            Encoding::F32,
            ValueOp::Keep,
        )?;
        let output = self.source(
            format!("{source}linear_attn.out_proj.weight"),
            &[hidden, values],
        )?;
        let columns = self.tiled_value_columns(width);
        self.push(
            format!("{block}ssm_out.weight"),
            &[values, hidden],
            Encoding::Q8_0,
            Recipe::Columns {
                source: output,
                columns,
            },
            ValueOp::Keep,
        );
        Ok(())
    }

    fn ngram(&mut self, block: &str, source: &str) -> Result<(), LayoutError> {
        let geometry = self.geometry;
        let hidden = geometry.hidden;
        let wide = geometry.stream_width();
        self.direct(
            format!("{block}ple_conv1d.weight"),
            format!("{source}ple.conv1d.weight"),
            &[wide, 1, geometry.ngram_convolution_kernel],
            Encoding::F32,
            ValueOp::Keep,
        )?;
        self.direct(
            format!("{block}ple_key.weight"),
            format!("{source}ple.key_proj.weight"),
            &[wide, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )?;
        for name in ["norm_conv", "norm_key", "norm_query"] {
            self.direct(
                format!("{block}ple_{name}.weight"),
                format!("{source}ple.{name}.weight"),
                &[wide],
                Encoding::F32,
                ValueOp::PlusOne,
            )?;
        }
        self.direct(
            format!("{block}ple_value.weight"),
            format!("{source}ple.value_proj.weight"),
            &[hidden, hidden],
            Encoding::Q8_0,
            ValueOp::Keep,
        )
    }

    /// One block's tensors in name order. `layer` is `None` for the next-token block.
    fn block(
        &mut self,
        block: &str,
        source: &str,
        layer: Option<usize>,
    ) -> Result<(), LayoutError> {
        let linear = layer.is_some_and(|layer| self.geometry.linear_layers[layer]);
        let (gate_up, down, inject) = match layer {
            Some(layer) => (
                if WIDE_GATE_UP_LAYERS.contains(&layer) {
                    (Encoding::Q5K, Fit::LeastSquares)
                } else {
                    (Encoding::Q4K, Fit::LeastSquares)
                },
                if WIDE_DOWN_LAYERS.contains(&layer) {
                    (Encoding::Q8_0, Fit::Reference)
                } else {
                    (Encoding::Q5_1, Fit::LeastSquares)
                },
                Encoding::F32,
            ),
            // The next-token head is 8-bit throughout.
            None => (
                (Encoding::Q8_0, Fit::Reference),
                (Encoding::Q8_0, Fit::Reference),
                Encoding::Q8_0,
            ),
        };
        if linear {
            self.linear_attention_head(block, source)?;
        } else {
            self.full_attention(block, source)?;
        }
        self.feed_forward(block, source, gate_up, down)?;
        self.hyper_connection(
            &format!("{block}hc_attn"),
            &format!("{source}attn_hyper_connection"),
            Some(inject),
        )?;
        self.hyper_connection(
            &format!("{block}hc_ffn"),
            &format!("{source}mlp_hyper_connection"),
            Some(inject),
        )?;
        if !linear {
            self.indexer(block, source)?;
        }
        if layer.is_some_and(|layer| self.geometry.ngram_layers.contains(&layer)) {
            self.ngram(block, source)?;
        }
        if linear {
            self.linear_attention_state(block, source)?;
        }
        Ok(())
    }
}

fn check_geometry(geometry: &Geometry) -> Result<(), LayoutError> {
    let positive = [
        geometry.hidden,
        geometry.vocabulary,
        geometry.experts,
        geometry.expert_width,
        geometry.shared_expert_width,
        geometry.streams,
        geometry.stream_rank,
        geometry.attention_heads,
        geometry.attention_key_value_heads,
        geometry.attention_head_width,
        geometry.indexer_heads,
        geometry.indexer_key_value_heads,
        geometry.indexer_head_width,
        geometry.linear_key_heads,
        geometry.linear_value_heads,
        geometry.linear_key_width,
        geometry.linear_value_width,
        geometry.convolution_kernel,
        geometry.ngram_parts,
        geometry.ngram_convolution_kernel,
    ];
    if positive.contains(&0) || geometry.linear_layers.is_empty() {
        return Err(LayoutError::Geometry(
            "model geometry has an empty dimension",
        ));
    }
    if !geometry
        .linear_value_heads
        .is_multiple_of(geometry.linear_key_heads)
    {
        return Err(LayoutError::Geometry(
            "value heads are not a multiple of key heads",
        ));
    }
    if geometry.next_token_layers != 1 {
        return Err(LayoutError::Geometry("expected one next-token layer"));
    }
    let [ngram_layer] = geometry.ngram_layers.as_slice() else {
        return Err(LayoutError::Geometry("expected one n-gram embedding layer"));
    };
    if *ngram_layer >= geometry.linear_layers.len() {
        return Err(LayoutError::Geometry("n-gram layer is out of range"));
    }
    Ok(())
}

/// Lays out the converted model for a checkpoint with these tensors.
pub fn layout(geometry: &Geometry, sources: &SourceIndex) -> Result<Layout, LayoutError> {
    check_geometry(geometry)?;
    let hidden = geometry.hidden;
    let trunk = "model.language_model.";

    let mut builder = Builder {
        geometry,
        sources,
        tensors: Vec::new(),
    };
    builder.direct(
        "output.weight".to_owned(),
        "lm_head.weight".to_owned(),
        &[geometry.vocabulary, hidden],
        Encoding::Q8_0,
        ValueOp::Keep,
    )?;
    builder.hyper_connection("output_hc", &format!("{trunk}hyper_connection_mixer"), None)?;

    // The n-gram table is stored in parts, concatenated in index order.
    let ngram_layer = geometry.ngram_layers[0];
    let mut runs = Vec::new();
    let mut ngram_width = None;
    for part in 0..geometry.ngram_parts {
        let name = format!(
            "{trunk}layers.{ngram_layer}.ple.ple_embedding.ngram_embedding.shard_{part}.weight"
        );
        let (rows, width) = match sources.get(&name) {
            None => return Err(LayoutError::Missing(name)),
            Some((Dtype::Bf16, shape)) => match shape.as_slice() {
                [rows, width] if *rows > 0 && *width > 0 => (*rows, *width),
                _ => return Err(LayoutError::Shape(name)),
            },
            Some(_) => return Err(LayoutError::Shape(name)),
        };
        if *ngram_width.get_or_insert(width) != width {
            return Err(LayoutError::Shape(name));
        }
        runs.push(RowRun {
            source: name,
            first_row: 0,
            rows,
        });
    }
    let ngram_rows: u64 = runs.iter().map(|run| run.rows).sum();
    let ngram_width = ngram_width.ok_or(LayoutError::Geometry("n-gram table has no parts"))?;
    builder.push(
        "per_layer_token_embd.weight".to_owned(),
        &[ngram_width, ngram_rows],
        Encoding::Iq4Nl,
        Recipe::Rows(runs),
        ValueOp::Keep,
    );
    builder.direct(
        "token_embd.weight".to_owned(),
        format!("{trunk}embed_tokens.weight"),
        &[geometry.vocabulary, hidden],
        Encoding::Q8_0,
        ValueOp::Keep,
    )?;
    for layer in 0..geometry.linear_layers.len() {
        builder.block(
            &format!("blk.{layer}."),
            &format!("{trunk}layers.{layer}."),
            Some(layer),
        )?;
    }
    let target = std::mem::take(&mut builder.tensors);

    // The next-token head is numbered as the block after the trunk.
    let block = format!("blk.{}.", geometry.linear_layers.len());
    builder.block(&block, "mtp.layers.0.", None)?;
    let embedding = builder.source("mtp.fc_embedding.weight".to_owned(), &[hidden, hidden])?;
    let state = builder.source("mtp.fc_hidden.weight".to_owned(), &[hidden, hidden])?;
    builder.push(
        format!("{block}nextn.eh_proj.weight"),
        &[2 * hidden, hidden],
        Encoding::Q8_0,
        Recipe::Wide {
            left: embedding,
            right: state,
        },
        ValueOp::Keep,
    );
    builder.direct(
        format!("{block}nextn.enorm.weight"),
        "mtp.pre_fc_norm_embedding.weight".to_owned(),
        &[hidden],
        Encoding::F32,
        ValueOp::PlusOne,
    )?;
    builder.hyper_connection(
        &format!("{block}nextn.hc_head"),
        "mtp.hyper_connection_mixer",
        None,
    )?;
    builder.direct(
        format!("{block}nextn.hnorm.weight"),
        "mtp.pre_fc_norm_hidden.weight".to_owned(),
        &[geometry.stream_width()],
        Encoding::F32,
        ValueOp::PlusOne,
    )?;

    Ok(Layout {
        target,
        next_token: builder.tensors,
    })
}

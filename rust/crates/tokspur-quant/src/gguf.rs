//! GGUF v3 header writer: metadata records and the tensor directory.

/// One metadata value, in the GGUF types the converted files use.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U16(u16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Text(Vec<u8>),
    I32s(Vec<i32>),
    U64s(Vec<u64>),
    Texts(Vec<Vec<u8>>),
}

impl Value {
    /// A string value.
    pub fn text(text: &str) -> Self {
        Self::Text(text.as_bytes().to_vec())
    }
}

/// One entry of a file's tensor directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub name: String,
    /// Innermost dimension first.
    pub dimensions: Vec<u64>,
    pub kind: u32,
    /// Offset from the start of the data section.
    pub offset: u64,
}

/// The data alignment GGUF assumes when a file does not state one.
pub const ALIGNMENT: u64 = 32;

/// `value` rounded up to the data alignment.
pub const fn aligned(value: u64) -> u64 {
    value.div_ceil(ALIGNMENT) * ALIGNMENT
}

fn push_text(out: &mut Vec<u8>, text: &[u8]) {
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text);
}

fn push_value(out: &mut Vec<u8>, value: &Value) {
    let (kind, element): (u32, Option<u32>) = match value {
        Value::U16(_) => (2, None),
        Value::U32(_) => (4, None),
        Value::I32(_) => (5, None),
        Value::F32(_) => (6, None),
        Value::Bool(_) => (7, None),
        Value::Text(_) => (8, None),
        Value::I32s(_) => (9, Some(5)),
        Value::U64s(_) => (9, Some(10)),
        Value::Texts(_) => (9, Some(8)),
    };
    out.extend_from_slice(&kind.to_le_bytes());
    if let Some(element) = element {
        out.extend_from_slice(&element.to_le_bytes());
    }
    match value {
        Value::U16(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::U32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::I32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::F32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::Bool(value) => out.push(u8::from(*value)),
        Value::Text(text) => push_text(out, text),
        Value::I32s(items) => {
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                out.extend_from_slice(&item.to_le_bytes());
            }
        }
        Value::U64s(items) => {
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                out.extend_from_slice(&item.to_le_bytes());
            }
        }
        Value::Texts(items) => {
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                push_text(out, item);
            }
        }
    }
}

/// The header of a GGUF v3 file, padded so the data section starts on the alignment.
pub fn header(metadata: &[(String, Value)], tensors: &[Record]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        push_text(&mut out, key.as_bytes());
        push_value(&mut out, value);
    }
    for tensor in tensors {
        push_text(&mut out, tensor.name.as_bytes());
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&tensor.kind.to_le_bytes());
        out.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    out.resize(aligned(out.len() as u64) as usize, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokspur_gguf::Gguf;

    #[test]
    fn the_runtime_reader_accepts_a_written_header() {
        let metadata = vec![
            ("general.architecture".to_owned(), Value::text("qwen4exp")),
            ("split.no".to_owned(), Value::U16(1)),
            ("split.tensors.count".to_owned(), Value::I32(2)),
            ("qwen4exp.block_count".to_owned(), Value::U32(48)),
            ("qwen4exp.rope.freq_base".to_owned(), Value::F32(1.0e7)),
            (
                "tokenizer.ggml.add_bos_token".to_owned(),
                Value::Bool(false),
            ),
            ("qwen4exp.ple.layers".to_owned(), Value::I32s(vec![1])),
            (
                "qwen4exp.ple.head_offsets".to_owned(),
                Value::U64s(vec![0, 20_000_003]),
            ),
            (
                "general.tags".to_owned(),
                Value::Texts(vec![b"one".to_vec(), b"two".to_vec()]),
            ),
        ];
        let tensors = vec![
            Record {
                name: "a.weight".to_owned(),
                dimensions: vec![32, 2],
                kind: 8,
                offset: 0,
            },
            Record {
                name: "b".to_owned(),
                dimensions: vec![8],
                kind: 0,
                offset: aligned(68),
            },
        ];
        let mut bytes = header(&metadata, &tensors);
        assert!((bytes.len() as u64).is_multiple_of(ALIGNMENT));
        let data_offset = bytes.len() as u64;
        bytes.resize(bytes.len() + aligned(68) as usize + 32, 0);

        let gguf = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(gguf.data_offset(), data_offset);
        assert_eq!(
            gguf.string_value("general.architecture").unwrap(),
            b"qwen4exp"
        );
        assert_eq!(gguf.unsigned_value("split.no").unwrap(), 1);
        assert_eq!(gguf.unsigned_value("qwen4exp.block_count").unwrap(), 48);
        assert_eq!(
            gguf.floating_value("qwen4exp.rope.freq_base").unwrap(),
            1.0e7
        );
        assert_eq!(gguf.unsigned_array("qwen4exp.ple.layers").unwrap(), vec![1]);
        assert_eq!(
            gguf.unsigned_array("qwen4exp.ple.head_offsets").unwrap(),
            vec![0, 20_000_003]
        );
        assert_eq!(
            gguf.string_array("general.tags").unwrap(),
            vec![b"one".as_slice(), b"two".as_slice()]
        );
        let names: Vec<&[u8]> = gguf.tensors().iter().map(|t| t.name.as_slice()).collect();
        assert_eq!(names, vec![b"a.weight".as_slice(), b"b".as_slice()]);
    }
}

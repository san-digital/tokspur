//! The GGUF tokenizer records, built from the checkpoint's `tokenizer.json`.
//!
//! The rules follow llama.cpp's converter for byte-pair vocabularies: ids without a token become unused padding entries, added tokens are control tokens when they are marked special or look like `<|...|>`, and user-defined tokens otherwise.

use std::fmt;

use tokspur_json::{Json, JsonError, JsonLimits, parse_json_limited};

/// GGUF token kinds.
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;

/// The three tokenizer arrays GGUF stores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vocabulary {
    pub tokens: Vec<Vec<u8>>,
    pub kinds: Vec<i32>,
    pub merges: Vec<Vec<u8>>,
}

/// Why `tokenizer.json` could not be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerError {
    Json(JsonError),
    Format(&'static str),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => error.fmt(formatter),
            Self::Format(reason) => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<JsonError> for TokenizerError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

fn field<'a>(value: &'a Json, key: &str) -> Result<&'a Json, TokenizerError> {
    value.find(key)?.ok_or(TokenizerError::Format(
        "tokenizer.json lacks a required field",
    ))
}

/// Whether an added token that is not marked special should still be a control token.
fn looks_special(token: &[u8]) -> bool {
    const KNOWN: [&[u8]; 4] = [b"<pad>", b"<mask>", b"<2mass>", b"[@BOS@]"];
    let wrapped = |open: &[u8], close: &[u8]| token.starts_with(open) && token.ends_with(close);
    KNOWN.contains(&token)
        || wrapped(b"<|", b"|>")
        || wrapped("<\u{ff5c}".as_bytes(), "\u{ff5c}>".as_bytes())
        || wrapped(b"<unused", b">")
}

/// Replaces the SentencePiece word-boundary mark with a space, as the converter does for user-defined tokens.
fn spaced(token: &[u8]) -> Vec<u8> {
    let mark = "\u{2581}".as_bytes();
    let mut out = Vec::with_capacity(token.len());
    let mut rest = token;
    while !rest.is_empty() {
        if rest.starts_with(mark) {
            out.push(b' ');
            rest = &rest[mark.len()..];
        } else {
            out.push(rest[0]);
            rest = &rest[1..];
        }
    }
    out
}

/// Builds the tokenizer arrays for a vocabulary of `size` ids.
pub fn vocabulary(tokenizer_json: &[u8], size: usize) -> Result<Vocabulary, TokenizerError> {
    let limits = JsonLimits::new(tokenizer_json.len(), 16_000_000, 16);
    let json = parse_json_limited(tokenizer_json, limits)?;
    let model = field(&json, "model")?;
    if field(model, "type")?.string()? != b"BPE" {
        return Err(TokenizerError::Format(
            "tokenizer is not byte-pair encoding",
        ));
    }

    let mut tokens: Vec<Option<Vec<u8>>> = vec![None; size];
    let mut kinds = vec![UNUSED; size];
    for (token, id) in field(model, "vocab")?.object()? {
        let id = usize::try_from(id.unsigned_integer()?)
            .ok()
            .filter(|&id| id < size)
            .ok_or(TokenizerError::Format("token id is outside the vocabulary"))?;
        if tokens[id].replace(token.clone()).is_some() {
            return Err(TokenizerError::Format("two tokens share an id"));
        }
        kinds[id] = NORMAL;
    }
    for added in field(&json, "added_tokens")?.array()? {
        let id = usize::try_from(field(added, "id")?.unsigned_integer()?)
            .ok()
            .filter(|&id| id < size)
            .ok_or(TokenizerError::Format(
                "added token id is outside the vocabulary",
            ))?;
        let content = field(added, "content")?.string()?;
        if tokens[id].as_deref().is_some_and(|token| token != content) {
            return Err(TokenizerError::Format(
                "added token disagrees with the vocabulary",
            ));
        }
        if field(added, "special")?.boolean()? || looks_special(content) {
            tokens[id] = Some(content.to_vec());
            kinds[id] = CONTROL;
        } else {
            tokens[id] = Some(spaced(content));
            kinds[id] = USER_DEFINED;
        }
    }
    let tokens = tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.unwrap_or_else(|| format!("[PAD{id}]").into_bytes()))
        .collect();

    let mut merges = Vec::new();
    for merge in field(model, "merges")?.array()? {
        merges.push(match merge {
            Json::String(pair) => pair.clone(),
            other => {
                let [left, right] = other.array()?.as_slice() else {
                    return Err(TokenizerError::Format("merge is not a pair"));
                };
                let mut pair = left.string()?.to_vec();
                pair.push(b' ');
                pair.extend_from_slice(right.string()?);
                pair
            }
        });
    }
    Ok(Vocabulary {
        tokens,
        kinds,
        merges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "added_tokens": [
            {"id": 3, "content": "<|end|>", "special": true},
            {"id": 4, "content": "<tool_call>", "special": false},
            {"id": 5, "content": "<|quiet|>", "special": false},
            {"id": 6, "content": "▁word", "special": false}
        ],
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2, "<|end|>": 3},
            "merges": ["a b", ["ab", "a"]]
        }
    }"#;

    #[test]
    fn builds_tokens_kinds_and_merges() {
        let vocabulary = vocabulary(SAMPLE.as_bytes(), 9).unwrap();
        let tokens: Vec<&[u8]> = vocabulary.tokens.iter().map(Vec::as_slice).collect();
        assert_eq!(
            tokens,
            vec![
                b"a".as_slice(),
                b"b",
                b"ab",
                b"<|end|>",
                b"<tool_call>",
                b"<|quiet|>",
                b" word",
                b"[PAD7]",
                b"[PAD8]"
            ]
        );
        assert_eq!(vocabulary.kinds, vec![1, 1, 1, 3, 4, 3, 4, 5, 5]);
        assert_eq!(vocabulary.merges, vec![b"a b".to_vec(), b"ab a".to_vec()]);
    }

    #[test]
    fn refuses_ids_outside_the_vocabulary_and_other_models() {
        assert!(vocabulary(SAMPLE.as_bytes(), 3).is_err());
        let other = SAMPLE.replace("BPE", "Unigram");
        assert!(vocabulary(other.as_bytes(), 9).is_err());
        let clash = SAMPLE.replace(r#""ab": 2"#, r#""ab": 1"#);
        assert!(vocabulary(clash.as_bytes(), 9).is_err());
    }
}

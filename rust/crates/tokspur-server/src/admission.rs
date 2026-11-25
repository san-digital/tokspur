//! Request, tool-schema and worker-readiness admission from `native/src/serve_protocol.cpp` (everything except `ServeSession`). CPU only; no tool is ever executed here.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use std::sync::Arc;
use std::time::SystemTimeError;
use tokspur_json::{
    Json, JsonError, JsonLimits, Kind, from_chars_f64, from_chars_i64, parse_json,
    parse_json_limited, scalar_str, to_chars_shortest, unicode_scalars,
};

use crate::pattern::compile;
use tokspur_text::response::{ParameterSpec, ParameterType, ResponseError, ToolSpec};
use tokspur_wire::mtp_wire::MtpWireError;

/// Policy governing whether tool parameter `enum` membership is enforced at the server or deferred to the agent executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolEnumValidation {
    #[default]
    Strict,
    Executor,
}

/// Parses `--tool-enum-validation` values (`strict` | `executor`).
pub fn parse_tool_enum_validation(value: &str) -> Result<ToolEnumValidation> {
    match value {
        "strict" => Ok(ToolEnumValidation::Strict),
        "executor" => Ok(ToolEnumValidation::Executor),
        _ => Err(AdmissionError::Refused(
            "tool-enum-validation must be strict or executor",
        )),
    }
}

/// Canonical short name for `ToolEnumValidation`.
pub const fn tool_enum_validation_name(value: ToolEnumValidation) -> &'static str {
    match value {
        ToolEnumValidation::Strict => "strict",
        ToolEnumValidation::Executor => "executor",
    }
}

/// Canonical startup policy identifier for `ToolEnumValidation`.
pub const fn tool_enum_validation_policy(value: ToolEnumValidation) -> &'static str {
    match value {
        ToolEnumValidation::Strict => "strict-server-enums-v1",
        ToolEnumValidation::Executor => "original-agent-executor-enums-v1",
    }
}

/// Policy governing whether tool argument schema properties/required/types are enforced at the server or deferred to the agent executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolArgValidation {
    #[default]
    Strict,
    Executor,
}

/// Parses `--tool-arg-validation` values (`strict` | `executor`).
pub fn parse_tool_arg_validation(value: &str) -> Result<ToolArgValidation> {
    match value {
        "strict" => Ok(ToolArgValidation::Strict),
        "executor" => Ok(ToolArgValidation::Executor),
        _ => Err(AdmissionError::Refused(
            "tool-arg-validation must be strict or executor",
        )),
    }
}

/// Canonical short name for `ToolArgValidation`.
pub const fn tool_arg_validation_name(value: ToolArgValidation) -> &'static str {
    match value {
        ToolArgValidation::Strict => "strict",
        ToolArgValidation::Executor => "executor",
    }
}

/// Canonical startup policy identifier for `ToolArgValidation`.
pub const fn tool_arg_validation_policy(value: ToolArgValidation) -> &'static str {
    match value {
        ToolArgValidation::Strict => "strict-server-args-v1",
        ToolArgValidation::Executor => "executor-deferred-args-v1",
    }
}

/// The C++ `FormatError` raised by serving admission.
#[derive(Clone)]
pub enum AdmissionError {
    /// `require`: a request, schema or readiness check refused.
    Refused(&'static str),
    /// A JSON accessor, parse or dump refused a value.
    Json(JsonError),
    /// A response parameter marker check refused a value.
    Response(ResponseError),
    /// System clock before UNIX epoch when building an API timestamp.
    Clock(SystemTimeError),
    /// Persisting a private prefix trace record failed.
    TracePersist(Arc<std::io::Error>),
    /// The private prefix trace cannot take a record: it is full, or it failed earlier. A fault of the server, not of the request.
    PrefixTrace(&'static str),
    /// MTP wire validation refused inside a session.
    Mtp(MtpWireError),
}

impl PartialEq for AdmissionError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Refused(left), Self::Refused(right)) => left == right,
            (Self::Json(left), Self::Json(right)) => left == right,
            (Self::Response(left), Self::Response(right)) => left == right,
            (Self::Clock(left), Self::Clock(right)) => left.duration() == right.duration(),
            (Self::TracePersist(left), Self::TracePersist(right)) => {
                left.kind() == right.kind() && left.raw_os_error() == right.raw_os_error()
            }
            (Self::PrefixTrace(left), Self::PrefixTrace(right)) => left == right,
            (Self::Mtp(left), Self::Mtp(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for AdmissionError {}

impl fmt::Debug for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::Json(error) => formatter.debug_tuple("Json").field(error).finish(),
            Self::Response(error) => formatter.debug_tuple("Response").field(error).finish(),
            Self::Clock(_) => formatter.write_str("Clock([redacted])"),
            Self::TracePersist(_) => formatter.write_str("TracePersist([redacted])"),
            Self::PrefixTrace(reason) => {
                formatter.debug_tuple("PrefixTrace").field(reason).finish()
            }
            Self::Mtp(error) => formatter.debug_tuple("Mtp").field(error).finish(),
        }
    }
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) | Self::PrefixTrace(reason) => formatter.write_str(reason),
            Self::Json(error) => error.fmt(formatter),
            Self::Response(error) => error.fmt(formatter),
            Self::Clock(_) => formatter.write_str("invalid API clock"),
            Self::TracePersist(_) => formatter.write_str("cannot persist private prefix trace"),
            Self::Mtp(MtpWireError::Refused(reason)) => formatter.write_str(reason),
            Self::Mtp(MtpWireError::MissingField(_)) => formatter.write_str("missing field"),
            Self::Mtp(MtpWireError::Json(error)) => error.fmt(formatter),
            Self::Mtp(MtpWireError::WorkerError(msg)) => formatter.write_str(msg),
        }
    }
}

impl std::error::Error for AdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Response(error) => Some(error),
            Self::Clock(error) => Some(error),
            Self::TracePersist(error) => Some(error.as_ref()),
            Self::Mtp(error) => Some(error),
            Self::Refused(_) | Self::PrefixTrace(_) => None,
        }
    }
}

impl From<JsonError> for AdmissionError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<ResponseError> for AdmissionError {
    fn from(error: ResponseError) -> Self {
        Self::Response(error)
    }
}

type Result<T> = std::result::Result<T, AdmissionError>;

pub(crate) fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(AdmissionError::Refused(why))
    }
}

pub(crate) fn field(value: &Json, key: impl AsRef<[u8]>) -> Result<&Json> {
    value
        .find(key)?
        .ok_or(AdmissionError::Refused("missing required API field"))
}

fn keys(value: &Json, allowed: &[&str]) -> Result<()> {
    for (key, _) in value.object()? {
        require(
            allowed.iter().any(|name| name.as_bytes() == key.as_slice()),
            "unsupported API/schema field",
        )?;
    }
    Ok(())
}

pub(crate) fn number(value: &Json, low: f64, high: f64) -> Result<f64> {
    let text = value.number_lexeme()?;
    match from_chars_f64(text) {
        Some(result) if result.is_finite() && result >= low && result <= high => Ok(result),
        _ => Err(AdmissionError::Refused(
            "API number outside supported range",
        )),
    }
}

/// `real(double)`: `std::to_chars` then the owned parser, so the lexeme is exactly the C++ one.
pub(crate) fn real(value: f64) -> Result<Json> {
    let text =
        to_chars_shortest(value).ok_or(AdmissionError::Refused("cannot encode API number"))?;
    Ok(parse_json(text.as_bytes())?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Decimal {
    negative: bool,
    digits: Vec<u8>,
    exponent: i64,
}

fn decimal(value: &Json) -> Result<Decimal> {
    let text = value.number_lexeme()?.as_bytes();
    require(text.len() <= 128, "tool numeric lexeme bound")?;
    let mut i = usize::from(text.first() == Some(&b'-'));
    let negative = i != 0;
    let mut digits = Vec::new();
    let mut fraction: i64 = 0;
    let mut after_point = false;
    while i < text.len() && text[i] != b'e' && text[i] != b'E' {
        if text[i] == b'.' {
            after_point = true;
        } else {
            digits.push(text[i]);
            if after_point {
                fraction += 1;
            }
        }
        i += 1;
    }
    let mut exponent: i64 = 0;
    if i < text.len() {
        i += 1;
        let mut minus = false;
        if matches!(text.get(i), Some(b'-' | b'+')) {
            minus = text[i] == b'-';
            i += 1;
        }
        let rest = std::str::from_utf8(&text[i..]).ok();
        match rest.and_then(from_chars_i64) {
            Some(parsed) if parsed <= 1_000_000 => exponent = parsed,
            _ => return Err(AdmissionError::Refused("tool numeric exponent bound")),
        }
        if minus {
            exponent = -exponent;
        }
    }
    exponent -= fraction;
    let Some(first) = digits.iter().position(|&c| c != b'0') else {
        return Ok(Decimal {
            negative: false,
            digits: b"0".to_vec(),
            exponent: 0,
        });
    };
    digits.drain(..first);
    while digits.last() == Some(&b'0') {
        digits.pop();
        exponent += 1;
    }
    Ok(Decimal {
        negative,
        digits,
        exponent,
    })
}

fn compare(left: &Decimal, right: &Decimal) -> i32 {
    if left.digits == b"0" && right.digits == b"0" {
        return 0;
    }
    if left.negative != right.negative {
        return if left.negative { -1 } else { 1 };
    }
    let mut order = 0;
    if left.digits == b"0" {
        order = -1;
    } else if right.digits == b"0" {
        order = 1;
    } else {
        let l = left.digits.len() as i64 + left.exponent;
        let r = right.digits.len() as i64 + right.exponent;
        if l != r {
            order = if l < r { -1 } else { 1 };
        } else {
            for i in 0..left.digits.len().max(right.digits.len()) {
                let a = left.digits.get(i).copied().unwrap_or(b'0');
                let b = right.digits.get(i).copied().unwrap_or(b'0');
                if a != b {
                    order = if a < b { -1 } else { 1 };
                    break;
                }
            }
        }
    }
    if left.negative { -order } else { order }
}

/// The `multipleOf` divisor as its significant digits and decimal exponent. Eighteen digits keep every remainder step below 10^19, inside `u64`; the shortest lexeme that round-trips a double has at most seventeen.
fn divisor(value: &Json) -> Result<(u64, i64)> {
    let step = decimal(value)?;
    require(
        !step.negative && step.digits != b"0",
        "multipleOf must be positive",
    )?;
    require(step.digits.len() <= 18, "multipleOf digit bound")?;
    let digits = step
        .digits
        .iter()
        .fold(0u64, |total, &c| total * 10 + u64::from(c - b'0'));
    Ok((digits, step.exponent))
}

/// Whether `value` divided by the divisor is an integer, in exact decimal arithmetic.
fn multiple(value: &Decimal, (digits, exponent): (u64, i64)) -> bool {
    if value.digits == b"0" {
        return true;
    }
    // The value's digits do not end in zero, so they are never divisible by the power of ten that a negative shift leaves in the denominator.
    let shift = value.exponent - exponent;
    if shift < 0 {
        return false;
    }
    let mut rest = value
        .digits
        .iter()
        .fold(0u64, |rest, &c| (rest * 10 + u64::from(c - b'0')) % digits);
    // The divisor is below 2^60, so it holds at most 59 factors of two and fewer of five: beyond 60, further powers of ten cannot change whether the remainder is zero.
    for _ in 0..shift.min(60) {
        rest = (rest * 10) % digits;
    }
    rest == 0
}

/// Appends a form in which two values are the same bytes exactly when JSON Schema calls them equal: numbers by exact decimal value, object members in name order, everything else as `dump` writes it.
fn canonical(value: &Json, out: &mut Vec<u8>) -> Result<()> {
    match value.kind() {
        Kind::Number => {
            let n = decimal(value)?;
            if n.negative {
                out.push(b'-');
            }
            out.extend_from_slice(&n.digits);
            out.push(b'e');
            out.extend_from_slice(n.exponent.to_string().as_bytes());
        }
        Kind::Array => {
            out.push(b'[');
            for item in value.array()? {
                canonical(item, out)?;
                out.push(b',');
            }
            out.push(b']');
        }
        Kind::Object => {
            let mut members: Vec<&(Vec<u8>, Json)> = value.object()?.iter().collect();
            members.sort_by(|left, right| left.0.cmp(&right.0));
            out.push(b'{');
            for (name, item) in members {
                out.extend_from_slice(&Json::String(name.clone()).dump()?);
                out.push(b':');
                canonical(item, out)?;
                out.push(b',');
            }
            out.push(b'}');
        }
        Kind::String | Kind::Boolean | Kind::Null => out.extend_from_slice(&value.dump()?),
    }
    Ok(())
}

fn safe_name(name: &[u8], function: bool) -> Result<()> {
    require(
        !name.is_empty() && name.len() <= if function { 64 } else { 1024 },
        "tool name bound",
    )?;
    for &c in name {
        require(
            (32..127).contains(&c) && c != b'<' && c != b'>',
            "unsupported tool name",
        )?;
        if function {
            require(
                c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.',
                "unsupported function name",
            )?;
        }
    }
    Ok(())
}

fn schema_type(schema: &Json) -> Result<&[u8]> {
    if schema.find("anyOf")?.is_some() {
        return Ok(b"any");
    }
    let value = field(schema, "type")?.string()?;
    require(
        matches!(
            value,
            b"string" | b"boolean" | b"number" | b"integer" | b"object" | b"array" | b"null"
        ),
        "unsupported tool schema type",
    )?;
    Ok(value)
}

fn schema_admit(schema: &Json, depth: usize, nodes: &mut usize) -> Result<()> {
    require(
        depth <= 16 && {
            *nodes += 1;
            *nodes <= 4096
        },
        "tool schema complexity bound",
    )?;
    keys(
        schema,
        &[
            "type",
            "anyOf",
            "title",
            "default",
            "examples",
            "format",
            "$schema",
            "description",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
            "minLength",
            "maxLength",
            "pattern",
            "enum",
            "const",
            "minimum",
            "exclusiveMinimum",
            "maximum",
            "exclusiveMaximum",
            "multipleOf",
        ],
    )?;
    let kind = schema_type(schema)?;
    if let Some(dialect) = schema.find("$schema")? {
        require(
            depth == 0 && dialect.string()? == b"https://json-schema.org/draft/2020-12/schema",
            "unsupported tool schema dialect",
        )?;
    }
    if let Some(description) = schema.find("description")? {
        require(
            description.string()?.len() <= 16384,
            "schema description bound",
        )?;
    }
    for name in ["title", "format"] {
        if let Some(value) = schema.find(name)? {
            require(value.string()?.len() <= 16384, "schema annotation bound")?;
        }
    }
    if let Some(examples) = schema.find("examples")? {
        require(examples.array()?.len() <= 128, "schema examples bound")?;
    }
    if kind == b"any" {
        keys(
            schema,
            &[
                "anyOf",
                "$schema",
                "title",
                "description",
                "default",
                "examples",
                "format",
            ],
        )?;
        let alternatives = field(schema, "anyOf")?.array()?;
        require(
            !alternatives.is_empty() && alternatives.len() <= 8,
            "schema union bound",
        )?;
        for branch in alternatives {
            schema_admit(branch, depth + 1, nodes)?;
        }
        return Ok(());
    }
    if kind == b"object" {
        let properties = schema.find("properties")?;
        if let Some(properties) = properties {
            require(properties.object()?.len() <= 32, "too many tool parameters")?;
            for (name, child) in properties.object()? {
                safe_name(name, false)?;
                schema_admit(child, depth + 1, nodes)?;
            }
        }
        if let Some(additional) = schema.find("additionalProperties")? {
            require(
                !additional.boolean()?,
                "additional tool properties are unsupported",
            )?;
        }
        if let Some(required) = schema.find("required")? {
            let mut seen = BTreeSet::new();
            for item in required.array()? {
                let name = item.string()?;
                let present = match properties {
                    Some(properties) => properties.find(name)?.is_some(),
                    None => false,
                };
                require(
                    present && seen.insert(name),
                    "invalid required tool parameter",
                )?;
            }
        }
    } else {
        require(
            schema.find("properties")?.is_none()
                && schema.find("required")?.is_none()
                && schema.find("additionalProperties")?.is_none(),
            "object schema keyword on other type",
        )?;
    }
    if kind == b"array" {
        schema_admit(field(schema, "items")?, depth + 1, nodes)?;
        let low = match schema.find("minItems")? {
            Some(low) => Some(low.unsigned_integer()?),
            None => None,
        };
        let high = match schema.find("maxItems")? {
            Some(high) => Some(high.unsigned_integer()?),
            None => None,
        };
        // Validation refuses any array above 4096 items, so a larger minimum could never be met. A larger maximum stays admitted: the same cap still applies at validation.
        if let Some(low) = low {
            require(low <= 4096, "minItems exceeds the tool array bound")?;
        }
        if let (Some(low), Some(high)) = (low, high) {
            require(low <= high, "inverted schema bounds")?;
        }
        if let Some(unique) = schema.find("uniqueItems")? {
            unique.boolean()?;
        }
    } else {
        require(
            schema.find("items")?.is_none()
                && schema.find("minItems")?.is_none()
                && schema.find("maxItems")?.is_none()
                && schema.find("uniqueItems")?.is_none(),
            "array schema keyword on other type",
        )?;
    }
    if kind == b"string" {
        let low = match schema.find("minLength")? {
            Some(low) => Some(low.unsigned_integer()?),
            None => None,
        };
        let high = match schema.find("maxLength")? {
            Some(high) => Some(high.unsigned_integer()?),
            None => None,
        };
        // The response parser caps a generated tool call at 64 KiB (`ResponseLimits::call_bytes`), and a code point takes at least one byte, so no call this server emits could meet a larger minimum. A larger maximum stays admitted: it only fails to constrain.
        if let Some(low) = low {
            require(low <= 65536, "minLength exceeds the tool call byte limit")?;
        }
        if let (Some(low), Some(high)) = (low, high) {
            require(low <= high, "inverted schema bounds")?;
        }
        if let Some(pattern) = schema.find("pattern")? {
            compile(pattern.string()?)?;
        }
    } else {
        require(
            schema.find("minLength")?.is_none()
                && schema.find("maxLength")?.is_none()
                && schema.find("pattern")?.is_none(),
            "string schema keyword on other type",
        )?;
    }
    if let Some(values) = schema.find("enum")? {
        require(
            kind != b"object"
                && kind != b"array"
                && !values.array()?.is_empty()
                && values.array()?.len() <= 128,
            "unsupported enum",
        )?;
        // Admission checks enum members against the same type, without enum.
        let base = Json::Object(
            schema
                .object()?
                .iter()
                .filter(|(key, _)| key.as_slice() != b"enum")
                .cloned()
                .collect(),
        );
        for value in values.array()? {
            validate_tool_value(&base, value)?;
        }
    }
    if let Some(value) = schema.find("const")? {
        require(kind != b"object" && kind != b"array", "unsupported const")?;
        // Admission checks the constant against the same type, without const.
        let base = Json::Object(
            schema
                .object()?
                .iter()
                .filter(|(key, _)| key.as_slice() != b"const")
                .cloned()
                .collect(),
        );
        validate_tool_value(&base, value)?;
    }
    if schema.find("minimum")?.is_some()
        || schema.find("maximum")?.is_some()
        || schema.find("exclusiveMinimum")?.is_some()
        || schema.find("exclusiveMaximum")?.is_some()
        || schema.find("multipleOf")?.is_some()
    {
        require(
            kind == b"integer" || kind == b"number",
            "numeric bound on other schema type",
        )?;
        if let Some(low) = schema.find("minimum")? {
            number(low, -1e300, 1e300)?;
            decimal(low)?;
        }
        if let Some(high) = schema.find("maximum")? {
            number(high, -1e300, 1e300)?;
            decimal(high)?;
        }
        if let Some(low) = schema.find("exclusiveMinimum")? {
            number(low, -1e300, 1e300)?;
            decimal(low)?;
            if let Some(high) = schema.find("maximum")? {
                require(
                    compare(&decimal(low)?, &decimal(high)?) < 0,
                    "inverted schema bounds",
                )?;
            }
        }
        if let Some(high) = schema.find("exclusiveMaximum")? {
            number(high, -1e300, 1e300)?;
            decimal(high)?;
            if let Some(low) = schema.find("minimum")? {
                require(
                    compare(&decimal(low)?, &decimal(high)?) < 0,
                    "inverted schema bounds",
                )?;
            }
            if let Some(low) = schema.find("exclusiveMinimum")? {
                require(
                    compare(&decimal(low)?, &decimal(high)?) < 0,
                    "inverted schema bounds",
                )?;
            }
        }
        if let (Some(low), Some(high)) = (schema.find("minimum")?, schema.find("maximum")?) {
            require(
                compare(&decimal(low)?, &decimal(high)?) <= 0,
                "inverted schema bounds",
            )?;
        }
        if let Some(step) = schema.find("multipleOf")? {
            number(step, -1e300, 1e300)?;
            divisor(step)?;
        }
    }
    Ok(())
}

fn parameter_type(kind: &[u8]) -> ParameterType {
    match kind {
        b"string" => ParameterType::RawString,
        b"boolean" => ParameterType::Boolean,
        b"number" | b"integer" => ParameterType::Number,
        b"object" => ParameterType::Object,
        b"array" => ParameterType::Array,
        b"any" => ParameterType::JsonValue,
        _ => ParameterType::NullValue,
    }
}

fn validate_executor_string(raw: &[u8]) -> Result<()> {
    require(!raw.contains(&0), "NUL byte inside tool parameter")
}

fn validate_executor_node(value: &Json, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    require(
        depth <= 32 && *nodes <= 16384,
        "executor tool argument complexity bound",
    )?;
    match value.kind() {
        Kind::String => validate_executor_string(value.string()?),
        Kind::Number => {
            number(value, -1e300, 1e300)?;
            Ok(())
        }
        Kind::Array => {
            require(value.array()?.len() <= 4096, "tool array bound")?;
            for item in value.array()? {
                validate_executor_node(item, depth + 1, nodes)?;
            }
            Ok(())
        }
        Kind::Object => {
            require(value.object()?.len() <= 256, "too many tool parameters")?;
            let mut seen: BTreeSet<&[u8]> = BTreeSet::new();
            for (key, item) in value.object()? {
                if depth == 0 {
                    safe_name(key, false)?;
                }
                require(
                    seen.insert(key.as_slice()),
                    "duplicate executor tool property",
                )?;
                validate_executor_string(key)?;
                validate_executor_node(item, depth + 1, nodes)?;
            }
            Ok(())
        }
        Kind::Boolean | Kind::Null => Ok(()),
    }
}

/// Validates executor-deferred tool arguments with bounded depth/size, finite numbers and representable strings.
pub fn validate_executor_tool_arguments(value: &Json) -> Result<()> {
    require(
        value.kind() == Kind::Object,
        "executor tool arguments must be a JSON object",
    )?;
    let mut nodes = 0;
    validate_executor_node(value, 0, &mut nodes)
}

fn thinking_history(message: &Json, role: &[u8], max_request_bytes: usize) -> Result<()> {
    let Some(blocks) = message.find("thinking_blocks")? else {
        return Ok(());
    };
    require(
        role == b"assistant",
        "thinking blocks require assistant history",
    )?;
    require(blocks.array()?.len() <= 4096, "thinking block count bound")?;
    let mut text = Vec::new();
    for block in blocks.array()? {
        keys(block, &["type", "thinking", "signature"])?;
        require(
            field(block, "type")?.string()? == b"thinking",
            "unsupported thinking block type",
        )?;
        if let Some(signature) = block.find("signature")? {
            require(
                signature.string()?.is_empty(),
                "signed thinking blocks are unsupported",
            )?;
        }
        let part = field(block, "thinking")?.string()?;
        require(
            part.len() <= max_request_bytes - text.len(),
            "thinking history bound",
        )?;
        text.extend_from_slice(part);
    }
    let reasoning = match message.find("reasoning_content")? {
        Some(value) if !value.is_null() => value.string()?,
        _ => b"",
    };
    require(
        text == reasoning,
        "thinking history differs from reasoning content",
    )
}

fn history(
    messages: &Json,
    tools: &[Vec<u8>],
    schemas: &[Json],
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
    max_request_bytes: usize,
) -> Result<()> {
    require(
        !messages.array()?.is_empty() && messages.array()?.len() <= 4096,
        "API history bound",
    )?;
    let mut pending: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut used: BTreeSet<Vec<u8>> = BTreeSet::new();
    for message in messages.array()? {
        keys(
            message,
            &[
                "role",
                "content",
                "reasoning_content",
                "thinking_blocks",
                "tool_calls",
                "tool_call_id",
                "name",
            ],
        )?;
        let role = field(message, "role")?.string()?;
        require(
            matches!(
                role,
                b"system" | b"developer" | b"user" | b"assistant" | b"tool"
            ),
            "unsupported API message role",
        )?;
        thinking_history(message, role, max_request_bytes)?;
        if let Some(content) = message.find("content")? {
            if content.kind() == Kind::Array {
                require(content.array()?.len() <= 4096, "text content part bound")?;
                for part in content.array()? {
                    keys(part, &["type", "text"])?;
                    require(
                        field(part, "type")?.string()? == b"text",
                        "only text content parts supported",
                    )?;
                    field(part, "text")?.string()?;
                }
            } else {
                require(
                    content.is_null() || content.kind() == Kind::String,
                    "unsupported message content",
                )?;
            }
        }
        if role == b"tool" {
            require(
                message.find("tool_calls")?.is_none()
                    && message.find("reasoning_content")?.is_none(),
                "invalid tool result fields",
            )?;
            require(
                !field(message, "content")?.is_null(),
                "tool result requires text content",
            )?;
            let id = field(message, "tool_call_id")?.string()?;
            let Some(found) = pending.get(id).cloned() else {
                return Err(AdmissionError::Refused(
                    "unsolicited or duplicate tool result",
                ));
            };
            if let Some(name) = message.find("name")? {
                require(
                    name.string()? == found.as_slice(),
                    "tool result name/ID disagreement",
                )?;
            }
            pending.remove(id);
        } else {
            require(
                pending.is_empty(),
                "missing tool results before next message",
            )?;
            require(
                message.find("tool_call_id")?.is_none() && message.find("name")?.is_none(),
                "tool identity field outside tool result",
            )?;
            if role != b"assistant" {
                require(
                    !field(message, "content")?.is_null(),
                    "message requires text content",
                )?;
            }
            if let Some(reasoning) = message.find("reasoning_content")? {
                require(
                    role == b"assistant"
                        && (reasoning.is_null() || reasoning.kind() == Kind::String),
                    "invalid reasoning history",
                )?;
            }
            if let Some(calls) = message.find("tool_calls")? {
                require(
                    role == b"assistant"
                        && !calls.array()?.is_empty()
                        && calls.array()?.len() <= 128,
                    "invalid assistant tool history",
                )?;
                for call in calls.array()? {
                    keys(call, &["id", "type", "function", "index"])?;
                    if let Some(index) = call.find("index")? {
                        require(
                            index.unsigned_integer()? < 128,
                            "historical tool index bound",
                        )?;
                    }
                    require(
                        field(call, "type")?.string()? == b"function",
                        "unsupported historical call type",
                    )?;
                    let id = field(call, "id")?.string()?.to_vec();
                    require(
                        !id.is_empty() && id.len() <= 128 && used.insert(id.clone()),
                        "invalid/duplicate historical tool ID",
                    )?;
                    let function = field(call, "function")?;
                    keys(function, &["name", "arguments"])?;
                    let name = field(function, "name")?.string()?;
                    let found = tools.iter().position(|tool| tool.as_slice() == name);
                    let Some(index) = found else {
                        return Err(AdmissionError::Refused(
                            "historical function absent from admitted tools",
                        ));
                    };
                    let arguments = field(function, "arguments")?;
                    require(
                        arguments.kind() == Kind::Object || arguments.kind() == Kind::String,
                        "invalid historical tool arguments",
                    )?;
                    let parsed;
                    let value = if arguments.kind() == Kind::String {
                        // The server emits `arguments` from a call of at most 64 KiB, and escaping can grow each byte to six (`\u000a`). Accept everything this server can emit, so a client that echoes it verbatim is not refused on its next turn.
                        parsed = parse_json_limited(
                            arguments.string()?,
                            JsonLimits::new(6 * 65536, 65536, 33),
                        )?;
                        &parsed
                    } else {
                        arguments
                    };
                    if tool_args == ToolArgValidation::Executor {
                        validate_executor_tool_arguments(value)?;
                    } else {
                        validate_tool_value_with_policy(&schemas[index], value, tool_enums)?;
                    }
                    pending.entry(id).or_insert_with(|| name.to_vec());
                }
            }
        }
    }
    require(
        pending.is_empty(),
        "request ends with unanswered tool calls",
    )
}

/// The response parser's tool manifest, as JSON (`parser_manifest`).
pub(crate) fn parser_manifest(tools: &[ToolSpec]) -> Json {
    const TYPES: [&str; 7] = [
        "string", "boolean", "number", "object", "array", "null", "any",
    ];
    Json::Array(
        tools
            .iter()
            .map(|tool| {
                let parameters = tool
                    .parameters
                    .iter()
                    .map(|p| {
                        Json::new_object([
                            ("name", Json::from(p.name.clone())),
                            ("type", Json::from(TYPES[p.r#type as usize])),
                            ("required", Json::Bool(p.required)),
                        ])
                    })
                    .collect();
                Json::new_object([
                    ("name", Json::from(tool.name.clone())),
                    ("parameters", Json::Array(parameters)),
                ])
            })
            .collect(),
    )
}

/// One SSE `chat.completion.chunk`.
pub(crate) fn chunk(id: &[u8], alias: &[u8], created: u64, delta: Json, finish: Json) -> Json {
    Json::new_object([
        ("id", Json::from(id)),
        ("object", Json::from("chat.completion.chunk")),
        ("created", Json::integer(created)),
        ("model", Json::from(alias)),
        (
            "choices",
            Json::Array(vec![Json::new_object([
                ("index", Json::integer(0)),
                ("delta", delta),
                ("finish_reason", finish),
            ])]),
        ),
    ])
}

pub(crate) fn worker_execution(value: &Json, selected: &str) -> Result<()> {
    require(
        selected == "direct" || selected == "recurrent-replay",
        "unsupported serving startup execution mode",
    )?;
    require(
        field(value, "small_gemv_column_splits")?.unsigned_integer()? == 16
            && field(value, "block_gemv_formats")?.unsigned_integer()? == 3
            && field(value, "stack_execution")?.string()? == selected.as_bytes(),
        "worker execution policy changed",
    )?;
    if selected == "recurrent-replay" {
        require(
            field(value, "replay_prepared")?.boolean()?
                && field(value, "recurrent_replay_captures")?.unsigned_integer()? == 24
                && field(value, "mapped_weight_region_limit_bytes")?.unsigned_integer()? == 4 << 30,
            "owned recurrent replay preparation/capture/region identity mismatch",
        )?;
    }
    Ok(())
}

pub(crate) fn broker_storage(value: &Json, ready: &Json) -> Result<()> {
    let slots = field(ready, "checkpoint_slots")?.unsigned_integer()?;
    let budget = field(ready, "checkpoint_combined_budget_bytes")?.unsigned_integer()?;
    require(
        slots != 0 && slots <= 4 && budget != 0 && budget <= (512 << 20),
        "invalid serving broker storage preparation",
    )?;
    require(
        field(value, "checkpoint_slots")?.unsigned_integer()? == slots
            && field(value, "checkpoint_combined_budget_bytes")?.unsigned_integer()? == budget,
        "serving broker configuration changed",
    )?;
    let mut total: u64 = 0;
    for name in [
        "checkpoint_host_bytes",
        "checkpoint_store_host_bytes",
        "checkpoint_gpu_bytes",
    ] {
        let bytes = field(value, name)?.unsigned_integer()?;
        require(
            bytes != 0
                && bytes <= budget.wrapping_sub(total)
                && bytes == field(ready, name)?.unsigned_integer()?,
            "serving broker storage charge changed",
        )?;
        total = total.wrapping_add(bytes);
    }
    Ok(())
}

/// CPU-only startup admission for the fixed owned target policy. Every field is mandatory and type-checked; absent metadata never implies the direct path.
pub fn validate_worker_ready(
    ready: &Json,
    stack_execution: &str,
    checkpoint_slots: u32,
    checkpoint_budget: u64,
) -> Result<()> {
    require(
        field(ready, "event")?.string()? == b"ready"
            && field(ready, "maximum_generated_tokens")?.unsigned_integer()? == 4096,
        "owned worker readiness/execution identity mismatch",
    )?;
    worker_execution(ready, stack_execution)?;
    require(
        checkpoint_slots <= 4
            && checkpoint_budget <= (512 << 20)
            && (checkpoint_slots != 0) == (checkpoint_budget != 0),
        "invalid serving broker preparation",
    )?;
    if checkpoint_slots != 0 {
        require(
            stack_execution == "direct"
                && field(ready, "checkpoint_broker")?.boolean()?
                && field(ready, "broker_epoch")?.unsigned_integer()? != 0
                && field(ready, "checkpoint_slots")?.unsigned_integer()?
                    == u64::from(checkpoint_slots)
                && field(ready, "checkpoint_combined_budget_bytes")?.unsigned_integer()?
                    == checkpoint_budget,
            "owned broker readiness identity mismatch",
        )?;
        broker_storage(ready, ready)?;
    } else if let Some(enabled) = ready.find("checkpoint_broker")? {
        require(!enabled.boolean()?, "unexpected serving broker capability")?;
    }
    Ok(())
}

/// Validates the supported schema subset recursively under strict `enum` enforcement.
pub fn validate_tool_value(schema: &Json, value: &Json) -> Result<()> {
    validate_tool_value_with_policy(schema, value, ToolEnumValidation::Strict)
}

/// Validates the supported schema subset recursively under the selected `ToolEnumValidation` policy.
pub fn validate_tool_value_with_policy(
    schema: &Json,
    value: &Json,
    tool_enums: ToolEnumValidation,
) -> Result<()> {
    let kind = schema_type(schema)?;
    if kind == b"any" {
        return require(
            field(schema, "anyOf")?
                .array()?
                .iter()
                .any(|branch| validate_tool_value_with_policy(branch, value, tool_enums).is_ok()),
            "tool value matches no schema alternative",
        );
    }
    match kind {
        b"string" => {
            require(value.kind() == Kind::String, "tool value is not string")?;
            if tool_enums == ToolEnumValidation::Executor {
                validate_executor_string(value.string()?)?;
            }
            let minimum = schema.find("minLength")?;
            let maximum = schema.find("maxLength")?;
            let pattern = schema.find("pattern")?;
            if minimum.is_some() || maximum.is_some() || pattern.is_some() {
                // JSON Schema counts a string's length, and matches a pattern, in Unicode code points, not bytes.
                let text = value.string()?;
                let (scalars, length) = if pattern.is_some() {
                    let scalars = unicode_scalars(text).map_err(JsonError::from)?;
                    let length = scalars.len();
                    (scalars, length as u64)
                } else {
                    let length = scalar_str(text).map_err(JsonError::from)?.chars().count();
                    (Vec::new(), length as u64)
                };
                if let Some(minimum) = minimum {
                    require(
                        length >= minimum.unsigned_integer()?,
                        "tool string below minLength",
                    )?;
                }
                if let Some(maximum) = maximum {
                    require(
                        length <= maximum.unsigned_integer()?,
                        "tool string above maxLength",
                    )?;
                }
                if let Some(pattern) = pattern {
                    require(
                        compile(pattern.string()?)?.matches(&scalars),
                        "tool string does not match pattern",
                    )?;
                }
            }
        }
        b"boolean" => require(value.kind() == Kind::Boolean, "tool value is not boolean")?,
        b"null" => require(value.is_null(), "tool value is not null")?,
        b"number" | b"integer" => {
            number(value, -1e300, 1e300)?;
            let n = decimal(value)?;
            let safe = decimal(&parse_json(b"9007199254740991")?)?;
            let mut magnitude = n.clone();
            magnitude.negative = false;
            require(
                kind != b"integer"
                    || ((n.digits == b"0" || n.exponent >= 0)
                        && compare(&magnitude, &safe) <= 0
                        && !value.number_lexeme()?.contains(['.', 'e', 'E'])),
                "tool integer must use exact safe-integer syntax",
            )?;
            if let Some(minimum) = schema.find("minimum")? {
                require(
                    compare(&n, &decimal(minimum)?) >= 0,
                    "tool value below minimum",
                )?;
            }
            if let Some(maximum) = schema.find("maximum")? {
                require(
                    compare(&n, &decimal(maximum)?) <= 0,
                    "tool value above maximum",
                )?;
            }
            if let Some(minimum) = schema.find("exclusiveMinimum")? {
                require(
                    compare(&n, &decimal(minimum)?) > 0,
                    "tool value at or below exclusive minimum",
                )?;
            }
            if let Some(maximum) = schema.find("exclusiveMaximum")? {
                require(
                    compare(&n, &decimal(maximum)?) < 0,
                    "tool value at or above exclusive maximum",
                )?;
            }
            if let Some(step) = schema.find("multipleOf")? {
                require(
                    multiple(&n, divisor(step)?),
                    "tool value is not a multiple of multipleOf",
                )?;
            }
        }
        b"array" => {
            let items = value.array()?;
            require(items.len() <= 4096, "tool array bound")?;
            if let Some(minimum) = schema.find("minItems")? {
                require(
                    (items.len() as u64) >= minimum.unsigned_integer()?,
                    "tool array below minItems",
                )?;
            }
            if let Some(maximum) = schema.find("maxItems")? {
                require(
                    (items.len() as u64) <= maximum.unsigned_integer()?,
                    "tool array above maxItems",
                )?;
            }
            for item in items {
                validate_tool_value_with_policy(field(schema, "items")?, item, tool_enums)?;
            }
            // After the items are validated, so every number below has an admitted lexeme and nesting is bounded by the schema.
            if let Some(unique) = schema.find("uniqueItems")?
                && unique.boolean()?
            {
                let mut seen = BTreeSet::new();
                for item in items {
                    let mut key = Vec::new();
                    canonical(item, &mut key)?;
                    require(seen.insert(key), "tool array repeats an item")?;
                }
            }
        }
        _ => {
            let properties = schema.find("properties")?;
            for (name, item) in value.object()? {
                let child = match properties {
                    Some(properties) => properties.find(name)?,
                    None => None,
                };
                let Some(child) = child else {
                    return Err(AdmissionError::Refused("unknown tool property"));
                };
                validate_tool_value_with_policy(child, item, tool_enums)?;
            }
            if let Some(required) = schema.find("required")? {
                for name in required.array()? {
                    require(
                        value.find(name.string()?)?.is_some(),
                        "missing required tool property",
                    )?;
                }
            }
        }
    }
    if tool_enums == ToolEnumValidation::Strict
        && let Some(values) = schema.find("enum")?
    {
        let mut found = false;
        for candidate in values.array()? {
            if candidate.kind() == Kind::Number && value.kind() == Kind::Number {
                found |= compare(&decimal(candidate)?, &decimal(value)?) == 0;
            } else {
                found |= candidate.dump()? == value.dump()?;
            }
        }
        require(found, "tool value outside enum")?;
    }
    // A constant is an enum of one value, so it follows the enum policy.
    if tool_enums == ToolEnumValidation::Strict
        && let Some(expected) = schema.find("const")?
    {
        let same = if expected.kind() == Kind::Number && value.kind() == Kind::Number {
            compare(&decimal(expected)?, &decimal(value)?) == 0
        } else {
            expected.dump()? == value.dump()?
        };
        require(same, "tool value differs from const")?;
    }
    Ok(())
}

/// Checks announced controller sampling modes before rendering or worker submission.
pub fn validate_calibrated_sampling(sampling: &Json, ready: &Json) -> Result<()> {
    if ready.kind() != Kind::Object
        || ready
            .find("native_measured_controller")?
            .is_none_or(|value| value != &Json::Bool(true))
    {
        return Ok(());
    }
    let temperature = number(field(sampling, "temperature")?, 0.0, 100.0)?;
    let mode: &[u8] = if temperature == 0.0 {
        b"greedy".as_slice()
    } else {
        require(
            temperature == 1.0 && number(field(sampling, "min_p")?, 0.0, 1.0)? == 0.0,
            "sampling policy is not calibrated by measured profile",
        )?;
        let top_k = field(sampling, "top_k")?.unsigned_integer()?;
        let top_p = number(field(sampling, "top_p")?, 0.0, 1.0)?;
        if top_k == 20 && top_p == 0.95 {
            b"coding"
        } else {
            require(
                top_p == 1.0 && matches!(top_k, 0 | 20),
                "sampling policy is not calibrated by measured profile",
            )?;
            if top_k == 20 { b"topk" } else { b"full" }
        }
    };
    let modes = field(ready, "native_controller_sampling_modes")?.array()?;
    require(
        !modes.is_empty()
            && modes.len() <= 4
            && modes
                .iter()
                .any(|value| value.string().is_ok_and(|value| value == mode)),
        "sampling mode is not calibrated by measured profile",
    )
}

/// `AdmittedChat`.
#[derive(Clone)]
pub struct AdmittedChat {
    pub render: Json,
    pub sampling: Json,
    pub manifest: Vec<ToolSpec>,
    pub schemas: Vec<Json>,
    pub names: Vec<Vec<u8>>,
    pub seed: Option<u64>,
    pub tool_choice: Vec<u8>,
    pub named_tool_choice: bool,
    pub single_tool_call: bool,
    pub stream: bool,
    pub include_usage: bool,
    pub thinking: bool,
    pub thinking_budget: Option<u32>,
    pub maximum_tokens: u32,
}

impl fmt::Debug for AdmittedChat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Messages, tool schemas and names stay out of debug output.
        formatter
            .debug_struct("AdmittedChat")
            .field("tools", &self.names.len())
            .field("stream", &self.stream)
            .field("thinking", &self.thinking)
            .field("maximum_tokens", &self.maximum_tokens)
            .finish_non_exhaustive()
    }
}

/// `admit_chat`: validates one OpenAI-style request without side effects under strict `enum` enforcement.
pub fn admit_chat(request: &Json, alias: &[u8]) -> Result<AdmittedChat> {
    admit_chat_with_policy(request, alias, ToolEnumValidation::Strict)
}

/// `admit_chat`: validates one OpenAI-style request without side effects under the selected `ToolEnumValidation` policy.
pub fn admit_chat_with_policy(
    request: &Json,
    alias: &[u8],
    tool_enums: ToolEnumValidation,
) -> Result<AdmittedChat> {
    admit_chat_with_policies(request, alias, tool_enums, ToolArgValidation::Strict)
}

/// `admit_chat`: validates one OpenAI-style request without side effects under the selected `ToolEnumValidation` and `ToolArgValidation` policies.
pub fn admit_chat_with_policies(
    request: &Json,
    alias: &[u8],
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
) -> Result<AdmittedChat> {
    admit_chat_with_context(request, alias, tool_enums, tool_args, 65_536)
}

/// How many times longer than the received body a request's canonical dump can be. The writer spells every control character as six bytes (`\u000a`) where a client may send two (`\n`), and nothing else grows.
const DUMP_GROWTH: usize = 3;

/// How many bytes shorter than its dump a client could have sent `value`: JSON has a two-byte escape for `\b`, `\t`, `\n`, `\f` and `\r`, which the writer spells as six. Call after a dump has bounded the nesting.
fn short_escape_savings(value: &Json) -> usize {
    fn text(bytes: &[u8]) -> usize {
        4 * bytes
            .iter()
            .filter(|&&c| matches!(c, 0x08..=0x0a | 0x0c | 0x0d))
            .count()
    }
    match value {
        Json::String(bytes) => text(bytes),
        Json::Array(items) => items.iter().map(short_escape_savings).sum(),
        Json::Object(fields) => fields
            .iter()
            .map(|(name, item)| text(name) + short_escape_savings(item))
            .sum(),
        Json::Null | Json::Bool(_) | Json::Number(_) => 0,
    }
}

/// `admit_chat`: validates one OpenAI-style request without side effects under the selected policies and configured context length.
pub fn admit_chat_with_context(
    request: &Json,
    alias: &[u8],
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
    context_length: u32,
) -> Result<AdmittedChat> {
    require(
        context_length == 65_536 || context_length == 131_072,
        "unsupported serving context length",
    )?;
    let max_request_bytes = if context_length > 65_536 {
        4 * 1024 * 1024
    } else {
        1024 * 1024
    };
    // The request is measured as a client could have sent it, not as the writer spells it: a body within the limit must not be refused because its newlines and tabs dump to six bytes each. The dump still refuses what only a value built in code can hold (invalid UTF-8, a duplicate key, nesting beyond 64).
    let dumped = request.dump_limited(DUMP_GROWTH * max_request_bytes)?;
    require(
        dumped.len() - short_escape_savings(request) <= max_request_bytes,
        "chat request exceeds byte limit",
    )?;
    keys(
        request,
        &[
            "model",
            "messages",
            "n",
            "parallel_tool_calls",
            "stream",
            "stream_options",
            "keep_alive",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "tools",
            "tool_choice",
            "enable_thinking",
            "preserve_thinking",
            "reasoning_effort",
            "thinking_budget",
            "chat_template_kwargs",
            "presence_penalty",
            "repetition_penalty",
        ],
    )?;
    require(
        field(request, "model")?.string()? == alias,
        "unknown model alias",
    )?;
    let mut out = AdmittedChat {
        render: Json::Null,
        sampling: Json::Null,
        manifest: Vec::new(),
        schemas: Vec::new(),
        names: Vec::new(),
        seed: None,
        tool_choice: Vec::new(),
        named_tool_choice: false,
        single_tool_call: false,
        stream: false,
        include_usage: false,
        thinking: true,
        thinking_budget: None,
        maximum_tokens: 8192,
    };
    if let Some(n) = request.find("n")? {
        require(n.unsigned_integer()? == 1, "only n=1 is supported")?;
    }
    if let Some(parallel) = request.find("parallel_tool_calls")? {
        out.single_tool_call = !parallel.boolean()?;
    }
    if let Some(p) = request.find("stream")? {
        out.stream = p.boolean()?;
    }
    if let Some(options) = request.find("stream_options")? {
        require(out.stream, "stream_options requires streaming")?;
        keys(options, &["include_usage"])?;
        if let Some(include) = options.find("include_usage")? {
            out.include_usage = include.boolean()?;
        }
    }
    // Numeric lifetime hints do not change the owned worker's supervised lifetime.
    if let Some(keep) = request.find("keep_alive")? {
        if keep.kind() == Kind::String {
            let value = keep.string()?;
            let numeric = value
                .strip_suffix(b"ms")
                .or_else(|| value.strip_suffix(b"s"))
                .or_else(|| value.strip_suffix(b"m"))
                .or_else(|| value.strip_suffix(b"h"))
                .unwrap_or(value);
            require(
                !value.is_empty()
                    && value.len() <= 64
                    && (value == b"-1"
                        || (!numeric.is_empty()
                            && numeric.iter().all(|c| c.is_ascii_digit() || *c == b'.')
                            && numeric.iter().filter(|c| **c == b'.').count() <= 1
                            && numeric.iter().any(u8::is_ascii_digit))),
                "invalid keep_alive duration hint",
            )?;
        } else {
            number(keep, -1.0, 1e12)?;
        }
    }
    let template_options = request.find("chat_template_kwargs")?;
    if let Some(template) = template_options {
        keys(
            template,
            &[
                "enable_thinking",
                "preserve_thinking",
                "reasoning_effort",
                "thinking_budget",
            ],
        )?;
    }
    let mut canonical_options = Vec::new();
    for name in ["enable_thinking", "preserve_thinking", "reasoning_effort"] {
        let outer = request.find(name)?;
        let inner = match template_options {
            Some(template) => template.find(name)?,
            None => None,
        };
        if name == "reasoning_effort" {
            if let Some(p) = outer {
                p.string()?;
            }
            if let Some(p) = inner {
                p.string()?;
            }
        } else {
            if let Some(p) = outer {
                p.boolean()?;
            }
            if let Some(p) = inner {
                p.boolean()?;
            }
        }
        if let (Some(a), Some(b)) = (outer, inner) {
            require(a.dump()? == b.dump()?, "conflicting chat template options")?;
        }
        if let Some(selected) = outer.or(inner) {
            canonical_options.push((name.as_bytes().to_vec(), selected.clone()));
        }
    }
    let outer_budget = request.find("thinking_budget")?;
    let inner_budget = match template_options {
        Some(template) => template.find("thinking_budget")?,
        None => None,
    };
    let mut raw_thinking_budget = None;
    if let Some(p) = outer_budget {
        let n = p.unsigned_integer()?;
        require(n <= 32768, "thinking budget must be 0..32768")?;
        raw_thinking_budget = Some(n as u32);
    }
    if let Some(p) = inner_budget {
        let n = p.unsigned_integer()?;
        require(n <= 32768, "thinking budget must be 0..32768")?;
        if let Some(outer_val) = raw_thinking_budget {
            require(outer_val == n as u32, "conflicting chat template options")?;
        }
        raw_thinking_budget = Some(n as u32);
    }
    let options = Json::Object(canonical_options);
    let mut explicit_thinking = None;
    if let Some(p) = options.find("enable_thinking")? {
        let val = p.boolean()?;
        explicit_thinking = Some(val);
        out.thinking = val;
    }
    if let Some(budget) = raw_thinking_budget {
        if budget == 0 {
            require(
                explicit_thinking != Some(true),
                "conflicting thinking options",
            )?;
            out.thinking = false;
        } else if explicit_thinking != Some(false) {
            // A positive budget only caps thinking. With thinking explicitly off there is nothing to cap, so the budget is ignored and not refused: clients keep a fixed budget and toggle thinking.
            out.thinking_budget = Some(budget);
        }
    }
    if let Some(p) = options.find("reasoning_effort")? {
        let effort = p.string()?;
        require(
            effort == b"xhigh"
                || effort == b"high"
                || effort == b"medium"
                || effort == b"low"
                || (!out.thinking && effort == b"none"),
            "unsupported reasoning effort",
        )?;
    }
    require(
        !(request.find("max_tokens")?.is_some()
            && request.find("max_completion_tokens")?.is_some()),
        "provide only one output limit",
    )?;
    let limit = match request.find("max_tokens")? {
        Some(limit) => Some(limit),
        None => request.find("max_completion_tokens")?,
    };
    if let Some(limit) = limit {
        let n = limit.unsigned_integer()?;
        require(n != 0 && n <= 32768, "output limit must be 1..32768")?;
        out.maximum_tokens = n as u32;
    }
    for name in ["presence_penalty", "repetition_penalty"] {
        if let Some(p) = request.find(name)? {
            let neutral = parse_json(if name == "presence_penalty" {
                b"0"
            } else {
                b"1"
            })?;
            require(
                compare(&decimal(p)?, &decimal(&neutral)?) == 0,
                "non-neutral sampling penalty is unsupported",
            )?;
        }
    }
    let temperature = match request.find("temperature")? {
        Some(value) => number(value, 0.0, 100.0)?,
        None => 0.0,
    };
    let top_p = match request.find("top_p")? {
        Some(value) => number(value, 0.0, 1.0)?,
        None => 1.0,
    };
    require(top_p > 0.0, "top_p must be positive")?;
    let min_p = match request.find("min_p")? {
        Some(value) => number(value, 0.0, 1.0)?,
        None => 0.0,
    };
    let top_k = match request.find("top_k")? {
        Some(value) => value.unsigned_integer()?,
        None => 0,
    };
    require(top_k <= 248_320, "top_k bound")?;
    out.sampling = Json::new_object([
        ("temperature", real(temperature)?),
        ("top_k", Json::integer(top_k)),
        ("top_p", real(top_p)?),
        ("min_p", real(min_p)?),
    ]);
    if let Some(seed) = request.find("seed")? {
        out.seed = Some(seed.unsigned_integer()?);
    }
    let mut declarations: Vec<Json> = Vec::new();
    let mut names: BTreeSet<Vec<u8>> = BTreeSet::new();
    if let Some(tools) = request.find("tools")? {
        require(tools.array()?.len() <= 64, "API tool limit")?;
        for declaration in tools.array()? {
            keys(declaration, &["type", "function"])?;
            require(
                field(declaration, "type")?.string()? == b"function",
                "unsupported tool type",
            )?;
            let function = field(declaration, "function")?;
            keys(function, &["name", "description", "parameters", "strict"])?;
            let name = field(function, "name")?.string()?.to_vec();
            safe_name(&name, true)?;
            require(names.insert(name.clone()), "duplicate tool name")?;
            if let Some(description) = function.find("description")? {
                require(
                    description.string()?.len() <= 16384,
                    "tool description bound",
                )?;
            }
            if let Some(strict) = function.find("strict")? {
                strict.boolean()?;
            }
            let schema = match function.find("parameters")? {
                Some(parameters) => parameters.clone(),
                None => parse_json(
                    br#"{"type":"object","properties":{},"additionalProperties":false}"#,
                )?,
            };
            let mut nodes = 0;
            schema_admit(&schema, 0, &mut nodes)?;
            require(
                schema_type(&schema)? == b"object",
                "function parameters must be object",
            )?;
            let mut manifest = ToolSpec {
                name: name.clone(),
                parameters: Vec::new(),
            };
            let mut required: BTreeSet<Vec<u8>> = BTreeSet::new();
            if let Some(p) = schema.find("required")? {
                for value in p.array()? {
                    required.insert(value.string()?.to_vec());
                }
            }
            if let Some(p) = schema.find("properties")? {
                for (key, value) in p.object()? {
                    manifest.parameters.push(ParameterSpec {
                        name: key.clone(),
                        r#type: parameter_type(schema_type(value)?),
                        required: required.contains(key),
                    });
                }
            }
            out.manifest.push(manifest);
            out.schemas.push(schema);
            out.names.push(name);
            declarations.push(declaration.clone());
        }
    }
    out.tool_choice = b"auto".to_vec();
    if let Some(choice) = request.find("tool_choice")? {
        if choice.kind() == Kind::String {
            out.tool_choice = choice.string()?.to_vec();
            require(
                matches!(out.tool_choice.as_slice(), b"auto" | b"none" | b"required"),
                "unsupported tool_choice",
            )?;
        } else {
            out.named_tool_choice = true;
            keys(choice, &["type", "function"])?;
            require(
                field(choice, "type")?.string()? == b"function",
                "unsupported named tool_choice",
            )?;
            let function = field(choice, "function")?;
            keys(function, &["name"])?;
            out.tool_choice = field(function, "name")?.string()?.to_vec();
            require(
                names.contains(&out.tool_choice),
                "unknown named tool_choice",
            )?;
        }
    }
    let choice = out.tool_choice.clone();
    require(
        choice == b"auto" || choice == b"none" || !declarations.is_empty(),
        "tool choice requires tools",
    )?;
    let messages = field(request, "messages")?;
    history(
        messages,
        &out.names,
        &out.schemas,
        tool_enums,
        tool_args,
        max_request_bytes,
    )?;
    let mut rendered_messages = messages.array()?.clone();
    if out.named_tool_choice || choice == b"required" {
        let instruction = if !out.named_tool_choice {
            b"For this response, call at least one of the declared tools. Do not answer without a tool call.".to_vec()
        } else {
            let mut text = b"For this response, call only the declared function named ".to_vec();
            text.extend_from_slice(&choice);
            text.extend_from_slice(b". Do not answer without that tool call.");
            text
        };
        rendered_messages.insert(
            0,
            Json::new_object([
                ("role", Json::from("system")),
                ("content", Json::String(instruction)),
            ]),
        );
    }
    if !out.named_tool_choice && choice == b"none" {
        declarations.clear();
        out.manifest.clear();
        out.schemas.clear();
        out.names.clear();
    } else if out.named_tool_choice {
        let index = out
            .names
            .iter()
            .position(|name| *name == choice)
            .unwrap_or(out.names.len());
        let (Some(declaration), Some(manifest), Some(schema), Some(name)) = (
            declarations.get(index).cloned(),
            out.manifest.get(index).cloned(),
            out.schemas.get(index).cloned(),
            out.names.get(index).cloned(),
        ) else {
            return Err(AdmissionError::Refused("unknown named tool_choice"));
        };
        declarations = vec![declaration];
        out.manifest = vec![manifest];
        out.schemas = vec![schema];
        out.names = vec![name];
    }
    let mut render = vec![
        (b"messages".to_vec(), Json::Array(rendered_messages)),
        (b"tools".to_vec(), Json::Array(declarations)),
        (b"enable_thinking".to_vec(), Json::Bool(out.thinking)),
    ];
    for name in ["preserve_thinking", "reasoning_effort"] {
        if let Some(p) = options.find(name)? {
            render.push((name.as_bytes().to_vec(), p.clone()));
        }
    }
    out.render = Json::Object(render);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexeme(text: &str) -> Json {
        parse_json(text.as_bytes()).unwrap()
    }

    #[test]
    fn decimal_normalizes_zeros_signs_and_exponents() {
        let d = decimal(&lexeme("-12.3400e2")).unwrap();
        assert_eq!(
            d,
            Decimal {
                negative: true,
                digits: b"1234".to_vec(),
                exponent: 0
            }
        );
        assert_eq!(
            decimal(&lexeme("-0.000")).unwrap(),
            Decimal {
                negative: false,
                digits: b"0".to_vec(),
                exponent: 0
            }
        );
        assert_eq!(
            decimal(&lexeme("1e1000001")),
            Err(AdmissionError::Refused("tool numeric exponent bound"))
        );
    }

    #[test]
    fn compare_orders_by_magnitude_then_digits() {
        let parse = |text: &str| decimal(&lexeme(text)).unwrap();
        assert_eq!(compare(&parse("10"), &parse("9.99")), 1);
        assert_eq!(compare(&parse("-10"), &parse("-9.99")), -1);
        assert_eq!(compare(&parse("1.50"), &parse("15e-1")), 0);
        assert_eq!(compare(&parse("0"), &parse("-0.0")), 0);
        assert_eq!(compare(&parse("-1"), &parse("0")), -1);
    }

    #[test]
    fn opencode_schema_dialect_and_exclusive_bound_are_checked() {
        let schema = lexeme(
            r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"timeout":{"type":"number","exclusiveMinimum":0,"maximum":60}},"required":["timeout"]}"#,
        );
        schema_admit(&schema, 0, &mut 0).unwrap();
        for value in ["0.0001", "1", "60"] {
            let value = lexeme(&format!(r#"{{"timeout":{value}}}"#));
            validate_tool_value(&schema, &value).unwrap();
        }
        for value in ["0", "-0.0", "-1", "60.01"] {
            let value = lexeme(&format!(r#"{{"timeout":{value}}}"#));
            assert!(validate_tool_value(&schema, &value).is_err());
        }
        for text in [
            r#"{"type":"number","exclusiveMinimum":true}"#,
            r#"{"type":"string","exclusiveMinimum":0}"#,
            r#"{"type":"number","exclusiveMinimum":1,"maximum":1}"#,
            r#"{"type":"object","$schema":"https://example.invalid/schema"}"#,
            r#"{"type":"object","$schema":12}"#,
            r#"{"type":"object","$schema":"https://json-schema.org/draft/2020-12/schema","$ref":"remote"}"#,
        ] {
            assert!(schema_admit(&lexeme(text), 0, &mut 0).is_err());
        }
        assert!(schema_admit(&schema, 1, &mut 0).is_err());
    }

    #[test]
    fn array_schema_min_items_and_max_items_are_checked() {
        let schema = lexeme(
            r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3}},"required":["tags"]}"#,
        );
        schema_admit(&schema, 0, &mut 0).unwrap();
        for value in [r#"["a"]"#, r#"["a","b"]"#, r#"["a","b","c"]"#] {
            let value = lexeme(&format!(r#"{{"tags":{value}}}"#));
            validate_tool_value(&schema, &value).unwrap();
        }
        for value in [r#"[]"#, r#"["a","b","c","d"]"#] {
            let value = lexeme(&format!(r#"{{"tags":{value}}}"#));
            assert!(validate_tool_value(&schema, &value).is_err());
        }
        let exact_zero =
            lexeme(r#"{"type":"array","items":{"type":"integer"},"minItems":0,"maxItems":0}"#);
        schema_admit(&exact_zero, 0, &mut 0).unwrap();
        validate_tool_value(&exact_zero, &lexeme("[]")).unwrap();
        assert!(validate_tool_value(&exact_zero, &lexeme("[1]")).is_err());

        // The minimum is bounded by the 4096-item validation cap; the maximum is not, because the cap still holds.
        for text in [
            r#"{"type":"array","items":{"type":"string"},"minItems":4096}"#,
            r#"{"type":"array","items":{"type":"string"},"maxItems":4097}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":4096,"maxItems":18446744073709551615}"#,
        ] {
            schema_admit(&lexeme(text), 0, &mut 0).unwrap();
        }
        assert_eq!(
            schema_admit(
                &lexeme(r#"{"type":"array","items":{"type":"string"},"minItems":4097}"#),
                0,
                &mut 0
            ),
            Err(AdmissionError::Refused(
                "minItems exceeds the tool array bound"
            ))
        );
        assert_eq!(
            schema_admit(
                &lexeme(
                    r#"{"type":"array","items":{"type":"string"},"minItems":4097,"maxItems":2}"#
                ),
                0,
                &mut 0
            ),
            Err(AdmissionError::Refused(
                "minItems exceeds the tool array bound"
            ))
        );

        for text in [
            r#"{"type":"array","items":{"type":"string"},"minItems":4097}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":18446744073709551616}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":-1}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":1.0}"#,
            r#"{"type":"array","items":{"type":"string"},"maxItems":"2"}"#,
            r#"{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2}"#,
            r#"{"type":"string","minItems":1}"#,
            r#"{"type":"object","properties":{},"maxItems":2}"#,
            r#"{"anyOf":[{"type":"array","items":{"type":"string"}}],"minItems":1}"#,
        ] {
            assert!(
                schema_admit(&lexeme(text), 0, &mut 0).is_err(),
                "expected refusal for {text}"
            );
        }
    }

    /// The refusal text of admitting one schema, or empty when it is admitted.
    fn admitted(text: &str) -> String {
        match schema_admit(&lexeme(text), 0, &mut 0) {
            Ok(()) => String::new(),
            Err(error) => error.to_string(),
        }
    }

    /// The refusal text of validating one value, or empty when it is valid.
    fn validated(schema: &str, value: &str) -> String {
        match validate_tool_value(&lexeme(schema), &lexeme(value)) {
            Ok(()) => String::new(),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn exclusive_maximum_mirrors_exclusive_minimum() {
        let schema = r#"{"type":"number","minimum":0,"exclusiveMaximum":1}"#;
        assert_eq!(admitted(schema), "");
        for value in ["0", "-0.0", "0.5", "9.9e-1", "0.99999999999999999999"] {
            assert_eq!(validated(schema, value), "", "{value}");
        }
        for value in ["1", "1.0", "1e0", "1.00000000000000000001", "2"] {
            assert_eq!(
                validated(schema, value),
                "tool value at or above exclusive maximum",
                "{value}"
            );
        }
        assert_eq!(validated(schema, "-0.1"), "tool value below minimum");
        let open = r#"{"type":"integer","exclusiveMinimum":1,"exclusiveMaximum":4}"#;
        assert_eq!(admitted(open), "");
        assert_eq!(validated(open, "2"), "");
        assert_eq!(validated(open, "3"), "");
        assert_eq!(
            validated(open, "1"),
            "tool value at or below exclusive minimum"
        );
        assert_eq!(
            validated(open, "4"),
            "tool value at or above exclusive maximum"
        );

        // An excluded upper bound leaves nothing when a lower bound reaches it, inclusive or not.
        for (text, reason) in [
            (
                r#"{"type":"number","minimum":1,"exclusiveMaximum":1}"#,
                "inverted schema bounds",
            ),
            (
                r#"{"type":"number","minimum":2,"exclusiveMaximum":1}"#,
                "inverted schema bounds",
            ),
            (
                r#"{"type":"number","exclusiveMinimum":1,"exclusiveMaximum":1}"#,
                "inverted schema bounds",
            ),
            (
                r#"{"type":"number","exclusiveMinimum":2,"exclusiveMaximum":1}"#,
                "inverted schema bounds",
            ),
            (
                r#"{"type":"number","exclusiveMaximum":true}"#,
                "expected JSON number",
            ),
            (
                r#"{"type":"number","exclusiveMaximum":1e301}"#,
                "API number outside supported range",
            ),
            (
                r#"{"type":"string","exclusiveMaximum":0}"#,
                "numeric bound on other schema type",
            ),
            (
                r#"{"anyOf":[{"type":"number"}],"exclusiveMaximum":1}"#,
                "unsupported API/schema field",
            ),
        ] {
            assert_eq!(admitted(text), reason, "{text}");
        }
        // These bounds are one step apart in the nineteenth decimal place: the same double, ordered as decimals.
        for text in [
            r#"{"type":"number","minimum":1,"exclusiveMaximum":1.0000000000000000001}"#,
            r#"{"type":"number","exclusiveMinimum":1,"exclusiveMaximum":1.0000000000000000001}"#,
            r#"{"type":"number","maximum":5,"exclusiveMaximum":5}"#,
        ] {
            assert_eq!(admitted(text), "", "{text}");
        }
    }

    #[test]
    fn string_length_is_counted_in_code_points() {
        let schema = r#"{"type":"string","minLength":2,"maxLength":3}"#;
        assert_eq!(admitted(schema), "");
        // Two and three code points, whether each takes one, two, three or four bytes, and however it is escaped. A letter with a combining accent is two.
        for value in [
            r#""ab""#,
            r#""abc""#,
            r#""éé""#,
            r#""日本語""#,
            r#""😀😀""#,
            r#""😀😀😀""#,
            r#""ab""#,
            r#""é""#,
        ] {
            assert_eq!(validated(schema, value), "", "{value}");
        }
        // One code point each, although the last two are four bytes and two UTF-16 units.
        for value in [r#""""#, r#""a""#, r#""é""#, r#""😀""#, r#""😀""#] {
            assert_eq!(
                validated(schema, value),
                "tool string below minLength",
                "{value}"
            );
        }
        for value in [r#""abcd""#, r#""日本語x""#, r#""😀😀😀😀""#] {
            assert_eq!(
                validated(schema, value),
                "tool string above maxLength",
                "{value}"
            );
        }
        assert_eq!(validated(schema, "5"), "tool value is not string");
        let empty = r#"{"type":"string","minLength":0,"maxLength":0}"#;
        assert_eq!(admitted(empty), "");
        assert_eq!(validated(empty, r#""""#), "");
        assert_eq!(validated(empty, r#""a""#), "tool string above maxLength");
        // A value built in memory can hold bytes that are not UTF-8. Its length cannot be counted, so it is refused as the JSON codec refuses it.
        assert_eq!(
            validate_tool_value(&lexeme(schema), &Json::String(vec![b'a', 0xff]))
                .unwrap_err()
                .to_string(),
            "invalid UTF-8 input"
        );

        // The minimum is bounded by the 64 KiB a generated call can hold; the maximum is not, because it only fails to constrain.
        for text in [
            r#"{"type":"string","minLength":65536}"#,
            r#"{"type":"string","maxLength":65537}"#,
            r#"{"type":"string","minLength":65536,"maxLength":18446744073709551615}"#,
            r#"{"type":"string","enum":["abc"],"minLength":3}"#,
        ] {
            assert_eq!(admitted(text), "", "{text}");
        }
        for (text, reason) in [
            (
                r#"{"type":"string","minLength":65537}"#,
                "minLength exceeds the tool call byte limit",
            ),
            (
                r#"{"type":"string","minLength":65537,"maxLength":2}"#,
                "minLength exceeds the tool call byte limit",
            ),
            (
                r#"{"type":"string","minLength":3,"maxLength":2}"#,
                "inverted schema bounds",
            ),
            (
                r#"{"type":"string","minLength":-1}"#,
                "JSON unsigned integer out of range or nonintegral",
            ),
            (
                r#"{"type":"string","minLength":1.0}"#,
                "JSON unsigned integer out of range or nonintegral",
            ),
            (
                r#"{"type":"string","minLength":18446744073709551616}"#,
                "JSON unsigned integer out of range or nonintegral",
            ),
            (
                r#"{"type":"string","maxLength":"2"}"#,
                "expected JSON unsigned integer",
            ),
            (
                r#"{"type":"integer","minLength":1}"#,
                "string schema keyword on other type",
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"maxLength":2}"#,
                "string schema keyword on other type",
            ),
            (
                r#"{"anyOf":[{"type":"string"}],"minLength":1}"#,
                "unsupported API/schema field",
            ),
            (
                r#"{"type":"string","enum":["ab"],"minLength":3}"#,
                "tool string below minLength",
            ),
        ] {
            assert_eq!(admitted(text), reason, "{text}");
        }
    }

    #[test]
    fn unique_items_compare_as_json_schema_equality() {
        let key = |text: &str| {
            let mut out = Vec::new();
            canonical(&lexeme(text), &mut out).unwrap();
            String::from_utf8(out).unwrap()
        };
        assert_eq!(key("1.50"), "15e-1");
        assert_eq!(key("-1.5e3"), "-15e2");
        assert_eq!(key("-0.0"), "0e0");
        assert_eq!(
            key(r#"{"b":[1.0,"1"],"a":null}"#),
            r#"{"a":null,"b":[1e0,"1",],}"#
        );
        assert_eq!(
            key(r#"{"b":[1.0,"1"],"a":null}"#),
            key(r#"{"a":null,"b":[1,"1"]}"#)
        );
        // A number never shares a form with a string, a literal or an array that spells it.
        for (left, right) in [
            ("1", r#""1e0""#),
            ("0", "false"),
            ("[1,2]", "[2,1]"),
            ("[[1],2]", "[[1,2]]"),
            (r#"["a,b"]"#, r#"["a","b"]"#),
            (r#"{"a":{"b":1}}"#, r#"{"a":{},"b":1}"#),
        ] {
            assert_ne!(key(left), key(right), "{left} {right}");
        }

        let numbers = r#"{"type":"array","items":{"type":"number"},"uniqueItems":true}"#;
        assert_eq!(admitted(numbers), "");
        for value in ["[]", "[1]", "[1,2,3]", "[1,1.00000000000000000001]"] {
            assert_eq!(validated(numbers, value), "", "{value}");
        }
        for value in [
            "[1,1]",
            "[1,2,1]",
            "[1,1.0]",
            "[10,1e1]",
            "[0,-0.0]",
            "[1.5,15e-1]",
        ] {
            assert_eq!(
                validated(numbers, value),
                "tool array repeats an item",
                "{value}"
            );
        }
        // Items are validated first, so a wrong item is reported before a repeat.
        assert_eq!(validated(numbers, r#"[1,1,"x"]"#), "expected JSON number");
        let objects = r#"{"type":"array","items":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"string"}}},"uniqueItems":true}"#;
        assert_eq!(
            validated(objects, r#"[{"a":1,"b":"x"},{"a":1,"b":"y"}]"#),
            ""
        );
        assert_eq!(
            validated(objects, r#"[{"a":1,"b":"x"},{"b":"x","a":1.0}]"#),
            "tool array repeats an item"
        );
        let repeats = r#"{"type":"array","items":{"type":"number"},"uniqueItems":false}"#;
        assert_eq!(admitted(repeats), "");
        assert_eq!(validated(repeats, "[1,1]"), "");
        // The largest array validation accepts stays unique item by item.
        let many = format!(
            "[{}]",
            (0..4096)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        assert_eq!(validated(numbers, &many), "");
        assert_eq!(
            validated(numbers, &many.replacen("[0,", "[4095.0,", 1)),
            "tool array repeats an item"
        );

        for (text, reason) in [
            (
                r#"{"type":"array","items":{"type":"string"},"uniqueItems":1}"#,
                "expected JSON boolean",
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"uniqueItems":"true"}"#,
                "expected JSON boolean",
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2,"uniqueItems":1}"#,
                "inverted schema bounds",
            ),
            (
                r#"{"type":"string","uniqueItems":true}"#,
                "array schema keyword on other type",
            ),
            (
                r#"{"type":"object","properties":{},"uniqueItems":false}"#,
                "array schema keyword on other type",
            ),
            (
                r#"{"anyOf":[{"type":"array","items":{"type":"string"}}],"uniqueItems":true}"#,
                "unsupported API/schema field",
            ),
        ] {
            assert_eq!(admitted(text), reason, "{text}");
        }
    }

    #[test]
    fn constant_is_an_enum_of_one_value() {
        let text = r#"{"type":"string","const":"a"}"#;
        assert_eq!(admitted(text), "");
        assert_eq!(validated(text, r#""a""#), "");
        assert_eq!(validated(text, r#""a""#), "");
        assert_eq!(validated(text, r#""b""#), "tool value differs from const");
        assert_eq!(validated(text, "1"), "tool value is not string");
        let number = r#"{"type":"number","const":1.50}"#;
        assert_eq!(admitted(number), "");
        assert_eq!(validated(number, "1.5"), "");
        assert_eq!(validated(number, "15e-1"), "");
        assert_eq!(
            validated(number, "1.50000000000000000001"),
            "tool value differs from const"
        );
        for (schema, value) in [
            (r#"{"type":"integer","const":0}"#, "-0"),
            (r#"{"type":"boolean","const":true}"#, "true"),
            (r#"{"type":"null","const":null}"#, "null"),
            (r#"{"type":"string","enum":["a"],"const":"a"}"#, r#""a""#),
        ] {
            assert_eq!(admitted(schema), "", "{schema}");
            assert_eq!(validated(schema, value), "", "{schema}");
        }
        // The enum policy defers the constant with the enum; the type is still checked.
        let schema = lexeme(text);
        let deferred = ToolEnumValidation::Executor;
        assert_eq!(
            validate_tool_value_with_policy(&schema, &lexeme(r#""b""#), deferred),
            Ok(())
        );
        assert_eq!(
            validate_tool_value_with_policy(&schema, &lexeme("1"), deferred),
            Err(AdmissionError::Refused("tool value is not string"))
        );

        // The constant must itself satisfy the rest of the schema, as each enum member must.
        for (text, reason) in [
            (r#"{"type":"string","const":1}"#, "tool value is not string"),
            (
                r#"{"type":"integer","const":1.0}"#,
                "tool integer must use exact safe-integer syntax",
            ),
            (
                r#"{"type":"integer","const":5,"maximum":3}"#,
                "tool value above maximum",
            ),
            (
                r#"{"type":"string","const":"abc","maxLength":2}"#,
                "tool string above maxLength",
            ),
            (
                r#"{"type":"string","enum":["a","b"],"const":"a"}"#,
                "tool value differs from const",
            ),
            (
                r#"{"type":"object","properties":{},"const":{}}"#,
                "unsupported const",
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"const":[]}"#,
                "unsupported const",
            ),
            (
                r#"{"anyOf":[{"type":"string"}],"const":"a"}"#,
                "unsupported API/schema field",
            ),
        ] {
            assert_eq!(admitted(text), reason, "{text}");
        }
    }

    #[test]
    fn multiple_of_is_exact_decimal_division() {
        let is = |value: &str, step: &str| {
            multiple(
                &decimal(&lexeme(value)).unwrap(),
                divisor(&lexeme(step)).unwrap(),
            )
        };
        for (value, step) in [
            ("0", "7"),
            ("-10", "5"),
            ("1.23", "0.01"),
            // 0.3 / 0.1 is not an integer in binary floating point.
            ("0.3", "0.1"),
            ("3", "1.5"),
            ("3e1", "1.5"),
            ("1e1", "2.5"),
            ("1e3", "8"),
            ("1e300", "8"),
            ("3e299", "3"),
            // 2^53 + 1 does not survive a double.
            ("9007199254740993", "3"),
            // The largest powers of two and five a divisor can hold.
            ("1e59", "576460752303423488"),
            ("1e25", "298023223876953125"),
            // The largest divisor: every remainder step stays inside u64.
            ("122999999999999999877", "999999999999999999"),
            ("999999999999999999e40", "999999999999999999"),
        ] {
            assert!(is(value, step), "{value} {step}");
        }
        for (value, step) in [
            ("7", "5"),
            ("1.234", "0.01"),
            ("1e-3", "0.01"),
            ("0.30000000000000004", "0.1"),
            ("1", "1.5"),
            ("1e1", "1.5"),
            ("1e2", "8"),
            ("1e300", "3"),
            ("9007199254740992", "3"),
            ("1e58", "576460752303423488"),
            ("1e24", "298023223876953125"),
            ("999999999999999998999999999999999998", "999999999999999999"),
        ] {
            assert!(!is(value, step), "{value} {step}");
        }
        assert_eq!(
            divisor(&lexeme("1.50")),
            Ok((15, -1)),
            "trailing zeros are not significant"
        );
        assert_eq!(divisor(&lexeme("1234567890123456780e3")).unwrap().1, 4);

        let schema = r#"{"type":"integer","minimum":0,"exclusiveMaximum":100,"multipleOf":25}"#;
        assert_eq!(admitted(schema), "");
        assert_eq!(validated(schema, "75"), "");
        assert_eq!(
            validated(schema, "30"),
            "tool value is not a multiple of multipleOf"
        );
        assert_eq!(
            validated(schema, "100"),
            "tool value at or above exclusive maximum"
        );
        for (text, reason) in [
            (
                r#"{"type":"number","multipleOf":0}"#,
                "multipleOf must be positive",
            ),
            (
                r#"{"type":"number","multipleOf":-2}"#,
                "multipleOf must be positive",
            ),
            (
                r#"{"type":"number","multipleOf":1234567890123456789}"#,
                "multipleOf digit bound",
            ),
            (
                r#"{"type":"number","multipleOf":"2"}"#,
                "expected JSON number",
            ),
            (
                r#"{"type":"number","multipleOf":1e301}"#,
                "API number outside supported range",
            ),
            (
                r#"{"type":"string","multipleOf":2}"#,
                "numeric bound on other schema type",
            ),
            (
                r#"{"anyOf":[{"type":"integer"}],"multipleOf":2}"#,
                "unsupported API/schema field",
            ),
            (
                r#"{"type":"integer","enum":[4,5],"multipleOf":2}"#,
                "tool value is not a multiple of multipleOf",
            ),
            // The enum members are validated before the divisor is admitted, so validation refuses a zero divisor itself and never divides by it.
            (
                r#"{"type":"integer","enum":[4],"multipleOf":0}"#,
                "multipleOf must be positive",
            ),
        ] {
            assert_eq!(admitted(text), reason, "{text}");
        }
    }

    #[test]
    fn pattern_is_enforced_on_strings() {
        let schema = r#"{"type":"string","pattern":"^[a-z]+(?:-[a-z]+)*$","maxLength":8}"#;
        assert_eq!(admitted(schema), "");
        for value in [r#""a""#, r#""ab-cd""#] {
            assert_eq!(validated(schema, value), "", "{value}");
        }
        for value in [r#""""#, r#""a-""#, r#""A""#, r#""a b""#, r#""a\nb""#] {
            assert_eq!(
                validated(schema, value),
                "tool string does not match pattern",
                "{value}"
            );
        }
        // The length bounds are checked first.
        assert_eq!(
            validated(schema, r#""abcdefghi""#),
            "tool string above maxLength"
        );
        assert_eq!(validated(schema, "1"), "tool value is not string");
        // A pattern is not anchored unless it says so.
        let anywhere = r#"{"type":"string","pattern":"b"}"#;
        assert_eq!(validated(anywhere, r#""abc""#), "");
        assert_eq!(
            validated(anywhere, r#""ac""#),
            "tool string does not match pattern"
        );
        // A character is one code point, as it is for the length keywords.
        let single = r#"{"type":"string","pattern":"^.$"}"#;
        assert_eq!(validated(single, r#""\ud83d\ude00""#), "");
        assert_eq!(
            validated(single, r#""e\u0301""#),
            "tool string does not match pattern"
        );
    }

    #[test]
    fn pattern_is_admitted_only_inside_the_supported_grammar() {
        for text in [
            r#"{"type":"string","pattern":""}"#,
            r#"{"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"}"#,
            r#"{"type":"string","pattern":"^https\\:\\/\\/"}"#,
            r#"{"type":"string","enum":["ab","ac"],"pattern":"^a"}"#,
            r#"{"type":"string","const":"ab","pattern":"^a"}"#,
            r#"{"type":"array","items":{"type":"string","pattern":"^a"}}"#,
            r#"{"anyOf":[{"type":"number"},{"type":"string","pattern":"^\\d+$"}]}"#,
            r#"{"type":"string","pattern":"a{4095}"}"#,
        ] {
            assert_eq!(admitted(text), "", "{text}");
        }
        for (text, reason) in [
            (
                r#"{"type":"string","pattern":"(a)\\1"}"#,
                "unsupported pattern",
            ),
            (
                r#"{"type":"string","pattern":"(?<n>a)"}"#,
                "unsupported pattern",
            ),
            (
                r#"{"type":"string","pattern":"\\p{L}"}"#,
                "unsupported pattern",
            ),
            (r#"{"type":"string","pattern":"a{"}"#, "unsupported pattern"),
            (
                r#"{"type":"string","pattern":"a{4096}"}"#,
                "pattern exceeds the tool pattern bound",
            ),
            (r#"{"type":"string","pattern":7}"#, "expected JSON string"),
            (
                r#"{"type":"integer","pattern":"^1"}"#,
                "string schema keyword on other type",
            ),
            (
                r#"{"anyOf":[{"type":"string"}],"pattern":"a"}"#,
                "unsupported API/schema field",
            ),
            // Enum members and a constant are checked against the pattern, and the pattern is read before them.
            (
                r#"{"type":"string","enum":["ab","b"],"pattern":"^a"}"#,
                "tool string does not match pattern",
            ),
            (
                r#"{"type":"string","const":"b","pattern":"^a"}"#,
                "tool string does not match pattern",
            ),
            (
                r#"{"type":"string","enum":[1],"pattern":"("}"#,
                "unsupported pattern",
            ),
            (
                r#"{"type":"string","minLength":3,"maxLength":2,"pattern":"("}"#,
                "inverted schema bounds",
            ),
        ] {
            assert_eq!(admitted(text), reason, "{text}");
        }
    }

    #[test]
    fn sampling_values_keep_the_cpp_to_chars_lexemes() {
        let admitted = admit_chat(
            &lexeme(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":0.00001,"top_p":0.7}"#),
            b"m",
        )
        .unwrap();
        assert_eq!(
            admitted.sampling.dump().unwrap(),
            br#"{"temperature":1e-05,"top_k":0,"top_p":0.7,"min_p":0}"#
        );
    }

    #[test]
    fn admitted_chat_debug_counts_tools_without_messages_or_names() {
        // Arrange
        let request = lexeme(
            r#"{"model":"m","messages":[{"role":"user","content":"private prompt"}],"tools":[{"type":"function","function":{"name":"private_tool","parameters":{"type":"object","properties":{}}}}],"max_tokens":7}"#,
        );

        // Act
        let rendered = format!("{:?}", admit_chat(&request, b"m").unwrap());

        // Assert
        assert_eq!(
            rendered,
            "AdmittedChat { tools: 1, stream: false, thinking: true, maximum_tokens: 7, .. }"
        );
        assert!(!rendered.contains("private"));
    }

    /// A chat request whose one message is `escapes` two-byte JSON escapes of `control`, as a client would send it.
    fn escaped_body(control: &[u8; 2], escapes: usize) -> Vec<u8> {
        let mut body = br#"{"model":"m","messages":[{"role":"user","content":""#.to_vec();
        for _ in 0..escapes {
            body.extend_from_slice(control);
        }
        body.extend_from_slice(br#""}]}"#);
        body
    }

    #[test]
    fn a_body_within_the_request_limit_is_admitted_whatever_its_escapes_dump_to() {
        for (context_length, limit, values) in [
            (65_536, 1024 * 1024, 150_000),
            (131_072, 4 * 1024 * 1024, 300_000),
        ] {
            for control in [br"\n", br"\t"] {
                // Arrange: as long as the received-body limit allows, nearly all of it escaped control characters.
                let frame = escaped_body(control, 0).len();
                let body = escaped_body(control, (limit - frame) / 2);
                assert!(limit - body.len() < 2);
                let request =
                    parse_json_limited(&body, JsonLimits::new(limit, values, 32)).unwrap();
                // The canonical dump is almost three times the body, far past the limit.
                assert!(request.dump().unwrap().len() > 2 * limit);

                // Act
                let admitted = admit_chat_with_context(
                    &request,
                    b"m",
                    ToolEnumValidation::Strict,
                    ToolArgValidation::Strict,
                    context_length,
                );

                // Assert
                assert!(admitted.is_ok(), "{:?}", admitted.err());
            }
        }
    }

    /// A chat request built in code, with `content` as its one message.
    fn built(content: Vec<u8>) -> Json {
        Json::new_object([
            ("model", Json::from("m")),
            (
                "messages",
                Json::Array(vec![Json::new_object([
                    ("role", Json::from("user")),
                    ("content", Json::String(content)),
                ])]),
            ),
        ])
    }

    #[test]
    fn a_request_no_body_within_the_limit_could_carry_is_still_refused() {
        // Arrange: built in code, so nothing bounded these as received.
        let limit = 1024 * 1024;
        let fits_as_sent = built(vec![b'\n'; limit / 2 - 64]);
        let too_long = built(vec![b'a'; limit]);
        // \u0001 has no short escape: a client needs all six bytes.
        let long_escapes = built(vec![0x01; limit / 6]);
        let beyond_any_dump = built(vec![b'a'; 3 * limit]);
        let invalid = built(vec![0xc0, 0x80]);

        // Act and assert
        assert!(fits_as_sent.dump().unwrap().len() > 2 * limit);
        assert!(admit_chat(&fits_as_sent, b"m").is_ok());
        assert_eq!(
            admit_chat(&too_long, b"m").unwrap_err().to_string(),
            "chat request exceeds byte limit"
        );
        assert_eq!(
            admit_chat(&long_escapes, b"m").unwrap_err().to_string(),
            "chat request exceeds byte limit"
        );
        assert_eq!(
            admit_chat(&beyond_any_dump, b"m").unwrap_err().to_string(),
            "JSON output byte limit exceeded"
        );
        assert_eq!(
            admit_chat(&invalid, b"m").unwrap_err().to_string(),
            "invalid UTF-8 input"
        );
        assert_eq!(
            short_escape_savings(&built(b"\x08\t\n\x0b\x0c\r\x1f a".to_vec())),
            20
        );
    }

    #[test]
    fn executor_enum_policy_defers_membership_and_accepts_literal_markers() {
        assert_eq!(
            parse_tool_enum_validation("strict"),
            Ok(ToolEnumValidation::Strict)
        );
        assert_eq!(
            parse_tool_enum_validation("executor"),
            Ok(ToolEnumValidation::Executor)
        );
        assert_eq!(
            parse_tool_enum_validation("permissive"),
            Err(AdmissionError::Refused(
                "tool-enum-validation must be strict or executor"
            ))
        );
        assert_eq!(
            tool_enum_validation_policy(ToolEnumValidation::Strict),
            "strict-server-enums-v1"
        );
        assert_eq!(
            tool_enum_validation_policy(ToolEnumValidation::Executor),
            "original-agent-executor-enums-v1"
        );

        let schema = lexeme(
            r#"{"type":"object","properties":{"cmd":{"type":"string","enum":["cargo test"]}},"required":["cmd"],"additionalProperties":false}"#,
        );
        let outside = lexeme(r#"{"cmd":"cargo test --frozen"}"#);
        assert_eq!(
            validate_tool_value(&schema, &outside),
            Err(AdmissionError::Refused("tool value outside enum"))
        );
        assert_eq!(
            validate_tool_value_with_policy(&schema, &outside, ToolEnumValidation::Executor),
            Ok(())
        );

        for value in [
            r#"{"cmd":"echo <|im_start|>"}"#,
            r#"{"cmd":"echo <|endoftext|>"}"#,
        ] {
            assert_eq!(
                validate_tool_value_with_policy(
                    &schema,
                    &lexeme(value),
                    ToolEnumValidation::Executor
                ),
                Ok(())
            );
        }
        assert!(
            validate_tool_value_with_policy(
                &schema,
                &lexeme(r#"{"cmd":"\u0000"}"#),
                ToolEnumValidation::Executor
            )
            .is_err()
        );
    }
}

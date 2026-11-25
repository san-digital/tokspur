//! Port of `native/tests/test_serve_protocol.cpp`, check for check.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};

use tokspur_json::{Json, Kind, parse_json};
use tokspur_server::admission::{
    ToolArgValidation, ToolEnumValidation, admit_chat, admit_chat_with_context,
    admit_chat_with_policies, admit_chat_with_policy, parse_tool_arg_validation,
    parse_tool_enum_validation, tool_arg_validation_policy, tool_enum_validation_policy,
    validate_executor_tool_arguments, validate_tool_value, validate_tool_value_with_policy,
    validate_worker_ready,
};
use tokspur_server::session::{ServeCodec, ServeError, ServeSession, ServeSink, SessionOptions};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response_protocol::ResponseAdmission;

static CHECKS: AtomicUsize = AtomicUsize::new(0);

#[track_caller]
fn check(value: bool) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    assert!(value, "serving protocol check failed");
}

/// `refusal`: the C++ `FormatError` family (admission, JSON, text and V8 admission), not a worker or transport error.
#[track_caller]
fn refusal<T>(result: Result<T, ServeError>) {
    let refused = match result {
        Err(ServeError::Admission(_) | ServeError::Text(_)) => true,
        #[cfg(feature = "verify8")]
        Err(ServeError::Verify8(_)) => true,
        Err(other) => panic!("expected FormatError, got {other}"),
        Ok(_) => false,
    };
    check(refused);
}

#[track_caller]
fn refusal_admission<T, E>(result: Result<T, E>) {
    check(result.is_err());
}

fn parse(text: &str) -> Json {
    parse_json(text.as_bytes()).unwrap()
}

fn with(input: &Json, key: &str, value: Json) -> Json {
    let mut fields = input.object().unwrap().clone();
    for (name, item) in &mut fields {
        if name.as_slice() == key.as_bytes() {
            *item = value;
            return Json::Object(fields);
        }
    }
    fields.push((key.as_bytes().to_vec(), value));
    Json::Object(fields)
}

fn without(input: &Json, key: &str) -> Json {
    Json::Object(
        input
            .object()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.as_slice() != key.as_bytes())
            .cloned()
            .collect(),
    )
}

fn replay_metadata(input: &Json) -> Json {
    let input = with(input, "replay_prepared", Json::Bool(true));
    let input = with(&input, "recurrent_replay_captures", Json::integer(24));
    with(
        &input,
        "mapped_weight_region_limit_bytes",
        Json::integer(4 << 30),
    )
}

fn worker_ready_admission() {
    let ready = parse(
        r#"{"event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct","maximum_generated_tokens":4096}"#,
    );
    validate_worker_ready(&ready, "direct", 0, 0).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    validate_worker_ready(
        &with(&ready, "unrelated_diagnostic", Json::Bool(true)),
        "direct",
        0,
        0,
    )
    .unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    for (key, expected) in ready.object().unwrap() {
        let key = std::str::from_utf8(key).unwrap();
        refusal_admission(validate_worker_ready(&without(&ready, key), "direct", 0, 0));
        for invalid in [
            Json::Null,
            Json::Bool(true),
            Json::Bool(false),
            Json::Array(Vec::new()),
            Json::Object(Vec::new()),
        ] {
            refusal_admission(validate_worker_ready(
                &with(&ready, key, invalid),
                "direct",
                0,
                0,
            ));
        }
        if expected.kind() == Kind::String {
            for invalid in [Json::from(""), Json::from("wrong"), Json::integer(1)] {
                refusal_admission(validate_worker_ready(
                    &with(&ready, key, invalid),
                    "direct",
                    0,
                    0,
                ));
            }
        } else {
            let n = expected.unsigned_integer().unwrap();
            for invalid in [
                Json::integer(0),
                Json::integer(n - 1),
                Json::integer(n + 1),
                Json::from(n.to_string()),
                parse(&format!("{n}.0")),
                parse(&format!("{n}e0")),
            ] {
                refusal_admission(validate_worker_ready(
                    &with(&ready, key, invalid),
                    "direct",
                    0,
                    0,
                ));
            }
        }
    }
    for mode in ["indirect-compatible", "recurrent-replay", "capture"] {
        refusal_admission(validate_worker_ready(
            &with(&ready, "stack_execution", Json::from(mode)),
            "direct",
            0,
            0,
        ));
    }
    for invalid in [
        Json::Null,
        Json::Bool(true),
        Json::from("ready"),
        Json::Array(Vec::new()),
        Json::Object(Vec::new()),
    ] {
        refusal_admission(validate_worker_ready(&invalid, "direct", 0, 0));
    }
    // Refusals are pure: a later exact ready record remains admissible.
    validate_worker_ready(&ready, "direct", 0, 0).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    let replay = replay_metadata(&with(
        &ready,
        "stack_execution",
        Json::from("recurrent-replay"),
    ));
    validate_worker_ready(&replay, "recurrent-replay", 0, 0).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    refusal_admission(validate_worker_ready(&replay, "direct", 0, 0));
    refusal_admission(validate_worker_ready(&ready, "recurrent-replay", 0, 0));
    for mode in ["", "capture", "indirect-compatible", "unknown"] {
        refusal_admission(validate_worker_ready(&replay, mode, 0, 0));
        refusal_admission(validate_worker_ready(&ready, mode, 0, 0));
    }
    for key in [
        "replay_prepared",
        "recurrent_replay_captures",
        "mapped_weight_region_limit_bytes",
    ] {
        refusal_admission(validate_worker_ready(
            &without(&replay, key),
            "recurrent-replay",
            0,
            0,
        ));
        for invalid in [
            Json::Null,
            Json::from("true"),
            Json::Array(Vec::new()),
            Json::Object(Vec::new()),
            Json::Bool(false),
        ] {
            refusal_admission(validate_worker_ready(
                &with(&replay, key, invalid),
                "recurrent-replay",
                0,
                0,
            ));
        }
        if key == "replay_prepared" {
            refusal_admission(validate_worker_ready(
                &with(&replay, key, Json::integer(1)),
                "recurrent-replay",
                0,
                0,
            ));
        } else {
            let n = replay
                .find(key)
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap();
            for invalid in [
                Json::Bool(true),
                Json::integer(0),
                Json::integer(n - 1),
                Json::integer(n + 1),
                Json::from(n.to_string()),
                parse(&format!("{n}.0")),
                parse(&format!("{n}e0")),
            ] {
                refusal_admission(validate_worker_ready(
                    &with(&replay, key, invalid),
                    "recurrent-replay",
                    0,
                    0,
                ));
            }
        }
    }
    validate_worker_ready(&replay, "recurrent-replay", 0, 0).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
}

fn encode(text: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(b"<|im_end|>") {
            result.push(1000);
            i += 10;
        } else if text[i..].starts_with(b"<|endoftext|>") {
            result.push(1001);
            i += 13;
        } else if text[i..].starts_with(b"<|im_start|>") {
            result.push(1002);
            i += 12;
        } else {
            result.push(u32::from(text[i]) + 1);
            i += 1;
        }
    }
    result
}

fn decode(ids: &[u32]) -> Vec<u8> {
    let mut result = Vec::new();
    for &id in ids {
        match id {
            1000 => result.extend_from_slice(b"<|im_end|>"),
            1001 => result.extend_from_slice(b"<|endoftext|>"),
            1002 => result.extend_from_slice(b"<|im_start|>"),
            _ => {
                check(id > 0 && id <= 256);
                result.push((id - 1) as u8);
            }
        }
    }
    result
}

fn codec() -> ServeCodec {
    ServeCodec {
        control_ids: Vec::new(),
        nonordinary_ids: Vec::new(),
        render: Box::new(|request: &Json| {
            let messages = request.find("messages").unwrap().unwrap().array().unwrap();
            let prompt = messages
                .last()
                .unwrap()
                .find("content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                .to_vec();
            // The fake renderer owns fixture terminators, as the real chat template owns role framing.
            let prompt = String::from_utf8(prompt)
                .unwrap()
                .replace("[fixture_end]", "<|im_end|>")
                .into_bytes();
            Ok(RenderedChat {
                prompt,
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| Ok(encode(&text.prompt))),
        decode: Box::new(|ids: &[u32]| Ok(decode(ids))),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    }
}

fn session(mode: &str, broker_ready: Json) -> Result<ServeSession, ServeError> {
    ServeSession::new(
        codec(),
        SessionOptions {
            stack_execution: mode.to_owned(),
            broker_ready,
            ..SessionOptions::default()
        },
    )
}

fn request(text: &str, stream: bool, thinking: bool) -> Json {
    let text = text.replace("<|im_end|>", "[fixture_end]");
    Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from(text)),
            ])]),
        ),
        ("max_tokens", Json::integer(256)),
        ("stream", Json::Bool(stream)),
        ("enable_thinking", Json::Bool(thinking)),
    ])
}

fn plain(text: &str) -> Json {
    request(text, false, false)
}

fn tools() -> Json {
    parse(
        r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}]"#,
    )
}

type Event<'a> = &'a mut dyn FnMut(&Json) -> Result<(), ServeError>;

#[derive(Default)]
struct Fake {
    position: u32,
    seed: u64,
    sampling: Json,
    output: Vec<u32>,
    next: usize,
    resets: usize,
    prefills: usize,
    generations: usize,
    corrupt: String,
    stack_execution: String,
}

fn fixture_failure(message: &'static str) -> ServeError {
    ServeError::Admission(tokspur_server::admission::AdmissionError::Refused(message))
}

impl Fake {
    fn new() -> Self {
        Self {
            stack_execution: "direct".to_owned(),
            ..Self::default()
        }
    }

    fn reply(&mut self, text: &str) {
        self.output = encode(text.as_bytes());
        self.next = 0;
    }

    fn call(&mut self, job: &Json, event: Event<'_>) -> Result<Json, ServeError> {
        if job.find("reset")?.unwrap().boolean()? {
            self.position = 0;
            self.resets += 1;
            self.sampling = job.find("sampling")?.unwrap().clone();
            self.seed = job.find("seed")?.unwrap().unsigned_integer()?;
        }
        let generation = job.find("generate")?.unwrap().unsigned_integer()? != 0;
        check(job.find("tokens")?.unwrap().array()?.len() == if generation { 0 } else { 1 });
        let stopped = generation && matches!(self.output[self.next], 1000 | 1001);
        let mut ids = Vec::new();
        self.position += 1;
        if generation {
            self.generations += 1;
            let id = self.output[self.next];
            self.next += 1;
            ids.push(Json::integer(u64::from(id)));
            event(&Json::new_object([
                ("event", Json::from("token")),
                ("id", Json::integer(u64::from(id))),
                ("committed_prefix", Json::integer(u64::from(self.position))),
            ]))?;
        } else {
            self.prefills += 1;
        }
        let mut policy = self.sampling.object()?.clone();
        policy.push((b"valid".to_vec(), Json::Bool(true)));
        policy.push((
            b"entropy_prefix".to_vec(),
            Json::integer(u64::from(self.position)),
        ));
        policy.push((b"seed".to_vec(), Json::integer(self.seed)));
        let status = if generation {
            if stopped { "stopped" } else { "length" }
        } else {
            "prefilled"
        };
        let mut done = Json::new_object([
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(u64::from(self.position))),
            ("generated_ids", Json::Array(ids)),
            ("status", Json::from(status)),
            (
                "finish_reason",
                if generation {
                    Json::from(if stopped { "stop" } else { "length" })
                } else {
                    Json::Null
                },
            ),
            (
                "stop_id",
                if stopped {
                    Json::integer(u64::from(self.output[self.next - 1]))
                } else {
                    Json::Null
                },
            ),
            ("continuation_allowed", Json::Bool(!stopped)),
            ("sampling", Json::Object(policy)),
            ("small_gemv_column_splits", Json::integer(16)),
            ("block_gemv_formats", Json::integer(3)),
            ("stack_execution", Json::from(self.stack_execution.as_str())),
        ]);
        if self.stack_execution == "recurrent-replay" {
            done = replay_metadata(&done);
        }
        let corrupt = self.corrupt.as_str();
        if generation && corrupt == "prefix" {
            done = with(
                &done,
                "committed_prefix",
                Json::integer(u64::from(self.position) + 1),
            );
        }
        if generation && corrupt == "policy" {
            done = with(&done, "block_gemv_formats", Json::integer(1));
        }
        if generation && corrupt == "execution" {
            let other = if self.stack_execution == "direct" {
                "recurrent-replay"
            } else {
                "direct"
            };
            done = with(&done, "stack_execution", Json::from(other));
        }
        if generation && corrupt == "replay-prepared" {
            done = with(&done, "replay_prepared", Json::Bool(false));
        }
        if generation && corrupt == "replay-captures" {
            done = with(&done, "recurrent_replay_captures", Json::integer(23));
        }
        if generation && corrupt == "replay-region" {
            done = with(&done, "mapped_weight_region_limit_bytes", Json::integer(0));
        }
        if generation && corrupt == "replay-missing" {
            done = without(&done, "replay_prepared");
        }
        if stopped && corrupt == "terminal" {
            done = with(&done, "stop_id", Json::integer(1001));
        }
        if generation && corrupt == "throw" {
            return Err(fixture_failure("fixture final failure after token"));
        }
        Ok(done)
    }
}

fn broker_metadata(value: &Json) -> Json {
    let value = with(value, "checkpoint_slots", Json::integer(2));
    let value = with(
        &value,
        "checkpoint_combined_budget_bytes",
        Json::integer(1 << 20),
    );
    let value = with(&value, "checkpoint_host_bytes", Json::integer(512));
    let value = with(&value, "checkpoint_store_host_bytes", Json::integer(256));
    with(&value, "checkpoint_gpu_bytes", Json::integer(1024))
}

fn broker_ready() -> Json {
    broker_metadata(&parse(
        r#"{"event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct","maximum_generated_tokens":4096,"checkpoint_broker":true,"broker_epoch":77}"#,
    ))
}

#[derive(Clone, Copy, Default)]
struct Saved {
    prefix: usize,
    order: usize,
    valid: bool,
}

struct BrokerFake {
    base: Fake,
    saved: [Saved; 2],
    history: Vec<u32>,
    command: u64,
    lease: u64,
    order: usize,
    begins: usize,
    captures: usize,
    hits: usize,
    last_restore: bool,
    fail_suffix: bool,
    control_corrupt: String,
}

impl BrokerFake {
    fn new() -> Self {
        Self {
            base: Fake::new(),
            saved: [Saved::default(); 2],
            history: Vec::new(),
            command: 0,
            lease: 0,
            order: 0,
            begins: 0,
            captures: 0,
            hits: 0,
            last_restore: false,
            fail_suffix: false,
            control_corrupt: String::new(),
        }
    }

    fn stamp(&self, value: &Json) -> Json {
        let value = broker_metadata(value);
        let retained = usize::from(self.saved[0].valid) + usize::from(self.saved[1].valid);
        let value = with(
            &value,
            "checkpoint_retained",
            Json::integer(retained as u64),
        );
        let value = with(
            &value,
            "broker_epoch",
            Json::integer(if self.control_corrupt == "epoch" {
                78
            } else {
                77
            }),
        );
        let mut value = with(&value, "command_id", Json::integer(self.command));
        if self.control_corrupt == "storage" {
            value = with(&value, "checkpoint_gpu_bytes", Json::integer(1025));
        }
        let lease = if self.control_corrupt == "lease" {
            self.lease.wrapping_sub(1)
        } else {
            self.lease
        };
        with(&value, "lease_id", Json::integer(lease))
    }

    fn call(&mut self, job: &Json, event: Event<'_>) -> Result<Json, ServeError> {
        self.command += 1;
        if let Some(op) = job.find("op")? {
            check(job.find("tokens")?.is_none() && job.find("generate")?.is_none());
            let begin = op.string()? == b"branch_begin";
            let mut outcome = "captured";
            let mut common = self.base.position as usize;
            if begin {
                self.begins += 1;
                self.lease += 1;
                self.last_restore = job.find("allow_restore")?.unwrap().boolean()?;
                let future = job.find("prompt_ids")?.unwrap().array()?;
                common = 0;
                if self.last_restore {
                    while common < future.len()
                        && common < self.history.len()
                        && u64::from(self.history[common]) == future[common].unsigned_integer()?
                    {
                        common += 1;
                    }
                }
                let mut selected: Option<usize> = None;
                for i in 0..self.saved.len() {
                    if !self.last_restore {
                        break;
                    }
                    let s = self.saved[i];
                    if s.valid
                        && s.prefix <= common
                        && selected.is_none_or(|j| {
                            s.prefix > self.saved[j].prefix
                                || (s.prefix == self.saved[j].prefix
                                    && s.order > self.saved[j].order)
                        })
                    {
                        selected = Some(i);
                    }
                }
                if let Some(j) = selected {
                    self.base.position = self.saved[j].prefix as u32;
                    self.hits += 1;
                    outcome = "restored";
                    let position = self.base.position as usize;
                    for record in &mut self.saved {
                        if record.valid && record.prefix > position {
                            record.valid = false;
                        }
                    }
                } else {
                    self.base.position = 0;
                    self.base.resets += 1;
                    for record in &mut self.saved {
                        record.valid = false;
                    }
                    outcome = "cold_miss";
                }
                self.history.truncate(self.base.position as usize);
                self.history.resize(self.base.position as usize, 0);
                self.base.sampling = job.find("sampling")?.unwrap().clone();
                self.base.seed = job.find("seed")?.unwrap().unsigned_integer()?;
            } else {
                check(op.string()? == b"branch_capture" && self.base.position != 0);
                self.captures += 1;
                let mut at = 0;
                for i in 0..self.saved.len() {
                    if !self.saved[i].valid {
                        at = i;
                        break;
                    }
                    if self.saved[i].order < self.saved[at].order {
                        at = i;
                    }
                }
                self.order += 1;
                self.saved[at] = Saved {
                    prefix: self.base.position as usize,
                    order: self.order,
                    valid: true,
                };
            }
            let position = u64::from(self.base.position);
            let mut policy = self.base.sampling.object()?.clone();
            policy.push((b"valid".to_vec(), Json::Bool(true)));
            policy.push((b"entropy_prefix".to_vec(), Json::integer(position)));
            policy.push((b"seed".to_vec(), Json::integer(self.base.seed)));
            if self.control_corrupt == "token" {
                event(&self.stamp(&Json::new_object([
                    ("event", Json::from("token")),
                    ("id", Json::integer(1)),
                    ("committed_prefix", Json::integer(position + 1)),
                ])))?;
            }
            let lcp = common as u64 + u64::from(self.control_corrupt == "lcp");
            return Ok(self.stamp(&Json::new_object([
                ("event", Json::from("control_done")),
                ("op", op.clone()),
                ("outcome", Json::from(outcome)),
                ("model_steps", Json::integer(0)),
                ("generated_ids", Json::Array(Vec::new())),
                ("committed_prefix", Json::integer(position)),
                ("common_prefix", Json::integer(lcp)),
                ("sampling", Json::Object(policy)),
                ("logits_available", Json::Bool(position != 0)),
                ("stop_id", Json::Null),
                ("continuation_allowed", Json::Bool(position != 0)),
            ])));
        }
        if job.find("reset")?.unwrap().boolean()? {
            self.lease += 1;
            self.history.clear();
            for record in &mut self.saved {
                record.valid = false;
            }
        }
        let generated = job.find("generate")?.unwrap().unsigned_integer()? != 0;
        let mut tokens: Vec<u32> = Vec::new();
        let stamps = |token: &Json| -> Json { token.clone() };
        let done = {
            let history = &mut tokens;
            let snapshot = self.stamp_context();
            let mut forward = |token: &Json| -> Result<(), ServeError> {
                history.push(token.find("id")?.unwrap().unsigned_integer()? as u32);
                event(&snapshot.apply(&stamps(token)))
            };
            self.base.call(job, &mut forward)?
        };
        self.history.extend(tokens);
        if !generated {
            self.history
                .push(job.find("tokens")?.unwrap().array()?[0].unsigned_integer()? as u32);
        }
        if self.fail_suffix && !generated {
            return Err(fixture_failure(
                "fake suffix failure after committed model input",
            ));
        }
        Ok(self.stamp(&done))
    }

    /// The fields `stamp` adds, captured before the base fake runs.
    fn stamp_context(&self) -> StampContext {
        StampContext {
            retained: usize::from(self.saved[0].valid) + usize::from(self.saved[1].valid),
            epoch: if self.control_corrupt == "epoch" {
                78
            } else {
                77
            },
            command: self.command,
            storage: self.control_corrupt == "storage",
            lease: if self.control_corrupt == "lease" {
                self.lease.wrapping_sub(1)
            } else {
                self.lease
            },
        }
    }
}

struct StampContext {
    retained: usize,
    epoch: u64,
    command: u64,
    storage: bool,
    lease: u64,
}

impl StampContext {
    fn apply(&self, value: &Json) -> Json {
        let value = broker_metadata(value);
        let value = with(
            &value,
            "checkpoint_retained",
            Json::integer(self.retained as u64),
        );
        let value = with(&value, "broker_epoch", Json::integer(self.epoch));
        let mut value = with(&value, "command_id", Json::integer(self.command));
        if self.storage {
            value = with(&value, "checkpoint_gpu_bytes", Json::integer(1025));
        }
        with(&value, "lease_id", Json::integer(self.lease))
    }
}

fn message(result: &Json) -> &Json {
    result.find("choices").unwrap().unwrap().array().unwrap()[0]
        .find("message")
        .unwrap()
        .unwrap()
}

fn finish(result: &Json) -> &Json {
    result.find("choices").unwrap().unwrap().array().unwrap()[0]
        .find("finish_reason")
        .unwrap()
        .unwrap()
}

fn delta(chunk: &Json) -> &Json {
    chunk.find("choices").unwrap().unwrap().array().unwrap()[0]
        .find("delta")
        .unwrap()
        .unwrap()
}

fn startup_execution_policy() {
    for invalid in ["", "capture", "indirect-compatible", "unknown"] {
        refusal(session(invalid, Json::Null));
    }
    for mode in ["direct", "recurrent-replay"] {
        let mut serving = session(mode, Json::Null).unwrap();
        let fake = RefCell::new(Fake::new());
        fake.borrow_mut().stack_execution = mode.to_owned();
        let chunks: RefCell<Vec<Json>> = RefCell::new(Vec::new());
        let mut push = |part: &Json| {
            chunks.borrow_mut().push(part.clone());
            Ok(())
        };
        let mut never = || false;
        let mut seed = || 42;
        let mut sink = ServeSink {
            chunk: Some(&mut push),
            cancelled: &mut never,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);
        fake.borrow_mut().reply("valid<|im_end|>");
        let result = serving
            .complete(
                &request("A", true, false),
                b"startup-policy",
                &mut call,
                &mut sink,
            )
            .unwrap();
        check(finish(&result).string().unwrap() == b"stop");
        let submitted = fake.borrow().prefills + fake.borrow().generations;
        for body_key in ["stack_execution", "stack-execution"] {
            refusal(serving.complete(
                &with(&plain("A"), body_key, Json::from(mode)),
                b"body-switch",
                &mut call,
                &mut sink,
            ));
        }
        check(submitted == fake.borrow().prefills + fake.borrow().generations);
        let tool_request = with(&request("B", true, false), "tools", tools());
        for corrupt in [
            "execution",
            "replay-prepared",
            "replay-captures",
            "replay-region",
            "replay-missing",
        ] {
            if mode == "direct" && corrupt != "execution" {
                continue;
            }
            chunks.borrow_mut().clear();
            fake.borrow_mut().corrupt = corrupt.to_owned();
            fake.borrow_mut()
                .reply("<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>");
            refusal(serving.complete(&tool_request, b"execution-mismatch", &mut call, &mut sink));
            check(serving.committed_tokens() == 0);
            for part in chunks.borrow().iter() {
                check(delta(part).find("tool_calls").unwrap().is_none());
            }
            fake.borrow_mut().corrupt.clear();
            fake.borrow_mut().reply("recovered<|im_end|>");
            let result = serving
                .complete(&plain("A"), b"after-policy-refusal", &mut call, &mut sink)
                .unwrap();
            check(finish(&result).string().unwrap() == b"stop");
        }
        serving.invalidate();
        fake.borrow_mut().stack_execution = if mode == "direct" {
            "recurrent-replay"
        } else {
            "direct"
        }
        .to_owned();
        fake.borrow_mut().reply("never accepted<|im_end|>");
        let generations = fake.borrow().generations;
        refusal(serving.complete(&plain("A"), b"startup-mode-mismatch", &mut call, &mut sink));
        check(serving.committed_tokens() == 0 && fake.borrow().generations == generations);
    }
}

fn broker_sessions() {
    let ready = broker_ready();
    validate_worker_ready(&ready, "direct", 2, 1 << 20).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    refusal_admission(validate_worker_ready(&ready, "direct", 0, 0));
    for key in [
        "checkpoint_broker",
        "broker_epoch",
        "checkpoint_host_bytes",
        "checkpoint_gpu_bytes",
    ] {
        refusal_admission(validate_worker_ready(
            &without(&ready, key),
            "direct",
            2,
            1 << 20,
        ));
    }
    let mut serving = session("direct", ready.clone()).unwrap();
    let fake = RefCell::new(BrokerFake::new());
    let cancel = Cell::new(false);
    let mut cancelled = || cancel.get();
    let mut seed = || 42;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut seed,
        prefix_trace: None,
    };
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);
    fake.borrow_mut().base.reply("X<|im_end|>");
    serving
        .complete(&plain("AB"), b"broker-first", &mut call, &mut sink)
        .unwrap();
    {
        let f = fake.borrow();
        check(f.begins == 1 && f.base.resets == 1 && f.captures == 1 && f.base.prefills == 2);
    }
    fake.borrow_mut().base.reply("Y<|im_end|>");
    serving
        .complete(
            &plain("ABX<|im_end|>C"),
            b"broker-append",
            &mut call,
            &mut sink,
        )
        .unwrap();
    {
        let f = fake.borrow();
        check(f.begins == 1 && f.base.resets == 1 && f.captures == 2);
    }
    let before = fake.borrow().base.prefills;
    fake.borrow_mut().base.reply("Z<|im_end|>");
    let branch = serving
        .complete(&plain("ABD"), b"broker-branch", &mut call, &mut sink)
        .unwrap();
    let cached = |result: &Json| {
        result
            .find("usage")
            .unwrap()
            .unwrap()
            .find("prompt_tokens_details")
            .unwrap()
            .unwrap()
            .find("cached_tokens")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
    };
    check(
        fake.borrow().hits == 1
            && fake.borrow().base.prefills == before + 1
            && cached(&branch) == 2,
    );
    let captures = fake.borrow().captures;
    fake.borrow_mut().base.reply("Z<|im_end|>");
    serving
        .complete(&plain("ABD"), b"broker-exact", &mut call, &mut sink)
        .unwrap();
    check(fake.borrow().captures == captures && fake.borrow().hits == 2);
    fake.borrow_mut().fail_suffix = true;
    fake.borrow_mut().base.reply("unused<|im_end|>");
    refusal(serving.complete(&plain("ABE"), b"broker-suffix-failed", &mut call, &mut sink));
    check(serving.committed_tokens() == 0);
    fake.borrow_mut().fail_suffix = false;
    fake.borrow_mut().base.reply("safe<|im_end|>");
    serving
        .complete(&plain("AB"), b"broker-after-suffix", &mut call, &mut sink)
        .unwrap();
    check(!fake.borrow().last_restore);
    cancel.set(true);
    let controls = fake.borrow().begins;
    refusal(serving.complete(&plain("Q"), b"broker-before-cancel", &mut call, &mut sink));
    check(fake.borrow().begins == controls);
    cancel.set(false);
    let mut after_control = |job: &Json, event: Event<'_>| {
        let done = fake.borrow_mut().call(job, event);
        if job
            .find("op")?
            .is_some_and(|op| op.string().is_ok_and(|op| op == b"branch_begin"))
        {
            cancel.set(true);
        }
        done
    };
    refusal(serving.complete(
        &plain("Q"),
        b"broker-after-control-cancel",
        &mut after_control,
        &mut sink,
    ));
    check(serving.committed_tokens() == 0);
    cancel.set(false);
    fake.borrow_mut().base.reply("safe<|im_end|>");
    serving
        .complete(&plain("Q"), b"broker-after-cancel", &mut call, &mut sink)
        .unwrap();
    check(!fake.borrow().last_restore);
    let greedy_begins = fake.borrow().begins;
    let ordinary_resets = fake.borrow().base.resets;
    fake.borrow_mut().base.reply("safe<|im_end|>");
    let stochastic = with(&plain("S"), "temperature", parse("0.5"));
    let stochastic = with(&stochastic, "seed", Json::integer(7));
    serving
        .complete(
            &stochastic,
            b"broker-stochastic-reset",
            &mut call,
            &mut sink,
        )
        .unwrap();
    check(
        fake.borrow().begins == greedy_begins && fake.borrow().base.resets == ordinary_resets + 1,
    );
    for corrupt in ["epoch", "lease", "token", "lcp", "storage"] {
        let mut invalid = session("direct", ready.clone()).unwrap();
        let broken = RefCell::new(BrokerFake::new());
        broken.borrow_mut().control_corrupt = corrupt.to_owned();
        let mut bad = |job: &Json, event: Event<'_>| broken.borrow_mut().call(job, event);
        refusal(invalid.complete(&plain("A"), b"broker-invalid-control", &mut bad, &mut sink));
        check(invalid.committed_tokens() == 0);
    }
}

fn literal_marker_serving() {
    let body = "```xml\n<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call>\n```\nDone.";
    for terminal in ["<|im_end|>", "<|endoftext|>"] {
        let mut serving = ServeSession::new(codec(), SessionOptions::default()).unwrap();
        let mut fake = Fake::new();
        fake.reply(&format!("{body}{terminal}"));
        let mut call = |job: &Json, event: Event<'_>| fake.call(job, event);
        let chunks = RefCell::new(Vec::new());
        let mut push = |value: &Json| {
            chunks.borrow_mut().push(value.clone());
            Ok(())
        };
        let mut cancelled = || false;
        let mut new_seed = || 44;
        let mut sink = ServeSink {
            chunk: Some(&mut push),
            cancelled: &mut cancelled,
            new_seed: &mut new_seed,
            prefix_trace: None,
        };
        let job = with(&request("literal", true, false), "tools", tools());
        let result = serving
            .complete(&job, b"literal-stop", &mut call, &mut sink)
            .unwrap();
        check(
            message(&result)
                .find("content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == body.as_bytes(),
        );
        check(message(&result).find("tool_calls").unwrap().is_none());
        check(finish(&result).string().unwrap() == b"stop");
        let mut streamed = Vec::new();
        for chunk in chunks.borrow().iter() {
            let delta = chunk.find("choices").unwrap().unwrap().array().unwrap()[0]
                .find("delta")
                .unwrap()
                .unwrap();
            if let Some(content) = delta.find("content").unwrap() {
                streamed.extend_from_slice(content.string().unwrap());
            }
        }
        check(streamed == body.as_bytes());
    }
}

/// A stop token while reasoning is still open ends the reasoning there: the client gets what was produced with `finish_reason: "stop"`, not an error after its reasoning deltas.
fn open_thinking_stop_serving() {
    let reasoning = "weighing the options";
    for terminal in ["<|im_end|>", "<|endoftext|>"] {
        for stream in [false, true] {
            let mut serving = ServeSession::new(codec(), SessionOptions::default()).unwrap();
            let mut fake = Fake::new();
            fake.reply(&format!("{reasoning}{terminal}"));
            let mut call = |job: &Json, event: Event<'_>| fake.call(job, event);
            let chunks = RefCell::new(Vec::new());
            let mut push = |value: &Json| {
                chunks.borrow_mut().push(value.clone());
                Ok(())
            };
            let mut cancelled = || false;
            let mut new_seed = || 44;
            let mut sink = ServeSink {
                chunk: Some(&mut push),
                cancelled: &mut cancelled,
                new_seed: &mut new_seed,
                prefix_trace: None,
            };
            let result = serving
                .complete(
                    &request("think", stream, true),
                    b"open-thinking",
                    &mut call,
                    &mut sink,
                )
                .unwrap();
            let reply = message(&result);
            check(
                reply
                    .find("reasoning_content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == reasoning.as_bytes(),
            );
            check(
                reply
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    .is_empty(),
            );
            check(reply.find("tool_calls").unwrap().is_none());
            check(finish(&result).string().unwrap() == b"stop");
            // The session stays usable for the next turn.
            check(serving.committed_tokens() > 0);
            let chunks = chunks.borrow();
            check(chunks.is_empty() != stream);
            if !stream {
                continue;
            }
            let mut streamed = Vec::new();
            for chunk in chunks.iter() {
                let delta = delta(chunk);
                check(delta.find("content").unwrap().is_none());
                if let Some(text) = delta.find("reasoning_content").unwrap() {
                    streamed.extend_from_slice(text.string().unwrap());
                }
            }
            check(streamed == reasoning.as_bytes());
            let last = chunks.last().unwrap();
            check(
                last.find("choices").unwrap().unwrap().array().unwrap()[0]
                    .find("finish_reason")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == b"stop",
            );
        }
    }
    // A required tool is still required, and a tool call cut off by the stop is still refused.
    for (job, raw, public_error) in [
        (
            with(
                &with(&request("think", false, true), "tools", tools()),
                "tool_choice",
                Json::from("required"),
            ),
            "weighing the options<|im_end|>",
            "required tool was not produced",
        ),
        (
            with(&request("think", false, true), "tools", tools()),
            "done</think><tool_call><function=read_file><|im_end|>",
            "incomplete or invalid stopped response",
        ),
    ] {
        let mut serving = ServeSession::new(codec(), SessionOptions::default()).unwrap();
        let mut fake = Fake::new();
        fake.reply(raw);
        let error = serving
            .complete(
                &job,
                b"open-thinking-refused",
                &mut |job, event| fake.call(job, event),
                &mut ServeSink {
                    chunk: None,
                    cancelled: &mut || false,
                    new_seed: &mut || 44,
                    prefix_trace: None,
                },
            )
            .unwrap_err();
        check(error.to_string() == public_error);
    }
}

fn parallel_tool_streams() {
    for valid in [true, false] {
        let mut serving = session("direct", Json::Null).unwrap();
        let fake = RefCell::new(Fake::new());
        let chunks = RefCell::new(Vec::new());
        let frame = |name: &str, path: &str| {
            format!(
                "<tool_call><function={name}><parameter=path>\n{path}\n</parameter></function></tool_call>"
            )
        };
        fake.borrow_mut().reply(&format!(
            "{}\n{}<|im_end|>",
            frame("read_file", "a.rs"),
            frame(if valid { "read_file" } else { "unknown" }, "b.rs")
        ));
        let job = with(
            &with(&request("fixture", true, false), "tools", tools()),
            "max_tokens",
            Json::integer(512),
        );
        let mut chunk = |value: &Json| {
            chunks.borrow_mut().push(value.clone());
            Ok(())
        };
        let mut cancelled = || false;
        let mut seed = || 42;
        let mut sink = ServeSink {
            chunk: Some(&mut chunk),
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);
        let result = serving.complete(&job, b"parallel", &mut call, &mut sink);
        if valid {
            let result = result.unwrap();
            check(
                message(&result)
                    .find("tool_calls")
                    .unwrap()
                    .unwrap()
                    .array()
                    .unwrap()
                    .len()
                    == 2,
            );
            let mut index = 0;
            for chunk in chunks.borrow().iter() {
                let Some(calls) = delta(chunk).find("tool_calls").unwrap() else {
                    continue;
                };
                let calls = calls.array().unwrap();
                check(
                    calls.len() == 1
                        && calls[0]
                            .find("index")
                            .unwrap()
                            .unwrap()
                            .unsigned_integer()
                            .unwrap()
                            == index,
                );
                let args = calls[0]
                    .find("function")
                    .unwrap()
                    .unwrap()
                    .find("arguments")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap();
                check(
                    parse_json(args)
                        .unwrap()
                        .find("path")
                        .unwrap()
                        .unwrap()
                        .string()
                        .unwrap()
                        == if index == 0 { b"a.rs" } else { b"b.rs" },
                );
                index += 1;
            }
            check(index == 2);
        } else {
            refusal(result);
            for chunk in chunks.borrow().iter() {
                check(delta(chunk).find("tool_calls").unwrap().is_none());
            }
        }
    }
}

fn main() {
    literal_marker_serving();
    if std::env::args().nth(1).as_deref() == Some("--response-failure-trace-case") {
        response_failure_trace_cases();
        return;
    }
    if std::env::args().any(|arg| arg == "--list") {
        println!("serve_protocol: test");
        return;
    }
    parallel_tool_streams();
    open_thinking_stop_serving();
    // Running totals after each C++ function, to compare with an instrumented copy of test_serve_protocol.cpp.
    let section = |name: &str| {
        if std::env::var_os("TOKSPUR_SERVE_SECTIONS").is_some() {
            println!("{name} {}", CHECKS.load(Ordering::Relaxed));
        }
    };
    let diagnostic_child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--response-failure-trace-case")
        .output()
        .unwrap();
    assert!(
        diagnostic_child.status.success(),
        "response failure child failed: {}",
        String::from_utf8_lossy(&diagnostic_child.stderr)
    );
    assert!(
        !diagnostic_child
            .stdout
            .windows(b"private-failure-canary".len())
            .any(|w| w == b"private-failure-canary"),
    );
    worker_ready_admission();
    section("s1");
    startup_execution_policy();
    section("s2");
    broker_sessions();
    section("s3");
    for text in [
        r#"{"model":"wrong","messages":[]}"#,
        r#"{"model":"tokspur","messages":[],"n":2}"#,
        r#"{"model":"tokspur","messages":[],"max_tokens":0}"#,
        r#"{"model":"tokspur","messages":[],"max_tokens":4097}"#,
        r#"{"model":"tokspur","messages":[],"temperature":-1}"#,
        r#"{"model":"tokspur","messages":[],"top_p":0}"#,
        r#"{"model":"tokspur","messages":[],"max_tokens":1,"max_completion_tokens":1}"#,
        r#"{"model":"tokspur","messages":[{"role":"tool","content":"x","tool_call_id":"fake"}]}"#,
        r#"{"model":"tokspur","messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#,
    ] {
        refusal_admission(admit_chat(&parse(text), b"tokspur"));
    }
    let admitted = admit_chat(&with(&plain("A"), "tools", tools()), b"tokspur").unwrap();
    check(admitted.manifest.len() == 1 && admitted.names[0] == b"read_file");
    let text_parts = with(
        &plain("A"),
        "messages",
        parse(r#"[{"role":"user","content":[{"type":"text","text":"hello"}]}]"#),
    );
    check(admit_chat(&text_parts, b"tokspur").unwrap().maximum_tokens == 256);
    for schema in [
        r#"{"type":"object","oneOf":[]}"#,
        r#"{"type":"object","additionalProperties":true}"#,
        r#"{"type":"array"}"#,
        r#"{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2}"#,
        r#"{"type":"array","items":{"type":"string"},"minItems":-1}"#,
        r#"{"type":"array","items":{"type":"string"},"minItems":4097}"#,
        r#"{"type":"array","items":{"type":"string"},"minItems":1.0}"#,
        r#"{"type":"array","items":{"type":"string"},"maxItems":"2"}"#,
        r#"{"anyOf":[{"type":"array","items":{"type":"string"}}],"minItems":1}"#,
        r#"{"type":"string","minItems":1}"#,
        r#"{"type":["string","null"]}"#,
        r#"{"type":"object","required":["missing"]}"#,
    ] {
        let declaration = tools().array().unwrap()[0].clone();
        let function = with(
            declaration.find("function").unwrap().unwrap(),
            "parameters",
            parse(schema),
        );
        let tools = Json::Array(vec![with(&declaration, "function", function)]);
        refusal_admission(admit_chat(&with(&plain("A"), "tools", tools), b"tokspur"));
    }
    let schema = parse(
        r#"{"type":"object","properties":{"n":{"type":"integer","minimum":1,"maximum":3},"values":{"type":"array","items":{"type":"string","enum":["x","y"]},"minItems":1,"maxItems":2}},"required":["n"]}"#,
    );
    validate_tool_value(&schema, &parse(r#"{"n":2,"values":["x","y"]}"#)).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    for value in [
        r#"{"n":1.5}"#,
        r#"{"n":1.0}"#,
        r#"{"n":4}"#,
        r#"{"n":9007199254740993}"#,
        r#"{"n":2,"values":[]}"#,
        r#"{"n":2,"values":["x","y","x"]}"#,
        r#"{"n":2,"values":["z"]}"#,
        r#"{"n":2,"extra":true}"#,
        r#"{}"#,
    ] {
        refusal_admission(validate_tool_value(&schema, &parse(value)));
    }
    refusal_admission(validate_tool_value(
        &parse(r#"{"type":"integer"}"#),
        &parse("1.00000000000000001"),
    ));
    refusal_admission(validate_tool_value(
        &parse(r#"{"type":"number","enum":[1]}"#),
        &parse("1.00000000000000001"),
    ));
    refusal_admission(validate_tool_value(
        &parse(r#"{"type":"number","enum":[9007199254740992]}"#),
        &parse("9007199254740993"),
    ));
    refusal_admission(validate_tool_value(
        &parse(r#"{"type":"number","maximum":1}"#),
        &parse("1.00000000000000001"),
    ));
    validate_tool_value(&parse(r#"{"type":"number","enum":[1.0]}"#), &parse("1e0")).unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);
    validate_tool_value(
        &parse(r#"{"type":"number","minimum":-1,"maximum":0}"#),
        &parse("-0.1"),
    )
    .unwrap();
    CHECKS.fetch_add(1, Ordering::Relaxed);

    let chunks: RefCell<Vec<Json>> = RefCell::new(Vec::new());
    let cancel = Cell::new(false);
    let seeds = Cell::new(0usize);
    let mut push = |chunk: &Json| {
        chunks.borrow_mut().push(chunk.clone());
        Ok(())
    };
    let mut cancelled = || cancel.get();
    let mut seed = || {
        seeds.set(seeds.get() + 1);
        42
    };
    let mut sink = ServeSink {
        chunk: Some(&mut push),
        cancelled: &mut cancelled,
        new_seed: &mut seed,
        prefix_trace: None,
    };
    let mut serving = session("direct", Json::Null).unwrap();
    let fake = RefCell::new(Fake::new());
    fake.borrow_mut().reply("Hello 🌍<|im_end|>");
    let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);
    let result = serving
        .complete(&request("A", true, false), b"first", &mut call, &mut sink)
        .unwrap();
    check(
        message(&result)
            .find("content")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == "Hello 🌍".as_bytes()
            && finish(&result).string().unwrap() == b"stop",
    );
    let mut streamed = Vec::new();
    for event in chunks.borrow().iter() {
        let reparsed = parse_json(&event.dump().unwrap()).unwrap(); // Every SSE JSON payload has complete UTF-8.
        if let Some(part) = delta(&reparsed).find("content").unwrap() {
            streamed.extend_from_slice(part.string().unwrap());
        }
    }
    check(streamed == "Hello 🌍".as_bytes());
    check(fake.borrow().resets == 1 && seeds.get() == 1 && !chunks.borrow().is_empty());
    let old = serving.committed_tokens();
    fake.borrow_mut().reply("Next<|im_end|>");
    let cached_tokens = |result: &Json| {
        result
            .find("usage")
            .unwrap()
            .unwrap()
            .find("prompt_tokens_details")
            .unwrap()
            .unwrap()
            .find("cached_tokens")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
    };
    let result = serving
        .complete(
            &plain("AHello 🌍<|im_end|>B"),
            b"second",
            &mut call,
            &mut sink,
        )
        .unwrap();
    check(fake.borrow().resets == 1 && seeds.get() == 1 && cached_tokens(&result) == old as u64);
    fake.borrow_mut().reply("Changed<|im_end|>");
    let changed = with(
        &plain("AHello 🌍<|im_end|>BNext<|im_end|>C"),
        "temperature",
        parse("0.5"),
    );
    serving
        .complete(&changed, b"third", &mut call, &mut sink)
        .unwrap();
    check(fake.borrow().resets == 2 && seeds.get() == 2);
    let before = fake.borrow().prefills + fake.borrow().generations;
    refusal(serving.complete(
        &with(&plain("A"), "n", Json::integer(2)),
        b"invalid",
        &mut call,
        &mut sink,
    ));
    check(before == fake.borrow().prefills + fake.borrow().generations);
    refusal(serving.complete(
        &plain(&"A".repeat(65536)),
        b"overflow",
        &mut call,
        &mut sink,
    ));
    check(before == fake.borrow().prefills + fake.borrow().generations);
    {
        let large_req = plain(&"A".repeat(1_100_000));
        refusal_admission(admit_chat(&large_req, b"tokspur"));
        check(
            admit_chat_with_context(
                &large_req,
                b"tokspur",
                ToolEnumValidation::Strict,
                ToolArgValidation::Strict,
                131_072,
            )
            .unwrap()
            .maximum_tokens
                == 256,
        );
        refusal_admission(admit_chat_with_context(
            &plain(&"A".repeat(4_200_000)),
            b"tokspur",
            ToolEnumValidation::Strict,
            ToolArgValidation::Strict,
            131_072,
        ));
        refusal_admission(admit_chat_with_context(
            &plain("A"),
            b"tokspur",
            ToolEnumValidation::Strict,
            ToolArgValidation::Strict,
            100_000,
        ));
        refusal_admission(ServeSession::new(
            codec(),
            SessionOptions {
                context_length: 100_000,
                ..SessionOptions::default()
            },
        ));
        let mut session128k = ServeSession::new(
            codec(),
            SessionOptions {
                context_length: 131_072,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let fake128k = RefCell::new(Fake::new());
        let mut call128k = |job: &Json, event: Event<'_>| fake128k.borrow_mut().call(job, event);
        let mut cancelled128k = || false;
        let mut seed128k = || 77u64;
        let mut sink128k = ServeSink {
            chunk: None,
            cancelled: &mut cancelled128k,
            new_seed: &mut seed128k,
            prefix_trace: None,
        };
        refusal(session128k.complete(
            &plain(&"A".repeat(131_072)),
            b"overflow128k",
            &mut call128k,
            &mut sink128k,
        ));
        check(fake128k.borrow().prefills + fake128k.borrow().generations == 0);
        fake128k.borrow_mut().reply("ok<|im_end|>");
        let r128k = session128k
            .complete(
                &plain(&"A".repeat(65_536)),
                b"admitted128k",
                &mut call128k,
                &mut sink128k,
            )
            .unwrap();
        check(
            finish(&r128k).string().unwrap() == b"stop" && session128k.committed_tokens() == 65_539,
        );
        fake128k.borrow_mut().reply("xyz");
        let append128k = format!("{}ok<|im_end|>{}", "A".repeat(65_536), "B".repeat(65_531));
        let r128k_clamp = session128k
            .complete(
                &with(&plain(&append128k), "max_tokens", Json::integer(100)),
                b"clamped128k",
                &mut call128k,
                &mut sink128k,
            )
            .unwrap();
        check(
            finish(&r128k_clamp).string().unwrap() == b"length"
                && message(&r128k_clamp)
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == b"xy"
                && session128k.committed_tokens() == 131_072,
        );
    }
    chunks.borrow_mut().clear();
    fake.borrow_mut().reply("why</think>\n\nVisible<|im_end|>");
    let result = serving
        .complete(&request("T", true, true), b"thinking", &mut call, &mut sink)
        .unwrap();
    check(
        message(&result)
            .find("reasoning_content")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == b"why"
            && message(&result)
                .find("content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"\n\nVisible",
    );
    let tool_request = with(&request("U", true, false), "tools", tools());
    let tool_response = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>";
    chunks.borrow_mut().clear();
    fake.borrow_mut().reply(tool_response);
    let result = serving
        .complete(&tool_request, b"tool", &mut call, &mut sink)
        .unwrap();
    let first_call = &message(&result)
        .find("tool_calls")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()[0];
    check(
        finish(&result).string().unwrap() == b"tool_calls"
            && first_call.find("id").unwrap().unwrap().string().unwrap() == b"tool_call_0",
    );
    let tool_chunks = chunks
        .borrow()
        .iter()
        .filter(|chunk| delta(chunk).find("tool_calls").unwrap().is_some())
        .count();
    check(tool_chunks == 1);
    let history_request = with(&plain("R"), "tools", tools());
    let history_request = with(
        &history_request,
        "messages",
        parse(
            r#"[{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},{"role":"tool","tool_call_id":"a","content":"file"},{"role":"user","content":"R"}]"#,
        ),
    );
    check(
        admit_chat(&history_request, b"tokspur")
            .unwrap()
            .manifest
            .len()
            == 1,
    );
    let mut bad_history = history_request
        .find("messages")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()
        .clone();
    bad_history.push(parse(
        r#"{"role":"tool","tool_call_id":"a","content":"duplicate"}"#,
    ));
    refusal_admission(admit_chat(
        &with(&history_request, "messages", Json::Array(bad_history)),
        b"tokspur",
    ));
    let mut bad_history = history_request
        .find("messages")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()
        .clone();
    bad_history[1] = with(&bad_history[1], "name", Json::from("other"));
    refusal_admission(admit_chat(
        &with(&history_request, "messages", Json::Array(bad_history)),
        b"tokspur",
    ));
    for corrupt in ["prefix", "policy", "terminal", "throw"] {
        chunks.borrow_mut().clear();
        fake.borrow_mut().corrupt = corrupt.to_owned();
        fake.borrow_mut().reply(tool_response);
        refusal(serving.complete(&tool_request, b"bad", &mut call, &mut sink));
        check(serving.committed_tokens() == 0);
        for chunk in chunks.borrow().iter() {
            check(delta(chunk).find("tool_calls").unwrap().is_none());
        }
    }
    fake.borrow_mut().corrupt.clear();
    fake.borrow_mut().reply("safe<|im_end|>");
    let result = serving
        .complete(&plain("A"), b"recovery", &mut call, &mut sink)
        .unwrap();
    check(finish(&result).string().unwrap() == b"stop");
    chunks.borrow_mut().clear();
    fake.borrow_mut().reply(tool_response);
    let limited = with(&tool_request, "max_tokens", Json::integer(10));
    let result = serving
        .complete(&limited, b"truncated", &mut call, &mut sink)
        .unwrap();
    check(
        finish(&result).string().unwrap() == b"length"
            && message(&result).find("tool_calls").unwrap().is_none(),
    );
    chunks.borrow_mut().clear();
    fake.borrow_mut().reply("not a call<|im_end|>");
    refusal(serving.complete(
        &with(&tool_request, "tool_choice", Json::from("required")),
        b"required",
        &mut call,
        &mut sink,
    ));
    for name in ["auto", "none", "required"] {
        let mut declarations = tools().array().unwrap().clone();
        declarations.push(Json::new_object([
            ("type", Json::from("function")),
            ("function", Json::new_object([("name", Json::from(name))])),
        ]));
        let named = with(&plain("named"), "tools", Json::Array(declarations));
        let named = with(
            &named,
            "tool_choice",
            Json::new_object([
                ("type", Json::from("function")),
                ("function", Json::new_object([("name", Json::from(name))])),
            ]),
        );
        let admitted = admit_chat(&named, b"tokspur").unwrap();
        check(admitted.names == vec![name.as_bytes().to_vec()]);
        check(
            admitted
                .render
                .find("tools")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .len()
                == 1,
        );
        let instruction = admitted
            .render
            .find("messages")
            .unwrap()
            .unwrap()
            .array()
            .unwrap()[0]
            .find("content")
            .unwrap()
            .unwrap()
            .string()
            .unwrap();
        check(
            instruction
                .windows(name.len() + 6)
                .any(|part| part == format!("named {name}").as_bytes()),
        );
        fake.borrow_mut().reply("not a call<|im_end|>");
        refusal(serving.complete(&named, b"named-missing", &mut call, &mut sink));
        fake.borrow_mut().reply(&format!(
            "<tool_call><function={name}></function></tool_call><|im_end|>"
        ));
        let selected = serving
            .complete(&named, b"named-ok", &mut call, &mut sink)
            .unwrap();
        check(finish(&selected).string().unwrap() == b"tool_calls");
        check(
            message(&selected)
                .find("tool_calls")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()[0]
                .find("function")
                .unwrap()
                .unwrap()
                .find("name")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == name.as_bytes(),
        );
    }
    cancel.set(true);
    let previous = fake.borrow().prefills + fake.borrow().generations;
    let previous_tokens = serving.committed_tokens();
    refusal(serving.complete(&plain("A"), b"cancelled", &mut call, &mut sink));
    check(
        fake.borrow().prefills + fake.borrow().generations == previous
            && serving.committed_tokens() == previous_tokens,
    );
    cancel.set(false);
    fake.borrow_mut().reply("more<|im_end|>");
    let mut cancel_after_commit = |job: &Json, event: Event<'_>| {
        let done = fake.borrow_mut().call(job, event);
        if job
            .find("generate")?
            .is_some_and(|n| n.unsigned_integer().is_ok_and(|n| n != 0))
        {
            cancel.set(true);
        }
        done
    };
    refusal(serving.complete(
        &request("C", true, false),
        b"midcancel",
        &mut cancel_after_commit,
        &mut sink,
    ));
    check(serving.committed_tokens() == 0);
    cancel.set(false);
    fake.borrow_mut()
        .reply("<tool_call><function=unknown></function></tool_call><|im_end|>");
    refusal(serving.complete(&tool_request, b"unknown", &mut call, &mut sink));
    fake.borrow_mut().reply("<tool_call><function=read_file><parameter=path>\ntrue\n</parameter></function></tool_call><|im_end|>");
    let result = serving
        .complete(&tool_request, b"rawstring", &mut call, &mut sink)
        .unwrap();
    let arguments = message(&result)
        .find("tool_calls")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()[0]
        .find("function")
        .unwrap()
        .unwrap()
        .find("arguments")
        .unwrap()
        .unwrap()
        .string()
        .unwrap()
        .to_vec();
    check(arguments == br#"{"path":"true"}"#);
    executor_enum_contract();
    executor_arg_contract();
    generator_schema_keywords();
    tool_schema_patterns();
    thinking_budget_contract();
    println!(
        "PASS owned serving protocol checks={} (model-free; no tools executed)",
        CHECKS.load(Ordering::Relaxed)
    );
}

fn executor_enum_contract() {
    let deferred = ToolEnumValidation::Executor;
    check(
        parse_tool_enum_validation("strict") == Ok(ToolEnumValidation::Strict)
            && parse_tool_enum_validation("executor") == Ok(deferred),
    );
    check(tool_enum_validation_policy(deferred) == "original-agent-executor-enums-v1");
    refusal_admission(parse_tool_enum_validation("permissive"));

    let declarations = parse(
        r#"[{"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string","enum":["cargo test --offline"]}},"required":["command"],"additionalProperties":false}}}]"#,
    );
    let job = with(&plain("A"), "tools", declarations);
    let literal = "cargo test --offline 2>&1 | head -60";
    let frame = format!(
        "<tool_call><function=run_command><parameter=command>\n{literal}\n</parameter></function></tool_call><|im_end|>"
    );
    let mut cancelled = || false;
    let mut seed = || 44u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut seed,
        prefix_trace: None,
    };

    let mut strict = ServeSession::new(
        codec(),
        SessionOptions {
            tool_enums: ToolEnumValidation::Strict,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    let rejected = RefCell::new(Fake::new());
    rejected.borrow_mut().reply(&frame);
    let mut rejected_call =
        |command: &Json, event: Event<'_>| rejected.borrow_mut().call(command, event);
    refusal(strict.complete(&job, b"strict-enum", &mut rejected_call, &mut sink));

    let mut executor = ServeSession::new(
        codec(),
        SessionOptions {
            tool_enums: deferred,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    let accepted = RefCell::new(Fake::new());
    accepted.borrow_mut().reply(&frame);
    let mut accepted_call =
        |command: &Json, event: Event<'_>| accepted.borrow_mut().call(command, event);
    let result = executor
        .complete(&job, b"executor-enum", &mut accepted_call, &mut sink)
        .unwrap();
    let proposed = message(&result)
        .find("tool_calls")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()[0]
        .clone();
    let args_json = parse_json(
        proposed
            .find("function")
            .unwrap()
            .unwrap()
            .find("arguments")
            .unwrap()
            .unwrap()
            .string()
            .unwrap(),
    )
    .unwrap();
    check(
        args_json
            .find("command")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == literal.as_bytes(),
    );

    let prior = vec![
        message(&result).clone(),
        Json::new_object([
            ("role", Json::from("tool")),
            (
                "tool_call_id",
                proposed.find("id").unwrap().unwrap().clone(),
            ),
            ("content", Json::from(r#"{"error":"command not allowed"}"#)),
        ]),
        Json::new_object([
            ("role", Json::from("user")),
            ("content", Json::from("continue")),
        ]),
    ];
    let next = with(&job, "messages", Json::Array(prior.clone()));
    check(
        admit_chat_with_policy(&next, b"tokspur", deferred)
            .unwrap()
            .manifest
            .len()
            == 1,
    );
    refusal_admission(admit_chat(&next, b"tokspur"));

    accepted.borrow_mut().reply("done<|im_end|>");
    let after = executor
        .complete(
            &next,
            b"after-executor-error",
            &mut accepted_call,
            &mut sink,
        )
        .unwrap();
    check(finish(&after).string().unwrap() == b"stop");

    for (value, allowed) in [
        (r#"{"command":1}"#, false),
        (r#"{"command":"bad","extra":true}"#, false),
        (r#"{}"#, false),
        (r#"{"command":"<|im_start|>"}"#, true),
        (r#"{"command":"<|endoftext|>"}"#, true),
    ] {
        let function = with(
            proposed.find("function").unwrap().unwrap(),
            "arguments",
            Json::from(value),
        );
        let mut bad = prior.clone();
        bad[0] = with(
            &bad[0],
            "tool_calls",
            Json::Array(vec![with(&proposed, "function", function)]),
        );
        check(
            admit_chat_with_policy(
                &with(&next, "messages", Json::Array(bad)),
                b"tokspur",
                deferred,
            )
            .is_ok()
                == allowed,
        );
    }

    for key in ["tool_enum_validation", "tool-enum-validation"] {
        refusal_admission(admit_chat_with_policy(
            &with(&job, key, Json::from("executor")),
            b"tokspur",
            deferred,
        ));
    }

    let nested = parse(
        r#"{"type":"object","properties":{"items":{"type":"array","items":{"type":"integer","enum":[1],"maximum":3}}},"required":["items"],"additionalProperties":false}"#,
    );
    validate_tool_value_with_policy(&nested, &parse(r#"{"items":[2,3]}"#), deferred).unwrap();
    check(true);
    refusal_admission(validate_tool_value(&nested, &parse(r#"{"items":[2]}"#)));
    for value in [
        r#"{"items":[4]}"#,
        r#"{"items":["2"]}"#,
        r#"{"items":[2],"extra":true}"#,
        r#"{}"#,
    ] {
        refusal_admission(validate_tool_value_with_policy(
            &nested,
            &parse(value),
            deferred,
        ));
    }

    let malformed = parse(
        r#"[{"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string","enum":[1]}}}}}]"#,
    );
    refusal_admission(admit_chat_with_policy(
        &with(&job, "tools", malformed),
        b"tokspur",
        deferred,
    ));

    for raw in [
        "<tool_call><function=unknown></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=extra>\nbad\n</parameter></function></tool_call><|im_end|>",
        "<tool_call><function=run_command></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=command>\n<|im_start|>\n</parameter></function></tool_call><|im_end|>",
        "<tool_call><function=run_command><parameter=command>\n<|endoftext|>\n</parameter></function></tool_call><|im_end|>",
    ] {
        accepted.borrow_mut().reply(raw);
        refusal(executor.complete(&job, b"still-refused", &mut accepted_call, &mut sink));
    }
}

/// Execute the production session and response parser in a child process with a generated target, without HTTP, model weights or GPU execution.
/// The refusal text of one call, or empty when it is admitted.
fn refusal_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => String::new(),
        Err(error) => error.to_string(),
    }
}

/// Checks one table row, naming the row when it differs.
#[track_caller]
fn expect_text(actual: String, expected: &str, row: &str) {
    if actual != expected {
        eprintln!("schema keyword row {row}: got '{actual}', expected '{expected}'");
    }
    check(actual == expected);
}

/// The same rows as `generator_schema_keywords` in `native/tests/test_serve_protocol.cpp`, with the same refusal text.
fn generator_schema_keywords() {
    // Tool declarations: each schema is one property of the parameters object. Empty text means admitted.
    for (schema, expected) in [
        (r#"{"type":"number","exclusiveMaximum":1}"#, ""),
        (
            r#"{"type":"integer","minimum":0,"exclusiveMaximum":10}"#,
            "",
        ),
        (
            r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}"#,
            "",
        ),
        (r#"{"type":"number","maximum":5,"exclusiveMaximum":5}"#, ""),
        (r#"{"type":"number","exclusiveMaximum":-1e300}"#, ""),
        (
            r#"{"type":"number","minimum":1,"exclusiveMaximum":1.0000000000000000001}"#,
            "",
        ),
        (
            r#"{"type":"number","exclusiveMaximum":true}"#,
            "expected JSON number",
        ),
        (
            r#"{"type":"number","exclusiveMaximum":"1"}"#,
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
            r#"{"type":"number","minimum":1,"exclusiveMaximum":1}"#,
            "inverted schema bounds",
        ),
        (
            r#"{"type":"number","minimum":1.0000000000000000001,"exclusiveMaximum":1}"#,
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
            r#"{"type":"number","exclusiveMaximum":0,"minimum":"x"}"#,
            "expected JSON number",
        ),
        (
            r#"{"anyOf":[{"type":"number"}],"exclusiveMaximum":1}"#,
            "unsupported API/schema field",
        ),
        (r#"{"type":"string","minLength":1,"maxLength":3}"#, ""),
        (r#"{"type":"string","minLength":0,"maxLength":0}"#, ""),
        (r#"{"type":"string","minLength":65536}"#, ""),
        (r#"{"type":"string","maxLength":65537}"#, ""),
        (
            r#"{"type":"string","minLength":65536,"maxLength":18446744073709551615}"#,
            "",
        ),
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
            r#"{"type":"string","maxLength":true,"minLength":1.5}"#,
            "JSON unsigned integer out of range or nonintegral",
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
        (r#"{"type":"string","enum":["abc"],"minLength":3}"#, ""),
        (
            r#"{"type":"string","enum":["ab"],"minLength":3}"#,
            "tool string below minLength",
        ),
        (
            r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#,
            "",
        ),
        (
            r#"{"type":"array","items":{"type":"string"},"uniqueItems":false}"#,
            "",
        ),
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
        (r#"{"type":"string","const":"a"}"#, ""),
        (r#"{"type":"integer","const":3}"#, ""),
        (r#"{"type":"number","const":1.50}"#, ""),
        (r#"{"type":"boolean","const":true}"#, ""),
        (r#"{"type":"null","const":null}"#, ""),
        (r#"{"type":"string","enum":["a"],"const":"a"}"#, ""),
        (r#"{"type":"string","const":1}"#, "tool value is not string"),
        (
            r#"{"type":"boolean","const":"true"}"#,
            "tool value is not boolean",
        ),
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
            r#"{"type":"number","const":1e301}"#,
            "API number outside supported range",
        ),
        (
            r#"{"type":"string","enum":["a","b"],"const":"a"}"#,
            "tool value differs from const",
        ),
        (
            r#"{"type":"string","enum":["a"],"const":"b"}"#,
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
        (r#"{"type":"integer","multipleOf":5}"#, ""),
        (r#"{"type":"number","multipleOf":0.01}"#, ""),
        (r#"{"type":"number","multipleOf":1e-7}"#, ""),
        (r#"{"type":"number","multipleOf":0.30000000000000004}"#, ""),
        (r#"{"type":"number","multipleOf":123456789012345678}"#, ""),
        (r#"{"type":"number","multipleOf":1234567890123456780}"#, ""),
        (
            r#"{"type":"number","multipleOf":1234567890123456789}"#,
            "multipleOf digit bound",
        ),
        (
            r#"{"type":"number","multipleOf":0}"#,
            "multipleOf must be positive",
        ),
        (
            r#"{"type":"number","multipleOf":0.0}"#,
            "multipleOf must be positive",
        ),
        (
            r#"{"type":"number","multipleOf":-2}"#,
            "multipleOf must be positive",
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
            r#"{"type":"number","minimum":2,"maximum":1,"multipleOf":0}"#,
            "inverted schema bounds",
        ),
        (
            r#"{"type":"string","multipleOf":2}"#,
            "numeric bound on other schema type",
        ),
        (
            r#"{"anyOf":[{"type":"integer"}],"multipleOf":2}"#,
            "unsupported API/schema field",
        ),
        (r#"{"type":"integer","enum":[4,6],"multipleOf":2}"#, ""),
        (
            r#"{"type":"integer","enum":[4,5],"multipleOf":2}"#,
            "tool value is not a multiple of multipleOf",
        ),
        (
            r#"{"type":"integer","enum":[4],"multipleOf":0}"#,
            "multipleOf must be positive",
        ),
    ] {
        let declaration = tools().array().unwrap()[0].clone();
        let function = with(
            declaration.find("function").unwrap().unwrap(),
            "parameters",
            parse(&format!(
                r#"{{"type":"object","properties":{{"p":{schema}}}}}"#
            )),
        );
        let tools = Json::Array(vec![with(&declaration, "function", function)]);
        expect_text(
            refusal_text(admit_chat(&with(&plain("A"), "tools", tools), b"tokspur")),
            expected,
            schema,
        );
    }
    // Strict validation of one value against one schema.
    let validation: &[(&str, &[(&str, &str)])] = &[
        (
            r#"{"type":"number","minimum":0,"exclusiveMaximum":1}"#,
            &[
                (r#"0"#, ""),
                (r#"-0.0"#, ""),
                (r#"0.5"#, ""),
                (r#"9.9e-1"#, ""),
                (r#"0.99999999999999999999"#, ""),
                (r#"1"#, "tool value at or above exclusive maximum"),
                (r#"1.0"#, "tool value at or above exclusive maximum"),
                (r#"1e0"#, "tool value at or above exclusive maximum"),
                (
                    r#"1.00000000000000000001"#,
                    "tool value at or above exclusive maximum",
                ),
                (r#"2"#, "tool value at or above exclusive maximum"),
                (r#"-0.1"#, "tool value below minimum"),
            ],
        ),
        (
            r#"{"type":"integer","exclusiveMinimum":1,"exclusiveMaximum":4}"#,
            &[
                (r#"2"#, ""),
                (r#"3"#, ""),
                (r#"1"#, "tool value at or below exclusive minimum"),
                (r#"4"#, "tool value at or above exclusive maximum"),
            ],
        ),
        (
            r#"{"type":"number","maximum":1,"exclusiveMaximum":2}"#,
            &[
                (r#"1"#, ""),
                (r#"1.5"#, "tool value above maximum"),
                (r#"2"#, "tool value above maximum"),
            ],
        ),
        (
            r#"{"type":"number","exclusiveMaximum":-1.5}"#,
            &[
                (r#"-1.6"#, ""),
                (r#"-1.5"#, "tool value at or above exclusive maximum"),
                (r#"-1.50"#, "tool value at or above exclusive maximum"),
                (r#"-1.4"#, "tool value at or above exclusive maximum"),
            ],
        ),
        (
            r#"{"type":"string","minLength":2,"maxLength":3}"#,
            &[
                (r#""ab""#, ""),
                (r#""abc""#, ""),
                (r#""""#, "tool string below minLength"),
                (r#""a""#, "tool string below minLength"),
                (r#""abcd""#, "tool string above maxLength"),
                (r#""é""#, "tool string below minLength"),
                (r#""éé""#, ""),
                (r#""日本語""#, ""),
                (r#""日本語x""#, "tool string above maxLength"),
                (r#""😀""#, "tool string below minLength"),
                (r#""😀😀""#, ""),
                (r#""😀😀😀""#, ""),
                (r#""😀😀😀😀""#, "tool string above maxLength"),
                (r#""\ud83d\ude00""#, "tool string below minLength"),
                (r#""\u0061\u0062""#, ""),
                (
                    r#""\ud83d\ude00\ud83d\ude00\ud83d\ude00\ud83d\ude00""#,
                    "tool string above maxLength",
                ),
                (r#""e\u0301""#, ""),
                (r#"5"#, "tool value is not string"),
            ],
        ),
        (
            r#"{"type":"string","minLength":0,"maxLength":0}"#,
            &[(r#""""#, ""), (r#""a""#, "tool string above maxLength")],
        ),
        (
            r#"{"type":"array","items":{"type":"number"},"uniqueItems":true}"#,
            &[
                (r#"[]"#, ""),
                (r#"[1]"#, ""),
                (r#"[1,2,3]"#, ""),
                (r#"[1,1.00000000000000000001]"#, ""),
                (r#"[1,1]"#, "tool array repeats an item"),
                (r#"[1,2,1]"#, "tool array repeats an item"),
                (r#"[1,1.0]"#, "tool array repeats an item"),
                (r#"[1,1e0]"#, "tool array repeats an item"),
                (r#"[10,1e1]"#, "tool array repeats an item"),
                (r#"[0,-0.0]"#, "tool array repeats an item"),
                (r#"[1.5,15e-1]"#, "tool array repeats an item"),
                (r#"[1,1,"x"]"#, "expected JSON number"),
            ],
        ),
        (
            r#"{"type":"array","items":{"type":"number"},"uniqueItems":false}"#,
            &[(r#"[1,1]"#, "")],
        ),
        (
            r#"{"type":"array","items":{"type":"integer"},"minItems":2,"uniqueItems":true}"#,
            &[
                (r#"[1,2]"#, ""),
                (r#"[1]"#, "tool array below minItems"),
                (r#"[1,1]"#, "tool array repeats an item"),
            ],
        ),
        (
            r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#,
            &[
                (r#"["a","b"]"#, ""),
                (r#"["a","A"]"#, ""),
                (r#"["é","e\u0301"]"#, ""),
                (r#"["a","a"]"#, "tool array repeats an item"),
                (r#"["\u0061","a"]"#, "tool array repeats an item"),
            ],
        ),
        (
            r#"{"type":"array","items":{"anyOf":[{"type":"integer"},{"type":"string"},{"type":"boolean"},{"type":"null"}]},"uniqueItems":true}"#,
            &[
                (r#"[1,"1",true,null]"#, ""),
                (r#"[1,"1e0"]"#, ""),
                (r#"[0,false]"#, ""),
                (r#"["true",true]"#, ""),
                (r#"["null",null]"#, ""),
                (r#"[null,null]"#, "tool array repeats an item"),
                (r#"[true,true]"#, "tool array repeats an item"),
                (r#"[1,"1",1]"#, "tool array repeats an item"),
            ],
        ),
        (
            r#"{"type":"array","items":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"string"}}},"uniqueItems":true}"#,
            &[
                (r#"[{"a":1,"b":"x"},{"a":1,"b":"y"}]"#, ""),
                (r#"[{"a":1},{"b":"1"}]"#, ""),
                (r#"[{"a":1},{}]"#, ""),
                (
                    r#"[{"a":1,"b":"x"},{"b":"x","a":1.0}]"#,
                    "tool array repeats an item",
                ),
                (r#"[{},{}]"#, "tool array repeats an item"),
            ],
        ),
        (
            r#"{"type":"array","items":{"type":"array","items":{"type":"number"}},"uniqueItems":true}"#,
            &[
                (r#"[[1,2],[2,1]]"#, ""),
                (r#"[[1],[1,1]]"#, ""),
                (r#"[[],[0]]"#, ""),
                (r#"[[1,2],[1.0,2e0]]"#, "tool array repeats an item"),
                (r#"[[],[]]"#, "tool array repeats an item"),
            ],
        ),
        (
            r#"{"type":"string","const":"a"}"#,
            &[
                (r#""a""#, ""),
                (r#""\u0061""#, ""),
                (r#""b""#, "tool value differs from const"),
                (r#""A""#, "tool value differs from const"),
                (r#"1"#, "tool value is not string"),
            ],
        ),
        (
            r#"{"type":"number","const":1.50}"#,
            &[
                (r#"1.5"#, ""),
                (r#"15e-1"#, ""),
                (r#"1.50000000000000000001"#, "tool value differs from const"),
                (r#"1"#, "tool value differs from const"),
            ],
        ),
        (
            r#"{"type":"integer","const":0}"#,
            &[
                (r#"0"#, ""),
                (r#"-0"#, ""),
                (r#"1"#, "tool value differs from const"),
            ],
        ),
        (
            r#"{"type":"boolean","const":true}"#,
            &[
                (r#"true"#, ""),
                (r#"false"#, "tool value differs from const"),
            ],
        ),
        (
            r#"{"type":"null","const":null}"#,
            &[(r#"null"#, ""), (r#"0"#, "tool value is not null")],
        ),
        (
            r#"{"type":"integer","multipleOf":5}"#,
            &[
                (r#"0"#, ""),
                (r#"5"#, ""),
                (r#"-10"#, ""),
                (r#"7"#, "tool value is not a multiple of multipleOf"),
                (r#"-3"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":0.01}"#,
            &[
                (r#"1.23"#, ""),
                (r#"0.1"#, ""),
                (r#"1e2"#, ""),
                (r#"-4.5"#, ""),
                (r#"0.0"#, ""),
                (r#"1.234"#, "tool value is not a multiple of multipleOf"),
                (r#"1e-3"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":0.1}"#,
            &[
                (r#"0.3"#, ""),
                (r#"1e300"#, ""),
                (
                    r#"0.30000000000000004"#,
                    "tool value is not a multiple of multipleOf",
                ),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":1.5}"#,
            &[
                (r#"3"#, ""),
                (r#"4.5"#, ""),
                (r#"3e1"#, ""),
                (r#"1.5e299"#, ""),
                (r#"1"#, "tool value is not a multiple of multipleOf"),
                (r#"1e1"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":2.5}"#,
            &[
                (r#"5"#, ""),
                (r#"7.5"#, ""),
                (r#"1e1"#, ""),
                (r#"1"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":8}"#,
            &[
                (r#"1e3"#, ""),
                (r#"1e300"#, ""),
                (r#"1e2"#, "tool value is not a multiple of multipleOf"),
                (r#"4"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":3}"#,
            &[
                (r#"3e299"#, ""),
                (r#"9007199254740993"#, ""),
                (
                    r#"9007199254740992"#,
                    "tool value is not a multiple of multipleOf",
                ),
                (r#"1e300"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":1e-7}"#,
            &[
                (r#"0.0000001"#, ""),
                (r#"1"#, ""),
                (
                    r#"0.00000005"#,
                    "tool value is not a multiple of multipleOf",
                ),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":123456789012345678}"#,
            &[
                (r#"123456789012345678"#, ""),
                (r#"246913578024691356"#, ""),
                (r#"1234567890123456780"#, ""),
                (
                    r#"123456789012345679"#,
                    "tool value is not a multiple of multipleOf",
                ),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":576460752303423488}"#,
            &[
                (r#"1e59"#, ""),
                (r#"1e60"#, ""),
                (r#"1e58"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":298023223876953125}"#,
            &[
                (r#"1e25"#, ""),
                (r#"1e24"#, "tool value is not a multiple of multipleOf"),
            ],
        ),
        (
            r#"{"type":"number","multipleOf":999999999999999999}"#,
            &[
                (r#"999999999999999999"#, ""),
                (r#"122999999999999999877"#, ""),
                (r#"999999999999999999e40"#, ""),
                (
                    r#"999999999999999998999999999999999998"#,
                    "tool value is not a multiple of multipleOf",
                ),
            ],
        ),
        (
            r#"{"type":"integer","minimum":0,"exclusiveMaximum":100,"multipleOf":25}"#,
            &[
                (r#"75"#, ""),
                (r#"0"#, ""),
                (r#"100"#, "tool value at or above exclusive maximum"),
                (r#"30"#, "tool value is not a multiple of multipleOf"),
                (r#"50.0"#, "tool integer must use exact safe-integer syntax"),
                (r#"-25"#, "tool value below minimum"),
            ],
        ),
    ];
    for (schema, rows) in validation {
        let parsed = parse(schema);
        for (value, expected) in *rows {
            expect_text(
                refusal_text(validate_tool_value(&parsed, &parse(value))),
                expected,
                &format!("{schema} {value}"),
            );
        }
    }
    // A value built in memory can hold bytes that are not UTF-8. Its length cannot be counted, so it is refused as the JSON codec refuses it.
    expect_text(
        refusal_text(validate_tool_value(
            &parse(r#"{"type":"string","maxLength":4}"#),
            &Json::String(vec![b'a', 0xff]),
        )),
        "invalid UTF-8 input",
        "bytes that are not UTF-8",
    );
    // With enum validation deferred to the executor only const is deferred with it; every bound is still enforced.
    for (schema, value, expected) in [
        (r#"{"type":"string","const":"a"}"#, r#""b""#, ""),
        (
            r#"{"type":"string","const":"a"}"#,
            r#"1"#,
            "tool value is not string",
        ),
        (
            r#"{"type":"number","const":1,"exclusiveMaximum":2}"#,
            r#"1.5"#,
            "",
        ),
        (
            r#"{"type":"number","const":1,"exclusiveMaximum":2}"#,
            r#"2"#,
            "tool value at or above exclusive maximum",
        ),
        (
            r#"{"type":"integer","multipleOf":5}"#,
            r#"7"#,
            "tool value is not a multiple of multipleOf",
        ),
        (
            r#"{"type":"string","maxLength":1}"#,
            r#""ab""#,
            "tool string above maxLength",
        ),
        (
            r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#,
            r#"[1,1]"#,
            "tool array repeats an item",
        ),
    ] {
        expect_text(
            refusal_text(validate_tool_value_with_policy(
                &parse(schema),
                &parse(value),
                ToolEnumValidation::Executor,
            )),
            expected,
            &format!("{schema} {value}"),
        );
    }
    // A recorded call: strict argument validation refuses each violation with its own text, and executor validation defers all of them.
    let declared = parse(
        r#"[{"type":"function","function":{"name":"save","parameters":{"type":"object","properties":{"name":{"type":"string","minLength":2,"maxLength":4},"tags":{"type":"array","items":{"type":"string"},"uniqueItems":true},"ratio":{"type":"number","exclusiveMaximum":1},"step":{"type":"integer","multipleOf":5},"kind":{"type":"string","const":"file"}},"additionalProperties":false}}}]"#,
    );
    for (arguments, strict, deferred_enum) in [
        (
            r#"{"name":"ab","tags":["a","b"],"ratio":0.5,"step":10,"kind":"file"}"#,
            "",
            "",
        ),
        (
            r#"{"name":"a"}"#,
            "tool string below minLength",
            "tool string below minLength",
        ),
        (
            r#"{"name":"abcde"}"#,
            "tool string above maxLength",
            "tool string above maxLength",
        ),
        (
            r#"{"tags":["a","a"]}"#,
            "tool array repeats an item",
            "tool array repeats an item",
        ),
        (
            r#"{"ratio":1}"#,
            "tool value at or above exclusive maximum",
            "tool value at or above exclusive maximum",
        ),
        (
            r#"{"step":7}"#,
            "tool value is not a multiple of multipleOf",
            "tool value is not a multiple of multipleOf",
        ),
        (r#"{"kind":"dir"}"#, "tool value differs from const", ""),
    ] {
        let history = parse(&format!(
            r#"[{{"role":"user","content":"save it"}},{{"role":"assistant","tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"save","arguments":{arguments}}}}}]}},{{"role":"tool","tool_call_id":"call_1","content":"saved"}},{{"role":"user","content":"again"}}]"#
        ));
        let job = with(
            &with(&plain("A"), "tools", declared.clone()),
            "messages",
            history,
        );
        expect_text(
            refusal_text(admit_chat(&job, b"tokspur")),
            strict,
            arguments,
        );
        expect_text(
            refusal_text(admit_chat_with_policy(
                &job,
                b"tokspur",
                ToolEnumValidation::Executor,
            )),
            deferred_enum,
            arguments,
        );
        expect_text(
            refusal_text(admit_chat_with_policies(
                &job,
                b"tokspur",
                ToolEnumValidation::Strict,
                ToolArgValidation::Executor,
            )),
            "",
            arguments,
        );
    }
    // A generated call takes the same validator: the strict session refuses with its text and the executor session passes the call on.
    let job = with(&plain("A"), "tools", declared);
    let mut cancelled = || false;
    let mut seed = || 44u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut seed,
        prefix_trace: None,
    };
    for (parameter, value, expected) in [
        ("name", r#"abc"#, ""),
        ("name", r#"a"#, "tool string below minLength"),
        ("name", r#"abcde"#, "tool string above maxLength"),
        ("tags", r#"["a","a"]"#, "tool array repeats an item"),
        ("ratio", r#"1"#, "tool value at or above exclusive maximum"),
        ("step", r#"7"#, "tool value is not a multiple of multipleOf"),
        ("kind", r#"dir"#, "tool value differs from const"),
    ] {
        let frame = format!(
            "<tool_call><function=save><parameter={parameter}>\n{value}\n</parameter></function></tool_call><|im_end|>"
        );
        let mut strict = ServeSession::new(codec(), SessionOptions::default()).unwrap();
        let rejected = RefCell::new(Fake::new());
        rejected.borrow_mut().reply(&frame);
        let mut rejected_call =
            |command: &Json, event: Event<'_>| rejected.borrow_mut().call(command, event);
        expect_text(
            refusal_text(strict.complete(&job, b"strict-keyword", &mut rejected_call, &mut sink)),
            expected,
            &frame,
        );
        let mut executor = ServeSession::new(
            codec(),
            SessionOptions {
                tool_args: ToolArgValidation::Executor,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let accepted = RefCell::new(Fake::new());
        accepted.borrow_mut().reply(&frame);
        let mut accepted_call =
            |command: &Json, event: Event<'_>| accepted.borrow_mut().call(command, event);
        let result = executor
            .complete(&job, b"executor-keyword", &mut accepted_call, &mut sink)
            .unwrap();
        check(
            message(&result)
                .find("tool_calls")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .len()
                == 1,
        );
    }
}

/// The same rows as `tool_schema_patterns` in `native/tests/test_serve_protocol.cpp`, with the same refusal text. Both are written from one table.
fn tool_schema_patterns() {
    // Tool declarations: each schema is one property of the parameters object. Empty text means admitted.
    for (schema, expected) in [
        (
            r#"{"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?=.{1,253}\\.?$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[-0-9a-zA-Z]{0,61}[0-9a-zA-Z])?)*\\.?$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^P(?:(\\d+W)|(?!.*W)(?=\\d|T\\d)(\\d+Y)?(\\d+M)?(\\d+D)?(T(?=\\d)(\\d+H)?(\\d+M)?(\\d+([.,]\\d+)?S)?)?)$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?:(?:\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|(?:02)-(?:0[1-9]|1\\d|2[0-8])))T(?:(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?(?:Z|([+-](?:[01]\\d|2[0-3]):[0-5]\\d)))$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^$|^(?:[0-9a-zA-Z+/]{4})*(?:(?:[0-9a-zA-Z+/]{2}==)|(?:[0-9a-zA-Z+/]{3}=))?$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*(?:\\d{0,3}|(?=[\\d.]{1,6}0*$)\\d{0,3}\\.\\d{0,2}0*$)"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":"^[a-zA-Z0-9_-]{21}$"}"#, ""),
        (r#"{"type":"string","pattern":"^[cC][^\\s-]{8,}$"}"#, ""),
        (r#"{"type":"string","pattern":"^[0-9a-z]+$"}"#, ""),
        (
            r#"{"type":"string","pattern":"^[0-9A-HJKMNP-TV-Za-hjkmnp-tv-z]{26}$"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":"^[0-9a-vA-V]{20}$"}"#, ""),
        (
            r#"{"type":"string","pattern":"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\/([0-9]|[1-2][0-9]|3[0-2])$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|::|([0-9a-fA-F]{1,4})?::([0-9a-fA-F]{1,4}:?){0,6})\\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":"^[A-Za-z0-9_-]*$"}"#, ""),
        (
            r#"{"type":"string","pattern":"^\\+(?:[0-9]){6,14}[0-9]$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^https:\\/\\/a b_c-d\\..*"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":".*\\.json$"}"#, ""),
        (r#"{"type":"string","pattern":"a\\+b \\(c\\)"}"#, ""),
        (
            r#"{"type":"string","pattern":"^id--?\\d+(?:\\.\\d+)?-(a|b)$"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":"^[^A-Z]*$"}"#, ""),
        (
            r#"{"type":"string","pattern":"^[0-9a-fA-F]{8}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{12}$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(([a-f0-9]{1,4}:){7}|::([a-f0-9]{1,4}:){0,6}|([a-f0-9]{1,4}:){1}:([a-f0-9]{1,4}:){0,5}|([a-f0-9]{1,4}:){2}:([a-f0-9]{1,4}:){0,4}|([a-f0-9]{1,4}:){3}:([a-f0-9]{1,4}:){0,3}|([a-f0-9]{1,4}:){4}:([a-f0-9]{1,4}:){0,2}|([a-f0-9]{1,4}:){5}:([a-f0-9]{1,4}:){0,1})([a-f0-9]{1,4}|(((25[0-5])|(2[0-4][0-9])|(1[0-9]{2})|([0-9]{1,2}))\\.){3}((25[0-5])|(2[0-4][0-9])|(1[0-9]{2})|([0-9]{1,2})))$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]*$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^https\\:\\/\\/a\\ b\\_c\\-d\\."}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*(?:\\d{0,8}|(?=[\\d.]{1,9}0*$)\\d*\\.\\d*0*$)"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d{0,3}0*$"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^\\s*(\\d*\\.?\\d+)\\s*(\\w+)?"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^\\d{3}-\\d{2}-\\d{4}$"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":"^[a-z_][a-z0-9_]*$"}"#, ""),
        (
            r#"{"type":"string","pattern":"^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"^(?P<area>\\d{3})-\\d{4}$"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(a)\\1"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\1"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\9"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?<n>a)"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?<n>a)\\k<n>"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\k<n>"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\p{L}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\P{Lu}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[\\p{L}]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?i)a"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?i:a)"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?#c)a"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\uD83D"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\uDE00"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\uD83Dx"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\uD83D\\u0041"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\u{D800}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\\ud83d\ude00"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[\\\ud83d\ude00]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\a"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\e"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\A"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\Z"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\z"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\G"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\Q"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[\\B]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[\\1]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[\\c1]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\c1"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\c"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\00"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\01"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\x1"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\xg0"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\u12"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\u{}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\u{110000}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"\\u{1F600"}"#,
            "unsupported pattern",
        ),
        (r#"{"type":"string","pattern":"\\"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"*"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"+a"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"?"}"#, "unsupported pattern"),
        (
            r#"{"type":"string","pattern":"a**"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a+*"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a*??"}"#,
            "unsupported pattern",
        ),
        (r#"{"type":"string","pattern":"a{"}"#, "unsupported pattern"),
        (
            r#"{"type":"string","pattern":"a{1"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a{,2}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a{1,2"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a{2,1}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a{x}"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"{1}"}"#,
            "unsupported pattern",
        ),
        (r#"{"type":"string","pattern":"a}"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"a]"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"^*"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"$+"}"#, "unsupported pattern"),
        (
            r#"{"type":"string","pattern":"\\b?"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?=a)*"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?<!a){2}"}"#,
            "unsupported pattern",
        ),
        (r#"{"type":"string","pattern":"("}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":")"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"(a"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"a)"}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"(?"}"#, "unsupported pattern"),
        (
            r#"{"type":"string","pattern":"(?a)"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"(?<"}"#,
            "unsupported pattern",
        ),
        (r#"{"type":"string","pattern":"["}"#, "unsupported pattern"),
        (r#"{"type":"string","pattern":"[a"}"#, "unsupported pattern"),
        (
            r#"{"type":"string","pattern":"[a-"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[b-a]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[\\d-z]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[a-\\w]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"[a--]"}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"a{4096}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(?:a{64}){64}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"a{0,2048}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(?:(?:(?:a{20}){20}){20}){20}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(){4097}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"a{5000,4000}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"a{99999999999999999999999999}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"a{4096,}"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(((((((((((((((((((((((((((((((((a)))))))))))))))))))))))))))))))))"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=(?=a"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (
            r#"{"type":"string","pattern":"(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})(?:(?:(?=a)){0})"}"#,
            "pattern exceeds the tool pattern bound",
        ),
        (r#"{"type":"string","pattern":"a{4095}"}"#, ""),
        (r#"{"type":"string","pattern":"(?:a{63}){65}"}"#, ""),
        (r#"{"type":"string","pattern":"a{0,2047}"}"#, ""),
        (r#"{"type":"string","pattern":"(?:(?:(?:a*)*)*)*"}"#, ""),
        (r#"{"type":"string","pattern":"(){4096}"}"#, ""),
        (
            r#"{"type":"string","pattern":"((((((((((((((((((((((((((((((((a))))))))))))))))))))))))))))))))"}"#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)(?=a)"}"#,
            "",
        ),
        (r#"{"type":"string","pattern":7}"#, "expected JSON string"),
        (
            r#"{"type":"string","pattern":null}"#,
            "expected JSON string",
        ),
        (
            r#"{"type":"integer","pattern":"^1"}"#,
            "string schema keyword on other type",
        ),
        (
            r#"{"type":"array","items":{"type":"string"},"pattern":"a"}"#,
            "string schema keyword on other type",
        ),
        (
            r#"{"type":"array","items":{"type":"string","pattern":"^a"}}"#,
            "",
        ),
        (
            r#"{"anyOf":[{"type":"string"}],"pattern":"a"}"#,
            "unsupported API/schema field",
        ),
        (
            r#"{"anyOf":[{"type":"number"},{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"}]}"#,
            "",
        ),
        (
            r#"{"anyOf":[{"type":"number"},{"type":"string","pattern":"(a)\\1"}]}"#,
            "unsupported pattern",
        ),
        (r#"{"type":"string","pattern":"^a","enum":["ab","ac"]}"#, ""),
        (
            r#"{"type":"string","pattern":"^a","enum":["ab","b"]}"#,
            "tool string does not match pattern",
        ),
        (r#"{"type":"string","pattern":"^a","const":"ab"}"#, ""),
        (
            r#"{"type":"string","pattern":"^a","const":"b"}"#,
            "tool string does not match pattern",
        ),
        (
            r#"{"type":"string","pattern":"(","minLength":3,"maxLength":2}"#,
            "inverted schema bounds",
        ),
        (
            r#"{"type":"string","pattern":"(","minLength":65537}"#,
            "minLength exceeds the tool call byte limit",
        ),
        (
            r#"{"type":"string","pattern":"(","enum":[1]}"#,
            "unsupported pattern",
        ),
        (
            r#"{"type":"string","pattern":"^b","enum":["abc"],"minLength":4}"#,
            "tool string below minLength",
        ),
        (
            r#"{"type":"string","pattern":"^b","enum":["abc"],"minLength":3}"#,
            "tool string does not match pattern",
        ),
    ] {
        let declaration = tools().array().unwrap()[0].clone();
        let function = with(
            declaration.find("function").unwrap().unwrap(),
            "parameters",
            parse(&format!(
                r#"{{"type":"object","properties":{{"p":{schema}}}}}"#
            )),
        );
        let tools = Json::Array(vec![with(&declaration, "function", function)]);
        expect_text(
            refusal_text(admit_chat(&with(&plain("A"), "tools", tools), b"tokspur")),
            expected,
            schema,
        );
    }
    // The pattern text is bounded too, whatever it compiles to: a class of 16382 members is one step.
    for (members, expected) in [
        (16382, ""),
        (16383, "pattern exceeds the tool pattern bound"),
    ] {
        let schema = Json::Object(vec![
            (b"type".to_vec(), Json::String(b"string".to_vec())),
            (
                b"pattern".to_vec(),
                Json::String(format!("[{}]", "a".repeat(members)).into_bytes()),
            ),
        ]);
        expect_text(
            refusal_text(validate_tool_value(&schema, &Json::String(b"a".to_vec()))),
            expected,
            "pattern text length",
        );
    }
    // Strict validation of one string against one pattern: the strings it matches, then the strings it does not.
    let matching: &[(&str, &[&str], &[&str])] = &[
        (
            r#"{"type":"string","pattern":"b"}"#,
            &[r#""b""#, r#""abc""#],
            &[r#""""#, r#""a""#, r#""B""#],
        ),
        (
            r#"{"type":"string","pattern":"^b"}"#,
            &[r#""b""#, r#""bc""#],
            &[r#""ab""#, r#""a\nb""#],
        ),
        (
            r#"{"type":"string","pattern":"b$"}"#,
            &[r#""b""#, r#""ab""#],
            &[r#""ba""#, r#""b\n""#],
        ),
        (
            r#"{"type":"string","pattern":"^$"}"#,
            &[r#""""#],
            &[r#""a""#, r#""\n""#],
        ),
        (
            r#"{"type":"string","pattern":""}"#,
            &[r#""""#, r#""a""#],
            &[],
        ),
        (
            r#"{"type":"string","pattern":"^a|b$"}"#,
            &[r#""ax""#, r#""xb""#],
            &[r#""xa""#, r#""bx""#],
        ),
        (
            r#"{"type":"string","pattern":"a^b"}"#,
            &[],
            &[r#""ab""#, r#""a\nb""#],
        ),
        (
            r#"{"type":"string","pattern":"a$b"}"#,
            &[],
            &[r#""ab""#, r#""a\nb""#],
        ),
        (
            r#"{"type":"string","pattern":"^[a-c]$"}"#,
            &[r#""a""#, r#""c""#],
            &[r#""d""#, r#""-""#, r#""ab""#],
        ),
        (
            r#"{"type":"string","pattern":"^[^a-c]$"}"#,
            &[r#""d""#, r#""\n""#, r#""\ud83d\ude00""#],
            &[r#""a""#, r#""""#],
        ),
        (
            r#"{"type":"string","pattern":"^[-a]$"}"#,
            &[r#""-""#, r#""a""#],
            &[r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^[a-]$"}"#,
            &[r#""-""#, r#""a""#],
            &[r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^[a\\-c]$"}"#,
            &[r#""a""#, r#""-""#, r#""c""#],
            &[r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^[+--]$"}"#,
            &[r#""+""#, r#"",""#, r#""-""#],
            &[r#"".""#],
        ),
        (
            r#"{"type":"string","pattern":"^[a-c-e]$"}"#,
            &[r#""b""#, r#""-""#, r#""e""#],
            &[r#""d""#],
        ),
        (
            r#"{"type":"string","pattern":"^[A-Za-z0-9-_]$"}"#,
            &[r#""Q""#, r#""5""#, r#""-""#, r#""_""#],
            &[r#""+""#],
        ),
        (
            r#"{"type":"string","pattern":"^[\\d-]$"}"#,
            &[r#""7""#, r#""-""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^[\\w.]$"}"#,
            &[r#""a""#, r#""_""#, r#"".""#],
            &[r#""-""#],
        ),
        (
            r#"{"type":"string","pattern":"^[^\\s-]$"}"#,
            &[r#""a""#],
            &[r#"" ""#, r#""-""#, r#""\t""#],
        ),
        (
            r#"{"type":"string","pattern":"^[\\b]$"}"#,
            &[r#""\b""#],
            &[r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^[.*+?(){}|^$\\[\\]\\\\]$"}"#,
            &[r#"".""#, r#""{""#, r#""^""#, r#""[""#, r#""]""#, r#""\\""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^[]$"}"#,
            &[],
            &[r#""""#, r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^[^]$"}"#,
            &[r#""a""#, r#""\n""#, r#""\udbff\udfff""#],
            &[r#""""#, r#""ab""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\d$"}"#,
            &[r#""0""#, r#""9""#],
            &[r#""\u0663""#, r#""a""#, r#""\uff11""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\D$"}"#,
            &[r#""\u0663""#, r#""a""#, r#""\n""#],
            &[r#""0""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\w$"}"#,
            &[r#""a""#, r#""Z""#, r#""0""#, r#""_""#],
            &[r#""\u00e9""#, r#""-""#, r#""\u017f""#, r#""\u212a""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\W$"}"#,
            &[r#""\u00e9""#, r#""-""#, r#"" ""#],
            &[r#""a""#, r#""_""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\s$"}"#,
            &[
                r#""\t""#,
                r#""\n""#,
                r#""\u000b""#,
                r#""\f""#,
                r#""\r""#,
                r#"" ""#,
                r#""\u00a0""#,
                r#""\u1680""#,
                r#""\u2000""#,
                r#""\u200a""#,
                r#""\u2028""#,
                r#""\u2029""#,
                r#""\u202f""#,
                r#""\u205f""#,
                r#""\u3000""#,
                r#""\ufeff""#,
            ],
            &[
                r#""\u0085""#,
                r#""\u180e""#,
                r#""\u200b""#,
                r#""\u2060""#,
                r#""a""#,
                r#""\u001c""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^\\S$"}"#,
            &[r#""\u0085""#, r#""\u180e""#, r#""\u200b""#, r#""a""#],
            &[r#"" ""#, r#""\u00a0""#, r#""\u2028""#, r#""\ufeff""#],
        ),
        (
            r#"{"type":"string","pattern":"^.$"}"#,
            &[
                r#""a""#,
                r#""\u0085""#,
                r#""\t""#,
                r#""\u000b""#,
                r#""\ud83d\ude00""#,
            ],
            &[r#""\n""#, r#""\r""#, r#""\u2028""#, r#""\u2029""#, r#""""#],
        ),
        (
            r#"{"type":"string","pattern":"^.{2}$"}"#,
            &[r#""e\u0301""#, r#""\ud83d\ude00a""#],
            &[r#""\ud83d\ude00""#, r#""\u00e9""#],
        ),
        (
            r#"{"type":"string","pattern":"^\ud83d\ude00+$"}"#,
            &[r#""\ud83d\ude00""#, r#""\ud83d\ude00\ud83d\ude00""#],
            &[r#""""#, r#""\ud83d\ude01""#],
        ),
        (
            r#"{"type":"string","pattern":"^[\ud83d\ude00-\ud83d\ude4f]$"}"#,
            &[r#""\ud83d\ude00""#, r#""\ud83d\ude4f""#],
            &[r#""\ud83d\ude50""#, r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\u{1F600}\\uD83D\\uDE00$"}"#,
            &[r#""\ud83d\ude00\ud83d\ude00""#],
            &[r#""\ud83d\ude00""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\u00e9\\xe9\\u{0000e9}$"}"#,
            &[r#""\u00e9\u00e9\u00e9""#],
            &[r#""eee""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\t\\n\\v\\f\\r\\0\\cJ\\cj$"}"#,
            &[r#""\t\n\u000b\f\r\u0000\n\n""#],
            &[r#""\t\n\u000b\f\r0\n\n""#],
        ),
        (
            r#"{"type":"string","pattern":"^a*$"}"#,
            &[r#""""#, r#""aaa""#],
            &[r#""b""#, r#""ab""#],
        ),
        (
            r#"{"type":"string","pattern":"^a+?$"}"#,
            &[r#""a""#, r#""aaa""#],
            &[r#""""#],
        ),
        (
            r#"{"type":"string","pattern":"^a??b$"}"#,
            &[r#""b""#, r#""ab""#],
            &[r#""aab""#],
        ),
        (
            r#"{"type":"string","pattern":"^a{2}$"}"#,
            &[r#""aa""#],
            &[r#""a""#, r#""aaa""#],
        ),
        (
            r#"{"type":"string","pattern":"^a{2,}$"}"#,
            &[r#""aa""#, r#""aaaaa""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^a{1,3}?$"}"#,
            &[r#""a""#, r#""aaa""#],
            &[r#""""#, r#""aaaa""#],
        ),
        (
            r#"{"type":"string","pattern":"^a{0}b$"}"#,
            &[r#""b""#],
            &[r#""ab""#],
        ),
        (
            r#"{"type":"string","pattern":"^a{0,0}$"}"#,
            &[r#""""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^(ab|c)+$"}"#,
            &[r#""ab""#, r#""c""#, r#""abcab""#],
            &[r#""a""#, r#""ac""#, r#""""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?:a|)b$"}"#,
            &[r#""ab""#, r#""b""#],
            &[r#""aab""#],
        ),
        (
            r#"{"type":"string","pattern":"^(|a|)$"}"#,
            &[r#""""#, r#""a""#],
            &[r#""aa""#],
        ),
        (
            r#"{"type":"string","pattern":"^()*a(?:)+$"}"#,
            &[r#""a""#],
            &[r#""""#, r#""aa""#],
        ),
        (
            r#"{"type":"string","pattern":"^(a*)*$"}"#,
            &[r#""""#, r#""aaaa""#],
            &[r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^(a*)+b$"}"#,
            &[r#""b""#, r#""aab""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^(a|ab)(c|bcd)(d*)$"}"#,
            &[r#""abcd""#, r#""ac""#, r#""abc""#, r#""acdd""#],
            &[r#""ab""#, r#""abd""#],
        ),
        (
            r#"{"type":"string","pattern":"^((a)|(b))*?c$"}"#,
            &[r#""c""#, r#""abbac""#],
            &[r#""abd""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?:a{2}){2,3}$"}"#,
            &[r#""aaaa""#, r#""aaaaaa""#],
            &[r#""aa""#, r#""aaaaa""#, r#""aaaaaaaa""#],
        ),
        (
            r#"{"type":"string","pattern":"\\bcat\\b"}"#,
            &[r#""cat""#, r#""a cat.""#, r#""cat-flap""#],
            &[r#""cats""#, r#""concat""#, r#""\u00e9cat_""#],
        ),
        (
            r#"{"type":"string","pattern":"\\Bcat"}"#,
            &[r#""concat""#, r#""_cat""#],
            &[r#""cat""#, r#"" cat""#, r#""\u00e9cat""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\b"}"#,
            &[r#""a""#],
            &[r#""""#, r#""-a""#, r#""\u00e9""#],
        ),
        (
            r#"{"type":"string","pattern":"\\b$"}"#,
            &[r#""a""#, r#""-a""#],
            &[r#""""#, r#""a-""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\B$"}"#,
            &[r#""""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?=.*\\d)(?=.*[a-z]).{4,}$"}"#,
            &[r#""ab12""#, r#""1a..""#],
            &[r#""abcd""#, r#""1234""#, r#""a1""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?!ab)\\w+$"}"#,
            &[r#""ba""#, r#""a""#, r#""aab""#],
            &[r#""ab""#, r#""abc""#],
        ),
        (
            r#"{"type":"string","pattern":"a(?=b)"}"#,
            &[r#""ab""#, r#""xab""#],
            &[r#""a""#, r#""ac""#, r#""ba""#],
        ),
        (
            r#"{"type":"string","pattern":"a(?!b)"}"#,
            &[r#""a""#, r#""ac""#, r#""aba""#],
            &[r#""ab""#, r#""xab""#],
        ),
        (
            r#"{"type":"string","pattern":"(?<=a)b"}"#,
            &[r#""ab""#, r#""xab""#],
            &[r#""b""#, r#""cb""#],
        ),
        (
            r#"{"type":"string","pattern":"(?<!a)b"}"#,
            &[r#""b""#, r#""cb""#, r#""abb""#],
            &[r#""ab""#, r#""xab""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?=a$)a$"}"#,
            &[r#""a""#],
            &[r#""aa""#],
        ),
        (
            r#"{"type":"string","pattern":"^a(?<=^a)$"}"#,
            &[r#""a""#],
            &[r#""aa""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?:(?=a)\\w|-)+$"}"#,
            &[r#""a-a""#, r#""-""#, r#""aa""#],
            &[r#""ab""#, r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?=(?!ab)a)\\w+$"}"#,
            &[r#""ac""#, r#""a""#],
            &[r#""ab""#, r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?=a(?<=^a)b)ab$"}"#,
            &[r#""ab""#],
            &[r#""abb""#],
        ),
        (
            r#"{"type":"string","pattern":"(?<=(?<!x)a)b"}"#,
            &[r#""ab""#],
            &[r#""xab""#, r#""b""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?!)"}"#,
            &[],
            &[r#""""#, r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?=)$"}"#,
            &[r#""""#],
            &[r#""a""#],
        ),
        (
            r#"{"type":"string","pattern":"(?=\\b)a"}"#,
            &[r#""a""#, r#"" a""#],
            &[r#""ba""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?=(a|b)*c)\\w+$"}"#,
            &[r#""abc""#, r#""c""#, r#""cab""#],
            &[r#""abd""#],
        ),
        (
            r#"{"type":"string","pattern":"(?<=\\d{2})-(?=\\d{2})"}"#,
            &[r#""12-34""#, r#""x99-00y""#],
            &[r#""1-23""#, r#""12-3""#, r#""ab-cd""#],
        ),
        (
            r#"{"type":"string","pattern":"^https\\:\\/\\/a\\ b\\_c\\-d\\.$"}"#,
            &[r#""https://a b_c-d.""#],
            &[r#""https://a b_c-dx""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\@\\#\\%\\&\\,\\;\\<\\=\\>\\~\\`\\!\\'\\\"$"}"#,
            &[r#""@#%&,;<=>~`!'\"""#],
            &[r#""@""#],
        ),
        (
            r#"{"type":"string","pattern":"^na\\\u00efve\\\u2028$"}"#,
            &[r#""na\u00efve\u2028""#],
            &[r#""naive""#],
        ),
        (
            r#"{"type":"string","pattern":"^[\\ -\\~]+$"}"#,
            &[r#""a b~""#],
            &[r#""\t""#, r#""\u00e9""#],
        ),
        (
            r#"{"type":"string","pattern":"^\\^\\$\\\\\\.\\*\\+\\?\\(\\)\\[\\]\\{\\}\\|$"}"#,
            &[r#""^$\\.*+?()[]{}|""#],
            &[r#""^""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"}"#,
            &[
                r#""a@b.co""#,
                r#""first.last+tag@sub.example.org""#,
                r#""o'neil@x-y.io""#,
            ],
            &[
                r#"".a@b.co""#,
                r#""a..b@c.de""#,
                r#""a.@b.co""#,
                r#""a@b""#,
                r#""a@-b.co""#,
                r#""a b@c.de""#,
                r#""a@b.c""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^(?=.{1,253}\\.?$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[-0-9a-zA-Z]{0,61}[0-9a-zA-Z])?)*\\.?$"}"#,
            &[
                r#""example.com""#,
                r#""a""#,
                r#""x-1.y.""#,
                r#""localhost""#,
            ],
            &[
                r#""""#,
                r#""-a.com""#,
                r#""a-.com""#,
                r#""a..b""#,
                r#""a_b.com""#,
                r#""aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^P(?:(\\d+W)|(?!.*W)(?=\\d|T\\d)(\\d+Y)?(\\d+M)?(\\d+D)?(T(?=\\d)(\\d+H)?(\\d+M)?(\\d+([.,]\\d+)?S)?)?)$"}"#,
            &[
                r#""P1W""#,
                r#""P1Y2M3D""#,
                r#""PT1H""#,
                r#""P1DT2H3M4.5S""#,
                r#""PT0,5S""#,
            ],
            &[
                r#""P""#,
                r#""PT""#,
                r#""P1WT1H""#,
                r#""P1S""#,
                r#""1Y""#,
                r#""P1Y2W""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)$"}"#,
            &[
                r#""123e4567-e89b-12d3-a456-426614174000""#,
                r#""00000000-0000-0000-0000-000000000000""#,
            ],
            &[
                r#""123e4567-e89b-02d3-a456-426614174000""#,
                r#""123e4567e89b12d3a456426614174000""#,
                r#""""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$"}"#,
            &[r#""0.0.0.0""#, r#""192.168.1.255""#],
            &[
                r#""256.1.1.1""#,
                r#""1.2.3""#,
                r#""01.2.3.4""#,
                r#""1.2.3.4 ""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))$"}"#,
            &[
                r#""::""#,
                r#""::1""#,
                r#""2001:db8::8a2e:370:7334""#,
                r#""1:2:3:4:5:6:7:8""#,
            ],
            &[
                r#""1:2:3:4:5:6:7""#,
                r#"":::""#,
                r#""g::1""#,
                r#""1::2::3""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^(?:(?:\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\\d|30)|(?:02)-(?:0[1-9]|1\\d|2[0-8])))T(?:(?:[01]\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d(?:\\.\\d+)?)?(?:Z|([+-](?:[01]\\d|2[0-3]):[0-5]\\d)))$"}"#,
            &[
                r#""2024-02-29T23:59:59Z""#,
                r#""2025-12-31T00:00:00.123+05:30""#,
                r#""2000-01-01T12:30Z""#,
            ],
            &[
                r#""2023-02-29T00:00:00Z""#,
                r#""2024-13-01T00:00:00Z""#,
                r#""2024-01-01 00:00:00Z""#,
                r#""2024-01-01T24:00:00Z""#,
                r#""2024-01-01T00:00:00""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^$|^(?:[0-9a-zA-Z+/]{4})*(?:(?:[0-9a-zA-Z+/]{2}==)|(?:[0-9a-zA-Z+/]{3}=))?$"}"#,
            &[r#""""#, r#""QUJD""#, r#""QUI=""#, r#""QQ==""#],
            &[r#""QUJ""#, r#""QQ=""#, r#""QU JD""#, r#""QUJD==""#],
        ),
        (
            r#"{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"}"#,
            &[r#""1""#, r#""-1.5""#, r#""+.5""#, r#""007""#, r#""1.""#],
            &[
                r#""""#,
                r#""-""#,
                r#""+.""#,
                r#"".""#,
                r#""1.2.3""#,
                r#""1e5""#,
                r#""--1""#,
            ],
        ),
        (
            r#"{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*(?:\\d{0,3}|(?=[\\d.]{1,6}0*$)\\d{0,3}\\.\\d{0,2}0*$)"}"#,
            &[r#""123""#, r#""123.45""#, r#""-0.5""#, r#""1.50""#],
            &[r#""-""#, r#"".""#, r#""+""#],
        ),
        (
            r#"{"type":"string","pattern":"^[cC][^\\s-]{8,}$"}"#,
            &[r#""cjld2cjxh0000qzrmn831i7rn""#, r#""C12345678""#],
            &[r#""c1234567""#, r#""c1234 5678""#, r#""djld2cjxh0000""#],
        ),
        (
            r#"{"type":"string","pattern":"^id--?\\d+(?:\\.\\d+)?-(a|b)$"}"#,
            &[r#""id-1-a""#, r#""id--2.5-b""#],
            &[r#""id-1-c""#, r#""id-1.-a""#],
        ),
        (
            r#"{"type":"string","pattern":"^https\\:\\/\\/a\\ b\\_c\\-d\\."}"#,
            &[r#""https://a b_c-d.x""#],
            &[r#""http://a b_c-d.""#],
        ),
    ];
    for (schema, yes, no) in matching {
        let parsed = parse(schema);
        for (values, expected) in [(*yes, ""), (*no, "tool string does not match pattern")] {
            for value in values {
                expect_text(
                    refusal_text(validate_tool_value(&parsed, &parse(value))),
                    expected,
                    &format!("{schema} {value}"),
                );
            }
        }
    }
    // Matching follows every thread at once, so nested quantifiers that would take a backtracking matcher longer than this suite runs are answered at once.
    let long = Json::String("a".repeat(20000).into_bytes());
    for (schema, expected) in [
        (
            r#"{"type":"string","pattern":"^(a*)*b$"}"#,
            "tool string does not match pattern",
        ),
        (
            r#"{"type":"string","pattern":"^(a|a)*b$"}"#,
            "tool string does not match pattern",
        ),
        (
            r#"{"type":"string","pattern":"^(a+)+b$"}"#,
            "tool string does not match pattern",
        ),
        (
            r#"{"type":"string","pattern":"(.*a){12}b"}"#,
            "tool string does not match pattern",
        ),
        (r#"{"type":"string","pattern":"^(?=(a*)*$)(a|aa)+$"}"#, ""),
        (
            r#"{"type":"string","pattern":"^(?!(a+)+$)"}"#,
            "tool string does not match pattern",
        ),
        (r#"{"type":"string","pattern":"(?<=(a*)*)a$"}"#, ""),
        (r#"{"type":"string","pattern":"^(?:a{64}){63}a{0,31}"}"#, ""),
    ] {
        expect_text(
            refusal_text(validate_tool_value(&parse(schema), &long)),
            expected,
            schema,
        );
    }
    // A value built in memory can hold bytes that are not UTF-8. It cannot be read as code points, so it is refused as the JSON codec refuses it.
    expect_text(
        refusal_text(validate_tool_value(
            &parse(r#"{"type":"string","pattern":"a"}"#),
            &Json::String(b"a\xff".to_vec()),
        )),
        "invalid UTF-8 input",
        "bytes that are not UTF-8",
    );
    // With enum validation deferred to the executor a pattern is still enforced.
    for (schema, value, expected) in [
        (
            r#"{"type":"string","pattern":"^a","enum":["ab"]}"#,
            r#""ac""#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^a","enum":["ab"]}"#,
            r#""b""#,
            "tool string does not match pattern",
        ),
        (
            r#"{"type":"string","pattern":"^a","const":"ab"}"#,
            r#""ac""#,
            "",
        ),
        (
            r#"{"type":"string","pattern":"^a","const":"ab"}"#,
            r#""b""#,
            "tool string does not match pattern",
        ),
    ] {
        expect_text(
            refusal_text(validate_tool_value_with_policy(
                &parse(schema),
                &parse(value),
                ToolEnumValidation::Executor,
            )),
            expected,
            &format!("{schema} {value}"),
        );
    }
    // A recorded call: strict argument validation refuses a string that does not match, and executor validation defers it.
    let declared = parse(
        r#"[{"type":"function","function":{"name":"tag","parameters":{"type":"object","properties":{"slug":{"type":"string","pattern":"^[a-z]+(?:-[a-z]+)*$","maxLength":12},"email":{"type":"string","pattern":"^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"},"amount":{"anyOf":[{"type":"number"},{"type":"string","pattern":"^(?!^[-+.]*$)[+-]?0*\\d*\\.?\\d*$"}]}},"additionalProperties":false}}}]"#,
    );
    for (arguments, strict) in [
        (r#"{"slug":"ab-cd","email":"a@b.co","amount":"1.5"}"#, ""),
        (r#"{"slug":"ab--cd"}"#, "tool string does not match pattern"),
        (r#"{"slug":"Ab"}"#, "tool string does not match pattern"),
        (
            r#"{"slug":"ab-cd-ef-gh-ij"}"#,
            "tool string above maxLength",
        ),
        (
            r#"{"email":"a..b@c.de"}"#,
            "tool string does not match pattern",
        ),
        (r#"{"amount":2}"#, ""),
        (
            r#"{"amount":"-."}"#,
            "tool value matches no schema alternative",
        ),
    ] {
        let history = parse(&format!(
            r#"[{{"role":"user","content":"tag it"}},{{"role":"assistant","tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"tag","arguments":{arguments}}}}}]}},{{"role":"tool","tool_call_id":"call_1","content":"tagged"}},{{"role":"user","content":"again"}}]"#
        ));
        let job = with(
            &with(&plain("A"), "tools", declared.clone()),
            "messages",
            history,
        );
        expect_text(
            refusal_text(admit_chat(&job, b"tokspur")),
            strict,
            arguments,
        );
        expect_text(
            refusal_text(admit_chat_with_policy(
                &job,
                b"tokspur",
                ToolEnumValidation::Executor,
            )),
            strict,
            arguments,
        );
        expect_text(
            refusal_text(admit_chat_with_policies(
                &job,
                b"tokspur",
                ToolEnumValidation::Strict,
                ToolArgValidation::Executor,
            )),
            "",
            arguments,
        );
    }
    // A generated call takes the same validator: the strict session refuses with its text and the executor session passes the call on.
    let job = with(&plain("A"), "tools", declared);
    let mut cancelled = || false;
    let mut seed = || 44u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut seed,
        prefix_trace: None,
    };
    for (parameter, value, expected) in [
        ("slug", "ab-cd", ""),
        ("slug", "ab--cd", "tool string does not match pattern"),
        ("slug", "ab-cd-ef-gh-ij", "tool string above maxLength"),
        ("email", "a@b.co", ""),
        ("email", ".a@b.co", "tool string does not match pattern"),
    ] {
        let frame = format!(
            "<tool_call><function=tag><parameter={parameter}>\n{value}\n</parameter></function></tool_call><|im_end|>"
        );
        let mut strict = ServeSession::new(codec(), SessionOptions::default()).unwrap();
        let rejected = RefCell::new(Fake::new());
        rejected.borrow_mut().reply(&frame);
        let mut rejected_call =
            |command: &Json, event: Event<'_>| rejected.borrow_mut().call(command, event);
        expect_text(
            refusal_text(strict.complete(&job, b"strict-pattern", &mut rejected_call, &mut sink)),
            expected,
            &frame,
        );
        let mut executor = ServeSession::new(
            codec(),
            SessionOptions {
                tool_args: ToolArgValidation::Executor,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let accepted = RefCell::new(Fake::new());
        accepted.borrow_mut().reply(&frame);
        let mut accepted_call =
            |command: &Json, event: Event<'_>| accepted.borrow_mut().call(command, event);
        let result = executor
            .complete(&job, b"executor-pattern", &mut accepted_call, &mut sink)
            .unwrap();
        check(
            message(&result)
                .find("tool_calls")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .len()
                == 1,
        );
    }
}

fn response_failure_trace_cases() {
    use tokspur_text::response::{ResponseLimits, ResponseParser, ResponseStatus};
    // Nonstreaming still cross-checks incremental text against the final decode.
    let mut inconsistent = codec();
    inconsistent.decode = Box::new(|ids| {
        let mut bytes = decode(ids);
        if ids.len() > 1 {
            bytes[0] = b'x';
        }
        Ok(bytes)
    });
    let mut session_under_test =
        ServeSession::new(inconsistent, SessionOptions::default()).unwrap();
    let mut target = Fake::new();
    target.reply("ok<|im_end|>");
    let error = session_under_test
        .complete(
            &plain("A"),
            b"final-prefix-check",
            &mut |job, event| target.call(job, event),
            &mut ServeSink {
                chunk: None,
                cancelled: &mut || false,
                new_seed: &mut || 44,
                prefix_trace: None,
            },
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "final response disagrees with streamed prefix"
    );
    assert_eq!(session_under_test.committed_tokens(), 0);

    for (raw, phase, detail, public_error) in [
        (
            "private-failure-canary\0tail<|im_end|>",
            "feed",
            "NUL byte in response",
            "response bytes/frame validation failed",
        ),
        (
            "<tool_call><function=read_file><|im_end|>",
            "finish",
            "truncated response tool call",
            "incomplete or invalid stopped response",
        ),
    ] {
        for trace_enabled in [false, true] {
            for trace_fails in [false, true] {
                let fake = RefCell::new(Fake::new());
                let mut serving = session("direct", Json::Null).unwrap();
                let events = RefCell::new(Vec::<Json>::new());
                let mut trace = |record: &Json| {
                    events.borrow_mut().push(record.clone());
                    if trace_fails
                        && record.find("event")?.unwrap().string()? == b"response_validation_failed"
                    {
                        return Err(fixture_failure("injected private trace failure"));
                    }
                    Ok(())
                };
                let mut call = |job: &Json, event: Event<'_>| fake.borrow_mut().call(job, event);
                let mut no_cancel = || false;
                let mut new_seed = || 44;
                let mut sink = ServeSink {
                    chunk: None,
                    cancelled: &mut no_cancel,
                    new_seed: &mut new_seed,
                    prefix_trace: if trace_enabled {
                        Some(&mut trace)
                    } else {
                        None
                    },
                };
                fake.borrow_mut().reply("warm<|im_end|>");
                serving
                    .complete(&plain("A"), b"warm", &mut call, &mut sink)
                    .unwrap();
                check(serving.committed_tokens() > 0);
                events.borrow_mut().clear();
                fake.borrow_mut().reply(raw);
                let job = with(&plain("B"), "tools", tools());
                let error = serving
                    .complete(&job, b"failed", &mut call, &mut sink)
                    .unwrap_err();
                check(error.to_string() == public_error);
                check(!error.to_string().contains("private-failure-canary"));
                check(serving.committed_tokens() == 0);
                let failures: Vec<Json> = events
                    .borrow()
                    .iter()
                    .filter(|event| {
                        event.find("event").unwrap().unwrap().string().unwrap()
                            == b"response_validation_failed"
                    })
                    .cloned()
                    .collect();
                check(failures.len() == usize::from(trace_enabled));
                check(!events.borrow().iter().any(|event| {
                    event.find("event").unwrap().unwrap().string().unwrap() == b"completion"
                }));
                if trace_enabled {
                    let failure = &failures[0];
                    check(
                        failure.find("phase").unwrap().unwrap().string().unwrap()
                            == phase.as_bytes(),
                    );
                    check(
                        failure.find("reason").unwrap().unwrap().string().unwrap()
                            == detail.as_bytes(),
                    );
                    check(
                        failure
                            .find("response_id")
                            .unwrap()
                            .unwrap()
                            .string()
                            .unwrap()
                            == b"failed",
                    );
                    let ids: Vec<u32> = failure
                        .find("generated_ids")
                        .unwrap()
                        .unwrap()
                        .array()
                        .unwrap()
                        .iter()
                        .map(|id| id.unsigned_integer().unwrap() as u32)
                        .collect();
                    check(ids == fake.borrow().output[..fake.borrow().next]);
                    check(
                        failure
                            .find("generated_count")
                            .unwrap()
                            .unwrap()
                            .unsigned_integer()
                            .unwrap()
                            == ids.len() as u64,
                    );
                    let admitted = admit_chat(&job, b"tokspur").unwrap();
                    let mut replay =
                        ResponseParser::new(admitted.manifest, false, ResponseLimits::default())
                            .unwrap();
                    for id in ids {
                        replay.feed(&decode(&[id]));
                    }
                    if phase == "finish" {
                        replay.finish();
                    }
                    check(replay.status() == ResponseStatus::Error && replay.error() == detail);
                }
                let resets = fake.borrow().resets;
                events.borrow_mut().clear();
                fake.borrow_mut().reply("ok<|im_end|>");
                serving
                    .complete(&plain("C"), b"after-failure", &mut call, &mut sink)
                    .unwrap();
                check(fake.borrow().resets == resets + 1 && serving.committed_tokens() > 0);
                if trace_enabled {
                    let borrow = events.borrow();
                    let prompt = borrow
                        .iter()
                        .find(|event| {
                            event.find("event").unwrap().unwrap().string().unwrap()
                                == b"prompt_prefix"
                        })
                        .unwrap();
                    check(
                        prompt
                            .find("prior_committed_tokens")
                            .unwrap()
                            .unwrap()
                            .unsigned_integer()
                            .unwrap()
                            == 0,
                    );
                }
            }
        }
    }
}

fn executor_arg_contract() {
    let deferred = ToolArgValidation::Executor;
    check(
        parse_tool_arg_validation("strict") == Ok(ToolArgValidation::Strict)
            && parse_tool_arg_validation("executor") == Ok(deferred),
    );
    refusal_admission(parse_tool_arg_validation("permissive"));
    check(tool_arg_validation_policy(ToolArgValidation::Strict) == "strict-server-args-v1");
    check(tool_arg_validation_policy(deferred) == "executor-deferred-args-v1");

    let declarations = parse(
        r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"],"additionalProperties":false}}}]"#,
    );
    let job = with(&plain("A"), "tools", declarations);
    let extra_frame = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter><parameter=line>\n10\n</parameter><parameter=limit>\nten\n</parameter></function></tool_call><|im_end|>";
    let mut cancelled = || false;
    let mut seed = || 44u64;
    let mut sink = ServeSink {
        chunk: None,
        cancelled: &mut cancelled,
        new_seed: &mut seed,
        prefix_trace: None,
    };

    let mut strict = ServeSession::new(codec(), SessionOptions::default()).unwrap();
    let rejected = RefCell::new(Fake::new());
    rejected.borrow_mut().reply(extra_frame);
    let mut rejected_call =
        |command: &Json, event: Event<'_>| rejected.borrow_mut().call(command, event);
    refusal(strict.complete(&job, b"strict-arg", &mut rejected_call, &mut sink));

    let mut executor = ServeSession::new(
        codec(),
        SessionOptions {
            tool_args: deferred,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    let accepted = RefCell::new(Fake::new());
    accepted.borrow_mut().reply(extra_frame);
    let mut accepted_call =
        |command: &Json, event: Event<'_>| accepted.borrow_mut().call(command, event);
    let result = executor
        .complete(&job, b"executor-arg", &mut accepted_call, &mut sink)
        .unwrap();
    let proposed = message(&result)
        .find("tool_calls")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()[0]
        .clone();
    let args_json = parse_json(
        proposed
            .find("function")
            .unwrap()
            .unwrap()
            .find("arguments")
            .unwrap()
            .unwrap()
            .string()
            .unwrap(),
    )
    .unwrap();
    check(args_json.find("path").unwrap().unwrap().string().unwrap() == b"a.rs");
    check(
        args_json
            .find("line")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
            == 10,
    );
    check(args_json.find("limit").unwrap().unwrap().string().unwrap() == b"ten");

    accepted
        .borrow_mut()
        .reply("<tool_call><function=read_file></function></tool_call><|im_end|>");
    let missing_res = executor
        .complete(&job, b"executor-missing", &mut accepted_call, &mut sink)
        .unwrap();
    let missing_args = parse_json(
        message(&missing_res)
            .find("tool_calls")
            .unwrap()
            .unwrap()
            .array()
            .unwrap()[0]
            .find("function")
            .unwrap()
            .unwrap()
            .find("arguments")
            .unwrap()
            .unwrap()
            .string()
            .unwrap(),
    )
    .unwrap();
    check(missing_args.object().unwrap().is_empty());

    let prior = vec![
        message(&result).clone(),
        Json::new_object([
            ("role", Json::from("tool")),
            (
                "tool_call_id",
                proposed.find("id").unwrap().unwrap().clone(),
            ),
            (
                "content",
                Json::from(r#"{"error":"unexpected argument line"}"#),
            ),
        ]),
        Json::new_object([
            ("role", Json::from("user")),
            ("content", Json::from("retry without line")),
        ]),
    ];
    let next = with(&job, "messages", Json::Array(prior.clone()));
    check(
        admit_chat_with_policies(&next, b"tokspur", ToolEnumValidation::Strict, deferred)
            .unwrap()
            .manifest
            .len()
            == 1,
    );
    refusal_admission(admit_chat(&next, b"tokspur"));

    accepted.borrow_mut().reply("done<|im_end|>");
    let after = executor
        .complete(&next, b"after-executor-arg", &mut accepted_call, &mut sink)
        .unwrap();
    check(finish(&after).string().unwrap() == b"stop");

    for (raw, allowed) in [
        (
            "<tool_call><function=unknown><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>",
            false,
        ),
        (
            "<tool_call><function=read_file><parameter=path>\n<|im_start|>\n</parameter></function></tool_call><|im_end|>",
            false,
        ),
        (
            "<tool_call><function=read_file><parameter=extra>\n{\"nested\":\"\\u003ctool_call\\u003e\"}\n</parameter></function></tool_call><|im_end|>",
            true,
        ),
        (
            "<tool_call><function=read_file><parameter=extra>\n{\"nested\":\"\\u003c|endoftext|\\u003e\"}\n</parameter></function></tool_call><|im_end|>",
            true,
        ),
    ] {
        accepted.borrow_mut().reply(raw);
        let result = executor.complete(&job, b"escaped-value", &mut accepted_call, &mut sink);
        if allowed {
            check(
                message(&result.unwrap())
                    .find("tool_calls")
                    .unwrap()
                    .unwrap()
                    .array()
                    .unwrap()
                    .len()
                    == 1,
            );
        } else {
            refusal(result);
        }
    }

    refusal_admission(validate_executor_tool_arguments(&parse(r#""str""#)));
    for (value, allowed) in [
        (r#""not-an-object""#, false),
        (r#"[1,2]"#, false),
        (r#"123"#, false),
        (r#"{"path":"<|im_start|>"}"#, true),
        (r#"{"path":"<|endoftext|>"}"#, true),
    ] {
        let function = with(
            proposed.find("function").unwrap().unwrap(),
            "arguments",
            Json::from(value),
        );
        let mut bad = prior.clone();
        bad[0] = with(
            &bad[0],
            "tool_calls",
            Json::Array(vec![with(&proposed, "function", function)]),
        );
        check(
            admit_chat_with_policies(
                &with(&next, "messages", Json::Array(bad)),
                b"tokspur",
                ToolEnumValidation::Strict,
                deferred,
            )
            .is_ok()
                == allowed,
        );
    }
}

fn thinking_budget_contract() {
    let base = parse(r#"{"model":"tokspur","messages":[{"role":"user","content":"Q"}]}"#);
    check(
        admit_chat(&base, b"tokspur")
            .unwrap()
            .thinking_budget
            .is_none(),
    );

    // Admission: top-level, chat_template_kwargs, matching both, and alongside reasoning_effort.
    let top = admit_chat(
        &with(&base, "thinking_budget", Json::integer(128)),
        b"tokspur",
    )
    .unwrap();
    check(top.thinking && top.thinking_budget == Some(128));
    check(top.render.find("thinking_budget").unwrap().is_none());

    let kw = admit_chat(
        &with(
            &base,
            "chat_template_kwargs",
            parse(r#"{"thinking_budget":256,"reasoning_effort":"medium"}"#),
        ),
        b"tokspur",
    )
    .unwrap();
    check(kw.thinking && kw.thinking_budget == Some(256));
    check(
        kw.render
            .find("reasoning_effort")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == b"medium",
    );

    let both = admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(512)),
            "chat_template_kwargs",
            parse(r#"{"thinking_budget":512}"#),
        ),
        b"tokspur",
    )
    .unwrap();
    check(both.thinking && both.thinking_budget == Some(512));

    let max_b = admit_chat(
        &with(&base, "thinking_budget", Json::integer(32768)),
        b"tokspur",
    )
    .unwrap();
    check(max_b.thinking && max_b.thinking_budget == Some(32768));

    // thinking_budget = 0 disables thinking and allows reasoning_effort = "none".
    let zero = admit_chat(
        &with(&base, "thinking_budget", Json::integer(0)),
        b"tokspur",
    )
    .unwrap();
    check(!zero.thinking && zero.thinking_budget.is_none());
    check(
        !zero
            .render
            .find("enable_thinking")
            .unwrap()
            .unwrap()
            .boolean()
            .unwrap(),
    );

    let zero_none = admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(0)),
            "reasoning_effort",
            Json::from("none"),
        ),
        b"tokspur",
    )
    .unwrap();
    check(!zero_none.thinking && zero_none.thinking_budget.is_none());

    let zero_false = admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(0)),
            "enable_thinking",
            Json::Bool(false),
        ),
        b"tokspur",
    )
    .unwrap();
    check(!zero_false.thinking && zero_false.thinking_budget.is_none());

    // Refusals: bounds, types, and conflicts.
    for invalid in [
        Json::Null,
        Json::Bool(true),
        Json::from("128"),
        parse("-1"),
        parse("1.5"),
        Json::integer(32769),
    ] {
        refusal_admission(admit_chat(
            &with(&base, "thinking_budget", invalid),
            b"tokspur",
        ));
    }
    refusal_admission(admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(0)),
            "enable_thinking",
            Json::Bool(true),
        ),
        b"tokspur",
    ));
    refusal_admission(admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(64)),
            "chat_template_kwargs",
            parse(r#"{"thinking_budget":128}"#),
        ),
        b"tokspur",
    ));
    // A positive budget with thinking explicitly off is admitted and ignored, wherever either option is given; it still has to be a valid budget.
    for request in [
        with(
            &with(&base, "thinking_budget", Json::integer(64)),
            "enable_thinking",
            Json::Bool(false),
        ),
        with(
            &base,
            "chat_template_kwargs",
            parse(r#"{"thinking_budget":64,"enable_thinking":false}"#),
        ),
        with(
            &with(&base, "thinking_budget", Json::integer(64)),
            "chat_template_kwargs",
            parse(r#"{"enable_thinking":false}"#),
        ),
        with(
            &with(
                &with(&base, "thinking_budget", Json::integer(64)),
                "enable_thinking",
                Json::Bool(false),
            ),
            "reasoning_effort",
            Json::from("none"),
        ),
    ] {
        let off = admit_chat(&request, b"tokspur").unwrap();
        check(!off.thinking && off.thinking_budget.is_none());
        check(
            !off.render
                .find("enable_thinking")
                .unwrap()
                .unwrap()
                .boolean()
                .unwrap(),
        );
    }
    refusal_admission(admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(32769)),
            "enable_thinking",
            Json::Bool(false),
        ),
        b"tokspur",
    ));
    refusal_admission(admit_chat(
        &with(
            &with(&base, "thinking_budget", Json::integer(64)),
            "reasoning_effort",
            Json::from("none"),
        ),
        b"tokspur",
    ));

    // Generation: forced cutoff without trailing newline + streaming verification.
    let chunks = RefCell::new(Vec::new());
    let mut chunk_cb = |chunk: &Json| {
        chunks.borrow_mut().push(chunk.clone());
        Ok(())
    };
    let mut cancelled = || false;
    let mut seed = || 42u64;
    let mut session = ServeSession::new(codec(), SessionOptions::default()).unwrap();
    let fake = RefCell::new(Fake::new());
    let mut call = |command: &Json, event: Event<'_>| fake.borrow_mut().call(command, event);

    {
        fake.borrow_mut().reply("abcdefVisible<|im_end|>");
        let req = with(
            &request("T", true, true),
            "thinking_budget",
            Json::integer(6),
        );
        let mut sink = ServeSink {
            chunk: Some(&mut chunk_cb),
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-cutoff", &mut call, &mut sink)
            .unwrap();
        check(finish(&res).string().unwrap() == b"stop");
        check(
            message(&res)
                .find("reasoning_content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"abcdef\n",
        );
        check(
            message(&res)
                .find("content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"\n\nVisible",
        );
        let mut s_think = Vec::new();
        let mut s_text = Vec::new();
        for ev in chunks.borrow().iter() {
            let delta = ev.find("choices").unwrap().unwrap().array().unwrap()[0]
                .find("delta")
                .unwrap()
                .unwrap();
            if let Some(t) = delta.find("reasoning_content").unwrap() {
                s_think.extend_from_slice(t.string().unwrap());
            }
            if let Some(c) = delta.find("content").unwrap() {
                s_text.extend_from_slice(c.string().unwrap());
            }
        }
        check(s_think == b"abcdef\n" && s_text == b"\n\nVisible");
    }

    // Forced cutoff when thinking already ends with '\n' (avoids double newline before </think>).
    {
        chunks.borrow_mut().clear();
        fake.borrow_mut().reply("abc\nDone<|im_end|>");
        let req = with(
            &request("T2", false, true),
            "thinking_budget",
            Json::integer(4),
        );
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-nl", &mut call, &mut sink)
            .unwrap();
        check(
            message(&res)
                .find("reasoning_content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"abc\n"
                && message(&res)
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == b"\n\nDone",
        );
    }

    // Forced cutoff inside an unclosed Markdown fence (```python\nx=1) + tool call in Phase 2.
    {
        let fence_think = "```py\nx=1";
        let post_close = "<tool_call><function=read_file><parameter=path>\na.rs\n</parameter></function></tool_call><|im_end|>";
        fake.borrow_mut()
            .reply(&format!("{fence_think}{post_close}"));
        let req = with(
            &with(&request("F", true, true), "tools", tools()),
            "thinking_budget",
            Json::integer(fence_think.len() as u64),
        );
        chunks.borrow_mut().clear();
        let mut sink = ServeSink {
            chunk: Some(&mut chunk_cb),
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-fence-tool", &mut call, &mut sink)
            .unwrap();
        check(finish(&res).string().unwrap() == b"tool_calls");
        check(
            message(&res)
                .find("reasoning_content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"```py\nx=1\n",
        );
        check(
            message(&res)
                .find("tool_calls")
                .unwrap()
                .unwrap()
                .array()
                .unwrap()
                .len()
                == 1,
        );
    }

    // Forced cutoff mid multi-byte UTF-8 character ("考" = 3 bytes; budget=1 steps 2 more bytes to complete the scalar before closing).
    {
        fake.borrow_mut().reply("考Yes<|im_end|>");
        let req = with(
            &request("U8", false, true),
            "thinking_budget",
            Json::integer(1),
        );
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-utf8", &mut call, &mut sink)
            .unwrap();
        check(
            message(&res)
                .find("reasoning_content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == "考\n".as_bytes()
                && message(&res)
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == b"\n\nYes",
        );
    }

    // Natural </think> before thinking_budget is reached.
    {
        fake.borrow_mut().reply("hi</think>\n\nNatural<|im_end|>");
        let req = with(
            &request("N", false, true),
            "thinking_budget",
            Json::integer(50),
        );
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-natural", &mut call, &mut sink)
            .unwrap();
        check(
            message(&res)
                .find("reasoning_content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"hi"
                && message(&res)
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == b"\n\nNatural",
        );
    }

    // The budget is reached after the model closed thinking on its own: on the close's last token (10) or inside the answer (13). Nothing is injected and the answer continues in phase 2.
    for budget in [10u64, 13] {
        fake.borrow_mut().reply("hi</think>\n\nNatural<|im_end|>");
        let req = with(
            &request("NB", false, true),
            "thinking_budget",
            Json::integer(budget),
        );
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-natural-boundary", &mut call, &mut sink)
            .unwrap();
        check(finish(&res).string().unwrap() == b"stop");
        check(
            message(&res)
                .find("reasoning_content")
                .unwrap()
                .unwrap()
                .string()
                .unwrap()
                == b"hi"
                && message(&res)
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    == b"\n\nNatural",
        );
        check(
            res.find("usage")
                .unwrap()
                .unwrap()
                .find("completion_tokens")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap()
                == 20,
        );
    }

    // Tight max_tokens truncation during forced close injection.
    {
        fake.borrow_mut().reply("abcdefghij");
        let req = with(
            &with(
                &request("L", false, true),
                "thinking_budget",
                Json::integer(3),
            ),
            "max_tokens",
            Json::integer(5),
        );
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: None,
        };
        let res = session
            .complete(&req, b"tb-tight", &mut call, &mut sink)
            .unwrap();
        check(finish(&res).string().unwrap() == b"length");
        check(
            res.find("usage")
                .unwrap()
                .unwrap()
                .find("completion_tokens")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap()
                == 5,
        );
    }
}

//! Generated worker metadata through the actual serving session and MTP reply admission.
//! These tests cover usage and failure lifecycle; they do not execute native cache storage or a model.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::Cell;
use std::collections::VecDeque;

use tokspur_json::{Json, parse_json};
use tokspur_server::session::{
    ServeCodec, ServeError, ServeSession, ServeSink, SessionOptions, TokenCallback,
};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response_protocol::ResponseAdmission;
use tokspur_wire::mtp_wire::MtpWireReply;

fn get<'a>(value: &'a Json, name: &str) -> &'a Json {
    value.find(name).unwrap().unwrap()
}

fn number(value: &Json, name: &str) -> u64 {
    get(value, name).unsigned_integer().unwrap()
}

fn put(value: &mut Json, name: &str, replacement: Json) {
    let mut fields = value.object().unwrap().clone();
    fields.retain(|(key, _)| key != name.as_bytes());
    fields.push((name.as_bytes().to_vec(), replacement));
    *value = Json::Object(fields);
}

#[derive(Clone)]
enum Counts {
    Restored(u64),
    Legacy,
    Fields(Option<Json>, Option<Json>),
}

impl Counts {
    fn write(&self, done: &mut Json, inputs: u64) {
        let (restored, replayed) = match self {
            Self::Restored(restored) => (
                Some(Json::integer(*restored)),
                Some(Json::integer(inputs.saturating_sub(*restored))),
            ),
            Self::Legacy => (None, None),
            Self::Fields(restored, replayed) => (restored.clone(), replayed.clone()),
        };
        if let Some(restored) = restored {
            put(done, "checkpoint_restored_tokens", restored);
        }
        if let Some(replayed) = replayed {
            put(done, "replayed_input_tokens", replayed);
        }
    }
}

fn ready() -> Json {
    parse_json(br#"{"event":"ready","maximum_generated_tokens":4096,"stack_execution":"direct","small_gemv_column_splits":16,"block_gemv_formats":3,"native_mtp_protocol":"native-mtp-depth2-lookahead-v1","native_mtp_depth":2,"native_mtp_draft_attention":"tiled128","native_mtp_input_packet_schema":1,"native_mtp_input_packet_max":512,"native_prefill_dense":"f32","native_prefill_grouped":"f32","native_mtp_priming_batch_capacity":512,"native_mtp_priming_batch_bytes":1024,"native_mtp_priming_batch_submits":0,"native_mtp_carry_scope":"http-generation-v1","native_mtp_generation":{"id":0,"active":false,"remaining":0,"pending":false},"native_mtp_head_calls":0,"native_mtp_prime_calls":0,"native_mtp_rounds":0,"native_mtp_proposed":0,"native_mtp_accepted":0,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0}"#).unwrap()
}

struct Worker {
    ready: Json,
    position: u64,
    primes: u64,
    seed: u64,
    sampling: Json,
    generation_id: u64,
    inputs: VecDeque<Counts>,
    output: Counts,
    input_packets: usize,
    resets: usize,
    cancel_after_input: Option<usize>,
    corrupt_prime_count: bool,
    /// Parked conversation length the next lane prompt restores, and the lane prompt lengths seen.
    lane: Option<u64>,
    inherited_lane: Option<(u64, u64)>,
    seed_opt_ins: usize,
    lane_prompts: Vec<usize>,
}

impl Worker {
    fn new(inputs: impl IntoIterator<Item = Counts>) -> Self {
        Self {
            ready: ready(),
            position: 0,
            primes: 0,
            seed: 0,
            sampling: Json::Null,
            generation_id: 0,
            inputs: inputs.into_iter().collect(),
            output: Counts::Restored(0),
            input_packets: 0,
            resets: 0,
            cancel_after_input: None,
            corrupt_prime_count: false,
            lane: None,
            inherited_lane: None,
            seed_opt_ins: 0,
            lane_prompts: Vec::new(),
        }
    }

    fn call(
        &mut self,
        job: &Json,
        event: TokenCallback<'_>,
        cancelled: &Cell<bool>,
    ) -> Result<Json, ServeError> {
        let reset = get(job, "reset").boolean().unwrap();
        if job.find("lane_inherit_seed").unwrap().is_some() {
            assert!(reset && get(job, "lane_inherit_seed").boolean().unwrap());
            self.seed_opt_ins += 1;
        }
        let inherited = if reset {
            self.inherited_lane.take()
        } else {
            None
        };
        if reset {
            self.resets += 1;
            self.position = 0;
            self.sampling = get(job, "sampling").clone();
            self.seed = inherited.map_or_else(|| number(job, "seed"), |(seed, _)| seed);
        }
        let inputs = get(job, "tokens").array().unwrap().len() as u64;
        let generate = number(job, "generate");
        let mut ids = Vec::new();
        let mut lane_restored = None;
        if let Some(lane) = job.find("lane_prompt").unwrap() {
            self.lane_prompts.push(lane.array().unwrap().len());
            lane_restored = self.lane.take().filter(|&length| length > inputs);
        }
        let counts = if inputs > 0 {
            assert_eq!(generate, 0);
            self.input_packets += 1;
            self.position += inputs;
            // Native restore credits logical prime calls even when it skips their execution.
            self.primes += inputs - u64::from(reset);
            if let Some(length) = lane_restored {
                self.position = length;
            }
            self.inputs.pop_front().unwrap_or(Counts::Restored(0))
        } else {
            assert_eq!(generate, 1);
            self.generation_id = number(get(job, "mtp_generation"), "id");
            self.position += 1;
            self.primes += 1;
            ids.push(Json::integer(u64::from(b'a')));
            event(&Json::new_object([
                ("event", Json::from("token")),
                ("id", Json::integer(u64::from(b'a'))),
                ("committed_prefix", Json::integer(self.position)),
                ("entropy_words", Json::integer(0)),
            ]))?;
            self.output.clone()
        };
        let mut sampled = self.sampling.clone();
        for (name, value) in [
            ("valid", Json::Bool(true)),
            ("entropy_prefix", Json::integer(self.position)),
            ("entropy_words", Json::integer(0)),
            ("seed", Json::integer(self.seed)),
        ] {
            put(&mut sampled, name, value);
        }
        let mut done = self.ready.clone();
        for (name, value) in [
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(self.position)),
            ("generated_ids", Json::Array(ids)),
            ("sampling", sampled),
            (
                "status",
                Json::from(if inputs > 0 { "prefilled" } else { "length" }),
            ),
            ("continuation_allowed", Json::Bool(true)),
            (
                "native_mtp_prime_calls",
                Json::integer(self.primes + u64::from(self.corrupt_prime_count)),
            ),
            (
                "native_mtp_generation",
                Json::new_object([
                    ("id", Json::integer(self.generation_id)),
                    ("active", Json::Bool(false)),
                    ("remaining", Json::integer(0)),
                    ("pending", Json::Bool(false)),
                ]),
            ),
            // Deliberately unrelated lifetime/absolute-prefix values must never become HTTP usage.
            ("shared_prefix_cache_restored_tokens", Json::integer(98_000)),
            ("checkpoint_ssd_restored_tokens", Json::integer(104_000)),
        ] {
            put(&mut done, name, value);
        }
        counts.write(&mut done, inputs);
        if let Some((seed, tokens)) = inherited {
            put(&mut done, "lane_seed", Json::integer(seed));
            put(&mut done, "lane_restored_tokens", Json::integer(tokens));
        }
        if let Some(length) = lane_restored {
            put(&mut done, "lane_restored_prefix", Json::integer(length));
            put(&mut done, "continuation_allowed", Json::Bool(false));
        }
        if inputs > 0 && self.cancel_after_input == Some(self.input_packets) {
            cancelled.set(true);
        }
        Ok(done)
    }
}

fn session() -> ServeSession {
    session_with(false)
}

fn session_with(lane_parking: bool) -> ServeSession {
    let codec = ServeCodec {
        control_ids: Vec::new(),
        nonordinary_ids: Vec::new(),
        render: Box::new(|request| {
            let messages = get(request, "messages").array().unwrap();
            Ok(RenderedChat {
                prompt: get(messages.last().unwrap(), "content")
                    .string()
                    .unwrap()
                    .to_vec(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| Ok(text.prompt.iter().map(|&byte| u32::from(byte)).collect())),
        decode: Box::new(|ids| Ok(ids.iter().map(|&id| u8::try_from(id).unwrap()).collect())),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    };
    ServeSession::new(
        codec,
        SessionOptions {
            mtp_ready: ready(),
            lane_parking,
            ..SessionOptions::default()
        },
    )
    .unwrap()
}

fn complete(
    session: &mut ServeSession,
    worker: &mut Worker,
    prompt: &str,
    stream: bool,
    cancelled: &Cell<bool>,
) -> (Result<Json, ServeError>, Vec<Json>, Vec<Json>) {
    complete_with_seed(session, worker, prompt, stream, cancelled, Some(44))
}

fn complete_with_seed(
    session: &mut ServeSession,
    worker: &mut Worker,
    prompt: &str,
    stream: bool,
    cancelled: &Cell<bool>,
    requested_seed: Option<u64>,
) -> (Result<Json, ServeError>, Vec<Json>, Vec<Json>) {
    let request = Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from(prompt)),
            ])]),
        ),
        ("max_tokens", Json::integer(1)),
        ("stream", Json::Bool(stream)),
        ("enable_thinking", Json::Bool(false)),
    ]);
    let mut request = request;
    if let Some(seed) = requested_seed {
        put(&mut request, "seed", Json::integer(seed));
    }
    if stream {
        put(
            &mut request,
            "stream_options",
            Json::new_object([("include_usage", Json::Bool(true))]),
        );
    }
    let mut traces = Vec::new();
    let mut chunks = Vec::new();
    let mut trace = |value: &Json| {
        traces.push(value.clone());
        Ok(())
    };
    let mut chunk = |value: &Json| {
        chunks.push(value.clone());
        Ok(())
    };
    let mut check_cancelled = || cancelled.get();
    let mut seed = || 44;
    let mut sink = ServeSink {
        chunk: Some(&mut chunk),
        cancelled: &mut check_cancelled,
        new_seed: &mut seed,
        prefix_trace: Some(&mut trace),
    };
    let result = session.complete(
        &request,
        b"cache-usage",
        &mut |job, event| worker.call(job, event, cancelled),
        &mut sink,
    );
    (result, traces, chunks)
}

fn cached(response: &Json) -> u64 {
    number(
        get(get(response, "usage"), "prompt_tokens_details"),
        "cached_tokens",
    )
}

fn assert_usage(response: &Json, traces: &[Json], chunks: &[Json], expected: u64) {
    assert_eq!(cached(response), expected);
    let completed = traces
        .iter()
        .find(|item| get(item, "event").string().unwrap() == b"completion")
        .unwrap();
    assert_eq!(number(completed, "cached_tokens"), expected);
    assert_eq!(
        number(completed, "new_tokens"),
        number(get(response, "usage"), "prompt_tokens") - expected
    );
    if !chunks.is_empty() {
        let final_chunk = chunks.last().unwrap();
        assert!(get(final_chunk, "choices").array().unwrap().is_empty());
        assert_eq!(get(final_chunk, "usage"), get(response, "usage"));
    }
}

#[test]
fn accumulates_packets_once_and_preserves_active_prefix_then_resets_request_usage() {
    let mut session = session();
    let mut worker = Worker::new([
        Counts::Restored(512),
        Counts::Restored(128),
        Counts::Restored(0),
    ]);
    let cancelled = Cell::new(false);
    let prompt = "x".repeat(1100);
    let (response, traces, chunks) = complete(&mut session, &mut worker, &prompt, true, &cancelled);
    assert!(
        !chunks.is_empty(),
        "streamed request must publish final usage"
    );
    assert_usage(&response.unwrap(), &traces, &chunks, 640);
    assert_eq!(worker.input_packets, 3);
    assert_eq!(number(&traces[0], "cached_tokens"), 0); // Admission trace precedes native restores.
    assert_eq!(session.committed_tokens(), 1101);
    assert_eq!(worker.resets, 1);

    worker
        .inputs
        .extend([Counts::Restored(256), Counts::Restored(0)]);
    let appended = format!("{prompt}a{}", "y".repeat(600));
    let (response, traces, chunks) =
        complete(&mut session, &mut worker, &appended, false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 1357); // 1101 active plus 256 newly restored.
    assert_eq!(worker.resets, 1);
    assert_eq!(number(&traces[0], "cached_tokens"), 1101);

    let (response, traces, chunks) =
        complete(&mut session, &mut worker, "fresh", false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 0);
    assert_eq!(worker.resets, 2);
}

#[test]
fn parked_lane_restore_past_the_first_packet_counts_as_cached_and_resumes_as_append() {
    let (mut plain, mut parked) = (session(), session_with(true));
    let cancelled = Cell::new(false);
    let prompt = "x".repeat(1100);

    // Off: no lane prompt is ever sent.
    let mut worker = Worker::new([]);
    let (response, _, _) = complete(&mut plain, &mut worker, &prompt, false, &cancelled);
    response.unwrap();
    assert!(worker.lane_prompts.is_empty());

    // On: the fresh first packet names the whole prompt; a 700-token parked lane restores past it.
    let mut worker = Worker::new([Counts::Restored(512)]);
    worker.lane = Some(700);
    let (response, traces, chunks) = complete(&mut parked, &mut worker, &prompt, false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 700);
    assert_eq!(worker.lane_prompts, [1100]);
    assert_eq!(worker.input_packets, 2); // 0..512 restored to 700, then 700..1100.
    assert_eq!(worker.resets, 1);
    assert_eq!(parked.committed_tokens(), 1101);

    // The restored conversation continues as an ordinary append with no lane prompt.
    let appended = format!("{prompt}a{}", "y".repeat(40));
    let (response, traces, chunks) =
        complete(&mut parked, &mut worker, &appended, false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 1101);
    assert_eq!(worker.lane_prompts, [1100]);
    assert_eq!(worker.resets, 1);

    // Single-packet fresh prompts (n == prompt.len() <= 512) omit lane_prompt while still
    // accounting for parked-lane tokens restored inside the single packet.
    let short_prompt = "z".repeat(400);
    worker.inputs.push_back(Counts::Restored(320));
    let (response, traces, chunks) =
        complete(&mut parked, &mut worker, &short_prompt, false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 320);
    assert_eq!(worker.lane_prompts, [1100]);
    assert_eq!(worker.resets, 2);
}

#[test]
fn parked_lanes_preserve_saved_thinking_across_interleaved_conversations() {
    let codec = ServeCodec {
        control_ids: Vec::new(),
        nonordinary_ids: Vec::new(),
        render: Box::new(|request| {
            let messages = get(request, "messages").array().unwrap();
            let mut prompt = Vec::new();
            for msg in messages {
                let role = get(msg, "role").string().unwrap();
                let content = get(msg, "content").string().unwrap();
                if role == b"user" {
                    prompt.extend_from_slice(b"U:");
                    prompt.extend_from_slice(content);
                    prompt.push(b'\n');
                } else {
                    assert_eq!(role, b"assistant");
                    prompt.extend_from_slice(b"<think>");
                    if let Some(r) = msg.find("reasoning_content").unwrap()
                        && !r.is_null()
                    {
                        prompt.extend_from_slice(r.string().unwrap());
                    }
                    prompt.extend_from_slice(b"</think>");
                    prompt.extend_from_slice(content);
                    prompt.extend_from_slice(b"<|im_end|>");
                }
            }
            prompt.extend_from_slice(b"<think>");
            Ok(RenderedChat {
                prompt,
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| {
            let mut out = Vec::new();
            let s = &text.prompt;
            let mut i = 0;
            while i < s.len() {
                if s[i..].starts_with(b"</think>") {
                    out.push(2000);
                    i += b"</think>".len();
                } else if s[i..].starts_with(b"<|im_end|>") {
                    out.push(1000);
                    i += b"<|im_end|>".len();
                } else {
                    out.push(u32::from(s[i]));
                    i += 1;
                }
            }
            Ok(out)
        }),
        decode: Box::new(|ids| {
            let mut out = Vec::new();
            for &id in ids {
                match id {
                    1000 => out.extend_from_slice(b"<|im_end|>"),
                    1001 => out.extend_from_slice(b"<|endoftext|>"),
                    2000 => out.extend_from_slice(b"</think>"),
                    _ => out.push(u8::try_from(id).unwrap()),
                }
            }
            Ok(out)
        }),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    };
    let mut session = ServeSession::new(
        codec,
        SessionOptions {
            mtp_ready: ready(),
            lane_parking: true,
            ..SessionOptions::default()
        },
    )
    .unwrap();

    let mut lifetime_primes = 0u64;
    let mut gen_id = 0u64;
    let mut run_turn = |session: &mut ServeSession,
                        messages: Vec<Json>,
                        think_out: &[u8],
                        ans_out: &[u8],
                        restored_prefix: u64|
     -> (Json, Vec<u32>) {
        let request = Json::new_object([
            ("model", Json::from("tokspur")),
            ("messages", Json::Array(messages)),
            ("max_tokens", Json::integer(32)),
            ("seed", Json::integer(44)),
            ("stream", Json::Bool(false)),
            ("enable_thinking", Json::Bool(true)),
        ]);
        let mut out_tokens: VecDeque<u32> = think_out.iter().map(|&b| u32::from(b)).collect();
        out_tokens.push_back(2000); // </think>
        out_tokens.extend(ans_out.iter().map(|&b| u32::from(b)));
        out_tokens.push_back(1000); // <|im_end|>

        let mut position = 0u64;
        let mut sampling = Json::Null;
        let mut submitted_prompt: Vec<u32> = Vec::new();
        let mut target = |job: &Json, event: TokenCallback<'_>| -> Result<Json, ServeError> {
            let reset = get(job, "reset").boolean().unwrap();
            if reset {
                position = 0;
                sampling = get(job, "sampling").clone();
            }
            let toks = get(job, "tokens").array().unwrap();
            let inputs = toks.len() as u64;
            let generate = number(job, "generate") as usize;
            let mut ids = Vec::new();
            let mut status = "prefilled";
            let mut stop_id = Json::Null;
            let mut cont_allowed = true;
            let mut restored_field = None;
            if inputs > 0 {
                for t in toks {
                    submitted_prompt.push(t.unsigned_integer().unwrap() as u32);
                }
                position += inputs;
                lifetime_primes += inputs - u64::from(reset);
                if reset && restored_prefix > 0 {
                    restored_field = Some((restored_prefix, inputs - restored_prefix));
                }
            } else {
                let gen_meta = get(job, "mtp_generation");
                gen_id = number(gen_meta, "id");
                let rem = number(gen_meta, "remaining");
                status = "length";
                for _ in 0..generate {
                    let tok = out_tokens.pop_front().unwrap();
                    position += 1;
                    lifetime_primes += 1;
                    ids.push(Json::integer(u64::from(tok)));
                    event(&Json::new_object([
                        ("event", Json::from("token")),
                        ("id", Json::integer(u64::from(tok))),
                        ("committed_prefix", Json::integer(position)),
                        ("entropy_words", Json::integer(0)),
                    ]))?;
                    if tok == 1000 {
                        status = "stopped";
                        stop_id = Json::integer(1000);
                        cont_allowed = false;
                        break;
                    }
                }
                let next_rem = if status == "stopped" {
                    0
                } else {
                    rem - ids.len() as u64
                };
                let _ = next_rem;
            }
            let next_rem = if inputs > 0 || status == "stopped" {
                0
            } else {
                number(get(job, "mtp_generation"), "remaining") - ids.len() as u64
            };
            let mut sampled = sampling.clone();
            for (name, value) in [
                ("valid", Json::Bool(true)),
                ("entropy_prefix", Json::integer(position)),
                ("entropy_words", Json::integer(0)),
                ("seed", Json::integer(44)),
            ] {
                put(&mut sampled, name, value);
            }
            let mut done = ready();
            for (name, value) in [
                ("event", Json::from("done")),
                ("committed_prefix", Json::integer(position)),
                ("generated_ids", Json::Array(ids)),
                ("sampling", sampled),
                ("status", Json::from(status)),
                ("stop_id", stop_id),
                ("continuation_allowed", Json::Bool(cont_allowed)),
                ("native_mtp_prime_calls", Json::integer(lifetime_primes)),
                (
                    "native_mtp_generation",
                    Json::new_object([
                        ("id", Json::integer(gen_id)),
                        ("active", Json::Bool(next_rem > 0)),
                        ("remaining", Json::integer(next_rem)),
                        ("pending", Json::Bool(false)),
                    ]),
                ),
            ] {
                put(&mut done, name, value);
            }
            if let Some((r, rep)) = restored_field {
                put(&mut done, "checkpoint_restored_tokens", Json::integer(r));
                put(&mut done, "replayed_input_tokens", Json::integer(rep));
            }
            Ok(done)
        };
        let mut check_cancelled = || false;
        let mut new_seed = || 44;
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut check_cancelled,
            new_seed: &mut new_seed,
            prefix_trace: None,
        };
        let resp = session
            .complete(&request, b"parked-think", &mut target, &mut sink)
            .unwrap();
        (resp, submitted_prompt)
    };

    let msg = |role: &str, content: &str| {
        Json::new_object([("role", Json::from(role)), ("content", Json::from(content))])
    };

    // Turn A1: generates reasoning "planA" and content "ansA".
    let (_, _) = run_turn(
        &mut session,
        vec![msg("user", "alpha")],
        b"planA",
        b"ansA",
        0,
    );
    let a1_committed = session.committed_tokens() as u64;

    // Turn B1 (short single-packet request): switches lanes, parking A1's saved reasoning.
    let (_, _) = run_turn(
        &mut session,
        vec![msg("user", "beta")],
        b"planB",
        b"ansB",
        0,
    );
    let b1_committed = session.committed_tokens() as u64;

    // Turn A2 (content-only client omits reasoning_content): parked reasoning restores "planA"
    // so the encoded prompt starts with A1's exact committed token IDs.
    let (resp_a2, prompt_a2) = run_turn(
        &mut session,
        vec![
            msg("user", "alpha"),
            msg("assistant", "ansA"),
            msg("user", "nextA"),
        ],
        b"planA2",
        b"ansA2",
        a1_committed,
    );
    assert_eq!(cached(&resp_a2), a1_committed);
    assert!(
        prompt_a2
            .windows(b"planA".len())
            .any(|w| w == b"planA".map(u32::from)),
        "restored prompt must include parked reasoning from A1"
    );

    // Turn B2 (content-only client omits reasoning_content): B1's parked reasoning also restores.
    let (resp_b2, prompt_b2) = run_turn(
        &mut session,
        vec![
            msg("user", "beta"),
            msg("assistant", "ansB"),
            msg("user", "nextB"),
        ],
        b"planB2",
        b"ansB2",
        b1_committed,
    );
    assert_eq!(cached(&resp_b2), b1_committed);
    assert!(
        prompt_b2
            .windows(b"planB".len())
            .any(|w| w == b"planB".map(u32::from)),
        "restored prompt must include parked reasoning from B1"
    );

    // Edited assistant content must NOT inject A1's parked reasoning.
    let (_, tampered_prompt) = run_turn(
        &mut session,
        vec![
            msg("user", "alpha"),
            msg("assistant", "EDITED"),
            msg("user", "nextA"),
        ],
        b"planT",
        b"ansT",
        0,
    );
    assert!(
        !tampered_prompt
            .windows(b"planA".len())
            .any(|w| w == b"planA".map(u32::from)),
        "edited history must not inherit parked reasoning"
    );
}

#[test]
fn implicit_seed_is_adopted_only_on_an_opted_in_parked_restore() {
    let cancelled = Cell::new(false);
    for (size, restored) in [(300, 200), (1100, 700)] {
        let prompt = "x".repeat(size);
        let mut session = session_with(true);
        let mut worker = Worker::new([Counts::Restored(restored.min(512))]);
        worker.lane = Some(restored);
        worker.inherited_lane = Some((99, restored));
        let (response, traces, chunks) =
            complete_with_seed(&mut session, &mut worker, &prompt, false, &cancelled, None);
        assert_usage(&response.unwrap(), &traces, &chunks, restored);
        // An explicit request for the inherited seed can append only if the session retained it.
        let next = format!("{prompt}ay");
        let (response, _, _) = complete_with_seed(
            &mut session,
            &mut worker,
            &next,
            false,
            &cancelled,
            Some(99),
        );
        assert_eq!(cached(&response.unwrap()), (size + 1) as u64);
        assert_eq!((worker.resets, worker.seed_opt_ins), (1, 1));
    }

    // An unsolicited report with the very same seed must still be refused.
    let mut session = session_with(true);
    let mut worker = Worker::new([Counts::Restored(2)]);
    worker.inherited_lane = Some((44, 2));
    let (result, _, _) = complete(&mut session, &mut worker, "fresh", false, &cancelled);
    assert!(result.is_err());
    assert_eq!(session.committed_tokens(), 0);
    assert_eq!(worker.seed_opt_ins, 0);

    // A seedless cache miss keeps the fresh seed supplied for the reset.
    let (result, _, _) =
        complete_with_seed(&mut session, &mut worker, "fresh", false, &cancelled, None);
    assert_eq!(cached(&result.unwrap()), 0);
    let (result, _, _) = complete(&mut session, &mut worker, "freshax", false, &cancelled);
    assert_eq!(cached(&result.unwrap()), 6);
    assert_eq!(worker.seed_opt_ins, 1);
}

#[test]
fn legacy_absence_and_present_zero_both_report_no_native_reuse() {
    for counts in [Counts::Legacy, Counts::Restored(0)] {
        let mut session = session();
        let mut worker = Worker::new([counts.clone()]);
        worker.output = counts;
        let (response, traces, chunks) =
            complete(&mut session, &mut worker, "cold", false, &Cell::new(false));
        assert_usage(&response.unwrap(), &traces, &chunks, 0);
    }
}

#[test]
fn malformed_packet_counts_refuse_without_publishing_usage_and_retry_cold() {
    let bad = [
        Counts::Fields(Some(Json::integer(1)), None),
        Counts::Fields(None, Some(Json::integer(511))),
        Counts::Fields(Some(Json::from("1")), Some(Json::integer(511))),
        Counts::Fields(Some(Json::Null), Some(Json::integer(512))),
        Counts::Fields(Some(Json::Bool(false)), Some(Json::integer(512))),
        Counts::Fields(Some(parse_json(b"1.0").unwrap()), Some(Json::integer(511))),
        Counts::Fields(Some(Json::integer(1)), Some(Json::integer(u64::MAX))),
        Counts::Fields(Some(parse_json(b"-1").unwrap()), Some(Json::integer(513))),
        Counts::Fields(Some(Json::integer(1)), Some(Json::from("511"))),
        Counts::Fields(Some(Json::integer(u64::MAX)), Some(Json::integer(0))),
        Counts::Fields(Some(Json::integer(513)), Some(Json::integer(0))),
        Counts::Fields(Some(Json::integer(1)), Some(Json::integer(510))),
    ];
    for counts in bad {
        let mut session = session();
        let mut worker = Worker::new([Counts::Restored(512), counts]);
        let cancelled = Cell::new(false);
        let (result, traces, chunks) = complete(
            &mut session,
            &mut worker,
            &"x".repeat(1024),
            true,
            &cancelled,
        );
        assert!(result.is_err());
        assert_eq!(session.committed_tokens(), 0);
        assert!(
            !traces
                .iter()
                .any(|item| get(item, "event").string().unwrap() == b"completion")
        );
        assert!(!chunks.iter().any(|item| {
            item.find("usage")
                .unwrap()
                .is_some_and(|usage| !usage.is_null())
        }));
        let (response, traces, chunks) =
            complete(&mut session, &mut worker, "retry", false, &cancelled);
        assert_usage(&response.unwrap(), &traces, &chunks, 0);
        assert_eq!(worker.resets, 2);
    }
}

#[test]
fn generation_cannot_claim_restored_inputs_and_existing_prime_validation_remains_strict() {
    for corrupt_prime in [false, true] {
        let mut session = session();
        let mut worker = Worker::new([Counts::Restored(4)]);
        worker.corrupt_prime_count = corrupt_prime;
        if !corrupt_prime {
            worker.output = Counts::Restored(1);
        }
        let (result, traces, _) =
            complete(&mut session, &mut worker, "cold", false, &Cell::new(false));
        assert!(result.is_err());
        assert_eq!(session.committed_tokens(), 0);
        assert!(
            !traces
                .iter()
                .any(|item| get(item, "event").string().unwrap() == b"completion")
        );
    }
}

#[test]
fn cancellation_after_admitted_restore_discards_request_usage_before_retry() {
    let mut session = session();
    let mut worker = Worker::new([Counts::Restored(512), Counts::Restored(128)]);
    worker.cancel_after_input = Some(2);
    let cancelled = Cell::new(false);
    let (result, traces, chunks) = complete(
        &mut session,
        &mut worker,
        &"x".repeat(1024),
        true,
        &cancelled,
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "request cancelled or deadline expired"
    );
    assert_eq!(session.committed_tokens(), 0);
    assert!(
        !traces
            .iter()
            .any(|item| get(item, "event").string().unwrap() == b"completion")
    );
    assert!(!chunks.iter().any(|item| {
        item.find("usage")
            .unwrap()
            .is_some_and(|usage| !usage.is_null())
    }));
    cancelled.set(false);
    worker.cancel_after_input = None;
    let (response, traces, chunks) =
        complete(&mut session, &mut worker, "retry", false, &cancelled);
    assert_usage(&response.unwrap(), &traces, &chunks, 0);
    assert_eq!(worker.resets, 2);
}

#[test]
fn restore_count_is_unavailable_until_the_entire_reply_is_admitted() {
    let ready = ready();
    let job = parse_json(br#"{"tokens":[1,2,3,4],"generate":0,"reset":true,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0},"seed":44}"#).unwrap();
    let mut worker = Worker::new([Counts::Restored(4)]);
    let done = worker
        .call(
            &job,
            &mut |_| panic!("prefill emitted token"),
            &Cell::new(false),
        )
        .unwrap();
    let mut reply = MtpWireReply::new(&ready, &ready, &job, false).unwrap();
    assert!(reply.restored_input_tokens().is_err());
    let mut bad = done.clone();
    put(&mut bad, "native_mtp_prime_calls", Json::integer(99));
    assert!(reply.finish(&bad).is_err());
    assert!(reply.restored_input_tokens().is_err());
    reply.finish(&done).unwrap();
    assert_eq!(reply.restored_input_tokens().unwrap(), 4);
}

#[test]
fn first_output_packet_streams_early_then_switches_to_steady_state_64_packets() {
    let mut ready64 = ready();
    put(
        &mut ready64,
        "native_mtp_output_packet_schema",
        Json::integer(1),
    );
    put(&mut ready64, "native_mtp_output_packet", Json::integer(64));

    let make_codec = || ServeCodec {
        control_ids: Vec::new(),
        nonordinary_ids: Vec::new(),
        render: Box::new(|request| {
            let messages = get(request, "messages").array().unwrap();
            Ok(RenderedChat {
                prompt: get(messages.last().unwrap(), "content")
                    .string()
                    .unwrap()
                    .to_vec(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| Ok(text.prompt.iter().map(|&byte| u32::from(byte)).collect())),
        decode: Box::new(|ids| Ok(ids.iter().map(|&id| u8::try_from(id).unwrap()).collect())),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    };

    assert!(
        ServeSession::new(
            make_codec(),
            SessionOptions {
                mtp_ready: ready(),
                mtp_first_output_packet: Some(16),
                ..SessionOptions::default()
            },
        )
        .is_err()
    );

    let mut session = ServeSession::new(
        make_codec(),
        SessionOptions {
            mtp_ready: ready64.clone(),
            mtp_first_output_packet: Some(8),
            ..SessionOptions::default()
        },
    )
    .unwrap();

    let request = Json::new_object([
        ("model", Json::from("tokspur")),
        (
            "messages",
            Json::Array(vec![Json::new_object([
                ("role", Json::from("user")),
                ("content", Json::from("hello")),
            ])]),
        ),
        ("max_tokens", Json::integer(80)),
        ("seed", Json::integer(44)),
        ("stream", Json::Bool(true)),
        ("enable_thinking", Json::Bool(false)),
    ]);

    let mut position = 0u64;
    let mut primes = 0u64;
    let mut sampling = Json::Null;
    let mut gen_widths = Vec::new();
    let streamed_content = Cell::new(0usize);
    let mut content_before_second_packet = 0usize;

    let mut chunk_cb = |value: &Json| {
        if let Some(choices) = value.find("choices").unwrap()
            && !choices.array().unwrap().is_empty()
            && choices.array().unwrap()[0]
                .find("delta")
                .unwrap()
                .unwrap()
                .find("content")
                .unwrap()
                .is_some()
        {
            streamed_content.set(streamed_content.get() + 1);
        }
        Ok(())
    };
    let mut check_cancelled = || false;
    let mut new_seed = || 44;
    let mut sink = ServeSink {
        chunk: Some(&mut chunk_cb),
        cancelled: &mut check_cancelled,
        new_seed: &mut new_seed,
        prefix_trace: None,
    };

    let mut target = |job: &Json, event: TokenCallback<'_>| -> Result<Json, ServeError> {
        let reset = get(job, "reset").boolean().unwrap();
        if reset {
            position = 0;
            sampling = get(job, "sampling").clone();
        }
        let inputs = get(job, "tokens").array().unwrap().len() as u64;
        let generate = number(job, "generate");
        let mut ids = Vec::new();
        let mut active = false;
        let mut remaining_after = 0u64;
        let mut gen_id = 0u64;
        if inputs > 0 {
            position += inputs;
            primes += inputs - u64::from(reset);
        } else {
            if generate == 64 && content_before_second_packet == 0 {
                content_before_second_packet = streamed_content.get();
            }
            gen_widths.push(generate);
            let gen_meta = get(job, "mtp_generation");
            gen_id = number(gen_meta, "id");
            let rem = number(gen_meta, "remaining");
            remaining_after = rem - generate;
            active = remaining_after > 0;
            for _ in 0..generate {
                position += 1;
                primes += 1;
                ids.push(Json::integer(u64::from(b'x')));
                event(&Json::new_object([
                    ("event", Json::from("token")),
                    ("id", Json::integer(u64::from(b'x'))),
                    ("committed_prefix", Json::integer(position)),
                    ("entropy_words", Json::integer(0)),
                ]))?;
            }
        }
        let mut sampled = sampling.clone();
        for (name, value) in [
            ("valid", Json::Bool(true)),
            ("entropy_prefix", Json::integer(position)),
            ("entropy_words", Json::integer(0)),
            ("seed", Json::integer(44)),
        ] {
            put(&mut sampled, name, value);
        }
        let mut done = ready64.clone();
        for (name, value) in [
            ("event", Json::from("done")),
            ("committed_prefix", Json::integer(position)),
            ("generated_ids", Json::Array(ids)),
            ("sampling", sampled),
            (
                "status",
                Json::from(if inputs > 0 { "prefilled" } else { "length" }),
            ),
            ("continuation_allowed", Json::Bool(true)),
            ("native_mtp_prime_calls", Json::integer(primes)),
            (
                "native_mtp_generation",
                Json::new_object([
                    ("id", Json::integer(gen_id)),
                    ("active", Json::Bool(active)),
                    ("remaining", Json::integer(remaining_after)),
                    ("pending", Json::Bool(false)),
                ]),
            ),
        ] {
            put(&mut done, name, value);
        }
        Ok(done)
    };

    let response = session
        .complete(&request, b"first-pkt-8", &mut target, &mut sink)
        .unwrap();
    assert_eq!(gen_widths, vec![8, 64, 8]);
    assert!(
        content_before_second_packet > 0,
        "first 8-token packet must flush SSE content chunks before starting steady-state 64-token packet"
    );
    let details = get(&response, "tokspur_private");
    assert_eq!(number(details, "native_mtp_first_output_packet"), 8);
    assert_eq!(number(details, "maximum_generation_width"), 64);
}

#[test]
fn phase_stamps_are_taken_in_order_only_when_enabled() {
    let mut session = session();
    let cancelled = Cell::new(false);
    let mut worker = Worker::new([Counts::Restored(0)]);
    let (response, _, _) = complete(&mut session, &mut worker, "cold", true, &cancelled);
    response.unwrap();
    let off = *session.phase_stamps();
    assert!(off.request_start.is_none() && off.prefill_start.is_none());
    assert!(off.first_packet_done.is_none() && off.generation_end.is_none());
    assert_eq!(off.output_packets, 0);

    session.set_phase_stamps(true);
    worker.inputs.extend([Counts::Restored(0)]);
    let (response, _, _) = complete(&mut session, &mut worker, "fresh", true, &cancelled);
    response.unwrap();
    let on = *session.phase_stamps();
    let order = [
        on.request_start.unwrap(),
        on.prefill_start.unwrap(),
        on.generation_start.unwrap(),
        on.first_packet_done.unwrap(),
        on.generation_end.unwrap(),
    ];
    assert!(order.windows(2).all(|pair| pair[0] <= pair[1]), "{on:?}");
    assert_eq!((on.output_packets, on.first_packet_tokens), (1, 1));
    assert_eq!((on.max_packet_tokens, on.completion_tokens), (1, 1));

    // A refusal before admission keeps only its own start, not the previous request's phases.
    let refused = session.complete(
        &Json::new_object([("model", Json::from("tokspur"))]),
        b"refused",
        &mut |_, _| unreachable!("a refused request starts no worker command"),
        &mut ServeSink {
            chunk: None,
            cancelled: &mut || false,
            new_seed: &mut || 44,
            prefix_trace: None,
        },
    );
    assert!(refused.is_err());
    let after = *session.phase_stamps();
    assert!(after.request_start.unwrap() >= on.generation_end.unwrap());
    assert!(after.prefill_start.is_none() && after.first_packet_done.is_none());
    assert_eq!(after.output_packets, 0);
}

#[test]
fn thinking_budget_mtp_v2_forced_close_and_natural_exit_within_one_generation() {
    let mut ready_v2 = ready();
    put(
        &mut ready_v2,
        "native_mtp_carry_scope",
        Json::from("http-generation-v2"),
    );
    let make_v2_gen = |id: u64,
                       active: bool,
                       remaining: u64,
                       pending: bool,
                       committed: u64,
                       emitted: u64,
                       overflow: u64,
                       remaining_compute: u64,
                       total_budget: u64,
                       continuation_allowed: bool| {
        Json::new_object([
            ("id", Json::integer(id)),
            ("active", Json::Bool(active)),
            ("remaining", Json::integer(remaining)),
            ("pending", Json::Bool(pending)),
            ("committed", Json::integer(committed)),
            ("emitted", Json::integer(emitted)),
            ("overflow", Json::integer(overflow)),
            ("remaining_compute", Json::integer(remaining_compute)),
            ("total_budget", Json::integer(total_budget)),
            ("continuation_allowed", Json::Bool(continuation_allowed)),
            ("stop_queued", Json::Bool(false)),
            ("stop_drained", Json::Bool(false)),
            (
                "stage_a_controller",
                Json::new_object([
                    ("ewma_alpha1", parse_json(b"0.45").unwrap()),
                    ("current_mode", Json::integer(0)),
                    ("consecutive_ordinary_steps", Json::integer(0)),
                    (
                        "controller_policy_digest",
                        Json::Array(vec![Json::integer(0); 32]),
                    ),
                ]),
            ),
        ])
    };
    put(
        &mut ready_v2,
        "native_mtp_generation",
        make_v2_gen(0, false, 0, false, 0, 0, 0, 0, 0, true),
    );

    let make_codec = || ServeCodec {
        control_ids: Vec::new(),
        nonordinary_ids: Vec::new(),
        render: Box::new(|request| {
            let messages = get(request, "messages").array().unwrap();
            Ok(RenderedChat {
                prompt: get(messages.last().unwrap(), "content")
                    .string()
                    .unwrap()
                    .to_vec(),
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        }),
        encode: Box::new(|text| {
            if text.prompt == b"</think>" {
                Ok(vec![2000])
            } else {
                Ok(text.prompt.iter().map(|&byte| u32::from(byte)).collect())
            }
        }),
        decode: Box::new(|ids| {
            let mut out = Vec::new();
            for &id in ids {
                match id {
                    1000 => out.extend_from_slice(b"<|im_end|>"),
                    1001 => out.extend_from_slice(b"<|endoftext|>"),
                    2000 => out.extend_from_slice(b"</think>"),
                    _ => out.push(u8::try_from(id).unwrap()),
                }
            }
            Ok(out)
        }),
        admission: ResponseAdmission {
            vocabulary: 4096,
            im_end: 1000,
            end_of_text: 1001,
            im_start: 1002,
        },
    };

    // Sub-case 1: the budget is reached. With thinking still open the close is forced as known input; with thinking already closed by the model nothing is injected. Either way phase 2 is one new generation and nothing is aborted.
    let bytes = |text: &[u8]| text.iter().map(|&b| u32::from(b)).collect::<Vec<u32>>();
    let natural_then_answer = [bytes(b"hi"), vec![2000], bytes(b"\n\nabc")].concat();
    for (budget, first, second, jobs, reasoning, content) in [
        (
            2u64,
            bytes(b"hi"),
            bytes(b"ok"),
            4usize,
            &b"hi\n"[..],
            &b"\n\nok"[..],
        ),
        (
            8,
            natural_then_answer,
            bytes(b"d"),
            3,
            &b"hi"[..],
            &b"\n\nabcd"[..],
        ),
    ] {
        let mut session = ServeSession::new(
            make_codec(),
            SessionOptions {
                mtp_ready: ready_v2.clone(),
                ..SessionOptions::default()
            },
        )
        .unwrap();
        session.set_phase_stamps(true);

        let request = Json::new_object([
            ("model", Json::from("tokspur")),
            (
                "messages",
                Json::Array(vec![Json::new_object([
                    ("role", Json::from("user")),
                    ("content", Json::from("go")),
                ])]),
            ),
            ("thinking_budget", Json::integer(budget)),
            ("max_tokens", Json::integer(12)),
            ("seed", Json::integer(44)),
            ("stream", Json::Bool(false)),
        ]);

        let mut position = 0u64;
        let mut primes = 0u64;
        let mut sampling = Json::Null;
        let mut seen_jobs = Vec::new();
        let mut phase1_tokens = VecDeque::from(first);
        let mut phase2_tokens: VecDeque<u32> = second.into_iter().chain([1000u32]).collect();

        let mut target = |job: &Json, event: TokenCallback<'_>| -> Result<Json, ServeError> {
            seen_jobs.push(job.clone());
            let reset = get(job, "reset").boolean().unwrap();
            if reset {
                position = 0;
                sampling = get(job, "sampling").clone();
            }
            let inputs = get(job, "tokens").array().unwrap().len() as u64;
            let generate = number(job, "generate");
            let mut ids = Vec::new();
            let mut status = "prefilled";
            let mut stop_id = Json::Null;
            let mut cont_allowed = true;
            let gen_state = if inputs > 0 {
                assert!(job.find("mtp_generation").unwrap().is_none());
                position += inputs;
                primes += inputs - u64::from(reset);
                make_v2_gen(
                    if reset { 0 } else { 1 },
                    false,
                    0,
                    false,
                    0,
                    0,
                    0,
                    0,
                    0,
                    true,
                )
            } else {
                let gen_meta = get(job, "mtp_generation");
                let gen_id = number(gen_meta, "id");
                let rem = number(gen_meta, "remaining");
                let total = number(gen_meta, "total_budget");
                status = "length";
                let queue = if gen_id == 1 {
                    &mut phase1_tokens
                } else {
                    &mut phase2_tokens
                };
                let mut emitted_this_pkt = 0u64;
                while emitted_this_pkt < generate {
                    let tok = queue.pop_front().unwrap();
                    position += 1;
                    primes += 1;
                    emitted_this_pkt += 1;
                    ids.push(Json::integer(u64::from(tok)));
                    event(&Json::new_object([
                        ("event", Json::from("token")),
                        ("id", Json::integer(u64::from(tok))),
                        ("committed_prefix", Json::integer(position)),
                        ("entropy_words", Json::integer(0)),
                    ]))?;
                    if tok == 1000 {
                        status = "stopped";
                        stop_id = Json::integer(1000);
                        cont_allowed = false;
                        break;
                    }
                }
                let rem_after = if status == "stopped" {
                    0
                } else {
                    rem - emitted_this_pkt
                };
                make_v2_gen(
                    gen_id,
                    rem_after > 0,
                    rem_after,
                    false,
                    emitted_this_pkt,
                    emitted_this_pkt,
                    0,
                    rem_after,
                    total,
                    cont_allowed,
                )
            };
            let mut sampled = sampling.clone();
            for (name, value) in [
                ("valid", Json::Bool(true)),
                ("entropy_prefix", Json::integer(position)),
                ("entropy_words", Json::integer(0)),
                ("seed", Json::integer(44)),
            ] {
                put(&mut sampled, name, value);
            }
            let mut done = ready_v2.clone();
            for (name, value) in [
                ("event", Json::from("done")),
                ("committed_prefix", Json::integer(position)),
                ("generated_ids", Json::Array(ids)),
                ("sampling", sampled),
                ("status", Json::from(status)),
                ("stop_id", stop_id),
                ("continuation_allowed", Json::Bool(cont_allowed)),
                ("native_mtp_prime_calls", Json::integer(primes)),
                ("native_mtp_generation", gen_state),
            ] {
                put(&mut done, name, value);
            }
            Ok(done)
        };

        let mut check_cancelled = || false;
        let mut new_seed = || 44;
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut check_cancelled,
            new_seed: &mut new_seed,
            prefix_trace: None,
        };

        let response = session
            .complete(&request, b"mtp-forced-close", &mut target, &mut sink)
            .unwrap();
        // Prefill, phase 1 (id=1), the forced-close input only when thinking was still open, phase 2 (id=2).
        assert_eq!(seen_jobs.len(), jobs);
        let known_inputs = seen_jobs
            .iter()
            .filter(|job| !get(job, "tokens").array().unwrap().is_empty())
            .count();
        assert_eq!(known_inputs, jobs - 2);
        let generation_ids: Vec<u64> = seen_jobs
            .iter()
            .filter_map(|job| job.find("mtp_generation").unwrap())
            .map(|generation| {
                assert!(generation.find("abort").unwrap().is_none());
                number(generation, "id")
            })
            .collect();
        assert_eq!(generation_ids, [1, 2]);
        let msg = get(&get(&response, "choices").array().unwrap()[0], "message");
        assert_eq!(get(msg, "reasoning_content").string().unwrap(), reasoning);
        assert_eq!(get(msg, "content").string().unwrap(), content);
        assert_eq!(session.phase_stamps().output_packets, 2);
        assert_eq!(
            number(get(&response, "tokspur_private"), "generation_batches"),
            2
        );
    }

    // Sub-case 2: the model closes thinking early under http-generation-v2 while one speculative overflow token is already committed. The same generation carries on through the answer: the overflow token leads the next ordinary packet, and nothing is drained separately or aborted.
    {
        let mut session = ServeSession::new(
            make_codec(),
            SessionOptions {
                mtp_ready: ready_v2.clone(),
                ..SessionOptions::default()
            },
        )
        .unwrap();
        session.set_phase_stamps(true);

        let request = Json::new_object([
            ("model", Json::from("tokspur")),
            (
                "messages",
                Json::Array(vec![Json::new_object([
                    ("role", Json::from("user")),
                    ("content", Json::from("go")),
                ])]),
            ),
            ("thinking_budget", Json::integer(12)),
            ("max_tokens", Json::integer(20)),
            ("seed", Json::integer(44)),
            ("stream", Json::Bool(false)),
        ]);

        let mut position = 0u64;
        let mut primes = 0u64;
        let mut head_calls = 0u64;
        let mut rounds = 0u64;
        let mut proposed = 0u64;
        let mut accepted = 0u64;
        let discarded = 0u64;
        let mut sampling = Json::Null;
        let mut carried = 0u64;
        let mut output_packets = 0usize;

        let mut target = |job: &Json, event: TokenCallback<'_>| -> Result<Json, ServeError> {
            let reset = get(job, "reset").boolean().unwrap();
            if reset {
                position = 0;
                sampling = get(job, "sampling").clone();
            }
            let inputs = get(job, "tokens").array().unwrap().len() as u64;
            let generate = number(job, "generate");
            let mut ids = Vec::new();
            let mut status = "prefilled";
            let mut stop_id = Json::Null;
            let mut cont_allowed = true;
            let gen_state = if inputs > 0 {
                position += inputs;
                primes += inputs - u64::from(reset);
                make_v2_gen(0, false, 0, false, 0, 0, 0, 0, 0, true)
            } else {
                let gen_meta = get(job, "mtp_generation");
                let gen_id = number(gen_meta, "id");
                assert!(gen_meta.find("abort").unwrap().is_none());
                assert_eq!(gen_id, 1);
                output_packets += 1;
                {
                    let begin = get(gen_meta, "begin").boolean().unwrap();
                    let rem = number(gen_meta, "remaining");
                    let total = number(gen_meta, "total_budget");
                    status = "length";
                    if begin {
                        // Emit 6 ordinary tokens ("think\n"), then an mtp_round committing 3 tokens:
                        // emitted token 7 ('\n'), emitted token 8 (2000 = "</think>"), and 1 overflow token ('A').
                        assert_eq!(generate, 8);
                        for &b in b"think\n" {
                            position += 1;
                            primes += 1;
                            ids.push(Json::integer(u64::from(b)));
                            event(&Json::new_object([
                                ("event", Json::from("token")),
                                ("id", Json::integer(u64::from(b))),
                                ("committed_prefix", Json::integer(position)),
                                ("entropy_words", Json::integer(0)),
                            ]))?;
                        }
                        let round_base = position;
                        let round_end = round_base + 3;
                        head_calls += 2;
                        primes += 2;
                        rounds += 1;
                        proposed += 2;
                        accepted += 2;
                        event(&Json::new_object([
                            ("event", Json::from("mtp_round")),
                            ("base", Json::integer(round_base)),
                            ("committed_prefix", Json::integer(round_end)),
                            ("count", Json::integer(3)),
                            ("accepted", Json::integer(2)),
                            ("proposed", Json::integer(2)),
                            ("head_calls", Json::integer(2)),
                            ("prime_extra", Json::integer(0)),
                            ("lookahead_anchor", Json::Bool(true)),
                            ("used_carried_anchor", Json::Bool(false)),
                            ("pending_anchor", Json::Bool(true)),
                        ]))?;
                        for (offset, tok) in [(1u64, u32::from(b'\n')), (2u64, 2000u32)] {
                            ids.push(Json::integer(u64::from(tok)));
                            event(&Json::new_object([
                                ("event", Json::from("token")),
                                ("id", Json::integer(u64::from(tok))),
                                ("committed_prefix", Json::integer(round_end)),
                                ("round_committed_prefix", Json::integer(round_end)),
                                ("token_prefix", Json::integer(round_base + offset)),
                                ("entropy_words", Json::integer(0)),
                            ]))?;
                        }
                        position = round_end; // 1 overflow token ('A') already committed in KV/anchor state!
                        make_v2_gen(gen_id, true, rem - 8, true, 9, 8, 1, total - 9, total, true)
                    } else {
                        // Continuation of the same generation with the rest of the phase budget: pops overflow 'A' with no GPU work, then decodes 'B' against the carried anchor and stops.
                        assert_eq!(generate, 4);
                        assert_eq!(rem, 4);
                        assert_eq!(number(gen_meta, "pending"), 1);
                        assert_eq!(number(gen_meta, "committed"), 9);
                        assert_eq!(number(gen_meta, "emitted"), 8);
                        let tok = u32::from(b'A');
                        ids.push(Json::integer(u64::from(tok)));
                        event(&Json::new_object([
                            ("event", Json::from("token")),
                            ("id", Json::integer(u64::from(tok))),
                            ("committed_prefix", Json::integer(position)),
                            ("round_committed_prefix", Json::integer(position)),
                            ("token_prefix", Json::integer(position)),
                            ("entropy_words", Json::integer(0)),
                        ]))?;
                        carried += 1;
                        for tok in [u32::from(b'B'), 1000u32] {
                            position += 1;
                            primes += 1;
                            ids.push(Json::integer(u64::from(tok)));
                            event(&Json::new_object([
                                ("event", Json::from("token")),
                                ("id", Json::integer(u64::from(tok))),
                                ("committed_prefix", Json::integer(position)),
                                ("entropy_words", Json::integer(0)),
                            ]))?;
                        }
                        status = "stopped";
                        stop_id = Json::integer(1000);
                        cont_allowed = false;
                        make_v2_gen(gen_id, false, 0, false, 11, 11, 0, total - 11, total, false)
                    }
                }
            };
            let mut sampled = sampling.clone();
            for (name, value) in [
                ("valid", Json::Bool(true)),
                ("entropy_prefix", Json::integer(position)),
                ("entropy_words", Json::integer(0)),
                ("seed", Json::integer(44)),
            ] {
                put(&mut sampled, name, value);
            }
            let mut done = ready_v2.clone();
            for (name, value) in [
                ("event", Json::from("done")),
                ("committed_prefix", Json::integer(position)),
                ("generated_ids", Json::Array(ids)),
                ("sampling", sampled),
                ("status", Json::from(status)),
                ("stop_id", stop_id),
                ("continuation_allowed", Json::Bool(cont_allowed)),
                ("native_mtp_head_calls", Json::integer(head_calls)),
                ("native_mtp_prime_calls", Json::integer(primes)),
                ("native_mtp_rounds", Json::integer(rounds)),
                ("native_mtp_proposed", Json::integer(proposed)),
                ("native_mtp_accepted", Json::integer(accepted)),
                ("native_mtp_carried_anchors", Json::integer(carried)),
                ("native_mtp_discarded_anchors", Json::integer(discarded)),
                ("native_mtp_generation", gen_state),
            ] {
                put(&mut done, name, value);
            }
            Ok(done)
        };

        let mut check_cancelled = || false;
        let mut new_seed = || 44;
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut check_cancelled,
            new_seed: &mut new_seed,
            prefix_trace: None,
        };

        let response = session
            .complete(&request, b"mtp-v2-natural-exit", &mut target, &mut sink)
            .unwrap();
        assert_eq!(output_packets, 2);
        assert_eq!((carried, discarded), (1, 0));
        let msg = get(&get(&response, "choices").array().unwrap()[0], "message");
        assert_eq!(
            get(msg, "reasoning_content").string().unwrap(),
            b"think\n\n"
        );
        assert_eq!(get(msg, "content").string().unwrap(), b"AB");
        // Two output packets of one generation; the response ends inside the thinking budget, so there is no phase 2.
        assert_eq!(session.phase_stamps().output_packets, 2);
        assert_eq!(
            number(get(&response, "tokspur_private"), "generation_batches"),
            2
        );
    }
}

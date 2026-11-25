//! `ServeSession`, ported one to one from `native/src/serve_protocol.cpp`.
//!
//! CPU-only seams: production binds the codec to the owned tokenizer and renderer and the target to exactly one owned worker; tests supply deterministic fixtures.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use tokspur_json::{Json, JsonError, Kind};
use tokspur_text::chat::RenderedChat;
use tokspur_text::response::{
    ResponseKind, ResponseLimits, ResponseParser, ResponseRecord, ResponseStatus,
};
use tokspur_text::response_protocol::{
    RESPONSE_MAXIMUM_ACCUMULATED_IDS, ResponseAdmission, ResponseProtocolError,
    parse_native_response_with_controls,
};

use crate::admission::{
    AdmissionError, AdmittedChat, ToolArgValidation, ToolEnumValidation, admit_chat_with_context,
    broker_storage, chunk, field, number, parser_manifest, real, require,
    validate_executor_tool_arguments, validate_tool_value_with_policy, validate_worker_ready,
    worker_execution,
};
use crate::http::HttpError;
use crate::request_timing::PhaseStamps;
use tokspur_wire::worker::WorkerError;

#[cfg(feature = "verify8")]
use tokspur_wire::verify8::{Verify8Context, Verify8Error, Verify8Reply};

/// Everything `ServeSession::complete` (and its callbacks) can fail with; `Display` is the C++ `what()` text.
pub enum ServeError {
    /// Admission or a session check refused (`FormatError`).
    Admission(AdmissionError),
    /// A text-layer operation (render, encode, decode, final parse) failed.
    Text(Box<dyn std::error::Error + Send + Sync>),
    /// The worker refused, failed or was cancelled.
    Worker(WorkerError),
    /// V8/branch8 reply admission refused.
    #[cfg(feature = "verify8")]
    Verify8(Verify8Error),
    /// Streaming to the client failed.
    Http(HttpError),
}

impl fmt::Debug for ServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::Text(_) => formatter.write_str("Text([redacted])"),
            Self::Worker(error) => formatter.debug_tuple("Worker").field(error).finish(),
            #[cfg(feature = "verify8")]
            Self::Verify8(error) => formatter.debug_tuple("Verify8").field(error).finish(),
            Self::Http(error) => formatter.debug_tuple("Http").field(error).finish(),
        }
    }
}

impl fmt::Display for ServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => error.fmt(formatter),
            Self::Text(error) => error.fmt(formatter),
            Self::Worker(error) => error.fmt(formatter),
            #[cfg(feature = "verify8")]
            Self::Verify8(error) => error.fmt(formatter),
            Self::Http(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Text(error) => Some(error.as_ref()),
            Self::Worker(error) => Some(error),
            #[cfg(feature = "verify8")]
            Self::Verify8(error) => Some(error),
            Self::Http(error) => Some(error),
        }
    }
}

impl From<AdmissionError> for ServeError {
    fn from(error: AdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl From<JsonError> for ServeError {
    fn from(error: JsonError) -> Self {
        Self::Admission(AdmissionError::Json(error))
    }
}

impl From<WorkerError> for ServeError {
    fn from(error: WorkerError) -> Self {
        Self::Worker(error)
    }
}

#[cfg(feature = "verify8")]
impl From<Verify8Error> for ServeError {
    fn from(error: Verify8Error) -> Self {
        Self::Verify8(error)
    }
}

impl From<HttpError> for ServeError {
    fn from(error: HttpError) -> Self {
        Self::Http(error)
    }
}

impl From<tokspur_wire::mtp_wire::MtpWireError> for ServeError {
    fn from(error: tokspur_wire::mtp_wire::MtpWireError) -> Self {
        Self::Admission(AdmissionError::Mtp(error))
    }
}

type Result<T> = std::result::Result<T, ServeError>;

/// Renders an admitted request into prompt text (`ServeCodec::render`).
pub type RenderFn = Box<dyn Fn(&Json) -> Result<RenderedChat> + Send>;
/// Encodes renderer output into token IDs while retaining its literal byte ranges (`ServeCodec::encode`).
pub type EncodeFn = Box<dyn Fn(&RenderedChat) -> Result<Vec<u32>> + Send>;
/// Decodes generated IDs into their exact bytes (`ServeCodec::decode`).
pub type DecodeFn = Box<dyn Fn(&[u32]) -> Result<Vec<u8>> + Send>;
/// Receives one streaming chunk (`ServeSink::chunk`).
pub type ChunkSink<'a> = &'a mut dyn FnMut(&Json) -> Result<()>;

/// `ServeCodec`.
pub struct ServeCodec {
    pub render: RenderFn,
    pub encode: EncodeFn,
    pub decode: DecodeFn,
    pub admission: ResponseAdmission,
    /// Added-token IDs of the response frame markers (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`). Empty keeps the byte-spelled framing.
    pub control_ids: Vec<u32>,
    /// All added tokens and reserved padding slots, including markers outside response framing.
    pub nonordinary_ids: Vec<u32>,
}

impl fmt::Debug for ServeCodec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ServeCodec").finish_non_exhaustive()
    }
}

/// The token-event callback a target receives.
pub type TokenCallback<'a> = &'a mut dyn FnMut(&Json) -> Result<()>;
/// `ServeTarget`: submits one worker command and returns its completion.
pub type ServeTarget<'a> = &'a mut dyn FnMut(&Json, TokenCallback<'_>) -> Result<Json>;

/// Receives one trace event (`ServeSink::prefix_trace`).
pub type TraceSink<'a> = &'a mut dyn FnMut(&Json) -> Result<()>;

/// `ServeSink`.
pub struct ServeSink<'a> {
    pub chunk: Option<ChunkSink<'a>>,
    pub cancelled: &'a mut dyn FnMut() -> bool,
    pub new_seed: &'a mut dyn FnMut() -> u64,
    pub prefix_trace: Option<TraceSink<'a>>,
}

/// The `ServeSession` constructor arguments after the codec.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub alias: Vec<u8>,
    /// Immutable startup choice, never a request field.
    pub stack_execution: String,
    pub tool_enums: ToolEnumValidation,
    pub tool_args: ToolArgValidation,
    pub broker_ready: Json,
    #[cfg(feature = "verify8")]
    pub verify8: bool,
    #[cfg(feature = "verify8")]
    pub verify_ready: Json,
    #[cfg(feature = "branch8")]
    pub branch8: bool,
    pub mtp_ready: Json,
    pub mtp_first_output_packet: Option<u32>,
    pub overlap_prefill: bool,
    pub lane_parking: bool,
    pub context_length: u32,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            alias: b"tokspur".to_vec(),
            stack_execution: "direct".to_owned(),
            tool_enums: ToolEnumValidation::Strict,
            tool_args: ToolArgValidation::Strict,
            broker_ready: Json::Null,
            #[cfg(feature = "verify8")]
            verify8: false,
            #[cfg(feature = "verify8")]
            verify_ready: Json::Null,
            #[cfg(feature = "branch8")]
            branch8: false,
            mtp_ready: Json::Null,
            mtp_first_output_packet: None,
            overlap_prefill: false,
            lane_parking: false,
            context_length: 65_536,
        }
    }
}

/// One persistent conversation lane over one owned target.
const MAX_PARKED_REASONING_LANES: usize = 16;

struct ParkedReasoning {
    committed: Vec<u32>,
    turns: Vec<(usize, Vec<u8>)>,
    sampling: Vec<u8>,
    seed: u64,
}

pub struct ServeSession {
    codec: ServeCodec,
    alias: Vec<u8>,
    stack_execution: String,
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
    context_length: u32,
    committed: Vec<u32>,
    sampling: Vec<u8>,
    seed: u64,
    reusable: bool,
    broker_ready: Json,
    broker_epoch: u64,
    broker_command: u64,
    broker_lease: u64,
    broker_resync: bool,
    #[cfg(feature = "verify8")]
    verify8: bool,
    #[cfg(feature = "verify8")]
    verify_ready: Json,
    #[cfg(feature = "verify8")]
    verify_previous: Json,
    /// Only a cold reset may rebase metadata after a drained cancellation.
    #[cfg(feature = "verify8")]
    verify_resync: bool,
    #[cfg(feature = "branch8")]
    branch8: bool,
    pub mtp: bool,
    pub mtp_ready: Json,
    pub mtp_previous: Json,
    pub mtp_first_output_packet: Option<u32>,
    pub mtp_resync: bool,
    pub mtp_generation_id: u64,
    pub mtp_committed: u64,
    pub mtp_emitted: u64,
    pub mtp_pending: u64,
    pub mtp_stage_a_controller: Option<tokspur_wire::mtp_wire::StageAControllerState>,
    pub overlap_prefill: bool,
    /// Name the whole prompt on a fresh packet so the worker can park and restore whole conversations.
    pub lane_parking: bool,
    /// Reasoning this session generated, keyed by the assistant message's ordinal in the request history.
    saved_reasoning: Vec<(usize, Vec<u8>)>,
    /// Reasoning saved from conversations parked on a lane switch, verified against their exact committed token IDs.
    parked_reasoning: Vec<ParkedReasoning>,
    /// Off by default; only the opt-in `request_timing` record reads the stamps.
    phase_stamps_enabled: bool,
    phase_stamps: PhaseStamps,
    last_req_telemetry: Option<RequestCheckpointTelemetry>,
}

#[derive(Clone, Copy, Default)]
struct RequestCheckpointTelemetry {
    ssd_restored: bool,
    reuse: Option<&'static str>,
    restored_tokens: u64,
    replayed_tokens: u64,
}

impl fmt::Debug for ServeSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Committed token IDs belong to a real conversation; count them only.
        formatter
            .debug_struct("ServeSession")
            .field("committed_tokens", &self.committed.len())
            .field("reusable", &self.reusable)
            .finish_non_exhaustive()
    }
}

fn is(value: &Json, expected: &str) -> std::result::Result<bool, JsonError> {
    Ok(value.string()? == expected.as_bytes())
}

impl ServeSession {
    pub fn new(mut codec: ServeCodec, options: SessionOptions) -> Result<Self> {
        require(
            codec
                .nonordinary_ids
                .iter()
                .all(|&id| id < codec.admission.vocabulary),
            "nonordinary token outside vocabulary",
        )?;
        codec.nonordinary_ids.sort_unstable();
        codec.nonordinary_ids.dedup();
        let mtp = !options.mtp_ready.is_null();
        let mut session = Self {
            codec,
            alias: options.alias,
            stack_execution: options.stack_execution,
            tool_enums: options.tool_enums,
            tool_args: options.tool_args,
            context_length: options.context_length,
            committed: Vec::new(),
            sampling: Vec::new(),
            seed: 0,
            reusable: false,
            broker_ready: options.broker_ready,
            broker_epoch: 0,
            broker_command: 0,
            broker_lease: 0,
            broker_resync: true,
            #[cfg(feature = "verify8")]
            verify8: options.verify8,
            #[cfg(feature = "verify8")]
            verify_ready: options.verify_ready,
            #[cfg(feature = "verify8")]
            verify_previous: Json::Null,
            #[cfg(feature = "verify8")]
            verify_resync: false,
            #[cfg(feature = "branch8")]
            branch8: options.branch8,
            mtp,
            mtp_ready: options.mtp_ready.clone(),
            mtp_previous: if mtp { options.mtp_ready } else { Json::Null },
            mtp_first_output_packet: options.mtp_first_output_packet,
            mtp_resync: false,
            mtp_generation_id: 0,
            mtp_committed: 0,
            mtp_emitted: 0,
            mtp_pending: 0,
            mtp_stage_a_controller: None,
            overlap_prefill: options.overlap_prefill,
            lane_parking: options.lane_parking && mtp,
            saved_reasoning: Vec::new(),
            parked_reasoning: Vec::new(),
            phase_stamps_enabled: false,
            phase_stamps: PhaseStamps::default(),
            last_req_telemetry: None,
        };
        require(!session.alias.is_empty(), "invalid serving codec")?;
        require(
            session.context_length == 65_536 || session.context_length == 131_072,
            "unsupported serving context length",
        )?;
        require(
            session.stack_execution == "direct" || session.stack_execution == "recurrent-replay",
            "unsupported serving startup execution mode",
        )?;
        let context_cap = session.context_length as usize;
        if session.mtp {
            let draft = tokspur_wire::mtp_wire::mtp_draft_attention(&session.mtp_ready)?;
            let carry = tokspur_wire::mtp_wire::mtp_request_carry(&session.mtp_ready)?;
            let depth = tokspur_wire::mtp_wire::mtp_depth(&session.mtp_ready)?;
            tokspur_wire::mtp_wire::validate_mtp_ready(&session.mtp_ready, draft, carry, depth)?;
            session.committed.reserve(context_cap);
        }
        if let Some(first_pkt) = session.mtp_first_output_packet {
            require(
                session.mtp,
                "MTP first-output-packet requires native MTP readiness",
            )?;
            let cap = tokspur_wire::mtp_wire::mtp_output_packet(&session.mtp_ready)?;
            require(
                (1..=cap).contains(&first_pkt),
                "MTP first-output-packet outside negotiated output packet bound",
            )?;
        }
        #[cfg(feature = "verify8")]
        if session.verify8 {
            require(
                session.stack_execution == "direct",
                "private HTTP V8 cannot use replay",
            )?;
            if session.branch8_mode() {
                #[cfg(feature = "branch8")]
                {
                    require(
                        !session.broker_ready.is_null()
                            && session.broker_ready.dump()? == session.verify_ready.dump()?,
                        "branch8 requires the same explicit broker/verifier readiness",
                    )?;
                    tokspur_wire::verify8::validate_branch8_ready(&session.verify_ready)?;
                    session.verify_previous = session.verify_ready.clone();
                }
            } else {
                require(
                    session.broker_ready.is_null(),
                    "private HTTP V8 cannot use checkpoints without explicit branch8 consent",
                )?;
                tokspur_wire::verify8::validate_verify8_ready(&session.verify_ready)?;
            }
            session.committed.reserve(context_cap);
        } else {
            require(
                session.verify_ready.is_null(),
                "unexpected private V8 readiness",
            )?;
        }
        #[cfg(feature = "branch8")]
        require(
            !session.branch8 || session.verify8,
            "branch8 requires explicit private V8 scheduling",
        )?;
        if !session.broker_ready.is_null() {
            let slots = field(&session.broker_ready, "checkpoint_slots")?.unsigned_integer()?;
            require(slots != 0 && slots <= 4, "serving broker slot bound")?;
            let budget = field(&session.broker_ready, "checkpoint_combined_budget_bytes")?
                .unsigned_integer()?;
            validate_worker_ready(
                &session.broker_ready,
                &session.stack_execution,
                slots as u32,
                budget,
            )?;
            session.broker_epoch =
                field(&session.broker_ready, "broker_epoch")?.unsigned_integer()?;
            session.committed.reserve(context_cap); // Before any restore publication.
        }
        Ok(session)
    }

    fn branch8_mode(&self) -> bool {
        #[cfg(feature = "branch8")]
        {
            self.branch8
        }
        #[cfg(not(feature = "branch8"))]
        {
            false
        }
    }

    fn verify8_mode(&self) -> bool {
        #[cfg(feature = "verify8")]
        {
            self.verify8
        }
        #[cfg(not(feature = "verify8"))]
        {
            false
        }
    }

    /// Turns on the phase stamps that the opt-in `request_timing` record reads. Off, no stamp is taken.
    pub fn set_phase_stamps(&mut self, enabled: bool) {
        self.phase_stamps_enabled = enabled;
        self.phase_stamps = PhaseStamps::default();
    }

    /// The latest request's phase stamps; all empty unless [`Self::set_phase_stamps`] turned them on.
    pub fn phase_stamps(&self) -> &PhaseStamps {
        &self.phase_stamps
    }

    /// Overlays request-level prefill checkpoint telemetry onto the terminal `done` record.
    pub fn apply_request_telemetry(&self, done: &Json) -> Json {
        let Some(tel) = self.last_req_telemetry else {
            return done.clone();
        };
        let Ok(fields) = done.object() else {
            return done.clone();
        };
        let mut out = Vec::with_capacity(fields.len() + 4);
        let mut has_ssd = false;
        let mut has_reuse = false;
        let mut has_restored = false;
        let mut has_replayed = false;
        for (key, value) in fields {
            match key.as_slice() {
                b"checkpoint_ssd_restored" => {
                    has_ssd = true;
                    let val = value.boolean().unwrap_or(false) || tel.ssd_restored;
                    out.push((key.clone(), Json::Bool(val)));
                }
                b"checkpoint_reuse" => {
                    has_reuse = true;
                    let val = tel.reuse.map(Json::from).unwrap_or_else(|| value.clone());
                    out.push((key.clone(), val));
                }
                b"checkpoint_restored_tokens" => {
                    has_restored = true;
                    out.push((key.clone(), Json::integer(tel.restored_tokens)));
                }
                b"replayed_input_tokens" => {
                    has_replayed = true;
                    out.push((key.clone(), Json::integer(tel.replayed_tokens)));
                }
                _ => out.push((key.clone(), value.clone())),
            }
        }
        if !has_ssd && tel.ssd_restored {
            out.push((b"checkpoint_ssd_restored".to_vec(), Json::Bool(true)));
        }
        if !has_reuse && let Some(reuse) = tel.reuse {
            out.push((b"checkpoint_reuse".to_vec(), Json::from(reuse)));
        }
        if !has_restored {
            out.push((
                b"checkpoint_restored_tokens".to_vec(),
                Json::integer(tel.restored_tokens),
            ));
        }
        if !has_replayed {
            out.push((
                b"replayed_input_tokens".to_vec(),
                Json::integer(tel.replayed_tokens),
            ));
        }
        Json::Object(out)
    }

    /// The first output packet's worker command has returned; `batches` counts packets so far.
    fn stamp_output_packet(&mut self, batches: u64, tokens: usize) {
        if self.phase_stamps_enabled && batches == 1 {
            self.phase_stamps.first_packet_done = Some(std::time::Instant::now());
            self.phase_stamps.first_packet_tokens = tokens as u64;
        }
    }

    /// Drops reusable state after any failure.
    pub fn invalidate(&mut self) {
        self.saved_reasoning.clear();
        self.parked_reasoning.clear();
        self.committed.clear();
        self.sampling.clear();
        self.reusable = false;
        self.last_req_telemetry = None;
        self.seed = 0;
        self.broker_resync = true;
        self.mtp_resync = true;
        self.mtp_committed = 0;
        self.mtp_emitted = 0;
        self.mtp_pending = 0;
        self.mtp_stage_a_controller = None;
        #[cfg(feature = "verify8")]
        {
            self.verify_resync = true;
        }
    }

    pub fn committed_tokens(&self) -> usize {
        self.committed.len()
    }

    /// The request with saved reasoning restored to assistant turns that arrived without it.
    fn fill_saved_reasoning(render: &Json, saved: &[(usize, Vec<u8>)]) -> Result<Option<Json>> {
        if saved.is_empty() {
            return Ok(None);
        }
        let mut fields = render.object()?.clone();
        let Some((_, messages)) = fields.iter_mut().find(|(name, _)| name == b"messages") else {
            return Ok(None);
        };
        let mut list = messages.array()?.clone();
        let mut ordinal = 0;
        let mut filled = false;
        for message in &mut list {
            if field(message, "role")?.string()? != b"assistant" {
                continue;
            }
            let absent = message.find("reasoning_content")?.is_none_or(Json::is_null);
            if absent && let Some((_, text)) = saved.iter().find(|(at, _)| *at == ordinal) {
                let mut entry = message.object()?.clone();
                entry.retain(|(name, _)| name != b"reasoning_content");
                entry.push((b"reasoning_content".to_vec(), Json::String(text.clone())));
                *message = Json::Object(entry);
                filled = true;
            }
            ordinal += 1;
        }
        // Nothing restored renders the same prompt again, so skip the second render and encode.
        if !filled {
            return Ok(None);
        }
        *messages = Json::Array(list);
        Ok(Some(Json::Object(fields)))
    }

    fn park_active_reasoning(&mut self) {
        if !self.lane_parking || !self.reusable || self.committed.is_empty() {
            return;
        }
        let committed = self.committed.clone();
        let turns = std::mem::take(&mut self.saved_reasoning);
        let sampling = self.sampling.clone();
        let seed = self.seed;
        self.parked_reasoning.retain(|old| {
            !(old.sampling == sampling
                && old.seed == seed
                && old.committed.len() <= committed.len()
                && committed.starts_with(&old.committed))
        });
        if self.parked_reasoning.len() >= MAX_PARKED_REASONING_LANES {
            self.parked_reasoning.remove(0);
        }
        self.parked_reasoning.push(ParkedReasoning {
            committed,
            turns,
            sampling,
            seed,
        });
    }

    /// Rebuilds a canonical prompt on the committed token spelling when both spell the same text.
    ///
    /// Sampling can commit a token split that re-encoding the same bytes does not reproduce. Reuse requires matching added tokens, reserved padding slots and admission markers, with equal bytes between them. Only the canonical tail after the committed terminal is appended.
    fn respelled_on_committed(
        &self,
        committed: &[u32],
        canonical: &[u32],
    ) -> Result<Option<Vec<u32>>> {
        let admission = self.codec.admission;
        let frame = |id: u32| {
            id == admission.im_start
                || admission.is_stop(id)
                || self.codec.control_ids.contains(&id)
                || self.codec.nonordinary_ids.binary_search(&id).is_ok()
        };
        // A committed terminal ends both spellings on a token boundary, so the tail joins cleanly.
        if !committed.last().is_some_and(|&id| admission.is_stop(id)) {
            return Ok(None);
        }
        let shared = committed
            .iter()
            .zip(canonical)
            .take_while(|(live, fresh)| live == fresh)
            .count();
        let (mut live, mut fresh) = (shared, shared);
        while live < committed.len() {
            let Some(live_run) = committed[live..].iter().position(|&id| frame(id)) else {
                return Ok(None);
            };
            let Some(fresh_run) = canonical[fresh..].iter().position(|&id| frame(id)) else {
                return Ok(None);
            };
            let (live_end, fresh_end) = (live + live_run, fresh + fresh_run);
            if committed[live_end] != canonical[fresh_end] {
                return Ok(None);
            }
            let (mut live_changed, mut fresh_changed) = (live_end, fresh_end);
            while live < live_changed
                && fresh < fresh_changed
                && committed[live] == canonical[fresh]
            {
                live += 1;
                fresh += 1;
            }
            while live < live_changed
                && fresh < fresh_changed
                && committed[live_changed - 1] == canonical[fresh_changed - 1]
            {
                live_changed -= 1;
                fresh_changed -= 1;
            }
            if (live < live_changed || fresh < fresh_changed)
                && (self.codec.decode)(&committed[live..live_changed])?
                    != (self.codec.decode)(&canonical[fresh..fresh_changed])?
            {
                return Ok(None);
            }
            (live, fresh) = (live_end + 1, fresh_end + 1);
        }
        let mut prompt = committed.to_vec();
        prompt.extend_from_slice(&canonical[fresh..]);
        Ok(Some(prompt))
    }

    /// Serves one chat completion. Refusals before worker state can change keep reusable state. Later failures invalidate it.
    pub fn complete(
        &mut self,
        input: &Json,
        response_id: &[u8],
        target: ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
    ) -> Result<Json> {
        let request_start = std::time::Instant::now();
        self.last_req_telemetry = None;
        if self.phase_stamps_enabled {
            self.phase_stamps = PhaseStamps {
                request_start: Some(request_start),
                ..PhaseStamps::default()
            };
        }
        let request = admit_chat_with_context(
            input,
            &self.alias,
            self.tool_enums,
            self.tool_args,
            self.context_length,
        )?;
        crate::admission::validate_calibrated_sampling(&request.sampling, &self.mtp_ready)?;
        if !self.mtp && self.verify8_mode() {
            require(
                number(field(&request.sampling, "temperature")?, 0.0, 100.0)? == 0.0,
                "private HTTP V8 is greedy-only; nonzero temperature is explicitly refused",
            )?;
        }
        let context_cap = self.context_length as usize;
        let policy = request.sampling.dump()?;
        let compatible_sampling =
            self.sampling == policy && request.seed.is_none_or(|seed| seed == self.seed);
        type CachedReasoningPrompt<'a> = (&'a [(usize, Vec<u8>)], Option<Vec<u32>>);
        let rendered = (self.codec.render)(&request.render)?;
        let base_prompt = (self.codec.encode)(&rendered)?;
        let mut prompt = base_prompt.clone();
        let mut restored_reasoning: Option<Vec<(usize, Vec<u8>)>> = None;
        let mut restored_lane_idx: Option<usize> = None;
        let mut cached_by_turns: Vec<CachedReasoningPrompt<'_>> = Vec::new();
        if self.reusable && !prompt.starts_with(&self.committed) {
            let mut restored = None;
            if !self.saved_reasoning.is_empty() {
                if let Some(filled) =
                    Self::fill_saved_reasoning(&request.render, &self.saved_reasoning)?
                {
                    let candidate = (self.codec.encode)(&(self.codec.render)(&filled)?)?;
                    cached_by_turns.push((&self.saved_reasoning, Some(candidate.clone())));
                    if candidate.len() < context_cap && candidate.starts_with(&self.committed) {
                        prompt = candidate;
                    } else {
                        restored = Some(candidate);
                    }
                } else {
                    cached_by_turns.push((&self.saved_reasoning, None));
                }
            }
            if compatible_sampling
                && !prompt.starts_with(&self.committed)
                && let Some(respelled) = self.respelled_on_committed(
                    &self.committed,
                    restored.as_deref().unwrap_or(&prompt),
                )?
                && respelled.len() < context_cap
            {
                prompt = respelled;
            }
        }
        if self.lane_parking
            && (!self.reusable || !compatible_sampling || !prompt.starts_with(&self.committed))
            && !self.parked_reasoning.is_empty()
        {
            let mut best: Option<(usize, Vec<u32>)> = None;
            let mut ambiguous_seed = false;
            for (idx, lane) in self.parked_reasoning.iter().enumerate() {
                if let Some((cur, _)) = &best
                    && lane.committed.len() < self.parked_reasoning[*cur].committed.len()
                {
                    continue;
                }
                if lane.sampling != policy || request.seed.is_some_and(|seed| seed != lane.seed) {
                    continue;
                }
                let candidate: &[u32] = if lane.turns.is_empty() {
                    &base_prompt
                } else if let Some(pos) = cached_by_turns
                    .iter()
                    .position(|(turns, _)| *turns == lane.turns.as_slice())
                {
                    cached_by_turns[pos].1.as_deref().unwrap_or(&base_prompt)
                } else if let Some(filled) =
                    Self::fill_saved_reasoning(&request.render, &lane.turns)?
                {
                    let encoded = (self.codec.encode)(&(self.codec.render)(&filled)?)?;
                    let slot = cached_by_turns.len();
                    cached_by_turns.push((&lane.turns, Some(encoded)));
                    cached_by_turns[slot].1.as_deref().unwrap_or(&base_prompt)
                } else {
                    cached_by_turns.push((&lane.turns, None));
                    &base_prompt
                };
                let matched = if candidate.len() > lane.committed.len()
                    && candidate.len() < context_cap
                    && candidate.starts_with(&lane.committed)
                {
                    candidate.to_vec()
                } else if let Some(respelled) =
                    self.respelled_on_committed(&lane.committed, candidate)?
                    && respelled.len() > lane.committed.len()
                    && respelled.len() < context_cap
                {
                    respelled
                } else {
                    continue;
                };
                if let Some((cur, _)) = &best {
                    if lane.committed.len() > self.parked_reasoning[*cur].committed.len() {
                        best = Some((idx, matched));
                        ambiguous_seed = false;
                    } else if lane.seed != self.parked_reasoning[*cur].seed {
                        ambiguous_seed = true;
                    }
                } else {
                    best = Some((idx, matched));
                    ambiguous_seed = false;
                }
            }
            if !ambiguous_seed && let Some((idx, candidate)) = best {
                prompt = candidate;
                restored_reasoning = Some(self.parked_reasoning[idx].turns.clone());
                restored_lane_idx = Some(idx);
            }
        }
        require(
            !prompt.is_empty() && prompt.len() < context_cap,
            if self.context_length == 131_072 {
                "request leaves no generation space in the 128K context"
            } else {
                "request leaves no generation space in the 64K context"
            },
        )?;
        let maximum_tokens =
            (request.maximum_tokens as usize).min(context_cap - prompt.len()) as u32;
        for &id in &prompt {
            require(
                id < self.codec.admission.vocabulary,
                "rendered token outside vocabulary",
            )?;
        }
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(AdmissionError::Clock)?
            .as_secs();
        let append = self.reusable
            && compatible_sampling
            && prompt.len() >= self.committed.len()
            && prompt.starts_with(&self.committed);
        let cached_prefix = if append { self.committed.len() } else { 0 };
        let mut common = 0;
        if self.reusable {
            while common < self.committed.len()
                && common < prompt.len()
                && self.committed[common] == prompt[common]
            {
                common += 1;
            }
        }
        let seed = if append {
            self.seed
        } else if let Some(seed) = request.seed {
            seed
        } else {
            (sink.new_seed)()
        };
        require(
            !response_id.is_empty() && response_id.len() <= 96,
            "response ID bound",
        )?;
        let parser = ResponseParser::new_with_executor_parameters(
            request.manifest.clone(),
            request.thinking,
            ResponseLimits::default(),
            self.tool_args == ToolArgValidation::Executor,
        )
        .map_err(|error| ServeError::Text(Box::new(error)))?
        .with_token_framing();
        let parser = if self.codec.control_ids.is_empty() {
            parser
        } else {
            parser.with_control_tokens()
        };
        let mut completion = Completion {
            request_start,
            maximum_tokens,
            request: &request,
            prompt: &prompt,
            policy: &policy,
            seed,
            created,
            response_id,
            append,
            common,
            cached_prefix,
            parser,
            cursor: StreamCursor::default(),
            streamed_content: Vec::new(),
            streamed_thinking: Vec::new(),
            generated: Vec::with_capacity(maximum_tokens as usize),
            position: 0,
            state_touched: false,
            restored_reasoning,
            restored_lane_idx,
            counters: Counters::default(),
        };
        let result = completion.run(self, target, sink);
        if result.is_ok() && self.mtp && completion.counters.input_batches > 0 {
            self.last_req_telemetry = Some(RequestCheckpointTelemetry {
                ssd_restored: completion.counters.ssd_restored,
                reuse: completion.counters.checkpoint_reuse,
                restored_tokens: completion.counters.restored_input_tokens,
                replayed_tokens: completion.counters.replayed_input_tokens,
            });
        }
        if self.phase_stamps_enabled {
            self.phase_stamps.output_packets = completion.counters.output_batches;
            self.phase_stamps.max_packet_tokens = completion.counters.output_width as u64;
            self.phase_stamps.completion_tokens = completion.generated.len() as u64;
        }
        if result.is_err() && completion.state_touched {
            self.invalidate();
        }
        result
    }
}

#[derive(Default)]
struct Counters {
    input_batches: u64,
    restored_input_tokens: u64,
    replayed_input_tokens: u64,
    ssd_restored: bool,
    checkpoint_reuse: Option<&'static str>,
    output_batches: u64,
    generation_rounds: u64,
    #[allow(dead_code)]
    verified_rounds: u64,
    accepted_tokens: u64,
    input_width: usize,
    output_width: usize,
    overlap_pipelined: bool,
    #[allow(dead_code)]
    checkpoint_captures: u64,
    #[allow(dead_code)]
    checkpoint_restores: u64,
    #[allow(dead_code)]
    checkpoint_cold_resyncs: u64,
}

/// How far the parser's text and thinking records have been streamed.
///
/// While a response is incomplete the parser only appends: a new record goes on the end and only the last record's text grows (tool records stay withheld until `finish`). The unsent bytes are therefore the tail of the record at `record` plus every record after it, so a token costs its own bytes, not the whole response so far.
#[derive(Default)]
struct StreamCursor {
    record: usize,
    offset: usize,
}

/// The bytes one `StreamCursor::advance` found unsent.
#[derive(Default)]
struct StreamDelta {
    thinking: Vec<u8>,
    content: Vec<u8>,
}

impl StreamCursor {
    fn advance(&mut self, records: &[ResponseRecord]) -> Result<StreamDelta> {
        const NONMONOTONIC: &str = "nonmonotonic response parser";
        let mut delta = StreamDelta::default();
        let Some(last) = records.last() else {
            require(self.record == 0 && self.offset == 0, NONMONOTONIC)?;
            return Ok(delta);
        };
        require(self.record < records.len(), NONMONOTONIC)?;
        let mut offset = self.offset;
        for record in &records[self.record..] {
            require(offset <= record.text.len(), NONMONOTONIC)?;
            let unsent = &record.text[offset..];
            match record.kind {
                ResponseKind::Text => delta.content.extend_from_slice(unsent),
                ResponseKind::Thinking => delta.thinking.extend_from_slice(unsent),
                ResponseKind::Tool => {}
            }
            offset = 0;
        }
        self.record = records.len() - 1;
        self.offset = last.text.len();
        Ok(delta)
    }
}

/// The per-request state the C++ lambdas capture by reference.
struct Completion<'r> {
    request_start: std::time::Instant,
    maximum_tokens: u32,
    request: &'r AdmittedChat,
    prompt: &'r [u32],
    policy: &'r [u8],
    seed: u64,
    created: u64,
    response_id: &'r [u8],
    append: bool,
    common: usize,
    cached_prefix: usize,
    parser: ResponseParser,
    cursor: StreamCursor,
    streamed_content: Vec<u8>,
    streamed_thinking: Vec<u8>,
    generated: Vec<u32>,
    position: usize,
    state_touched: bool,
    restored_reasoning: Option<Vec<(usize, Vec<u8>)>>,
    restored_lane_idx: Option<usize>,
    counters: Counters,
}

fn cancelled(sink: &mut ServeSink<'_>) -> Result<()> {
    Ok(require(
        !(sink.cancelled)(),
        "request cancelled or deadline expired",
    )?)
}

impl Completion<'_> {
    fn emit(
        &self,
        session: &ServeSession,
        sink: &mut ServeSink<'_>,
        delta: Json,
        finish: Json,
    ) -> Result<()> {
        if self.request.stream {
            require(sink.chunk.is_some(), "missing streaming sink")?;
            cancelled(sink)?;
            let mut event = chunk(
                self.response_id,
                &session.alias,
                self.created,
                delta,
                finish,
            );
            if self.request.include_usage {
                let mut fields = event.object()?.clone();
                fields.push((b"usage".to_vec(), Json::Null));
                event = Json::Object(fields);
            }
            if let Some(send) = sink.chunk.as_mut() {
                send(&event)?;
            }
        }
        Ok(())
    }

    fn flush(&mut self, session: &ServeSession, sink: &mut ServeSink<'_>) -> Result<()> {
        // The finished response is still checked against everything streamed, in `run`.
        let unsent = self.cursor.advance(self.parser.records())?;
        if !unsent.thinking.is_empty() {
            let delta =
                Json::new_object([("reasoning_content", Json::from(unsent.thinking.as_slice()))]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        if !unsent.content.is_empty() {
            let delta = Json::new_object([("content", Json::from(unsent.content.as_slice()))]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        self.streamed_thinking.extend_from_slice(&unsent.thinking);
        self.streamed_content.extend_from_slice(&unsent.content);
        Ok(())
    }

    fn broker_reply(&self, session: &mut ServeSession, value: &Json, fresh: bool) -> Result<()> {
        if session.broker_epoch == 0 {
            return Ok(());
        }
        if session.branch8_mode() {
            require(
                is(field(value, "broker_protocol")?, "branch8-v1")?
                    && field(value, "broker_batch_width")?.unsigned_integer()? == 8,
                "serving branch8 wire identity changed",
            )?;
        }
        require(
            field(value, "broker_epoch")?.unsigned_integer()? == session.broker_epoch,
            "serving worker epoch changed",
        )?;
        let command = field(value, "command_id")?.unsigned_integer()?;
        let lease = field(value, "lease_id")?.unsigned_integer()?;
        require(
            command > session.broker_command
                && (session.broker_resync || command == session.broker_command.wrapping_add(1))
                && if fresh {
                    lease > session.broker_lease
                } else {
                    lease != 0 && lease == session.broker_lease
                },
            "stale serving broker command/lease",
        )?;
        broker_storage(value, &session.broker_ready)?;
        require(
            field(value, "checkpoint_retained")?.unsigned_integer()?
                <= field(&session.broker_ready, "checkpoint_slots")?.unsigned_integer()?,
            "serving broker retained count exceeds capacity",
        )?;
        session.broker_command = command;
        session.broker_lease = lease;
        session.broker_resync = false;
        Ok(())
    }

    fn sampling_matches(&self, value: &Json, prefix: usize, seed: u64) -> Result<()> {
        let sampled = field(value, "sampling")?;
        require(
            field(sampled, "valid")?.boolean()?
                && field(sampled, "entropy_prefix")?.unsigned_integer()? == prefix as u64
                && field(sampled, "seed")?.unsigned_integer()? == seed,
            "worker sampling prefix/seed mismatch",
        )?;
        for (name, value) in self.request.sampling.object()? {
            require(
                number(field(sampled, name)?, 0.0, 1e9)? == number(value, 0.0, 1e9)?,
                "worker sampling policy changed",
            )?;
        }
        Ok(())
    }

    fn control(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
        operation: &str,
        fresh: bool,
    ) -> Result<Json> {
        cancelled(sink)?;
        let done = target(&job, &mut |_: &Json| {
            Err(AdmissionError::Refused("zero-step broker control emitted a token").into())
        })?;
        cancelled(sink)?;
        require(
            is(field(&done, "event")?, "control_done")?
                && is(field(&done, "op")?, operation)?
                && field(&done, "model_steps")?.unsigned_integer()? == 0
                && field(&done, "generated_ids")?.array()?.is_empty(),
            "invalid zero-step broker completion",
        )?;
        #[cfg(feature = "branch8")]
        if session.branch8 {
            tokspur_wire::verify8::validate_branch8_control(
                &session.verify_ready,
                &session.verify_previous,
                &job,
                &done,
                session.verify_resync,
            )?;
            if operation == "branch_capture" {
                self.counters.checkpoint_captures += 1;
            } else {
                self.counters.checkpoint_restores +=
                    u64::from(is(field(&done, "outcome")?, "restored")?);
                self.counters.checkpoint_cold_resyncs += u64::from(session.verify_resync);
            }
            session.verify_previous = done.clone();
            session.verify_resync = false;
        }
        self.broker_reply(session, &done, fresh)?;
        let prefix = field(&done, "committed_prefix")?.unsigned_integer()?;
        require(
            prefix <= self.prompt.len() as u64,
            "broker restored beyond future prompt",
        )?;
        self.sampling_matches(&done, prefix as usize, self.seed)?;
        let logits = field(&done, "logits_available")?.boolean()?;
        let stop = field(&done, "stop_id")?;
        require(
            logits == (prefix != 0)
                && (stop.is_null()
                    || (logits
                        && (stop.unsigned_integer()?
                            == u64::from(session.codec.admission.im_end)
                            || stop.unsigned_integer()?
                                == u64::from(session.codec.admission.end_of_text))))
                && field(&done, "continuation_allowed")?.boolean()? == (logits && stop.is_null()),
            "invalid broker terminal metadata",
        )?;
        Ok(done)
    }

    fn step(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
        input_token: Option<u32>,
        decode: bool,
    ) -> Result<Json> {
        cancelled(sink)?;
        let mut events: Vec<u32> = Vec::new();
        let position = self.position;
        let vocabulary = session.codec.admission.vocabulary;
        let (epoch, next_command, lease) = (
            session.broker_epoch,
            session.broker_command.wrapping_add(1),
            session.broker_lease,
        );
        let done = target(&job, &mut |event: &Json| {
            require(
                decode && is(field(event, "event")?, "token")?,
                "unexpected worker token event",
            )?;
            let id = field(event, "id")?.unsigned_integer()?;
            require(
                id < u64::from(vocabulary) && events.is_empty(),
                "invalid worker token count/ID",
            )?;
            require(
                field(event, "committed_prefix")?.unsigned_integer()? == position as u64 + 1,
                "worker token prefix mismatch",
            )?;
            events.push(id as u32);
            if epoch != 0 {
                require(
                    field(event, "broker_epoch")?.unsigned_integer()? == epoch
                        && field(event, "command_id")?.unsigned_integer()? == next_command
                        && field(event, "lease_id")?.unsigned_integer()? == lease,
                    "stale serving broker token event",
                )?;
            }
            Ok(())
        })?;
        require(
            is(field(&done, "event")?, "done")?
                && field(&done, "committed_prefix")?.unsigned_integer()?
                    == self.position as u64 + 1,
            "worker completion prefix mismatch",
        )?;
        worker_execution(&done, &session.stack_execution)?;
        self.broker_reply(session, &done, field(&job, "reset")?.boolean()?)?;
        self.sampling_matches(&done, self.position + 1, self.seed)?;
        let ids = field(&done, "generated_ids")?.array()?;
        require(
            ids.len() == usize::from(decode) && events.len() == ids.len(),
            "worker generated count mismatch",
        )?;
        if decode {
            require(
                ids[0].unsigned_integer()? == u64::from(events[0]),
                "worker token/done disagreement",
            )?;
        }
        self.position += 1;
        let token = match input_token {
            Some(token) => token,
            None => events[0],
        };
        session.committed.push(token);
        Ok(done)
    }

    #[cfg(feature = "verify8")]
    fn run_many(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
    ) -> Result<Json> {
        cancelled(sink)?;
        // The gate reads readiness and the previous reply; snapshots keep it independent of the session state this command updates.
        let ready = session.verify_ready.clone();
        let previous = session.verify_previous.clone();
        let mut reply = Verify8Reply::new(Verify8Context {
            ready: &ready,
            previous: &previous,
            job: &job,
            before: self.position as u32,
            policy: &self.request.sampling,
            seed: self.seed,
            reset_resync: session.verify_resync,
            branch8: session.branch8_mode(),
        })?;
        let branch8 = session.branch8_mode();
        let (epoch, next_command, lease) = (
            session.broker_epoch,
            session.broker_command.wrapping_add(1),
            session.broker_lease,
        );
        let done = target(&job, &mut |event: &Json| {
            if branch8 {
                require(
                    field(event, "broker_epoch")?.unsigned_integer()? == epoch
                        && field(event, "command_id")?.unsigned_integer()? == next_command
                        && field(event, "lease_id")?.unsigned_integer()? == lease
                        && is(field(event, "broker_protocol")?, "branch8-v1")?
                        && field(event, "broker_batch_width")?.unsigned_integer()? == 8,
                    "stale serving branch8 batched event",
                )?;
            }
            Ok(reply.event(event)?)
        })?;
        reply.finish(&done)?;
        drop(reply);
        cancelled(sink)?; // No response parsing or history publication before full admission.
        if branch8 {
            self.broker_reply(session, &done, field(&job, "reset")?.boolean()?)?;
        }
        let tokens = field(&job, "tokens")?.array()?;
        let count = tokens.len();
        if count != 0 {
            self.counters.input_batches += 1;
            self.counters.input_width = self.counters.input_width.max(count);
            for id in tokens {
                session.committed.push(id.unsigned_integer()? as u32);
            }
        } else {
            self.counters.output_batches += 1;
            let generated = field(&done, "generated_ids")?.array()?;
            self.counters.output_width = self.counters.output_width.max(generated.len());
            session.stamp_output_packet(self.counters.output_batches, generated.len());
            self.counters.generation_rounds += field(&done, "lookup_rounds")?.unsigned_integer()?;
            self.counters.verified_rounds +=
                field(&done, "lookup_verified_rounds")?.unsigned_integer()?;
            self.counters.accepted_tokens +=
                field(&done, "lookup_accepted_tokens")?.unsigned_integer()?;
            for id in generated {
                session.committed.push(id.unsigned_integer()? as u32);
            }
        }
        self.position = field(&done, "committed_prefix")?.unsigned_integer()? as usize;
        session.verify_previous = done.clone();
        session.verify_resync = false;
        Ok(done)
    }

    fn run_mtp(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        job: Json,
    ) -> Result<Json> {
        cancelled(sink)?;
        let ready = session.mtp_ready.clone();
        let previous = session.mtp_previous.clone();
        let mut reply =
            tokspur_wire::mtp_wire::MtpWireReply::new(&ready, &previous, &job, session.mtp_resync)?;
        let done = target(&job, &mut |event: &Json| {
            reply.event(event)?;
            Ok(())
        })?;
        reply.finish(&done)?;
        cancelled(sink)?;
        worker_execution(&done, &session.stack_execution)?;
        let committed_prefix = field(&done, "committed_prefix")?.unsigned_integer()?;
        let seed = reply.lane_seed()?.unwrap_or(self.seed);
        self.sampling_matches(&done, committed_prefix as usize, seed)?;
        let inputs = field(&job, "tokens")?.array()?;
        if !inputs.is_empty() {
            let restored_in_pkt = u64::from(reply.restored_input_tokens()?);
            self.counters.restored_input_tokens += restored_in_pkt;
            self.counters.replayed_input_tokens +=
                (inputs.len() as u64).saturating_sub(restored_in_pkt);
            if let Some(ssd) = done.find("checkpoint_ssd_restored")?
                && ssd.kind() == Kind::Boolean
                && ssd.boolean()?
            {
                self.counters.ssd_restored = true;
            }
            if let Some(reuse) = done.find("checkpoint_reuse")?
                && reuse.kind() == Kind::String
            {
                let candidate = match reuse.string()? {
                    b"restored" => Some(("restored", 3u8)),
                    b"ssd_restored" => Some(("ssd_restored", 2u8)),
                    b"cold_miss" => Some(("cold_miss", 1u8)),
                    _ => None,
                };
                if let Some((label, rank)) = candidate {
                    let cur_rank = match self.counters.checkpoint_reuse {
                        Some("restored") => 3u8,
                        Some("ssd_restored") => 2u8,
                        Some("cold_miss") => 1u8,
                        _ => 0u8,
                    };
                    if rank > cur_rank {
                        self.counters.checkpoint_reuse = Some(label);
                    }
                }
            }
            self.counters.input_batches += 1;
            self.counters.input_width = self.counters.input_width.max(inputs.len());
            if let Some(ov) = done.find("prefill_generation_overlap")?
                && ov.kind() == Kind::String
                && ov.string()? == b"16_token_b10_pipelined"
            {
                self.counters.overlap_pipelined = true;
            }
            for id in inputs {
                session.committed.push(id.unsigned_integer()? as u32);
            }
        } else {
            let outputs = field(&done, "generated_ids")?.array()?;
            self.counters.output_batches += 1;
            self.counters.output_width = self.counters.output_width.max(outputs.len());
            session.stamp_output_packet(self.counters.output_batches, outputs.len());
            let done_rounds = field(&done, "native_mtp_rounds")?.unsigned_integer()?;
            let prev_rounds =
                field(&session.mtp_previous, "native_mtp_rounds")?.unsigned_integer()?;
            self.counters.generation_rounds += done_rounds.saturating_sub(prev_rounds);
            let done_acc = field(&done, "native_mtp_accepted")?.unsigned_integer()?;
            let prev_acc =
                field(&session.mtp_previous, "native_mtp_accepted")?.unsigned_integer()?;
            self.counters.accepted_tokens += done_acc.saturating_sub(prev_acc);
            for id in outputs {
                session.committed.push(id.unsigned_integer()? as u32);
            }
            if let Some(gen_meta) = done.find("native_mtp_generation")? {
                if let Some(comm) = gen_meta.find("committed")? {
                    session.mtp_committed = comm.unsigned_integer()?;
                }
                if let Some(emit) = gen_meta.find("emitted")? {
                    session.mtp_emitted = emit.unsigned_integer()?;
                }
                if let Some(ovf) = gen_meta.find("overflow")? {
                    session.mtp_pending = ovf.unsigned_integer()?;
                }
                if let Some(sc) = gen_meta.find("stage_a_controller")?
                    && sc.kind() == Kind::Object
                {
                    let mut ctrl = tokspur_wire::mtp_wire::StageAControllerState::default();
                    if let Some(ea) = sc.find("ewma_alpha1")? {
                        ctrl.ewma_alpha1 = number(ea, 0.0, 1.0)?;
                    }
                    if let Some(cm) = sc.find("current_mode")? {
                        ctrl.current_mode = cm.unsigned_integer()? as u8;
                    }
                    if let Some(co) = sc.find("consecutive_ordinary_steps")? {
                        ctrl.consecutive_ordinary_steps = co.unsigned_integer()? as u8;
                    }
                    if let Some(cpd) = sc.find("controller_policy_digest")?
                        && cpd.kind() == Kind::Array
                    {
                        let arr = cpd.array()?;
                        if arr.len() == 32 {
                            for (i, elem) in arr.iter().enumerate() {
                                ctrl.controller_policy_digest[i] = elem.unsigned_integer()? as u8;
                            }
                        }
                    }
                    if let Some(m) = sc.find("measured")? {
                        ctrl.measured =
                            Some(tokspur_wire::mtp_wire::MeasuredObservation::parse(m)?);
                    }
                    if let Some(expected) =
                        session.mtp_ready.find("native_controller_policy_digest")?
                    {
                        require(
                            sc.find("controller_policy_digest")? == Some(expected),
                            "controller policy differs from startup negotiation",
                        )?;
                    }
                    session.mtp_stage_a_controller = Some(ctrl);
                }
            }
        }
        self.seed = seed;
        self.position = committed_prefix as usize;
        session.mtp_previous = done.clone();
        session.mtp_resync = false;
        Ok(done)
    }

    /// Private replay evidence for a refused response, before session invalidation removes its tokens.
    fn response_failure(
        &self,
        sink: &mut ServeSink<'_>,
        phase: &'static str,
        finish_reason: Option<&'static str>,
        fallback_reason: &'static str,
        primary: ServeError,
    ) -> ServeError {
        if let Some(trace) = sink.prefix_trace.as_mut() {
            let reason = if self.parser.error().is_empty() {
                fallback_reason
            } else {
                self.parser.error()
            };
            // Parser errors contain fixed descriptions, never generated bytes or tool values. IDs and the admitted parser manifest remain confined to the already opt-in, bounded private trace.
            let event = Json::new_object([
                ("event", Json::from("response_validation_failed")),
                ("schema", Json::integer(1)),
                ("response_id", Json::from(self.response_id)),
                ("phase", Json::from(phase)),
                (
                    "finish_reason",
                    finish_reason.map_or(Json::Null, Json::from),
                ),
                ("reason", Json::from(reason)),
                (
                    "generated_count",
                    Json::integer(self.generated.len() as u64),
                ),
                (
                    "generated_scope",
                    Json::from("generated-ids-consumed-by-session"),
                ),
                (
                    "generated_ids",
                    Json::Array(
                        self.generated
                            .iter()
                            .map(|&id| Json::integer(u64::from(id)))
                            .collect(),
                    ),
                ),
                (
                    "parser_bytes_received",
                    Json::integer(self.parser.bytes_received() as u64),
                ),
                ("initial_thinking", Json::Bool(self.request.thinking)),
                ("parser_manifest", parser_manifest(&self.request.manifest)),
            ]);
            if trace(&event).is_err() {
                // The primary validation error takes precedence over a diagnostic-write failure, preserving the public HTTP error bytes.
                return primary;
            }
        }
        primary
    }

    fn encode_forced_tokens(session: &ServeSession, text: &[u8]) -> Result<Vec<u32>> {
        let rendered = RenderedChat {
            prompt: text.to_vec(),
            ..RenderedChat::default()
        };
        let ids = (session.codec.encode)(&rendered)?;
        require(!ids.is_empty(), "forced thinking close encoded no tokens")?;
        let admission = session.codec.admission;
        for &id in &ids {
            require(
                id < admission.vocabulary && id != admission.im_start && !admission.is_stop(id),
                "forced thinking close token outside vocabulary or terminal",
            )?;
        }
        Ok(ids)
    }

    fn build_mtp_generation_field(
        session: &ServeSession,
        begin: bool,
        remaining: u64,
        total_budget: u64,
    ) -> Result<Json> {
        let is_v2 = if let Some(scope) = session.mtp_ready.find("native_mtp_carry_scope")? {
            scope.string()? == b"http-generation-v2"
        } else {
            false
        };
        if is_v2 {
            let committed = session.mtp_committed;
            let emitted = session.mtp_emitted;
            let pending = session.mtp_pending;
            let remaining_compute = total_budget.saturating_sub(committed);
            let mut fields: Vec<(&str, Json)> = vec![
                ("id", Json::integer(session.mtp_generation_id)),
                ("begin", Json::Bool(begin)),
                ("remaining", Json::integer(remaining)),
                ("committed", Json::integer(committed)),
                ("emitted", Json::integer(emitted)),
                ("pending", Json::integer(pending)),
                ("remaining_compute", Json::integer(remaining_compute)),
                ("total_budget", Json::integer(total_budget)),
            ];
            if let Some(ref ctrl) = session.mtp_stage_a_controller {
                let digest_arr: Vec<Json> = ctrl
                    .controller_policy_digest
                    .iter()
                    .map(|&b| Json::integer(b as u64))
                    .collect();
                let ctrl_obj = Json::new_object([
                    ("ewma_alpha1", real(ctrl.ewma_alpha1)?),
                    ("current_mode", Json::integer(ctrl.current_mode as u64)),
                    (
                        "consecutive_ordinary_steps",
                        Json::integer(ctrl.consecutive_ordinary_steps as u64),
                    ),
                    ("controller_policy_digest", Json::Array(digest_arr)),
                ]);
                let ctrl_obj = if let Some(measured) = ctrl.measured {
                    let mut object = ctrl_obj.object()?.to_vec();
                    object.push((b"measured".to_vec(), measured.json()?));
                    Json::Object(object)
                } else {
                    ctrl_obj
                };
                fields.push(("stage_a_controller", ctrl_obj));
            }
            Ok(Json::new_object(fields))
        } else {
            Ok(Json::new_object([
                ("id", Json::integer(session.mtp_generation_id)),
                ("begin", Json::Bool(begin)),
                ("remaining", Json::integer(remaining)),
            ]))
        }
    }

    fn run_generation_phase(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
        phase_limit: usize,
        stopped: &mut bool,
    ) -> Result<()> {
        if *stopped || self.generated.len() >= phase_limit {
            return Ok(());
        }
        let phase_budget = phase_limit - self.generated.len();
        let admission = session.codec.admission;
        let stop_ids = || {
            Json::Array(vec![
                Json::integer(u64::from(admission.im_end)),
                Json::integer(u64::from(admission.end_of_text)),
            ])
        };
        if session.mtp {
            session.mtp_committed = 0;
            session.mtp_emitted = 0;
            session.mtp_pending = 0;
            session.mtp_stage_a_controller = None;
            let scoped_carry = tokspur_wire::mtp_wire::mtp_request_carry(&session.mtp_ready)?;
            if scoped_carry {
                require(
                    session.mtp_generation_id < u64::MAX,
                    "MTP generation ID exhausted",
                )?;
                session.mtp_generation_id += 1;
            }
            let output_packet =
                tokspur_wire::mtp_wire::mtp_output_packet(&session.mtp_ready)? as usize;
            let mut phase_emitted = 0usize;
            while phase_emitted < phase_budget {
                let packet_cap = match session.mtp_first_output_packet {
                    Some(first_pkt) if self.generated.is_empty() => {
                        (first_pkt as usize).min(output_packet)
                    }
                    _ => output_packet,
                };
                let n = packet_cap.min(phase_budget - phase_emitted);
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(Vec::new())),
                    (b"generate".to_vec(), Json::integer(n as u64)),
                    (b"reset".to_vec(), Json::Bool(false)),
                    (b"stop_ids".to_vec(), stop_ids()),
                ];
                if scoped_carry {
                    let gen_obj = Self::build_mtp_generation_field(
                        session,
                        phase_emitted == 0,
                        (phase_budget - phase_emitted) as u64,
                        phase_budget as u64,
                    )?;
                    job.push((b"mtp_generation".to_vec(), gen_obj));
                }
                let done = self.run_mtp(session, target, sink, Json::Object(job))?;
                for token in field(&done, "generated_ids")?.array()? {
                    let id = token.unsigned_integer()? as u32;
                    require(
                        id != admission.im_start && !*stopped,
                        "MTP unexpected generated role/terminal",
                    )?;
                    *stopped = admission.is_stop(id);
                    self.generated.push(id);
                    phase_emitted += 1;
                    self.feed(session, sink, id)?;
                }
                if *stopped {
                    break;
                }
            }
        } else if session.verify8_mode() {
            #[cfg(feature = "verify8")]
            while self.generated.len() < phase_limit {
                let n = 8.min(phase_limit - self.generated.len());
                let job = Json::new_object([
                    ("tokens", Json::Array(Vec::new())),
                    ("generate", Json::integer(n as u64)),
                    ("reset", Json::Bool(false)),
                    ("stop_ids", stop_ids()),
                ]);
                let done = self.run_many(session, target, sink, job)?;
                for token in field(&done, "generated_ids")?.array()? {
                    let id = token.unsigned_integer()? as u32;
                    require(
                        id != admission.im_start && !*stopped,
                        "private V8 unexpected generated role/terminal",
                    )?;
                    *stopped = admission.is_stop(id);
                    self.generated.push(id);
                    self.feed(session, sink, id)?;
                }
                if *stopped {
                    break;
                }
            }
        } else {
            while self.generated.len() < phase_limit {
                let job = Json::new_object([
                    ("tokens", Json::Array(Vec::new())),
                    ("generate", Json::integer(1)),
                    ("reset", Json::Bool(false)),
                    ("stop_ids", stop_ids()),
                ]);
                let done = self.step(session, target, sink, job, None, true)?;
                let id = session.committed.last().copied().unwrap_or(0);
                require(
                    id != admission.im_start,
                    "unexpected generated role/terminal token",
                )?;
                *stopped = admission.is_stop(id);
                require(
                    is(
                        field(&done, "finish_reason")?,
                        if *stopped { "stop" } else { "length" },
                    )? && is(
                        field(&done, "status")?,
                        if *stopped { "stopped" } else { "length" },
                    )?,
                    "worker termination mismatch",
                )?;
                if *stopped {
                    require(
                        field(&done, "stop_id")?.unsigned_integer()? == u64::from(id)
                            && !field(&done, "continuation_allowed")?.boolean()?,
                        "invalid terminal admission",
                    )?;
                } else {
                    require(
                        field(&done, "stop_id")?.is_null()
                            && field(&done, "continuation_allowed")?.boolean()?,
                        "invalid continuation admission",
                    )?;
                }
                self.generated.push(id);
                self.feed(session, sink, id)?;
                if *stopped {
                    break;
                }
            }
        }
        Ok(())
    }

    fn inject_forced_thinking_close(
        &mut self,
        session: &mut ServeSession,
        target: &mut ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
    ) -> Result<Option<usize>> {
        let nl_ids = if self.parser.thinking_ends_with_newline() {
            Vec::new()
        } else {
            Self::encode_forced_tokens(session, b"\n")?
        };
        let close_ids = Self::encode_forced_tokens(session, b"</think>")?;
        let nl2_ids = Self::encode_forced_tokens(session, b"\n\n")?;
        let mut inject = Vec::with_capacity(nl_ids.len() + close_ids.len() + nl2_ids.len());
        inject.extend_from_slice(&nl_ids);
        let close_offset = inject.len();
        inject.extend_from_slice(&close_ids);
        inject.extend_from_slice(&nl2_ids);
        let rem = self.maximum_tokens as usize - self.generated.len();
        if inject.len() > rem {
            inject.truncate(rem);
        }
        if inject.is_empty() {
            return Ok(None);
        }
        let forced_close_index = if inject.len() > close_offset {
            Some(self.generated.len() + close_offset)
        } else {
            None
        };
        if session.mtp {
            let input_packet =
                tokspur_wire::mtp_wire::mtp_input_packet(&session.mtp_ready)? as usize;
            let final_logits = tokspur_wire::mtp_wire::mtp_final_input_logits(&session.mtp_ready)?;
            let mut at = 0usize;
            while at < inject.len() {
                cancelled(sink)?;
                let n = input_packet.min(inject.len() - at);
                let ids = inject[at..at + n]
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect();
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(ids)),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(false)),
                ];
                if final_logits {
                    job.push((
                        b"prefill_final".to_vec(),
                        Json::Bool(at + n == inject.len()),
                    ));
                }
                self.run_mtp(session, target, sink, Json::Object(job))?;
                at += n;
            }
        } else if session.verify8_mode() {
            #[cfg(feature = "verify8")]
            {
                let mut at = 0usize;
                while at < inject.len() {
                    let n = 8.min(inject.len() - at);
                    let ids = inject[at..at + n]
                        .iter()
                        .map(|&id| Json::integer(u64::from(id)))
                        .collect();
                    let job = Json::new_object([
                        ("tokens", Json::Array(ids)),
                        ("generate", Json::integer(0)),
                        ("reset", Json::Bool(false)),
                    ]);
                    self.run_many(session, target, sink, job)?;
                    at += n;
                }
            }
        } else {
            for &token in &inject {
                let job = Json::new_object([
                    ("tokens", Json::Array(vec![Json::integer(u64::from(token))])),
                    ("generate", Json::integer(0)),
                    ("reset", Json::Bool(false)),
                ]);
                let done = self.step(session, target, sink, job, Some(token), false)?;
                require(
                    is(field(&done, "status")?, "prefilled")?
                        && field(&done, "finish_reason")?.is_null()
                        && field(&done, "stop_id")?.is_null(),
                    "invalid prefill termination",
                )?;
            }
        }
        for (i, &id) in inject.iter().enumerate() {
            if i == close_offset {
                self.parser.prepare_forced_thinking_close();
            }
            self.generated.push(id);
            self.feed(session, sink, id)?;
        }
        Ok(forced_close_index)
    }

    /// Feeds one generated ID's bytes to the parser and streams new deltas.
    fn feed(&mut self, session: &ServeSession, sink: &mut ServeSink<'_>, id: u32) -> Result<()> {
        if session.codec.admission.is_stop(id) {
            return Ok(());
        }
        let bytes = (session.codec.decode)(&[id])?;
        let status = if session.codec.control_ids.contains(&id) {
            self.parser.feed_control(&bytes)
        } else {
            self.parser.feed(&bytes)
        };
        if status == ResponseStatus::Error {
            return Err(self.response_failure(
                sink,
                "feed",
                None,
                "response parser rejected generated bytes",
                AdmissionError::Refused("response bytes/frame validation failed").into(),
            ));
        }
        if self.request.stream {
            self.flush(session, sink)?;
        }
        Ok(())
    }

    /// The C++ `try` block of `complete`.
    fn run(
        &mut self,
        session: &mut ServeSession,
        mut target: ServeTarget<'_>,
        sink: &mut ServeSink<'_>,
    ) -> Result<Json> {
        let target = &mut target;
        let request = self.request;
        let prompt = self.prompt;
        cancelled(sink)?;
        if let Some(trace) = sink.prefix_trace.as_mut() {
            let ids = Json::Array(
                prompt
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect(),
            );
            let prior = Json::Array(
                session
                    .committed
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect(),
            );
            let event = Json::new_object([
                ("event", Json::from("prompt_prefix")),
                ("schema", Json::integer(1)),
                ("response_id", Json::from(self.response_id)),
                ("prompt_ids", ids),
                ("prior_committed_ids", prior),
                (
                    "prior_committed_tokens",
                    Json::integer(session.committed.len() as u64),
                ),
                ("common_prefix", Json::integer(self.common as u64)),
                ("reusable", Json::Bool(session.reusable)),
                ("append", Json::Bool(self.append)),
                (
                    "sampling_matches",
                    Json::Bool(session.sampling == self.policy),
                ),
                (
                    "seed_matches",
                    Json::Bool(self.request.seed.is_none_or(|s| s == session.seed)),
                ),
                ("prompt_tokens", Json::integer(prompt.len() as u64)),
                ("cached_tokens", Json::integer(self.cached_prefix as u64)),
                (
                    "new_tokens",
                    Json::integer((prompt.len() - self.cached_prefix) as u64),
                ),
            ]);
            trace(&event)?;
        }
        self.emit(
            session,
            sink,
            Json::new_object([("role", Json::from("assistant"))]),
            Json::Null,
        )?;
        self.state_touched = true;
        let prefill_start = std::time::Instant::now();
        if session.phase_stamps_enabled {
            session.phase_stamps.prefill_start = Some(prefill_start);
        }
        self.position = if self.append {
            session.committed.len()
        } else {
            0
        };
        let mut began = false;
        let mut cached_prefix = self.cached_prefix;
        if session.broker_epoch != 0
            && !self.append
            && (session.branch8_mode()
                || session.mtp
                || number(field(&request.sampling, "temperature")?, 0.0, 100.0)? == 0.0)
        {
            let future = Json::Array(
                prompt
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect(),
            );
            let reuse = session.reusable;
            let job = Json::new_object([
                ("op", Json::from("branch_begin")),
                ("prompt_ids", future),
                ("generation_requested", Json::Bool(true)),
                ("allow_restore", Json::Bool(reuse)),
                ("sampling", request.sampling.clone()),
                ("seed", Json::integer(self.seed)),
            ]);
            let done = self.control(session, target, sink, job, "branch_begin", true)?;
            let prefix = field(&done, "committed_prefix")?.unsigned_integer()?;
            let lcp = field(&done, "common_prefix")?.unsigned_integer()?;
            let outcome = field(&done, "outcome")?.string()?;
            require(
                lcp == if reuse { self.common as u64 } else { 0 }
                    && prefix <= lcp
                    && ((outcome == b"cold_miss" && prefix == 0)
                        || (outcome == b"restored" && prefix != 0 && reuse)),
                "broker exact-LCP transaction mismatch",
            )?;
            require(
                prefix < prompt.len() as u64 || field(&done, "stop_id")?.is_null(),
                "terminal restored prompt cannot generate without new input",
            )?;
            // Reserved at preparation; shrinking cannot allocate after restore.
            session.committed.resize(prefix as usize, 0);
            self.position = prefix as usize;
            cached_prefix = self.position;
            began = true;
        } else if !self.append {
            if let Some(idx) = self.restored_lane_idx.take() {
                session.parked_reasoning.remove(idx);
            }
            session.park_active_reasoning();
            session.committed.clear();
        }
        if !self.append {
            session.reusable = false;
        }
        let input_start = self.position;
        if session.mtp {
            while self.position < prompt.len() {
                cancelled(sink)?;
                let input_packet =
                    tokspur_wire::mtp_wire::mtp_input_packet(&session.mtp_ready)? as usize;
                let mut n = input_packet.min(prompt.len() - self.position);
                if session.overlap_prefill && prompt.len() - self.position > 16 {
                    let bulk_remaining = (prompt.len() - 16) - self.position;
                    n = n.min(bulk_remaining);
                }
                let ids = prompt[self.position..self.position + n]
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect();
                let fresh = !self.append && !began && self.position == 0;
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(ids)),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(fresh)),
                ];
                if fresh {
                    job.push((b"sampling".to_vec(), request.sampling.clone()));
                    job.push((b"seed".to_vec(), Json::integer(self.seed)));
                }
                if fresh && session.lane_parking && request.seed.is_none() {
                    job.push((b"lane_inherit_seed".to_vec(), Json::Bool(true)));
                }
                let lane_prompt = fresh && session.lane_parking && n < prompt.len();
                if lane_prompt {
                    let all = prompt
                        .iter()
                        .map(|&id| Json::integer(u64::from(id)))
                        .collect();
                    job.push((b"lane_prompt".to_vec(), Json::Array(all)));
                }
                if tokspur_wire::mtp_wire::mtp_final_input_logits(&session.mtp_ready)? {
                    job.push((
                        b"prefill_final".to_vec(),
                        Json::Bool(self.position + n == prompt.len()),
                    ));
                }
                let packet_end = self.position + n;
                self.run_mtp(session, target, sink, Json::Object(job))?;
                if lane_prompt && self.position > packet_end {
                    // The worker restored a parked conversation past this packet; its tokens are this prompt's prefix.
                    require(
                        self.position < prompt.len() && session.committed.len() == packet_end,
                        "parked-lane restore exceeds the prompt",
                    )?;
                    session
                        .committed
                        .extend_from_slice(&prompt[packet_end..self.position]);
                    self.counters.restored_input_tokens += (self.position - packet_end) as u64;
                }
            }
        } else if session.verify8_mode() {
            #[cfg(feature = "verify8")]
            while self.position < prompt.len() {
                let mut n = 8.min(prompt.len() - self.position);
                if session.branch8_mode() {
                    n = n.min(512 - self.position % 512); // Never step over a useful ancestor boundary.
                }
                let ids = prompt[self.position..self.position + n]
                    .iter()
                    .map(|&id| Json::integer(u64::from(id)))
                    .collect();
                let fresh = !self.append && !began && self.position == 0;
                let mut job = vec![
                    (b"tokens".to_vec(), Json::Array(ids)),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(fresh)),
                ];
                if fresh {
                    job.push((b"sampling".to_vec(), request.sampling.clone()));
                    job.push((b"seed".to_vec(), Json::integer(self.seed)));
                    job.push((b"prompt_lookup".to_vec(), Json::Bool(true)));
                }
                self.run_many(session, target, sink, Json::Object(job))?;
                if session.branch8_mode()
                    && (self.position.is_multiple_of(512) || self.position == prompt.len())
                {
                    let capture = Json::new_object([("op", Json::from("branch_capture"))]);
                    let captured =
                        self.control(session, target, sink, capture, "branch_capture", false)?;
                    require(
                        is(field(&captured, "outcome")?, "captured")?
                            && field(&captured, "committed_prefix")?.unsigned_integer()?
                                == self.position as u64
                            && field(&captured, "common_prefix")?.unsigned_integer()?
                                == self.position as u64,
                        "branch8 scheduled capture mismatch",
                    )?;
                }
            }
        } else {
            for (i, &token) in prompt.iter().enumerate().skip(self.position) {
                let fresh = !self.append && !began && i == 0;
                let mut job = vec![
                    (
                        b"tokens".to_vec(),
                        Json::Array(vec![Json::integer(u64::from(token))]),
                    ),
                    (b"generate".to_vec(), Json::integer(0)),
                    (b"reset".to_vec(), Json::Bool(fresh)),
                ];
                if fresh {
                    job.push((b"sampling".to_vec(), request.sampling.clone()));
                    job.push((b"seed".to_vec(), Json::integer(self.seed)));
                }
                let done =
                    self.step(session, target, sink, Json::Object(job), Some(token), false)?;
                require(
                    is(field(&done, "status")?, "prefilled")?
                        && field(&done, "finish_reason")?.is_null()
                        && field(&done, "stop_id")?.is_null(),
                    "invalid prefill termination",
                )?;
            }
        }
        if session.broker_epoch != 0 && self.position > input_start && !session.branch8_mode() {
            let capture = Json::new_object([("op", Json::from("branch_capture"))]);
            let done = self.control(session, target, sink, capture, "branch_capture", false)?;
            require(
                is(field(&done, "outcome")?, "captured")?
                    && field(&done, "committed_prefix")?.unsigned_integer()?
                        == self.position as u64
                    && field(&done, "common_prefix")?.unsigned_integer()? == self.position as u64,
                "broker input-boundary capture mismatch",
            )?;
        }
        // Active/broker reuse precedes the submitted input packets. Each admitted packet contributes only its own restored suffix, once per request.
        require(
            self.counters.restored_input_tokens <= (prompt.len() - cached_prefix) as u64,
            "restored input accounting exceeds uncached prompt",
        )?;
        cached_prefix += self.counters.restored_input_tokens as usize;
        let admission = session.codec.admission;
        let generation_start = std::time::Instant::now();
        if session.phase_stamps_enabled {
            session.phase_stamps.generation_start = Some(generation_start);
        }
        let mut stopped = false;
        let mut forced_thinking_close_index = None;
        let max_total = self.maximum_tokens as usize;
        if let Some(budget) = request.thinking_budget
            && request.thinking
            && (budget as usize) < max_total
        {
            self.run_generation_phase(session, target, sink, budget as usize, &mut stopped)?;
            while !stopped
                && self.parser.in_thinking()
                && self.parser.has_incomplete_utf8()
                && self.generated.len() < max_total
            {
                let next_limit = self.generated.len() + 1;
                self.run_generation_phase(session, target, sink, next_limit, &mut stopped)?;
            }
            if !stopped && self.parser.in_thinking() && self.generated.len() < max_total {
                forced_thinking_close_index =
                    self.inject_forced_thinking_close(session, target, sink)?;
            }
            if !stopped && self.generated.len() < max_total {
                self.run_generation_phase(session, target, sink, max_total, &mut stopped)?;
            }
        } else {
            self.run_generation_phase(session, target, sink, max_total, &mut stopped)?;
        }
        // Nonstreaming still validates each token above. Assemble its parser
        // records once for the same final prefix check used by streaming.
        if !request.stream {
            self.flush(session, sink)?;
        }
        let ids = Json::Array(
            self.generated
                .iter()
                .map(|&id| Json::integer(u64::from(id)))
                .collect(),
        );
        let mut final_fields = vec![
            (b"ids".to_vec(), ids),
            (b"tools".to_vec(), parser_manifest(&request.manifest)),
            (b"initial_thinking".to_vec(), Json::Bool(request.thinking)),
            (
                b"finish_reason".to_vec(),
                Json::from(if stopped { "stop" } else { "length" }),
            ),
            (
                b"stop_id".to_vec(),
                if stopped {
                    Json::integer(u64::from(*self.generated.last().ok_or(
                        AdmissionError::Refused("stopped response lacks terminal token"),
                    )?))
                } else {
                    Json::Null
                },
            ),
        ];
        if session.tool_args == ToolArgValidation::Executor {
            final_fields.push((b"executor_parameters".to_vec(), Json::Bool(true)));
        }
        if let Some(idx) = forced_thinking_close_index {
            final_fields.push((
                b"forced_thinking_close_index".to_vec(),
                Json::integer(idx as u64),
            ));
        }
        let final_request = Json::Object(final_fields);
        let decode = |ids: &[u32]| (session.codec.decode)(ids);
        let finished = parse_native_response_with_controls(
            &final_request,
            admission,
            Some(&decode),
            RESPONSE_MAXIMUM_ACCUMULATED_IDS,
            &session.codec.control_ids,
        )
        .map_err(|error| {
            if sink.prefix_trace.is_some() && stopped {
                // Only after the authoritative accumulated parser has refused: finalise the existing incremental parser to recover its detailed reason without changing admission or success paths.
                self.parser.finish();
            }
            let reason = match &error {
                ResponseProtocolError::Admit(reason) => *reason,
                ResponseProtocolError::Access(_) => "final response field validation failed",
                ResponseProtocolError::Configure(_) => "final response parser configuration failed",
                ResponseProtocolError::Decode(_) => "final response decoding failed",
            };
            self.response_failure(
                sink,
                "finish",
                Some(if stopped { "stop" } else { "length" }),
                reason,
                ServeError::Text(Box::new(error)),
            )
        })?;
        require(
            !request.single_tool_call || field(&finished, "tool_calls")?.array()?.len() <= 1,
            "parallel tool calls are disabled for this request",
        )?;
        let mut calls: Vec<Json> = Vec::new();
        for call in field(&finished, "tool_calls")?.array()? {
            let function = field(call, "function")?;
            let name = field(function, "name")?.string()?;
            let Some(index) = request
                .names
                .iter()
                .position(|candidate| candidate.as_slice() == name)
            else {
                return Err(AdmissionError::Refused("unadmitted final tool").into());
            };
            if session.tool_args == ToolArgValidation::Executor {
                validate_executor_tool_arguments(field(function, "arguments")?)?;
            } else {
                validate_tool_value_with_policy(
                    &request.schemas[index],
                    field(function, "arguments")?,
                    session.tool_enums,
                )?;
            }
            let mut id = self.response_id.to_vec();
            id.extend_from_slice(format!("_call_{}", calls.len()).as_bytes());
            calls.push(Json::new_object([
                ("id", Json::String(id)),
                ("type", Json::from("function")),
                (
                    "function",
                    Json::new_object([
                        ("name", Json::from(name)),
                        (
                            "arguments",
                            Json::String(field(function, "arguments")?.dump()?),
                        ),
                    ]),
                ),
            ]));
        }
        let choice = request.tool_choice.as_slice();
        require(
            (!request.named_tool_choice && (choice == b"auto" || choice == b"none"))
                || !stopped
                || !calls.is_empty(),
            "required tool was not produced",
        )?;
        cancelled(sink)?;
        let content = field(&finished, "content")?.string()?.to_vec();
        let thinking = field(&finished, "reasoning_content")?.string()?.to_vec();
        require(
            content.starts_with(&self.streamed_content)
                && thinking.starts_with(&self.streamed_thinking),
            "final response disagrees with streamed prefix",
        )?;
        if thinking.len() > self.streamed_thinking.len() {
            let delta = Json::new_object([(
                "reasoning_content",
                Json::from(&thinking[self.streamed_thinking.len()..]),
            )]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        if content.len() > self.streamed_content.len() {
            let delta = Json::new_object([(
                "content",
                Json::from(&content[self.streamed_content.len()..]),
            )]);
            self.emit(session, sink, delta, Json::Null)?;
        }
        for (i, call) in calls.iter().enumerate() {
            let mut fields = call.object()?.clone();
            fields.push((b"index".to_vec(), Json::integer(i as u64)));
            self.emit(
                session,
                sink,
                Json::new_object([("tool_calls", Json::Array(vec![Json::Object(fields)]))]),
                Json::Null,
            )?;
        }
        let finish = if !calls.is_empty() {
            "tool_calls"
        } else if stopped {
            "stop"
        } else {
            "length"
        };
        self.emit(session, sink, Json::Object(Vec::new()), Json::from(finish))?;
        session.sampling = self.policy.to_vec();
        session.seed = self.seed;
        session.reusable = true;
        if !self.append {
            session.saved_reasoning = self.restored_reasoning.take().unwrap_or_default();
        }
        if !thinking.is_empty() {
            let mut ordinal = 0;
            for message in field(&request.render, "messages")?.array()? {
                ordinal += usize::from(field(message, "role")?.string()? == b"assistant");
            }
            session.saved_reasoning.push((ordinal, thinking.clone()));
        }
        let generation_end = std::time::Instant::now();
        if session.phase_stamps_enabled {
            session.phase_stamps.generation_end = Some(generation_end);
        }
        let admission_ns = prefill_start
            .saturating_duration_since(self.request_start)
            .as_nanos() as u64;
        let prefill_ns = generation_start
            .saturating_duration_since(prefill_start)
            .as_nanos() as u64;
        let generation_ns = generation_end
            .saturating_duration_since(generation_start)
            .as_nanos() as u64;
        let server_admission_wall_ms = admission_ns as f64 / 1e6;
        let server_prefill_wall_ms = prefill_ns as f64 / 1e6;
        let server_generation_wall_ms = generation_ns as f64 / 1e6;
        let server_request_wall_ms =
            server_admission_wall_ms + server_prefill_wall_ms + server_generation_wall_ms;
        let server_generation_throughput =
            if !self.generated.is_empty() && server_generation_wall_ms > 0.0 {
                (self.generated.len() as f64 * 1000.0) / server_generation_wall_ms
            } else {
                0.0
            };
        let generated = self.generated.len() as u64;
        let prompt_tokens = prompt.len() as u64;
        let prefill_generation_overlap = if self.counters.overlap_pipelined {
            "16_token_b10_pipelined"
        } else {
            "none"
        };
        let mut message = vec![
            (b"role".to_vec(), Json::from("assistant")),
            (b"content".to_vec(), Json::String(content)),
        ];
        if !thinking.is_empty() {
            message.push((b"reasoning_content".to_vec(), Json::String(thinking)));
        }
        if !calls.is_empty() {
            message.push((b"tool_calls".to_vec(), Json::Array(calls)));
        }
        let usage = Json::new_object([
            ("prompt_tokens", Json::integer(prompt_tokens)),
            ("completion_tokens", Json::integer(generated)),
            ("total_tokens", Json::integer(prompt_tokens + generated)),
            (
                "prompt_tokens_details",
                Json::new_object([("cached_tokens", Json::integer(cached_prefix as u64))]),
            ),
        ]);
        let result = vec![
            (b"id".to_vec(), Json::from(self.response_id)),
            (b"object".to_vec(), Json::from("chat.completion")),
            (b"created".to_vec(), Json::integer(self.created)),
            (b"model".to_vec(), Json::from(session.alias.clone())),
            (
                b"choices".to_vec(),
                Json::Array(vec![Json::new_object([
                    ("index", Json::integer(0)),
                    ("message", Json::Object(message)),
                    ("finish_reason", Json::from(finish)),
                ])]),
            ),
            (b"usage".to_vec(), usage.clone()),
        ];
        if let Some(trace) = sink.prefix_trace.as_mut() {
            let fields: Vec<(&str, Json)> = vec![
                ("event", Json::from("completion")),
                ("schema", Json::integer(1)),
                ("response_id", Json::from(self.response_id)),
                ("prompt_tokens", Json::integer(prompt_tokens)),
                ("cached_tokens", Json::integer(cached_prefix as u64)),
                (
                    "new_tokens",
                    Json::integer(prompt_tokens - cached_prefix as u64),
                ),
                ("completion_tokens", Json::integer(generated)),
                ("total_tokens", Json::integer(prompt_tokens + generated)),
                ("finish_reason", Json::from(finish)),
                ("stopped_on_eos", Json::Bool(stopped)),
                ("server_admission_wall_ms", real(server_admission_wall_ms)?),
                ("server_prefill_wall_ms", real(server_prefill_wall_ms)?),
                (
                    "server_generation_wall_ms",
                    real(server_generation_wall_ms)?,
                ),
                ("server_request_wall_ms", real(server_request_wall_ms)?),
                (
                    "server_generation_throughput",
                    real(server_generation_throughput)?,
                ),
                (
                    "prefill_generation_overlap",
                    Json::from(prefill_generation_overlap),
                ),
            ];
            let mut byte_fields: Vec<(Vec<u8>, Json)> = fields
                .into_iter()
                .map(|(k, v)| (k.as_bytes().to_vec(), v))
                .collect();
            if session.mtp_previous.kind() == Kind::Object {
                for trace_key in [
                    "peak_metal_buffer_bytes",
                    "memory",
                    "ple_cache",
                    "stateless_prefill_ple_rows",
                    "checkpoint_ssd_enabled",
                    "checkpoint_ssd_occupied_bytes",
                    "checkpoint_ssd_spills",
                    "checkpoint_ssd_restores",
                    "checkpoint_ssd_restored_tokens",
                ] {
                    if let Some(v) = session.mtp_previous.find(trace_key)? {
                        byte_fields.push((trace_key.as_bytes().to_vec(), v.clone()));
                    }
                }
            }
            let trace_event = Json::Object(byte_fields);
            trace(&trace_event)?;
        }
        if self.request.stream && self.request.include_usage {
            cancelled(sink)?;
            let final_chunk = Json::new_object([
                ("id", Json::from(self.response_id)),
                ("object", Json::from("chat.completion.chunk")),
                ("created", Json::integer(self.created)),
                ("model", Json::from(session.alias.clone())),
                ("choices", Json::Array(Vec::new())),
                ("usage", usage),
            ]);
            if let Some(send) = sink.chunk.as_mut() {
                send(&final_chunk)?;
            }
        }
        let result = self.private_details(session, result)?;
        Ok(Json::Object(result))
    }

    /// `tokspur_private`: scheduling counters, never a performance claim.
    fn private_details(
        &self,
        session: &ServeSession,
        mut result: Vec<(Vec<u8>, Json)>,
    ) -> Result<Vec<(Vec<u8>, Json)>> {
        if session.mtp {
            let c = &self.counters;
            let mut details = vec![
                (b"native_mtp".to_vec(), Json::Bool(true)),
                (
                    b"native_mtp_protocol".to_vec(),
                    field(&session.mtp_ready, "native_mtp_protocol")?.clone(),
                ),
                (
                    b"known_input_batches".to_vec(),
                    Json::integer(c.input_batches),
                ),
                (
                    b"maximum_known_input_width".to_vec(),
                    Json::integer(c.input_width as u64),
                ),
                (
                    b"generation_batches".to_vec(),
                    Json::integer(c.output_batches),
                ),
                (
                    b"maximum_generation_width".to_vec(),
                    Json::integer(c.output_width as u64),
                ),
                (b"mtp_rounds".to_vec(), Json::integer(c.generation_rounds)),
                (
                    b"mtp_accepted_tokens".to_vec(),
                    Json::integer(c.accepted_tokens),
                ),
                (b"performance_qualified".to_vec(), Json::Bool(false)),
            ];
            if let Ok(512 | 1024 | 2048 | 4096 | 8192) =
                tokspur_wire::mtp_wire::mtp_input_packet(&session.mtp_ready)
            {
                for name in &[
                    "native_mtp_input_packet_schema",
                    "native_mtp_input_packet_max",
                    "native_mtp_priming_batch_capacity",
                    "native_mtp_priming_batch_bytes",
                    "native_prefill_dense",
                ] {
                    if let Ok(v) = field(&session.mtp_ready, name) {
                        details.push((name.as_bytes().to_vec(), v.clone()));
                    }
                }
                for name in &[
                    "native_prefill_grouped",
                    "native_prefill_grouped_kernel",
                    "native_prefill_dense_kernel",
                ] {
                    if let Ok(v) = field(&session.mtp_ready, name) {
                        details.push((name.as_bytes().to_vec(), v.clone()));
                    }
                }
            }
            if let Some(first_pkt) = session.mtp_first_output_packet {
                details.push((
                    b"native_mtp_first_output_packet".to_vec(),
                    Json::integer(u64::from(first_pkt)),
                ));
            }
            result.push((b"tokspur_private".to_vec(), Json::Object(details)));
        } else {
            #[cfg(feature = "verify8")]
            if session.verify8 {
                let c = &self.counters;
                let details = vec![
                    (b"native_verify8".to_vec(), Json::Bool(true)),
                    (b"verification_max_tokens".to_vec(), Json::integer(8)),
                    (
                        b"known_input_batches".to_vec(),
                        Json::integer(c.input_batches),
                    ),
                    (
                        b"maximum_known_input_width".to_vec(),
                        Json::integer(c.input_width as u64),
                    ),
                    (
                        b"generation_batches".to_vec(),
                        Json::integer(c.output_batches),
                    ),
                    (
                        b"maximum_generation_width".to_vec(),
                        Json::integer(c.output_width as u64),
                    ),
                    (
                        b"lookup_rounds".to_vec(),
                        Json::integer(c.generation_rounds),
                    ),
                    (
                        b"lookup_verified_rounds".to_vec(),
                        Json::integer(c.verified_rounds),
                    ),
                    (
                        b"lookup_accepted_tokens".to_vec(),
                        Json::integer(c.accepted_tokens),
                    ),
                    (
                        b"hc_exact_final_logit_sha256".to_vec(),
                        field(&session.verify_previous, "hc_exact_final_logit_sha256")?.clone(),
                    ),
                    (b"performance_qualified".to_vec(), Json::Bool(false)),
                ];
                #[cfg(feature = "branch8")]
                let details = self.branch8_details(session, details)?;
                result.push((b"tokspur_private".to_vec(), Json::Object(details)));
            }
        }
        Ok(result)
    }

    /// The branch8 additions to `tokspur_private`.
    #[cfg(feature = "branch8")]
    fn branch8_details(
        &self,
        session: &ServeSession,
        mut details: Vec<(Vec<u8>, Json)>,
    ) -> Result<Vec<(Vec<u8>, Json)>> {
        let c = &self.counters;
        {
            if session.branch8 {
                details.push((b"branch8_reuse".to_vec(), Json::Bool(true)));
                details.push((b"broker_protocol".to_vec(), Json::from("branch8-v1")));
                details.push((
                    b"checkpoint_captures".to_vec(),
                    Json::integer(c.checkpoint_captures),
                ));
                details.push((
                    b"checkpoint_restores".to_vec(),
                    Json::integer(c.checkpoint_restores),
                ));
                details.push((
                    b"checkpoint_cold_resyncs".to_vec(),
                    Json::integer(c.checkpoint_cold_resyncs),
                ));
                details.push((
                    b"checkpoint_slots".to_vec(),
                    field(&session.verify_ready, "checkpoint_slots")?.clone(),
                ));
            }
        }
        Ok(details)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::admit_chat_with_policies;

    fn codec() -> ServeCodec {
        ServeCodec {
            control_ids: Vec::new(),
            nonordinary_ids: Vec::new(),
            render: Box::new(|_| Err(AdmissionError::Refused("unused").into())),
            encode: Box::new(|_| Ok(Vec::new())),
            decode: Box::new(|_| Ok(Vec::new())),
            admission: ResponseAdmission {
                vocabulary: 300,
                im_end: 1,
                end_of_text: 2,
                im_start: 3,
            },
        }
    }

    #[test]
    fn session_debug_counts_committed_tokens_without_their_ids() {
        // Arrange
        let mut session = ServeSession::new(codec(), SessionOptions::default()).unwrap();
        session.committed = vec![151_644, 872, 198];

        // Act
        let rendered = format!("{session:?}");

        // Assert
        assert_eq!(
            rendered,
            "ServeSession { committed_tokens: 3, reusable: false, .. }"
        );
        assert!(!rendered.contains("151644") && !rendered.contains("872"));
    }

    /// What `flush` did before the cursor: rebuild both streams from every record.
    fn rebuilt(records: &[ResponseRecord]) -> (Vec<u8>, Vec<u8>) {
        let mut thinking = Vec::new();
        let mut content = Vec::new();
        for record in records {
            match record.kind {
                ResponseKind::Text => content.extend_from_slice(&record.text),
                ResponseKind::Thinking => thinking.extend_from_slice(&record.text),
                ResponseKind::Tool => {}
            }
        }
        (thinking, content)
    }

    #[test]
    fn stream_cursor_emits_the_same_deltas_as_a_full_rebuild() {
        // Arrange: thinking, text either side of a withheld tool call, multi-byte UTF-8 and a held `<` marker prefix.
        let response =
            "plan 日本 <b\n</think>answer 🙂 a<b <tool_call><function=ping></function></tool_call>";
        let tools = vec![tokspur_text::response::ToolSpec::new("ping", Vec::new())];
        let mut parser = ResponseParser::new(tools, true, ResponseLimits::default()).unwrap();
        let mut cursor = StreamCursor::default();
        let (mut thinking, mut content) = (Vec::new(), Vec::new());

        // Act: one byte per feed, the smallest step the cursor can see.
        for byte in response.bytes() {
            assert_eq!(parser.feed(&[byte]), ResponseStatus::Incomplete);
            let before = (thinking.len(), content.len());
            let delta = cursor.advance(parser.records()).unwrap();
            thinking.extend_from_slice(&delta.thinking);
            content.extend_from_slice(&delta.content);

            // Assert: every step matches the rebuild, byte for byte.
            let (all_thinking, all_content) = rebuilt(parser.records());
            assert_eq!(delta.thinking, all_thinking[before.0..]);
            assert_eq!(delta.content, all_content[before.1..]);
        }
        assert!(!thinking.is_empty() && !content.is_empty());
        assert_eq!(parser.records().len(), 2);
    }

    #[test]
    fn stream_cursor_refuses_records_that_shrink() {
        // Arrange
        let mut parser = ResponseParser::new(Vec::new(), false, ResponseLimits::default()).unwrap();
        parser.feed(b"hello world");
        let mut cursor = StreamCursor::default();
        cursor.advance(parser.records()).unwrap();
        let mut shorter = parser.records().to_vec();
        shorter[0].text.truncate(5);

        // Act + Assert
        assert!(cursor.advance(&shorter).is_err());
        assert!(cursor.advance(&[]).is_err());
    }

    #[test]
    fn codec_debug_shows_no_fields() {
        // Act
        let rendered = format!("{:?}", codec());

        // Assert
        assert_eq!(rendered, "ServeCodec { .. }");
    }

    #[test]
    fn omitted_reasoning_requires_an_exact_prefix_and_prework_failure_preserves_state() {
        let encode = |text: &[u8]| text.iter().map(|b| u32::from(*b) + 4).collect::<Vec<_>>();
        let mut text = codec();
        text.encode = Box::new(move |raw| Ok(encode(&raw.prompt)));
        text.render = Box::new(|request| {
            let mut prompt = Vec::new();
            for message in field(request, "messages")?.array()? {
                let assistant = field(message, "role")?.string()? == b"assistant";
                if assistant && let Some(reasoning) = message.find("reasoning_content")? {
                    prompt.extend_from_slice(b"[reasoning:");
                    prompt.extend_from_slice(reasoning.string()?);
                    prompt.push(b']');
                }
                prompt.extend_from_slice(field(message, "content")?.string()?);
                if assistant {
                    prompt.push(b';');
                }
            }
            Ok(RenderedChat {
                prompt,
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        });
        let mut session = ServeSession::new(text, SessionOptions::default()).unwrap();
        session.reusable = true;
        session.committed = encode(b"A[reasoning:why]Visible;");
        session.saved_reasoning.push((0, b"why".to_vec()));
        for (first, answer, expected_append) in [
            ("A", "Visible", true),
            ("X", "Visible", false),
            ("A", "Edited", false),
        ] {
            let input = tokspur_json::parse_json(format!(r#"{{"model":"tokspur","messages":[{{"role":"user","content":"{first}"}},{{"role":"assistant","content":"{answer}"}},{{"role":"user","content":"B"}}]}}"#).as_bytes()).unwrap();
            let admitted = admit_chat_with_policies(
                &input,
                b"tokspur",
                ToolEnumValidation::Strict,
                ToolArgValidation::Strict,
            )
            .unwrap();
            session.sampling = admitted.sampling.dump().unwrap();
            let before = session.committed.clone();
            let mut observed = Json::Null;
            let mut trace = |value: &Json| {
                observed = value.clone();
                Err(AdmissionError::Refused("controlled trace refusal").into())
            };
            let mut cancelled = || false;
            let mut seed = || 42;
            let mut sink = ServeSink {
                chunk: None,
                cancelled: &mut cancelled,
                new_seed: &mut seed,
                prefix_trace: Some(&mut trace),
            };
            let mut target = |_: &Json, _: TokenCallback<'_>| -> Result<Json> {
                panic!("prework refusal submitted a worker command")
            };
            assert!(
                session
                    .complete(&input, b"test", &mut target, &mut sink)
                    .is_err()
            );
            assert_eq!(
                field(&observed, "append").unwrap().boolean().unwrap(),
                expected_append
            );
            let ids = field(&observed, "prompt_ids")
                .unwrap()
                .array()
                .unwrap()
                .iter()
                .map(|value| value.unsigned_integer().unwrap() as u32)
                .collect::<Vec<_>>();
            assert_eq!(ids.starts_with(&before), expected_append);
            if !expected_append {
                assert!(!ids.windows(3).any(|part| part == encode(b"why")));
            }
            assert_eq!(session.committed, before);
            assert!(session.reusable);
        }
    }

    /// Bytes map to `byte + 4`, `;` is the assistant terminal and `MERGED` spells `Vi` as one token.
    const MERGED: u32 = 299;

    fn respelling_codec() -> ServeCodec {
        let mut text = codec();
        text.encode = Box::new(|raw| {
            Ok(raw
                .prompt
                .iter()
                .map(|&b| if b == b';' { 1 } else { u32::from(b) + 4 })
                .collect())
        });
        text.decode = Box::new(|ids| {
            let mut bytes = Vec::new();
            for &id in ids {
                match id {
                    MERGED => bytes.extend_from_slice(b"Vi"),
                    1 => bytes.push(b';'),
                    _ => bytes.push((id - 4) as u8),
                }
            }
            Ok(bytes)
        });
        text.render = Box::new(|request| {
            let mut prompt = Vec::new();
            for message in field(request, "messages")?.array()? {
                prompt.extend_from_slice(field(message, "content")?.string()?);
                if field(message, "role")?.string()? == b"assistant" {
                    prompt.push(b';');
                }
            }
            Ok(RenderedChat {
                prompt,
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        });
        text
    }

    /// Admits `A`, the echoed `answer` and `B`, then returns the prefix trace of the refused request.
    fn respelling_trace(session: &mut ServeSession, answer: &str) -> Json {
        let input = tokspur_json::parse_json(format!(r#"{{"model":"tokspur","messages":[{{"role":"user","content":"A"}},{{"role":"assistant","content":"{answer}"}},{{"role":"user","content":"B"}}]}}"#).as_bytes()).unwrap();
        let admitted = admit_chat_with_policies(
            &input,
            b"tokspur",
            ToolEnumValidation::Strict,
            ToolArgValidation::Strict,
        )
        .unwrap();
        session.sampling = admitted.sampling.dump().unwrap();
        let mut observed = Json::Null;
        let mut trace = |value: &Json| {
            observed = value.clone();
            Err(AdmissionError::Refused("controlled trace refusal").into())
        };
        let mut cancelled = || false;
        let mut seed = || 42;
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: Some(&mut trace),
        };
        let mut target = |_: &Json, _: TokenCallback<'_>| -> Result<Json> {
            panic!("prework refusal submitted a worker command")
        };
        assert!(
            session
                .complete(&input, b"test", &mut target, &mut sink)
                .is_err()
        );
        observed
    }

    fn trace_ids(trace: &Json) -> Vec<u32> {
        field(trace, "prompt_ids")
            .unwrap()
            .array()
            .unwrap()
            .iter()
            .map(|value| value.unsigned_integer().unwrap() as u32)
            .collect()
    }

    /// A live prefix whose answer `Visible` was sampled as `MERGED` plus the remaining bytes.
    fn respelled_session(terminal: bool) -> ServeSession {
        let mut session = ServeSession::new(respelling_codec(), SessionOptions::default()).unwrap();
        session.reusable = true;
        session.committed = vec![u32::from(b'A') + 4, MERGED];
        session
            .committed
            .extend(b"sible".iter().map(|&b| u32::from(b) + 4));
        if terminal {
            session.committed.push(1);
        }
        session
    }

    #[test]
    fn same_text_in_another_token_spelling_appends_to_the_live_prefix() {
        // Arrange
        let mut session = respelled_session(true);
        let committed = session.committed.clone();

        // Act
        let trace = respelling_trace(&mut session, "Visible");

        // Assert
        let mut expected = committed;
        expected.push(u32::from(b'B') + 4);
        assert!(field(&trace, "append").unwrap().boolean().unwrap());
        assert_eq!(trace_ids(&trace), expected);
    }

    #[test]
    fn edited_text_does_not_append_to_a_respelled_prefix() {
        // Arrange
        let mut session = respelled_session(true);

        // Act
        let trace = respelling_trace(&mut session, "Visibly");

        // Assert
        assert!(!field(&trace, "append").unwrap().boolean().unwrap());
        assert!(!trace_ids(&trace).contains(&MERGED));
    }

    #[test]
    fn respelled_prefix_without_a_committed_terminal_does_not_append() {
        // Arrange
        let mut session = respelled_session(false);

        // Act
        let trace = respelling_trace(&mut session, "Visible");

        // Assert
        assert!(!field(&trace, "append").unwrap().boolean().unwrap());
        assert!(!trace_ids(&trace).contains(&MERGED));
    }

    #[test]
    fn nonordinary_tokens_must_not_match_ordinary_text() {
        for (marker, spelling) in [
            (280, "<think>"),
            (281, "</think>"),
            (282, "<tool_call>"),
            (283, "</tool_call>"),
            (284, "<tool_response>"),
            (285, "</tool_response>"),
            (286, "<added>"),
            (287, "[PAD287]"),
        ] {
            let mut text = respelling_codec();
            text.nonordinary_ids = vec![marker];
            let decode = text.decode;
            text.decode = Box::new(move |ids| {
                let mut bytes = Vec::new();
                for &id in ids {
                    if id == marker {
                        bytes.extend_from_slice(spelling.as_bytes());
                    } else {
                        bytes.extend(decode(&[id])?);
                    }
                }
                Ok(bytes)
            });
            let mut session = ServeSession::new(text, SessionOptions::default()).unwrap();
            let mut ordinary = vec![u32::from(b'A') + 4];
            ordinary.extend(spelling.bytes().map(|b| u32::from(b) + 4));
            ordinary.push(1);
            let special = vec![u32::from(b'A') + 4, marker, 1];
            for (committed, mut canonical) in
                [(special.clone(), ordinary.clone()), (ordinary, special)]
            {
                session.committed = committed;
                canonical.push(u32::from(b'B') + 4);
                assert!(
                    session
                        .respelled_on_committed(&session.committed, &canonical)
                        .unwrap()
                        .is_none(),
                    "{spelling}"
                );
            }
            session.committed = vec![MERGED, marker, u32::from(b'x') + 4, 1];
            let canonical = vec![
                u32::from(b'V') + 4,
                u32::from(b'i') + 4,
                marker,
                u32::from(b'x') + 4,
                1,
                u32::from(b'B') + 4,
            ];
            let mut expected = session.committed.clone();
            expected.push(u32::from(b'B') + 4);
            assert_eq!(
                session
                    .respelled_on_committed(&session.committed, &canonical)
                    .unwrap(),
                Some(expected)
            );
        }
    }

    #[test]
    fn nonordinary_ids_are_validated_and_sorted() {
        let mut text = codec();
        text.nonordinary_ids = vec![285, 280, 285];
        let session = ServeSession::new(text, SessionOptions::default()).unwrap();
        assert_eq!(session.codec.nonordinary_ids, vec![280, 285]);
        let mut invalid = codec();
        invalid.nonordinary_ids = vec![invalid.admission.vocabulary];
        assert!(ServeSession::new(invalid, SessionOptions::default()).is_err());
    }

    fn trace_without_worker(session: &mut ServeSession, input: &Json) -> Json {
        let mut observed = Json::Null;
        let mut trace = |value: &Json| {
            observed = value.clone();
            Err(AdmissionError::Refused("controlled trace refusal").into())
        };
        let mut cancelled = || false;
        let mut seed = || 42;
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut seed,
            prefix_trace: Some(&mut trace),
        };
        let mut target = |_: &Json, _: TokenCallback<'_>| -> Result<Json> {
            panic!("unexpected worker request")
        };
        assert!(
            session
                .complete(input, b"test", &mut target, &mut sink)
                .is_err()
        );
        observed
    }

    #[test]
    fn changed_sampling_or_seed_preserves_canonical_prompt_on_reset() {
        let original = tokspur_json::parse_json(br#"{"model":"tokspur","seed":42,"temperature":0.7,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}]}"#).unwrap();
        let admitted = admit_chat_with_policies(
            &original,
            b"tokspur",
            ToolEnumValidation::Strict,
            ToolArgValidation::Strict,
        )
        .unwrap();
        for (seed, temperature, append) in [(99, 0.7, false), (42, 0.5, false), (42, 0.7, true)] {
            let input = tokspur_json::parse_json(format!(r#"{{"model":"tokspur","seed":{seed},"temperature":{temperature},"messages":[{{"role":"user","content":"A"}},{{"role":"assistant","content":"Visible"}},{{"role":"user","content":"B"}}]}}"#).as_bytes()).unwrap();
            let mut session = respelled_session(true);
            session.sampling = admitted.sampling.dump().unwrap();
            session.seed = 42;
            let expected =
                (session.codec.encode)(&(session.codec.render)(&admitted.render).unwrap()).unwrap();
            let observed = trace_without_worker(&mut session, &input);
            assert_eq!(
                field(&observed, "append").unwrap().boolean().unwrap(),
                append
            );
            let ids = trace_ids(&observed);
            assert_eq!(ids.contains(&MERGED), append);
            if !append {
                assert_eq!(ids, expected);
            }
        }
    }

    #[test]
    fn identical_tokens_around_changed_spans_do_not_need_decoding() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let decoded = Arc::new(AtomicUsize::new(0));
        let seen = decoded.clone();
        let mut session = respelled_session(true);
        session.codec.decode = Box::new(move |ids| {
            seen.fetch_add(ids.len(), Ordering::Relaxed);
            let mut bytes = Vec::new();
            for &id in ids {
                if id == MERGED {
                    bytes.extend_from_slice(b"Vi");
                } else {
                    bytes.push((id - 4) as u8);
                }
            }
            Ok(bytes)
        });
        let equal = vec![u32::from(b'x') + 4; 20_000];
        session.committed = vec![MERGED, 1];
        session.committed.extend(&equal);
        session.committed.push(1);
        session.committed.extend(&equal);
        session.committed.push(MERGED);
        session.committed.extend(&equal);
        session.committed.push(1);
        let mut canonical = vec![u32::from(b'V') + 4, u32::from(b'i') + 4, 1];
        canonical.extend(&equal);
        canonical.push(1);
        canonical.extend(&equal);
        canonical.extend([u32::from(b'V') + 4, u32::from(b'i') + 4]);
        canonical.extend(&equal);
        canonical.extend([1, u32::from(b'B') + 4]);
        assert!(
            session
                .respelled_on_committed(&session.committed, &canonical)
                .unwrap()
                .is_some()
        );
        assert_eq!(decoded.load(Ordering::Relaxed), 6);
    }

    fn parked_mtp_ready() -> Json {
        tokspur_json::parse_json(br#"{"event":"ready","maximum_generated_tokens":4096,"stack_execution":"direct","mapped_weight_region_limit_bytes":0,"small_gemv_column_splits":16,"block_gemv_formats":3,"native_mtp_protocol":"native-mtp-depth2-lookahead-v1","native_mtp_depth":2,"native_mtp_draft_attention":"tiled128","native_mtp_input_packet_schema":1,"native_mtp_input_packet_max":512,"native_prefill_dense":"f32","native_prefill_grouped":"f32","native_mtp_priming_batch_capacity":512,"native_mtp_priming_batch_bytes":1024,"native_mtp_priming_batch_submits":0,"native_mtp_carry_scope":"http-generation-v1","native_mtp_generation":{"id":0,"active":false,"remaining":0,"pending":false},"native_mtp_head_calls":0,"native_mtp_prime_calls":0,"native_mtp_rounds":0,"native_mtp_proposed":0,"native_mtp_accepted":0,"native_mtp_recovery_rounds":0,"native_mtp_carried_anchors":0,"native_mtp_discarded_anchors":0}"#).unwrap()
    }

    fn parked_reasoning_codec() -> ServeCodec {
        let mut text = codec();
        text.encode = Box::new(|raw| {
            let mut out = Vec::new();
            let s = &raw.prompt;
            let mut i = 0;
            while i < s.len() {
                if s[i..].starts_with(b"</think>") {
                    out.push(281);
                    i += b"</think>".len();
                } else if s[i] == b';' {
                    out.push(1);
                    i += 1;
                } else {
                    out.push(u32::from(s[i]) + 4);
                    i += 1;
                }
            }
            Ok(out)
        });
        text.decode = Box::new(|ids| {
            let mut bytes = Vec::new();
            for &id in ids {
                match id {
                    MERGED => bytes.extend_from_slice(b"Vi"),
                    1 => bytes.extend_from_slice(b"<|im_end|>"),
                    2 => bytes.extend_from_slice(b"<|endoftext|>"),
                    3 => bytes.extend_from_slice(b"<|im_start|>"),
                    281 => bytes.extend_from_slice(b"</think>"),
                    _ => bytes.push((id - 4) as u8),
                }
            }
            Ok(bytes)
        });
        text.render = Box::new(|request| {
            let mut prompt = Vec::new();
            for message in field(request, "messages")?.array()? {
                let assistant = field(message, "role")?.string()? == b"assistant";
                if assistant
                    && let Some(reasoning) = message.find("reasoning_content")?
                    && !reasoning.is_null()
                {
                    prompt.extend_from_slice(b"[reasoning:");
                    prompt.extend_from_slice(reasoning.string()?);
                    prompt.extend_from_slice(b"</think>");
                }
                prompt.extend_from_slice(field(message, "content")?.string()?);
                if assistant {
                    prompt.push(b';');
                }
            }
            Ok(RenderedChat {
                prompt,
                normalized_messages: Json::Array(Vec::new()),
                parsed_tool_arguments: 0,
                hoisted_system_messages: 0,
                literal_ranges: Vec::new(),
            })
        });
        text.nonordinary_ids = vec![281];
        text
    }

    fn sampling_policy_bytes(seed: Option<u64>, temperature: &str) -> Vec<u8> {
        let seed_part = seed.map_or_else(String::new, |s| format!(r#""seed":{s},"#));
        let input = tokspur_json::parse_json(
            format!(
                r#"{{"model":"tokspur",{seed_part}"temperature":{temperature},"messages":[{{"role":"user","content":"A"}}]}}"#
            )
            .as_bytes(),
        )
        .unwrap();
        admit_chat_with_policies(
            &input,
            b"tokspur",
            ToolEnumValidation::Strict,
            ToolArgValidation::Strict,
        )
        .unwrap()
        .sampling
        .dump()
        .unwrap()
    }

    #[test]
    fn parked_lane_with_enable_thinking_false_parks_and_matches_respelled_prefix() {
        let mut session = ServeSession::new(
            parked_reasoning_codec(),
            SessionOptions {
                mtp_ready: parked_mtp_ready(),
                lane_parking: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert!(
            session
                .respelled_on_committed(&[], &[u32::from(b'A') + 4])
                .unwrap()
                .is_none()
        );

        let policy = sampling_policy_bytes(Some(42), "0.7");
        let mut lane_a = vec![u32::from(b'A') + 4, MERGED];
        lane_a.extend(
            b"sible;"
                .iter()
                .map(|&b| if b == b';' { 1 } else { u32::from(b) + 4 }),
        );
        session.reusable = true;
        session.committed = lane_a.clone();
        session.sampling = policy.clone();
        session.seed = 42;
        assert!(session.saved_reasoning.is_empty());

        // Switching to conversation B parks A even though enable_thinking was false (empty saved_reasoning).
        session.park_active_reasoning();
        assert_eq!(session.parked_reasoning.len(), 1);
        assert!(session.parked_reasoning[0].turns.is_empty());
        assert_eq!(session.parked_reasoning[0].sampling, policy);
        assert_eq!(session.parked_reasoning[0].seed, 42);

        // Simulate active conversation B on the live lane.
        session.committed = vec![u32::from(b'Z') + 4, u32::from(b'B') + 4, 1];
        session.sampling = sampling_policy_bytes(Some(99), "0.7");
        session.seed = 99;

        let return_a = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":42,"temperature":0.7,"enable_thinking":false,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace = trace_without_worker(&mut session, &return_a);
        assert!(!field(&trace, "append").unwrap().boolean().unwrap());
        let mut expected = lane_a;
        expected.push(u32::from(b'B') + 4);
        assert_eq!(trace_ids(&trace), expected);
    }

    #[test]
    fn parked_lane_with_echoed_reasoning_falls_back_to_base_prompt_and_respells() {
        let mut session = ServeSession::new(
            parked_reasoning_codec(),
            SessionOptions {
                mtp_ready: parked_mtp_ready(),
                lane_parking: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let policy = sampling_policy_bytes(Some(42), "0.7");
        let mut lane_a = Vec::new();
        lane_a.extend(b"A[reasoning:".iter().map(|&b| u32::from(b) + 4));
        lane_a.push(MERGED);
        lane_a.extend(b"s".iter().map(|&b| u32::from(b) + 4));
        lane_a.push(281); // </think>
        lane_a.push(MERGED);
        lane_a.extend(b"sible".iter().map(|&b| u32::from(b) + 4));
        lane_a.push(1);

        session.reusable = true;
        session.committed = lane_a.clone();
        session.saved_reasoning = vec![(0, b"Vis".to_vec())];
        session.sampling = policy;
        session.seed = 42;
        session.park_active_reasoning();

        session.committed = vec![u32::from(b'Z') + 4, 1];
        session.seed = 99;

        // Client echoes reasoning_content: fill_saved_reasoning returns None, so base_prompt is respelled.
        let echoed = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":42,"temperature":0.7,"messages":[{"role":"user","content":"A"},{"role":"assistant","reasoning_content":"Vis","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace = trace_without_worker(&mut session, &echoed);
        let mut expected = lane_a;
        expected.push(u32::from(b'B') + 4);
        assert_eq!(trace_ids(&trace), expected);

        // Tampered echoed reasoning_content must miss and stay canonical.
        let tampered = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":42,"temperature":0.7,"messages":[{"role":"user","content":"A"},{"role":"assistant","reasoning_content":"Other","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace_tampered = trace_without_worker(&mut session, &tampered);
        assert!(!trace_ids(&trace_tampered).contains(&MERGED));
    }

    #[test]
    fn parked_lane_hoists_and_deduplicates_render_and_encode_calls() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let render_calls = Arc::new(AtomicUsize::new(0));
        let encode_calls = Arc::new(AtomicUsize::new(0));
        let rc = Arc::clone(&render_calls);
        let ec = Arc::clone(&encode_calls);

        let mut text = parked_reasoning_codec();
        let inner_render = text.render;
        let inner_encode = text.encode;
        text.render = Box::new(move |req| {
            rc.fetch_add(1, Ordering::Relaxed);
            inner_render(req)
        });
        text.encode = Box::new(move |rendered| {
            ec.fetch_add(1, Ordering::Relaxed);
            inner_encode(rendered)
        });

        let mut session = ServeSession::new(
            text,
            SessionOptions {
                mtp_ready: parked_mtp_ready(),
                lane_parking: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();

        let policy_match = sampling_policy_bytes(Some(42), "0.7");
        let policy_other = sampling_policy_bytes(Some(42), "0.5");
        let shared_turns = vec![(0, b"why".to_vec())];
        let mut exact_a = (session.codec.encode)(
            &(session.codec.render)(
                &tokspur_json::parse_json(
                    br#"{"messages":[{"role":"user","content":"A"},{"role":"assistant","reasoning_content":"why","content":"Visible"}]}"#,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        // Reset counters after fixture setup.
        render_calls.store(0, Ordering::Relaxed);
        encode_calls.store(0, Ordering::Relaxed);

        // Case 1: 4 empty-turn lanes + 2 echoed-turn lanes with echoed reasoning -> only 1 render + 1 encode.
        for s in [10u64, 11, 12, 13] {
            session.parked_reasoning.push(ParkedReasoning {
                committed: vec![u32::from(b'X') + 4, 1],
                turns: Vec::new(),
                sampling: policy_match.clone(),
                seed: s,
            });
        }
        session.parked_reasoning.push(ParkedReasoning {
            committed: vec![u32::from(b'Y') + 4, 1],
            turns: shared_turns.clone(),
            sampling: policy_match.clone(),
            seed: 42,
        });
        let echoed_req = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":42,"temperature":0.7,"messages":[{"role":"user","content":"A"},{"role":"assistant","reasoning_content":"why","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let _ = trace_without_worker(&mut session, &echoed_req);
        assert_eq!(render_calls.load(Ordering::Relaxed), 1);
        assert_eq!(encode_calls.load(Ordering::Relaxed), 1);

        // Case 2: Omitted reasoning with 3 lanes sharing identical non-empty turns,
        // 1 incompatible sampling lane, and 1 shorter lane after a longer match.
        session.parked_reasoning.clear();
        let mut respelled_prefix = exact_a.clone();
        // Replace canonical "Vi" after "</think>" (281) with MERGED.
        let think_end = respelled_prefix.iter().position(|&id| id == 281).unwrap();
        respelled_prefix.splice(think_end + 1..think_end + 3, [MERGED]);
        exact_a.clear();
        let _ = exact_a;

        // Longer matching lane first (so shorter lanes are skipped by < best_len without rendering).
        session.parked_reasoning.push(ParkedReasoning {
            committed: respelled_prefix.clone(),
            turns: shared_turns.clone(),
            sampling: policy_match.clone(),
            seed: 42,
        });
        // Same turns, non-matching prefix, same length -> reuses cached_by_turns (0 extra render/encode).
        let mut non_matching_same_len = respelled_prefix.clone();
        non_matching_same_len[0] = u32::from(b'Q') + 4;
        session.parked_reasoning.push(ParkedReasoning {
            committed: non_matching_same_len,
            turns: shared_turns.clone(),
            sampling: policy_match.clone(),
            seed: 42,
        });
        // Different turns, incompatible policy -> filtered before render/encode.
        session.parked_reasoning.push(ParkedReasoning {
            committed: respelled_prefix.clone(),
            turns: vec![(0, b"incompatible".to_vec())],
            sampling: policy_other,
            seed: 42,
        });
        // Different turns, strictly shorter than best -> filtered before render/encode.
        session.parked_reasoning.push(ParkedReasoning {
            committed: vec![u32::from(b'A') + 4, 1],
            turns: vec![(0, b"shorter".to_vec())],
            sampling: policy_match,
            seed: 42,
        });

        render_calls.store(0, Ordering::Relaxed);
        encode_calls.store(0, Ordering::Relaxed);
        let omitted_req = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":42,"temperature":0.7,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace = trace_without_worker(&mut session, &omitted_req);
        assert_eq!(render_calls.load(Ordering::Relaxed), 2);
        assert_eq!(encode_calls.load(Ordering::Relaxed), 2);
        assert!(trace_ids(&trace).contains(&MERGED));
    }

    #[test]
    fn parked_lane_respects_per_lane_sampling_seed_and_ambiguous_seed_ties() {
        let mut session = ServeSession::new(
            parked_reasoning_codec(),
            SessionOptions {
                mtp_ready: parked_mtp_ready(),
                lane_parking: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let policy_07 = sampling_policy_bytes(None, "0.7");
        let mut respelled_a = vec![u32::from(b'A') + 4, MERGED];
        respelled_a.extend(b"sible".iter().map(|&b| u32::from(b) + 4));
        respelled_a.push(1);
        let mut canonical_a = Vec::new();
        canonical_a.extend(b"AVisible".iter().map(|&b| u32::from(b) + 4));
        canonical_a.push(1);

        // Park two lanes with the same text/policy but different seeds (11 vs 22).
        // Superseding must preserve both because seeds differ.
        session.reusable = true;
        session.sampling = policy_07.clone();
        session.committed = respelled_a.clone();
        session.seed = 11;
        session.park_active_reasoning();

        session.committed = respelled_a.clone();
        session.seed = 22;
        session.park_active_reasoning();
        assert_eq!(session.parked_reasoning.len(), 2);

        // Switch live lane to conversation B.
        session.committed = vec![u32::from(b'Z') + 4, 1];
        session.seed = 99;

        // With seed omitted, two matching lanes of equal length trigger ambiguous_seed -> miss!
        let req_no_seed = tokspur_json::parse_json(
            br#"{"model":"tokspur","temperature":0.7,"enable_thinking":false,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace_ambig = trace_without_worker(&mut session, &req_no_seed);
        assert!(!trace_ids(&trace_ambig).contains(&MERGED));

        // Explicit seed=22 disambiguates and hits lane 2 (respelled).
        let req_seed_22 = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":22,"temperature":0.7,"enable_thinking":false,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace_22 = trace_without_worker(&mut session, &req_seed_22);
        assert!(trace_ids(&trace_22).contains(&MERGED));

        // Mismatched temperature=0.5 misses both parked lanes.
        let req_05 = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":22,"temperature":0.5,"enable_thinking":false,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"}]}"#,
        )
        .unwrap();
        let trace_05 = trace_without_worker(&mut session, &req_05);
        assert!(!trace_ids(&trace_05).contains(&MERGED));

        // Adding a strictly longer lane (seed 33) overrides the shorter tie even when seed is omitted.
        let mut longer = respelled_a.clone();
        longer.push(u32::from(b'B') + 4);
        longer.push(u32::from(b'C') + 4);
        longer.push(1);
        session.committed = longer.clone();
        session.seed = 33;
        session.park_active_reasoning();
        assert_eq!(session.parked_reasoning.len(), 3);

        // Now complete() end-to-end with seed=None, verifying lane_inherit_seed + lane_seed adoption
        // + consumed lane removal + live lane parking + subsequent append.
        session.committed = vec![u32::from(b'Z') + 4, 1];
        session.seed = 99;
        let req_longer_no_seed = tokspur_json::parse_json(
            br#"{"model":"tokspur","temperature":0.7,"max_tokens":16,"enable_thinking":false,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"},{"role":"assistant","content":"C"},{"role":"user","content":"D"}]}"#,
        )
        .unwrap();
        let ready = parked_mtp_ready();
        let mut saw_inherit_seed = false;
        let mut resets = 0usize;
        let mut position = 0u64;
        let mut primes = 0u64;
        let mut gen_id = 0u64;
        let mut out_queue =
            std::collections::VecDeque::from([u32::from(b'O') + 4, u32::from(b'K') + 4, 1]);
        let mut target = |job: &Json, event: TokenCallback<'_>| -> Result<Json> {
            let reset = field(job, "reset")?.boolean()?;
            if reset {
                resets += 1;
                position = 0;
                if let Some(inherit) = job.find("lane_inherit_seed")?
                    && inherit.boolean()?
                {
                    saw_inherit_seed = true;
                }
            }
            let toks = field(job, "tokens")?.array()?;
            let inputs = toks.len() as u64;
            let generate = field(job, "generate")?.unsigned_integer()? as usize;
            let mut done = ready.clone();
            let mut put_field = |name: &str, val: Json| {
                let mut fields = done.object().unwrap().clone();
                fields.retain(|(k, _)| k != name.as_bytes());
                fields.push((name.as_bytes().to_vec(), val));
                done = Json::Object(fields);
            };
            put_field("event", Json::from("done"));
            if inputs > 0 {
                let first_pkt = position == 0 && reset;
                position += inputs;
                primes += inputs - u64::from(reset);
                put_field("committed_prefix", Json::integer(position));
                put_field("generated_ids", Json::Array(Vec::new()));
                put_field("status", Json::from("prefilled"));
                put_field("continuation_allowed", Json::Bool(true));
                put_field(
                    "sampling",
                    Json::new_object([
                        ("valid", Json::Bool(true)),
                        ("seed", Json::integer(33)),
                        ("entropy_prefix", Json::integer(position)),
                        ("entropy_words", Json::integer(0)),
                        ("temperature", Json::integer(0.7 as u64)),
                    ]),
                );
                // Replace sampling with exact policy from job.
                let mut samp = field(job, "sampling")?.object()?.clone();
                for (k, v) in [
                    ("valid", Json::Bool(true)),
                    ("seed", Json::integer(33)),
                    ("entropy_prefix", Json::integer(position)),
                    ("entropy_words", Json::integer(0)),
                ] {
                    samp.retain(|(name, _)| name != k.as_bytes());
                    samp.push((k.as_bytes().to_vec(), v));
                }
                put_field("sampling", Json::Object(samp));
                if first_pkt {
                    put_field(
                        "checkpoint_restored_tokens",
                        Json::integer(longer.len() as u64),
                    );
                    put_field(
                        "replayed_input_tokens",
                        Json::integer(inputs - longer.len() as u64),
                    );
                    put_field("lane_seed", Json::integer(33));
                    put_field("lane_restored_tokens", Json::integer(longer.len() as u64));
                } else {
                    put_field("checkpoint_restored_tokens", Json::integer(0));
                    put_field("replayed_input_tokens", Json::integer(inputs));
                }
                put_field("native_mtp_prime_calls", Json::integer(primes));
                put_field("native_mtp_priming_batch_submits", Json::integer(1));
                put_field(
                    "native_mtp_generation",
                    Json::new_object([
                        ("id", Json::integer(gen_id)),
                        ("active", Json::Bool(false)),
                        ("remaining", Json::integer(0)),
                        ("pending", Json::Bool(false)),
                    ]),
                );
            } else {
                gen_id += 1;
                let mut emitted = Vec::new();
                while emitted.len() < generate
                    && let Some(id) = out_queue.pop_front()
                {
                    position += 1;
                    emitted.push(id);
                    event(&Json::new_object([
                        ("event", Json::from("token")),
                        ("id", Json::integer(u64::from(id))),
                        ("committed_prefix", Json::integer(position)),
                        ("entropy_words", Json::integer(0)),
                    ]))?;
                    if id == 1 {
                        break;
                    }
                }
                primes += emitted.len() as u64;
                let stopped = emitted.last().copied() == Some(1);
                put_field("committed_prefix", Json::integer(position));
                put_field(
                    "generated_ids",
                    Json::Array(
                        emitted
                            .iter()
                            .map(|&id| Json::integer(u64::from(id)))
                            .collect(),
                    ),
                );
                put_field(
                    "status",
                    Json::from(if stopped { "stopped" } else { "length" }),
                );
                put_field(
                    "finish_reason",
                    Json::from(if stopped { "stop" } else { "length" }),
                );
                put_field(
                    "stop_id",
                    if stopped {
                        Json::integer(1)
                    } else {
                        Json::Null
                    },
                );
                put_field("continuation_allowed", Json::Bool(!stopped));
                let mut samp = tokspur_json::parse_json(
                    br#"{"temperature":0.7,"top_k":0,"top_p":1,"min_p":0}"#,
                )
                .unwrap()
                .object()
                .unwrap()
                .clone();
                samp.push((b"valid".to_vec(), Json::Bool(true)));
                samp.push((b"seed".to_vec(), Json::integer(33)));
                samp.push((b"entropy_prefix".to_vec(), Json::integer(position)));
                samp.push((b"entropy_words".to_vec(), Json::integer(0)));
                put_field("sampling", Json::Object(samp));
                put_field("native_mtp_prime_calls", Json::integer(primes));
                put_field("native_mtp_priming_batch_submits", Json::integer(1));
                put_field(
                    "native_mtp_generation",
                    Json::new_object([
                        ("id", Json::integer(gen_id)),
                        ("active", Json::Bool(false)),
                        ("remaining", Json::integer(0)),
                        ("pending", Json::Bool(false)),
                    ]),
                );
            }
            Ok(done)
        };
        let mut cancelled = || false;
        let mut new_seed = || 777;
        let mut sink = ServeSink {
            chunk: None,
            cancelled: &mut cancelled,
            new_seed: &mut new_seed,
            prefix_trace: None,
        };
        let resp = session
            .complete(
                &req_longer_no_seed,
                b"resp-parked-1",
                &mut target,
                &mut sink,
            )
            .unwrap();
        assert!(saw_inherit_seed);
        assert_eq!(resets, 1);
        assert_eq!(session.seed, 33);
        assert_eq!(
            field(field(&resp, "usage").unwrap(), "prompt_tokens_details")
                .unwrap()
                .find("cached_tokens")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap(),
            longer.len() as u64
        );
        // Consumed lane 2 (seed 33) was removed, and previous live lane (seed 99) was parked!
        assert_eq!(session.parked_reasoning.len(), 3);
        assert!(session.parked_reasoning.iter().any(|l| l.seed == 99));
        assert!(!session.parked_reasoning.iter().any(|l| l.seed == 33));

        // Subsequent turn on the restored conversation with seed=None appends cleanly.
        let turn_next = tokspur_json::parse_json(
            br#"{"model":"tokspur","temperature":0.7,"enable_thinking":false,"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"Visible"},{"role":"user","content":"B"},{"role":"assistant","content":"C"},{"role":"user","content":"D"},{"role":"assistant","content":"OK"},{"role":"user","content":"E"}]}"#,
        )
        .unwrap();
        let trace_append = trace_without_worker(&mut session, &turn_next);
        assert!(field(&trace_append, "append").unwrap().boolean().unwrap());
    }

    #[test]
    fn parked_lane_mixed_thinking_and_non_thinking_turns_preserve_ordinals() {
        let mut session = ServeSession::new(
            parked_reasoning_codec(),
            SessionOptions {
                mtp_ready: parked_mtp_ready(),
                lane_parking: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let policy = sampling_policy_bytes(Some(42), "0.7");
        // Turn 0 had enable_thinking=false (no reasoning), Turn 1 had thinking "plan" + MERGED content.
        let rendered_t1 = (session.codec.render)(
            &tokspur_json::parse_json(
                br#"{"messages":[{"role":"user","content":"U0"},{"role":"assistant","content":"A0"},{"role":"user","content":"U1"},{"role":"assistant","reasoning_content":"plan","content":"Visible"}]}"#,
            )
            .unwrap(),
        )
        .unwrap();
        let mut committed_a = (session.codec.encode)(&rendered_t1).unwrap();
        let vi_pos = committed_a
            .windows(2)
            .position(|w| w == [u32::from(b'V') + 4, u32::from(b'i') + 4])
            .unwrap();
        committed_a.splice(vi_pos..vi_pos + 2, [MERGED]);

        session.reusable = true;
        session.committed = committed_a.clone();
        session.saved_reasoning = vec![(1, b"plan".to_vec())];
        session.sampling = policy;
        session.seed = 42;
        session.park_active_reasoning();

        session.committed = vec![u32::from(b'Z') + 4, 1];
        session.seed = 99;

        // Return to A with Turn 2 (enable_thinking=false, omitted reasoning_content on both assistant messages).
        let return_a = tokspur_json::parse_json(
            br#"{"model":"tokspur","seed":42,"temperature":0.7,"enable_thinking":false,"messages":[{"role":"user","content":"U0"},{"role":"assistant","content":"A0"},{"role":"user","content":"U1"},{"role":"assistant","content":"Visible"},{"role":"user","content":"U2"}]}"#,
        )
        .unwrap();
        let trace = trace_without_worker(&mut session, &return_a);
        let mut expected = committed_a;
        expected.push(u32::from(b'U') + 4);
        expected.push(u32::from(b'2') + 4);
        assert_eq!(trace_ids(&trace), expected);
    }

    #[test]
    fn apply_request_telemetry_overlays_ssd_restore_onto_generation_done() {
        let mut session = ServeSession::new(
            parked_reasoning_codec(),
            SessionOptions {
                mtp_ready: parked_mtp_ready(),
                lane_parking: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let gen_done = tokspur_json::parse_json(
            br#"{"event":"done","checkpoint_reuse":"off","checkpoint_ssd_restored":false,"checkpoint_restored_tokens":0,"replayed_input_tokens":0,"hc_exact_final_logit_sha256":"abc"}"#,
        )
        .unwrap();

        // Without prior request telemetry, done is returned unchanged.
        assert_eq!(session.apply_request_telemetry(&gen_done), gen_done);

        session.last_req_telemetry = Some(RequestCheckpointTelemetry {
            ssd_restored: true,
            reuse: Some("ssd_restored"),
            restored_tokens: 61440,
            replayed_tokens: 12291,
        });
        let enriched = session.apply_request_telemetry(&gen_done);
        assert!(
            field(&enriched, "checkpoint_ssd_restored")
                .unwrap()
                .boolean()
                .unwrap()
        );
        assert_eq!(
            field(&enriched, "checkpoint_reuse")
                .unwrap()
                .string()
                .unwrap(),
            b"ssd_restored"
        );
        assert_eq!(
            field(&enriched, "checkpoint_restored_tokens")
                .unwrap()
                .unsigned_integer()
                .unwrap(),
            61440
        );
        assert_eq!(
            field(&enriched, "replayed_input_tokens")
                .unwrap()
                .unsigned_integer()
                .unwrap(),
            12291
        );
        assert_eq!(
            field(&enriched, "hc_exact_final_logit_sha256")
                .unwrap()
                .string()
                .unwrap(),
            b"abc"
        );

        session.invalidate();
        assert_eq!(session.apply_request_telemetry(&gen_done), gen_done);
    }
}

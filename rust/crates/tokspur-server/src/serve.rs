//! `server_cli`, ported one to one from `native/src/serve.cpp`: options, private state, asset pins, the owned worker, readiness publication, the loopback accept loop and the single generation thread.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fmt;
use std::fs::{DirBuilder, File, OpenOptions, TryLockError};
use std::io::{ErrorKind, Read, Write};
use std::os::fd::AsFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tokspur_gguf::{GgufError, MappedFile, ModelFiles};
use tokspur_json::{Json, JsonError, JsonLimits, from_chars_u32, parse_json_limited};
use tokspur_sys::fd as sysfd;
use tokspur_text::chat::{ChatRenderer, extract_flash_next_literals};
use tokspur_text::response_protocol::ResponseAdmission;
use tokspur_text::sha256::{hex, sha256};
use tokspur_wire::worker::{CallbackFailed, OwnedWorker, WorkerError, WorkerOptions};

use crate::admission::{
    AdmissionError, ToolArgValidation, ToolEnumValidation, admit_chat_with_context,
    parse_tool_arg_validation, parse_tool_enum_validation, tool_arg_validation_name,
    tool_arg_validation_policy, tool_enum_validation_name, tool_enum_validation_policy,
    validate_worker_ready,
};
use crate::http::{Connection, HttpError, JSON_RESPONSE_BYTES, Limits, Listener};
use crate::native_text::{NativeText, TextError};
use crate::request_timing::{DeltaKinds, Outcome, RequestTiming, delta_kinds, emits_record};
use crate::session::{ServeCodec, ServeError, ServeSession, ServeSink, SessionOptions};

const TOKENIZER_PIN: &str = tokspur_text::assets::TOKENIZER_SHA256;
const TEMPLATE_PIN: &str = tokspur_text::assets::TEMPLATE_SHA256;

/// Startup and shutdown failures, as the C++ exceptions reaching `main`.
pub enum ServerError {
    /// `require` in serve.cpp (`FormatError`).
    Refused(&'static str),
    /// Private state directory or file operation failed (`FormatError` with an optional I/O cause).
    StateIo {
        reason: &'static str,
        source: Option<std::io::Error>,
    },
    /// Exclusive `server.lock` acquisition failed.
    StateLock(TryLockError),
    Admission(AdmissionError),
    Text(TextError),
    Gguf(GgufError),
    Worker(WorkerError),
    Http(HttpError),
    Serve(ServeError),
    Chat(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Debug for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
            Self::StateIo { reason, source } => formatter
                .debug_struct("StateIo")
                .field("reason", reason)
                .field(
                    "source",
                    &if source.is_some() {
                        "Some([redacted])"
                    } else {
                        "None"
                    },
                )
                .finish(),
            Self::StateLock(_) => formatter.write_str("StateLock([redacted])"),
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::Text(error) => formatter.debug_tuple("Text").field(error).finish(),
            Self::Gguf(error) => formatter.debug_tuple("Gguf").field(error).finish(),
            Self::Worker(error) => formatter.debug_tuple("Worker").field(error).finish(),
            Self::Http(error) => formatter.debug_tuple("Http").field(error).finish(),
            Self::Serve(error) => formatter.debug_tuple("Serve").field(error).finish(),
            Self::Chat(_) => formatter.write_str("Chat([redacted])"),
        }
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) | Self::StateIo { reason, .. } => formatter.write_str(reason),
            Self::StateLock(_) => formatter.write_str("another server owns this state directory"),
            Self::Admission(error) => error.fmt(formatter),
            Self::Text(error) => error.fmt(formatter),
            Self::Gguf(error) => error.fmt(formatter),
            Self::Worker(error) => error.fmt(formatter),
            Self::Http(error) => error.fmt(formatter),
            Self::Serve(error) => error.fmt(formatter),
            Self::Chat(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(_) => None,
            Self::StateIo { source, .. } => source
                .as_ref()
                .map(|error| error as &(dyn std::error::Error + 'static)),
            Self::StateLock(error) => Some(error),
            Self::Admission(error) => Some(error),
            Self::Text(error) => Some(error),
            Self::Gguf(error) => Some(error),
            Self::Worker(error) => Some(error),
            Self::Http(error) => Some(error),
            Self::Serve(error) => Some(error),
            Self::Chat(error) => Some(error.as_ref()),
        }
    }
}

impl From<AdmissionError> for ServerError {
    fn from(error: AdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl From<JsonError> for ServerError {
    fn from(error: JsonError) -> Self {
        Self::Admission(AdmissionError::Json(error))
    }
}

impl From<TextError> for ServerError {
    fn from(error: TextError) -> Self {
        Self::Text(error)
    }
}

impl From<GgufError> for ServerError {
    fn from(error: GgufError) -> Self {
        Self::Gguf(error)
    }
}

impl From<WorkerError> for ServerError {
    fn from(error: WorkerError) -> Self {
        Self::Worker(error)
    }
}

impl From<HttpError> for ServerError {
    fn from(error: HttpError) -> Self {
        Self::Http(error)
    }
}

impl From<ServeError> for ServerError {
    fn from(error: ServeError) -> Self {
        Self::Serve(error)
    }
}

type Result<T> = std::result::Result<T, ServerError>;

fn require(value: bool, why: &'static str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(ServerError::Refused(why))
    }
}

fn random_word() -> u64 {
    let mut bytes = [0u8; 8];
    tokspur_sys::random_bytes(&mut bytes);
    u64::from_ne_bytes(bytes)
}

fn random_hex(bytes: usize) -> Result<String> {
    require(bytes <= 64, "random ID bound")?;
    let mut data = [0u8; 64];
    tokspur_sys::random_bytes(&mut data[..bytes]);
    Ok(hex(&data[..bytes]))
}

/// Constant-time over `right`'s length, as in C++.
fn equal_secret(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for (i, &byte) in right.iter().enumerate() {
        difference |= usize::from(byte ^ left.get(i).copied().unwrap_or(0));
    }
    difference == 0
}

struct PrivateStateLock {
    file: File,
    released: bool,
}

impl fmt::Debug for PrivateStateLock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PrivateStateLock(..)")
    }
}

impl PrivateStateLock {
    fn acquire(file: File) -> Result<Self> {
        file.try_lock().map_err(ServerError::StateLock)?;
        Ok(Self {
            file,
            released: false,
        })
    }

    fn release(&mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        self.file.unlock().map_err(|source| ServerError::StateIo {
            reason: "cannot release private state lock",
            source: Some(source),
        })?;
        self.released = true;
        Ok(())
    }
}

impl Drop for PrivateStateLock {
    fn drop(&mut self) {
        // Release ownership before closing this handle, including when a forked child still holds a duplicate.
        if let Err(error) = self.release() {
            // Drop cannot return an error. Report it here before File closes the descriptor.
            let line = format!("tokspur-server failed to release private state lock: {error}\n");
            if std::io::stderr().write_all(line.as_bytes()).is_err() {
                // stderr is unavailable, so no other output channel remains.
            }
        }
    }
}

/// The private state directory, its exclusive lock and the API key.
struct PrivateState {
    _directory: File,
    _lock: PrivateStateLock,
    key: String,
    prefix_trace: Option<File>,
    prefix_trace_bytes: u64,
    /// Why the prefix trace takes no more records, once it has refused or failed to persist one. Tracing fails closed: every later request is refused with this reason rather than served untraced.
    prefix_trace_refusal: Option<&'static str>,
    telemetry_trace: Option<File>,
    telemetry_trace_bytes: u64,
}

impl fmt::Debug for PrivateState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PrivateState(..)")
    }
}

fn owned_mode(metadata: &std::fs::Metadata, mode: u32) -> bool {
    metadata.uid() == tokspur_sys::effective_uid() && metadata.mode() & 0o777 == mode
}

/// The most bytes the private telemetry trace file may hold. The size already on disk counts, so the bound holds across server starts.
const TELEMETRY_TRACE_LIMIT: u64 = 16 << 20;

/// Members of the worker's `done` record that derive from the reply and so never reach the telemetry trace file. `generated_ids` are the reply's token IDs, which the tokenizer turns straight back into text. `final_logit_top8` is the model's prediction after the whole conversation. Production builds repeat it inside `branch8_session`, so the members are dropped at every depth.
const TELEMETRY_WITHHELD: [&[u8]; 2] = [b"generated_ids", b"final_logit_top8"];

/// A copy of the worker's `done` record without the members in [`TELEMETRY_WITHHELD`]. The counters, timings and memory figures stay. The worker's line parser refuses nesting beyond 12 and the writer beyond 64, which bounds the recursion.
fn telemetry_record(record: &Json) -> Json {
    match record {
        Json::Object(fields) => Json::Object(
            fields
                .iter()
                .filter(|(name, _)| !TELEMETRY_WITHHELD.contains(&name.as_slice()))
                .map(|(name, value)| (name.clone(), telemetry_record(value)))
                .collect(),
        ),
        Json::Array(items) => Json::Array(items.iter().map(telemetry_record).collect()),
        other => other.clone(),
    }
}

/// Whether `record_bytes` more bytes fit under [`TELEMETRY_TRACE_LIMIT`] when `used` bytes are already on disk. A file that is already past the limit has no room.
fn telemetry_trace_has_room(used: u64, record_bytes: usize) -> bool {
    u64::try_from(record_bytes)
        .is_ok_and(|bytes| bytes <= TELEMETRY_TRACE_LIMIT.saturating_sub(used))
}

/// The most bytes the private prefix trace file may hold.
const PREFIX_TRACE_LIMIT: u64 = 16 << 20;

/// The refusal once a record no longer fits the private prefix trace. Sent to clients, so it names no path; the startup refusal names the file.
const PREFIX_TRACE_FULL: &str = "private prefix trace is full (16 MiB bound); no request is served until the server is restarted with a new trace file";

const PREFIX_TRACE_UNAVAILABLE: &str = "private prefix trace is unavailable";

impl PrivateState {
    fn enable_prefix_trace(&mut self) -> Result<()> {
        const PREFIX_REFUSED: &str = "private prefix trace requires a new owned mode 0600 file";
        // An earlier trace is never appended to or overwritten, so the operator has to move it away. Say which file.
        const PREFIX_EXISTS: &str = "private prefix trace requires a new file: move or remove prefix-trace.jsonl from the state directory, or start without --private-prefix-trace";
        let file = sysfd::openat(
            self._directory.as_fd(),
            c"prefix-trace.jsonl",
            sysfd::O_WRONLY | sysfd::O_CREAT | sysfd::O_EXCL | sysfd::O_NOFOLLOW | sysfd::O_CLOEXEC,
            0o600,
        )
        .map_err(|error| ServerError::StateIo {
            reason: if error.kind() == ErrorKind::AlreadyExists {
                PREFIX_EXISTS
            } else {
                PREFIX_REFUSED
            },
            source: Some(error),
        })?;
        let state = file.metadata().map_err(|error| ServerError::StateIo {
            reason: PREFIX_REFUSED,
            source: Some(error),
        })?;
        if !(state.is_file() && state.nlink() == 1 && owned_mode(&state, 0o600)) {
            return Err(ServerError::StateIo {
                reason: PREFIX_REFUSED,
                source: None,
            });
        }
        sysfd::fsync(self._directory.as_fd()).map_err(|error| ServerError::StateIo {
            reason: "cannot persist private prefix trace entry",
            source: Some(error),
        })?;
        self.prefix_trace = Some(file);
        self.prefix_trace_bytes = 0;
        self.prefix_trace_refusal = None;
        Ok(())
    }

    fn write_worker_pid(&mut self, pid: u32) -> Result<()> {
        const PID_OPEN_REFUSED: &str = "cannot open private worker.pid";
        let mut file = sysfd::openat(
            self._directory.as_fd(),
            c"worker.pid",
            sysfd::O_WRONLY | sysfd::O_CREAT | sysfd::O_NOFOLLOW | sysfd::O_CLOEXEC,
            0o600,
        )
        .map_err(|error| ServerError::StateIo {
            reason: PID_OPEN_REFUSED,
            source: Some(error),
        })?;
        let state = file.metadata().map_err(|error| ServerError::StateIo {
            reason: PID_OPEN_REFUSED,
            source: Some(error),
        })?;
        if !(state.is_file() && state.nlink() == 1 && owned_mode(&state, 0o600)) {
            return Err(ServerError::StateIo {
                reason: PID_OPEN_REFUSED,
                source: None,
            });
        }
        use std::io::{Seek, SeekFrom};
        file.set_len(0)
            .and_then(|()| file.seek(SeekFrom::Start(0)).map(|_| ()))
            .map_err(|error| ServerError::StateIo {
                reason: "cannot reset worker.pid",
                source: Some(error),
            })?;
        let content = format!("{pid}\n");
        file.write_all(content.as_bytes())
            .map_err(|error| ServerError::StateIo {
                reason: "cannot write worker.pid",
                source: Some(error),
            })?;
        sysfd::fsync(file.as_fd())
            .and_then(|()| sysfd::fsync(self._directory.as_fd()))
            .map_err(|error| ServerError::StateIo {
                reason: "cannot persist worker.pid",
                source: Some(error),
            })?;
        Ok(())
    }

    fn write_telemetry_trace(&mut self, record: &Json) -> Result<()> {
        if self.telemetry_trace.is_none() {
            const TELEMETRY_OPEN_REFUSED: &str = "cannot open private telemetry trace file";
            let mut file = sysfd::openat(
                self._directory.as_fd(),
                c"telemetry.jsonl",
                sysfd::O_WRONLY | sysfd::O_CREAT | sysfd::O_NOFOLLOW | sysfd::O_CLOEXEC,
                0o600,
            )
            .map_err(|error| ServerError::StateIo {
                reason: TELEMETRY_OPEN_REFUSED,
                source: Some(error),
            })?;
            let state = file.metadata().map_err(|error| ServerError::StateIo {
                reason: TELEMETRY_OPEN_REFUSED,
                source: Some(error),
            })?;
            if !(state.is_file() && state.nlink() == 1 && owned_mode(&state, 0o600)) {
                return Err(ServerError::StateIo {
                    reason: TELEMETRY_OPEN_REFUSED,
                    source: None,
                });
            }
            use std::io::{Seek, SeekFrom};
            // Records from earlier server starts stay in the file, so their size counts toward the bound.
            self.telemetry_trace_bytes =
                file.seek(SeekFrom::End(0))
                    .map_err(|error| ServerError::StateIo {
                        reason: "cannot seek private telemetry trace file",
                        source: Some(error),
                    })?;
            self.telemetry_trace = Some(file);
        }
        let mut bytes = telemetry_record(record).dump_limited(4 * 1024 * 1024)?;
        bytes.push(b'\n');
        if !telemetry_trace_has_room(self.telemetry_trace_bytes, bytes.len()) {
            return Err(ServerError::Refused(
                "private telemetry trace exceeds 16 MiB bound",
            ));
        }
        if let Some(file) = self.telemetry_trace.as_mut() {
            file.write_all(&bytes)
                .and_then(|()| sysfd::fsync(file.as_fd()))
                .map_err(|error| ServerError::StateIo {
                    reason: "cannot persist private telemetry trace",
                    source: Some(error),
                })?;
        }
        self.telemetry_trace_bytes += bytes.len() as u64;
        Ok(())
    }

    fn write_prefix_trace(&mut self, record: &Json) -> std::result::Result<(), ServeError> {
        let refused = |reason| ServeError::Admission(AdmissionError::PrefixTrace(reason));
        if let Some(reason) = self.prefix_trace_refusal {
            return Err(refused(reason));
        }
        let Some(file) = self.prefix_trace.as_mut() else {
            self.prefix_trace_refusal = Some(PREFIX_TRACE_UNAVAILABLE);
            return Err(refused(PREFIX_TRACE_UNAVAILABLE));
        };
        let mut bytes = record
            .dump_limited(4 * 1024 * 1024)
            .map_err(|e| ServeError::Admission(AdmissionError::Json(e)))?;
        bytes.push(b'\n');
        if bytes.len() as u64 > PREFIX_TRACE_LIMIT - self.prefix_trace_bytes {
            self.prefix_trace_refusal = Some(PREFIX_TRACE_FULL);
            return Err(refused(PREFIX_TRACE_FULL));
        }
        if let Err(error) = file
            .write_all(&bytes)
            .and_then(|()| sysfd::fsync(file.as_fd()))
        {
            self.prefix_trace_refusal = Some(PREFIX_TRACE_UNAVAILABLE);
            return Err(ServeError::Admission(AdmissionError::TracePersist(
                Arc::new(error),
            )));
        }
        self.prefix_trace_bytes += bytes.len() as u64;
        Ok(())
    }

    fn new(path: &Path) -> Result<Self> {
        const DIR_REFUSED: &str = "state directory must be owned private mode0700, not a symlink";
        const LOCK_REFUSED: &str = "invalid private server lock";
        const KEY_OPEN_REFUSED: &str =
            "API key must be an owned mode 0600 single-link 65-byte file";
        require(path.is_absolute(), "state directory must be absolute")?;
        if let Err(error) = DirBuilder::new().mode(0o700).create(path)
            && error.kind() != ErrorKind::AlreadyExists
        {
            return Err(ServerError::StateIo {
                reason: "cannot create private state directory",
                source: Some(error),
            });
        }
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(sysfd::O_DIRECTORY | sysfd::O_NOFOLLOW)
            .open(path)
            .map_err(|error| ServerError::StateIo {
                reason: DIR_REFUSED,
                source: Some(error),
            })?;
        let dir_state = directory.metadata().map_err(|error| ServerError::StateIo {
            reason: DIR_REFUSED,
            source: Some(error),
        })?;
        if !(dir_state.is_dir() && owned_mode(&dir_state, 0o700)) {
            return Err(ServerError::StateIo {
                reason: DIR_REFUSED,
                source: None,
            });
        }
        let lock = sysfd::openat(
            directory.as_fd(),
            c"server.lock",
            sysfd::O_RDWR | sysfd::O_CREAT | sysfd::O_NOFOLLOW | sysfd::O_CLOEXEC,
            0o600,
        )
        .map_err(|error| ServerError::StateIo {
            reason: LOCK_REFUSED,
            source: Some(error),
        })?;
        let lock_state = lock.metadata().map_err(|error| ServerError::StateIo {
            reason: LOCK_REFUSED,
            source: Some(error),
        })?;
        if !(lock_state.is_file() && lock_state.nlink() == 1 && owned_mode(&lock_state, 0o600)) {
            return Err(ServerError::StateIo {
                reason: LOCK_REFUSED,
                source: None,
            });
        }
        let lock = PrivateStateLock::acquire(lock)?;
        match sysfd::openat(
            directory.as_fd(),
            c"api-key",
            sysfd::O_WRONLY | sysfd::O_CREAT | sysfd::O_EXCL | sysfd::O_NOFOLLOW | sysfd::O_CLOEXEC,
            0o600,
        ) {
            Ok(mut created) => {
                let data = random_hex(32)? + "\n";
                created
                    .write_all(data.as_bytes())
                    .map_err(|error| ServerError::StateIo {
                        reason: "cannot initialize API key",
                        source: Some(error),
                    })?;
                sysfd::fsync(created.as_fd())
                    .and_then(|()| sysfd::fsync(directory.as_fd()))
                    .map_err(|error| ServerError::StateIo {
                        reason: "cannot persist API key",
                        source: Some(error),
                    })?;
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(ServerError::StateIo {
                    reason: "cannot create private API key",
                    source: Some(error),
                });
            }
        }
        let mut input = sysfd::openat(
            directory.as_fd(),
            c"api-key",
            sysfd::O_RDONLY | sysfd::O_NOFOLLOW | sysfd::O_CLOEXEC,
            0,
        )
        .map_err(|error| ServerError::StateIo {
            reason: KEY_OPEN_REFUSED,
            source: Some(error),
        })?;
        let key_state = input.metadata().map_err(|error| ServerError::StateIo {
            reason: KEY_OPEN_REFUSED,
            source: Some(error),
        })?;
        if !(key_state.is_file()
            && key_state.nlink() == 1
            && owned_mode(&key_state, 0o600)
            && key_state.size() == 65)
        {
            return Err(ServerError::StateIo {
                reason: KEY_OPEN_REFUSED,
                source: None,
            });
        }
        let mut bytes = [0u8; 65];
        input
            .read_exact(&mut bytes)
            .map_err(|error| ServerError::StateIo {
                reason: "cannot read API key",
                source: Some(error),
            })?;
        let key = &bytes[..64];
        require(
            bytes[64] == b'\n' && key.iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')),
            "malformed API key",
        )?;
        let key = key.iter().copied().map(char::from).collect();
        Ok(Self {
            _directory: directory,
            _lock: lock,
            key,
            prefix_trace: None,
            prefix_trace_bytes: 0,
            prefix_trace_refusal: None,
            telemetry_trace: None,
            telemetry_trace_bytes: 0,
        })
    }
}

fn template_data(path: &Path) -> Result<Vec<u8>> {
    let file = MappedFile::new(path)?;
    require(
        file.size() != 0 && file.size() <= 65536,
        "template byte bound",
    )?;
    let bytes = file.bytes()?;
    require(
        hex(&sha256(&bytes)) == TEMPLATE_PIN,
        "template pin mismatch",
    )?;
    file.assert_unchanged()?;
    Ok(bytes)
}

fn error_json(message: &[u8], kind: &str) -> Json {
    let message = &message[..message.len().min(2048)];
    Json::new_object([(
        "error",
        Json::new_object([
            ("message", Json::from(message)),
            ("type", Json::from(kind)),
            ("code", Json::from(kind)),
        ]),
    )])
}

struct Options {
    worker: WorkerOptions,
    tokenizer: PathBuf,
    model_template: PathBuf,
    state: PathBuf,
    port: u16,
    request_seconds: u32,
    owner_lifetime_stdin: bool,
    alias: String,
    prefix_trace: bool,
    telemetry_stdout: bool,
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
    mtp_first_output_packet: Option<u32>,
}

fn integer(text: &str, low: u32, high: u32) -> Result<u32> {
    match from_chars_u32(text) {
        Some(value) if value >= low && value <= high => Ok(value),
        _ => Err(ServerError::Refused("server option outside bound")),
    }
}

fn options(args: &[OsString]) -> Result<Options> {
    let mut out = Options {
        worker: WorkerOptions::default(),
        tokenizer: PathBuf::new(),
        model_template: PathBuf::new(),
        state: PathBuf::new(),
        port: 18081,
        request_seconds: 3600,
        owner_lifetime_stdin: false,
        alias: "tokspur".to_owned(),
        prefix_trace: false,
        telemetry_stdout: false,
        tool_enums: ToolEnumValidation::Strict,
        tool_args: ToolArgValidation::Strict,
        mtp_first_output_packet: None,
    };
    let mut seen = BTreeSet::new();
    let mut allow = false;
    let mut i = 1;
    while i < args.len() {
        let raw = args[i].as_bytes();
        if !raw.starts_with(b"--") {
            out.worker.shards.push(PathBuf::from(&args[i]));
            i += 1;
            continue;
        }
        i += 1;
        require(
            seen.insert(raw.to_vec()) && i < args.len(),
            "duplicate server option or missing value",
        )?;
        // Option names and string values are compared as bytes in C++; text that is not UTF-8 matches no option and fails every value check.
        let option = std::str::from_utf8(raw).unwrap_or("");
        let path = PathBuf::from(&args[i]);
        let value = args[i].to_str().unwrap_or("\u{fffd}");
        match option {
            "--worker" => out.worker.executable = path.clone(),
            "--worker-burst-pacing" => {
                require(
                    matches!(value, "off" | "on"),
                    "--worker-burst-pacing requires off or on",
                )?;
                out.worker.burst_pacing = value == "on";
            }
            "--derived-pack" => {
                require(!value.is_empty(), "derived pack path cannot be empty")?;
                out.worker.derived_pack = Some(path.clone());
            }
            "--derived-pack-auth-workers" => {
                out.worker.derived_pack_auth_workers = Some(integer(value, 1, 16)?);
            }
            "--derived-pack-trust-mode" => {
                require(
                    value == "full" || value == "local-seal",
                    "derived pack trust mode must be full or local-seal",
                )?;
                out.worker.derived_pack_trust_mode = Some(value.to_owned());
            }
            "--native-scalar-channel" => {
                require(
                    value == "on" || value == "off",
                    "private native-scalar-channel must be off or on",
                )?;
                out.worker.scalar_channel = value == "on";
            }
            "--native-mtp-sidecar" => {
                require(!value.is_empty(), "native MTP sidecar cannot be empty")?;
                out.worker.mtp_sidecar = path.clone();
            }
            "--drafter" | "--native-drafter" => {
                require(
                    value == "mtp" || value == "dflash2",
                    "drafter must be mtp or dflash2",
                )?;
                out.worker.drafter = Some(value.to_owned());
            }
            "--dflash2-dir" | "--native-dflash2-dir" => {
                require(!value.is_empty(), "DFlash2 directory cannot be empty")?;
                out.worker.dflash2_dir = path.clone();
            }
            "--dflash2-precision" | "--native-dflash2-precision" => {
                require(
                    value == "q8_0" || value == "bf16",
                    "DFlash2 precision must be q8_0 or bf16",
                )?;
                out.worker.dflash2_precision = Some(value.to_owned());
            }
            "--native-mtp-depth" | "--mtp-depth" => {
                let depth = integer(value, 1, 7)?;
                require(
                    depth == 1 || depth == 2 || depth == 3 || depth == 7,
                    "native MTP depth must be 1, 2, 3 or 7",
                )?;
                out.worker.mtp_depth = Some(depth);
            }
            "--native-mtp-request-carry" => {
                require(
                    value == "on" || value == "off",
                    "native MTP request carry must be off or on",
                )?;
                out.worker.mtp_request_carry = value == "on";
            }
            "--native-mtp-draft-attention" => {
                require(
                    value == "serial"
                        || value == "tiled128"
                        || value == "tiled256"
                        || value == "split16",
                    "unsupported native MTP draft-attention mode",
                )?;
                out.worker.mtp_draft_attention = Some(value.to_owned());
            }
            "--native-mtp-output-packet" => {
                require(
                    value == "8" || value == "64",
                    "native MTP output packet must be 8 or 64",
                )?;
                out.worker.mtp_output_packet = Some(if value == "8" { 8 } else { 64 });
            }
            "--native-mtp-first-output-packet" | "--first-output-packet" => {
                out.mtp_first_output_packet = Some(integer(value, 1, 64)?);
            }
            "--native-qsa-sparse" => {
                require(
                    value == "on" || value == "off",
                    "native sparse QSA must be off or on",
                )?;
                out.worker.qsa_sparse = Some(value == "on");
            }
            "--native-gdn-ported" => {
                require(
                    value == "on" || value == "off",
                    "native ported GDN must be off or on",
                )?;
                out.worker.gdn_ported = Some(value == "on");
            }
            "--native-gdn-prefill-ported" => {
                require(
                    value == "on" || value == "off",
                    "native prefill-only ported GDN must be off or on",
                )?;
                out.worker.gdn_prefill_ported = Some(value == "on");
            }
            "--native-prefill-input512" => {
                require(
                    value == "on" || value == "off",
                    "native input512 must be off or on",
                )?;
                out.worker.prefill_input512 = value == "on";
            }
            "--native-prefill-dense" => {
                require(
                    value == "f16" || value == "f32" || value == "f32-staged" || value == "nax",
                    "native prefill dense must select a compiled f16/f32/nax backend",
                )?;
                out.worker.prefill_dense = Some(value.to_owned());
            }
            "--native-prefill-grouped" => {
                require(
                    value == "serial" || value == "f32" || value == "auto" || value == "nax",
                    "native prefill grouped must be serial, f32, auto or nax",
                )?;
                out.worker.prefill_grouped = Some(value.to_owned());
            }
            "--native-prefill-hc-rows" => {
                require(
                    value == "on" || value == "off",
                    "native HC rows must be off or on",
                )?;
                out.worker.prefill_hc_rows = Some(value == "on");
            }
            "--native-prefill-router-window" => {
                require(
                    value == "on" || value == "off",
                    "native router window must be off or on",
                )?;
                out.worker.prefill_router_window = Some(value == "on");
            }
            "--native-prefill-hc-down-window" => {
                require(
                    value == "on" || value == "off",
                    "native HC down window must be off or on",
                )?;
                out.worker.prefill_hc_down_window = Some(value == "on");
            }
            "--native-prefill-hc-down-f32" => {
                require(
                    value == "on" || value == "off",
                    "native HC down batching must be off or on",
                )?;
                out.worker.prefill_hc_down_f32 = Some(value == "on");
            }
            "--native-prefill-final-logits" => {
                require(
                    value == "on" || value == "off",
                    "native final-input logits must be off or on",
                )?;
                out.worker.prefill_final_logits = value == "on";
            }
            "--native-gdn-old-order-batch" => {
                require(
                    value == "on" || value == "off",
                    "native old-order GDN must be off or on",
                )?;
                out.worker.gdn_old_order_batch = value == "on";
            }
            "--native-q8-three-rows" => {
                require(
                    value == "on" || value == "off",
                    "native three-row Q8 must be on or off",
                )?;
                out.worker.q8_three_rows = Some(value == "on");
            }
            "--native-concurrent-dispatch" => {
                require(
                    value == "on" || value == "off",
                    "native concurrent dispatch must be off or on",
                )?;
                out.worker.concurrent_dispatch = Some(value == "on");
            }
            "--native-raw-decode" | "--raw-decode" => {
                require(
                    value == "on" || value == "off",
                    "native raw-decode must be off or on",
                )?;
                out.worker.raw_decode = Some(value == "on");
            }
            "--native-mtp-draft-slice" => {
                require(
                    value == "on" || value == "off",
                    "native MTP draft slice must be off or on",
                )?;
                out.worker.mtp_draft_slice = Some(value == "on");
            }
            "--native-verify-nax" | "--verify-nax" => {
                require(
                    value == "on" || value == "off",
                    "native verify NAX must be off or on",
                )?;
                out.worker.verify_nax = Some(value == "on");
            }
            "--native-verify-nax-m8" | "--verify-nax-m8" => {
                require(
                    value == "on" || value == "off",
                    "native verify NAX m8 must be off or on",
                )?;
                out.worker.verify_nax_m8 = Some(value == "on");
            }
            "--native-verify-hc-gdn-fused" | "--verify-hc-gdn-fused" => {
                require(
                    value == "on" || value == "off",
                    "native verify HC+GDN fused must be off or on",
                )?;
                out.worker.verify_hc_gdn_fused = Some(value == "on");
            }
            "--native-host-logit-views" => {
                require(
                    value == "on" || value == "off",
                    "native host logit views must be off or on",
                )?;
                out.worker.host_logit_views = Some(value == "on");
            }
            "--native-logit-topk" | "--logit-topk" => {
                require(
                    value == "on" || value == "off",
                    "native logit top-k must be off or on",
                )?;
                out.worker.logit_topk = Some(value == "on");
            }
            "--native-ple-async-gather" | "--ple-async-gather" => {
                require(
                    value == "on" || value == "off",
                    "native PLE async gather must be off or on",
                )?;
                out.worker.ple_async_gather = Some(value == "on");
            }
            "--native-ple-ram-resident" | "--ple-ram-resident" => {
                require(
                    value == "on" || value == "off",
                    "native PLE RAM residency must be off or on",
                )?;
                out.worker.ple_ram_resident = Some(value == "on");
            }
            "--native-mtp-fast-round" | "--mtp-fast-round" => {
                require(
                    value == "on" || value == "off",
                    "native MTP fast round must be off or on",
                )?;
                out.worker.mtp_fast_round = Some(value == "on");
            }
            "--native-dflash2-defer-context" => {
                require(
                    value == "on" || value == "off",
                    "native DFlash2 deferred context must be off or on",
                )?;
                out.worker.dflash2_defer_context = Some(value == "on");
            }
            "--native-prepared-validation" => {
                require(
                    value == "on" || value == "off",
                    "native prepared validation must be off or on",
                )?;
                out.worker.prepared_validation = value == "on";
            }
            "--native-verify-cost-profile" => {
                require(!value.is_empty(), "empty measured cost profile")?;
                out.worker.verify_cost_profile = Some(PathBuf::from(value));
            }
            "--native-verify-m8-splits" => {
                require(!value.is_empty(), "empty M8 split policy")?;
                out.worker.verify_m8_splits = Some(PathBuf::from(value));
            }
            "--native-dflash2-draft-policy" => {
                require(
                    value == "always" || value == "auto",
                    "native DFlash2 draft policy must be always or auto",
                )?;
                out.worker.dflash2_draft_policy = value.to_owned();
            }
            "--native-dflash2-verify-rows" | "--dflash2-verify-rows" => {
                require(
                    value == "all" || value == "auto",
                    "native DFlash2 verify rows must be all or auto",
                )?;
                out.worker.dflash2_verify_rows = Some(value.to_owned());
            }
            "--native-dflash2-max-drafts" | "--dflash2-max-drafts" => {
                let md = integer(value, 1, 7)? as u32;
                out.worker.dflash2_max_drafts = Some(md);
            }
            "--native-dflash2-vocab-slice" | "--dflash2-vocab-slice" => {
                require(
                    out.worker.dflash2_vocab_gather.is_none(),
                    "cannot combine --dflash2-vocab-slice and --dflash2-vocab-gather",
                )?;
                let slice = if value == "full" {
                    0u32
                } else {
                    integer(value, 0, 248_319)? as u32
                };
                require(
                    slice == 0 || (slice >= 256 && slice.is_multiple_of(32)),
                    "DFlash2 vocab slice must be 0, full, or a multiple of 32 in [256, 248319]",
                )?;
                out.worker.dflash2_vocab_slice = Some(slice);
            }
            "--native-dflash2-vocab-gather" | "--dflash2-vocab-gather" => {
                require(
                    out.worker.dflash2_vocab_slice.is_none(),
                    "cannot combine --dflash2-vocab-slice and --dflash2-vocab-gather",
                )?;
                tokspur_wire::worker::validate_dflash2_vocab_gather(value, 248_320)
                    .map_err(ServerError::Refused)?;
                out.worker.dflash2_vocab_gather = Some(value.to_owned());
            }
            "--native-known-token-capacity" | "--known-token-capacity" => {
                let cap = integer(value, 0, 2048)? as u32;
                require(
                    cap == 0 || cap == 8 || cap == 512 || cap == 1024 || cap == 2048,
                    "known-token-capacity must be 0, 8, 512, 1024, or 2048",
                )?;
                out.worker.known_token_capacity = Some(cap);
            }
            "--tool-enum-validation" => {
                out.tool_enums = parse_tool_enum_validation(value)?;
            }
            "--tool-arg-validation" => {
                out.tool_args = parse_tool_arg_validation(value)?;
            }
            #[cfg(feature = "fast-math")]
            "--native-fast-math" => {
                require(
                    value == "staged" || value == "channel",
                    "private native-fast-math must be staged or channel",
                )?;
                out.worker.fast_channel = Some(value == "channel");
            }
            #[cfg(feature = "verify8")]
            "--native-verify8" => {
                require(
                    value == "off" || value == "on",
                    "private native-verify8 must be off or on",
                )?;
                out.worker.verify8 = value == "on";
            }
            #[cfg(feature = "branch8")]
            "--native-branch8" => {
                require(
                    value == "off" || value == "on",
                    "private native-branch8 must be off or on",
                )?;
                out.worker.branch8 = value == "on";
            }
            #[cfg(feature = "qsa-stage-v8")]
            "--native-qsa-stage-v8" => {
                require(
                    value == "off" || value == "on",
                    "private native-qsa-stage-v8 must be off or on",
                )?;
                out.worker.stage8_present = true;
                out.worker.stage8_enabled = value == "on";
            }
            "--stack-execution" => {
                require(
                    value == "direct" || value == "recurrent-replay",
                    "unsupported serving startup execution mode",
                )?;
                out.worker.stack_execution = value.to_owned();
            }
            "--policy" => out.worker.policy = path.clone(),
            "--library" => out.worker.library = path.clone(),
            "--checkpoint-slots" => out.worker.checkpoint_slots = integer(value, 0, 4)?,
            "--checkpoint-budget-bytes" => {
                out.worker.checkpoint_budget_bytes = u64::from(integer(value, 0, 512 << 20)?)
            }
            "--tokenizer" => out.tokenizer = path.clone(),
            "--template" => out.model_template = path.clone(),
            "--state-dir" => out.state = path.clone(),
            "--model-alias" => {
                require(
                    !value.is_empty()
                        && value.len() <= 64
                        && value
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')),
                    "invalid model alias",
                )?;
                out.alias = value.to_owned();
            }
            "--port" => out.port = integer(value, 0, 65535)? as u16,
            "--owner-lifetime-stdin" => {
                require(
                    matches!(value, "on" | "off"),
                    "owner lifetime requires on/off",
                )?;
                out.owner_lifetime_stdin = value == "on";
            }
            "--request-timeout-seconds" => out.request_seconds = integer(value, 1, 86400)?,
            "--startup-timeout-seconds" => {
                out.worker.startup_timeout =
                    Duration::from_secs(u64::from(integer(value, 1, 3600)?));
            }
            "--step-timeout-seconds" => {
                out.worker.step_timeout = Duration::from_secs(u64::from(integer(value, 1, 300)?));
            }
            "--allow-full-model" => {
                require(
                    value == "true",
                    "server requires explicit full-model consent",
                )?;
                allow = true;
            }
            "--private-prefix-trace" => {
                require(
                    value == "off" || value == "on",
                    "private-prefix-trace must be off or on",
                )?;
                out.prefix_trace = value == "on";
            }
            "--telemetry-stdout" => {
                require(
                    value == "off" || value == "on",
                    "telemetry-stdout must be off or on",
                )?;
                out.telemetry_stdout = value == "on";
            }
            "--checkpoint-ssd" => {
                require(
                    value == "off" || value == "on" || value == "true" || value == "false",
                    "checkpoint-ssd must be off or on",
                )?;
                out.worker.checkpoint_ssd = value == "on" || value == "true";
            }
            "--checkpoint-cap-gib" | "--checkpoint-ssd-budget-gib" => {
                let gib: f64 = value
                    .parse()
                    .map_err(|_| ServerError::Refused("invalid SSD budget GiB"))?;
                require(
                    (0.0..=32.0).contains(&gib),
                    "checkpoint SSD budget must be 0.0 to 32.0 GiB",
                )?;
                out.worker.checkpoint_ssd_budget_bytes = (gib * (1024.0 * 1024.0 * 1024.0)) as u64;
            }
            "--checkpoint-floor-gib" | "--checkpoint-ssd-floor-gib" => {
                let gib: f64 = value
                    .parse()
                    .map_err(|_| ServerError::Refused("invalid SSD floor GiB"))?;
                require(
                    (0.0..=100.0).contains(&gib),
                    "checkpoint SSD floor must be 0.0 to 100.0 GiB",
                )?;
                out.worker.checkpoint_ssd_floor_bytes = (gib * (1024.0 * 1024.0 * 1024.0)) as u64;
            }
            "--checkpoint-dir" | "--checkpoint-ssd-dir" => {
                require(
                    !value.is_empty(),
                    "checkpoint SSD directory cannot be empty",
                )?;
                out.worker.checkpoint_ssd_dir = PathBuf::from(value);
            }
            "--checkpoint-policy" => {
                require(
                    value == "recompute-cost" || value == "lru",
                    "checkpoint policy must be recompute-cost or lru",
                )?;
                out.worker.checkpoint_policy = value.to_owned();
            }
            "--overlap-prefill" => {
                require(
                    value == "off" || value == "on" || value == "true" || value == "false",
                    "overlap-prefill must be off or on",
                )?;
                out.worker.overlap_prefill = value == "on" || value == "true";
            }
            "--native-swiglu-fused" | "--swiglu-fused" => {
                require(
                    value == "off" || value == "on",
                    "native SwiGLU fusion must be off or on",
                )?;
                out.worker.swiglu_fused = Some(value == "on");
            }
            "--native-q8-gemv-rg4" | "--q8-gemv-rg4" => {
                require(
                    value == "off" || value == "on" || value == "wide",
                    "native 4-row/SG Q8 GEMV must be off, on or wide",
                )?;
                out.worker.q8_gemv_rg4 = Some(value.to_owned());
            }
            "--native-q8-three-msg" | "--q8-three-msg" => {
                require(
                    value == "off" || value == "on" || value == "wide",
                    "native multi-SG three-row Q8 must be off, on or wide",
                )?;
                out.worker.q8_three_msg = Some(value.to_owned());
            }
            "--native-moe-three-union" | "--moe-three-union" => {
                require(
                    value == "off" || value == "on",
                    "native unioned three-row MoE must be off or on",
                )?;
                out.worker.moe_three_union = Some(value == "on");
            }
            "--native-mtp-lookup" | "--mtp-lookup" => {
                require(
                    value == "off" || value == "bypass" || value == "hybrid",
                    "native MTP lookup must be off, bypass or hybrid",
                )?;
                out.worker.mtp_lookup = Some(value.to_owned());
            }
            "--native-mtp-lookup-match" | "--mtp-lookup-match" => {
                out.worker.mtp_lookup_match = Some(integer(value, 2, 16)?);
            }
            "--native-dflash2-lookup" | "--dflash2-lookup" => {
                require(
                    value == "off" || value == "bypass" || value == "hybrid",
                    "native DFlash2 lookup must be off, bypass or hybrid",
                )?;
                out.worker.dflash2_lookup = Some(value.to_owned());
            }
            "--native-dflash2-lookup-match" | "--dflash2-lookup-match" => {
                out.worker.dflash2_lookup_match = Some(integer(value, 2, 16)?);
            }
            "--native-prefill-m64" | "--prefill-m64" => {
                require(
                    value == "off" || value == "on" || value == "short",
                    "native prefill M64 must be off, on or short",
                )?;
                out.worker.prefill_m64 = Some(value.to_owned());
            }
            "--native-prefill-multiseg" | "--prefill-multiseg" => {
                require(
                    value == "off" || value == "on",
                    "native prefill multiseg must be off or on",
                )?;
                out.worker.prefill_multiseg = Some(value == "on");
            }
            "--native-prefill-swizzle" | "--prefill-swizzle" => {
                require(
                    value == "off" || value == "on",
                    "native prefill swizzle must be off or on",
                )?;
                out.worker.prefill_swizzle = Some(value == "on");
            }
            "--native-prefill-router-fast" | "--prefill-router-fast" => {
                require(
                    value == "off" || value == "on",
                    "native prefill router fast must be off or on",
                )?;
                out.worker.prefill_router_fast = Some(value == "on");
            }
            "--native-qsa-prefill-fused" | "--qsa-prefill-fused" => {
                require(
                    value == "off" || value == "on",
                    "native fused QSA prefill must be off or on",
                )?;
                out.worker.qsa_prefill_fused = Some(value == "on");
            }
            "--native-hc-consumer-fused" | "--hc-consumer-fused" => {
                require(
                    value == "off" || value == "on",
                    "native fused HC consumer must be off or on",
                )?;
                out.worker.hc_consumer_fused = Some(value == "on");
            }
            "--native-ple-overlap" | "--ple-overlap" => {
                require(
                    value == "off" || value == "on",
                    "native PLE overlap must be off or on",
                )?;
                out.worker.ple_overlap = Some(value == "on");
            }
            "--shared-prefix-cache" | "--native-shared-prefix-cache" => {
                require(
                    value == "off" || value == "on",
                    "shared prefix cache must be off or on",
                )?;
                out.worker.shared_prefix_cache = Some(value == "on");
            }
            "--parked-lanes-mib" | "--native-parked-lanes-mib" => {
                out.worker.parked_lanes_mib = integer(value, 0, 16_384)? as u32;
            }
            "--context-length" | "--native-context-length" => {
                let ctx = integer(value, 65_536, 131_072)?;
                require(
                    ctx == 65_536 || ctx == 131_072,
                    "context-length must be 65536 or 131072",
                )?;
                out.worker.context_length = ctx;
            }
            _ => return Err(ServerError::Refused("unknown server option")),
        }
        i += 1;
    }
    require(
        allow
            && out.tokenizer.is_absolute()
            && out.model_template.is_absolute()
            && !out.state.as_os_str().is_empty(),
        "server requires consent, absolute tokenizer/template/state and owned worker assets",
    )?;
    require(
        out.worker.mtp_sidecar.as_os_str().is_empty()
            || out.worker.dflash2_dir.as_os_str().is_empty(),
        "cannot combine MTP sidecar and DFlash2 drafter",
    )?;
    require(
        out.worker.drafter.as_deref() != Some("dflash2")
            || !out.worker.dflash2_dir.as_os_str().is_empty(),
        "dflash2 drafter requires --dflash2-dir",
    )?;
    require(
        out.worker.drafter.as_deref() != Some("mtp")
            || out.worker.dflash2_dir.as_os_str().is_empty(),
        "mtp drafter refuses --dflash2-dir",
    )?;
    if !out.worker.dflash2_dir.as_os_str().is_empty() {
        if out.worker.mtp_depth.is_none() {
            out.worker.mtp_depth = Some(7);
        }
        if out.worker.mtp_draft_attention.is_none() {
            out.worker.mtp_draft_attention = Some("split16".to_owned());
        }
        out.worker.mtp_request_carry = true;
    }
    if out.worker.checkpoint_ssd_budget_bytes == 0 {
        out.worker.checkpoint_ssd = false;
    }
    let w = &out.worker;
    require(
        (w.checkpoint_slots != 0) == (w.checkpoint_budget_bytes != 0)
            && (w.checkpoint_slots == 0 || w.stack_execution == "direct"),
        "HTTP broker requires paired bounded direct checkpoint preparation",
    )?;
    let scalar_transport = tokspur_wire::worker::scalar_channel_transport(w);
    let concurrent_transport = tokspur_wire::worker::owned_concurrent_transport(w);
    let has_spec = tokspur_wire::worker::has_speculative(w);
    require(
        !w.scalar_channel || scalar_transport,
        "private scalar-channel requires explicit channel/Stage-ON direct execution without V8/branch/history",
    )?;
    require(
        !has_spec
            || (scalar_transport
                && (w.mtp_sidecar.as_os_str().is_empty() || w.mtp_sidecar.is_absolute())
                && (w.dflash2_dir.as_os_str().is_empty() || w.dflash2_dir.is_absolute())),
        "native MTP/DFlash2 requires its distinct explicit fixed channel consent and absolute asset path",
    )?;
    require(
        !w.prefill_input512 || concurrent_transport,
        "input512 requires explicit fixed depth-two MTP scalar transport",
    )?;
    require(
        !w.prefill_final_logits || w.prefill_input512,
        "native final-input logits require input512",
    )?;
    require(
        w.prefill_hc_rows.is_none() || concurrent_transport,
        "HC rows require fixed MTP transport",
    )?;
    let dense_ok = matches!(w.prefill_dense.as_deref(), Some("f32-staged" | "nax"));
    let grouped_ok = matches!(w.prefill_grouped.as_deref(), Some("f32" | "auto" | "nax"));
    require(
        !w.prefill_hc_rows.unwrap_or(false)
            || (w.prefill_input512
                && dense_ok
                && grouped_ok
                && !w.prefill_hc_down_f32.unwrap_or(false)),
        "HC rows require staged FP32 or NAX input512 and scalar HC down",
    )?;
    require(
        w.prefill_router_window.is_none() || concurrent_transport,
        "router window require fixed MTP transport",
    )?;
    require(
        !w.prefill_router_window.unwrap_or(false)
            || (w.prefill_input512
                && dense_ok
                && grouped_ok
                && !w.prefill_hc_down_f32.unwrap_or(false)),
        "router window requires staged FP32 or NAX input512 and scalar HC down",
    )?;
    require(
        !w.prefill_router_fast.unwrap_or(false)
            || (w.prefill_router_window.unwrap_or(false)
                && (w.prefill_dense.as_deref() == Some("nax")
                    || w.prefill_grouped.as_deref() == Some("nax"))),
        "fast BF16 prefill router requires router window and NAX prefill",
    )?;
    require(
        w.prefill_hc_down_window.is_none() || concurrent_transport,
        "HC down window requires fixed MTP transport",
    )?;
    require(
        !w.prefill_hc_down_window.unwrap_or(false)
            || (w.prefill_input512
                && dense_ok
                && grouped_ok
                && w.prefill_hc_rows.unwrap_or(false)
                && w.prefill_router_window.unwrap_or(false)
                && !w.prefill_hc_down_f32.unwrap_or(false)),
        "HC down window requires staged FP32 or NAX input512, exact HC rows, exact router window and reassociating HC down off",
    )?;
    require(
        w.qsa_sparse.is_none() || concurrent_transport,
        "sparse QSA requires fixed MTP transport",
    )?;
    require(
        !w.qsa_sparse.unwrap_or(false)
            || (w.prefill_input512
                && dense_ok
                && grouped_ok
                && w.prefill_hc_rows.unwrap_or(false)
                && w.prefill_router_window.unwrap_or(false)
                && w.prefill_hc_down_window.unwrap_or(false)
                && !w.prefill_hc_down_f32.unwrap_or(false)),
        "sparse QSA requires staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off",
    )?;
    require(
        w.gdn_ported.is_none() || concurrent_transport,
        "ported GDN requires fixed MTP transport",
    )?;
    require(
        !w.gdn_ported.unwrap_or(false)
            || (w.prefill_input512
                && dense_ok
                && grouped_ok
                && w.prefill_hc_rows.unwrap_or(false)
                && w.prefill_router_window.unwrap_or(false)
                && w.prefill_hc_down_window.unwrap_or(false)
                && !w.prefill_hc_down_f32.unwrap_or(false)),
        "ported GDN requires staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off",
    )?;
    require(
        w.gdn_prefill_ported.is_none() || concurrent_transport,
        "prefill-only ported GDN requires fixed MTP transport",
    )?;
    require(
        !w.gdn_prefill_ported.unwrap_or(false)
            || (!w.gdn_ported.unwrap_or(false)
                && w.prefill_input512
                && dense_ok
                && grouped_ok
                && w.prefill_hc_rows.unwrap_or(false)
                && w.prefill_router_window.unwrap_or(false)
                && w.prefill_hc_down_window.unwrap_or(false)
                && !w.prefill_hc_down_f32.unwrap_or(false)),
        "prefill-only ported GDN requires --native-gdn-ported off, staged FP32 or NAX input512, exact HC rows, exact router window, exact HC down window and reassociating HC down off",
    )?;
    require(
        w.prefill_hc_down_f32.is_none() || concurrent_transport,
        "native HC down selection requires fixed MTP transport",
    )?;
    require(
        !w.prefill_hc_down_f32.unwrap_or(false)
            || (w.prefill_input512
                && w.prefill_dense.as_deref() == Some("f32")
                && matches!(w.prefill_grouped.as_deref(), Some("f32" | "auto"))),
        "native HC down batching requires explicit f32/f32 input512",
    )?;
    require(
        !w.gdn_old_order_batch
            || (w.prefill_input512
                && (w.prefill_dense.as_deref() == Some("f32")
                    || w.prefill_dense.as_deref() == Some("f32-staged"))
                && matches!(w.prefill_grouped.as_deref(), Some("f32" | "auto"))),
        "native old-order GDN requires explicit f32/f32 input512",
    )?;
    if dense_ok {
        require(
            w.prefill_input512 && grouped_ok && !w.prefill_hc_down_f32.unwrap_or(false),
            "staged/NAX dense backend requires input512 compatible grouped mode and HC down off",
        )?;
    }
    if w.prefill_grouped.as_deref() == Some("nax") {
        require(
            w.prefill_input512 && dense_ok && !w.prefill_hc_down_f32.unwrap_or(false),
            "NAX grouped prefill requires staged/NAX dense and HC down off",
        )?;
    }
    require(
        w.prefill_dense.is_none() || w.prefill_input512,
        "native prefill dense requires input512",
    )?;
    require(
        w.prefill_grouped.is_none() || w.prefill_input512,
        "native prefill grouped requires input512",
    )?;
    require(
        w.dflash2_precision.is_none() || !w.dflash2_dir.as_os_str().is_empty(),
        "DFlash2 precision requires --dflash2-dir",
    )?;
    require(
        w.dflash2_max_drafts.is_none() || !w.dflash2_dir.as_os_str().is_empty(),
        "DFlash2 max drafts requires --dflash2-dir",
    )?;
    require(
        w.dflash2_vocab_slice.is_none() || !w.dflash2_dir.as_os_str().is_empty(),
        "DFlash2 vocab slice requires --dflash2-dir",
    )?;
    require(
        w.dflash2_vocab_gather.is_none() || !w.dflash2_dir.as_os_str().is_empty(),
        "DFlash2 vocab gather requires --dflash2-dir",
    )?;
    require(
        !(w.prepared_validation
            || w.verify_cost_profile.is_some()
            || w.verify_m8_splits.is_some()
            || w.dflash2_draft_policy == "auto")
            || concurrent_transport,
        "runtime checking controls require the native MTP carrier",
    )?;
    require(
        w.dflash2_draft_policy != "auto"
            || (w.verify_cost_profile.is_some()
                && !w.dflash2_dir.as_os_str().is_empty()
                && w.dflash2_verify_rows.as_deref() == Some("auto")),
        "automatic drafting requires a compatible DFlash2 profile and automatic rows",
    )?;
    require(
        w.verify_m8_splits.is_none() || w.verify_nax_m8 == Some(true),
        "M8 split override requires explicit M8 verification",
    )?;
    require(
        !w.logit_topk.unwrap_or(false) || has_spec,
        "native logit top-k requires a speculative drafter",
    )?;
    require(
        !w.ple_async_gather.unwrap_or(false) || has_spec,
        "native PLE async gather requires a speculative drafter",
    )?;
    require(
        !w.ple_ram_resident.unwrap_or(false) || has_spec,
        "native PLE RAM residency requires a speculative drafter",
    )?;
    require(
        !w.mtp_fast_round.unwrap_or(false) || !w.mtp_sidecar.as_os_str().is_empty(),
        "native MTP fast round requires --mtp-sidecar",
    )?;
    require(
        !w.verify_hc_gdn_fused.unwrap_or(false)
            || (has_spec
                && (w.verify_nax == Some(true)
                    || (!w.dflash2_dir.as_os_str().is_empty() && w.verify_nax != Some(false)))),
        "native verify HC+GDN fused requires --native-verify-nax on",
    )?;
    require(
        w.mtp_depth.is_none()
            || (has_spec
                && if !w.dflash2_dir.as_os_str().is_empty() {
                    w.mtp_depth == Some(7)
                        && w.mtp_request_carry
                        && w.mtp_draft_attention.as_deref() == Some("split16")
                } else {
                    w.mtp_depth == Some(1)
                        || (matches!(w.mtp_depth, Some(2 | 3))
                            && w.mtp_request_carry
                            && w.mtp_draft_attention.as_deref() == Some("tiled128"))
                }),
        "explicit speculative HTTP depth requires matching scoped carry/attention",
    )?;
    require(
        !w.mtp_request_carry || has_spec,
        "native MTP request carry requires its sidecar or DFlash2 directory",
    )?;
    require(
        w.concurrent_dispatch.is_none() || concurrent_transport,
        "concurrent HTTP requires explicit fixed depth-two/seven carry scalar transport",
    )?;
    require(
        w.mtp_output_packet.is_none() || concurrent_transport,
        "output-packet selection requires explicit fixed depth-two/seven carry scalar transport",
    )?;
    require(
        out.mtp_first_output_packet.is_none() || concurrent_transport,
        "first-output-packet selection requires explicit fixed depth-two/seven carry scalar transport",
    )?;
    if let Some(first_pkt) = out.mtp_first_output_packet {
        require(
            first_pkt <= w.mtp_output_packet.unwrap_or(8),
            "first-output-packet cannot exceed native MTP output packet",
        )?;
    }
    require(
        w.q8_three_rows.is_none() || concurrent_transport,
        "three-row Q8 HTTP requires the current fixed MTP transport",
    )?;
    require(
        w.mtp_draft_attention.is_none() || has_spec,
        "native MTP draft-attention requires its sidecar or DFlash2 directory",
    )?;
    #[cfg(feature = "verify8")]
    {
        #[cfg(feature = "branch8")]
        let branch_transport =
            w.branch8 && w.verify8 && w.checkpoint_slots != 0 && w.checkpoint_budget_bytes != 0;
        #[cfg(not(feature = "branch8"))]
        let branch_transport = false;
        #[cfg(feature = "branch8")]
        require(
            !w.branch8 || branch_transport,
            "private native-branch8 requires explicit V8 and prepared history",
        )?;
        #[cfg(feature = "qsa-stage-v8")]
        require(
            !w.stage8_present
                || ((branch_transport || scalar_transport) && w.stack_execution == "direct"),
            "private staged HTTP requires explicit branch8/V8/direct/history consent",
        )?;
        #[cfg(feature = "fast-math")]
        require(
            w.fast_channel.is_none()
                || ((branch_transport || scalar_transport) && w.stage8_present && w.stage8_enabled),
            "private fast HTTP math requires explicit staged-ON branch8/V8/history consent",
        )?;
        require(
            !w.verify8
                || (w.stack_execution == "direct"
                    && (branch_transport
                        || (w.checkpoint_slots == 0 && w.checkpoint_budget_bytes == 0))),
            "private native-verify8 requires direct execution without checkpoints",
        )?;
    }
    // A differently named parent is intentional: the existing target's worker exclusion must continue to reject every other tokspur-native model child.
    let name = args
        .first()
        .and_then(|arg0| Path::new(arg0).file_name().map(std::ffi::OsStr::to_owned));
    require(
        name.is_some_and(|name| name == "tokspur-server"),
        "owned HTTP parent must be named tokspur-server",
    )?;
    Ok(out)
}

/// A protocol line on stdout. iostream failures are silent in C++; with the default `SIGPIPE` a closed pipe ends the process the same way.
fn print_line(value: &Json) -> Result<()> {
    let mut line = value.dump()?;
    line.push(b'\n');
    let mut out = std::io::stdout().lock();
    if out.write_all(&line).and_then(|()| out.flush()).is_err() {
        // `std::cout << ... << std::endl` sets badbit and carries on.
    }
    Ok(())
}

/// Only report a pack format confirmed by the native worker after it opens the weights.
fn pack_startup_message(ready: &Json, requested_pack: bool) -> &'static [u8] {
    const UNAVAILABLE: &[u8] = b"tokspur: weight pack identity unavailable\n";
    let schema = ready.find("derived_pack_schema");
    let recipe = ready.find("derived_pack_recipe");
    match (requested_pack, schema, recipe) {
        (false, Ok(None), Ok(None)) => b"tokspur: loaded raw GGUF weights\n",
        (true, Ok(Some(schema)), Ok(Some(recipe))) => {
            match (schema.unsigned_integer(), recipe.string()) {
                (Ok(2), Ok(b"tokspur-pack-v2")) => b"tokspur: loaded Pack v2 (tokspur-pack-v2)\n",
                (Ok(3), Ok(b"tokspur-pack-v3")) => b"tokspur: loaded Pack v3 (tokspur-pack-v3)\n",
                _ => UNAVAILABLE,
            }
        }
        _ => UNAVAILABLE,
    }
}

struct Job {
    connection: Connection,
    request: Json,
    /// The accept instant; only the opt-in `request_timing` record reads it.
    arrival: Instant,
    deadline: Instant,
}

/// Startup choices the readiness line and routing read.
struct Serving<'a> {
    listener: &'a Listener,
    ready_value: &'a Json,
    telemetry_stdout: bool,
    worker_pid: Option<u32>,
    guardian_pid: Option<u32>,
    alias: &'a str,
    stack_execution: &'a str,
    tool_enums: ToolEnumValidation,
    tool_args: ToolArgValidation,
    mtp_first_output_packet: Option<u32>,
    checkpoint_slots: u32,
    checkpoint_budget: u64,
    context_length: u32,
    #[cfg(feature = "verify8")]
    verify8: bool,
    #[cfg(feature = "branch8")]
    branch8: bool,
    #[cfg(feature = "qsa-stage-v8")]
    stage8_present: bool,
    #[cfg(feature = "fast-math")]
    fast_channel: Option<bool>,
    authorization: &'a str,
    request_seconds: u32,
    available: &'a AtomicBool,
    pending: &'a Mutex<Option<Job>>,
    condition: &'a Condvar,
}

fn json_content_type(value: &str) -> bool {
    let mut parts = value.split(';');
    if !parts
        .next()
        .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("application/json"))
    {
        return false;
    }
    let mut charset = false;
    for parameter in parts {
        let Some((name, value)) = parameter.split_once('=') else {
            return false;
        };
        if !name.trim().eq_ignore_ascii_case("charset") || charset {
            return false;
        }
        charset = true;
        let mut value = value.trim();
        if value.starts_with('"') || value.ends_with('"') {
            let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) else {
                return false;
            };
            value = inner;
        }
        if !value.eq_ignore_ascii_case("utf-8") {
            return false;
        }
    }
    true
}

fn monitor_owner_stdin() -> Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("cli-lifetime".to_owned())
        .spawn(|| {
            let mut input = std::io::stdin().lock();
            let mut byte = [0u8; 1];
            loop {
                match input.read(&mut byte) {
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    _ => break,
                }
            }
            tokspur_sys::request_stop();
            std::thread::sleep(Duration::from_secs(2));
            // A lost owner cannot keep inference running. Kernel descriptor closure lets the existing native guardian stop its exact worker if orderly shutdown has stalled.
            std::process::exit(1);
        })
        .map_err(|source| ServerError::StateIo {
            reason: "cannot monitor CLI lifetime",
            source: Some(source),
        })
}

/// `server_cli`.
pub fn server_cli(args: &[OsString]) -> Result<()> {
    let mut config = options(args)?;
    require(
        tokspur_sys::install_stop_handlers(),
        "cannot arm native server shutdown handlers",
    )?;
    let _owner_lifetime = if config.owner_lifetime_stdin {
        Some(monitor_owner_stdin()?)
    } else {
        None
    };
    config.worker.startup_cancelled = Some(Box::new(tokspur_sys::stop_requested));
    require(
        !tokspur_sys::stop_requested(),
        "startup cancelled before asset preparation",
    )?;
    let state_dir = config.state.clone();
    let telemetry_stdout = config.telemetry_stdout;
    let mut state = PrivateState::new(&config.state)?;
    if config.prefix_trace {
        state.enable_prefix_trace()?;
    }
    let listener = Listener::new(config.port, 4)?; // Port and state are reserved before loading a model.
    let tokenizer = NativeText::new(&config.tokenizer, TOKENIZER_PIN)?;
    let literals = extract_flash_next_literals(&template_data(&config.model_template)?)
        .map_err(|error| ServerError::Chat(Box::new(error)))?;
    let renderer =
        ChatRenderer::new(literals).map_err(|error| ServerError::Chat(Box::new(error)))?;
    tokenizer.verify_gguf(&ModelFiles::new(&config.worker.shards)?)?;
    let marker = |text: &[u8]| -> Result<u32> {
        let ids = tokenizer.encode(text)?;
        require(
            ids.len() == 1 && tokenizer.decode_bytes(&ids, false)? == text,
            "unsupported terminal token identity",
        )?;
        Ok(ids[0])
    };
    let admission = ResponseAdmission {
        vocabulary: tokenizer.vocabulary_size() as u32,
        im_end: marker(b"<|im_end|>")?,
        end_of_text: marker(b"<|endoftext|>")?,
        im_start: marker(b"<|im_start|>")?,
    };
    // Frame markers count only when they arrive as the tokenizer's own added tokens. A tokenizer that lacks any of them keeps the byte-spelled framing.
    let control_ids = tokspur_text::response::CONTROL_MARKERS
        .iter()
        .map(|marker| tokenizer.added_token(marker))
        .collect::<Option<Vec<u32>>>()
        .unwrap_or_default();
    // The constructor arms its independent guardian before exec/model loading.
    require(
        !tokspur_sys::stop_requested(),
        "startup cancelled before model launch",
    )?;
    let stack_execution = config.worker.stack_execution.clone();
    let checkpoint_slots = config.worker.checkpoint_slots;
    let checkpoint_budget = config.worker.checkpoint_budget_bytes;
    let context_length = config.worker.context_length;
    #[cfg(feature = "verify8")]
    let verify8 = config.worker.verify8;
    #[cfg(feature = "branch8")]
    let branch8 = config.worker.branch8;
    #[cfg(feature = "qsa-stage-v8")]
    let stage8_present = config.worker.stage8_present;
    #[cfg(feature = "fast-math")]
    let fast_channel = config.worker.fast_channel;
    let has_mtp_sidecar = tokspur_wire::worker::has_speculative(&config.worker);
    let overlap_prefill = config.worker.overlap_prefill;
    require(
        config.worker.parked_lanes_mib == 0
            || (has_mtp_sidecar && config.worker.shared_prefix_cache == Some(true)),
        "parked lanes require a speculative drafter and the shared prefix cache",
    )?;
    let lane_parking = config.worker.parked_lanes_mib > 0;
    let requested_pack = config.worker.derived_pack.is_some();
    let mut worker = OwnedWorker::new(std::mem::take(&mut config.worker))?;
    let worker_pid = worker.worker_pid();
    let guardian_pid = worker.guardian_pid();
    if let Some(pid) = worker_pid {
        state.write_worker_pid(pid)?;
    }
    validate_worker_ready(
        worker.ready(),
        &stack_execution,
        checkpoint_slots,
        checkpoint_budget,
    )?;
    let ready_value = worker.ready().clone();
    // The readiness protocol on stdout is also consumed by supervisors.
    if std::io::stderr()
        .write_all(pack_startup_message(&ready_value, requested_pack))
        .is_err()
    {
        // An unavailable diagnostic stream does not prevent serving.
    }
    let tokenizer = Arc::new(tokenizer);
    let renderer = Arc::new(renderer);
    let (encoder, decoder) = (Arc::clone(&tokenizer), Arc::clone(&tokenizer));
    let codec = ServeCodec {
        render: Box::new(move |input| {
            renderer
                .render(input)
                .map_err(|error| ServeError::Text(Box::new(error)))
        }),
        encode: Box::new(move |input| {
            encoder
                .encode_rendered(input)
                .map_err(|error| ServeError::Text(Box::new(error)))
        }),
        decode: Box::new(move |ids| {
            decoder
                .decode_bytes(ids, false)
                .map_err(|error| ServeError::Text(Box::new(error)))
        }),
        admission,
        control_ids,
        nonordinary_ids: tokenizer.nonordinary_token_ids(),
    };
    let mut session = ServeSession::new(
        codec,
        SessionOptions {
            alias: config.alias.as_bytes().to_vec(),
            stack_execution: stack_execution.clone(),
            tool_enums: config.tool_enums,
            tool_args: config.tool_args,
            broker_ready: if checkpoint_slots != 0 {
                ready_value.clone()
            } else {
                Json::Null
            },
            #[cfg(feature = "verify8")]
            verify8,
            #[cfg(feature = "verify8")]
            verify_ready: if verify8 {
                ready_value.clone()
            } else {
                Json::Null
            },
            #[cfg(feature = "branch8")]
            branch8,
            mtp_ready: if has_mtp_sidecar {
                ready_value.clone()
            } else {
                Json::Null
            },
            mtp_first_output_packet: config.mtp_first_output_packet,
            overlap_prefill,
            lane_parking,
            context_length,
        },
    )?;
    let shutdown = AtomicBool::new(false);
    let available = AtomicBool::new(true);
    let pending: Mutex<Option<Job>> = Mutex::new(None);
    let condition = Condvar::new();
    let authorization = format!("Bearer {}", state.key);
    let serving = Serving {
        listener: &listener,
        ready_value: &ready_value,
        telemetry_stdout,
        worker_pid: if telemetry_stdout { worker_pid } else { None },
        guardian_pid: if telemetry_stdout { guardian_pid } else { None },
        alias: &config.alias,
        stack_execution: &stack_execution,
        tool_enums: config.tool_enums,
        tool_args: config.tool_args,
        mtp_first_output_packet: config.mtp_first_output_packet,
        checkpoint_slots,
        checkpoint_budget,
        context_length,
        #[cfg(feature = "verify8")]
        verify8,
        #[cfg(feature = "branch8")]
        branch8,
        #[cfg(feature = "qsa-stage-v8")]
        stage8_present,
        #[cfg(feature = "fast-math")]
        fast_channel,
        authorization: &authorization,
        request_seconds: config.request_seconds,
        available: &available,
        pending: &pending,
        condition: &condition,
    };
    let worker_ref = &mut worker;
    let state = Mutex::new(state);
    let state_ref = &state;
    let prefix_trace = config.prefix_trace;
    let failure = std::thread::scope(|scope| {
        let generation = scope.spawn(|| {
            generation_loop(
                worker_ref,
                &mut session,
                &shutdown,
                &available,
                &pending,
                &condition,
                state_ref,
                prefix_trace,
                telemetry_stdout,
            );
        });
        let result = serving.publish_and_accept();
        shutdown.store(true, Ordering::SeqCst);
        condition.notify_one();
        if generation.join().is_err() {
            // A panicking generation thread is std::terminate in C++.
            std::process::abort();
        }
        result
    });
    worker.close()?; // A clean exit is not reported unless the entire owned group drains.
    drop(std::fs::remove_file(state_dir.join("worker.pid")));
    failure?;
    let mut state = state
        .into_inner()
        .map_err(|_| ServerError::Refused("private state is poisoned"))?;
    state._lock.release()?;
    print_line(&Json::new_object([
        ("event", Json::from("end")),
        ("owned_worker_exit_verified", Json::Bool(true)),
        ("tools_executed", Json::Bool(false)),
    ]))
}

#[allow(clippy::too_many_arguments)]
fn generation_loop(
    worker: &mut OwnedWorker,
    session: &mut ServeSession,
    shutdown: &AtomicBool,
    available: &AtomicBool,
    pending: &Mutex<Option<Job>>,
    condition: &Condvar,
    state: &Mutex<PrivateState>,
    prefix_trace: bool,
    telemetry_stdout: bool,
) {
    loop {
        let job;
        {
            let Ok(guard) = pending.lock() else {
                std::process::abort();
            };
            let waited = condition.wait_timeout_while(guard, Duration::from_millis(100), |slot| {
                !shutdown.load(Ordering::SeqCst) && slot.is_none()
            });
            let Ok((mut guard, _)) = waited else {
                std::process::abort();
            };
            if shutdown.load(Ordering::SeqCst) {
                *guard = None;
                break;
            }
            if guard.is_none() {
                let live = worker.alive();
                available.store(live, Ordering::SeqCst);
                if !live {
                    session.invalidate();
                }
                continue;
            }
            job = guard.take();
        }
        if let Some(job) = job {
            run_job(
                worker,
                session,
                job,
                shutdown,
                available,
                state,
                prefix_trace,
                telemetry_stdout,
            );
        }
    }
}

/// Any failure while serving one job; `Display` is the C++ `what()`.
#[derive(Debug)]
enum JobError {
    Refused(&'static str),
    Serve(ServeError),
    Http(HttpError),
    Json(JsonError),
    Server(ServerError),
}

impl fmt::Display for JobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => formatter.write_str(reason),
            Self::Serve(error) => error.fmt(formatter),
            Self::Http(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
            Self::Server(error) => error.fmt(formatter),
        }
    }
}

impl From<ServeError> for JobError {
    fn from(error: ServeError) -> Self {
        Self::Serve(error)
    }
}

impl From<HttpError> for JobError {
    fn from(error: HttpError) -> Self {
        Self::Http(error)
    }
}

impl From<JsonError> for JobError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

impl From<ServerError> for JobError {
    fn from(error: ServerError) -> Self {
        Self::Server(error)
    }
}

/// Whether a job failed on the server's own private prefix trace rather than on the request or the generation.
fn trace_failure(error: &JobError) -> bool {
    matches!(
        error,
        JobError::Serve(ServeError::Admission(
            AdmissionError::PrefixTrace(_) | AdmissionError::TracePersist(_)
        ))
    )
}

/// The error type a failed job reports. A trace failure is the server's fault whether or not generation began.
fn failure_kind(error: &JobError, target_started: bool) -> &'static str {
    if trace_failure(error) {
        "server_error"
    } else if target_started {
        "generation_failed_no_retry"
    } else {
        "invalid_request_error"
    }
}

/// The status of a failed job that has not begun streaming. `expired` is whether the request deadline has passed.
fn failure_status(error: &JobError, expired: bool, target_started: bool) -> i32 {
    if expired {
        408
    } else if trace_failure(error) {
        500
    } else if target_started {
        502
    } else {
        400
    }
}

#[allow(clippy::too_many_arguments)]
fn run_job(
    worker: &mut OwnedWorker,
    session: &mut ServeSession,
    job: Job,
    shutdown: &AtomicBool,
    available: &AtomicBool,
    state: &Mutex<PrivateState>,
    prefix_trace: bool,
    telemetry_stdout: bool,
) {
    let streaming = Cell::new(false);
    let target_started = Cell::new(false);
    let connection = RefCell::new(job.connection);
    let deadline = job.deadline;
    session.set_phase_stamps(telemetry_stdout);
    let timing = telemetry_stdout.then(|| {
        RefCell::new(RequestTiming::new(
            job.arrival,
            Instant::now(),
            job.request
                .find("stream")
                .ok()
                .flatten()
                .is_some_and(|stream| matches!(stream, Json::Bool(true))),
        ))
    });
    let timed_id = RefCell::new(None::<String>);
    let cancelled = || {
        shutdown.load(Ordering::SeqCst)
            || tokspur_sys::stop_requested()
            || Instant::now() >= deadline
            || connection.borrow_mut().disconnected()
    };
    let outcome = (|| -> std::result::Result<(), JobError> {
        if cancelled() {
            return Err(JobError::Refused("queued request cancelled or expired"));
        }
        let response_id = format!("chatcmpl-{}", random_hex(16)?);
        if timing.is_some() {
            *timed_id.borrow_mut() = Some(response_id.clone());
        }
        let mut target =
            |request: &Json,
             on_token: &mut dyn FnMut(&Json) -> std::result::Result<(), ServeError>| {
                target_started.set(true);
                let mut probe = || Ok::<bool, CallbackFailed>(cancelled());
                let mut sink = |event: &Json| on_token(event).map_err(|_| CallbackFailed);
                worker
                    .request(request, deadline, Some(&mut probe), Some(&mut sink))
                    .map_err(ServeError::from)
            };
        let mut chunk = |event: &Json| -> std::result::Result<(), ServeError> {
            let mut connection = connection.borrow_mut();
            let write_start = timing.as_ref().map(|_| Instant::now());
            if !streaming.get() {
                connection.begin_sse(deadline.min(Instant::now() + Duration::from_secs(5)))?;
                streaming.set(true);
            }
            connection.send_sse_data(
                &event.dump_limited(1024 * 1024)?,
                deadline.min(Instant::now() + Duration::from_secs(5)),
            )?;
            if let (Some(timing), Some(start)) = (timing.as_ref(), write_start) {
                timing
                    .borrow_mut()
                    .sse_write(start, Instant::now(), delta_kinds(event));
            }
            Ok(())
        };
        let mut trace_sink = |record: &Json| -> std::result::Result<(), ServeError> {
            let mut state = state.lock().map_err(|_| {
                ServeError::Admission(AdmissionError::PrefixTrace("poisoned private state lock"))
            })?;
            state.write_prefix_trace(record)
        };
        let mut is_cancelled = || cancelled();
        let mut new_seed = random_word;
        let mut sink = ServeSink {
            chunk: Some(&mut chunk),
            cancelled: &mut is_cancelled,
            new_seed: &mut new_seed,
            prefix_trace: if prefix_trace {
                Some(&mut trace_sink)
            } else {
                None
            },
        };
        let result =
            session.complete(&job.request, response_id.as_bytes(), &mut target, &mut sink)?;
        let mut connection = connection.borrow_mut();
        let write_start = timing.as_ref().map(|_| Instant::now());
        if streaming.get() {
            connection.send_sse_data(
                b"[DONE]",
                deadline.min(Instant::now() + Duration::from_secs(5)),
            )?;
            connection.finish_sse(deadline.min(Instant::now() + Duration::from_secs(5)))?;
            if let (Some(timing), Some(start)) = (timing.as_ref(), write_start) {
                timing
                    .borrow_mut()
                    .sse_write(start, Instant::now(), DeltaKinds::default());
            }
        } else {
            connection.send_json(
                200,
                &result.dump_limited(JSON_RESPONSE_BYTES)?,
                deadline.min(Instant::now() + Duration::from_secs(5)),
            )?;
            if let (Some(timing), Some(start)) = (timing.as_ref(), write_start) {
                timing.borrow_mut().json_write(start, Instant::now());
            }
        }
        Ok(())
    })();
    let print_timing = |session: &ServeSession, outcome: Outcome, error: Option<(&str, &[u8])>| {
        if let Some(timing) = timing.as_ref()
            && emits_record(telemetry_stdout, outcome)
        {
            let id = timed_id.borrow();
            let record = timing.borrow().record_with_error(
                id.as_deref().map(str::as_bytes),
                outcome,
                error,
                session.phase_stamps(),
                Instant::now(),
            );
            drop(print_line(&record));
        }
    };
    if outcome.is_ok() {
        let last_resp = worker.last_response();
        if match last_resp.find("event") {
            Ok(Some(e)) => e.string().is_ok_and(|s| s == b"done"),
            _ => false,
        } {
            if let Ok(mut state_guard) = state.lock() {
                drop(state_guard.write_telemetry_trace(last_resp));
            }
            if telemetry_stdout {
                drop(print_line(&telemetry_record(last_resp)));
            }
        }
    }
    let Err(error) = outcome else {
        print_timing(session, Outcome::Ok, None);
        return;
    };
    // Classified before the abort drain, in the order the status below is chosen, and only when the record is on.
    let failure = timing.as_ref().map(|_| {
        Outcome::of_failure(
            Instant::now() >= deadline,
            connection.borrow_mut().disconnected()
                || shutdown.load(Ordering::SeqCst)
                || tokspur_sys::stop_requested(),
            target_started.get(),
        )
    });
    if target_started.get() {
        let drain_start = Instant::now();
        drop(worker.abort_mtp_generation(drain_start + worker.options().drain_timeout));
        if let Some(timing) = timing.as_ref() {
            timing.borrow_mut().abort_drain(drain_start, Instant::now());
        }
        session.invalidate();
    }
    available.store(worker.alive(), Ordering::SeqCst);
    let mut connection = connection.borrow_mut();
    if !connection.disconnected()
        && !shutdown.load(Ordering::SeqCst)
        && !tokspur_sys::stop_requested()
    {
        let reply_deadline = Instant::now() + Duration::from_secs(2);
        let kind = failure_kind(&error, target_started.get());
        let Ok(payload) = error_json(error.to_string().as_bytes(), kind).dump_limited(8192) else {
            // The C++ builds this payload outside its inner try: std::terminate.
            std::process::abort();
        };
        let sent = if streaming.get() {
            connection
                .send_sse_data(&payload, reply_deadline)
                .and_then(|()| connection.finish_sse(reply_deadline))
        } else {
            let status = failure_status(&error, Instant::now() >= deadline, target_started.get());
            connection.send_json(status, &payload, reply_deadline)
        };
        if sent.is_err() {
            // Broken clients cannot waive worker draining/cleanup.
        }
    }
    drop(connection);
    if let Some(failure) = failure {
        // The same type and message the error reply carries, whether or not the client was still there to read it.
        let kind = failure_kind(&error, target_started.get());
        print_timing(session, failure, Some((kind, error.to_string().as_bytes())));
    }
}

fn ready_field<'a>(ready: &'a Json, name: &str) -> Result<&'a Json> {
    // Every copied field was validated before readiness publication.
    ready
        .find(name)?
        .ok_or(ServerError::Refused("missing required API field"))
}

/// How long a peer has to send its whole request.
const READ_SECONDS: Duration = Duration::from_secs(5);
/// Peers that may be partway through sending a request at once.
const READING_PEERS: usize = 64;

/// A connection with its arrival time and its request, or the refusal reading it produced.
type Arrival = (
    Connection,
    Instant,
    std::result::Result<crate::http::Request, HttpError>,
);

/// A routing failure, handled like the C++ inner `catch` clauses.
enum Failure {
    Http(HttpError),
    Other,
}

impl From<HttpError> for Failure {
    fn from(error: HttpError) -> Self {
        Self::Http(error)
    }
}

impl From<JsonError> for Failure {
    fn from(_: JsonError) -> Self {
        Self::Other
    }
}

impl From<AdmissionError> for Failure {
    fn from(_: AdmissionError) -> Self {
        Self::Other
    }
}

fn answer(connection: &mut Connection, code: i32, body: &Json) -> std::result::Result<(), Failure> {
    let body = body.dump_limited(8192)?;
    connection.send_json(code, &body, Instant::now() + Duration::from_secs(2))?;
    Ok(())
}

/// Turns away a peer whose request cannot be taken, with a 503 rather than a bare close. Written without waiting: the accept thread calls this, and it reads every other peer.
fn refuse_busy(mut connection: Connection) {
    let Ok(body) = error_json(
        b"server is busy with other connections; retry shortly",
        "server_busy",
    )
    .dump_limited(8192) else {
        return;
    };
    if connection.refuse_json(503, &body).is_err() {
        // The peer is gone or not reading; there is nobody to tell.
    }
}

/// Hands a whole request, or the refusal reading it produced, to the request workers. A full queue answers the peer 503. `false` once the workers are gone.
fn hand_off(sender: &std::sync::mpsc::SyncSender<Arrival>, arrival: Arrival) -> bool {
    match sender.try_send(arrival) {
        Ok(()) => true,
        Err(std::sync::mpsc::TrySendError::Full((connection, _, _))) => {
            refuse_busy(connection);
            true
        }
        Err(std::sync::mpsc::TrySendError::Disconnected(_)) => false,
    }
}

/// Adds an accepted peer to those still sending their request. A full table gives up its oldest unfinished peer, answering it 503: a client that sends its request at once is read long before it becomes the oldest.
fn admit_reading(reading: &mut Vec<(Connection, Instant)>, connection: Connection, now: Instant) {
    if reading.len() >= READING_PEERS {
        let (oldest, _) = reading.remove(0);
        refuse_busy(oldest);
    }
    reading.push((connection, now));
}

impl Serving<'_> {
    fn ready_line(&self) -> Result<Json> {
        let mut ready = vec![
            (b"event".to_vec(), Json::from("ready")),
            (
                b"endpoint".to_vec(),
                Json::from(format!("http://127.0.0.1:{}/v1", self.listener.port())),
            ),
            (b"model".to_vec(), Json::from(self.alias)),
            (b"one_active_generation".to_vec(), Json::Bool(true)),
            (b"waiting_requests".to_vec(), Json::integer(1)),
            (
                b"stack_execution".to_vec(),
                Json::from(self.stack_execution),
            ),
            (
                b"tool_enum_validation".to_vec(),
                Json::from(tool_enum_validation_name(self.tool_enums)),
            ),
            (
                b"tool_enum_validation_policy".to_vec(),
                Json::from(tool_enum_validation_policy(self.tool_enums)),
            ),
            (
                b"checkpoint_slots".to_vec(),
                Json::integer(u64::from(self.checkpoint_slots)),
            ),
            (
                b"checkpoint_combined_budget_bytes".to_vec(),
                Json::integer(self.checkpoint_budget),
            ),
            (b"external_engine".to_vec(), Json::Bool(false)),
            (b"tools_executed".to_vec(), Json::Bool(false)),
            (b"performance_qualified".to_vec(), Json::Bool(false)),
        ];
        if let Some(pid) = self.worker_pid {
            ready.push((b"worker_pid".to_vec(), Json::integer(u64::from(pid))));
        }
        if let Some(pid) = self.guardian_pid {
            ready.push((b"guardian_pid".to_vec(), Json::integer(u64::from(pid))));
        }
        if self.telemetry_stdout {
            ready.push((b"native_ready".to_vec(), self.ready_value.clone()));
        }
        let copy = |ready: &mut Vec<(Vec<u8>, Json)>, names: &[&str]| -> Result<()> {
            for name in names {
                ready.push((
                    name.as_bytes().to_vec(),
                    ready_field(self.ready_value, name)?.clone(),
                ));
            }
            Ok(())
        };
        #[cfg(feature = "verify8")]
        {
            ready.push((b"native_verify8".to_vec(), Json::Bool(self.verify8)));
            if self.verify8 {
                copy(
                    &mut ready,
                    &[
                        "verification_max_tokens",
                        "verification_head_many",
                        "lookup_proposal_tokens",
                        "known_token_capacity",
                        "known_residual_reuse",
                        "hc_exact_mode",
                        "wide_q8_exact_enabled",
                        "wide_q8_many_enabled",
                        "lookup_verification_budget_bytes",
                        "lookup_verification_charged_bytes",
                    ],
                )?;
            }
        }
        #[cfg(feature = "branch8")]
        {
            ready.push((b"native_branch8".to_vec(), Json::Bool(self.branch8)));
            if self.branch8 {
                copy(&mut ready, &["broker_protocol", "broker_batch_width"])?;
                ready.push((b"checkpoint_capture_stride".to_vec(), Json::integer(512)));
            }
        }
        #[cfg(feature = "qsa-stage-v8")]
        if self.stage8_present {
            copy(
                &mut ready,
                &[
                    "qsa_stage_v8_prepared",
                    "qsa_stage_v8_enabled",
                    "qsa_stage_v8_scratch_logical_bytes",
                    "qsa_stage_v8_scratch_charged_bytes",
                    "qsa_stage_v8_calls",
                    "qsa_stage_v8_physical_dispatches",
                    "qsa_stage_v8_extra_dispatches",
                ],
            )?;
        }
        if self.stack_execution == "recurrent-replay" {
            copy(
                &mut ready,
                &[
                    "replay_prepared",
                    "recurrent_replay_captures",
                    "mapped_weight_region_limit_bytes",
                ],
            )?;
        }
        #[cfg(feature = "fast-math")]
        if let Some(channel) = self.fast_channel {
            ready.push((
                b"native_fast_math".to_vec(),
                Json::from(if channel { "channel" } else { "staged" }),
            ));
            copy(
                &mut ready,
                &[
                    "ple_gate_transform",
                    "ple_gate_transform_schema",
                    "qsa_channel_chronological_mode",
                    "qsa_channel_chronological_schema",
                    "qsa_channel_factor_logical_bytes",
                    "qsa_channel_factor_charged_bytes",
                    "qsa_channel_calls",
                    "qsa_channel_extra_dispatches",
                ],
            )?;
        }
        if self.ready_value.object().is_ok()
            && let Some(drafter) = self.ready_value.find("native_drafter")?
        {
            ready.push((b"native_drafter".to_vec(), drafter.clone()));
            copy(
                &mut ready,
                &[
                    "native_dflash2_checkpoint_sha256",
                    "native_dflash2_weight_bytes",
                    "native_mtp_depth",
                    "native_mtp_protocol",
                ],
            )?;
            if let Some(vnax) = self.ready_value.find("native_verify_nax")? {
                ready.push((b"native_verify_nax".to_vec(), vnax.clone()));
            }
            if let Some(prec) = self.ready_value.find("native_dflash2_precision")? {
                ready.push((b"native_dflash2_precision".to_vec(), prec.clone()));
            }
            for opt_name in [
                "native_dflash2_max_drafts",
                "native_dflash2_vocab_slice",
                "native_dflash2_vocab_gather",
                "native_dflash2_vocab_gather_rows",
            ] {
                if let Some(v) = self.ready_value.find(opt_name)? {
                    ready.push((opt_name.as_bytes().to_vec(), v.clone()));
                }
            }
        }
        if self.ready_value.object().is_ok() {
            for opt_name in [
                "native_verify_hc_gdn_fused",
                "native_logit_topk",
                "native_ple_async_gather",
                "native_ple_ram_resident",
                "native_mtp_fast_round",
            ] {
                if let Some(v) = self.ready_value.find(opt_name)? {
                    ready.push((opt_name.as_bytes().to_vec(), v.clone()));
                }
            }
        }
        if let Some(first_pkt) = self.mtp_first_output_packet {
            ready.push((
                b"native_mtp_first_output_packet".to_vec(),
                Json::integer(u64::from(first_pkt)),
            ));
        }
        if self.tool_args != ToolArgValidation::Strict {
            ready.push((
                b"tool_arg_validation".to_vec(),
                Json::from(tool_arg_validation_name(self.tool_args)),
            ));
            ready.push((
                b"tool_arg_validation_policy".to_vec(),
                Json::from(tool_arg_validation_policy(self.tool_args)),
            ));
        }
        if self.context_length != 65_536 {
            ready.push((
                b"context_length".to_vec(),
                Json::integer(u64::from(self.context_length)),
            ));
        }
        Ok(Json::Object(ready))
    }

    /// Readiness publication and the accept loop (the C++ outer `try`).
    fn publish_and_accept(&self) -> Result<()> {
        print_line(&self.ready_line()?)?;
        let stopped = AtomicBool::new(false);
        std::thread::scope(|scope| {
            // Whole requests, and refusals that still owe an answer. Reading happens on this thread for every peer at once, so a peer that sends nothing, or stalls partway, never occupies a worker.
            let (sender, receiver) = std::sync::mpsc::sync_channel::<Arrival>(8);
            let receiver = Arc::new(Mutex::new(receiver));
            let (failures, failed) = std::sync::mpsc::channel();
            let mut workers = Vec::new();
            for _ in 0..4 {
                let receiver = Arc::clone(&receiver);
                let failures = failures.clone();
                let stopped = &stopped;
                workers.push(scope.spawn(move || {
                    loop {
                        let next = match receiver.lock() {
                            Ok(receiver) => receiver.recv(),
                            Err(_) => {
                                return Err(ServerError::Refused("HTTP reader queue is poisoned"));
                            }
                        };
                        let Ok((connection, arrival, read)) = next else {
                            return Ok(());
                        };
                        if stopped.load(Ordering::SeqCst) || tokspur_sys::stop_requested() {
                            return Ok(());
                        }
                        if let Err(error) = self.route_connection(connection, arrival, read) {
                            if failures.send(error).is_err() {
                                return Ok(());
                            }
                            return Ok(());
                        }
                    }
                }));
            }
            drop(failures);
            let limits = Limits {
                body_bytes: if self.context_length > 65_536 {
                    4 * 1024 * 1024
                } else {
                    1024 * 1024
                },
                ..Limits::default()
            };
            // Peers still sending their request, oldest first.
            let mut reading: Vec<(Connection, Instant)> = Vec::new();
            let result = 'serve: loop {
                if tokspur_sys::stop_requested() {
                    break Ok(());
                }
                match failed.try_recv() {
                    Ok(error) => break Err(error),
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        break Err(ServerError::Refused("HTTP readers stopped"));
                    }
                }
                let handles: Vec<i32> = reading
                    .iter()
                    .map(|(connection, _)| connection.native_handle())
                    .collect();
                let ready = match self.listener.poll_input(&handles, 100) {
                    Ok(ready) => ready,
                    Err(_) if tokspur_sys::stop_requested() => break Ok(()),
                    Err(error) => break Err(error.into()),
                };
                let now = Instant::now();
                let mut unfinished = Vec::with_capacity(reading.len());
                for (index, (mut connection, arrival)) in reading.drain(..).enumerate() {
                    let deadline = arrival + READ_SECONDS;
                    if !ready[index + 1] && now < deadline {
                        unfinished.push((connection, arrival));
                        continue;
                    }
                    let read = match connection.poll_request(deadline) {
                        Ok(None) => {
                            unfinished.push((connection, arrival));
                            continue;
                        }
                        Ok(Some(request)) => Ok(request),
                        // Nobody is left to answer.
                        Err(error) if error.disconnected() => continue,
                        Err(error) => Err(error),
                    };
                    if !hand_off(&sender, (connection, arrival, read)) {
                        break 'serve Err(ServerError::Refused("HTTP readers stopped"));
                    }
                }
                reading = unfinished;
                if !ready[0] {
                    continue;
                }
                let connection = match self
                    .listener
                    .accept(Instant::now() + Duration::from_millis(1), limits)
                {
                    Ok(connection) => connection,
                    Err(_) if tokspur_sys::stop_requested() => break Ok(()),
                    Err(error) if error.status() == 408 => continue,
                    Err(error) => break Err(error.into()),
                };
                admit_reading(&mut reading, connection, Instant::now());
            };
            drop(reading);
            stopped.store(true, Ordering::SeqCst);
            drop(sender);
            for reader in workers {
                match reader.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => return Err(error),
                    Err(_) => return Err(ServerError::Refused("HTTP reader failed")),
                }
            }
            result
        })
    }

    fn route_connection(
        &self,
        mut connection: Connection,
        arrival: Instant,
        read: std::result::Result<crate::http::Request, HttpError>,
    ) -> Result<()> {
        let routed = match self.route(&mut connection, read) {
            Ok(Some(input)) => self.enqueue(connection, input, arrival),
            Ok(None) => Ok(()),
            Err(failure) => Err((connection, failure)),
        };
        let Err((mut connection, failure)) = routed else {
            return Ok(());
        };
        let deadline = Instant::now() + Duration::from_secs(1);
        let sent = match failure {
            Failure::Http(error) if error.disconnected() => Ok(()),
            Failure::Http(error) => {
                let status = if error.status() != 0 {
                    error.status()
                } else {
                    400
                };
                let body = error_json(b"invalid or timed-out HTTP request", "http_error").dump()?;
                connection.send_json(status, &body, deadline)
            }
            Failure::Other => {
                let body = error_json(b"invalid chat request", "invalid_request_error").dump()?;
                connection.send_json(400, &body, deadline)
            }
        };
        if sent.is_err() {
            // The C++ swallows a failed refusal write.
        }
        Ok(())
    }

    /// Queues an admitted chat, or answers 429 (under the queue lock, as the C++ does). A failed answer is handled like any routing failure.
    fn enqueue(
        &self,
        mut connection: Connection,
        input: Json,
        arrival: Instant,
    ) -> std::result::Result<(), (Connection, Failure)> {
        let Ok(mut slot) = self.pending.lock() else {
            std::process::abort();
        };
        if slot.is_some() {
            let answered = answer(
                &mut connection,
                429,
                &error_json(b"one waiting request already admitted", "queue_full"),
            );
            drop(slot);
            return answered.map_err(|failure| (connection, failure));
        }
        *slot = Some(Job {
            connection,
            request: input,
            arrival,
            deadline: arrival + Duration::from_secs(u64::from(self.request_seconds)),
        });
        self.condition.notify_one();
        Ok(())
    }

    /// One request's routing (the C++ inner `try`). `Some` is an admitted chat for the generation thread; `None` means it was answered here.
    fn route(
        &self,
        connection: &mut Connection,
        read: std::result::Result<crate::http::Request, HttpError>,
    ) -> std::result::Result<Option<Json>, Failure> {
        let request = read?;
        if !equal_secret(
            request.header("authorization").as_bytes(),
            self.authorization.as_bytes(),
        ) {
            answer(
                connection,
                401,
                &error_json(b"invalid API credentials", "authentication_error"),
            )?;
            return Ok(None);
        }
        if !request.header("origin").is_empty() {
            answer(
                connection,
                403,
                &error_json(
                    b"cross-origin browser access is not enabled; use the trusted local client proxy",
                    "origin_refused",
                ),
            )?;
            return Ok(None);
        }
        let live = self.available.load(Ordering::SeqCst);
        if request.method == "GET" && request.path == "/health" {
            let status = Json::new_object([
                (
                    "status",
                    Json::from(if live { "ready" } else { "worker_failed" }),
                ),
                ("model", Json::from(self.alias)),
            ]);
            answer(connection, if live { 200 } else { 503 }, &status)?;
            return Ok(None);
        }
        if request.method == "GET" && request.path == "/v1/models" {
            let models = Json::new_object([
                ("object", Json::from("list")),
                (
                    "data",
                    Json::Array(vec![Json::new_object([
                        ("id", Json::from(self.alias)),
                        ("object", Json::from("model")),
                        ("created", Json::integer(0)),
                        ("owned_by", Json::from("tokspur")),
                    ])]),
                ),
            ]);
            answer(connection, 200, &models)?;
            return Ok(None);
        }
        if request.method != "POST" || request.path != "/v1/chat/completions" {
            answer(
                connection,
                404,
                &error_json(b"unknown endpoint", "not_found"),
            )?;
            return Ok(None);
        }
        let content_type = request.header("content-type");
        if !json_content_type(content_type) {
            answer(
                connection,
                415,
                &error_json(b"application/json required", "unsupported_media_type"),
            )?;
            return Ok(None);
        }
        if !self.available.load(Ordering::SeqCst) {
            answer(
                connection,
                503,
                &error_json(
                    b"owned worker is unavailable; no automatic restart or generation retry",
                    "worker_unavailable",
                ),
            )?;
            return Ok(None);
        }
        let (max_bytes, max_values) = if self.context_length > 65_536 {
            (4 * 1024 * 1024, 300_000)
        } else {
            (1024 * 1024, 150_000)
        };
        let input = parse_json_limited(&request.body, JsonLimits::new(max_bytes, max_values, 32))?;
        admit_chat_with_context(
            &input,
            self.alias.as_bytes(),
            self.tool_enums,
            self.tool_args,
            self.context_length,
        )?; // No mutation on invalid admission.
        Ok(Some(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "fast-math")]
    #[test]
    fn cli_launch_flags_pass_server_startup_validation() {
        use tokspur_cli::serve::{SpeculativeAsset, build_carrier_flags};

        for cli_args in [
            vec!["serve"],
            vec!["serve", "--speed-preset"],
            vec![
                "serve",
                "--prefill-dense",
                "f32",
                "--prefill-grouped",
                "serial",
            ],
            vec![
                "serve",
                "--drafter",
                "dflash2",
                "--dflash2-dir",
                "/tmp/dflash2",
            ],
            vec![
                "serve",
                "--speed-preset",
                "--drafter",
                "dflash2",
                "--dflash2-dir",
                "/tmp/dflash2",
            ],
        ] {
            let tokspur_cli::Command::Serve(cli) =
                tokspur_cli::parse_args(cli_args.clone()).unwrap().command
            else {
                panic!("expected serve options");
            };
            let asset = if cli.drafter == "mtp" {
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf"))
            } else {
                SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2"))
            };
            let flags = build_carrier_flags(&cli, asset, Path::new("/tmp/state")).unwrap();
            let mut args: Vec<OsString> = [
                "tokspur-server",
                "--allow-full-model",
                "true",
                "--tokenizer",
                "/tmp/tokenizer.json",
                "--template",
                "/tmp/template",
                "--state-dir",
                "/tmp/state",
            ]
            .into_iter()
            .map(OsString::from)
            .collect();
            args.extend(flags.into_iter().map(OsString::from));
            let parsed = options(&args).unwrap_or_else(|error| panic!("CLI {cli_args:?}: {error}"));
            assert!(!parsed.worker.checkpoint_ssd);
            assert!(!parsed.worker.prefill_router_window.unwrap_or(false));
            assert!(!parsed.worker.prefill_hc_down_window.unwrap_or(false));
            assert!(!parsed.worker.qsa_sparse.unwrap_or(false));
            assert!(!parsed.worker.gdn_ported.unwrap_or(false));
        }
    }

    #[test]
    fn ssd_budget_preserves_explicit_policy_in_either_order() {
        let base: Vec<OsString> = [
            "tokspur-server",
            "--allow-full-model",
            "true",
            "--tokenizer",
            "/tmp/tokenizer.json",
            "--template",
            "/tmp/template",
            "--state-dir",
            "/tmp/state",
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        assert!(!options(&base).unwrap().worker.checkpoint_ssd);
        for budget_flag in ["--checkpoint-cap-gib", "--checkpoint-ssd-budget-gib"] {
            for budget in ["0", "8"] {
                let mut budget_only = base.clone();
                budget_only.extend([OsString::from(budget_flag), OsString::from(budget)]);
                assert!(!options(&budget_only).unwrap().worker.checkpoint_ssd);
                for policy in ["off", "on", "false", "true"] {
                    for policy_first in [false, true] {
                        let mut argv = base.clone();
                        let policy_pair = ["--checkpoint-ssd", policy];
                        let budget_pair = [budget_flag, budget];
                        for pair in if policy_first {
                            [policy_pair, budget_pair]
                        } else {
                            [budget_pair, policy_pair]
                        } {
                            argv.extend(pair.map(OsString::from));
                        }
                        let selected = options(&argv).unwrap().worker;
                        assert_eq!(
                            selected.checkpoint_ssd,
                            matches!(policy, "on" | "true") && budget != "0"
                        );
                        assert_eq!(
                            selected.checkpoint_ssd_budget_bytes,
                            if budget == "0" { 0 } else { 8 << 30 }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn native_sync_selectors_parse_and_refuse_invalid_values() {
        let base: Vec<OsString> = [
            "tokspur-server",
            "--allow-full-model",
            "true",
            "--tokenizer",
            "/tmp/tokenizer.json",
            "--template",
            "/tmp/template",
            "--state-dir",
            "/tmp/state",
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        let defaults = options(&base).unwrap();
        assert_eq!(defaults.worker.host_logit_views, None);
        assert_eq!(defaults.worker.dflash2_defer_context, None);
        for flag in [
            "--native-host-logit-views",
            "--native-dflash2-defer-context",
        ] {
            for value in ["off", "on"] {
                let mut argv = base.clone();
                argv.extend([OsString::from(flag), OsString::from(value)]);
                let parsed = options(&argv).unwrap();
                let selected = if flag == "--native-host-logit-views" {
                    parsed.worker.host_logit_views
                } else {
                    parsed.worker.dflash2_defer_context
                };
                assert_eq!(selected, Some(value == "on"));
                argv.extend([OsString::from(flag), OsString::from(value)]);
                assert!(matches!(options(&argv), Err(ServerError::Refused(_))));
            }
            for value in ["true", "false", "1", "", "auto"] {
                let mut argv = base.clone();
                argv.extend([OsString::from(flag), OsString::from(value)]);
                assert!(matches!(options(&argv), Err(ServerError::Refused(_))));
            }
            let mut argv = base.clone();
            argv.push(OsString::from(flag));
            assert!(matches!(options(&argv), Err(ServerError::Refused(_))));
        }
    }

    #[test]
    #[cfg(feature = "fast-math")]
    fn shared_prefix_cache_admits_dflash2_without_opening_assets() {
        let args: Vec<OsString> = [
            "tokspur-server",
            "--allow-full-model",
            "true",
            "--tokenizer",
            "/tmp/tokenizer.json",
            "--template",
            "/tmp/template",
            "--state-dir",
            "/tmp/state",
            "--native-scalar-channel",
            "on",
            "--native-fast-math",
            "channel",
            "--native-qsa-stage-v8",
            "on",
            "--shared-prefix-cache",
            "on",
            "--dflash2-dir",
            "/tmp/dflash2",
        ]
        .into_iter()
        .map(OsString::from)
        .collect();
        assert!(options(&args).is_ok());
    }

    #[test]
    fn private_state_debug_hides_the_key() {
        // Arrange
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temporary = TemporaryState(std::env::temp_dir().join(format!(
            "tokspur-state-debug-{}-{nanos}",
            std::process::id()
        )));
        let state = PrivateState::new(&temporary.0).unwrap();

        // Act
        let rendered = format!("{state:?}");

        // Assert
        assert_eq!(rendered, "PrivateState(..)");
        assert_eq!(format!("{:?}", state._lock), "PrivateStateLock(..)");
    }

    #[test]
    fn telemetry_record_withholds_reply_members_at_every_depth() {
        // Arrange
        let done = tokspur_json::parse_json(
            br#"{"event":"done","generated_ids":[11,22],"committed_prefix":7,"final_logit_top8":[{"id":3,"logit":1.5}],"branch8_session":{"stop_id":2,"final_logit_top8":[{"id":3,"logit":1.5}],"hc_exact_final_logit_sha256":"ab"},"profile":[{"generated_ids":[5],"wall_ms":1.25}],"memory":{"physical_footprint_bytes":4096}}"#,
        )
        .unwrap();

        // Act
        let persisted = telemetry_record(&done).dump().unwrap();

        // Assert
        assert_eq!(
            persisted,
            br#"{"event":"done","committed_prefix":7,"branch8_session":{"stop_id":2,"hc_exact_final_logit_sha256":"ab"},"profile":[{"wall_ms":1.25}],"memory":{"physical_footprint_bytes":4096}}"#
        );
    }

    #[test]
    fn telemetry_trace_room_counts_the_bytes_already_on_disk() {
        // Assert
        assert!(telemetry_trace_has_room(0, 1));
        assert!(telemetry_trace_has_room(TELEMETRY_TRACE_LIMIT - 10, 10));
        assert!(!telemetry_trace_has_room(TELEMETRY_TRACE_LIMIT - 10, 11));
        assert!(!telemetry_trace_has_room(TELEMETRY_TRACE_LIMIT, 1));
        // An earlier build did not bound the file across starts, so it can already be past the limit.
        assert!(!telemetry_trace_has_room(TELEMETRY_TRACE_LIMIT + 1, 1));
        assert!(!telemetry_trace_has_room(u64::MAX, 1));
    }

    /// A private state directory under the system temporary directory, removed when dropped.
    struct TemporaryState(std::path::PathBuf);

    impl Drop for TemporaryState {
        fn drop(&mut self) {
            if std::fs::remove_dir_all(&self.0).is_err() {
                // Best-effort cleanup of a temporary test fixture.
            }
        }
    }

    #[test]
    fn private_state_releases_ownership_while_a_duplicate_descriptor_survives() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let state = TemporaryState(std::env::temp_dir().join(format!(
            "tokspur-state-release-{}-{nanos}",
            std::process::id()
        )));
        let first = PrivateState::new(&state.0).unwrap();
        let duplicate = first._lock.file.try_clone().unwrap();
        drop(first);
        let second = PrivateState::new(&state.0).unwrap();
        drop(duplicate);
        assert!(matches!(
            PrivateState::new(&state.0),
            Err(ServerError::StateLock(_))
        ));
        drop(second);
        PrivateState::new(&state.0).unwrap();
    }

    #[test]
    fn repeated_release_does_not_unlock_a_replacement_owner() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let state = TemporaryState(std::env::temp_dir().join(format!(
            "tokspur-state-release-twice-{}-{nanos}",
            std::process::id()
        )));
        let mut first = PrivateState::new(&state.0).unwrap();
        first._lock.release().unwrap();
        let second = PrivateState::new(&state.0).unwrap();
        first._lock.release().unwrap();
        drop(first);
        assert!(matches!(
            PrivateState::new(&state.0),
            Err(ServerError::StateLock(_))
        ));
        drop(second);
        PrivateState::new(&state.0).unwrap();
    }

    #[test]
    fn telemetry_trace_keeps_no_reply_data_and_its_bound_across_server_starts() {
        // Arrange
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let state = TemporaryState(std::env::temp_dir().join(format!(
            "tokspur-telemetry-trace-{}-{nanos}",
            std::process::id()
        )));
        let path = state.0.join("telemetry.jsonl");
        let done = tokspur_json::parse_json(
            br#"{"event":"done","generated_ids":[11,22],"final_logit_top8":[{"id":3,"logit":1.5}],"committed_prefix":7}"#,
        )
        .unwrap();

        // Act
        let mut first = PrivateState::new(&state.0).unwrap();
        first.write_telemetry_trace(&done).unwrap();
        let written = std::fs::read(&path).unwrap();
        drop(first);
        let mut second = PrivateState::new(&state.0).unwrap();
        second.write_telemetry_trace(&done).unwrap();
        let counted = second.telemetry_trace_bytes;
        let total = std::fs::metadata(&path).unwrap().len();
        drop(second);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(TELEMETRY_TRACE_LIMIT)
            .unwrap();
        let mut third = PrivateState::new(&state.0).unwrap();
        let refused = third.write_telemetry_trace(&done);
        let after = std::fs::metadata(&path).unwrap().len();

        // Assert
        let record_bytes = u64::try_from(written.len()).unwrap();
        assert_eq!(written, b"{\"event\":\"done\",\"committed_prefix\":7}\n");
        assert_eq!(counted, total);
        assert_eq!(total, 2 * record_bytes);
        assert!(matches!(refused, Err(ServerError::Refused(_))));
        assert_eq!(after, TELEMETRY_TRACE_LIMIT);
    }

    #[test]
    fn server_and_serve_errors_keep_unbroken_source_chains_and_redact_debug() {
        use std::error::Error;

        let inner = tokspur_json::parse_json(b"\"\xc0\x80\"").unwrap_err();
        let admission = AdmissionError::Json(inner);
        let serve = ServeError::Admission(admission);
        let server = ServerError::Serve(serve);

        let s1 = server.source().expect("ServerError must expose ServeError");
        let s2 = s1.source().expect("ServeError must expose AdmissionError");
        let s3 = s2.source().expect("AdmissionError must expose JsonError");
        let s4 = s3.source().expect("JsonError must expose UnicodeError");
        assert!(s4.source().is_none());

        let state_err = ServerError::StateIo {
            reason: "cannot read API key",
            source: Some(std::io::Error::other(
                "/Users/secret/.tokspur/state/api-key",
            )),
        };
        assert_eq!(state_err.to_string(), "cannot read API key");
        assert!(state_err.source().is_some());
        assert!(!format!("{state_err:?}").contains("secret"));
    }

    #[test]
    fn pack_startup_message_uses_loaded_identity_for_custom_pack_paths() {
        let mut args = vec![
            "tokspur-server".into(),
            "--allow-full-model".into(),
            "true".into(),
            "--tokenizer".into(),
            "/tmp/tokenizer".into(),
            "--template".into(),
            "/tmp/template".into(),
            "--state-dir".into(),
            "/tmp/state".into(),
        ];
        let raw_requested_pack = options(&args).unwrap().worker.derived_pack.is_some();
        args.extend([
            OsString::from("--derived-pack"),
            OsString::from("/private/secret/renamed-v3.gguf"),
        ]);
        let custom_requested_pack = options(&args).unwrap().worker.derived_pack.is_some();
        let ready = |data: &[u8]| tokspur_json::parse_json(data).unwrap();
        let v2 = ready(
            br#"{"event":"ready","derived_pack_schema":2,"derived_pack_recipe":"tokspur-pack-v2"}"#,
        );
        let v3 = ready(
            br#"{"event":"ready","derived_pack_schema":3,"derived_pack_recipe":"tokspur-pack-v3"}"#,
        );
        let raw = ready(br#"{"event":"ready"}"#);
        assert_eq!(
            pack_startup_message(&v2, custom_requested_pack),
            b"tokspur: loaded Pack v2 (tokspur-pack-v2)\n"
        );
        assert_eq!(
            pack_startup_message(&v3, custom_requested_pack),
            b"tokspur: loaded Pack v3 (tokspur-pack-v3)\n"
        );
        assert_eq!(
            pack_startup_message(&raw, raw_requested_pack),
            b"tokspur: loaded raw GGUF weights\n"
        );
        let unavailable = b"tokspur: weight pack identity unavailable\n";
        assert_eq!(pack_startup_message(&raw, true), unavailable);
        assert_eq!(pack_startup_message(&v3, false), unavailable);
        for contradictory in [
            r#"{"derived_pack_schema":3}"#,
            r#"{"derived_pack_schema":2,"derived_pack_recipe":"tokspur-pack-v3"}"#,
            r#"{"derived_pack_schema":2,"derived_pack_recipe":"secret\ntext"}"#,
        ] {
            assert_eq!(
                pack_startup_message(&ready(contradictory.as_bytes()), true),
                unavailable
            );
        }
    }

    #[test]
    fn ready_line_emits_tool_enum_validation_and_policy_in_cpp_order() {
        let listener = Listener::new(0, 1).unwrap();
        let ready_value = Json::Null;
        let available = AtomicBool::new(true);
        let pending = Mutex::new(None);
        let condition = Condvar::new();
        for (policy, expected_name, expected_policy) in [
            (
                ToolEnumValidation::Strict,
                "strict",
                "strict-server-enums-v1",
            ),
            (
                ToolEnumValidation::Executor,
                "executor",
                "original-agent-executor-enums-v1",
            ),
        ] {
            let serving = Serving {
                listener: &listener,
                ready_value: &ready_value,
                telemetry_stdout: false,
                worker_pid: None,
                guardian_pid: None,
                alias: "tokspur",
                stack_execution: "direct",
                tool_enums: policy,
                tool_args: ToolArgValidation::Strict,
                mtp_first_output_packet: None,
                checkpoint_slots: 4,
                checkpoint_budget: 536_870_912,
                context_length: 65_536,
                #[cfg(feature = "verify8")]
                verify8: false,
                #[cfg(feature = "branch8")]
                branch8: false,
                #[cfg(feature = "qsa-stage-v8")]
                stage8_present: false,
                #[cfg(feature = "fast-math")]
                fast_channel: None,
                authorization: "Bearer test",
                request_seconds: 600,
                available: &available,
                pending: &pending,
                condition: &condition,
            };
            let ready = serving.ready_line().unwrap();
            let fields = ready.object().unwrap();
            let keys: Vec<&[u8]> = fields.iter().map(|(key, _)| key.as_slice()).collect();
            let stack_at = keys
                .iter()
                .position(|&key| key == b"stack_execution")
                .unwrap();
            assert_eq!(keys[stack_at + 1], b"tool_enum_validation");
            assert_eq!(keys[stack_at + 2], b"tool_enum_validation_policy");
            assert_eq!(keys[stack_at + 3], b"checkpoint_slots");
            assert_eq!(
                ready
                    .find("tool_enum_validation")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap(),
                expected_name.as_bytes()
            );
            assert_eq!(
                ready
                    .find("tool_enum_validation_policy")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap(),
                expected_policy.as_bytes()
            );
        }
    }

    #[test]
    fn ready_line_preserves_complete_native_ready_only_with_telemetry() {
        let listener = Listener::new(0, 1).unwrap();
        let available = AtomicBool::new(true);
        let pending = Mutex::new(None);
        let condition = Condvar::new();
        for mapping in [
            "",
            ",\"native_pack_shared_mapping\":true",
            ",\"native_pack_shared_mapping\":null",
        ] {
            let fixture = format!(
                r#"{{"event":"ready","derived_pack_schema":3,"derived_pack_recipe":"tokspur-pack-v3","imported_regions":17,"generated_marker":"native-ready-only","future":{{"values":[null,false,1.2300e+02,18446744073709551615],"text":"generated\\ntext"}}{mapping}}}"#
            );
            let native_ready = tokspur_json::parse_json(fixture.as_bytes()).unwrap();
            let mut serving = Serving {
                listener: &listener,
                ready_value: &native_ready,
                telemetry_stdout: false,
                worker_pid: None,
                guardian_pid: None,
                alias: "tokspur",
                stack_execution: "direct",
                tool_enums: ToolEnumValidation::Strict,
                tool_args: ToolArgValidation::Strict,
                mtp_first_output_packet: None,
                checkpoint_slots: 0,
                checkpoint_budget: 0,
                context_length: 65_536,
                #[cfg(feature = "verify8")]
                verify8: false,
                #[cfg(feature = "branch8")]
                branch8: false,
                #[cfg(feature = "qsa-stage-v8")]
                stage8_present: false,
                #[cfg(feature = "fast-math")]
                fast_channel: None,
                authorization: "Bearer test",
                request_seconds: 600,
                available: &available,
                pending: &pending,
                condition: &condition,
            };
            let without_telemetry = serving.ready_line().unwrap();
            assert!(without_telemetry.find("native_ready").unwrap().is_none());
            assert!(
                !String::from_utf8(without_telemetry.dump().unwrap())
                    .unwrap()
                    .contains("native-ready-only")
            );

            // Telemetry is controlled by its explicit flag, even without PID metadata.
            serving.telemetry_stdout = true;
            let with_telemetry = serving.ready_line().unwrap();
            let nested = with_telemetry.find("native_ready").unwrap().unwrap();
            assert_eq!(nested, &native_ready);
            assert_eq!(nested.dump().unwrap(), fixture.as_bytes());
            let mut remaining = with_telemetry.object().unwrap().to_vec();
            remaining.retain(|(key, _)| key != b"native_ready");
            assert_eq!(Json::Object(remaining), without_telemetry);
        }
    }

    /// A fresh private state directory for one test.
    fn temporary_state(name: &str) -> TemporaryState {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        TemporaryState(
            std::env::temp_dir().join(format!("tokspur-{name}-{}-{nanos}", std::process::id())),
        )
    }

    #[test]
    fn a_full_prefix_trace_is_a_server_error_and_stays_refused() {
        // Arrange
        let directory = temporary_state("prefix-trace-full");
        let mut state = PrivateState::new(&directory.0).unwrap();
        state.enable_prefix_trace().unwrap();
        let record = Json::new_object([("event", Json::from("prompt_prefix"))]);
        state.write_prefix_trace(&record).unwrap();
        let line = state.prefix_trace_bytes;
        // One byte short of room for the same record again.
        state.prefix_trace_bytes = PREFIX_TRACE_LIMIT - line + 1;

        // Act
        let first = state.write_prefix_trace(&record).unwrap_err();
        // Tracing fails closed: room found later does not resume it.
        state.prefix_trace_bytes = line;
        let later = state.write_prefix_trace(&record).unwrap_err();

        // Assert
        for error in [first, later] {
            assert_eq!(error.to_string(), PREFIX_TRACE_FULL);
            let error = JobError::Serve(error);
            for target_started in [false, true] {
                assert_eq!(failure_kind(&error, target_started), "server_error");
                assert_eq!(failure_status(&error, false, target_started), 500);
            }
            assert_eq!(failure_status(&error, true, false), 408);
        }
        assert_eq!(
            std::fs::read(directory.0.join("prefix-trace.jsonl")).unwrap(),
            b"{\"event\":\"prompt_prefix\"}\n"
        );
        // A write failure is the server's fault too.
        let persist = JobError::Serve(ServeError::Admission(AdmissionError::TracePersist(
            Arc::new(std::io::Error::other("disk")),
        )));
        assert_eq!(failure_kind(&persist, false), "server_error");
        assert_eq!(failure_status(&persist, false, false), 500);
        // Request and generation failures keep their classes.
        let refused = JobError::Serve(ServeError::Admission(AdmissionError::Refused(
            "missing required API field",
        )));
        assert_eq!(failure_kind(&refused, false), "invalid_request_error");
        assert_eq!(failure_status(&refused, false, false), 400);
        assert_eq!(failure_kind(&refused, true), "generation_failed_no_retry");
        assert_eq!(failure_status(&refused, false, true), 502);
        assert_eq!(failure_status(&refused, true, true), 408);
    }

    #[test]
    fn an_existing_prefix_trace_refuses_startup_naming_the_file_and_is_kept() {
        // Arrange
        let directory = temporary_state("prefix-trace-exists");
        let record = Json::new_object([("event", Json::from("prompt_prefix"))]);
        let mut earlier = PrivateState::new(&directory.0).unwrap();
        earlier.enable_prefix_trace().unwrap();
        earlier.write_prefix_trace(&record).unwrap();
        drop(earlier);
        let mut restarted = PrivateState::new(&directory.0).unwrap();

        // Act
        let refusal = restarted.enable_prefix_trace().unwrap_err();

        // Assert
        assert_eq!(
            refusal.to_string(),
            "private prefix trace requires a new file: move or remove prefix-trace.jsonl from the state directory, or start without --private-prefix-trace"
        );
        assert_eq!(
            std::fs::read(directory.0.join("prefix-trace.jsonl")).unwrap(),
            b"{\"event\":\"prompt_prefix\"}\n"
        );
        // Without a trace file nothing is traced, and that too is the server's fault.
        let unopened = restarted.write_prefix_trace(&record).unwrap_err();
        assert_eq!(unopened.to_string(), PREFIX_TRACE_UNAVAILABLE);
        assert_eq!(
            failure_status(&JobError::Serve(unopened), false, false),
            500
        );
    }

    /// A connection over a socketpair, with the peer's end.
    fn peered() -> (Connection, std::os::unix::net::UnixStream) {
        let (server, peer) = std::os::unix::net::UnixStream::pair().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        let connection = Connection::new(server.into(), Limits::default()).unwrap();
        (connection, peer)
    }

    /// Everything the server wrote before closing its end.
    fn answered(mut peer: std::os::unix::net::UnixStream) -> String {
        let mut response = String::new();
        peer.read_to_string(&mut response).unwrap();
        response
    }

    fn assert_busy(response: &str) {
        assert!(
            response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
            "{response}"
        );
        assert!(response.contains("\r\nRetry-After: 5\r\n"), "{response}");
        assert!(
            response.ends_with(
                r#"{"error":{"message":"server is busy with other connections; retry shortly","type":"server_busy","code":"server_busy"}}"#
            ),
            "{response}"
        );
    }

    #[test]
    fn a_full_hand_off_queue_answers_503_instead_of_closing() {
        // Arrange
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Arrival>(1);
        let (queued, queued_peer) = peered();
        let (refused, refused_peer) = peered();
        let request = || Ok(crate::http::Request::default());

        // Act
        let taken = hand_off(&sender, (queued, Instant::now(), request()));
        let full = hand_off(&sender, (refused, Instant::now(), request()));

        // Assert
        assert!(taken && full);
        assert_busy(&answered(refused_peer));
        // The queued peer is still waiting for a worker.
        let (connection, _, _) = receiver.try_recv().unwrap();
        assert!(receiver.try_recv().is_err());
        drop(connection);
        assert_eq!(answered(queued_peer), "");
        // Stopped workers are reported, not answered.
        drop(receiver);
        let (late, _late_peer) = peered();
        assert!(!hand_off(&sender, (late, Instant::now(), request())));
    }

    #[test]
    fn a_full_reading_table_answers_its_oldest_peer_503() {
        // Arrange
        let mut reading = Vec::new();
        let mut peers = Vec::new();
        for _ in 0..READING_PEERS {
            let (connection, peer) = peered();
            admit_reading(&mut reading, connection, Instant::now());
            peers.push(peer);
        }
        let second_handle = reading[1].0.native_handle();
        let (newest, _newest_peer) = peered();
        let newest_handle = newest.native_handle();

        // Act
        admit_reading(&mut reading, newest, Instant::now());

        // Assert
        assert_eq!(reading.len(), READING_PEERS);
        assert_eq!(reading[0].0.native_handle(), second_handle);
        assert_eq!(reading[READING_PEERS - 1].0.native_handle(), newest_handle);
        assert_busy(&answered(peers.remove(0)));
        // Every other peer is still being read.
        drop(reading);
        for peer in peers {
            assert_eq!(answered(peer), "");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod readiness_network_tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpStream;
    use std::process::{Child, Command, Stdio};

    struct OwnedTestChild(Child);
    impl Drop for OwnedTestChild {
        fn drop(&mut self) {
            // Keep cleanup from replacing an assertion that is already unwinding.
            let _cleanup_kill = self.0.kill();
            let _cleanup_wait = self.0.wait();
        }
    }

    #[test]
    fn silent_peers_do_not_hold_health_or_shutdown() {
        if std::env::var_os("TOKSPUR_TEST_HTTP_READERS_CHILD").is_some() {
            assert!(tokspur_sys::install_stop_handlers());
            let _owner_lifetime = monitor_owner_stdin().unwrap();
            let listener = Listener::new(0, 8).unwrap();
            let ready = tokspur_json::parse_json(
                br#"{"event":"ready","generated_marker":"native-ready-only","imported_regions":17}"#,
            )
            .unwrap();
            let available = AtomicBool::new(true);
            let pending = Mutex::new(None);
            let condition = Condvar::new();
            let serving = Serving {
                listener: &listener,
                ready_value: &ready,
                telemetry_stdout: true,
                worker_pid: None,
                guardian_pid: None,
                alias: "tokspur",
                stack_execution: "direct",
                tool_enums: ToolEnumValidation::Strict,
                tool_args: ToolArgValidation::Strict,
                mtp_first_output_packet: None,
                checkpoint_slots: 0,
                checkpoint_budget: 0,
                context_length: 65_536,
                #[cfg(feature = "verify8")]
                verify8: false,
                #[cfg(feature = "branch8")]
                branch8: false,
                #[cfg(feature = "qsa-stage-v8")]
                stage8_present: false,
                #[cfg(feature = "fast-math")]
                fast_channel: None,
                authorization: "Bearer test",
                request_seconds: 3600,
                available: &available,
                pending: &pending,
                condition: &condition,
            };
            println!("{{\"test_port\":{}}}", serving.listener.port());
            serving.publish_and_accept().unwrap();
            return;
        }
        let mut child = OwnedTestChild(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "serve::readiness_network_tests::silent_peers_do_not_hold_health_or_shutdown",
                    "--nocapture",
                ])
                .env("TOKSPUR_TEST_HTTP_READERS_CHILD", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut lines = BufReader::new(child.0.stdout.take().unwrap());
        let port = loop {
            let mut line = String::new();
            assert!(
                lines.read_line(&mut line).unwrap() > 0,
                "reader child ended before readiness"
            );
            if let Some(at) = line.find('{')
                && let Ok(ready) = tokspur_json::parse_json(line[at..].trim().as_bytes())
                && let Some(port) = ready.find("test_port").unwrap()
            {
                break u16::try_from(port.unsigned_integer().unwrap()).unwrap();
            }
        };
        // More unauthenticated peers than there are request workers: some never send, some stall partway through their headers.
        let mut quiet = Vec::new();
        for index in 0..24 {
            let mut peer = TcpStream::connect(("127.0.0.1", port)).unwrap();
            if index % 2 == 1 {
                peer.write_all(b"GET /health HTTP/1.1\r\nHost: loc")
                    .unwrap();
            }
            quiet.push(peer);
        }
        std::thread::sleep(Duration::from_millis(30));
        for path in ["/health", "/v1/models"] {
            let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            write!(
                client,
                "GET {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test\r\n\r\n"
            )
            .unwrap();
            let mut response = String::new();
            client.read_to_string(&mut response).unwrap();
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            assert!(!response.contains("native_ready"), "{response}");
            assert!(!response.contains("native-ready-only"), "{response}");
            assert!(!response.contains("imported_regions"), "{response}");
        }
        // Enough further silent peers to fill the table of unfinished requests: the peers it gives up are told 503, oldest first, and a request is still served.
        for _ in quiet.len()..=READING_PEERS + 1 {
            quiet.push(TcpStream::connect(("127.0.0.1", port)).unwrap());
        }
        for mut evicted in quiet.drain(..2) {
            evicted
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let mut response = String::new();
            evicted.read_to_string(&mut response).unwrap();
            assert!(response.starts_with("HTTP/1.1 503 "), "{response}");
            assert!(response.contains("server_busy"), "{response}");
        }
        let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        write!(
            client,
            "GET /health HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test\r\n\r\n"
        )
        .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let started = Instant::now();
        drop(child.0.stdin.take());
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "HTTP readers did not stop"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(quiet);
    }
}

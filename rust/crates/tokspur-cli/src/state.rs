//! Private `.tokspur/` state management, shared `serve.lock` (`flock`), authoritative asset adoption and `psutil`-free inference worker detection.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::{self, File};
use std::io::Write;
use std::os::fd::AsFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt, symlink};
use std::path::{Component, Path, PathBuf};

use tokspur_json::{Json, JsonLimits, parse_json_limited};
use tokspur_sys::fd::{
    FileStat, O_CLOEXEC, O_CREAT, O_DIRECTORY, O_EXCL, O_NOFOLLOW, O_RDONLY, O_RDWR, O_WRONLY,
    fchmod, file_volume_uuid, flock_exclusive_nonblocking, fstat, fsync, lstat_at, lstat_path,
    openat, pread_exact, pwrite_all, set_nocache, unlink_at,
};
use tokspur_sys::host::{ProcessSnapshot, list_live_processes};
use tokspur_sys::net::port_available;
use tokspur_sys::random_bytes;
use tokspur_text::sha256::{Sha256, hex};

use crate::error::CliError;
use crate::layout::{RuntimeLayout, SCALAR_WORKER_NAME, ensure_no_symlink_components};

/// Darwin `<sys/param.h>` `MAXCOMLEN` (`16`); verified in `tokspur-sys/tests/sdk_bindings.rs`.
pub const DARWIN_MAXCOMLEN: usize = 16;

const MAX_JSON_FILE: usize = 16 * 1024 * 1024;
const ASSET_STREAM_CHUNK: usize = 8 * 1024 * 1024;

/// Exclusive non-blocking `flock` guard over `<tokspur_base>/state/serve.lock`, shared with Python `local.py:state_lock`.
pub struct StateLock {
    _file: File,
}

impl StateLock {
    pub fn acquire(tokspur_base: &Path) -> Result<Self, CliError> {
        let state_dir = private_dir(tokspur_base, Path::new("state"))?;
        let dir_fd = open_nofollow_dir(&state_dir, "private state directory")?;
        let file = openat(
            dir_fd.as_fd(),
            c"serve.lock",
            O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
            0o600,
        )
        .map_err(|source| CliError::Io {
            context: "cannot open serve.lock".to_owned(),
            source,
        })?;
        let info = fstat(file.as_fd()).map_err(|source| CliError::Io {
            context: "cannot stat serve.lock".to_owned(),
            source,
        })?;
        if !info.is_regular() {
            return Err(CliError::Validation(
                "serve.lock must be a regular file".to_owned(),
            ));
        }
        flock_exclusive_nonblocking(file.as_fd()).map_err(|err| {
            if err.kind() == std::io::ErrorKind::WouldBlock {
                CliError::StateLocked
            } else {
                CliError::Io {
                    context: "cannot lock serve.lock".to_owned(),
                    source: err,
                }
            }
        })?;
        Ok(Self { _file: file })
    }
}

pub fn open_nofollow_dir(path: &Path, label: &str) -> Result<File, CliError> {
    let cwd = File::open(".").map_err(|source| CliError::Io {
        context: "cannot open working directory".to_owned(),
        source,
    })?;
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| CliError::Validation(format!("{label} contains a NUL byte")))?;
    openat(
        cwd.as_fd(),
        &c_path,
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
        0,
    )
    .map_err(|source| CliError::Io {
        context: format!("cannot open {label}"),
        source,
    })
}

/// Ensures `<tokspur_base>/<relative>` exists with `0o700` permissions and zero symlink traversal.
pub fn private_dir(tokspur_base: &Path, relative: &Path) -> Result<PathBuf, CliError> {
    if relative.is_absolute() {
        return Err(CliError::Validation(
            "Private state subpath must be relative".to_owned(),
        ));
    }
    for comp in relative.components() {
        if matches!(comp, Component::ParentDir | Component::CurDir) {
            return Err(CliError::Validation(
                "Private state must not traverse parent directories".to_owned(),
            ));
        }
    }
    let root_parent = tokspur_base.parent().ok_or_else(|| {
        CliError::Validation("Private state base has no parent directory".to_owned())
    })?;
    ensure_no_symlink_components(root_parent, "state root")?;
    if !root_parent.exists() {
        fs::create_dir_all(root_parent).map_err(|source| CliError::Io {
            context: "cannot create state root parent".to_owned(),
            source,
        })?;
    }

    let base_name = tokspur_base
        .file_name()
        .ok_or_else(|| CliError::Validation("invalid .tokspur directory name".to_owned()))?;
    let mut current = root_parent.to_path_buf();
    let mut parts = vec![base_name];
    for comp in relative.components() {
        parts.push(comp.as_os_str());
    }

    for part in parts {
        current.push(part);
        let c_cur = CString::new(current.as_os_str().as_bytes())
            .map_err(|_| CliError::Validation("state path contains NUL".to_owned()))?;
        match lstat_path(&c_cur) {
            Ok(st) => {
                if st.is_symlink() {
                    return Err(CliError::Validation(
                        "Private state must not be a symlink".to_owned(),
                    ));
                }
                if !st.is_dir() {
                    return Err(CliError::Validation(
                        "Private state component is not a directory".to_owned(),
                    ));
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // Create at 0700 so no other user can open the directory before the fchmod below.
                fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&current)
                    .map_err(|source| CliError::Io {
                        context: "cannot create private state directory".to_owned(),
                        source,
                    })?;
            }
            Err(source) => {
                return Err(CliError::Io {
                    context: "cannot inspect private state directory".to_owned(),
                    source,
                });
            }
        }
        let dir_fd = open_nofollow_dir(&current, "private state directory")?;
        fchmod(dir_fd.as_fd(), 0o700).map_err(|source| CliError::Io {
            context: "cannot set 0700 on private state directory".to_owned(),
            source,
        })?;
    }
    Ok(current)
}

/// Formats `Json` with 2-space indentation and sorted object keys matching Python `json.dumps(..., indent=2, sort_keys=True)`.
pub fn format_pretty_sorted_json(value: &Json) -> Result<Vec<u8>, CliError> {
    let mut out = Vec::new();
    write_pretty_json(value, 0, &mut out)?;
    out.push(b'\n');
    Ok(out)
}

fn write_pretty_json(value: &Json, indent: usize, out: &mut Vec<u8>) -> Result<(), CliError> {
    match value {
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => {
            let encoded = value.dump().map_err(|source| CliError::Json {
                context: "failed to serialise JSON scalar",
                source,
            })?;
            out.extend_from_slice(&encoded);
        }
        Json::Array(items) => {
            if items.is_empty() {
                out.extend_from_slice(b"[]");
                return Ok(());
            }
            out.extend_from_slice(b"[\n");
            for (idx, item) in items.iter().enumerate() {
                out.extend(std::iter::repeat_n(b' ', indent + 2));
                write_pretty_json(item, indent + 2, out)?;
                if idx + 1 < items.len() {
                    out.push(b',');
                }
                out.push(b'\n');
            }
            out.extend(std::iter::repeat_n(b' ', indent));
            out.push(b']');
        }
        Json::Object(fields) => {
            if fields.is_empty() {
                out.extend_from_slice(b"{}");
                return Ok(());
            }
            let mut sorted: Vec<&(Vec<u8>, Json)> = fields.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            out.extend_from_slice(b"{\n");
            for (idx, (key, val)) in sorted.iter().enumerate() {
                out.extend(std::iter::repeat_n(b' ', indent + 2));
                let key_json =
                    Json::String((*key).clone())
                        .dump()
                        .map_err(|source| CliError::Json {
                            context: "failed to serialise JSON key",
                            source,
                        })?;
                out.extend_from_slice(&key_json);
                out.extend_from_slice(b": ");
                write_pretty_json(val, indent + 2, out)?;
                if idx + 1 < sorted.len() {
                    out.push(b',');
                }
                out.push(b'\n');
            }
            out.extend(std::iter::repeat_n(b' ', indent));
            out.push(b'}');
        }
    }
    Ok(())
}

/// Atomically writes `0o600` sorted JSON under `tokspur_base / rel_path`, refusing symlink destinations.
pub fn write_private_json(
    tokspur_base: &Path,
    rel_path: &Path,
    value: &Json,
) -> Result<PathBuf, CliError> {
    write_private_bytes(tokspur_base, rel_path, &format_pretty_sorted_json(value)?)
}

/// Atomically publishes owner-only bytes without following a destination symlink.
pub(crate) fn write_private_bytes(
    tokspur_base: &Path,
    rel_path: &Path,
    payload: &[u8],
) -> Result<PathBuf, CliError> {
    let parent_rel = rel_path.parent().unwrap_or_else(|| Path::new(""));
    let file_name = rel_path
        .file_name()
        .ok_or_else(|| CliError::Validation("invalid JSON filename".to_owned()))?;
    let parent_dir = private_dir(tokspur_base, parent_rel)?;
    let target_path = parent_dir.join(file_name);

    let dir_fd = open_nofollow_dir(&parent_dir, "private JSON parent directory")?;
    let c_name = CString::new(file_name.as_bytes())
        .map_err(|_| CliError::Validation("JSON filename contains NUL".to_owned()))?;
    if let Ok(existing) = lstat_at(dir_fd.as_fd(), &c_name)
        && existing.is_symlink()
    {
        return Err(CliError::Validation(
            "Refusing to overwrite a symlink".to_owned(),
        ));
    }

    let mut rnd = [0u8; 12];
    random_bytes(&mut rnd);
    let tmp_name_str = format!(".write-{}.tmp", hex(&rnd));
    let tmp_c = CString::new(tmp_name_str.as_bytes())
        .map_err(|_| CliError::Validation("invalid temp name".to_owned()))?;
    let tmp_file = openat(
        dir_fd.as_fd(),
        &tmp_c,
        O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY | O_CLOEXEC,
        0o600,
    )
    .map_err(|source| CliError::Io {
        context: "cannot create temporary JSON file".to_owned(),
        source,
    })?;

    let write_res = (|| -> Result<(), CliError> {
        pwrite_all(tmp_file.as_fd(), payload, 0).map_err(|source| CliError::Io {
            context: "cannot write temporary JSON file".to_owned(),
            source,
        })?;
        fchmod(tmp_file.as_fd(), 0o600).map_err(|source| CliError::Io {
            context: "cannot chmod temporary JSON file".to_owned(),
            source,
        })?;
        fsync(tmp_file.as_fd()).map_err(|source| CliError::Io {
            context: "cannot fsync temporary JSON file".to_owned(),
            source,
        })?;
        if let Ok(existing) = lstat_at(dir_fd.as_fd(), &c_name)
            && existing.is_symlink()
        {
            return Err(CliError::Validation(
                "Refusing to overwrite a symlink".to_owned(),
            ));
        }
        fs::rename(parent_dir.join(&tmp_name_str), &target_path).map_err(|source| {
            CliError::Io {
                context: "cannot atomically publish JSON file".to_owned(),
                source,
            }
        })?;
        fsync(dir_fd.as_fd()).map_err(|source| CliError::Io {
            context: "cannot fsync JSON parent directory".to_owned(),
            source,
        })?;
        Ok(())
    })();

    drop(tmp_file);
    if write_res.is_err() {
        let _ignored = unlink_at(dir_fd.as_fd(), &tmp_c, false);
    }
    write_res?;
    Ok(target_path)
}

pub fn read_json_file(path: &Path) -> Result<Json, CliError> {
    let bytes = fs::read(path).map_err(|source| CliError::Io {
        context: format!(
            "cannot read {}",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("JSON file")
        ),
        source,
    })?;
    parse_json_limited(&bytes, JsonLimits::new(MAX_JSON_FILE, 500_000, 64)).map_err(|source| {
        CliError::Json {
            context: "invalid JSON document",
            source,
        }
    })
}

#[derive(Clone, PartialEq, Eq)]
struct FileIdentity {
    device: i32,
    inode: u64,
    size: u64,
    mtime_ns: i128,
    ctime_ns: i128,
    volume_uuid: Option<String>,
}

impl std::fmt::Debug for FileIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileIdentity")
            .field("device", &self.device)
            .field("inode", &self.inode)
            .field("size", &self.size)
            .field("mtime_ns", &self.mtime_ns)
            .field("ctime_ns", &self.ctime_ns)
            .field(
                "volume_uuid",
                &self.volume_uuid.as_ref().map(|_| "[redacted]"),
            )
            .finish()
    }
}

impl FileIdentity {
    fn from_stat(st: FileStat, volume_uuid: Option<String>) -> Self {
        Self {
            device: st.dev,
            inode: st.ino,
            size: st.size,
            mtime_ns: st.mtime_ns,
            ctime_ns: st.ctime_ns,
            volume_uuid,
        }
    }

    fn of_resolved_path(path: &Path) -> Result<Self, CliError> {
        let f = File::open(path).map_err(|source| CliError::Io {
            context: "cannot open asset for identity check".to_owned(),
            source,
        })?;
        let st = fstat(f.as_fd()).map_err(|source| CliError::Io {
            context: "cannot stat asset descriptor".to_owned(),
            source,
        })?;
        let volume_uuid = file_volume_uuid(f.as_fd())
            .map_err(|source| CliError::Io {
                context: "cannot read asset volume identity".to_owned(),
                source,
            })?
            .map(|value| hex(&value));
        Ok(Self::from_stat(st, volume_uuid))
    }

    fn matches_adopted(&self, recorded: &Self) -> bool {
        let same_volume = match &recorded.volume_uuid {
            Some(uuid) => self.volume_uuid.as_ref() == Some(uuid),
            None => self.device == recorded.device,
        };
        same_volume
            && self.inode == recorded.inode
            && self.size == recorded.size
            && self.mtime_ns == recorded.mtime_ns
            && self.ctime_ns == recorded.ctime_ns
    }
}

fn recorded_volume_uuid(asset: &Json) -> Result<Option<String>, CliError> {
    let value = asset.find("volume_uuid").map_err(|source| CliError::Json {
        context: "invalid asset volume identity",
        source,
    })?;
    if value.is_none() {
        return Ok(None);
    }
    let text = json_str(asset, "volume_uuid")?;
    if text.len() != 32 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CliError::Validation("invalid asset volume UUID".to_owned()));
    }
    Ok(Some(text.to_ascii_lowercase()))
}

pub(crate) fn json_str<'a>(obj: &'a Json, key: &str) -> Result<&'a str, CliError> {
    let raw = obj
        .find(key)
        .map_err(|source| CliError::Json {
            context: "expected JSON object",
            source,
        })?
        .ok_or_else(|| {
            CliError::Validation(format!(
                "missing configuration field '{key}'; re-run adopt/configure"
            ))
        })?
        .string()
        .map_err(|source| CliError::Json {
            context: "expected JSON string field",
            source,
        })?;
    std::str::from_utf8(raw)
        .map_err(|_| CliError::Validation(format!("non-UTF-8 string in field '{key}'")))
}

pub(crate) fn json_u64(obj: &Json, key: &str) -> Result<u64, CliError> {
    obj.find(key)
        .map_err(|source| CliError::Json {
            context: "expected JSON object",
            source,
        })?
        .ok_or_else(|| {
            CliError::Validation(format!(
                "missing configuration field '{key}'; re-run adopt/configure"
            ))
        })?
        .unsigned_integer()
        .map_err(|source| CliError::Json {
            context: "expected JSON unsigned integer",
            source,
        })
}

fn json_i128(obj: &Json, key: &str) -> Result<i128, CliError> {
    let lexeme = obj
        .find(key)
        .map_err(|source| CliError::Json {
            context: "expected JSON object",
            source,
        })?
        .ok_or_else(|| {
            CliError::Validation(format!(
                "missing configuration field '{key}'; re-run adopt/configure"
            ))
        })?
        .number_lexeme()
        .map_err(|source| CliError::Json {
            context: "expected JSON number",
            source,
        })?;
    lexeme
        .parse::<i128>()
        .map_err(|_| CliError::Validation(format!("invalid integer in field '{key}'")))
}

pub(crate) fn optional_asset_role(obj: &Json) -> Result<Option<&str>, CliError> {
    let Some(val) = obj.find("role").map_err(|source| CliError::Json {
        context: "expected JSON object",
        source,
    })?
    else {
        return Ok(None);
    };
    let raw = val.string().map_err(|source| CliError::Json {
        context: "expected JSON string in field 'role'",
        source,
    })?;
    let role = std::str::from_utf8(raw)
        .map_err(|_| CliError::Validation("non-UTF-8 string in field 'role'".to_owned()))?;
    if role != "derived_pack" {
        return Err(CliError::Validation(format!(
            "unsupported asset role '{role}'"
        )));
    }
    Ok(Some(role))
}

fn group_candidate_file_specs<'a>(
    assets_profile: &'a Json,
    group: &str,
    include_legacy_draft: bool,
) -> Result<Vec<&'a Vec<Json>>, CliError> {
    let group_obj =
        assets_profile.find(group).ok().flatten().ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing group".to_owned())
        })?;
    let primary_files = group_obj
        .find("files")
        .ok()
        .flatten()
        .and_then(|f| f.array().ok())
        .ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing group files".to_owned())
        })?;
    let mut out = vec![primary_files];
    if let Some(variants) = group_obj
        .find("variants")
        .map_err(|source| CliError::Json {
            context: "invalid variants field in flash-next-assets.json",
            source,
        })?
    {
        let arr = variants.array().map_err(|source| CliError::Json {
            context: "variants in flash-next-assets.json must be an array",
            source,
        })?;
        for variant in arr {
            let v_files = variant
                .find("files")
                .ok()
                .flatten()
                .and_then(|f| f.array().ok())
                .ok_or_else(|| {
                    CliError::Validation(
                        "flash-next-assets.json variant missing files list".to_owned(),
                    )
                })?;
            out.push(v_files);
        }
    }
    if include_legacy_draft
        && group == "draft"
        && let Some(legacy_files) = assets_profile
            .find("legacy_dflash2_draft")
            .ok()
            .flatten()
            .and_then(|g| g.find("files").ok().flatten())
            .and_then(|f| f.array().ok())
    {
        out.push(legacy_files);
    }
    Ok(out)
}

pub(crate) fn dflash2_profile_group_key(assets_profile: &Json) -> Option<&'static str> {
    if assets_profile
        .find("dflash2_draft")
        .ok()
        .flatten()
        .is_some()
    {
        Some("dflash2_draft")
    } else if assets_profile
        .find("legacy_dflash2_draft")
        .ok()
        .flatten()
        .is_some()
    {
        Some("legacy_dflash2_draft")
    } else {
        None
    }
}

pub(crate) fn repo_leaf_name(repo: &str) -> Result<&str, CliError> {
    let Some((owner, leaf)) = repo.split_once('/') else {
        return Err(CliError::Validation(format!(
            "invalid repository identifier '{repo}'"
        )));
    };
    let valid_part = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && !part.contains('/')
            && part
                .bytes()
                .all(|b| matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'))
    };
    if !valid_part(owner) || !valid_part(leaf) {
        return Err(CliError::Validation(format!(
            "invalid repository identifier '{repo}'"
        )));
    }
    Ok(leaf)
}

pub(crate) fn directory_has_candidate_assets(
    assets_profile: &Json,
    group: &str,
    dir: &Path,
) -> Result<bool, CliError> {
    if !dir.is_dir() {
        return Ok(false);
    }
    let candidates = group_candidate_file_specs(assets_profile, group, false)?;
    for cand in candidates {
        let mut all_exist = true;
        for spec in cand {
            let name = json_str(spec, "name")?;
            if !dir.join(name).is_file() {
                all_exist = false;
                break;
            }
        }
        if all_exist {
            return Ok(true);
        }
    }
    Ok(false)
}

fn asset_record_json(
    resolved: &Path,
    digest: String,
    recorded_identity: FileIdentity,
    role: Option<&str>,
) -> Result<Json, CliError> {
    let resolved_str = resolved
        .to_str()
        .ok_or_else(|| CliError::Validation("asset path is not valid UTF-8".to_owned()))?;
    let device_u64 = u64::try_from(recorded_identity.device)
        .map_err(|_| CliError::Validation("negative st_dev on asset".to_owned()))?;
    let mtime_u64 = u64::try_from(recorded_identity.mtime_ns)
        .map_err(|_| CliError::Validation("negative mtime_ns on asset".to_owned()))?;
    let ctime_u64 = u64::try_from(recorded_identity.ctime_ns)
        .map_err(|_| CliError::Validation("negative ctime_ns on asset".to_owned()))?;

    let mut record = vec![
        ("path", Json::from(resolved_str)),
        ("sha256", Json::from(digest)),
        ("device", Json::integer(device_u64)),
        ("inode", Json::integer(recorded_identity.inode)),
        ("size", Json::integer(recorded_identity.size)),
        ("mtime_ns", Json::integer(mtime_u64)),
        ("ctime_ns", Json::integer(ctime_u64)),
    ];
    if let Some(uuid) = recorded_identity.volume_uuid {
        record.push(("volume_uuid", Json::from(uuid)));
    }
    if let Some(role_name) = role {
        record.push(("role", Json::from(role_name)));
    }
    Ok(Json::new_object(record))
}

pub(crate) fn verify_asset(path: &Path, spec: &Json) -> Result<Json, CliError> {
    let spec_name = json_str(spec, "name")?;
    let expected_size = json_u64(spec, "size")?;
    let expected_sha256 = json_str(spec, "sha256")?;
    let role = optional_asset_role(spec)?;
    let is_derived_pack = role == Some("derived_pack");

    let resolved = fs::canonicalize(path).map_err(|source| CliError::Io {
        context: format!("cannot resolve asset {spec_name}"),
        source,
    })?;
    let before = FileIdentity::of_resolved_path(&resolved)?;
    if before.size != expected_size {
        let short = resolved
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(spec_name);
        return Err(CliError::Validation(format!("Unexpected size: {short}")));
    }

    let parent = resolved
        .parent()
        .ok_or_else(|| CliError::Validation("asset path has no parent".to_owned()))?;
    let file_name = resolved
        .file_name()
        .ok_or_else(|| CliError::Validation("asset path has no filename".to_owned()))?;
    let dir_fd = open_nofollow_dir(parent, "asset parent directory")?;
    let c_name = CString::new(file_name.as_bytes())
        .map_err(|_| CliError::Validation("asset filename contains NUL".to_owned()))?;
    let file = openat(
        dir_fd.as_fd(),
        &c_name,
        O_RDONLY | O_NOFOLLOW | O_CLOEXEC,
        0,
    )
    .map_err(|source| CliError::Io {
        context: format!("cannot open asset {spec_name}"),
        source,
    })?;
    let opened = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: "cannot stat opened asset".to_owned(),
        source,
    })?;
    if !opened.is_regular() {
        return Err(CliError::Validation(
            "Asset must be a regular file".to_owned(),
        ));
    }
    if is_derived_pack && opened.nlink != 1 {
        return Err(CliError::Validation(
            "Derived pack asset must not have multiple hard links".to_owned(),
        ));
    }
    set_nocache(file.as_fd(), true).map_err(|source| CliError::Io {
        context: "cannot set F_NOCACHE on asset".to_owned(),
        source,
    })?;

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; ASSET_STREAM_CHUNK];
    let mut at = 0u64;
    while at < before.size {
        let take = (before.size - at).min(ASSET_STREAM_CHUNK as u64) as usize;
        pread_exact(file.as_fd(), &mut buf[..take], at).map_err(|source| CliError::Io {
            context: format!("failed reading asset {spec_name}"),
            source,
        })?;
        hasher.update(&buf[..take]);
        at += take as u64;
    }

    let after_open = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: "cannot re-stat opened asset".to_owned(),
        source,
    })?;
    if (after_open.dev, after_open.ino) != (before.device, before.inode) {
        return Err(CliError::Validation(
            "Asset changed while opening".to_owned(),
        ));
    }
    if FileIdentity::of_resolved_path(&resolved)? != before {
        return Err(CliError::Validation(
            "Asset changed during verification".to_owned(),
        ));
    }
    let digest = hex(&hasher.finalize());
    if digest != expected_sha256 {
        let short = resolved
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(spec_name);
        return Err(CliError::Validation(format!("Checksum mismatch: {short}")));
    }

    let mut recorded_identity = before;
    if is_derived_pack {
        if (after_open.mode & 0o222) != 0 {
            fchmod(file.as_fd(), 0o444).map_err(|source| CliError::Io {
                context: format!("cannot seal derived pack {spec_name} read-only (0444)"),
                source,
            })?;
        }
        let sealed = fstat(file.as_fd()).map_err(|source| CliError::Io {
            context: "cannot stat sealed derived pack".to_owned(),
            source,
        })?;
        if (sealed.mode & 0o222) != 0 || sealed.nlink != 1 {
            return Err(CliError::Validation(
                "Derived pack asset must be read-only with a single link".to_owned(),
            ));
        }
        drop(file);
        let post_seal = FileIdentity::of_resolved_path(&resolved)?;
        if (
            post_seal.device,
            post_seal.inode,
            post_seal.size,
            post_seal.mtime_ns,
        ) != (
            recorded_identity.device,
            recorded_identity.inode,
            recorded_identity.size,
            recorded_identity.mtime_ns,
        ) {
            return Err(CliError::Validation(
                "Asset changed during verification".to_owned(),
            ));
        }
        recorded_identity = post_seal;
    } else {
        drop(file);
    }

    asset_record_json(&resolved, digest, recorded_identity, role)
}

/// Verifies a downloaded `.part` file in a single streaming SHA-256 pass, unlinking on checksum mismatch, and atomically publishes `dest_path`.
pub(crate) fn publish_verified_part(
    part_path: &Path,
    dest_path: &Path,
    spec: &Json,
) -> Result<Json, CliError> {
    let spec_name = json_str(spec, "name")?;
    let expected_size = json_u64(spec, "size")?;
    let expected_sha256 = json_str(spec, "sha256")?;
    let role = optional_asset_role(spec)?;
    let is_derived_pack = role == Some("derived_pack");

    let parent = dest_path
        .parent()
        .ok_or_else(|| CliError::Validation("asset path has no parent".to_owned()))?;
    if part_path.parent() != Some(parent) {
        return Err(CliError::Validation(
            "partial asset path must share parent directory with destination".to_owned(),
        ));
    }
    let dest_name = dest_path
        .file_name()
        .ok_or_else(|| CliError::Validation("asset path has no filename".to_owned()))?;
    let part_name = part_path
        .file_name()
        .ok_or_else(|| CliError::Validation("partial asset path has no filename".to_owned()))?;
    let short = dest_name.to_str().unwrap_or(spec_name);

    let dir_fd = open_nofollow_dir(parent, "asset parent directory")?;
    let part_c = CString::new(part_name.as_bytes())
        .map_err(|_| CliError::Validation("asset filename contains NUL".to_owned()))?;
    let dest_c = CString::new(dest_name.as_bytes())
        .map_err(|_| CliError::Validation("asset filename contains NUL".to_owned()))?;

    let lstat_before = lstat_at(dir_fd.as_fd(), &part_c).map_err(|source| CliError::Io {
        context: format!("cannot stat downloaded {spec_name}"),
        source,
    })?;
    if !lstat_before.is_regular() {
        return Err(CliError::Validation(
            "Downloaded asset must be a regular file".to_owned(),
        ));
    }
    if lstat_before.nlink != 1 {
        return Err(CliError::Validation(
            "Downloaded asset must not have multiple hard links".to_owned(),
        ));
    }
    if lstat_before.size != expected_size {
        return Err(CliError::Validation(format!("Unexpected size: {short}")));
    }

    let file = openat(
        dir_fd.as_fd(),
        &part_c,
        O_RDONLY | O_NOFOLLOW | O_CLOEXEC,
        0,
    )
    .map_err(|source| CliError::Io {
        context: format!("cannot open downloaded {spec_name}"),
        source,
    })?;
    let opened = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: "cannot stat opened downloaded asset".to_owned(),
        source,
    })?;
    if !opened.is_regular()
        || opened.nlink != 1
        || (
            opened.dev,
            opened.ino,
            opened.size,
            opened.mtime_ns,
            opened.ctime_ns,
        ) != (
            lstat_before.dev,
            lstat_before.ino,
            lstat_before.size,
            lstat_before.mtime_ns,
            lstat_before.ctime_ns,
        )
    {
        return Err(CliError::Validation(
            "Asset changed while opening".to_owned(),
        ));
    }
    set_nocache(file.as_fd(), true).map_err(|source| CliError::Io {
        context: "cannot set F_NOCACHE on downloaded asset".to_owned(),
        source,
    })?;

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; ASSET_STREAM_CHUNK];
    let mut at = 0u64;
    while at < opened.size {
        let take = (opened.size - at).min(ASSET_STREAM_CHUNK as u64) as usize;
        pread_exact(file.as_fd(), &mut buf[..take], at).map_err(|source| CliError::Io {
            context: format!("failed reading downloaded {spec_name}"),
            source,
        })?;
        hasher.update(&buf[..take]);
        at += take as u64;
    }

    let after_hash = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: "cannot re-stat downloaded asset".to_owned(),
        source,
    })?;
    let lstat_after = lstat_at(dir_fd.as_fd(), &part_c).map_err(|source| CliError::Io {
        context: "cannot re-inspect downloaded asset".to_owned(),
        source,
    })?;
    if (
        after_hash.dev,
        after_hash.ino,
        after_hash.size,
        after_hash.mtime_ns,
        after_hash.ctime_ns,
    ) != (
        opened.dev,
        opened.ino,
        opened.size,
        opened.mtime_ns,
        opened.ctime_ns,
    ) || (
        lstat_after.dev,
        lstat_after.ino,
        lstat_after.size,
        lstat_after.mtime_ns,
        lstat_after.ctime_ns,
    ) != (
        opened.dev,
        opened.ino,
        opened.size,
        opened.mtime_ns,
        opened.ctime_ns,
    ) {
        return Err(CliError::Validation(
            "Asset changed during verification".to_owned(),
        ));
    }

    let digest = hex(&hasher.finalize());
    if digest != expected_sha256 {
        drop(file);
        let _ignored = unlink_at(dir_fd.as_fd(), &part_c, false);
        return Err(CliError::Validation(format!("Checksum mismatch: {short}")));
    }

    match lstat_at(dir_fd.as_fd(), &dest_c) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(CliError::Validation(format!(
                "Refusing to overwrite existing asset: {short}"
            )));
        }
        Err(source) => {
            return Err(CliError::Io {
                context: "cannot inspect destination asset".to_owned(),
                source,
            });
        }
    }

    let target_mode = if is_derived_pack { 0o444 } else { 0o644 };
    fchmod(file.as_fd(), target_mode).map_err(|source| CliError::Io {
        context: format!("cannot set permissions on {spec_name}"),
        source,
    })?;
    fsync(file.as_fd()).map_err(|source| CliError::Io {
        context: format!("cannot fsync {spec_name}"),
        source,
    })?;
    fs::rename(part_path, dest_path).map_err(|source| CliError::Io {
        context: format!("cannot publish verified asset {spec_name}"),
        source,
    })?;
    fsync(dir_fd.as_fd()).map_err(|source| CliError::Io {
        context: "cannot fsync asset parent directory".to_owned(),
        source,
    })?;

    let sealed = fstat(file.as_fd()).map_err(|source| CliError::Io {
        context: "cannot stat published asset".to_owned(),
        source,
    })?;
    if is_derived_pack && ((sealed.mode & 0o222) != 0 || sealed.nlink != 1) {
        return Err(CliError::Validation(
            "Derived pack asset must be read-only with a single link".to_owned(),
        ));
    }
    drop(file);

    let resolved = fs::canonicalize(dest_path).map_err(|source| CliError::Io {
        context: format!("cannot resolve asset {spec_name}"),
        source,
    })?;
    let published_identity = FileIdentity::of_resolved_path(&resolved)?;
    if (
        published_identity.device,
        published_identity.inode,
        published_identity.size,
        published_identity.mtime_ns,
    ) != (opened.dev, opened.ino, opened.size, opened.mtime_ns)
    {
        return Err(CliError::Validation(
            "Asset changed during verification".to_owned(),
        ));
    }
    asset_record_json(&resolved, digest, published_identity, role)
}

pub(crate) fn check_adoption_group(
    adoption: &Json,
    assets_profile: &Json,
    adoption_key: &str,
    profile_key: &str,
    include_legacy_draft: bool,
) -> Result<(), CliError> {
    let group_obj = adoption
        .find(adoption_key)
        .map_err(|source| CliError::Json {
            context: "invalid adoption.json",
            source,
        })?
        .ok_or_else(|| {
            CliError::Validation(format!(
                "missing configuration field '{adoption_key}'; re-run adopt/configure"
            ))
        })?;
    let dir_str = json_str(group_obj, "directory")?;
    let assets = group_obj
        .find("assets")
        .map_err(|source| CliError::Json {
            context: "invalid adoption group",
            source,
        })?
        .ok_or_else(|| {
            CliError::Validation(
                "missing configuration field 'assets'; re-run adopt/configure".to_owned(),
            )
        })?
        .array()
        .map_err(|source| CliError::Json {
            context: "adoption assets must be an array",
            source,
        })?;

    let candidate_specs =
        group_candidate_file_specs(assets_profile, profile_key, include_legacy_draft)?;
    let same_len: Vec<&Vec<Json>> = candidate_specs
        .into_iter()
        .filter(|spec_files| assets.len() == spec_files.len())
        .collect();
    if same_len.is_empty() {
        return Err(CliError::Validation(
            "Incomplete adopted asset set".to_owned(),
        ));
    }

    let mut matched_spec: Option<&Vec<Json>> = None;
    for spec_files in &same_len {
        let mut all_match = true;
        for (asset, spec) in assets.iter().zip(spec_files.iter()) {
            let spec_sha = json_str(spec, "sha256")?;
            let asset_sha = json_str(asset, "sha256")?;
            let spec_role = optional_asset_role(spec)?;
            let asset_role = optional_asset_role(asset)?;
            if asset_sha != spec_sha || asset_role != spec_role {
                all_match = false;
                break;
            }
        }
        if all_match {
            matched_spec = Some(*spec_files);
            break;
        }
    }
    let matched_spec = matched_spec.ok_or_else(|| {
        CliError::Validation("Asset manifest differs from the pinned profile".to_owned())
    })?;

    for (asset, spec) in assets.iter().zip(matched_spec.iter()) {
        let spec_name = json_str(spec, "name")?;
        let is_derived_pack = optional_asset_role(spec)? == Some("derived_pack");
        let asset_path = PathBuf::from(json_str(asset, "path")?);
        let actual = FileIdentity::of_resolved_path(&asset_path)
            .map_err(|_| CliError::Validation(format!("Asset changed; re-adopt: {spec_name}")))?;
        if is_derived_pack {
            let f = File::open(&asset_path).map_err(|_| {
                CliError::Validation(format!("Asset changed; re-adopt: {spec_name}"))
            })?;
            let st = fstat(f.as_fd()).map_err(|_| {
                CliError::Validation(format!("Asset changed; re-adopt: {spec_name}"))
            })?;
            if (st.mode & 0o222) != 0 || st.nlink != 1 {
                return Err(CliError::Validation(format!(
                    "Asset changed; re-adopt: {spec_name}"
                )));
            }
        }
        let recorded = FileIdentity {
            device: i32::try_from(json_i128(asset, "device")?)
                .map_err(|_| CliError::Validation("invalid recorded asset device".to_owned()))?,
            inode: json_u64(asset, "inode")?,
            size: json_u64(asset, "size")?,
            mtime_ns: json_i128(asset, "mtime_ns")?,
            ctime_ns: json_i128(asset, "ctime_ns")?,
            volume_uuid: recorded_volume_uuid(asset)?,
        };
        if recorded.device < 0 {
            return Err(CliError::Validation(
                "invalid recorded asset device".to_owned(),
            ));
        }
        if !actual.matches_adopted(&recorded) {
            return Err(CliError::Validation(format!(
                "Asset changed; re-adopt: {spec_name}"
            )));
        }
        let selected = Path::new(dir_str).join(spec_name);
        let resolved_selected = fs::canonicalize(&selected).map_err(|_| {
            CliError::Validation("Selected asset link changed; re-adopt".to_owned())
        })?;
        if resolved_selected != asset_path {
            return Err(CliError::Validation(
                "Selected asset link changed; re-adopt".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Validates full adoption against the pinned asset profile, including legacy draft records and the Python reference in `src/tokspur/local.py`.
pub fn check_adoption(adoption: &Json, assets_profile: &Json) -> Result<(), CliError> {
    let schema = adoption
        .find("schema")
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok());
    let verification = adoption
        .find("verification")
        .ok()
        .flatten()
        .and_then(|v| v.string().ok());
    if schema != Some(1) || verification != Some(b"full-sha256".as_slice()) {
        return Err(CliError::Validation(
            "Full asset verification is required".to_owned(),
        ));
    }

    for group in ["target", "draft"] {
        check_adoption_group(adoption, assets_profile, group, group, true)?;
    }
    if adoption
        .find("dflash2_draft")
        .map_err(|source| CliError::Json {
            context: "invalid adoption.json",
            source,
        })?
        .is_some()
    {
        let profile_key = dflash2_profile_group_key(assets_profile).ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing dflash2_draft".to_owned())
        })?;
        check_adoption_group(
            adoption,
            assets_profile,
            "dflash2_draft",
            profile_key,
            false,
        )?;
    }
    Ok(())
}

pub(crate) fn verify_group_directory(
    specs: &Json,
    profile_group: &str,
    label: &str,
    raw_dir: &Path,
    out: &mut dyn Write,
) -> Result<Json, CliError> {
    let directory = fs::canonicalize(raw_dir).map_err(|source| CliError::Io {
        context: format!("cannot resolve {label} directory"),
        source,
    })?;
    let dir_str = directory
        .to_str()
        .ok_or_else(|| CliError::Validation("directory path is not valid UTF-8".to_owned()))?;
    let candidates = group_candidate_file_specs(specs, profile_group, false)?;
    let spec_files = if candidates.len() == 1 {
        candidates[0]
    } else {
        let mut present = Vec::new();
        for cand in &candidates {
            let mut all_exist = true;
            for spec in *cand {
                let name = json_str(spec, "name")?;
                if !directory.join(name).exists() {
                    all_exist = false;
                    break;
                }
            }
            if all_exist {
                present.push(*cand);
            }
        }
        if present.len() > 1 {
            return Err(CliError::Validation(format!(
                "Ambiguous {label} asset variants present in {}",
                directory.display()
            )));
        }
        present.first().copied().unwrap_or(candidates[0])
    };
    let mut verified = Vec::with_capacity(spec_files.len());
    for spec in spec_files {
        let name = json_str(spec, "name")?;
        writeln!(out, "Verifying {label}: {name}").map_err(|source| CliError::Io {
            context: "stdout write failed".to_owned(),
            source,
        })?;
        out.flush().map_err(|source| CliError::Io {
            context: "stdout flush failed".to_owned(),
            source,
        })?;
        verified.push(verify_asset(&directory.join(name), spec)?);
    }
    Ok(Json::new_object([
        ("directory", Json::from(dir_str)),
        ("assets", Json::Array(verified)),
    ]))
}

pub(crate) fn preserved_dflash2_group(tokspur_base: &Path, specs: &Json) -> Option<Json> {
    let profile_key = dflash2_profile_group_key(specs)?;
    let existing = read_json_file(&tokspur_base.join("adoption.json")).ok()?;
    let group_obj = existing.find("dflash2_draft").ok().flatten()?.clone();
    check_adoption_group(&existing, specs, "dflash2_draft", profile_key, false).ok()?;
    Some(group_obj)
}

pub(crate) fn finalize_and_write_adoption(
    tokspur_base: &Path,
    specs: &Json,
    adopted: Json,
    out: &mut dyn Write,
) -> Result<Json, CliError> {
    let target = adopted
        .find("target")
        .map_err(|source| CliError::Json {
            context: "invalid target adoption",
            source,
        })?
        .ok_or_else(|| CliError::Validation("missing target adoption".to_owned()))?;
    let assets = target
        .find("assets")
        .map_err(|source| CliError::Json {
            context: "invalid target adoption assets",
            source,
        })?
        .ok_or_else(|| CliError::Validation("missing target assets".to_owned()))?
        .array()
        .map_err(|source| CliError::Json {
            context: "invalid target assets",
            source,
        })?;
    let first = assets
        .first()
        .ok_or_else(|| CliError::Validation("empty target assets".to_owned()))?;
    let first_path = Path::new(json_str(first, "path")?);
    if json_str(first, "sha256")? == crate::text_assets::SHARD1_SHA256
        || crate::text_assets::has_flash_next_text_metadata(first_path)
    {
        crate::text_assets::derive(tokspur_base, first_path)?;
        writeln!(
            out,
            "Pinned tokenizer and chat template derived from verified metadata."
        )
        .map_err(|source| CliError::Io {
            context: "stdout write failed".to_owned(),
            source,
        })?;
    }
    check_adoption(&adopted, specs)?;
    write_private_json(tokspur_base, Path::new("adoption.json"), &adopted)?;
    Ok(adopted)
}

pub(crate) fn adopt_weights_locked(
    layout: &RuntimeLayout,
    tokspur_base: &Path,
    target_dir: &Path,
    draft_dir: &Path,
    dflash2_dir: Option<&Path>,
    out: &mut dyn Write,
) -> Result<Json, CliError> {
    let specs = read_json_file(&layout.assets_profile_path())?;
    let mut fields: Vec<(&str, Json)> = vec![
        ("schema", Json::integer(1)),
        ("verification", Json::from("full-sha256")),
    ];
    for (group, raw_dir) in [("target", target_dir), ("draft", draft_dir)] {
        fields.push((
            group,
            verify_group_directory(&specs, group, group, raw_dir, out)?,
        ));
    }
    if let Some(df_dir) = dflash2_dir {
        let profile_key = dflash2_profile_group_key(&specs).ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing dflash2_draft".to_owned())
        })?;
        fields.push((
            "dflash2_draft",
            verify_group_directory(&specs, profile_key, "dflash2", df_dir, out)?,
        ));
    } else if let Some(existing_df) = preserved_dflash2_group(tokspur_base, &specs) {
        fields.push(("dflash2_draft", existing_df));
    }

    let adopted = finalize_and_write_adoption(tokspur_base, &specs, Json::new_object(fields), out)?;
    writeln!(
        out,
        "Existing weights verified in place. No model files copied or downloaded."
    )
    .map_err(|source| CliError::Io {
        context: "stdout write failed".to_owned(),
        source,
    })?;
    Ok(adopted)
}

/// Verifies `dflash2_dir` against the pinned asset profile and records `"dflash2_draft"` in `adoption.json` under an already-held `StateLock`.
pub(crate) fn record_dflash2_in_adoption(
    tokspur_base: &Path,
    specs: &Json,
    adoption: &Json,
    dflash2_dir: &Path,
    out: &mut dyn Write,
) -> Result<(Json, PathBuf), CliError> {
    let profile_key = dflash2_profile_group_key(specs).ok_or_else(|| {
        CliError::Validation("flash-next-assets.json missing dflash2_draft".to_owned())
    })?;
    let df_group = verify_group_directory(specs, profile_key, "dflash2", dflash2_dir, out)?;
    let resolved_dir = PathBuf::from(json_str(&df_group, "directory")?);
    let mut updated = adoption.clone();
    if let Json::Object(ref mut obj) = updated {
        obj.retain(|(k, _)| k != b"dflash2_draft");
        obj.push((b"dflash2_draft".to_vec(), df_group));
    }
    check_adoption(&updated, specs)?;
    write_private_json(tokspur_base, Path::new("adoption.json"), &updated)?;
    Ok((updated, resolved_dir))
}

/// Runs `tokspur adopt --target <dir> --draft <dir>`.
pub fn adopt_weights(
    layout: &RuntimeLayout,
    target_dir: &Path,
    draft_dir: &Path,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    adopt_weights_with_dflash2(layout, target_dir, draft_dir, None, out)
}

/// Runs `tokspur adopt` with optional `--dflash2-dir`.
pub fn adopt_weights_with_dflash2(
    layout: &RuntimeLayout,
    target_dir: &Path,
    draft_dir: &Path,
    dflash2_dir: Option<&Path>,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    let tokspur_base = layout.tokspur_base()?;
    let _lock = StateLock::acquire(&tokspur_base)?;
    adopt_weights_locked(
        layout,
        &tokspur_base,
        target_dir,
        draft_dir,
        dflash2_dir,
        out,
    )?;
    Ok(())
}

/// Runs `tokspur adopt --check-only`.
pub fn check_adopted_weights(layout: &RuntimeLayout, out: &mut dyn Write) -> Result<(), CliError> {
    let tokspur_base = layout.tokspur_base()?;
    let _lock = StateLock::acquire(&tokspur_base)?;
    let adoption = read_json_file(&tokspur_base.join("adoption.json"))?;
    let assets_profile = read_json_file(&layout.assets_profile_path())?;
    check_adoption(&adoption, &assets_profile)?;
    writeln!(
        out,
        "Existing weights verified in place. No model files copied or downloaded."
    )
    .map_err(|source| CliError::Io {
        context: "stdout write failed".to_owned(),
        source,
    })?;
    Ok(())
}

pub(crate) fn configure_runtime_locked(
    layout: &RuntimeLayout,
    tokspur_base: &Path,
    port: u32,
    require_port_free: bool,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    if !(1024..=65535).contains(&port) {
        return Err(CliError::Validation(
            "Use an unprivileged local port".to_owned(),
        ));
    }
    let port_u16 = port as u16;
    if require_port_free && !port_available(port_u16) {
        return Err(CliError::Validation(format!(
            "Port {port} is occupied; configuration left unchanged"
        )));
    }

    let adoption = read_json_file(&tokspur_base.join("adoption.json"))?;
    let assets_profile = read_json_file(&layout.assets_profile_path())?;
    check_adoption(&adoption, &assets_profile)?;
    let profile = read_json_file(&layout.hardware_profile_path())?;
    let public_model = json_str(&profile, "public_model")?.to_owned();

    let state = private_dir(tokspur_base, Path::new("state"))?;
    let owner_path = state.join("owner.json");
    let expected_owner = Json::new_object([
        ("project", Json::from("tokspur")),
        ("schema", Json::integer(1)),
    ]);
    if owner_path.exists() {
        let actual_owner = read_json_file(&owner_path)?;
        let owner_ok = actual_owner
            .find("project")
            .ok()
            .flatten()
            .and_then(|v| v.string().ok())
            == Some(b"tokspur".as_slice())
            && actual_owner
                .find("schema")
                .ok()
                .flatten()
                .and_then(|v| v.unsigned_integer().ok())
                == Some(1)
            && actual_owner.object().is_ok_and(|o| o.len() == 2);
        if !owner_ok {
            return Err(CliError::Validation("Unknown state owner".to_owned()));
        }
    }
    write_private_json(tokspur_base, Path::new("state/owner.json"), &expected_owner)?;

    // Keep a key only if the server would admit it: a single-link 0600 file of 64 lowercase hex digits and a newline. Anything else, such as the empty file an interrupted write could leave, is replaced. The replacement is published atomically, so an interruption leaves the old state.
    let key_file = state.join("api-key");
    let key_valid = match fs::symlink_metadata(&key_file) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(CliError::Validation(
                "state/api-key must not be a symlink".to_owned(),
            ));
        }
        Ok(meta) => {
            meta.file_type().is_file()
                && meta.len() == 65
                && meta.permissions().mode() & 0o777 == 0o600
                && meta.nlink() == 1
                && fs::read(&key_file).is_ok_and(|bytes| {
                    bytes.split_last().is_some_and(|(last, digits)| {
                        *last == b'\n'
                            && digits.len() == 64
                            && digits
                                .iter()
                                .all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
                    })
                })
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(source) => {
            return Err(CliError::Io {
                context: "cannot inspect state/api-key".to_owned(),
                source,
            });
        }
    };
    if !key_valid {
        let mut secret = [0u8; 32];
        random_bytes(&mut secret);
        write_private_bytes(
            tokspur_base,
            Path::new("state/api-key"),
            format!("{}\n", hex(&secret)).as_bytes(),
        )?;
    }

    let target_dir = PathBuf::from(json_str(
        adoption
            .find("target")
            .ok()
            .flatten()
            .ok_or_else(|| CliError::Validation("adoption missing target".to_owned()))?,
        "directory",
    )?);
    if target_dir.is_dir() {
        let model_dir = private_dir(tokspur_base, Path::new("state/models"))?;
        let link = model_dir.join(&public_model);
        if let Ok(meta) = fs::symlink_metadata(&link) {
            if meta.file_type().is_symlink() {
                let resolved_link = fs::canonicalize(&link).unwrap_or_default();
                let resolved_target =
                    fs::canonicalize(&target_dir).map_err(|source| CliError::Io {
                        context: "cannot resolve adopted target directory".to_owned(),
                        source,
                    })?;
                if resolved_link != resolved_target {
                    fs::remove_file(&link).map_err(|source| CliError::Io {
                        context: "cannot update model symlink".to_owned(),
                        source,
                    })?;
                    symlink(&target_dir, &link).map_err(|source| CliError::Io {
                        context: "cannot create model symlink".to_owned(),
                        source,
                    })?;
                }
            } else {
                return Err(CliError::Validation(
                    "Refusing to replace an existing model directory".to_owned(),
                ));
            }
        } else {
            symlink(&target_dir, &link).map_err(|source| CliError::Io {
                context: "cannot create model symlink".to_owned(),
                source,
            })?;
        }
    }

    write_private_json(
        tokspur_base,
        Path::new("configured.json"),
        &Json::new_object([
            ("port", Json::integer(u64::from(port))),
            ("profile", profile),
            ("model", Json::from(public_model)),
        ]),
    )?;
    writeln!(
        out,
        "Configured tokspur at http://127.0.0.1:{port}; inference not started."
    )
    .map_err(|source| CliError::Io {
        context: "stdout write failed".to_owned(),
        source,
    })?;
    Ok(())
}

/// Runs `tokspur configure --port <port>` (`cli.py:48-100`).
pub fn configure_runtime(
    layout: &RuntimeLayout,
    port: u32,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    let tokspur_base = layout.tokspur_base()?;
    let _lock = StateLock::acquire(&tokspur_base)?;
    configure_runtime_locked(layout, &tokspur_base, port, true, out)
}

/// Pure process-matching predicate ported one to one from `local.py:172-201`.
pub fn is_inference_worker_process(snap: &ProcessSnapshot, self_pid: i32) -> bool {
    if snap.pid == self_pid || snap.is_zombie() {
        return false;
    }
    const WORKER_NAMES: &[&str] = &[
        "omlx",
        "omlx-server",
        "splash",
        "llama-server",
        "tokspur-native",
        "branch8-target",
        "stage8-target",
        SCALAR_WORKER_NAME,
    ];

    let args_4 = &snap.argv[..snap.argv.len().min(4)];
    let mut names = BTreeSet::new();
    for arg in args_4 {
        if let Some(base) = Path::new(arg).file_name().and_then(|s| s.to_str()) {
            names.insert(base);
        }
    }
    if !snap.comm.is_empty() {
        names.insert(snap.comm.as_str());
    }

    let mut is_worker = names.iter().any(|n| WORKER_NAMES.contains(n));
    // Darwin p_comm fallback only when foreign/unreadable argv is empty; never match when readable argv disagrees.
    let scalar_short = &SCALAR_WORKER_NAME[..DARWIN_MAXCOMLEN];
    is_worker |= snap.argv.is_empty() && snap.comm == scalar_short;
    is_worker |= args_4.iter().any(|a| a == "mlx_lm.server");
    is_worker |= names.contains("ollama") && args_4.iter().any(|arg| arg == "runner");
    is_worker
}

/// Scans all live OS processes via `libproc` and returns active inference workers (`[{"pid": ..., "runtime": "inference worker"}]`).
pub fn inference_processes() -> Result<Vec<i32>, CliError> {
    let self_pid = std::process::id() as i32;
    let procs = list_live_processes().map_err(|source| CliError::Io {
        context: "cannot enumerate live OS processes".to_owned(),
        source,
    })?;
    let mut pids: Vec<i32> = procs
        .iter()
        .filter(|snap| is_inference_worker_process(snap, self_pid))
        .map(|snap| snap.pid)
        .collect();
    pids.sort_unstable();
    pids.dedup();
    Ok(pids)
}

/// Builds the sanitised child environment matching `local.py:isolated_environment`.
pub fn isolated_environment(tokspur_base: &Path) -> Result<BTreeMap<String, String>, CliError> {
    const BLOCKED_PREFIXES: &[&str] = &["HF_", "HUGGING_FACE_", "HUGGINGFACE_"];
    let mut env = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let k = key.into_string().map_err(|_| {
            CliError::Validation("environment variable names must be UTF-8".to_owned())
        })?;
        if BLOCKED_PREFIXES.iter().any(|p| k.starts_with(p))
            || k == "PYTHONPATH"
            || k == "PYTHONHOME"
        {
            continue;
        }
        let v = value.into_string().map_err(|_| {
            CliError::Validation("environment variable values must be UTF-8".to_owned())
        })?;
        env.insert(k, v);
    }
    let hf_home = private_dir(tokspur_base, Path::new("state/hf-home"))?;
    let tmp_dir = private_dir(tokspur_base, Path::new("state/tmp"))?;
    env.insert("HF_HOME".to_owned(), hf_home.to_string_lossy().into_owned());
    env.insert("HF_HUB_OFFLINE".to_owned(), "1".to_owned());
    env.insert("HF_HUB_DISABLE_IMPLICIT_TOKEN".to_owned(), "1".to_owned());
    env.insert("HF_HUB_DISABLE_TELEMETRY".to_owned(), "1".to_owned());
    env.insert("DO_NOT_TRACK".to_owned(), "1".to_owned());
    env.insert("TMPDIR".to_owned(), tmp_dir.to_string_lossy().into_owned());
    env.insert("NO_PROXY".to_owned(), "localhost,127.0.0.1,::1".to_owned());
    fs::set_permissions(&hf_home, fs::Permissions::from_mode(0o700)).map_err(|source| {
        CliError::Io {
            context: "cannot set state/hf-home permissions".to_owned(),
            source,
        }
    })?;
    fs::set_permissions(&tmp_dir, fs::Permissions::from_mode(0o700)).map_err(|source| {
        CliError::Io {
            context: "cannot set state/tmp permissions".to_owned(),
            source,
        }
    })?;
    Ok(env)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    #[test]
    fn file_identity_debug_redacts_volume_uuid() {
        let private_uuid = "1234567890abcdef1234567890abcdef";
        let identity = FileIdentity {
            device: 1,
            inode: 2,
            size: 3,
            mtime_ns: 4,
            ctime_ns: 5,
            volume_uuid: Some(private_uuid.to_owned()),
        };
        let rendered = format!("{identity:?}");
        assert!(!rendered.contains(private_uuid));
        assert!(rendered.contains("[redacted]"));
    }

    fn snap(pid: i32, comm: &str, argv: &[&str]) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            ppid: 1,
            pgid: pid,
            uid: 501,
            status: 2,
            start_tvsec: 100,
            start_tvusec: 0,
            comm: comm.to_owned(),
            name: comm.to_owned(),
            executable_path: None,
            argv: argv.iter().map(OsString::from).collect(),
        }
    }

    #[test]
    fn worker_classification_matches_python_local_contract() {
        let full = SCALAR_WORKER_NAME;
        let short = &SCALAR_WORKER_NAME[..DARWIN_MAXCOMLEN];
        let cases = [
            (
                snap(10, "", &["/x/build/owned-native/tokspur-native", "target"]),
                true,
            ),
            (snap(11, "omlx-server", &[]), true),
            (
                snap(12, "bash", &["bash", "-c", "echo tokspur-native"]),
                false,
            ),
            (snap(13, "Python", &["python", "-m", "pytest"]), false),
            (snap(14, "branch8-target", &[]), true),
            (snap(15, "", &["/x/private/branch8-target", "target"]), true),
            (snap(16, "stage8-target", &[]), true),
            (snap(17, "", &["/x/private/stage8-target", "target"]), true),
            (
                snap(201, "", &[&format!("/private/candidate/{full}"), "target"]),
                true,
            ),
            (snap(202, full, &[]), true),
            (snap(203, short, &[]), true),
            (snap(205, short, &["/usr/bin/unrelated"]), false),
            (snap(206, &format!("{short}x"), &[]), false),
            (snap(207, &short[..short.len() - 1], &[]), false),
            (
                snap(208, "unrelated", &["bash", "-c", &format!("echo {full}")]),
                false,
            ),
            (snap(209, "unrelated", &[]), false),
            (
                snap(210, full, &[&format!("/private/candidate/{full}")]),
                false,
            ),
            (snap(211, "ollama", &["ollama", "runner"]), true),
            (snap(212, "ollama", &["ollama", "list"]), false),
            (snap(214, "ollama", &["ollama", "serve"]), false),
            (
                snap(213, "python3", &["python3", "-m", "mlx_lm.server"]),
                true,
            ),
        ];
        for (p, expected) in cases {
            assert_eq!(
                is_inference_worker_process(&p, 210),
                expected,
                "pid={}",
                p.pid
            );
        }
    }
}

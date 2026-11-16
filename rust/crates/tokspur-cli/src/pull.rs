//! Pinned weight downloader (`tokspur pull`) with resumable `.part` transfers, single-pass SHA-256 verification and automatic state configuration.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{IsTerminal, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::Duration;

use tokspur_json::Json;

use crate::error::CliError;
use crate::layout::{
    RuntimeLayout, at_or_below_directory, ensure_no_symlink_components,
    validate_clean_absolute_path,
};
use crate::serve::SignalGuard;
use crate::state::{
    StateLock, configure_runtime_locked, dflash2_profile_group_key, finalize_and_write_adoption,
    json_str, json_u64, preserved_dflash2_group, publish_verified_part, read_json_file,
    repo_leaf_name, verify_asset,
};

const DEFAULT_HF_BASE_URL: &str = "https://huggingface.co";
const QWEN_LICENCE_URL: &str = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/34567a4712bc9766c4449e2e98e4468bfa24d915/LICENSE";

/// Options for `tokspur pull`.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct PullOptions {
    pub dir: Option<PathBuf>,
    pub dflash2: bool,
    pub dflash2_dir: Option<PathBuf>,
    pub port: Option<u32>,
}

impl fmt::Debug for PullOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PullOptions")
            .field("dir", &self.dir.as_ref().map(|_| "[redacted]"))
            .field("dflash2", &self.dflash2)
            .field(
                "dflash2_dir",
                &self.dflash2_dir.as_ref().map(|_| "[redacted]"),
            )
            .field("port", &self.port)
            .finish()
    }
}

fn resolve_hf_base_url() -> Result<(String, bool), CliError> {
    let Some(raw) = std::env::var_os("TOKSPUR_HF_BASE_URL").filter(|v| !v.is_empty()) else {
        return Ok((DEFAULT_HF_BASE_URL.to_owned(), false));
    };
    let text = raw
        .into_string()
        .map_err(|_| CliError::Validation("TOKSPUR_HF_BASE_URL must be valid UTF-8".to_owned()))?;
    let trimmed = text.trim_end_matches('/');
    let port_part = trimmed
        .strip_prefix("http://127.0.0.1:")
        .or_else(|| trimmed.strip_prefix("http://[::1]:"))
        .ok_or_else(|| {
            CliError::Validation(
                "TOKSPUR_HF_BASE_URL only permits loopback http://127.0.0.1:<port> or http://[::1]:<port>"
                    .to_owned(),
            )
        })?;
    let port = port_part.parse::<u16>().map_err(|_| {
        CliError::Validation("TOKSPUR_HF_BASE_URL has an invalid loopback port".to_owned())
    })?;
    if port == 0 {
        return Err(CliError::Validation(
            "TOKSPUR_HF_BASE_URL loopback port must be non-zero".to_owned(),
        ));
    }
    Ok((trimmed.to_owned(), true))
}

fn validate_pinned_revision(revision: &str) -> Result<&str, CliError> {
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(CliError::Validation(format!(
            "asset profile requires a 40-character lowercase hex revision, found '{revision}'"
        )));
    }
    Ok(revision)
}

fn validate_asset_rel_path(name: &str) -> Result<&Path, CliError> {
    if name.is_empty() {
        return Err(CliError::Validation(
            "asset filename must not be empty".to_owned(),
        ));
    }
    let rel = Path::new(name);
    if rel.is_absolute() {
        return Err(CliError::Validation(
            "asset filename must be a relative path".to_owned(),
        ));
    }
    for comp in rel.components() {
        match comp {
            Component::Normal(part) => {
                let Some(s) = part.to_str() else {
                    return Err(CliError::Validation(
                        "asset filename must be UTF-8".to_owned(),
                    ));
                };
                if s.is_empty()
                    || s == "."
                    || s == ".."
                    || !s.bytes().all(|b| {
                        matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-')
                    })
                {
                    return Err(CliError::Validation(format!(
                        "invalid asset filename component '{s}'"
                    )));
                }
            }
            _ => {
                return Err(CliError::Validation(format!(
                    "asset filename must not traverse directories: {name}"
                )));
            }
        }
    }
    Ok(rel)
}

fn resolve_destination_dir(
    layout: &RuntimeLayout,
    explicit: Option<&Path>,
    repo: &str,
    label: &str,
) -> Result<PathBuf, CliError> {
    let _leaf = repo_leaf_name(repo)?;
    if let Some(raw) = explicit {
        let abs = std::path::absolute(raw).map_err(|source| CliError::Io {
            context: format!("cannot resolve {label}"),
            source,
        })?;
        let clean = validate_clean_absolute_path(&abs, label)?;
        ensure_no_symlink_components(&clean, label)?;
        if let RuntimeLayout::Staged(staged) = layout
            && at_or_below_directory(&clean, &staged.package_dir)?
        {
            return Err(CliError::Validation(format!(
                "{label} must be outside the immutable package directory"
            )));
        }
        Ok(clean)
    } else {
        let managed = layout.managed_models_dir()?;
        let leaf = repo_leaf_name(repo)?;
        let dest = managed.join(leaf);
        ensure_no_symlink_components(&dest, label)?;
        Ok(dest)
    }
}

fn curl_environment() -> Result<BTreeMap<String, String>, CliError> {
    const BLOCKED_PREFIXES: &[&str] = &["HF_", "HUGGING_FACE_", "HUGGINGFACE_"];
    let mut env = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let Ok(k) = key.into_string() else {
            continue;
        };
        if BLOCKED_PREFIXES.iter().any(|p| k.starts_with(p))
            || matches!(
                k.as_str(),
                "PYTHONPATH" | "PYTHONHOME" | "TOKSPUR_HF_BASE_URL"
            )
        {
            continue;
        }
        if let Ok(v) = value.into_string() {
            env.insert(k, v);
        }
    }
    env.insert("HF_HUB_DISABLE_IMPLICIT_TOKEN".to_owned(), "1".to_owned());
    env.insert("DO_NOT_TRACK".to_owned(), "1".to_owned());
    Ok(env)
}

/// Fails once a stop signal has arrived, so an interrupted pull adopts and configures nothing.
fn ensure_not_interrupted() -> Result<(), CliError> {
    if tokspur_sys::stop_requested() {
        return Err(CliError::Validation(
            "Download interrupted; re-run 'tokspur pull' to resume".to_owned(),
        ));
    }
    Ok(())
}

/// Runs curl to completion. A stop signal sent only to the CLI (SIGTERM, or SIGINT from another process) does not reach curl, so it is passed on once and the wait continues until curl has gone.
fn run_curl(cmd: &mut Command) -> Result<ExitStatus, CliError> {
    let mut child = cmd.spawn().map_err(|source| CliError::Io {
        context: "cannot execute /usr/bin/curl".to_owned(),
        source,
    })?;
    let mut forwarded = false;
    loop {
        if let Some(status) = child.try_wait().map_err(|source| CliError::Io {
            context: "cannot wait for /usr/bin/curl".to_owned(),
            source,
        })? {
            return Ok(status);
        }
        if tokspur_sys::stop_requested() && !forwarded {
            forwarded = true;
            if let Err(source) = tokspur_sys::process::kill(child.id() as i32, tokspur_sys::SIGTERM)
                && source.raw_os_error() != Some(tokspur_sys::ESRCH)
            {
                return Err(CliError::Io {
                    context: "cannot interrupt /usr/bin/curl".to_owned(),
                    source,
                });
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// curl's exit status when a transfer would exceed `--max-filesize`.
const CURL_FILESIZE_EXCEEDED: i32 = 63;

/// The least `--max-filesize` given to curl. Before 8.11.0 curl also counts the cap against the body of each redirect it follows, and the model host answers `/resolve/` with a redirect whose body is a kilobyte or two, so a smaller cap fails the transfer on the redirect (and macOS ships 8.7.1).
const MIN_TRANSFER_CAP: u64 = 64 << 10;

/// The `--max-filesize` for a transfer into a partial file of `part_len` bytes: what the file still lacks, but never less than a redirect body may need. A file is held to its pin by the size check after the transfer.
fn transfer_cap(expected_size: u64, part_len: u64) -> u64 {
    expected_size.saturating_sub(part_len).max(MIN_TRANSFER_CAP)
}

/// The curl arguments for one asset transfer into `part_path`, of at most `max_transfer` bytes.
fn curl_arguments(
    url: &str,
    part_path: &Path,
    allow_http: bool,
    resume: bool,
    max_transfer: u64,
    progress_bar: bool,
) -> Vec<OsString> {
    // `-q` only works as the first argument. It stops curl reading the caller's ~/.curlrc, whose settings (netrc credentials, extra headers, a proxy, `insecure`) would otherwise apply to a download promised to send no tokens.
    let mut args: Vec<OsString> = [
        "-q",
        "--fail",
        "--location",
        "--proto",
        if allow_http { "=http,https" } else { "=https" },
        "--retry",
        "3",
        "--retry-delay",
        "2",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    if progress_bar {
        args.push("--progress-bar".into());
    } else {
        args.extend(["--silent".into(), "--show-error".into()]);
    }
    if resume {
        args.extend(["--continue-at".into(), "-".into()]);
    }
    // curl counts the cap against the body of each response, so on a resume it is what the partial file still lacks (see `transfer_cap`). A larger declared length is refused before anything is written, and a body with no declared length is cut off at the cap (curl 8.4.0 and later; an older curl only checks a declared length, and the size check after the transfer catches the rest).
    args.extend(["--max-filesize".into(), max_transfer.to_string().into()]);
    args.extend(["--output".into(), part_path.into(), "--".into(), url.into()]);
    args
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    if bytes >= (1u64 << 30) {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= (1u64 << 20) {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} B")
    }
}

#[allow(clippy::too_many_arguments)]
fn pull_single_asset(
    base_url: &str,
    allow_http: bool,
    env: &BTreeMap<String, String>,
    repo: &str,
    revision: &str,
    dest_dir: &Path,
    spec: &Json,
    cache: &mut BTreeMap<PathBuf, Json>,
    out: &mut dyn Write,
) -> Result<Json, CliError> {
    ensure_not_interrupted()?;
    let name = json_str(spec, "name")?;
    let expected_size = json_u64(spec, "size")?;
    let rel = validate_asset_rel_path(name)?;
    let dest_path = dest_dir.join(rel);
    if let Some(already) = cache.get(&dest_path) {
        return Ok(already.clone());
    }

    let parent = dest_path
        .parent()
        .ok_or_else(|| CliError::Validation("destination asset has no parent".to_owned()))?;
    ensure_no_symlink_components(parent, "download directory")?;
    fs::create_dir_all(parent).map_err(|source| CliError::Io {
        context: "cannot create model download directory".to_owned(),
        source,
    })?;
    ensure_no_symlink_components(parent, "download directory")?;

    let file_name_str = dest_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| CliError::Validation("asset filename must be UTF-8".to_owned()))?;
    let part_path = parent.join(format!("{file_name_str}.part"));

    match fs::symlink_metadata(&dest_path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(CliError::Validation(format!(
                "Refusing symlink asset path: {name}"
            )));
        }
        Ok(_) => {
            let record = verify_asset(&dest_path, spec)?;
            if let Ok(part_meta) = fs::symlink_metadata(&part_path)
                && part_meta.file_type().is_file()
            {
                let _ignored = fs::remove_file(&part_path);
            }
            writeln!(out, "Already verified: {name}").map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
            out.flush().map_err(|source| CliError::Io {
                context: "stdout flush failed".to_owned(),
                source,
            })?;
            cache.insert(dest_path, record.clone());
            return Ok(record);
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(CliError::Io {
                context: format!("cannot inspect {name}"),
                source,
            });
        }
    }

    let part_len = match fs::symlink_metadata(&part_path) {
        Ok(meta) => {
            if !meta.file_type().is_file() {
                return Err(CliError::Validation(format!(
                    "Partial download path must be a regular file: {name}.part"
                )));
            }
            if meta.nlink() != 1 {
                return Err(CliError::Validation(format!(
                    "Partial download file must not have multiple hard links: {name}.part"
                )));
            }
            if meta.len() > expected_size {
                fs::remove_file(&part_path).map_err(|source| CliError::Io {
                    context: "cannot remove oversized partial download".to_owned(),
                    source,
                })?;
                0
            } else {
                meta.len()
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0,
        Err(source) => {
            return Err(CliError::Io {
                context: format!("cannot inspect partial download {name}.part"),
                source,
            });
        }
    };

    if part_len < expected_size {
        let url = format!("{base_url}/{repo}/resolve/{revision}/{name}");
        if part_len > 0 {
            writeln!(
                out,
                "Resuming {repo}/{name} ({} / {})...",
                format_bytes(part_len),
                format_bytes(expected_size)
            )
            .map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
        } else {
            writeln!(
                out,
                "Downloading {repo}/{name} ({})...",
                format_bytes(expected_size)
            )
            .map_err(|source| CliError::Io {
                context: "stdout write failed".to_owned(),
                source,
            })?;
        }
        out.flush().map_err(|source| CliError::Io {
            context: "stdout flush failed".to_owned(),
            source,
        })?;

        let mut cmd = Command::new("/usr/bin/curl");
        cmd.env_clear().envs(env).args(curl_arguments(
            &url,
            &part_path,
            allow_http,
            part_len > 0,
            transfer_cap(expected_size, part_len),
            std::io::stderr().is_terminal(),
        ));

        let status = run_curl(&mut cmd)?;
        ensure_not_interrupted()?;
        // The cap can exceed what a small file or a nearly complete one still lacks. Nothing of a partial file past its pin can be resumed, so it goes now rather than on the next run.
        let oversized = fs::symlink_metadata(&part_path)
            .is_ok_and(|meta| meta.file_type().is_file() && meta.len() > expected_size);
        if oversized {
            let _ignored = fs::remove_file(&part_path);
        }
        if oversized || status.code() == Some(CURL_FILESIZE_EXCEEDED) {
            return Err(CliError::Validation(format!(
                "Refusing {repo}/{name}: the server offered more than the pinned {expected_size} bytes"
            )));
        }
        if !status.success() {
            return Err(CliError::Validation(format!(
                "Failed to download {repo}/{name}; re-run 'tokspur pull' to resume"
            )));
        }
    }

    writeln!(out, "Verifying: {name}").map_err(|source| CliError::Io {
        context: "stdout write failed".to_owned(),
        source,
    })?;
    out.flush().map_err(|source| CliError::Io {
        context: "stdout flush failed".to_owned(),
        source,
    })?;

    let record = publish_verified_part(&part_path, &dest_path, spec)?;
    cache.insert(dest_path, record.clone());
    Ok(record)
}

#[allow(clippy::too_many_arguments)]
fn pull_group(
    specs: &Json,
    group_key: &str,
    dest_dir: &Path,
    base_url: &str,
    allow_http: bool,
    env: &BTreeMap<String, String>,
    cache: &mut BTreeMap<PathBuf, Json>,
    out: &mut dyn Write,
) -> Result<Json, CliError> {
    let group_obj = specs.find(group_key).ok().flatten().ok_or_else(|| {
        CliError::Validation(format!("flash-next-assets.json missing {group_key}"))
    })?;
    let repo = json_str(group_obj, "repo")?;
    let _leaf = repo_leaf_name(repo)?;
    let revision = validate_pinned_revision(json_str(group_obj, "revision")?)?;
    let files = group_obj
        .find("files")
        .ok()
        .flatten()
        .and_then(|f| f.array().ok())
        .ok_or_else(|| {
            CliError::Validation(format!("flash-next-assets.json {group_key} missing files"))
        })?;

    ensure_no_symlink_components(dest_dir, "model destination directory")?;
    fs::create_dir_all(dest_dir).map_err(|source| CliError::Io {
        context: "cannot create model destination directory".to_owned(),
        source,
    })?;

    let mut verified = Vec::with_capacity(files.len());
    for spec in files {
        verified.push(pull_single_asset(
            base_url, allow_http, env, repo, revision, dest_dir, spec, cache, out,
        )?);
    }

    let resolved_dir = fs::canonicalize(dest_dir).map_err(|source| CliError::Io {
        context: "cannot resolve model destination directory".to_owned(),
        source,
    })?;
    let resolved_str = resolved_dir.to_str().ok_or_else(|| {
        CliError::Validation("model directory path is not valid UTF-8".to_owned())
    })?;

    Ok(Json::new_object([
        ("directory", Json::from(resolved_str)),
        ("assets", Json::Array(verified)),
    ]))
}

fn resolve_pull_port(
    layout: &RuntimeLayout,
    tokspur_base: &Path,
    explicit_port: Option<u32>,
) -> Result<u32, CliError> {
    if let Some(p) = explicit_port {
        if !(1024..=65535).contains(&p) {
            return Err(CliError::Validation(
                "Use an unprivileged local port".to_owned(),
            ));
        }
        return Ok(p);
    }
    let configured_path = tokspur_base.join("configured.json");
    if configured_path.is_file()
        && let Ok(existing) = read_json_file(&configured_path)
        && let Some(p) = existing
            .find("port")
            .ok()
            .flatten()
            .and_then(|v| v.unsigned_integer().ok())
            .and_then(|n| u32::try_from(n).ok())
        && (1024..=65535).contains(&p)
    {
        return Ok(p);
    }
    let profile = read_json_file(&layout.hardware_profile_path())?;
    Ok(profile
        .find("port")
        .ok()
        .flatten()
        .and_then(|v| v.unsigned_integer().ok())
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(18081))
}

/// Downloads and verifies pinned weights into the managed models directory (or `--dir`), writes `adoption.json`, and configures local state.
pub fn pull_weights(
    layout: &RuntimeLayout,
    opts: &PullOptions,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    if let Some(p) = opts.port
        && !(1024..=65535).contains(&p)
    {
        return Err(CliError::Validation(
            "Use an unprivileged local port".to_owned(),
        ));
    }
    let (base_url, allow_http) = resolve_hf_base_url()?;
    let env = curl_environment()?;
    let tokspur_base = layout.tokspur_base()?;
    let _lock = StateLock::acquire(&tokspur_base)?;
    let _signal_guard = SignalGuard::install()?;

    let specs = read_json_file(&layout.assets_profile_path())?;
    let target_group =
        specs.find("target").ok().flatten().ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing target".to_owned())
        })?;
    let draft_group =
        specs.find("draft").ok().flatten().ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing draft".to_owned())
        })?;
    let target_repo = json_str(target_group, "repo")?;
    let draft_repo = json_str(draft_group, "repo")?;

    let target_dir =
        resolve_destination_dir(layout, opts.dir.as_deref(), target_repo, "model directory")?;
    let draft_dir = if opts.dir.is_some() || target_repo == draft_repo {
        target_dir.clone()
    } else {
        resolve_destination_dir(layout, None, draft_repo, "draft directory")?
    };

    let include_dflash2 = opts.dflash2 || opts.dflash2_dir.is_some();
    let dflash2_plan = if include_dflash2 {
        let df_key = dflash2_profile_group_key(&specs).ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing dflash2_draft".to_owned())
        })?;
        let df_obj = specs.find(df_key).ok().flatten().ok_or_else(|| {
            CliError::Validation("flash-next-assets.json missing dflash2_draft".to_owned())
        })?;
        let df_repo = json_str(df_obj, "repo")?;
        let df_dir = resolve_destination_dir(
            layout,
            opts.dflash2_dir.as_deref(),
            df_repo,
            "DFlash2 directory",
        )?;
        Some((df_key, df_dir))
    } else {
        None
    };

    writeln!(
        out,
        "Model licence: Qwen Community License Agreement 1.0 ({QWEN_LICENCE_URL})"
    )
    .map_err(|source| CliError::Io {
        context: "stdout write failed".to_owned(),
        source,
    })?;
    if include_dflash2 {
        writeln!(
            out,
            "Drafter licence: Apache-2.0 checkpoint subject to Qwen Community License Agreement 1.0"
        )
        .map_err(|source| CliError::Io {
            context: "stdout write failed".to_owned(),
            source,
        })?;
    }
    out.flush().map_err(|source| CliError::Io {
        context: "stdout flush failed".to_owned(),
        source,
    })?;

    let mut verified_cache = BTreeMap::new();
    let mut fields: Vec<(&str, Json)> = vec![
        ("schema", Json::integer(1)),
        ("verification", Json::from("full-sha256")),
    ];
    fields.push((
        "target",
        pull_group(
            &specs,
            "target",
            &target_dir,
            &base_url,
            allow_http,
            &env,
            &mut verified_cache,
            out,
        )?,
    ));
    fields.push((
        "draft",
        pull_group(
            &specs,
            "draft",
            &draft_dir,
            &base_url,
            allow_http,
            &env,
            &mut verified_cache,
            out,
        )?,
    ));
    if let Some((df_key, df_dir)) = dflash2_plan {
        fields.push((
            "dflash2_draft",
            pull_group(
                &specs,
                df_key,
                &df_dir,
                &base_url,
                allow_http,
                &env,
                &mut verified_cache,
                out,
            )?,
        ));
    } else if let Some(existing_df) = preserved_dflash2_group(&tokspur_base, &specs) {
        fields.push(("dflash2_draft", existing_df));
    }

    // The last point at which a stop signal cancels the pull: once the adoption is written, the short configuration step runs to completion.
    ensure_not_interrupted()?;
    finalize_and_write_adoption(&tokspur_base, &specs, Json::new_object(fields), out)?;
    let port = resolve_pull_port(layout, &tokspur_base, opts.port)?;
    configure_runtime_locked(layout, &tokspur_base, port, opts.port.is_some(), out)?;

    if include_dflash2 {
        writeln!(
            out,
            "Ready. Start serving with: tokspur serve --speed-preset --drafter dflash2"
        )
    } else {
        writeln!(
            out,
            "Ready. Start serving with: tokspur serve --speed-preset"
        )
    }
    .map_err(|source| CliError::Io {
        context: "stdout write failed".to_owned(),
        source,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curl_arguments_skip_the_curlrc_and_cap_the_transfer() {
        for (allow_http, resume, progress_bar) in [(false, false, false), (true, true, true)] {
            let args = curl_arguments(
                "https://huggingface.co/owner/repo/resolve/rev/file.gguf",
                Path::new("/models/file.gguf.part"),
                allow_http,
                resume,
                4096,
                progress_bar,
            );
            assert!(
                args.windows(2)
                    .any(|w| w[0] == "--max-filesize" && w[1] == "4096")
            );
            // curl honours `-q` only in first position.
            assert_eq!(args[0], "-q");
            assert_eq!(args.iter().filter(|arg| *arg == "-q").count(), 1);
            let protocols = if allow_http { "=http,https" } else { "=https" };
            assert!(
                args.windows(2)
                    .any(|w| w[0] == "--proto" && w[1] == protocols)
            );
            assert_eq!(
                args.windows(2)
                    .any(|w| w[0] == "--continue-at" && w[1] == "-"),
                resume
            );
            assert_eq!(
                args[args.len() - 4..],
                [
                    "--output",
                    "/models/file.gguf.part",
                    "--",
                    "https://huggingface.co/owner/repo/resolve/rev/file.gguf"
                ]
            );
        }
    }

    #[test]
    fn transfer_cap_is_what_is_left_but_never_below_a_redirect_body() {
        const GIB: u64 = 1 << 30;
        assert_eq!(transfer_cap(40 * GIB, 0), 40 * GIB);
        assert_eq!(transfer_cap(40 * GIB, GIB), 39 * GIB);
        assert_eq!(
            transfer_cap(40 * GIB, 40 * GIB - MIN_TRANSFER_CAP - 1),
            MIN_TRANSFER_CAP + 1
        );
        // A small file, and a large one with little left, keep room for the redirect.
        assert_eq!(transfer_cap(1015, 0), MIN_TRANSFER_CAP);
        assert_eq!(transfer_cap(40 * GIB, 40 * GIB - 100), MIN_TRANSFER_CAP);
        assert_eq!(transfer_cap(1015, 1015), MIN_TRANSFER_CAP);
    }
}

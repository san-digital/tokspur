use std::ffi::CString;
use std::fs::{self, File};
use std::os::fd::AsFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use tokspur_json::{Json, from_chars_f64, parse_json, to_chars_shortest};
use tokspur_sys::fd::{
    O_CLOEXEC, O_CREAT, O_DIRECTORY, O_EXCL, O_NOFOLLOW, O_RDONLY, O_WRONLY, fchmod, fsync,
    lstat_at, openat, pwrite_all, unlink_at,
};
use tokspur_sys::random_bytes;
use tokspur_text::sha256::hex;

use crate::error::EvalError;

/// Formats `Json` with 2-space indentation and sorted object keys + trailing newline.
pub fn format_pretty_sorted_json(value: &Json) -> Result<Vec<u8>, EvalError> {
    let mut out = Vec::new();
    write_pretty_json(value, 0, &mut out)?;
    out.push(b'\n');
    Ok(out)
}

fn write_pretty_json(value: &Json, indent: usize, out: &mut Vec<u8>) -> Result<(), EvalError> {
    match value {
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => {
            let encoded = value.dump().map_err(|source| EvalError::Json {
                context: "failed to serialize JSON scalar",
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
                        .map_err(|source| EvalError::Json {
                            context: "failed to serialize JSON key",
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

pub fn json_f64(value: f64) -> Result<Json, EvalError> {
    if !value.is_finite() {
        return Err(EvalError::NumericsFailed(
            "cannot encode non-finite f64 into JSON".to_owned(),
        ));
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    let text = to_chars_shortest(normalized)
        .ok_or_else(|| EvalError::NumericsFailed("cannot format non-finite f64".to_owned()))?;
    parse_json(text.as_bytes()).map_err(|source| EvalError::Json {
        context: "failed to encode shortest f64 JSON number",
        source,
    })
}

pub fn json_usize(value: usize) -> Json {
    Json::integer(value as u64)
}

pub fn require_object_keys(fields: &[(Vec<u8>, Json)], allowed: &[&[u8]]) -> Result<(), EvalError> {
    for (k, _) in fields {
        if !allowed.contains(&k.as_slice()) {
            let key_str = String::from_utf8_lossy(k);
            return Err(EvalError::Validation(format!(
                "unknown JSON field '{key_str}'"
            )));
        }
    }
    Ok(())
}

pub fn required_field<'a>(
    fields: &'a [(Vec<u8>, Json)],
    key: &[u8],
) -> Result<&'a Json, EvalError> {
    fields
        .iter()
        .find(|(k, _)| k.as_slice() == key)
        .map(|(_, v)| v)
        .ok_or_else(|| {
            EvalError::Validation(format!(
                "missing required JSON field '{}'",
                String::from_utf8_lossy(key)
            ))
        })
}

pub fn as_str(val: &Json) -> Result<&str, EvalError> {
    let bytes = val.string().map_err(|source| EvalError::Json {
        context: "expected JSON string",
        source,
    })?;
    std::str::from_utf8(bytes)
        .map_err(|_| EvalError::Validation("JSON string is not valid UTF-8".to_owned()))
}

pub fn as_bool(val: &Json) -> Result<bool, EvalError> {
    val.boolean().map_err(|source| EvalError::Json {
        context: "expected JSON boolean",
        source,
    })
}

pub fn as_usize(val: &Json) -> Result<usize, EvalError> {
    let u = val.unsigned_integer().map_err(|source| EvalError::Json {
        context: "expected unsigned integer",
        source,
    })?;
    usize::try_from(u)
        .map_err(|_| EvalError::Validation("integer exceeds platform usize".to_owned()))
}

pub fn as_f64(val: &Json) -> Result<f64, EvalError> {
    let lexeme = val.number_lexeme().map_err(|source| EvalError::Json {
        context: "expected JSON number",
        source,
    })?;
    let parsed = from_chars_f64(lexeme)
        .ok_or_else(|| EvalError::Validation(format!("invalid f64 JSON number '{lexeme}'")))?;
    if !parsed.is_finite() {
        return Err(EvalError::NumericsFailed(format!(
            "non-finite f64 JSON number '{lexeme}'"
        )));
    }
    Ok(parsed)
}

/// Atomically writes `payload` with `0o600` permissions via `O_CREAT|O_EXCL|O_NOFOLLOW`, `fsync`, and `rename`.
pub fn atomic_write_private_bytes(path: &Path, payload: &[u8]) -> Result<(), EvalError> {
    let raw_parent = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = if raw_parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        raw_parent
    };
    let file_name = path
        .file_name()
        .ok_or_else(|| EvalError::Validation("target path has no filename".to_owned()))?;

    if !parent.exists() {
        fs::create_dir_all(parent).map_err(|source| EvalError::Io {
            context: "cannot create parent directory".to_owned(),
            source,
        })?;
    }

    let c_parent = CString::new(parent.as_os_str().as_bytes())
        .map_err(|_| EvalError::Validation("parent path contains NUL byte".to_owned()))?;
    let base_dir = File::open(parent).map_err(|source| EvalError::Io {
        context: "cannot open parent directory".to_owned(),
        source,
    })?;
    let dir_fd = openat(
        base_dir.as_fd(),
        &CString::new(".").map_err(|_| EvalError::Validation("bad dot path".to_owned()))?,
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
        0,
    )
    .map_err(|source| EvalError::Io {
        context: format!(
            "cannot open nofollow directory '{}'",
            String::from_utf8_lossy(c_parent.as_bytes())
        ),
        source,
    })?;

    let c_name = CString::new(file_name.as_bytes())
        .map_err(|_| EvalError::Validation("filename contains NUL byte".to_owned()))?;
    if let Ok(existing) = lstat_at(dir_fd.as_fd(), &c_name)
        && existing.is_symlink()
    {
        return Err(EvalError::Validation(
            "refusing to overwrite a symlink".to_owned(),
        ));
    }

    let mut rnd = [0u8; 12];
    random_bytes(&mut rnd);
    let tmp_name_str = format!(".tmp-{}.json", hex(&rnd));
    let tmp_c = CString::new(tmp_name_str.as_bytes())
        .map_err(|_| EvalError::Validation("temp filename contains NUL byte".to_owned()))?;

    let tmp_file = openat(
        dir_fd.as_fd(),
        &tmp_c,
        O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY | O_CLOEXEC,
        0o600,
    )
    .map_err(|source| EvalError::Io {
        context: "cannot create temporary private file".to_owned(),
        source,
    })?;

    let write_res = (|| -> Result<(), EvalError> {
        pwrite_all(tmp_file.as_fd(), payload, 0).map_err(|source| EvalError::Io {
            context: "cannot write temporary private file".to_owned(),
            source,
        })?;
        fchmod(tmp_file.as_fd(), 0o600).map_err(|source| EvalError::Io {
            context: "cannot set 0o600 on temporary private file".to_owned(),
            source,
        })?;
        fsync(tmp_file.as_fd()).map_err(|source| EvalError::Io {
            context: "cannot fsync temporary private file".to_owned(),
            source,
        })?;
        if let Ok(existing) = lstat_at(dir_fd.as_fd(), &c_name)
            && existing.is_symlink()
        {
            return Err(EvalError::Validation(
                "refusing to overwrite a symlink".to_owned(),
            ));
        }
        fs::rename(parent.join(&tmp_name_str), path).map_err(|source| EvalError::Io {
            context: "cannot atomically rename private file".to_owned(),
            source,
        })?;
        fsync(dir_fd.as_fd()).map_err(|source| EvalError::Io {
            context: "cannot fsync parent directory".to_owned(),
            source,
        })?;
        Ok(())
    })();

    drop(tmp_file);
    if write_res.is_err() {
        let _ignored = unlink_at(dir_fd.as_fd(), &tmp_c, false);
    }
    write_res
}

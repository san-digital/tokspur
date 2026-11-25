//! Content-addressed SHA-256 receipt cache verification and negative controls (Track 2 pattern).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path};

use tokspur_json::{Json, JsonLimits, parse_json_limited};
use tokspur_text::sha256::{Sha256, hex, sha256};

use crate::error::EvalError;
use crate::json_util::{
    as_str, atomic_write_private_bytes, format_pretty_sorted_json, require_object_keys,
    required_field,
};

pub const DOMAIN_TAG: &[u8] = b"tokspur-rust-receipt-v2\0";

pub fn push_len_prefixed(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedArtifactMeta {
    pub rel_path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptMeta {
    pub schema_version: String,
    pub action: String,
    pub arm: String,
    pub cpp_oracle: String,
    pub features: String,
    pub content_key: String,
    pub toolchain_sha256: String,
    pub cargo_lock_sha256: String,
    pub rust_mk_sha256: String,
    pub sources_sha256: String,
    pub status: String,
    pub artifacts: Vec<CachedArtifactMeta>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptKeySpec {
    pub action: String,
    pub arm: String,
    pub cpp_oracle: String,
    pub features: String,
    pub toolchain_items: BTreeMap<String, Vec<u8>>,
    pub flag_items: BTreeMap<String, Vec<u8>>,
    pub cargo_lock_bytes: Vec<u8>,
    pub rust_mk_bytes: Vec<u8>,
    pub source_digests: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptKeyDigest {
    pub content_key: String,
    pub toolchain_sha256: String,
    pub cargo_lock_sha256: String,
    pub rust_mk_sha256: String,
    pub sources_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&sha256(bytes))
}

pub fn compute_receipt_key(spec: &ReceiptKeySpec) -> ReceiptKeyDigest {
    let mut tc_buf = Vec::new();
    tc_buf.extend_from_slice(b"tokspur-toolchain-v1\0");
    for (k, v) in &spec.toolchain_items {
        push_len_prefixed(&mut tc_buf, k.as_bytes());
        push_len_prefixed(&mut tc_buf, v);
    }
    let toolchain_sha256 = sha256_hex(&tc_buf);
    let cargo_lock_sha256 = sha256_hex(&spec.cargo_lock_bytes);
    let rust_mk_sha256 = sha256_hex(&spec.rust_mk_bytes);

    let mut src_ctx = Sha256::new();
    src_ctx.update(b"tokspur-sources-v1\0");
    for (rel_path, digest) in &spec.source_digests {
        let mut entry = Vec::with_capacity(16 + rel_path.len() + digest.len());
        push_len_prefixed(&mut entry, rel_path.as_bytes());
        push_len_prefixed(&mut entry, digest.as_bytes());
        src_ctx.update(&entry);
    }
    let sources_sha256 = hex(&src_ctx.finalize());

    let mut key_buf = Vec::with_capacity(512);
    key_buf.extend_from_slice(DOMAIN_TAG);
    push_len_prefixed(&mut key_buf, spec.action.as_bytes());
    push_len_prefixed(&mut key_buf, spec.arm.as_bytes());
    push_len_prefixed(&mut key_buf, spec.cpp_oracle.as_bytes());
    push_len_prefixed(&mut key_buf, spec.features.as_bytes());
    push_len_prefixed(&mut key_buf, toolchain_sha256.as_bytes());
    push_len_prefixed(&mut key_buf, cargo_lock_sha256.as_bytes());
    push_len_prefixed(&mut key_buf, rust_mk_sha256.as_bytes());
    push_len_prefixed(&mut key_buf, &(spec.flag_items.len() as u64).to_le_bytes());
    for (k, v) in &spec.flag_items {
        push_len_prefixed(&mut key_buf, k.as_bytes());
        push_len_prefixed(&mut key_buf, v);
    }
    push_len_prefixed(&mut key_buf, sources_sha256.as_bytes());

    ReceiptKeyDigest {
        content_key: sha256_hex(&key_buf),
        toolchain_sha256,
        cargo_lock_sha256,
        rust_mk_sha256,
        sources_sha256,
    }
}

impl ReceiptMeta {
    pub fn to_pretty_json_bytes(&self) -> Result<Vec<u8>, EvalError> {
        let artifacts_json: Vec<Json> = self
            .artifacts
            .iter()
            .map(|a| {
                Json::new_object([
                    ("rel_path", Json::from(a.rel_path.as_str())),
                    ("sha256", Json::from(a.sha256.as_str())),
                    ("bytes", Json::integer(a.bytes)),
                ])
            })
            .collect();
        let root = Json::new_object([
            ("schema_version", Json::from(self.schema_version.as_str())),
            ("action", Json::from(self.action.as_str())),
            ("arm", Json::from(self.arm.as_str())),
            ("cpp_oracle", Json::from(self.cpp_oracle.as_str())),
            ("features", Json::from(self.features.as_str())),
            ("content_key", Json::from(self.content_key.as_str())),
            (
                "toolchain_sha256",
                Json::from(self.toolchain_sha256.as_str()),
            ),
            (
                "cargo_lock_sha256",
                Json::from(self.cargo_lock_sha256.as_str()),
            ),
            ("rust_mk_sha256", Json::from(self.rust_mk_sha256.as_str())),
            ("sources_sha256", Json::from(self.sources_sha256.as_str())),
            ("status", Json::from(self.status.as_str())),
            ("artifacts", Json::Array(artifacts_json)),
        ]);
        format_pretty_sorted_json(&root)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let root =
            parse_json_limited(bytes, JsonLimits::default()).map_err(|source| EvalError::Json {
                context: "failed parsing receipt JSON",
                source,
            })?;
        let fields = root.object().map_err(|source| EvalError::Json {
            context: "receipt must be JSON object",
            source,
        })?;
        require_object_keys(
            fields,
            &[
                b"schema_version",
                b"action",
                b"arm",
                b"cpp_oracle",
                b"features",
                b"content_key",
                b"toolchain_sha256",
                b"cargo_lock_sha256",
                b"rust_mk_sha256",
                b"sources_sha256",
                b"status",
                b"artifacts",
            ],
        )?;
        let art_arr = required_field(fields, b"artifacts")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "receipt.artifacts must be array",
                source,
            })?;
        let mut artifacts = Vec::with_capacity(art_arr.len());
        for a in art_arr {
            let af = a.object().map_err(|source| EvalError::Json {
                context: "artifact entry must be object",
                source,
            })?;
            require_object_keys(af, &[b"rel_path", b"sha256", b"bytes"])?;
            let bytes_u64 = required_field(af, b"bytes")?
                .unsigned_integer()
                .map_err(|source| EvalError::Json {
                    context: "artifact.bytes must be unsigned integer",
                    source,
                })?;
            artifacts.push(CachedArtifactMeta {
                rel_path: as_str(required_field(af, b"rel_path")?)?.to_owned(),
                sha256: as_str(required_field(af, b"sha256")?)?.to_owned(),
                bytes: bytes_u64,
            });
        }
        Ok(Self {
            schema_version: as_str(required_field(fields, b"schema_version")?)?.to_owned(),
            action: as_str(required_field(fields, b"action")?)?.to_owned(),
            arm: as_str(required_field(fields, b"arm")?)?.to_owned(),
            cpp_oracle: as_str(required_field(fields, b"cpp_oracle")?)?.to_owned(),
            features: as_str(required_field(fields, b"features")?)?.to_owned(),
            content_key: as_str(required_field(fields, b"content_key")?)?.to_owned(),
            toolchain_sha256: as_str(required_field(fields, b"toolchain_sha256")?)?.to_owned(),
            cargo_lock_sha256: as_str(required_field(fields, b"cargo_lock_sha256")?)?.to_owned(),
            rust_mk_sha256: as_str(required_field(fields, b"rust_mk_sha256")?)?.to_owned(),
            sources_sha256: as_str(required_field(fields, b"sources_sha256")?)?.to_owned(),
            status: as_str(required_field(fields, b"status")?)?.to_owned(),
            artifacts,
        })
    }
}

pub fn write_receipt_atomically(receipt_path: &Path, meta: &ReceiptMeta) -> Result<(), EvalError> {
    let encoded = meta.to_pretty_json_bytes()?;
    atomic_write_private_bytes(receipt_path, &encoded)
}

fn is_safe_artifact_rel_path(rel: &str, require_bin_prefix: bool) -> bool {
    if rel.is_empty() || rel.contains('\\') || rel.contains('\0') {
        return false;
    }
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return false;
    }
    let mut normals = Vec::new();
    for comp in rel_path.components() {
        match comp {
            Component::Normal(seg) if !seg.is_empty() => normals.push(seg),
            _ => return false,
        }
    }
    if normals.is_empty() {
        return false;
    }
    if require_bin_prefix {
        return normals.len() == 2 && normals[0] == "bin";
    }
    true
}

/// Fail-safe verification: returns `Ok(None)` on missing, malformed, mismatched, or corrupted receipt/artifact.
pub fn verify_receipt_file(
    receipt_path: &Path,
    expected_action: &str,
    expected_arm: &str,
    expected_content_key: &str,
    artifact_root: Option<&Path>,
    expected_outputs: &[&str],
) -> Result<Option<ReceiptMeta>, EvalError> {
    let Ok(receipt_md) = fs::symlink_metadata(receipt_path) else {
        return Ok(None);
    };
    if receipt_md.file_type().is_symlink() || !receipt_md.file_type().is_file() {
        return Ok(None);
    }
    let Ok(raw) = fs::read(receipt_path) else {
        return Ok(None);
    };
    let Ok(meta) = ReceiptMeta::from_json_bytes(&raw) else {
        return Ok(None);
    };
    if meta.schema_version != "1.0.0"
        || meta.status != "passed"
        || meta.action != expected_action
        || meta.arm != expected_arm
        || meta.content_key != expected_content_key
    {
        return Ok(None);
    }
    if meta.artifacts.len() != expected_outputs.len() {
        return Ok(None);
    }
    if expected_action == "fuzz-build" && (expected_outputs.is_empty() || meta.artifacts.is_empty())
    {
        return Ok(None);
    }
    if !meta.artifacts.is_empty() {
        let Some(root) = artifact_root else {
            return Ok(None);
        };
        let Ok(root_md) = fs::symlink_metadata(root) else {
            return Ok(None);
        };
        if root_md.file_type().is_symlink() || !root_md.file_type().is_dir() {
            return Ok(None);
        }
        let require_bin = expected_action == "fuzz-build";
        for (art, &exp_out) in meta.artifacts.iter().zip(expected_outputs.iter()) {
            if exp_out.is_empty()
                || exp_out.contains('\\')
                || exp_out.contains('\0')
                || !is_safe_artifact_rel_path(&art.rel_path, require_bin)
            {
                return Ok(None);
            }
            let expected_match = if require_bin {
                art.rel_path == format!("bin/{exp_out}")
                    || (exp_out.starts_with("bin/") && art.rel_path == exp_out)
            } else {
                art.rel_path == exp_out || art.rel_path == format!("bin/{exp_out}")
            };
            if !expected_match {
                return Ok(None);
            }
            let mut cur = root.to_path_buf();
            for comp in Path::new(&art.rel_path).components() {
                let Component::Normal(seg) = comp else {
                    return Ok(None);
                };
                cur.push(seg);
                let Ok(md) = fs::symlink_metadata(&cur) else {
                    return Ok(None);
                };
                if md.file_type().is_symlink() {
                    return Ok(None);
                }
            }
            let Ok(leaf_md) = fs::symlink_metadata(&cur) else {
                return Ok(None);
            };
            if !leaf_md.file_type().is_file() {
                return Ok(None);
            }
            let Ok(bytes) = fs::read(&cur) else {
                return Ok(None);
            };
            if bytes.len() as u64 != art.bytes || sha256_hex(&bytes) != art.sha256 {
                return Ok(None);
            }
        }
    }
    Ok(Some(meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokspur_sys::random_bytes;

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let mut rnd = [0u8; 8];
            random_bytes(&mut rnd);
            let dir = std::env::temp_dir().join(format!("tokspur_rc_{prefix}_{}", hex(&rnd)));
            fs::create_dir_all(&dir).unwrap();
            Self { path: dir }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ignored = fs::remove_dir_all(&self.path);
        }
    }

    fn base_spec(arm: &str, cpp_oracle: &str) -> ReceiptKeySpec {
        let mut toolchain_items = BTreeMap::new();
        toolchain_items.insert("rustc".to_owned(), b"rustc 1.99.1".to_vec());
        let mut flag_items = BTreeMap::new();
        flag_items.insert("RUSTFLAGS".to_owned(), Vec::new());
        let mut source_digests = BTreeMap::new();
        source_digests.insert(
            "rust/crates/tokspur-wire/src/worker.rs".to_owned(),
            sha256_hex(b"wire v1"),
        );
        if arm == "ordinary" {
            source_digests.insert(
                "rust/crates/tokspur-quant/src/lib.rs".to_owned(),
                sha256_hex(b"quant v1"),
            );
        }
        ReceiptKeySpec {
            action: "clippy".to_owned(),
            arm: arm.to_owned(),
            cpp_oracle: cpp_oracle.to_owned(),
            features: if arm == "ordinary" {
                String::new()
            } else {
                arm.to_owned()
            },
            toolchain_items,
            flag_items,
            cargo_lock_bytes: b"lock v1".to_vec(),
            rust_mk_bytes: b"rust.mk v1".to_vec(),
            source_digests,
        }
    }

    #[test]
    fn receipt_cache_negative_controls_mutation_and_corruption() {
        let ws_0 = base_spec("ordinary", "1");
        let fm_0 = base_spec("fast-math", "1");
        let ws_k0 = compute_receipt_key(&ws_0);
        let fm_k0 = compute_receipt_key(&fm_0);
        assert_ne!(ws_k0.content_key, fm_k0.content_key);

        // 1. Mutating tokspur-quant invalidates workspace ONLY, keeping fast-math hit!
        let mut ws_quant = base_spec("ordinary", "1");
        ws_quant.source_digests.insert(
            "rust/crates/tokspur-quant/src/lib.rs".to_owned(),
            sha256_hex(b"quant v2"),
        );
        let fm_quant = base_spec("fast-math", "1");
        assert_ne!(
            compute_receipt_key(&ws_quant).content_key,
            ws_k0.content_key
        );
        assert_eq!(
            compute_receipt_key(&fm_quant).content_key,
            fm_k0.content_key
        );

        // 2. Mutating tokspur-wire invalidates BOTH workspace and fast-math
        let mut fm_wire = base_spec("fast-math", "1");
        fm_wire.source_digests.insert(
            "rust/crates/tokspur-wire/src/worker.rs".to_owned(),
            sha256_hex(b"wire v2"),
        );
        assert_ne!(compute_receipt_key(&fm_wire).content_key, fm_k0.content_key);

        // 3. Toggling CPP_ORACLE or toolchain invalidates 100%
        let fm_no_oracle = base_spec("fast-math", "0");
        assert_ne!(
            compute_receipt_key(&fm_no_oracle).content_key,
            fm_k0.content_key
        );

        let mut fm_tc = base_spec("fast-math", "1");
        fm_tc
            .toolchain_items
            .insert("rustc".to_owned(), b"rustc 2.0.0".to_vec());
        assert_ne!(compute_receipt_key(&fm_tc).content_key, fm_k0.content_key);

        // 4. Atomic write, artifact verification, and fail-safe corruption detection
        let dir = TestDir::new("verify");
        let art_path = dir.path.join("wire-fast-math");
        let art_bytes = b"fake-macho-bytes";
        fs::write(&art_path, art_bytes).unwrap();
        let receipt_path = dir.path.join("clippy--fast-math.json");
        let meta = ReceiptMeta {
            schema_version: "1.0.0".to_owned(),
            action: "clippy".to_owned(),
            arm: "fast-math".to_owned(),
            cpp_oracle: "1".to_owned(),
            features: "fast-math".to_owned(),
            content_key: fm_k0.content_key.clone(),
            toolchain_sha256: fm_k0.toolchain_sha256,
            cargo_lock_sha256: fm_k0.cargo_lock_sha256,
            rust_mk_sha256: fm_k0.rust_mk_sha256,
            sources_sha256: fm_k0.sources_sha256,
            status: "passed".to_owned(),
            artifacts: vec![CachedArtifactMeta {
                rel_path: "wire-fast-math".to_owned(),
                sha256: sha256_hex(art_bytes),
                bytes: art_bytes.len() as u64,
            }],
        };
        write_receipt_atomically(&receipt_path, &meta).unwrap();
        assert!(
            verify_receipt_file(
                &receipt_path,
                "clippy",
                "fast-math",
                &fm_k0.content_key,
                Some(&dir.path),
                &["wire-fast-math"],
            )
            .unwrap()
            .is_some()
        );

        // Corrupt artifact -> fail-safe miss
        fs::write(&art_path, b"CORRUPTED_BYTES").unwrap();
        assert!(
            verify_receipt_file(
                &receipt_path,
                "clippy",
                "fast-math",
                &fm_k0.content_key,
                Some(&dir.path),
                &["wire-fast-math"],
            )
            .unwrap()
            .is_none()
        );

        // Corrupt receipt JSON -> fail-safe miss
        fs::write(&receipt_path, b"{corrupt-json").unwrap();
        assert!(
            verify_receipt_file(
                &receipt_path,
                "clippy",
                "fast-math",
                &fm_k0.content_key,
                Some(&dir.path),
                &["wire-fast-math"],
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn receipt_cache_rejects_path_traversal_symlinks_and_empty_fuzz_artifacts() {
        use std::os::unix::fs::symlink;

        assert_eq!(DOMAIN_TAG, b"tokspur-rust-receipt-v2\0");
        let mut spec = base_spec("fast-math", "1");
        spec.action = "fuzz-build".to_owned();
        let digest = compute_receipt_key(&spec);

        let dir = TestDir::new("security");
        let fuzz_root = dir.path.join("fuzz");
        let bin_dir = fuzz_root.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let outside = dir.path.join("outside");
        fs::write(&outside, b"outside-secret").unwrap();

        let receipt_path = dir.path.join("fuzz-build--fast-math.json");
        let mk_meta = |artifacts: Vec<CachedArtifactMeta>| ReceiptMeta {
            schema_version: "1.0.0".to_owned(),
            action: "fuzz-build".to_owned(),
            arm: "fast-math".to_owned(),
            cpp_oracle: "1".to_owned(),
            features: "fast-math".to_owned(),
            content_key: digest.content_key.clone(),
            toolchain_sha256: digest.toolchain_sha256.clone(),
            cargo_lock_sha256: digest.cargo_lock_sha256.clone(),
            rust_mk_sha256: digest.rust_mk_sha256.clone(),
            sources_sha256: digest.sources_sha256.clone(),
            status: "passed".to_owned(),
            artifacts,
        };

        // Empty fuzz-build artifacts must be rejected even if caller passes empty expected_outputs.
        write_receipt_atomically(&receipt_path, &mk_meta(Vec::new())).unwrap();
        assert!(
            verify_receipt_file(
                &receipt_path,
                "fuzz-build",
                "fast-math",
                &digest.content_key,
                Some(&fuzz_root),
                &[],
            )
            .unwrap()
            .is_none()
        );
        assert!(
            verify_receipt_file(
                &receipt_path,
                "fuzz-build",
                "fast-math",
                &digest.content_key,
                Some(&fuzz_root),
                &["wire-fast-math"],
            )
            .unwrap()
            .is_none()
        );

        // Path traversal ("../outside") and absolute path ("/etc/hosts") must be rejected.
        for bad_rel in ["../outside", "/etc/hosts", "bin/../outside"] {
            let meta = mk_meta(vec![CachedArtifactMeta {
                rel_path: bad_rel.to_owned(),
                sha256: sha256_hex(b"outside-secret"),
                bytes: b"outside-secret".len() as u64,
            }]);
            write_receipt_atomically(&receipt_path, &meta).unwrap();
            for action in ["fuzz-build", "clippy"] {
                let mut m2 = meta.clone();
                m2.action = action.to_owned();
                write_receipt_atomically(&receipt_path, &m2).unwrap();
                assert!(
                    verify_receipt_file(
                        &receipt_path,
                        action,
                        "fast-math",
                        &digest.content_key,
                        Some(&fuzz_root),
                        &[bad_rel],
                    )
                    .unwrap()
                    .is_none(),
                    "failed to reject {bad_rel} for {action}"
                );
            }
        }

        // Symlink inside bin/ must be rejected; real file in bin/ succeeds.
        let link_target = bin_dir.join("wire-fast-math");
        symlink(&outside, &link_target).unwrap();
        write_receipt_atomically(
            &receipt_path,
            &mk_meta(vec![CachedArtifactMeta {
                rel_path: "bin/wire-fast-math".to_owned(),
                sha256: sha256_hex(b"outside-secret"),
                bytes: b"outside-secret".len() as u64,
            }]),
        )
        .unwrap();
        assert!(
            verify_receipt_file(
                &receipt_path,
                "fuzz-build",
                "fast-math",
                &digest.content_key,
                Some(&fuzz_root),
                &["wire-fast-math"],
            )
            .unwrap()
            .is_none()
        );

        fs::remove_file(&link_target).unwrap();
        fs::write(&link_target, b"real-fuzz-bin").unwrap();
        write_receipt_atomically(
            &receipt_path,
            &mk_meta(vec![CachedArtifactMeta {
                rel_path: "bin/wire-fast-math".to_owned(),
                sha256: sha256_hex(b"real-fuzz-bin"),
                bytes: b"real-fuzz-bin".len() as u64,
            }]),
        )
        .unwrap();
        assert!(
            verify_receipt_file(
                &receipt_path,
                "fuzz-build",
                "fast-math",
                &digest.content_key,
                Some(&fuzz_root),
                &["wire-fast-math"],
            )
            .unwrap()
            .is_some()
        );
    }
}

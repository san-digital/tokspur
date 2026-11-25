//! Empirical V&V Manifest & SHA-256 single-use holdout locks with hash-chained runtime ledger.
//!
//! Uses pure `std` + `tokspur-json` (`to_chars_shortest`/`from_chars_f64` bit-exact round-trip,
//! strict unknown-field rejection, sorted 2-space pretty JSON) + `tokspur-text::sha256` +
//! `tokspur-sys` atomic `0o600` `O_CREAT|O_EXCL|O_NOFOLLOW` persistence.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::path::{Component, Path, PathBuf};

use tokspur_json::{Json, JsonLimits, parse_json_limited};
use tokspur_text::sha256::{hex, sha256};

use crate::error::EvalError;
use crate::json_util::{
    as_bool, as_f64, as_str, as_usize, atomic_write_private_bytes, format_pretty_sorted_json,
    json_f64, json_usize, require_object_keys, required_field,
};

pub const EPISTEMIC_VERIFICATION_HEADER: &str = "VERIFICATION ONLY: Deductive/Numerical Consistency Under Declared Assumptions; NOT Empirical Validation (Roache 1998; Oberkampf & Roy 2010).";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRole {
    Calibration,
    Holdout,
}

impl SplitRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Holdout => "holdout",
        }
    }

    fn from_str(s: &str) -> Result<Self, EvalError> {
        match s {
            "calibration" => Ok(Self::Calibration),
            "holdout" => Ok(Self::Holdout),
            other => Err(EvalError::ManifestIntegrityError(format!(
                "invalid split role '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WaveModeSpec {
    pub mode_k: usize,
    pub amplitude: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WaveCase {
    pub case_id: String,
    pub wave_speed_c: f64,
    pub domain_length_l: f64,
    pub nx: usize,
    pub nt: usize,
    pub cfl: f64,
    pub modes: Vec<WaveModeSpec>,
    pub reference_u_final: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WaveSplitDataset {
    pub split_id: String,
    pub role: SplitRole,
    pub domain: String,
    pub cases: Vec<WaveCase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitProvenance {
    pub partition_strategy: String,
    pub calibration_regime: String,
    pub holdout_regime: String,
    pub zero_leakage_policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationSplitEntry {
    pub split_id: String,
    pub relative_path: String,
    pub expected_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutSplitEntry {
    pub holdout_id: String,
    pub relative_path: String,
    pub expected_sha256: String,
    pub max_relative_l2_error: f64,
    pub single_use: bool,
    pub pre_registered_lock_commitment: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VvManifest {
    pub manifest_version: String,
    pub schema_id: String,
    pub pre_registration_id: String,
    pub created_utc: String,
    pub epistemic_header: String,
    pub split_provenance: SplitProvenance,
    pub calibration_splits: Vec<CalibrationSplitEntry>,
    pub holdout_splits: Vec<HoldoutSplitEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsumedHoldoutRecord {
    pub holdout_id: String,
    pub payload_sha256: String,
    pub lock_commitment: String,
    pub relative_l2_error: f64,
    pub passed: bool,
    pub consumption_token: String,
    pub prev_chain_hash: String,
    pub record_chain_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutLedger {
    pub ledger_version: String,
    pub pre_registration_id: String,
    pub entries: Vec<ConsumedHoldoutRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutEvalOutcome {
    pub holdout_id: String,
    pub payload_sha256: String,
    pub lock_commitment: String,
    pub consumption_token: String,
    pub cases_evaluated: usize,
    pub max_rel_l2_error: f64,
    pub max_allowed_rel_l2_error: f64,
    pub zero_calibration_leakage: bool,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&sha256(bytes))
}

pub fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn compute_lock_commitment(
    pre_registration_id: &str,
    holdout_id: &str,
    expected_sha256: &str,
    max_relative_l2_error: f64,
) -> String {
    let decimal = format!("{max_relative_l2_error:.6}");
    let msg = if decimal.parse::<f64>() == Ok(max_relative_l2_error) {
        format!("track4.vv.lock.v1|{pre_registration_id}|{holdout_id}|{expected_sha256}|{decimal}")
    } else {
        let bits = max_relative_l2_error.to_bits();
        format!(
            "track4.vv.lock.v2|{pre_registration_id}|{holdout_id}|{expected_sha256}|{bits:016x}"
        )
    };
    sha256_hex(msg.as_bytes())
}

pub fn compute_consumption_token(
    pre_registration_id: &str,
    holdout_id: &str,
    payload_sha256: &str,
    lock_commitment: &str,
    relative_l2_error: f64,
    passed: bool,
) -> String {
    let err_bits = if relative_l2_error == 0.0 {
        0.0_f64.to_bits()
    } else {
        relative_l2_error.to_bits()
    };
    let msg = format!(
        "track4.vv.consumed.v1|{pre_registration_id}|{holdout_id}|{payload_sha256}|{lock_commitment}|{err_bits:016x}|{passed}"
    );
    sha256_hex(msg.as_bytes())
}

pub fn compute_record_chain_hash(
    prev_chain_hash: &str,
    consumption_token: &str,
    payload_sha256: &str,
) -> String {
    let msg = format!("{prev_chain_hash}|{consumption_token}|{payload_sha256}");
    sha256_hex(msg.as_bytes())
}

// --- JSON codecs with strict deny_unknown_fields and bit-exact f64 ---

fn parse_json_bytes(bytes: &[u8], context: &'static str) -> Result<Json, EvalError> {
    parse_json_limited(bytes, JsonLimits::default())
        .map_err(|source| EvalError::Json { context, source })
}

impl WaveModeSpec {
    fn from_json(v: &Json) -> Result<Self, EvalError> {
        let fields = v.object().map_err(|source| EvalError::Json {
            context: "WaveModeSpec must be object",
            source,
        })?;
        require_object_keys(fields, &[b"mode_k", b"amplitude"])?;
        Ok(Self {
            mode_k: as_usize(required_field(fields, b"mode_k")?)?,
            amplitude: as_f64(required_field(fields, b"amplitude")?)?,
        })
    }

    fn to_json(&self) -> Result<Json, EvalError> {
        Ok(Json::new_object([
            ("mode_k", json_usize(self.mode_k)),
            ("amplitude", json_f64(self.amplitude)?),
        ]))
    }
}

impl WaveCase {
    fn from_json(v: &Json) -> Result<Self, EvalError> {
        let fields = v.object().map_err(|source| EvalError::Json {
            context: "WaveCase must be object",
            source,
        })?;
        require_object_keys(
            fields,
            &[
                b"case_id",
                b"wave_speed_c",
                b"domain_length_l",
                b"nx",
                b"nt",
                b"cfl",
                b"modes",
                b"reference_u_final",
            ],
        )?;
        let modes_arr = required_field(fields, b"modes")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "WaveCase.modes must be array",
                source,
            })?;
        let mut modes = Vec::with_capacity(modes_arr.len());
        for m in modes_arr {
            modes.push(WaveModeSpec::from_json(m)?);
        }
        let ref_arr = required_field(fields, b"reference_u_final")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "WaveCase.reference_u_final must be array",
                source,
            })?;
        let mut reference_u_final = Vec::with_capacity(ref_arr.len());
        for r in ref_arr {
            reference_u_final.push(as_f64(r)?);
        }
        Ok(Self {
            case_id: as_str(required_field(fields, b"case_id")?)?.to_owned(),
            wave_speed_c: as_f64(required_field(fields, b"wave_speed_c")?)?,
            domain_length_l: as_f64(required_field(fields, b"domain_length_l")?)?,
            nx: as_usize(required_field(fields, b"nx")?)?,
            nt: as_usize(required_field(fields, b"nt")?)?,
            cfl: as_f64(required_field(fields, b"cfl")?)?,
            modes,
            reference_u_final,
        })
    }

    fn to_json(&self) -> Result<Json, EvalError> {
        let mut modes_json = Vec::with_capacity(self.modes.len());
        for m in &self.modes {
            modes_json.push(m.to_json()?);
        }
        let mut ref_json = Vec::with_capacity(self.reference_u_final.len());
        for &r in &self.reference_u_final {
            ref_json.push(json_f64(r)?);
        }
        Ok(Json::new_object([
            ("case_id", Json::from(self.case_id.as_str())),
            ("wave_speed_c", json_f64(self.wave_speed_c)?),
            ("domain_length_l", json_f64(self.domain_length_l)?),
            ("nx", json_usize(self.nx)),
            ("nt", json_usize(self.nt)),
            ("cfl", json_f64(self.cfl)?),
            ("modes", Json::Array(modes_json)),
            ("reference_u_final", Json::Array(ref_json)),
        ]))
    }
}

impl WaveSplitDataset {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let root = parse_json_bytes(bytes, "failed parsing WaveSplitDataset JSON")?;
        let fields = root.object().map_err(|source| EvalError::Json {
            context: "WaveSplitDataset must be object",
            source,
        })?;
        require_object_keys(fields, &[b"split_id", b"role", b"domain", b"cases"])?;
        let cases_arr = required_field(fields, b"cases")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "WaveSplitDataset.cases must be array",
                source,
            })?;
        let mut cases = Vec::with_capacity(cases_arr.len());
        for c in cases_arr {
            cases.push(WaveCase::from_json(c)?);
        }
        Ok(Self {
            split_id: as_str(required_field(fields, b"split_id")?)?.to_owned(),
            role: SplitRole::from_str(as_str(required_field(fields, b"role")?)?)?,
            domain: as_str(required_field(fields, b"domain")?)?.to_owned(),
            cases,
        })
    }

    pub fn to_pretty_json_bytes(&self) -> Result<Vec<u8>, EvalError> {
        let mut cases_json = Vec::with_capacity(self.cases.len());
        for c in &self.cases {
            cases_json.push(c.to_json()?);
        }
        let root = Json::new_object([
            ("split_id", Json::from(self.split_id.as_str())),
            ("role", Json::from(self.role.as_str())),
            ("domain", Json::from(self.domain.as_str())),
            ("cases", Json::Array(cases_json)),
        ]);
        format_pretty_sorted_json(&root)
    }
}

impl SplitProvenance {
    fn from_json(v: &Json) -> Result<Self, EvalError> {
        let fields = v.object().map_err(|source| EvalError::Json {
            context: "SplitProvenance must be object",
            source,
        })?;
        require_object_keys(
            fields,
            &[
                b"partition_strategy",
                b"calibration_regime",
                b"holdout_regime",
                b"zero_leakage_policy",
            ],
        )?;
        Ok(Self {
            partition_strategy: as_str(required_field(fields, b"partition_strategy")?)?.to_owned(),
            calibration_regime: as_str(required_field(fields, b"calibration_regime")?)?.to_owned(),
            holdout_regime: as_str(required_field(fields, b"holdout_regime")?)?.to_owned(),
            zero_leakage_policy: as_str(required_field(fields, b"zero_leakage_policy")?)?
                .to_owned(),
        })
    }

    fn to_json(&self) -> Json {
        Json::new_object([
            (
                "partition_strategy",
                Json::from(self.partition_strategy.as_str()),
            ),
            (
                "calibration_regime",
                Json::from(self.calibration_regime.as_str()),
            ),
            ("holdout_regime", Json::from(self.holdout_regime.as_str())),
            (
                "zero_leakage_policy",
                Json::from(self.zero_leakage_policy.as_str()),
            ),
        ])
    }
}

impl CalibrationSplitEntry {
    fn from_json(v: &Json) -> Result<Self, EvalError> {
        let fields = v.object().map_err(|source| EvalError::Json {
            context: "CalibrationSplitEntry must be object",
            source,
        })?;
        require_object_keys(fields, &[b"split_id", b"relative_path", b"expected_sha256"])?;
        Ok(Self {
            split_id: as_str(required_field(fields, b"split_id")?)?.to_owned(),
            relative_path: as_str(required_field(fields, b"relative_path")?)?.to_owned(),
            expected_sha256: as_str(required_field(fields, b"expected_sha256")?)?.to_owned(),
        })
    }

    fn to_json(&self) -> Json {
        Json::new_object([
            ("split_id", Json::from(self.split_id.as_str())),
            ("relative_path", Json::from(self.relative_path.as_str())),
            ("expected_sha256", Json::from(self.expected_sha256.as_str())),
        ])
    }
}

impl HoldoutSplitEntry {
    fn from_json(v: &Json) -> Result<Self, EvalError> {
        let fields = v.object().map_err(|source| EvalError::Json {
            context: "HoldoutSplitEntry must be object",
            source,
        })?;
        require_object_keys(
            fields,
            &[
                b"holdout_id",
                b"relative_path",
                b"expected_sha256",
                b"max_relative_l2_error",
                b"single_use",
                b"pre_registered_lock_commitment",
            ],
        )?;
        Ok(Self {
            holdout_id: as_str(required_field(fields, b"holdout_id")?)?.to_owned(),
            relative_path: as_str(required_field(fields, b"relative_path")?)?.to_owned(),
            expected_sha256: as_str(required_field(fields, b"expected_sha256")?)?.to_owned(),
            max_relative_l2_error: as_f64(required_field(fields, b"max_relative_l2_error")?)?,
            single_use: as_bool(required_field(fields, b"single_use")?)?,
            pre_registered_lock_commitment: as_str(required_field(
                fields,
                b"pre_registered_lock_commitment",
            )?)?
            .to_owned(),
        })
    }

    fn to_json(&self) -> Result<Json, EvalError> {
        Ok(Json::new_object([
            ("holdout_id", Json::from(self.holdout_id.as_str())),
            ("relative_path", Json::from(self.relative_path.as_str())),
            ("expected_sha256", Json::from(self.expected_sha256.as_str())),
            (
                "max_relative_l2_error",
                json_f64(self.max_relative_l2_error)?,
            ),
            ("single_use", Json::Bool(self.single_use)),
            (
                "pre_registered_lock_commitment",
                Json::from(self.pre_registered_lock_commitment.as_str()),
            ),
        ]))
    }
}

impl VvManifest {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let root = parse_json_bytes(bytes, "failed parsing VvManifest JSON")?;
        let fields = root.object().map_err(|source| EvalError::Json {
            context: "VvManifest must be object",
            source,
        })?;
        require_object_keys(
            fields,
            &[
                b"manifest_version",
                b"schema_id",
                b"pre_registration_id",
                b"created_utc",
                b"epistemic_header",
                b"split_provenance",
                b"calibration_splits",
                b"holdout_splits",
            ],
        )?;
        let cal_arr = required_field(fields, b"calibration_splits")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "calibration_splits must be array",
                source,
            })?;
        let mut calibration_splits = Vec::with_capacity(cal_arr.len());
        for c in cal_arr {
            calibration_splits.push(CalibrationSplitEntry::from_json(c)?);
        }
        let hold_arr = required_field(fields, b"holdout_splits")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "holdout_splits must be array",
                source,
            })?;
        let mut holdout_splits = Vec::with_capacity(hold_arr.len());
        for h in hold_arr {
            holdout_splits.push(HoldoutSplitEntry::from_json(h)?);
        }
        Ok(Self {
            manifest_version: as_str(required_field(fields, b"manifest_version")?)?.to_owned(),
            schema_id: as_str(required_field(fields, b"schema_id")?)?.to_owned(),
            pre_registration_id: as_str(required_field(fields, b"pre_registration_id")?)?
                .to_owned(),
            created_utc: as_str(required_field(fields, b"created_utc")?)?.to_owned(),
            epistemic_header: as_str(required_field(fields, b"epistemic_header")?)?.to_owned(),
            split_provenance: SplitProvenance::from_json(required_field(
                fields,
                b"split_provenance",
            )?)?,
            calibration_splits,
            holdout_splits,
        })
    }

    pub fn to_pretty_json_bytes(&self) -> Result<Vec<u8>, EvalError> {
        let cal_json: Vec<Json> = self
            .calibration_splits
            .iter()
            .map(|c| c.to_json())
            .collect();
        let mut hold_json = Vec::with_capacity(self.holdout_splits.len());
        for h in &self.holdout_splits {
            hold_json.push(h.to_json()?);
        }
        let root = Json::new_object([
            (
                "manifest_version",
                Json::from(self.manifest_version.as_str()),
            ),
            ("schema_id", Json::from(self.schema_id.as_str())),
            (
                "pre_registration_id",
                Json::from(self.pre_registration_id.as_str()),
            ),
            ("created_utc", Json::from(self.created_utc.as_str())),
            (
                "epistemic_header",
                Json::from(self.epistemic_header.as_str()),
            ),
            ("split_provenance", self.split_provenance.to_json()),
            ("calibration_splits", Json::Array(cal_json)),
            ("holdout_splits", Json::Array(hold_json)),
        ]);
        format_pretty_sorted_json(&root)
    }
}

impl ConsumedHoldoutRecord {
    fn from_json(v: &Json) -> Result<Self, EvalError> {
        let fields = v.object().map_err(|source| EvalError::Json {
            context: "ConsumedHoldoutRecord must be object",
            source,
        })?;
        require_object_keys(
            fields,
            &[
                b"holdout_id",
                b"payload_sha256",
                b"lock_commitment",
                b"relative_l2_error",
                b"passed",
                b"consumption_token",
                b"prev_chain_hash",
                b"record_chain_hash",
            ],
        )?;
        Ok(Self {
            holdout_id: as_str(required_field(fields, b"holdout_id")?)?.to_owned(),
            payload_sha256: as_str(required_field(fields, b"payload_sha256")?)?.to_owned(),
            lock_commitment: as_str(required_field(fields, b"lock_commitment")?)?.to_owned(),
            relative_l2_error: as_f64(required_field(fields, b"relative_l2_error")?)?,
            passed: as_bool(required_field(fields, b"passed")?)?,
            consumption_token: as_str(required_field(fields, b"consumption_token")?)?.to_owned(),
            prev_chain_hash: as_str(required_field(fields, b"prev_chain_hash")?)?.to_owned(),
            record_chain_hash: as_str(required_field(fields, b"record_chain_hash")?)?.to_owned(),
        })
    }

    fn to_json(&self) -> Result<Json, EvalError> {
        Ok(Json::new_object([
            ("holdout_id", Json::from(self.holdout_id.as_str())),
            ("payload_sha256", Json::from(self.payload_sha256.as_str())),
            ("lock_commitment", Json::from(self.lock_commitment.as_str())),
            ("relative_l2_error", json_f64(self.relative_l2_error)?),
            ("passed", Json::Bool(self.passed)),
            (
                "consumption_token",
                Json::from(self.consumption_token.as_str()),
            ),
            ("prev_chain_hash", Json::from(self.prev_chain_hash.as_str())),
            (
                "record_chain_hash",
                Json::from(self.record_chain_hash.as_str()),
            ),
        ]))
    }
}

impl HoldoutLedger {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        let root = parse_json_bytes(bytes, "failed parsing HoldoutLedger JSON")?;
        let fields = root.object().map_err(|source| EvalError::Json {
            context: "HoldoutLedger must be object",
            source,
        })?;
        require_object_keys(
            fields,
            &[b"ledger_version", b"pre_registration_id", b"entries"],
        )?;
        let entries_arr = required_field(fields, b"entries")?
            .array()
            .map_err(|source| EvalError::Json {
                context: "HoldoutLedger.entries must be array",
                source,
            })?;
        let mut entries = Vec::with_capacity(entries_arr.len());
        for e in entries_arr {
            entries.push(ConsumedHoldoutRecord::from_json(e)?);
        }
        Ok(Self {
            ledger_version: as_str(required_field(fields, b"ledger_version")?)?.to_owned(),
            pre_registration_id: as_str(required_field(fields, b"pre_registration_id")?)?
                .to_owned(),
            entries,
        })
    }

    pub fn to_pretty_json_bytes(&self) -> Result<Vec<u8>, EvalError> {
        let mut entries_json = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            entries_json.push(e.to_json()?);
        }
        let root = Json::new_object([
            ("ledger_version", Json::from(self.ledger_version.as_str())),
            (
                "pre_registration_id",
                Json::from(self.pre_registration_id.as_str()),
            ),
            ("entries", Json::Array(entries_json)),
        ]);
        format_pretty_sorted_json(&root)
    }
}

// --- Path traversal and symlink safety ---

/// Canonicalizes `path` even when trailing components do not yet exist on disk.
pub fn canonicalize_allowing_nonexistent(path: &Path) -> Result<PathBuf, String> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("failed to read current_dir: {e}"))?
            .join(path)
    };
    if let Ok(canon) = std::fs::canonicalize(&abs) {
        return Ok(canon);
    }

    let mut cur = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::Prefix(p) => cur.push(p.as_os_str()),
            Component::RootDir => cur.push(Component::RootDir),
            Component::CurDir => {}
            Component::ParentDir => {
                if !cur.pop() {
                    return Err(format!(
                        "cannot traverse parent beyond root in '{}'",
                        path.display()
                    ));
                }
            }
            Component::Normal(seg) => {
                cur.push(seg);
                match std::fs::symlink_metadata(&cur) {
                    Ok(meta) => {
                        if let Ok(canon) = std::fs::canonicalize(&cur) {
                            cur = canon;
                        } else if meta.file_type().is_symlink() {
                            return Err(format!(
                                "dangling symlink encountered at '{}'",
                                cur.display()
                            ));
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(format!(
                            "failed inspecting path component '{}': {e}",
                            cur.display()
                        ));
                    }
                }
            }
        }
    }
    if let Ok(canon) = std::fs::canonicalize(&cur) {
        cur = canon;
    }
    Ok(cur)
}

pub fn resolve_safe_relative_path(base_dir: &Path, rel: &str) -> Result<PathBuf, EvalError> {
    if rel.trim().is_empty() || rel.contains('\\') {
        return Err(EvalError::ManifestIntegrityError(format!(
            "invalid relative split path '{rel}'"
        )));
    }
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err(EvalError::ManifestIntegrityError(format!(
            "split path '{rel}' must be relative"
        )));
    }
    for comp in rel_path.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(EvalError::ManifestIntegrityError(format!(
                    "disallowed path component in split path '{rel}'"
                )));
            }
        }
    }
    let canon_base = canonicalize_allowing_nonexistent(base_dir).map_err(|e| {
        EvalError::ManifestIntegrityError(format!(
            "invalid manifest base directory '{}': {e}",
            base_dir.display()
        ))
    })?;
    let joined = base_dir.join(rel_path);
    let canon_target = canonicalize_allowing_nonexistent(&joined).map_err(|e| {
        EvalError::ManifestIntegrityError(format!(
            "invalid split path '{rel}' under '{}': {e}",
            base_dir.display()
        ))
    })?;
    if !canon_target.starts_with(&canon_base) {
        return Err(EvalError::ManifestIntegrityError(format!(
            "split path '{rel}' escapes base directory via symlink"
        )));
    }
    Ok(canon_target)
}

// --- Zero-leakage fingerprinting ---

#[inline]
fn norm_float_bits(x: f64) -> u64 {
    if x.is_nan() {
        f64::NAN.to_bits()
    } else if x == 0.0 {
        0.0_f64.to_bits()
    } else {
        x.to_bits()
    }
}

pub fn canonical_case_fingerprint(c: &WaveCase) -> String {
    let mut coalesced: BTreeMap<usize, f64> = BTreeMap::new();
    for m in &c.modes {
        *coalesced.entry(m.mode_k).or_insert(0.0) += m.amplitude;
    }
    let mut s = format!(
        "c={:016x}|L={:016x}|nx={}|nt={}|cfl={:016x}|",
        norm_float_bits(c.wave_speed_c),
        norm_float_bits(c.domain_length_l),
        c.nx,
        c.nt,
        norm_float_bits(c.cfl)
    );
    for (k, amp) in coalesced {
        if amp == 0.0 {
            continue;
        }
        s.push_str(&format!("{k}:{:016x},", norm_float_bits(amp)));
    }
    sha256_hex(s.as_bytes())
}

// --- Manifest & Ledger validation ---

pub fn load_and_validate_manifest(manifest_dir: &Path) -> Result<VvManifest, EvalError> {
    let manifest_path = manifest_dir.join("manifest.json");
    let raw = std::fs::read(&manifest_path).map_err(|source| EvalError::Io {
        context: "cannot read manifest.json".to_owned(),
        source,
    })?;
    let manifest = VvManifest::from_json_bytes(&raw)?;

    if manifest.manifest_version != "1.0.0" {
        return Err(EvalError::ManifestIntegrityError(format!(
            "unsupported manifest_version '{}'",
            manifest.manifest_version
        )));
    }
    if manifest.schema_id != "track4.empirical_vv_manifest.v1" {
        return Err(EvalError::ManifestIntegrityError(format!(
            "unsupported schema_id '{}'",
            manifest.schema_id
        )));
    }
    if manifest.pre_registration_id.trim().is_empty() {
        return Err(EvalError::ManifestIntegrityError(
            "pre_registration_id cannot be empty".to_owned(),
        ));
    }
    if manifest.epistemic_header != EPISTEMIC_VERIFICATION_HEADER {
        return Err(EvalError::ManifestIntegrityError(
            "manifest epistemic_header does not match mandatory EPISTEMIC_VERIFICATION_HEADER"
                .to_owned(),
        ));
    }
    if manifest.calibration_splits.is_empty() || manifest.holdout_splits.is_empty() {
        return Err(EvalError::ManifestIntegrityError(
            "manifest must contain at least one calibration split and one holdout split".to_owned(),
        ));
    }

    let mut seen_cal_ids = BTreeSet::new();
    for cal in &manifest.calibration_splits {
        if cal.split_id.trim().is_empty() || !seen_cal_ids.insert(cal.split_id.clone()) {
            return Err(EvalError::ManifestIntegrityError(format!(
                "invalid or duplicate calibration split_id '{}'",
                cal.split_id
            )));
        }
        if !is_valid_sha256_hex(&cal.expected_sha256) {
            return Err(EvalError::ManifestIntegrityError(format!(
                "invalid calibration sha256 digest for '{}'",
                cal.split_id
            )));
        }
        let _resolved = resolve_safe_relative_path(manifest_dir, &cal.relative_path)?;
    }

    let mut seen_holdout_ids = BTreeSet::new();
    for hold in &manifest.holdout_splits {
        if hold.holdout_id.trim().is_empty() || !seen_holdout_ids.insert(hold.holdout_id.clone()) {
            return Err(EvalError::ManifestIntegrityError(format!(
                "invalid or duplicate holdout_id '{}'",
                hold.holdout_id
            )));
        }
        if !is_valid_sha256_hex(&hold.expected_sha256) {
            return Err(EvalError::ManifestIntegrityError(format!(
                "invalid holdout sha256 digest for '{}'",
                hold.holdout_id
            )));
        }
        if !hold.single_use {
            return Err(EvalError::ManifestIntegrityError(format!(
                "holdout '{}' must enforce single_use = true",
                hold.holdout_id
            )));
        }
        if !hold.max_relative_l2_error.is_finite()
            || hold.max_relative_l2_error <= 0.0
            || hold.max_relative_l2_error > 0.05
        {
            return Err(EvalError::ManifestIntegrityError(format!(
                "holdout '{}' max_relative_l2_error {} must be in (0, 0.05]",
                hold.holdout_id, hold.max_relative_l2_error
            )));
        }
        let expected_commitment = compute_lock_commitment(
            &manifest.pre_registration_id,
            &hold.holdout_id,
            &hold.expected_sha256,
            hold.max_relative_l2_error,
        );
        if hold.pre_registered_lock_commitment != expected_commitment {
            return Err(EvalError::ManifestIntegrityError(format!(
                "pre_registered_lock_commitment mismatch for '{}': expected {expected_commitment}, found {}",
                hold.holdout_id, hold.pre_registered_lock_commitment
            )));
        }
        let _resolved = resolve_safe_relative_path(manifest_dir, &hold.relative_path)?;
    }

    Ok(manifest)
}

pub fn load_and_verify_ledger(
    ledger_path: &Path,
    pre_registration_id: &str,
) -> Result<HoldoutLedger, EvalError> {
    if !ledger_path.exists() {
        return Ok(HoldoutLedger {
            ledger_version: "1.0.0".to_owned(),
            pre_registration_id: pre_registration_id.to_owned(),
            entries: Vec::new(),
        });
    }

    let raw = std::fs::read(ledger_path)
        .map_err(|e| EvalError::LedgerTampered(format!("failed reading ledger: {e}")))?;
    let ledger = HoldoutLedger::from_json_bytes(&raw)
        .map_err(|e| EvalError::LedgerTampered(format!("corrupted ledger JSON: {e}")))?;

    if ledger.ledger_version != "1.0.0" {
        return Err(EvalError::LedgerTampered(format!(
            "unexpected ledger_version '{}'",
            ledger.ledger_version
        )));
    }
    if ledger.pre_registration_id != pre_registration_id {
        return Err(EvalError::LedgerTampered(format!(
            "ledger pre_registration_id '{}' does not match manifest '{pre_registration_id}'",
            ledger.pre_registration_id
        )));
    }

    let mut expected_prev = "GENESIS".to_owned();
    for (idx, rec) in ledger.entries.iter().enumerate() {
        if !rec.relative_l2_error.is_finite() || rec.relative_l2_error < 0.0 {
            return Err(EvalError::LedgerTampered(format!(
                "non-finite or negative relative_l2_error at ledger entry {idx} ('{}')",
                rec.holdout_id
            )));
        }
        if rec.prev_chain_hash != expected_prev {
            return Err(EvalError::LedgerTampered(format!(
                "chain break at ledger entry {idx}: expected prev_chain_hash '{expected_prev}', got '{}'",
                rec.prev_chain_hash
            )));
        }
        let expected_token = compute_consumption_token(
            pre_registration_id,
            &rec.holdout_id,
            &rec.payload_sha256,
            &rec.lock_commitment,
            rec.relative_l2_error,
            rec.passed,
        );
        if rec.consumption_token != expected_token {
            return Err(EvalError::LedgerTampered(format!(
                "consumption_token mismatch at ledger entry {idx} ('{}')",
                rec.holdout_id
            )));
        }
        let expected_chain = compute_record_chain_hash(
            &rec.prev_chain_hash,
            &rec.consumption_token,
            &rec.payload_sha256,
        );
        if rec.record_chain_hash != expected_chain {
            return Err(EvalError::LedgerTampered(format!(
                "record_chain_hash mismatch at ledger entry {idx} ('{}')",
                rec.holdout_id
            )));
        }
        expected_prev = rec.record_chain_hash.clone();
    }

    Ok(ledger)
}

fn persist_ledger_atomically(ledger_path: &Path, ledger: &HoldoutLedger) -> Result<(), EvalError> {
    let encoded = ledger.to_pretty_json_bytes()?;
    atomic_write_private_bytes(ledger_path, &encoded)
}

fn validate_wave_case_finite(case: &WaveCase) -> Result<(), EvalError> {
    if !case.wave_speed_c.is_finite() || !case.domain_length_l.is_finite() || !case.cfl.is_finite()
    {
        return Err(EvalError::ManifestIntegrityError(format!(
            "case '{}' contains non-finite scalar parameter",
            case.case_id
        )));
    }
    if case.modes.iter().any(|m| !m.amplitude.is_finite()) {
        return Err(EvalError::ManifestIntegrityError(format!(
            "case '{}' contains non-finite mode amplitude",
            case.case_id
        )));
    }
    if case.reference_u_final.iter().any(|x| !x.is_finite()) {
        return Err(EvalError::ManifestIntegrityError(format!(
            "case '{}' contains non-finite reference_u_final value",
            case.case_id
        )));
    }
    Ok(())
}

pub fn verify_zero_calibration_leakage(
    cal_splits: &[(&str, &WaveSplitDataset)],
    holdout_sha256: &str,
    holdout: &WaveSplitDataset,
) -> Result<(), EvalError> {
    let mut cal_case_ids = BTreeSet::new();
    let mut cal_fingerprints = BTreeSet::new();

    for &(cal_sha, cal) in cal_splits {
        if cal_sha == holdout_sha256 {
            return Err(EvalError::CalibrationLeakageDetected(format!(
                "calibration split '{}' shares identical SHA-256 '{cal_sha}' with holdout '{}'",
                cal.split_id, holdout.split_id
            )));
        }
        if cal.split_id == holdout.split_id {
            return Err(EvalError::CalibrationLeakageDetected(format!(
                "calibration and holdout share split_id '{}'",
                holdout.split_id
            )));
        }
        for c in &cal.cases {
            validate_wave_case_finite(c)?;
            cal_case_ids.insert(c.case_id.as_str());
            cal_fingerprints.insert(canonical_case_fingerprint(c));
        }
    }

    for hc in &holdout.cases {
        validate_wave_case_finite(hc)?;
        if cal_case_ids.contains(hc.case_id.as_str()) {
            return Err(EvalError::CalibrationLeakageDetected(format!(
                "holdout case_id '{}' overlaps calibration split",
                hc.case_id
            )));
        }
        let fp = canonical_case_fingerprint(hc);
        if cal_fingerprints.contains(&fp) {
            return Err(EvalError::CalibrationLeakageDetected(format!(
                "holdout case '{}' shares identical physics tuple fingerprint '{fp}' with calibration split",
                hc.case_id
            )));
        }
    }

    Ok(())
}

fn discrete_l2_norm(u: &[f64], dx: f64) -> f64 {
    let sq_sum: f64 = u.iter().map(|&v| v * v).sum();
    (dx * sq_sum).sqrt()
}

fn evaluate_wave_case(case: &WaveCase) -> Result<f64, EvalError> {
    validate_wave_case_finite(case)?;
    if case.reference_u_final.len() != case.nx + 1 {
        return Err(EvalError::ManifestIntegrityError(format!(
            "holdout case '{}' reference_u_final length {} != nx+1 ({})",
            case.case_id,
            case.reference_u_final.len(),
            case.nx + 1
        )));
    }
    if case.modes.is_empty() {
        return Err(EvalError::ManifestIntegrityError(format!(
            "holdout case '{}' has zero modes",
            case.case_id
        )));
    }
    if case.domain_length_l <= 0.0 || case.wave_speed_c <= 0.0 || case.nx < 4 || case.nt < 1 {
        return Err(EvalError::ManifestIntegrityError(format!(
            "holdout case '{}' has invalid wave parameters",
            case.case_id
        )));
    }
    if case.cfl <= 0.0 || case.cfl > 1.0 {
        return Err(EvalError::NumericsFailed(format!(
            "CFL violation in holdout case '{}': {}",
            case.case_id, case.cfl
        )));
    }

    let n = case.nx;
    let dx = case.domain_length_l / (n as f64);
    let mut dt = case.cfl * dx / case.wave_speed_c;
    while case.wave_speed_c * dt / dx > 1.0 && dt > 0.0 {
        dt = dt.next_down();
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(EvalError::NumericsFailed(format!(
            "invalid timestep in holdout case '{}'",
            case.case_id
        )));
    }

    let mut u_prev = vec![0.0_f64; n + 1];
    for (j, val) in u_prev.iter_mut().enumerate().take(n).skip(1) {
        let x = (j as f64) * dx;
        let mut sum = 0.0;
        for m in &case.modes {
            sum += m.amplitude * ((m.mode_k as f64) * PI * x / case.domain_length_l).sin();
        }
        if !sum.is_finite() {
            return Err(EvalError::NumericsFailed(format!(
                "non-finite initial state in holdout case '{}'",
                case.case_id
            )));
        }
        *val = sum;
    }

    let r = case.wave_speed_c * dt / dx;
    let r2 = r * r;
    let c0 = 2.0 - 2.0 * r2;

    let mut u_curr = vec![0.0_f64; n + 1];
    for j in 1..n {
        u_curr[j] = u_prev[j] + 0.5 * r2 * (u_prev[j + 1] - 2.0 * u_prev[j] + u_prev[j - 1]);
    }

    let mut u_next = vec![0.0_f64; n + 1];
    for _step in 1..case.nt {
        for j in 1..n {
            u_next[j] = c0 * u_curr[j] + r2 * (u_curr[j - 1] + u_curr[j + 1]) - u_prev[j];
        }
        u_next[0] = 0.0;
        u_next[n] = 0.0;
        std::mem::swap(&mut u_prev, &mut u_curr);
        std::mem::swap(&mut u_curr, &mut u_next);
    }

    if u_curr.iter().any(|x| !x.is_finite()) {
        return Err(EvalError::NumericsFailed(format!(
            "non-finite final state in holdout case '{}'",
            case.case_id
        )));
    }

    let mut diff = Vec::with_capacity(n + 1);
    for (&u_num, &u_ref) in u_curr.iter().zip(case.reference_u_final.iter()) {
        diff.push(u_num - u_ref);
    }

    let err_l2 = discrete_l2_norm(&diff, dx);
    let ref_l2 = discrete_l2_norm(&case.reference_u_final, dx);
    if !ref_l2.is_finite() || ref_l2 <= 0.0 {
        return Err(EvalError::NumericsFailed(format!(
            "invalid reference L2 norm in holdout case '{}'",
            case.case_id
        )));
    }
    if !err_l2.is_finite() {
        return Err(EvalError::NumericsFailed(format!(
            "non-finite error L2 norm in holdout case '{}'",
            case.case_id
        )));
    }
    let rel_l2 = err_l2 / ref_l2;
    if !rel_l2.is_finite() {
        return Err(EvalError::NumericsFailed(format!(
            "non-finite relative L2 error in holdout case '{}'",
            case.case_id
        )));
    }
    Ok(rel_l2)
}

pub fn ensure_ledger_isolated_from_manifest(
    manifest_dir: &Path,
    manifest: &VvManifest,
    ledger_path: &Path,
) -> Result<(), EvalError> {
    let ledger_abs = canonicalize_allowing_nonexistent(ledger_path)
        .map_err(|e| EvalError::LedgerTampered(format!("invalid ledger_path: {e}")))?;
    let manifest_abs = canonicalize_allowing_nonexistent(&manifest_dir.join("manifest.json"))
        .map_err(|e| EvalError::ManifestIntegrityError(format!("invalid manifest path: {e}")))?;
    if ledger_abs == manifest_abs {
        return Err(EvalError::LedgerTampered(
            "ledger_path cannot overwrite manifest.json".to_owned(),
        ));
    }
    for cal in &manifest.calibration_splits {
        let p_abs = resolve_safe_relative_path(manifest_dir, &cal.relative_path)?;
        if ledger_abs == p_abs {
            return Err(EvalError::LedgerTampered(
                "ledger_path cannot overwrite calibration split file".to_owned(),
            ));
        }
    }
    for hold in &manifest.holdout_splits {
        let p_abs = resolve_safe_relative_path(manifest_dir, &hold.relative_path)?;
        if ledger_abs == p_abs {
            return Err(EvalError::LedgerTampered(
                "ledger_path cannot overwrite holdout split file".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Unlocks and evaluates a single-use holdout split, recording consumption into `ledger_path`
/// before returning any accuracy failure and refusing any subsequent unlock attempt by
/// `holdout_id`, `payload_sha256`, or `lock_commitment`.
pub fn unlock_and_evaluate_holdout(
    manifest_dir: &Path,
    holdout_id: &str,
    ledger_path: &Path,
) -> Result<HoldoutEvalOutcome, EvalError> {
    let manifest = load_and_validate_manifest(manifest_dir)?;
    ensure_ledger_isolated_from_manifest(manifest_dir, &manifest, ledger_path)?;

    let hold_entry = manifest
        .holdout_splits
        .iter()
        .find(|h| h.holdout_id == holdout_id)
        .ok_or_else(|| {
            EvalError::ManifestIntegrityError(format!(
                "holdout_id '{holdout_id}' not found in manifest"
            ))
        })?;

    // Verify all calibration split SHA-256 digests and parse them
    let mut cal_bytes_vec: Vec<(String, String, Vec<u8>)> =
        Vec::with_capacity(manifest.calibration_splits.len());
    for cal in &manifest.calibration_splits {
        let path = resolve_safe_relative_path(manifest_dir, &cal.relative_path)?;
        let bytes = std::fs::read(&path).map_err(|source| EvalError::Io {
            context: format!("cannot read calibration split '{}'", cal.split_id),
            source,
        })?;
        let actual_sha = sha256_hex(&bytes);
        if actual_sha != cal.expected_sha256 {
            return Err(EvalError::CalibrationHashMismatch {
                split_id: cal.split_id.clone(),
                expected: cal.expected_sha256.clone(),
                actual: actual_sha,
            });
        }
        cal_bytes_vec.push((cal.split_id.clone(), actual_sha, bytes));
    }

    // Atomic replacement changes the ledger inode. Hold its parent directory lock through the read, decision and write.
    let parent = ledger_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let ledger_dir = std::fs::create_dir_all(parent)
        .and_then(|()| std::fs::File::open(parent))
        .map_err(|source| EvalError::Io {
            context: "cannot open holdout ledger directory".to_owned(),
            source,
        })?;
    ledger_dir.lock().map_err(|source| EvalError::Io {
        context: "cannot lock holdout ledger directory".to_owned(),
        source,
    })?;

    let mut ledger = load_and_verify_ledger(ledger_path, &manifest.pre_registration_id)?;
    for rec in &ledger.entries {
        if rec.holdout_id == hold_entry.holdout_id
            || rec.payload_sha256 == hold_entry.expected_sha256
            || rec.lock_commitment == hold_entry.pre_registered_lock_commitment
        {
            return Err(EvalError::HoldoutAlreadyConsumed {
                holdout_id: hold_entry.holdout_id.clone(),
                payload_sha256: hold_entry.expected_sha256.clone(),
                consumption_token: rec.consumption_token.clone(),
            });
        }
    }

    // Read & SHA-256 verify holdout payload
    let hold_path = resolve_safe_relative_path(manifest_dir, &hold_entry.relative_path)?;
    let hold_bytes = std::fs::read(&hold_path).map_err(|source| EvalError::Io {
        context: format!("cannot read holdout split '{}'", hold_entry.holdout_id),
        source,
    })?;
    let hold_sha = sha256_hex(&hold_bytes);
    if hold_sha != hold_entry.expected_sha256 {
        return Err(EvalError::HoldoutHashMismatch {
            holdout_id: hold_entry.holdout_id.clone(),
            expected: hold_entry.expected_sha256.clone(),
            actual: hold_sha,
        });
    }

    let mut cal_datasets: Vec<(String, WaveSplitDataset)> = Vec::with_capacity(cal_bytes_vec.len());
    for (cal_id, cal_sha, bytes) in &cal_bytes_vec {
        let ds = WaveSplitDataset::from_json_bytes(bytes)?;
        if ds.role != SplitRole::Calibration || ds.split_id != *cal_id {
            return Err(EvalError::ManifestIntegrityError(format!(
                "calibration split '{cal_id}' metadata mismatch"
            )));
        }
        cal_datasets.push((cal_sha.clone(), ds));
    }

    let hold_ds = WaveSplitDataset::from_json_bytes(&hold_bytes)?;
    if hold_ds.role != SplitRole::Holdout || hold_ds.split_id != hold_entry.holdout_id {
        return Err(EvalError::ManifestIntegrityError(format!(
            "holdout split '{}' role or split_id mismatch",
            hold_entry.holdout_id
        )));
    }
    if hold_ds.cases.is_empty() {
        return Err(EvalError::ManifestIntegrityError(format!(
            "holdout split '{}' has zero cases",
            hold_entry.holdout_id
        )));
    }

    let cal_refs: Vec<(&str, &WaveSplitDataset)> = cal_datasets
        .iter()
        .map(|(s, ds)| (s.as_str(), ds))
        .collect();
    verify_zero_calibration_leakage(&cal_refs, &hold_sha, &hold_ds)?;

    let mut max_rel_l2 = 0.0_f64;
    for case in &hold_ds.cases {
        let rel_l2 = evaluate_wave_case(case)?;
        if !rel_l2.is_finite() {
            return Err(EvalError::NumericsFailed(format!(
                "non-finite relative L2 error for holdout case '{}'",
                case.case_id
            )));
        }
        if rel_l2 > max_rel_l2 {
            max_rel_l2 = rel_l2;
        }
    }
    let cases_evaluated = hold_ds.cases.len();

    let passed = max_rel_l2 <= hold_entry.max_relative_l2_error;
    let prev_chain_hash = ledger
        .entries
        .last()
        .map(|e| e.record_chain_hash.clone())
        .unwrap_or_else(|| "GENESIS".to_owned());
    let consumption_token = compute_consumption_token(
        &manifest.pre_registration_id,
        &hold_entry.holdout_id,
        &hold_sha,
        &hold_entry.pre_registered_lock_commitment,
        max_rel_l2,
        passed,
    );
    let record_chain_hash =
        compute_record_chain_hash(&prev_chain_hash, &consumption_token, &hold_sha);

    // Record consumption BEFORE checking accuracy threshold so failed evaluations still burn lock
    ledger.entries.push(ConsumedHoldoutRecord {
        holdout_id: hold_entry.holdout_id.clone(),
        payload_sha256: hold_sha.clone(),
        lock_commitment: hold_entry.pre_registered_lock_commitment.clone(),
        relative_l2_error: max_rel_l2,
        passed,
        consumption_token: consumption_token.clone(),
        prev_chain_hash,
        record_chain_hash,
    });
    persist_ledger_atomically(ledger_path, &ledger)?;
    drop(ledger_dir);

    if !passed {
        return Err(EvalError::HoldoutAccuracyFailed {
            holdout_id: hold_entry.holdout_id.clone(),
            measured: max_rel_l2,
            threshold: hold_entry.max_relative_l2_error,
        });
    }

    Ok(HoldoutEvalOutcome {
        holdout_id: hold_entry.holdout_id.clone(),
        payload_sha256: hold_sha,
        lock_commitment: hold_entry.pre_registered_lock_commitment.clone(),
        consumption_token,
        cases_evaluated,
        max_rel_l2_error: max_rel_l2,
        max_allowed_rel_l2_error: hold_entry.max_relative_l2_error,
        zero_calibration_leakage: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::fs::symlink;
    use tokspur_sys::random_bytes;

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let mut rnd = [0u8; 8];
            random_bytes(&mut rnd);
            let dir = std::env::temp_dir().join(format!("tokspur_eval_{prefix}_{}", hex(&rnd)));
            fs::create_dir_all(&dir).unwrap();
            Self { path: dir }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ignored = fs::remove_dir_all(&self.path);
        }
    }

    fn exact_wave_reference(
        length: f64,
        c: f64,
        nx: usize,
        nt: usize,
        cfl: f64,
        modes: &[WaveModeSpec],
    ) -> Vec<f64> {
        let case = WaveCase {
            case_id: "gen".to_owned(),
            wave_speed_c: c,
            domain_length_l: length,
            nx,
            nt,
            cfl,
            modes: modes.to_vec(),
            reference_u_final: vec![1.0; nx + 1],
        };
        let dx = length / (nx as f64);
        let mut dt = cfl * dx / c;
        while c * dt / dx > 1.0 && dt > 0.0 {
            dt = dt.next_down();
        }
        let mut u_prev = vec![0.0; nx + 1];
        for (j, val) in u_prev.iter_mut().enumerate().take(nx).skip(1) {
            let x = (j as f64) * dx;
            let mut s = 0.0;
            for m in modes {
                s += m.amplitude * ((m.mode_k as f64) * PI * x / length).sin();
            }
            *val = s;
        }
        let r = c * dt / dx;
        let r2 = r * r;
        let c0 = 2.0 - 2.0 * r2;
        let mut u_curr = vec![0.0; nx + 1];
        for j in 1..nx {
            u_curr[j] = u_prev[j] + 0.5 * r2 * (u_prev[j + 1] - 2.0 * u_prev[j] + u_prev[j - 1]);
        }
        let mut u_next = vec![0.0; nx + 1];
        for _ in 1..nt {
            for j in 1..nx {
                u_next[j] = c0 * u_curr[j] + r2 * (u_curr[j - 1] + u_curr[j + 1]) - u_prev[j];
            }
            std::mem::swap(&mut u_prev, &mut u_curr);
            std::mem::swap(&mut u_curr, &mut u_next);
        }
        let _ = case;
        u_curr
    }

    fn write_synthetic_manifest(dir: &Path, max_rel_err: f64) {
        let cal_modes = vec![WaveModeSpec {
            mode_k: 1,
            amplitude: 1.0,
        }];
        let cal_ref = exact_wave_reference(1.0, 1.0, 16, 8, 0.5, &cal_modes);
        let cal_ds = WaveSplitDataset {
            split_id: "cal_split_0".to_owned(),
            role: SplitRole::Calibration,
            domain: "wave1d".to_owned(),
            cases: vec![WaveCase {
                case_id: "cal_case_0".to_owned(),
                wave_speed_c: 1.0,
                domain_length_l: 1.0,
                nx: 16,
                nt: 8,
                cfl: 0.5,
                modes: cal_modes,
                reference_u_final: cal_ref,
            }],
        };
        let cal_bytes = cal_ds.to_pretty_json_bytes().unwrap();
        let cal_sha = sha256_hex(&cal_bytes);
        fs::write(dir.join("cal.json"), &cal_bytes).unwrap();

        let hold_modes = vec![
            WaveModeSpec {
                mode_k: 2,
                amplitude: 0.8,
            },
            WaveModeSpec {
                mode_k: 3,
                amplitude: 0.3,
            },
        ];
        let hold_ref = exact_wave_reference(1.0, 1.2, 20, 10, 0.6, &hold_modes);
        let hold_ds = WaveSplitDataset {
            split_id: "holdout_0".to_owned(),
            role: SplitRole::Holdout,
            domain: "wave1d".to_owned(),
            cases: vec![WaveCase {
                case_id: "hold_case_0".to_owned(),
                wave_speed_c: 1.2,
                domain_length_l: 1.0,
                nx: 20,
                nt: 10,
                cfl: 0.6,
                modes: hold_modes,
                reference_u_final: hold_ref,
            }],
        };
        let hold_bytes = hold_ds.to_pretty_json_bytes().unwrap();
        let hold_sha = sha256_hex(&hold_bytes);
        fs::write(dir.join("holdout.json"), &hold_bytes).unwrap();

        let pre_reg = "prereg-tokspur-test-v1";
        let commitment = compute_lock_commitment(pre_reg, "holdout_0", &hold_sha, max_rel_err);
        let manifest = VvManifest {
            manifest_version: "1.0.0".to_owned(),
            schema_id: "track4.empirical_vv_manifest.v1".to_owned(),
            pre_registration_id: pre_reg.to_owned(),
            created_utc: "2026-11-21T19:00:00Z".to_owned(),
            epistemic_header: EPISTEMIC_VERIFICATION_HEADER.to_owned(),
            split_provenance: SplitProvenance {
                partition_strategy: "stratified".to_owned(),
                calibration_regime: "cal".to_owned(),
                holdout_regime: "holdout".to_owned(),
                zero_leakage_policy: "sha256-disjoint".to_owned(),
            },
            calibration_splits: vec![CalibrationSplitEntry {
                split_id: "cal_split_0".to_owned(),
                relative_path: "cal.json".to_owned(),
                expected_sha256: cal_sha,
            }],
            holdout_splits: vec![HoldoutSplitEntry {
                holdout_id: "holdout_0".to_owned(),
                relative_path: "holdout.json".to_owned(),
                expected_sha256: hold_sha,
                max_relative_l2_error: max_rel_err,
                single_use: true,
                pre_registered_lock_commitment: commitment,
            }],
        };
        fs::write(
            dir.join("manifest.json"),
            manifest.to_pretty_json_bytes().unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn concurrent_unlocks_consume_one_record() {
        let dir = TestDir::new("concurrent");
        write_synthetic_manifest(dir.path(), 0.03);
        let ledger = dir.path().join("ledger.json");
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let tasks: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger)
                    })
                })
                .collect();
            let mut successes = 0;
            for task in tasks {
                match task.join().unwrap() {
                    Ok(_) => successes += 1,
                    Err(EvalError::HoldoutAlreadyConsumed { .. }) => {}
                    Err(error) => panic!("{error}"),
                }
            }
            assert_eq!(successes, 1);
        });
        let stored = load_and_verify_ledger(&ledger, "prereg-tokspur-test-v1").unwrap();
        assert_eq!(stored.entries.len(), 1);
        let threshold = 0.03_f64;
        assert_ne!(
            compute_lock_commitment("p", "h", "digest", threshold),
            compute_lock_commitment("p", "h", "digest", threshold.next_up())
        );
    }

    #[test]
    fn holdout_single_use_succeeds_once_and_refuses_replay_with_0600_ledger() {
        let dir = TestDir::new("single_use");
        write_synthetic_manifest(dir.path(), 0.03);
        let ledger_path = dir.path().join("ledger.json");

        let outcome = unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger_path).unwrap();
        assert!(outcome.zero_calibration_leakage);
        assert_eq!(outcome.cases_evaluated, 1);
        assert!(outcome.max_rel_l2_error <= 0.03);

        let mode = fs::metadata(&ledger_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);

        let ledger = load_and_verify_ledger(&ledger_path, "prereg-tokspur-test-v1").unwrap();
        assert_eq!(ledger.entries.len(), 1);
        assert_eq!(ledger.entries[0].prev_chain_hash, "GENESIS");
        assert!(ledger.entries[0].passed);

        // Replay by same holdout_id must fail
        let second = unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger_path);
        assert!(matches!(
            second,
            Err(EvalError::HoldoutAlreadyConsumed { .. })
        ));

        // Cloned payload under new holdout_id must still fail via payload_sha256 match
        let raw_manifest = fs::read(dir.path().join("manifest.json")).unwrap();
        let mut manifest = VvManifest::from_json_bytes(&raw_manifest).unwrap();
        let sha = manifest.holdout_splits[0].expected_sha256.clone();
        manifest.holdout_splits[0].holdout_id = "holdout_clone".to_owned();
        manifest.holdout_splits[0].pre_registered_lock_commitment = compute_lock_commitment(
            &manifest.pre_registration_id,
            "holdout_clone",
            &sha,
            manifest.holdout_splits[0].max_relative_l2_error,
        );
        let mut hold_ds =
            WaveSplitDataset::from_json_bytes(&fs::read(dir.path().join("holdout.json")).unwrap())
                .unwrap();
        hold_ds.split_id = "holdout_clone".to_owned();
        fs::write(
            dir.path().join("manifest.json"),
            manifest.to_pretty_json_bytes().unwrap(),
        )
        .unwrap();
        let cloned = unlock_and_evaluate_holdout(dir.path(), "holdout_clone", &ledger_path);
        assert!(matches!(
            cloned,
            Err(EvalError::HoldoutAlreadyConsumed { .. })
        ));
    }

    #[test]
    fn holdout_consume_before_threshold_burns_lock_and_numerics_failed_does_not() {
        let dir = TestDir::new("consume");
        write_synthetic_manifest(dir.path(), 0.01);
        let ledger_path = dir.path().join("ledger.json");

        // Invert sign of reference_u_final so relative L2 ~ 2.0 > 0.01
        let mut hold_ds =
            WaveSplitDataset::from_json_bytes(&fs::read(dir.path().join("holdout.json")).unwrap())
                .unwrap();
        for v in &mut hold_ds.cases[0].reference_u_final {
            *v = -*v;
        }
        let new_bytes = hold_ds.to_pretty_json_bytes().unwrap();
        let new_sha = sha256_hex(&new_bytes);
        fs::write(dir.path().join("holdout.json"), &new_bytes).unwrap();

        let mut manifest =
            VvManifest::from_json_bytes(&fs::read(dir.path().join("manifest.json")).unwrap())
                .unwrap();
        manifest.holdout_splits[0].expected_sha256 = new_sha.clone();
        manifest.holdout_splits[0].pre_registered_lock_commitment =
            compute_lock_commitment(&manifest.pre_registration_id, "holdout_0", &new_sha, 0.01);
        fs::write(
            dir.path().join("manifest.json"),
            manifest.to_pretty_json_bytes().unwrap(),
        )
        .unwrap();

        let err = unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger_path);
        assert!(matches!(err, Err(EvalError::HoldoutAccuracyFailed { .. })));

        // Lock was still burned atomically!
        let ledger = load_and_verify_ledger(&ledger_path, "prereg-tokspur-test-v1").unwrap();
        assert_eq!(ledger.entries.len(), 1);
        assert!(!ledger.entries[0].passed);
        let retry = unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger_path);
        assert!(matches!(
            retry,
            Err(EvalError::HoldoutAlreadyConsumed { .. })
        ));

        // NumericsFailed (e.g. CFL > 1.0) must NOT write a ledger entry
        let dir_num = TestDir::new("numerics");
        write_synthetic_manifest(dir_num.path(), 0.02);
        let mut bad_ds = WaveSplitDataset::from_json_bytes(
            &fs::read(dir_num.path().join("holdout.json")).unwrap(),
        )
        .unwrap();
        bad_ds.cases[0].cfl = 1.5;
        let bad_bytes = bad_ds.to_pretty_json_bytes().unwrap();
        let bad_sha = sha256_hex(&bad_bytes);
        fs::write(dir_num.path().join("holdout.json"), &bad_bytes).unwrap();
        let mut bad_m =
            VvManifest::from_json_bytes(&fs::read(dir_num.path().join("manifest.json")).unwrap())
                .unwrap();
        bad_m.holdout_splits[0].expected_sha256 = bad_sha.clone();
        bad_m.holdout_splits[0].pre_registered_lock_commitment =
            compute_lock_commitment(&bad_m.pre_registration_id, "holdout_0", &bad_sha, 0.02);
        fs::write(
            dir_num.path().join("manifest.json"),
            bad_m.to_pretty_json_bytes().unwrap(),
        )
        .unwrap();
        let num_ledger = dir_num.path().join("ledger.json");
        let num_err = unlock_and_evaluate_holdout(dir_num.path(), "holdout_0", &num_ledger);
        assert!(matches!(num_err, Err(EvalError::NumericsFailed(_))));
        assert!(!num_ledger.exists());
    }

    #[test]
    fn holdout_leakage_detected_for_sha256_and_coalesced_fingerprint() {
        let dir = TestDir::new("leakage");
        write_synthetic_manifest(dir.path(), 0.03);
        let ledger_path = dir.path().join("ledger.json");

        let mut cal_ds =
            WaveSplitDataset::from_json_bytes(&fs::read(dir.path().join("cal.json")).unwrap())
                .unwrap();
        let hold_ds =
            WaveSplitDataset::from_json_bytes(&fs::read(dir.path().join("holdout.json")).unwrap())
                .unwrap();

        // Inject disguised clone with split modes summing to the same amplitude & -0.0
        let mut leaked_case = hold_ds.cases[0].clone();
        leaked_case.case_id = "disguised_clone".to_owned();
        leaked_case.modes = vec![
            WaveModeSpec {
                mode_k: 2,
                amplitude: 0.5,
            },
            WaveModeSpec {
                mode_k: 2,
                amplitude: 0.3,
            },
            WaveModeSpec {
                mode_k: 3,
                amplitude: 0.3,
            },
            WaveModeSpec {
                mode_k: 9,
                amplitude: -0.0,
            },
        ];
        cal_ds.cases.push(leaked_case);
        let new_cal_bytes = cal_ds.to_pretty_json_bytes().unwrap();
        let new_cal_sha = sha256_hex(&new_cal_bytes);
        fs::write(dir.path().join("cal.json"), &new_cal_bytes).unwrap();

        let mut manifest =
            VvManifest::from_json_bytes(&fs::read(dir.path().join("manifest.json")).unwrap())
                .unwrap();
        manifest.calibration_splits[0].expected_sha256 = new_cal_sha;
        fs::write(
            dir.path().join("manifest.json"),
            manifest.to_pretty_json_bytes().unwrap(),
        )
        .unwrap();

        let err = unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger_path);
        assert!(matches!(err, Err(EvalError::CalibrationLeakageDetected(_))));
    }

    #[test]
    fn holdout_traversal_symlink_and_overwrite_rejected() {
        let dir = TestDir::new("traversal");
        let sub = dir.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        write_synthetic_manifest(&sub, 0.03);

        // Parent traversal rejected
        assert!(resolve_safe_relative_path(&sub, "../escape.json").is_err());
        assert!(resolve_safe_relative_path(&sub, "/etc/passwd").is_err());

        // Symlink escaping base_dir rejected
        let outside = dir.path().join("outside.json");
        fs::write(&outside, b"{}").unwrap();
        symlink(&outside, sub.join("link.json")).unwrap();
        assert!(resolve_safe_relative_path(&sub, "link.json").is_err());

        // Overwriting manifest via sub/../manifest.json rejected
        let alias = sub.join("nested/../manifest.json");
        fs::create_dir_all(sub.join("nested")).unwrap();
        let err = unlock_and_evaluate_holdout(&sub, "holdout_0", &alias);
        assert!(matches!(err, Err(EvalError::LedgerTampered(_))));
    }

    #[test]
    fn ledger_tamper_1ulp_and_unknown_json_fields_detected() {
        let dir = TestDir::new("tamper");
        write_synthetic_manifest(dir.path(), 0.03);
        let ledger_path = dir.path().join("ledger.json");

        unlock_and_evaluate_holdout(dir.path(), "holdout_0", &ledger_path).unwrap();

        // Unknown field in manifest must be rejected
        let raw_m = fs::read_to_string(dir.path().join("manifest.json")).unwrap();
        let tampered_m = raw_m.replacen("{", "{\"extra_unknown\": 1,", 1);
        fs::write(dir.path().join("manifest.json"), &tampered_m).unwrap();
        assert!(load_and_validate_manifest(dir.path()).is_err());

        // 1e-16 sub-ULP tamper in ledger relative_l2_error must be caught
        let mut ledger = load_and_verify_ledger(&ledger_path, "prereg-tokspur-test-v1").unwrap();
        ledger.entries[0].relative_l2_error += 1e-16;
        fs::write(&ledger_path, ledger.to_pretty_json_bytes().unwrap()).unwrap();
        let tamper_err = load_and_verify_ledger(&ledger_path, "prereg-tokspur-test-v1");
        assert!(matches!(tamper_err, Err(EvalError::LedgerTampered(_))));
    }
}

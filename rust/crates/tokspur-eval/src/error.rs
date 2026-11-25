use std::fmt;

use tokspur_json::JsonError;

pub enum EvalError {
    Io {
        context: String,
        source: std::io::Error,
    },
    Json {
        context: &'static str,
        source: JsonError,
    },
    Validation(String),
    ManifestIntegrityError(String),
    CalibrationHashMismatch {
        split_id: String,
        expected: String,
        actual: String,
    },
    HoldoutHashMismatch {
        holdout_id: String,
        expected: String,
        actual: String,
    },
    HoldoutAlreadyConsumed {
        holdout_id: String,
        payload_sha256: String,
        consumption_token: String,
    },
    CalibrationLeakageDetected(String),
    LedgerTampered(String),
    HoldoutAccuracyFailed {
        holdout_id: String,
        measured: f64,
        threshold: f64,
    },
    NumericsFailed(String),
    StatError(String),
    CacheError(String),
    Usage(&'static str),
}

impl fmt::Debug for EvalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { .. } => formatter.write_str("EvalError::Io([redacted])"),
            Self::Json { context, source } => formatter
                .debug_struct("EvalError::Json")
                .field("context", context)
                .field("source", source)
                .finish(),
            Self::Validation(_) => formatter.write_str("EvalError::Validation([redacted])"),
            Self::ManifestIntegrityError(_) => {
                formatter.write_str("EvalError::ManifestIntegrityError([redacted])")
            }
            Self::CalibrationHashMismatch { split_id, .. } => formatter
                .debug_struct("EvalError::CalibrationHashMismatch")
                .field("split_id", split_id)
                .finish(),
            Self::HoldoutHashMismatch { holdout_id, .. } => formatter
                .debug_struct("EvalError::HoldoutHashMismatch")
                .field("holdout_id", holdout_id)
                .finish(),
            Self::HoldoutAlreadyConsumed { holdout_id, .. } => formatter
                .debug_struct("EvalError::HoldoutAlreadyConsumed")
                .field("holdout_id", holdout_id)
                .finish(),
            Self::CalibrationLeakageDetected(_) => {
                formatter.write_str("EvalError::CalibrationLeakageDetected([redacted])")
            }
            Self::LedgerTampered(_) => formatter.write_str("EvalError::LedgerTampered([redacted])"),
            Self::HoldoutAccuracyFailed {
                holdout_id,
                measured,
                threshold,
            } => formatter
                .debug_struct("EvalError::HoldoutAccuracyFailed")
                .field("holdout_id", holdout_id)
                .field("measured", measured)
                .field("threshold", threshold)
                .finish(),
            Self::NumericsFailed(_) => formatter.write_str("EvalError::NumericsFailed([redacted])"),
            Self::StatError(_) => formatter.write_str("EvalError::StatError([redacted])"),
            Self::CacheError(_) => formatter.write_str("EvalError::CacheError([redacted])"),
            Self::Usage(msg) => formatter
                .debug_tuple("EvalError::Usage")
                .field(msg)
                .finish(),
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(formatter, "{context}: {source}"),
            Self::Json { context, source } => write!(formatter, "{context}: {source}"),
            Self::Validation(msg)
            | Self::ManifestIntegrityError(msg)
            | Self::CalibrationLeakageDetected(msg)
            | Self::LedgerTampered(msg)
            | Self::NumericsFailed(msg)
            | Self::StatError(msg)
            | Self::CacheError(msg) => formatter.write_str(msg),
            Self::CalibrationHashMismatch {
                split_id,
                expected,
                actual,
            } => write!(
                formatter,
                "calibration split '{split_id}' SHA-256 mismatch: expected {expected}, got {actual}"
            ),
            Self::HoldoutHashMismatch {
                holdout_id,
                expected,
                actual,
            } => write!(
                formatter,
                "holdout split '{holdout_id}' SHA-256 mismatch: expected {expected}, got {actual}"
            ),
            Self::HoldoutAlreadyConsumed {
                holdout_id,
                payload_sha256,
                consumption_token,
            } => write!(
                formatter,
                "holdout '{holdout_id}' (sha256={payload_sha256}) already consumed (token={consumption_token})"
            ),
            Self::HoldoutAccuracyFailed {
                holdout_id,
                measured,
                threshold,
            } => write!(
                formatter,
                "holdout '{holdout_id}' accuracy failed: measured {measured:.6e} > threshold {threshold:.6e}"
            ),
            Self::Usage(msg) => formatter.write_str(msg),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Validation(_)
            | Self::ManifestIntegrityError(_)
            | Self::CalibrationHashMismatch { .. }
            | Self::HoldoutHashMismatch { .. }
            | Self::HoldoutAlreadyConsumed { .. }
            | Self::CalibrationLeakageDetected(_)
            | Self::LedgerTampered(_)
            | Self::HoldoutAccuracyFailed { .. }
            | Self::NumericsFailed(_)
            | Self::StatError(_)
            | Self::CacheError(_)
            | Self::Usage(_) => None,
        }
    }
}

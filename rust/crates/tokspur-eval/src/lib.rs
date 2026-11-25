//! R5 evaluation, holdout manifest enforcement, sequential futility stopping, and receipt cache tooling.

#![forbid(unsafe_code)]

pub mod ci_tests;
pub mod error;
pub mod futility;
pub mod holdout;
pub mod json_util;
pub mod math;
pub mod receipt_cache;
pub mod transfer;

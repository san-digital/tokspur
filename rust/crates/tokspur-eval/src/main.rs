//! `tokspur-eval`: CLI for R5 holdout manifest enforcement and sequential 2-block futility stopping.

#![forbid(unsafe_code)]

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use tokspur_eval::error::EvalError;
use tokspur_eval::futility::{
    ContinueReason, FutilityConfig, FutilityVerdict, MetricDirection, PairedBlock, SweepCandidate,
    evaluate_sequential_futility,
};
use tokspur_eval::holdout::unlock_and_evaluate_holdout;
use tokspur_eval::json_util::{
    as_f64, as_str, as_usize, format_pretty_sorted_json, json_f64, json_usize, require_object_keys,
    required_field,
};
use tokspur_json::{Json, JsonLimits, parse_json_limited};

fn parse_holdout_args(args: &[String]) -> Result<Json, EvalError> {
    let [manifest_dir, holdout_id, ledger_path] = args else {
        return Err(EvalError::Usage(
            "usage: tokspur-eval holdout-check <manifest_dir> <holdout_id> <ledger_path>",
        ));
    };
    let outcome =
        unlock_and_evaluate_holdout(Path::new(manifest_dir), holdout_id, Path::new(ledger_path))?;

    Ok(Json::new_object([
        ("holdout_id", Json::from(outcome.holdout_id)),
        ("payload_sha256", Json::from(outcome.payload_sha256)),
        ("lock_commitment", Json::from(outcome.lock_commitment)),
        ("consumption_token", Json::from(outcome.consumption_token)),
        ("cases_evaluated", json_usize(outcome.cases_evaluated)),
        ("max_rel_l2_error", json_f64(outcome.max_rel_l2_error)?),
        (
            "max_allowed_rel_l2_error",
            json_f64(outcome.max_allowed_rel_l2_error)?,
        ),
        (
            "zero_calibration_leakage",
            Json::Bool(outcome.zero_calibration_leakage),
        ),
    ]))
}

fn parse_futility_args(args: &[String]) -> Result<Json, EvalError> {
    let [path_str] = args else {
        return Err(EvalError::Usage(
            "usage: tokspur-eval futility-eval <sweep.json>",
        ));
    };
    let raw = std::fs::read(path_str).map_err(|source| EvalError::Io {
        context: "cannot read sweep input JSON".to_owned(),
        source,
    })?;
    let root =
        parse_json_limited(&raw, JsonLimits::default()).map_err(|source| EvalError::Json {
            context: "failed parsing sweep input JSON",
            source,
        })?;
    let fields = root.object().map_err(|source| EvalError::Json {
        context: "sweep input must be JSON object",
        source,
    })?;
    require_object_keys(
        fields,
        &[
            b"min_blocks",
            b"fdr_alpha",
            b"min_worthwhile_gain_pct",
            b"candidates",
        ],
    )?;

    let mut cfg = FutilityConfig::default();
    if let Some(v) = root.find("min_blocks").map_err(|source| EvalError::Json {
        context: "lookup min_blocks",
        source,
    })? {
        cfg.min_blocks = as_usize(v)?;
    }
    if let Some(v) = root.find("fdr_alpha").map_err(|source| EvalError::Json {
        context: "lookup fdr_alpha",
        source,
    })? {
        cfg.fdr_alpha = as_f64(v)?;
    }
    if let Some(v) = root
        .find("min_worthwhile_gain_pct")
        .map_err(|source| EvalError::Json {
            context: "lookup min_worthwhile_gain_pct",
            source,
        })?
    {
        cfg.min_worthwhile_gain_pct = as_f64(v)?;
    }

    let cands_arr = required_field(fields, b"candidates")?
        .array()
        .map_err(|source| EvalError::Json {
            context: "candidates must be array",
            source,
        })?;
    let mut candidates = Vec::with_capacity(cands_arr.len());
    for c_val in cands_arr {
        let cf = c_val.object().map_err(|source| EvalError::Json {
            context: "candidate entry must be object",
            source,
        })?;
        require_object_keys(cf, &[b"candidate_id", b"direction", b"blocks"])?;
        let candidate_id = as_str(required_field(cf, b"candidate_id")?)?.to_owned();
        let dir_str = as_str(required_field(cf, b"direction")?)?;
        let direction = match dir_str {
            "lower_is_better" => MetricDirection::LowerIsBetter,
            "higher_is_better" => MetricDirection::HigherIsBetter,
            other => {
                return Err(EvalError::Validation(format!(
                    "invalid direction '{other}'; expected lower_is_better or higher_is_better"
                )));
            }
        };
        let blocks_arr =
            required_field(cf, b"blocks")?
                .array()
                .map_err(|source| EvalError::Json {
                    context: "blocks must be array",
                    source,
                })?;
        let mut blocks = Vec::with_capacity(blocks_arr.len());
        for b_val in blocks_arr {
            let bf = b_val.object().map_err(|source| EvalError::Json {
                context: "block must be object",
                source,
            })?;
            require_object_keys(bf, &[b"block", b"baseline", b"candidate"])?;
            let block_index = as_usize(required_field(bf, b"block")?)?;
            let base_arr =
                required_field(bf, b"baseline")?
                    .array()
                    .map_err(|source| EvalError::Json {
                        context: "baseline must be array",
                        source,
                    })?;
            let cand_arr = required_field(bf, b"candidate")?
                .array()
                .map_err(|source| EvalError::Json {
                    context: "candidate must be array",
                    source,
                })?;
            let mut baseline = Vec::with_capacity(base_arr.len());
            for x in base_arr {
                baseline.push(as_f64(x)?);
            }
            let mut candidate = Vec::with_capacity(cand_arr.len());
            for x in cand_arr {
                candidate.push(as_f64(x)?);
            }
            blocks.push(PairedBlock {
                block_index,
                baseline,
                candidate,
            });
        }
        candidates.push(SweepCandidate {
            candidate_id,
            direction,
            blocks,
        });
    }

    let report = evaluate_sequential_futility(&candidates, cfg)?;
    let mut cand_json = Vec::with_capacity(report.candidates.len());
    for c in &report.candidates {
        let verdict_str = match c.verdict {
            FutilityVerdict::StopFutile => "stop_futile",
            FutilityVerdict::Continue => "continue",
            FutilityVerdict::Winning => "winning",
        };
        let reason_json = match c.continue_reason {
            Some(ContinueReason::InsufficientBlocks) => Json::from("insufficient_blocks"),
            Some(ContinueReason::ExactTie) => Json::from("exact_tie"),
            Some(ContinueReason::PositiveOrBorderlineTrend) => {
                Json::from("positive_or_borderline_trend")
            }
            Some(ContinueReason::NonSignificantRegression) => {
                Json::from("non_significant_regression")
            }
            None => Json::Null,
        };
        cand_json.push(Json::new_object([
            ("candidate_id", Json::from(c.candidate_id.as_str())),
            ("blocks_observed", json_usize(c.blocks_observed)),
            ("total_pairs", json_usize(c.total_pairs)),
            ("mean_gain", json_f64(c.mean_gain)?),
            ("relative_gain_pct", json_f64(c.relative_gain_pct)?),
            ("n_effective", json_usize(c.wilcoxon.n_effective)),
            ("w_pos", json_f64(c.wilcoxon.w_pos)?),
            ("w_neg", json_f64(c.wilcoxon.w_neg)?),
            ("w_min", json_f64(c.wilcoxon.w_min)?),
            ("p_two_sided", json_f64(c.wilcoxon.p_two_sided)?),
            ("p_harm_one_sided", json_f64(c.wilcoxon.p_harm_one_sided)?),
            ("p_futility", json_f64(c.p_futility)?),
            ("q_futility", json_f64(c.q_futility)?),
            ("verdict", Json::from(verdict_str)),
            ("continue_reason", reason_json),
        ]));
    }

    Ok(Json::new_object([
        ("any_stopped", Json::Bool(report.any_stopped)),
        ("all_stopped", Json::Bool(report.all_stopped)),
        ("candidates", Json::Array(cand_json)),
    ]))
}

fn run(args: &[String]) -> Result<Json, EvalError> {
    match args.split_first() {
        Some((cmd, rest)) if cmd == "holdout-check" => parse_holdout_args(rest),
        Some((cmd, rest)) if cmd == "futility-eval" => parse_futility_args(rest),
        _ => Err(EvalError::Usage(
            "usage: tokspur-eval <holdout-check|futility-eval> ...",
        )),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args).and_then(|json| {
        let bytes = format_pretty_sorted_json(&json)?;
        std::io::stdout()
            .write_all(&bytes)
            .map_err(|source| EvalError::Io {
                context: "failed writing stdout".to_owned(),
                source,
            })
    }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let line = format!("tokspur-eval: {err}\n");
            if std::io::stderr().write_all(line.as_bytes()).is_err() {
                return ExitCode::from(2);
            }
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn eval_error_debug_redacts_paths() {
        let err = parse_futility_args(&["/nonexistent/secret-sweep-dir/sweep.json".to_owned()])
            .unwrap_err();
        assert!(err.source().is_some());
        let dbg = format!("{err:?}");
        assert!(!dbg.contains("secret-sweep-dir"));
    }
}

//! Paired A/B crossover blinding, balanced arm assignment, and Wilcoxon signed-rank evaluation.

use tokspur_text::sha256::sha256;

use crate::error::EvalError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferArm {
    ControlFirst,
    CandidateFirst,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedTransferObservation {
    pub subject_id: String,
    pub concept_id: String,
    pub raw_score: f64,
    pub exposed_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferEvalSummary {
    pub protocol_status: String,
    pub live_human_subjects_run: usize,
    pub paired_observations_count: usize,
    pub control_mean: f64,
    pub treatment_mean: f64,
    pub mean_delta: f64,
    pub relative_gain_pct: f64,
    pub std_dev_diff: f64,
    pub paired_t_stat: f64,
    pub wilcoxon_w_pos: f64,
    pub wilcoxon_w_neg: f64,
    pub wilcoxon_w_min: f64,
    pub cohens_d: f64,
    pub threshold_pct: f64,
    pub passed_threshold: bool,
}

fn ensure_finite_f64(value: f64, label: &str) -> Result<f64, EvalError> {
    if !value.is_finite() {
        return Err(EvalError::StatError(format!(
            "non-finite floating-point value for {label}"
        )));
    }
    Ok(value)
}

pub fn deterministic_blinding_code(subject_id: &str) -> String {
    let mut msg = Vec::with_capacity(32 + subject_id.len());
    msg.extend_from_slice(b"TRACK3_PRE_REG_TRANSFER_V1:");
    msg.extend_from_slice(subject_id.as_bytes());
    let digest = sha256(&msg);
    format!(
        "BLIND-{:02X}{:02X}{:02X}{:02X}",
        digest[0], digest[1], digest[2], digest[3]
    )
}

pub fn deterministic_balanced_crossover_arms(num_subjects: usize) -> Vec<TransferArm> {
    if num_subjects == 0 {
        return Vec::new();
    }
    let mut arms = Vec::with_capacity(num_subjects);
    for i in 0..num_subjects {
        arms.push(if i % 2 == 0 {
            TransferArm::ControlFirst
        } else {
            TransferArm::CandidateFirst
        });
    }
    for idx in (1..num_subjects).rev() {
        let mut msg = Vec::with_capacity(64);
        msg.extend_from_slice(b"TRACK3_PRE_REG_CROSSOVER_SHUFFLE_V1:");
        msg.extend_from_slice(&num_subjects.to_le_bytes());
        msg.extend_from_slice(&idx.to_le_bytes());
        let digest = sha256(&msg);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[0..8]);
        let j = (u64::from_le_bytes(bytes) as usize) % (idx + 1);
        arms.swap(idx, j);
    }
    arms
}

/// Computes exact Wilcoxon signed-rank `(w_pos, w_neg, w_min)` using average ranks for ties
/// and excluding zero/near-zero differences (`|d| <= 1e-12`).
pub fn exact_wilcoxon_signed_rank(diffs: &[f64]) -> (f64, f64, f64) {
    let mut nonzero: Vec<(f64, f64)> = diffs
        .iter()
        .copied()
        .filter(|d| d.is_finite() && d.abs() > 1e-12)
        .map(|d| (d.abs(), d))
        .collect();
    if nonzero.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    nonzero.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut w_pos = 0.0;
    let mut w_neg = 0.0;
    let mut i = 0usize;
    while i < nonzero.len() {
        let mut j = i + 1;
        while j < nonzero.len() && (nonzero[j].0 - nonzero[i].0).abs() <= 1e-12 {
            j += 1;
        }
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        for &(_, raw_d) in &nonzero[i..j] {
            if raw_d > 0.0 {
                w_pos += avg_rank;
            } else {
                w_neg += avg_rank;
            }
        }
        i = j;
    }
    (w_pos, w_neg, w_pos.min(w_neg))
}

pub fn evaluate_paired_transfer(
    observations: &[PairedTransferObservation],
    threshold_pct: f64,
) -> Result<TransferEvalSummary, EvalError> {
    ensure_finite_f64(threshold_pct, "threshold_pct")?;
    if observations.is_empty() {
        return Ok(TransferEvalSummary {
            protocol_status: "PRE_REGISTERED_NOT_EXECUTED".to_owned(),
            live_human_subjects_run: 0,
            paired_observations_count: 0,
            control_mean: 0.0,
            treatment_mean: 0.0,
            mean_delta: 0.0,
            relative_gain_pct: 0.0,
            std_dev_diff: 0.0,
            paired_t_stat: 0.0,
            wilcoxon_w_pos: 0.0,
            wilcoxon_w_neg: 0.0,
            wilcoxon_w_min: 0.0,
            cohens_d: 0.0,
            threshold_pct,
            passed_threshold: false,
        });
    }

    let mut diffs = Vec::with_capacity(observations.len());
    let mut control_sum = 0.0;
    let mut treatment_sum = 0.0;
    for (idx, obs) in observations.iter().enumerate() {
        ensure_finite_f64(obs.raw_score, &format!("observations[{idx}].raw_score"))?;
        ensure_finite_f64(
            obs.exposed_score,
            &format!("observations[{idx}].exposed_score"),
        )?;
        control_sum += obs.raw_score;
        treatment_sum += obs.exposed_score;
        diffs.push(obs.exposed_score - obs.raw_score);
    }

    let n = observations.len() as f64;
    let control_mean = control_sum / n;
    let treatment_mean = treatment_sum / n;
    let mean_delta = treatment_mean - control_mean;
    let relative_gain_pct = if control_mean.abs() > 1e-12 {
        100.0 * mean_delta / control_mean
    } else {
        100.0 * mean_delta
    };

    let std_dev_diff = if observations.len() >= 2 {
        let var = diffs
            .iter()
            .map(|d| {
                let c = d - mean_delta;
                c * c
            })
            .sum::<f64>()
            / (n - 1.0);
        var.sqrt()
    } else {
        0.0
    };

    let (paired_t_stat, cohens_d) = if std_dev_diff > 1e-12 {
        (
            mean_delta / (std_dev_diff / n.sqrt()),
            mean_delta / std_dev_diff,
        )
    } else if mean_delta.abs() <= 1e-12 {
        (0.0, 0.0)
    } else {
        let sign = mean_delta.signum();
        (sign * 100.0, sign * 10.0)
    };

    let (wilcoxon_w_pos, wilcoxon_w_neg, wilcoxon_w_min) = exact_wilcoxon_signed_rank(&diffs);
    let passed_threshold = observations.len() >= 2 && relative_gain_pct >= threshold_pct;

    Ok(TransferEvalSummary {
        protocol_status: "STATISTICAL_EVALUATION_COMPLETE".to_owned(),
        live_human_subjects_run: 0,
        paired_observations_count: observations.len(),
        control_mean,
        treatment_mean,
        mean_delta,
        relative_gain_pct,
        std_dev_diff,
        paired_t_stat,
        wilcoxon_w_pos,
        wilcoxon_w_neg,
        wilcoxon_w_min,
        cohens_d,
        threshold_pct,
        passed_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilcoxon_signed_rank_ties_and_zero_pairs_excluded() {
        let (w_pos, w_neg, w_min) = exact_wilcoxon_signed_rank(&[1.0, 1.0, -1.0, 2.0, 3.0, 4.0]);
        assert!((w_pos + w_neg - 21.0).abs() < 1e-12);
        assert!((w_min - w_neg).abs() < 1e-12);
        assert!((w_neg - 2.0).abs() < 1e-12);

        let (wp0, wn0, wm0) = exact_wilcoxon_signed_rank(&[0.0, 1e-13, -1e-13, 2.0, 3.0, 4.0, 5.0]);
        assert!((wp0 - 10.0).abs() < 1e-12);
        assert!((wn0 - 0.0).abs() < 1e-12);
        assert!((wm0 - 0.0).abs() < 1e-12);
    }

    #[test]
    fn crossover_arms_balance_and_blinding_is_deterministic() {
        for n in [2usize, 4, 8, 10, 16] {
            let arms = deterministic_balanced_crossover_arms(n);
            let c_count = arms
                .iter()
                .filter(|&&a| a == TransferArm::ControlFirst)
                .count();
            let t_count = arms
                .iter()
                .filter(|&&a| a == TransferArm::CandidateFirst)
                .count();
            assert_eq!(c_count, n / 2);
            assert_eq!(t_count, n / 2);
        }
        assert_eq!(
            deterministic_blinding_code("SUBJ-001"),
            deterministic_blinding_code("SUBJ-001")
        );
        assert!(deterministic_blinding_code("SUBJ-001").starts_with("BLIND-"));
    }
}

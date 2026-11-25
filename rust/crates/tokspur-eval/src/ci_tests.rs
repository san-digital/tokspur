//! Benjamini-Hochberg step-up FDR control (`apply_bh_fdr` + `apply_bh_fdr_pvalues`).

use crate::error::EvalError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiTestKind {
    PearsonCorrFisherZ,
    PartialCorrFisherZ,
    DiscreteContingencyChi2,
    WilcoxonSignedRank,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CiTestOutcome {
    pub x: String,
    pub y: String,
    pub conditioning_set: Vec<String>,
    pub test_kind: CiTestKind,
    pub testable: bool,
    pub sample_size: usize,
    pub degrees_of_freedom: usize,
    pub statistic: f64,
    pub auxiliary_metric: f64,
    pub p_value: f64,
    pub q_value: f64,
    pub rejected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BhFdrSummary {
    pub total_implied_cis: usize,
    pub tested: usize,
    pub skipped_untestable: usize,
    pub rejected: usize,
    pub min_p_value: Option<f64>,
    pub min_q_value: Option<f64>,
}

/// Applies Benjamini-Hochberg step-up FDR control at `fdr_alpha in (0, 1)` across
/// all testable outcomes in-place.
pub fn apply_bh_fdr(
    results: &mut [CiTestOutcome],
    fdr_alpha: f64,
) -> Result<BhFdrSummary, EvalError> {
    if !fdr_alpha.is_finite() || fdr_alpha <= 0.0 || fdr_alpha >= 1.0 {
        return Err(EvalError::StatError(format!(
            "invalid fdr_alpha {fdr_alpha}; must lie in (0, 1)"
        )));
    }

    let mut testable_indices = Vec::new();
    for (idx, r) in results.iter_mut().enumerate() {
        if r.testable {
            if !r.p_value.is_finite() || r.p_value < 0.0 || r.p_value > 1.0 {
                return Err(EvalError::StatError(format!(
                    "invalid p-value {} in CI test ({}, {})",
                    r.p_value, r.x, r.y
                )));
            }
            testable_indices.push(idx);
        } else {
            r.q_value = 1.0;
            r.rejected = false;
        }
    }

    let total_implied_cis = results.len();
    let tested = testable_indices.len();
    let skipped_untestable = total_implied_cis - tested;

    if tested == 0 {
        return Ok(BhFdrSummary {
            total_implied_cis,
            tested: 0,
            skipped_untestable,
            rejected: 0,
            min_p_value: None,
            min_q_value: None,
        });
    }

    testable_indices.sort_by(|&ia, &ib| {
        let a = &results[ia];
        let b = &results[ib];
        a.p_value
            .total_cmp(&b.p_value)
            .then_with(|| a.x.cmp(&b.x))
            .then_with(|| a.y.cmp(&b.y))
            .then_with(|| a.conditioning_set.cmp(&b.conditioning_set))
    });

    let mut running_min = 1.0_f64;
    let m_f = tested as f64;
    for rank_rev in (0..tested).rev() {
        let idx = testable_indices[rank_rev];
        let rank_1based = (rank_rev + 1) as f64;
        let scaled = (m_f * results[idx].p_value / rank_1based).min(1.0);
        running_min = running_min.min(scaled);
        results[idx].q_value = running_min;
        results[idx].rejected = running_min <= fdr_alpha;
    }

    let rejected = testable_indices
        .iter()
        .filter(|&&idx| results[idx].rejected)
        .count();
    let min_p_value = testable_indices
        .iter()
        .map(|&idx| results[idx].p_value)
        .min_by(f64::total_cmp);
    let min_q_value = testable_indices
        .iter()
        .map(|&idx| results[idx].q_value)
        .min_by(f64::total_cmp);

    Ok(BhFdrSummary {
        total_implied_cis,
        tested,
        skipped_untestable,
        rejected,
        min_p_value,
        min_q_value,
    })
}

/// Applies Benjamini-Hochberg step-up FDR control directly over a slice of p-values.
/// Returns `(q_value, rejected)` in input order.
pub fn apply_bh_fdr_pvalues(
    p_values: &[f64],
    fdr_alpha: f64,
) -> Result<Vec<(f64, bool)>, EvalError> {
    if !fdr_alpha.is_finite() || fdr_alpha <= 0.0 || fdr_alpha >= 1.0 {
        return Err(EvalError::StatError(format!(
            "Invalid BH-FDR alpha: {fdr_alpha}"
        )));
    }
    for &p in p_values {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(EvalError::StatError(format!("Invalid p-value: {p}")));
        }
    }
    let m = p_values.len();
    if m == 0 {
        return Ok(Vec::new());
    }

    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut q_sorted = vec![0.0_f64; m];
    let mut running = 1.0_f64;
    let m_f = m as f64;
    for rev_idx in (0..m).rev() {
        let rank = (rev_idx + 1) as f64;
        let raw_q = (m_f * indexed[rev_idx].1 / rank).min(1.0);
        running = running.min(raw_q);
        q_sorted[rev_idx] = running;
    }

    let mut out = vec![(0.0_f64, false); m];
    for (pos, &(orig_idx, _)) in indexed.iter().enumerate() {
        let q = q_sorted[pos];
        out[orig_idx] = (q, q <= fdr_alpha);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_outcome(x: &str, y: &str, testable: bool, p_value: f64) -> CiTestOutcome {
        CiTestOutcome {
            x: x.to_owned(),
            y: y.to_owned(),
            conditioning_set: Vec::new(),
            test_kind: CiTestKind::WilcoxonSignedRank,
            testable,
            sample_size: 12,
            degrees_of_freedom: 1,
            statistic: 0.0,
            auxiliary_metric: 0.0,
            p_value,
            q_value: 1.0,
            rejected: false,
        }
    }

    #[test]
    fn bh_fdr_step_up_monotone_and_skips_untestable() {
        let mut outcomes = vec![
            mk_outcome("A", "B", true, 0.001),
            mk_outcome("C", "D", true, 0.024),
            mk_outcome("E", "F", true, 0.80),
            mk_outcome("G", "H", false, 0.0001),
        ];
        let summary = apply_bh_fdr(&mut outcomes, 0.05).unwrap();
        assert_eq!(summary.total_implied_cis, 4);
        assert_eq!(summary.tested, 3);
        assert_eq!(summary.skipped_untestable, 1);
        assert_eq!(summary.rejected, 2);
        assert!((outcomes[0].q_value - 0.003).abs() < 1e-12);
        assert!((outcomes[1].q_value - 0.036).abs() < 1e-12);
        assert!((outcomes[2].q_value - 0.80).abs() < 1e-12);
        assert!(!outcomes[3].rejected);
        assert!((outcomes[3].q_value - 1.0).abs() < 1e-12);

        let direct = apply_bh_fdr_pvalues(&[0.002, 0.03, 0.025, 0.80], 0.05).unwrap();
        assert!(direct[0].1);
        assert!(!direct[3].1);
        assert!(direct[2].0 <= direct[1].0 && direct[1].0 <= direct[3].0);
    }

    #[test]
    fn bh_fdr_rejects_invalid_alpha_and_extreme_or_nan_pvalues() {
        assert!(apply_bh_fdr_pvalues(&[0.05], 0.0).is_err());
        assert!(apply_bh_fdr_pvalues(&[0.05], 1.0).is_err());
        assert!(apply_bh_fdr_pvalues(&[0.05], f64::NAN).is_err());
        assert!(apply_bh_fdr_pvalues(&[f64::NAN], 0.05).is_err());
        assert!(apply_bh_fdr_pvalues(&[-0.01], 0.05).is_err());
        assert!(apply_bh_fdr_pvalues(&[1.01], 0.05).is_err());

        let extremes = apply_bh_fdr_pvalues(&[0.0, 1.0], 0.05).unwrap();
        assert_eq!(extremes, vec![(0.0, true), (1.0, false)]);
    }
}

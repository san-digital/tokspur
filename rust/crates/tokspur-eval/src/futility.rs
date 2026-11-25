//! Evaluate a planned look at paired A/B blocks. Benjamini-Hochberg adjusts across candidates within this look. Repeated looks require a separate stopping design.

use std::collections::BTreeSet;
use std::f64::consts::FRAC_1_SQRT_2;

use crate::ci_tests::apply_bh_fdr_pvalues;
use crate::error::EvalError;
use crate::math::erfc_std;
use crate::transfer::{PairedTransferObservation, evaluate_paired_transfer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedBlock {
    pub block_index: usize,
    pub baseline: Vec<f64>,
    pub candidate: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SweepCandidate {
    pub candidate_id: String,
    pub direction: MetricDirection,
    pub blocks: Vec<PairedBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FutilityConfig {
    pub min_blocks: usize,
    pub fdr_alpha: f64,
    pub min_worthwhile_gain_pct: f64,
}

impl Default for FutilityConfig {
    fn default() -> Self {
        Self {
            min_blocks: 2,
            fdr_alpha: 0.05,
            min_worthwhile_gain_pct: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilcoxonTestResult {
    pub n_effective: usize,
    pub w_pos: f64,
    pub w_neg: f64,
    pub w_min: f64,
    pub p_two_sided: f64,
    pub p_harm_one_sided: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutilityVerdict {
    StopFutile,
    Continue,
    Winning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinueReason {
    InsufficientBlocks,
    ExactTie,
    PositiveOrBorderlineTrend,
    NonSignificantRegression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFutilityOutcome {
    pub candidate_id: String,
    pub blocks_observed: usize,
    pub total_pairs: usize,
    pub mean_gain: f64,
    pub relative_gain_pct: f64,
    pub wilcoxon: WilcoxonTestResult,
    pub p_futility: f64,
    pub q_futility: f64,
    pub verdict: FutilityVerdict,
    pub continue_reason: Option<ContinueReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SweepFutilityReport {
    pub candidates: Vec<CandidateFutilityOutcome>,
    pub any_stopped: bool,
    pub all_stopped: bool,
}

/// Computes exact Wilcoxon signed-rank statistic and two-sided / one-sided harm p-values
/// over signed improvements `gains` (where `gain > 0` means candidate improves over baseline).
pub fn wilcoxon_signed_rank_test(gains: &[f64]) -> Result<WilcoxonTestResult, EvalError> {
    for (idx, &g) in gains.iter().enumerate() {
        if !g.is_finite() {
            return Err(EvalError::StatError(format!(
                "non-finite difference at index {idx}"
            )));
        }
    }

    let mut nonzero: Vec<(f64, f64)> = gains
        .iter()
        .copied()
        .filter(|&g| g != 0.0)
        .map(|g| (g.abs(), g))
        .collect();
    let n_eff = nonzero.len();
    if n_eff == 0 {
        return Ok(WilcoxonTestResult {
            n_effective: 0,
            w_pos: 0.0,
            w_neg: 0.0,
            w_min: 0.0,
            p_two_sided: 1.0,
            p_harm_one_sided: 1.0,
        });
    }

    nonzero.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut doubled_ranks: Vec<usize> = Vec::with_capacity(n_eff);
    let mut tie_group_sizes: Vec<usize> = Vec::new();
    let mut w_pos = 0.0_f64;
    let mut w_neg = 0.0_f64;
    let mut w_pos_doubled: usize = 0;
    let mut w_neg_doubled: usize = 0;

    let mut i = 0usize;
    while i < n_eff {
        let mut j = i + 1;
        while j < n_eff && nonzero[j].0 == nonzero[i].0 {
            j += 1;
        }
        let tie_count = j - i;
        tie_group_sizes.push(tie_count);
        // Doubled average rank: (i + 1) + j is always an exact positive integer.
        let doubled_avg_rank = (i + 1) + j;
        let avg_rank = doubled_avg_rank as f64 / 2.0;
        for &(_, raw_g) in &nonzero[i..j] {
            doubled_ranks.push(doubled_avg_rank);
            if raw_g > 0.0 {
                w_pos += avg_rank;
                w_pos_doubled += doubled_avg_rank;
            } else {
                w_neg += avg_rank;
                w_neg_doubled += doubled_avg_rank;
            }
        }
        i = j;
    }

    let w_min = w_pos.min(w_neg);
    let w_min_doubled = w_pos_doubled.min(w_neg_doubled);

    let (p_two_sided, p_harm_one_sided) = if n_eff <= 60 {
        let total_doubled: usize = n_eff * (n_eff + 1);
        let mut dp: Vec<u128> = vec![0; total_doubled + 1];
        dp[0] = 1;
        for d in doubled_ranks {
            for s in (d..=total_doubled).rev() {
                dp[s] = dp[s].saturating_add(dp[s - d]);
            }
        }
        let denom = (1u128 << n_eff) as f64;
        let mut extreme_two: u128 = 0;
        let mut extreme_harm: u128 = 0;
        for (s, &count) in dp.iter().enumerate() {
            if count == 0 {
                continue;
            }
            if s.min(total_doubled - s) <= w_min_doubled {
                extreme_two = extreme_two.saturating_add(count);
            }
            if s <= w_pos_doubled {
                extreme_harm = extreme_harm.saturating_add(count);
            }
        }
        (
            (extreme_two as f64 / denom).clamp(0.0, 1.0),
            (extreme_harm as f64 / denom).clamp(0.0, 1.0),
        )
    } else {
        // Tie-corrected normal approximation with continuity correction.
        let nf = n_eff as f64;
        let mean_w = nf * (nf + 1.0) / 4.0;
        let tie_correction: f64 = tie_group_sizes
            .iter()
            .map(|&t| {
                let tf = t as f64;
                tf * tf * tf - tf
            })
            .sum();
        let var_w = ((nf * (nf + 1.0) * (2.0 * nf + 1.0)) - 0.5 * tie_correction) / 24.0;
        if var_w <= 1e-18 {
            (1.0, 1.0)
        } else {
            let sd_w = var_w.sqrt();
            let z_two = ((w_pos - mean_w).abs() - 0.5).max(0.0) / sd_w;
            let p_two = erfc_std(z_two * FRAC_1_SQRT_2).clamp(0.0, 1.0);
            // One-sided harm: P(W+ <= observed w_pos) under H0.
            let z_harm = (w_pos - mean_w + 0.5) / sd_w;
            let p_harm = (0.5 * erfc_std(-z_harm * FRAC_1_SQRT_2)).clamp(0.0, 1.0);
            (p_two, p_harm)
        }
    };

    Ok(WilcoxonTestResult {
        n_effective: n_eff,
        w_pos,
        w_neg,
        w_min,
        p_two_sided,
        p_harm_one_sided,
    })
}

fn validate_config(config: &FutilityConfig) -> Result<(), EvalError> {
    if config.min_blocks == 0 {
        return Err(EvalError::StatError(
            "FutilityConfig.min_blocks must be >= 1".to_owned(),
        ));
    }
    if !config.fdr_alpha.is_finite() || config.fdr_alpha <= 0.0 || config.fdr_alpha >= 1.0 {
        return Err(EvalError::StatError(format!(
            "Invalid BH-FDR alpha: {}",
            config.fdr_alpha
        )));
    }
    if !config.min_worthwhile_gain_pct.is_finite() || config.min_worthwhile_gain_pct < 0.0 {
        return Err(EvalError::StatError(
            "min_worthwhile_gain_pct must be finite and >= 0".to_owned(),
        ));
    }
    Ok(())
}

/// Evaluates one planned look across the supplied candidates after the required number of observed blocks.
pub fn evaluate_sequential_futility(
    candidates: &[SweepCandidate],
    config: FutilityConfig,
) -> Result<SweepFutilityReport, EvalError> {
    validate_config(&config)?;
    if candidates.is_empty() {
        return Ok(SweepFutilityReport {
            candidates: Vec::new(),
            any_stopped: false,
            all_stopped: false,
        });
    }

    struct RawStats {
        blocks_observed: usize,
        total_pairs: usize,
        mean_gain: f64,
        relative_gain_pct: f64,
        wilcoxon: WilcoxonTestResult,
        p_futility: f64,
        eligible_for_stop: bool,
        positive_trend: bool,
        insufficient_blocks: bool,
        exact_tie: bool,
    }

    let mut per_candidate = Vec::with_capacity(candidates.len());
    let mut candidate_ids = BTreeSet::new();
    for cand in candidates {
        if !candidate_ids.insert(cand.candidate_id.as_str()) {
            return Err(EvalError::StatError(format!(
                "duplicate candidate '{}'",
                cand.candidate_id
            )));
        }
        let mut gains = Vec::new();
        let mut obs = Vec::new();
        let mut block_indices = BTreeSet::new();
        for blk in &cand.blocks {
            if blk.baseline.len() != blk.candidate.len() {
                return Err(EvalError::StatError(format!(
                    "candidate '{}' block {} baseline/candidate length mismatch ({} vs {})",
                    cand.candidate_id,
                    blk.block_index,
                    blk.baseline.len(),
                    blk.candidate.len()
                )));
            }
            if blk.baseline.is_empty() || !block_indices.insert(blk.block_index) {
                return Err(EvalError::StatError(format!(
                    "candidate '{}' needs distinct non-empty blocks at block {}",
                    cand.candidate_id, blk.block_index
                )));
            }
            for (idx, (&base, &cval)) in blk.baseline.iter().zip(blk.candidate.iter()).enumerate() {
                if !base.is_finite() || !cval.is_finite() {
                    return Err(EvalError::StatError(format!(
                        "non-finite value in candidate '{}' block {} pair {}",
                        cand.candidate_id, blk.block_index, idx
                    )));
                }
                let gain = match cand.direction {
                    MetricDirection::LowerIsBetter => base - cval,
                    MetricDirection::HigherIsBetter => cval - base,
                };
                gains.push(gain);
                let (raw_s, exp_s) = match cand.direction {
                    MetricDirection::LowerIsBetter => (base, base + gain),
                    MetricDirection::HigherIsBetter => (base, cval),
                };
                obs.push(PairedTransferObservation {
                    subject_id: format!("blk{}_p{}", blk.block_index, idx),
                    concept_id: cand.candidate_id.clone(),
                    raw_score: raw_s,
                    exposed_score: exp_s,
                });
            }
        }

        let summary = evaluate_paired_transfer(&obs, config.min_worthwhile_gain_pct)?;
        let wilcoxon = wilcoxon_signed_rank_test(&gains)?;
        let blocks_observed = cand.blocks.len();
        let total_pairs = gains.len();
        let base_mean = summary.control_mean;
        let rel_gain_pct = if total_pairs == 0 {
            0.0
        } else if base_mean.abs() > 1e-12 {
            100.0 * summary.mean_delta / base_mean
        } else {
            100.0 * summary.mean_delta
        };

        let insufficient_blocks = blocks_observed < config.min_blocks || total_pairs == 0;
        let exact_tie = !insufficient_blocks && wilcoxon.n_effective == 0;
        let positive_trend = !insufficient_blocks
            && !exact_tie
            && (wilcoxon.w_pos >= wilcoxon.w_neg || summary.mean_delta > 0.0);

        let mut eligible_for_stop = false;
        let mut p_futility = 1.0;

        if !insufficient_blocks && !exact_tie && wilcoxon.w_neg > wilcoxon.w_pos {
            eligible_for_stop = true;
            p_futility = wilcoxon.p_two_sided;
        } else if !insufficient_blocks
            && !exact_tie
            && config.min_worthwhile_gain_pct > 0.0
            && wilcoxon.w_pos <= wilcoxon.w_neg
            && total_pairs >= 2
        {
            // Optional non-inferiority margin futility when candidate cannot reach target gain.
            let se_pct = if base_mean.abs() > 1e-12 {
                100.0 * (summary.std_dev_diff / (total_pairs as f64).sqrt()) / base_mean.abs()
            } else {
                100.0 * (summary.std_dev_diff / (total_pairs as f64).sqrt())
            };
            if se_pct > 1e-12 {
                let z = (rel_gain_pct - config.min_worthwhile_gain_pct) / se_pct;
                if z < 0.0 {
                    eligible_for_stop = true;
                    p_futility = (0.5 * erfc_std(-z * FRAC_1_SQRT_2)).clamp(0.0, 1.0);
                }
            } else if rel_gain_pct < config.min_worthwhile_gain_pct {
                eligible_for_stop = true;
                p_futility = 0.0;
            }
        }

        per_candidate.push(RawStats {
            blocks_observed,
            total_pairs,
            mean_gain: summary.mean_delta,
            relative_gain_pct: rel_gain_pct,
            wilcoxon,
            p_futility,
            eligible_for_stop,
            positive_trend,
            insufficient_blocks,
            exact_tie,
        });
    }

    let p_vec: Vec<f64> = per_candidate.iter().map(|c| c.p_futility).collect();
    let bh = apply_bh_fdr_pvalues(&p_vec, config.fdr_alpha)?;
    let improvement_p: Vec<_> = per_candidate
        .iter()
        .map(|c| {
            if c.insufficient_blocks || c.wilcoxon.w_pos <= c.wilcoxon.w_neg {
                1.0
            } else {
                c.wilcoxon.p_two_sided
            }
        })
        .collect();
    let improvement = apply_bh_fdr_pvalues(&improvement_p, config.fdr_alpha)?;

    let mut outcomes = Vec::with_capacity(candidates.len());
    let mut any_stopped = false;
    let mut all_stopped = true;
    for (index, (raw, (q_val, rejected))) in per_candidate.into_iter().zip(bh).enumerate() {
        let cand = &candidates[index];
        let (verdict, continue_reason) = if raw.insufficient_blocks {
            (
                FutilityVerdict::Continue,
                Some(ContinueReason::InsufficientBlocks),
            )
        } else if raw.exact_tie {
            (FutilityVerdict::Continue, Some(ContinueReason::ExactTie))
        } else if raw.eligible_for_stop && rejected {
            any_stopped = true;
            (FutilityVerdict::StopFutile, None)
        } else if raw.positive_trend {
            let v = if improvement[index].1 {
                FutilityVerdict::Winning
            } else {
                FutilityVerdict::Continue
            };
            (v, Some(ContinueReason::PositiveOrBorderlineTrend))
        } else {
            (
                FutilityVerdict::Continue,
                Some(ContinueReason::NonSignificantRegression),
            )
        };

        if verdict != FutilityVerdict::StopFutile {
            all_stopped = false;
        }

        outcomes.push(CandidateFutilityOutcome {
            candidate_id: cand.candidate_id.clone(),
            blocks_observed: raw.blocks_observed,
            total_pairs: raw.total_pairs,
            mean_gain: raw.mean_gain,
            relative_gain_pct: raw.relative_gain_pct,
            wilcoxon: raw.wilcoxon,
            p_futility: raw.p_futility,
            q_futility: q_val,
            verdict,
            continue_reason,
        });
    }

    Ok(SweepFutilityReport {
        candidates: outcomes,
        any_stopped,
        all_stopped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_lower_candidate(id: &str, blocks: Vec<(usize, Vec<f64>, Vec<f64>)>) -> SweepCandidate {
        SweepCandidate {
            candidate_id: id.to_owned(),
            direction: MetricDirection::LowerIsBetter,
            blocks: blocks
                .into_iter()
                .map(|(block_index, baseline, candidate)| PairedBlock {
                    block_index,
                    baseline,
                    candidate,
                })
                .collect(),
        }
    }

    #[test]
    fn signed_ranks_are_invariant_to_a_change_of_units() {
        let gains = [-1.0, 2.0, 3.0, -4.0, 5.0, 6.0];
        let scaled = gains.map(|gain| gain * 1e-14);
        assert_eq!(
            wilcoxon_signed_rank_test(&gains).unwrap(),
            wilcoxon_signed_rank_test(&scaled).unwrap()
        );
    }

    #[test]
    fn blocks_must_be_distinct_and_nonempty() {
        let block = (1, vec![10.0; 8], vec![12.0; 8]);
        for other in [block.clone(), (2, Vec::new(), Vec::new())] {
            let candidate = mk_lower_candidate("a", vec![block.clone(), other]);
            assert!(evaluate_sequential_futility(&[candidate], FutilityConfig::default()).is_err());
        }
    }

    #[test]
    fn improvement_is_adjusted_for_the_candidate_family() {
        let blocks = vec![
            (1, vec![8.0; 3], vec![7.0; 3]),
            (2, vec![8.0; 3], vec![7.0; 3]),
        ];
        let winner = mk_lower_candidate("a", blocks);
        let tie = mk_lower_candidate(
            "b",
            vec![(1, vec![8.0], vec![8.0]), (2, vec![8.0], vec![8.0])],
        );
        let report =
            evaluate_sequential_futility(&[winner, tie], FutilityConfig::default()).unwrap();
        assert_eq!(report.candidates[0].wilcoxon.p_two_sided, 0.03125);
        assert_eq!(report.candidates[0].verdict, FutilityVerdict::Continue);
    }

    #[test]
    fn futility_loser_2block_stops_and_winner_continues() {
        let loser = mk_lower_candidate(
            "loser_2block",
            vec![
                (
                    1,
                    vec![10.0, 10.2, 9.8, 10.1, 9.9, 10.3],
                    vec![12.5, 12.8, 12.2, 12.6, 12.4, 12.9],
                ),
                (
                    2,
                    vec![10.0, 10.1, 9.9, 10.2, 10.0, 10.1],
                    vec![12.6, 12.7, 12.5, 12.8, 12.4, 12.7],
                ),
            ],
        );
        let winner = mk_lower_candidate(
            "winner_2block",
            vec![
                (
                    1,
                    vec![12.0, 12.1, 11.9, 12.2, 12.0, 12.3],
                    vec![9.5, 9.4, 9.6, 9.5, 9.3, 9.7],
                ),
                (
                    2,
                    vec![12.1, 12.0, 12.2, 11.9, 12.1, 12.0],
                    vec![9.4, 9.5, 9.3, 9.6, 9.4, 9.5],
                ),
            ],
        );

        let report = evaluate_sequential_futility(&[loser, winner], FutilityConfig::default())
            .expect("evaluate sweep");
        assert!(report.any_stopped);
        assert!(!report.all_stopped);
        assert_eq!(report.candidates[0].verdict, FutilityVerdict::StopFutile);
        assert!(report.candidates[0].q_futility < 0.01);
        assert_eq!(report.candidates[1].verdict, FutilityVerdict::Winning);
        assert!((report.candidates[1].p_futility - 1.0).abs() < 1e-12);
    }

    #[test]
    fn futility_single_block_exact_tie_and_borderline_never_false_stop() {
        let single_block_only = mk_lower_candidate(
            "single_block_only",
            vec![(
                1,
                vec![10.0, 10.1, 10.2, 9.9, 10.0, 10.1],
                vec![12.0, 12.1, 12.2, 12.0, 12.1, 12.3],
            )],
        );
        let exact_tie = mk_lower_candidate(
            "exact_tie",
            vec![
                (
                    1,
                    vec![10.0, 10.0, 10.0, 10.0],
                    vec![10.0, 10.0, 10.0, 10.0],
                ),
                (
                    2,
                    vec![10.0, 10.0, 10.0, 10.0],
                    vec![10.0, 10.0, 10.0, 10.0],
                ),
            ],
        );
        let borderline_nonfutile = mk_lower_candidate(
            "borderline_nonfutile",
            vec![
                (
                    1,
                    vec![12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                    vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                ),
                (
                    2,
                    vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                    vec![10.1, 9.9, 10.0, 10.1, 9.9, 10.0],
                ),
            ],
        );

        let r_single =
            evaluate_sequential_futility(&[single_block_only], FutilityConfig::default()).unwrap();
        assert!(!r_single.any_stopped);
        assert_eq!(
            r_single.candidates[0].continue_reason,
            Some(ContinueReason::InsufficientBlocks)
        );

        let r_tie = evaluate_sequential_futility(&[exact_tie], FutilityConfig::default()).unwrap();
        assert!(!r_tie.any_stopped);
        assert_eq!(
            r_tie.candidates[0].continue_reason,
            Some(ContinueReason::ExactTie)
        );

        let r_border =
            evaluate_sequential_futility(&[borderline_nonfutile], FutilityConfig::default())
                .unwrap();
        assert!(!r_border.any_stopped);
        assert_ne!(r_border.candidates[0].verdict, FutilityVerdict::StopFutile);
    }

    #[test]
    fn wilcoxon_normal_fallback_handles_large_n_with_ties() {
        let gains: Vec<f64> = (0..80).map(|i| -((i % 7 + 1) as f64)).collect();
        let res = wilcoxon_signed_rank_test(&gains).unwrap();
        assert_eq!(res.n_effective, 80);
        assert!((res.w_pos - 0.0).abs() < 1e-12);
        assert!(res.w_neg > 3000.0);
        assert!(res.p_two_sided < 1e-10);
        assert!(res.p_harm_one_sided < 1e-10);
    }
}

# DFlash2 objective ablation

The v0.6.0 recommendation is to retain the shipped v0.5.0 DFlash2 checkpoint. The four-arm comparison found no clear holdout acceptance gain from changing either training objective.

## Setup

Both new arms used the same shipped block-16 checkpoint, taps `[5, 15, 25, 35, 45]`, 623 minibatches and training order as the completed hard-label and combined arms. Unary-only used soft verifier probabilities for unary loss and kept the hard selector objective. Selector-only kept hard unary labels and used soft verifier probabilities for the selector objective. The expected accepted length auxiliary followed its selector branch because it depends on that branch's candidate list.

The native Q8_0 worker evaluated all four arms on the same 16 code and 16 Japanese holdouts with matched prompt prefixes and seeds. The comparison reused the original hard-label and combined receipts, with the existing 32K vocabulary for code and CJK64K for Japanese.

## Holdout results

| Workload | Arm | Vocabulary loss | Top-16 loss | Selector mismatch | Expected acceptance |
|---|---|---:|---:|---:|---:|
| Code | Hard labels | 4.59% | 1.77% | 12.46% | 81.19% |
| Code | Unary-only | 4.43% | 1.97% | 12.55% | 81.05% |
| Code | Selector-only | 4.26% | 2.09% | 13.13% | 80.53% |
| Code | Combined | 4.09% | 2.08% | 13.71% | 80.12% |
| Japanese | Hard labels | 2.70% | 12.60% | 24.46% | 60.24% |
| Japanese | Unary-only | 2.75% | 12.46% | 24.39% | 60.40% |
| Japanese | Selector-only | 2.66% | 12.63% | 24.34% | 60.38% |
| Japanese | Combined | 3.02% | 12.69% | 24.00% | 60.30% |

| Workload | Arm vs hard labels | Acceptance change | Paired 95% interval | Prompts improved |
|---|---|---:|---:|---:|
| Code | Unary-only | −0.14 pp | −2.09 to +1.71 pp | 10/16 |
| Code | Selector-only | −0.66 pp | −2.03 to +0.81 pp | 6/16 |
| Code | Combined | −1.07 pp | −2.43 to +0.40 pp | 6/16 |
| Japanese | Unary-only | +0.15 pp | −0.87 to +1.18 pp | 6/16 |
| Japanese | Selector-only | +0.13 pp | −1.33 to +1.60 pp | 8/16 |
| Japanese | Combined | +0.05 pp | −1.26 to +1.32 pp | 8/16 |

The shipped checkpoint scored 80.12% on code and 59.51% on Japanese in the original run. Hard labels had the highest code estimate and unary-only the highest Japanese estimate. All paired expected-acceptance intervals between the four trained arms included zero. Hybrid timing was skipped because the holdouts did not establish an acceptance gain.

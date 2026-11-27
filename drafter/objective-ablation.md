# DFlash2 objective ablation

Retain the shipped v0.5.0 DFlash2 checkpoint (revision `1ddcc67fa6916f58136a6ea3e99d2e816da90a0a`) for v0.6.0. The holdout results give no clear acceptance gain from changing either training objective.

Both new arms used the same shipped block-16 checkpoint, taps `[5, 15, 25, 35, 45]`, 623 minibatches and training order as the completed hard-label and combined arms. Unary-only used soft verifier probabilities for unary loss and kept the hard selector objective. Selector-only kept hard unary labels and used soft verifier probabilities for the selector objective. The expected accepted length auxiliary followed its selector branch because it depends on that branch's candidate list.

The native Q8_0 worker evaluated all four arms on the same 16 code and 16 Japanese holdouts with matched input prefixes and seeds, reusing the original hard-label and combined receipts. Code used the existing 32K vocabulary and Japanese used CJK64K. Each alternative produced a different continuation from the hard-label control on every prompt, so the arms reached proposals with different token histories.

## Holdout results

Expected acceptance averages the probability overlap between verifier and drafter across the proposals reached in each arm. The paired 95% intervals use a prompt bootstrap. Accepted drafts per round is the observed total accepted drafts divided by the total neural verification rounds.

| Workload | Arm | Vocabulary loss | Top-16 loss | Selector mismatch | Expected acceptance | Accepted drafts per round |
|---|---|---:|---:|---:|---:|---:|
| Code | Shipped v0.5.0 | 4.46% | 2.27% | 13.15% | 80.12% | 2.548 |
| Code | Hard labels | 4.59% | 1.77% | 12.46% | 81.19% | 2.629 |
| Code | Unary-only | 4.43% | 1.97% | 12.55% | 81.05% | 2.639 |
| Code | Selector-only | 4.26% | 2.09% | 13.13% | 80.53% | 2.510 |
| Code | Combined | 4.09% | 2.08% | 13.71% | 80.12% | 2.467 |
| Japanese | Shipped v0.5.0 | 3.00% | 13.56% | 23.94% | 59.51% | 1.148 |
| Japanese | Hard labels | 2.70% | 12.60% | 24.46% | 60.24% | 1.136 |
| Japanese | Unary-only | 2.75% | 12.46% | 24.39% | 60.40% | 1.141 |
| Japanese | Selector-only | 2.66% | 12.63% | 24.34% | 60.38% | 1.138 |
| Japanese | Combined | 3.02% | 12.69% | 24.00% | 60.30% | 1.126 |

| Workload | Arm vs hard labels | Acceptance change | Paired 95% interval | Prompts improved |
|---|---|---:|---:|---:|
| Code | Unary-only | −0.14 pp | −2.09 to +1.71 pp | 10/16 |
| Code | Selector-only | −0.66 pp | −2.03 to +0.81 pp | 6/16 |
| Code | Combined | −1.07 pp | −2.43 to +0.40 pp | 6/16 |
| Japanese | Unary-only | +0.15 pp | −0.87 to +1.18 pp | 6/16 |
| Japanese | Selector-only | +0.13 pp | −1.33 to +1.60 pp | 8/16 |
| Japanese | Combined | +0.05 pp | −1.26 to +1.32 pp | 8/16 |

On code, unary-only accepted 0.091 more drafts per round than shipped, with a paired 95% interval of −0.127 to +0.318. On Japanese, every trained arm accepted fewer drafts per round than shipped by point estimate. All paired expected-acceptance intervals between trained arms included zero, so the results did not justify a checkpoint change or hybrid timing.

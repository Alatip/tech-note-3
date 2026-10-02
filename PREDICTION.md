# Pre-registration — tetrachoric + correlated Condorcet vs residualized double-fault

Written 2026-10-02 BEFORE any predictor was computed on the data. Bridge:
frag:spearman-brown-and-portfolio-limit -> arXiv 2607.20768 (Kim 2026).

## Claim being tested

For subsets of LLMs, a marginal-free dependence estimate (pairwise
tetrachoric correlation of correctness, estimated per difficulty stratum)
combined with the members' individual accuracies in a correlated-Condorcet
model predicts observed held-out majority-vote gain over the best member
better than Kim 2026's capability-residualized double-fault, and better than
the phi-based Kish n_eff that swarm-ceiling uses.

## Data (all public, no inference spend)

Primary, D1: Chen 2026 (HF `josefchen/co-failure-67-models`, CC-BY 4.0),
`matrix_marketE2_final.json`, dataset `mmlu_pro`: 52 models with full
coverage x 200 items, zero-shot, temperature 0. Closest available analogue
of Kim's MMLU-Pro matrix (Kim's own matrices are not released as of today).

Robustness: D2 `math_hard` 67 x 298; D3 `math500` (same file) 52 x 200;
D4 Open LLM Leaderboard v1 per-item cache in the Open LLM Leaderboard v1 per-item cache
(README, D4): 28 base models x 14,042 MMLU items — older, weaker models, but
enough items to make difficulty strata stable.

## Protocol (mirrors Kim 2026 where it can)

1. Subsets: all size-3 subsets (D1: 22,100). Size 4 on a random sample of
   20,000 subsets, seed 0, as secondary.
2. Item split: 20 seeds; each seed splits items 50/50 into TRAIN / TEST.
   Everything that defines a predictor (accuracies, correlations, strata,
   best-member choice, residualization fit) uses TRAIN only. The target uses
   TEST only.
3. Target: gain = acc_TEST(majority vote, correct-count rule, even ties =
   incorrect) - acc_TEST(member chosen as best on TRAIN). Held-out best, as in
   Kim's held-out variant.
4. Difficulty strata: terciles of item difficulty. Defining difficulty from
   the models outside each subset is too expensive per subset; instead, as in
   Kim's "non-circular split", models are split once per seed into DEFINERS
   (half) and EVALUATORS (half); difficulty terciles come from the definers'
   mean correctness; subsets are drawn from evaluators only. D1 therefore
   tests 26 evaluators -> 2,600 size-3 subsets per seed. The all-52-model
   analysis with circular (all-model) difficulty is reported as secondary.
5. Predictors (all computed on TRAIN):
   (a) phi-n_eff: mean pairwise phi of the k members' correctness vectors,
       n_eff = k / (1 + (k-1) phi_bar). Larger = more gain predicted.
   (b) tetrachoric + correlated Condorcet: per stratum s, per-model
       accuracies p_is and the full evaluator x evaluator tetrachoric matrix
       R_s (polygon `tetrachoric_vec`), projected to the nearest PSD matrix
       (eigenvalue clip at 1e-6, unit diagonal restored; distortion reported).
       Gaussian copula simulation: 20,000 latent draws per stratum with
       Cholesky(R_s), thresholded at Phi^-1(1-p_is), giving a simulated
       correctness matrix per stratum. Predicted majority-vote accuracy of a
       subset = stratum-weighted majority accuracy on the simulated matrix;
       predicted gain = that minus the TRAIN-best member's accuracy. This is
       the Boland/Ladha heterogeneous-competence correlated jury on a
       Gaussian latent. Variant (b0): same without strata (one R, one p).
   (c) Kim's residualized double-fault: mean pairwise double-fault rate
       (both wrong) over the subset, rank-transformed, OLS-residualized on
       ranked TRAIN best accuracy and ranked TRAIN mean accuracy across all
       subsets of the seed; predictor = minus the residual (less shared error
       = more gain).
   (d) capability-only controls: TRAIN best accuracy, TRAIN mean accuracy.
6. Metrics per seed: Spearman rho(predictor, gain) over subsets; AUC for
   gain > 0; for (b) only, which is on the gain scale, also Pearson R^2 and
   mean signed error (calibration). Report mean over 20 seeds with the
   seed-to-seed 2.5/97.5 percentiles AND a model-level cluster bootstrap
   (B=300, resample evaluator models with replacement, deduplicate) for the
   difference rho(b) - rho(c).

## Falsification criteria (fixed now)

- H1 CONFIRMED if on D1, size 3, non-circular protocol: mean Spearman of
  (b) exceeds that of (c) AND the 95% model-bootstrap interval of the
  difference excludes 0 AND (b) also exceeds (a). Otherwise H1 is NOT
  supported on this data, regardless of how the other datasets look.
- H1 REJECTED outright if (b) is not better than the capability-only
  control (d, best accuracy) on D1: then the latent model adds nothing over
  marginals, which is Kim's thesis.
- Strata question (A's "rho is not constant"): strata help if
  rho(b) - rho(b0) > 0 with the bootstrap interval excluding 0.
- Secondary expectation from the judge: AUC(gain>0) for (b) above 0.60.
  Kim reports 0.597 for residualized double-fault on his data; our data
  differ, so this is descriptive, not a criterion.
- Descriptive, for fragment A: phi-bar vs tetrachoric-bar and the implied
  n_eff per stratum; whether 1/rho behaves as a ceiling for majority vote
  (does predicted MV accuracy saturate with k at the copula's limit).

## What I expect (honest prior)

Prior ~40% that H1 is confirmed. Reasons it may fail: (i) with heterogeneous
accuracy the best-member term dominates gain, so every predictor is mostly
"how far below the best are the others" (Kim's finding); (ii) the Gaussian
copula underprices tail co-failure (Chen 2026); (iii) 100 TRAIN items per
stratum tercile (~33) make tetrachoric noisy on D1, which is why D4 is in.
Expected: (b) >= (c) in rank correlation, but difference small (<0.05)
and bootstrap interval touching 0 -> "not falsified, not confirmed".

# Already-done check — bridge frag:spearman-brown-and-portfolio-limit -> 2607.20768

Date: 2026-10-02. Searched: arXiv (web), Semantic Scholar citation graph of
2607.20768 (8 citing papers as of today), web search for tetrachoric /
latent-correlation / Gaussian-copula + Condorcet / majority vote on LLM
ensembles, 2025–2026.

Hypothesis under test: marginal-free dependence (tetrachoric correlation of
member errors, per difficulty stratum) + individual accuracies, plugged into a
correlated-Condorcet (heterogeneous-competence) model, predicts per-subset
majority-vote gain over the best member BETTER than capability-residualized
double-fault (Kim 2026's surviving metric) and than phi-based n_eff.

## Verdict: NOT clearly done. Ingredients exist in two concurrent papers; the
## head-to-head comparison on the Kim question (gain over best, per subset,
## per difficulty stratum) does not.

| paper | what it does | overlap with our test | gap |
|---|---|---|---|
| **Li & Hai, arXiv 2607.23931** (Jul 27 2026) "State-dependent error correlations shape voting thresholds in committees of AI agents" | 28 LLMs, 174k votes on 4 binary-screening benchmarks. Pairwise latent correlation by **tetrachoric inversion**, separate per state (good/bad case), heterogeneous Gaussian copula, PSD projection; predicts held-out cost-weighted loss of k-of-n rules. Full copula R²=0.967 vs independence 0.840. Zenodo 10.5281/zenodo.21303076 | **Closest.** Same estimator (tetrachoric), same copula, same voting family. | Target is loss under κ, not gain over best member; no comparison to double-fault/phi/residualized metrics; no difficulty strata; binary screening, not MC QA; does not cite Kim 2026. |
| **Chen, arXiv 2606.27288** (Jun 2026) "When Does Combining LMs Help? A Co-Failure Ceiling… 67 models" | Fits tetrachoric single-factor Gaussian copula to binary errors to predict the all-wrong rate β; shows the Gaussian floor underprices the co-failure tail (later audit: most of the tail was grader defects). Regresses majority-vote gain over best on ρ with accuracy-headroom control. Data: HF `josefchen/co-failure-67-models`, CC-BY 4.0 | Uses tetrachoric copula on LLM errors; has a gain-over-best regression. | Predicts β, not per-subset gain; one ρ, no strata; no head-to-head of estimators. **Its data is the usable public substitute for Kim's unreleased matrices.** |
| Kohli, arXiv 2605.29800 "Nine Judges, Two Effective Votes" | phi + Kish n_eff + item-aware Condorcet null, 9 judges | phi-based n_eff baseline (our predictor a) | explicitly phi, not tetrachoric; no subset-level prediction |
| Jha, arXiv 2608.16190 "Decorrelation Is Not Complementarity" (cites Kim) | 24 monitors; splits agreement into shared-detectability vs idiosyncratic error, skill predicts ensemble gain (Spearman 0.84) | same spirit (capability vs dependence) | Pearson/Spearman only, no latent model |
| Sasahara et al., arXiv 2609.14438 "A latent dimension of Condorcet's jury theorem for multiple AI advisers" (cites Kim) | binomial (independent) model of dissent vs reliability | title only | no correlation at all |
| other citers of Kim (2608.00285, 2608.18795, 2609.27110, 2609.32996, 2609.39229) | dispersion / self-consistency / hallucination aggregation | none | — |

Kim 2026 itself: Limitations item 7 says "Concurrent preprints study
quality-matched pools, all-model co-failure ceilings, and accuracy-adjusted
pair dependence" — i.e. the author knows the neighbours and did not run a
latent-correlation predictor. The release plan promises correctness matrices
(model × item) but as of 2026-10-02 no GitHub/HF/Zenodo link exists (arXiv
page, HTML, web search all empty).

## Consequence for the test

- Novelty that remains: (i) per-difficulty-stratum tetrachoric, (ii) the
  heterogeneous correlated-Condorcet prediction of **gain over best** per
  subset, (iii) the head-to-head against Kim's residualized double-fault and
  against phi-n_eff, with CIs. Expect a reviewer to call it incremental
  relative to Li & Hai; the answer to Kim's open question is still the new
  thing.
- Risk flagged by Chen: the Gaussian copula underprices tail co-failure. If
  predictor (b) loses, that is a candidate reason, and the result is still
  informative for KNOWLEDGE.md.
- Data: Kim's matrices are not public. Use Chen's released matrices
  (MMLU-Pro and MATH-500, 52 models with full coverage on 200 items each;
  MATH-Hard; 15-model pools) plus, as a second dataset with enough items for
  difficulty strata, the Open LLM Leaderboard v1 per-item cache already on
  disk in the Open LLM Leaderboard v1 per-item cache (README, D4). No inference spend.

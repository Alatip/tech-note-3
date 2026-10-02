# Results — tetrachoric + correlated Condorcet vs residualized double-fault

Run 2026-10-02 with `run.py` (numpy only, estimators copied from
swarm-ceiling/polygon with attribution in `stats.py`). Protocol and
falsification criteria: PREDICTION.md, written before the first run. Raw
numbers: `results/*.json` and `results/*.log`. No LLM inference was spent; all
data are public (Chen 2026 HF release, CC-BY 4.0; Open LLM Leaderboard v1
cache in swarm-ceiling, read-only). Kim 2026's own matrices are not released.

One deviation from the pre-registration, made after a 2-seed smoke test and
before the main run: two extra controls were added, (e) the same
heterogeneous-competence Condorcet model with the latent correlation forced to
identity (independence), and (f) the same copula with phi instead of
tetrachoric as the latent R. Without (e) the headline would have been
misread; see below.

## Headline table — Spearman rho(predictor, held-out gain over best), size 3

Mean over 20 item-split seeds, [2.5%, 97.5%] over seeds. Non-circular
protocol: difficulty terciles from half the models, subsets from the other
half. Predictors oriented so that positive = "predicts more gain".

| predictor | D1 Chen MMLU-Pro (26 eval, 2,600 subsets) | D2 Chen MATH-Hard (33, 5,456) | D3 Chen MATH-500 (26, 2,600) | D4 Leaderboard MMLU (14, 364; 7,021 test items) |
|---|---|---|---|---|
| (b) tetrachoric per stratum + Condorcet | **+0.810** [0.74, 0.88] | **+0.778** [0.70, 0.88] | **+0.796** [0.69, 0.85] | **+0.992** [0.98, 1.00] |
| (b0) tetrachoric pooled + Condorcet | +0.800 | +0.772 | +0.788 | +0.990 |
| (e) INDEPENDENT Condorcet, same accuracies | +0.799 | +0.776 | +0.798 | +0.967 |
| (f) phi copula + Condorcet | +0.811 | +0.783 | +0.801 | +0.986 |
| (c) Kim: double-fault residualized on best+mean | +0.195 [0.10, 0.27] | +0.281 | +0.259 | +0.175 |
| (c') raw double-fault (sign flipped) | +0.367 | +0.594 | +0.609 | +0.370 |
| (a) phi Kish n_eff | **−0.422** | −0.294 | −0.151 | **−0.802** |
| (d) −best accuracy | +0.028 | −0.130 | −0.184 | +0.039 |
| (d) mean accuracy | +0.502 | +0.654 | +0.676 | +0.609 |
| AUC(gain>0) for (b) | 0.872 | 0.832 | 0.857 | 0.998 |
| AUC(gain>0) for (c) | 0.552 | 0.578 | 0.560 | 0.463 |
| calibrated R² of (b) on the gain scale | 0.70 | 0.81 | 0.72 | 0.99 |
| gain>0 base rate | 8.7% | 22.1% | 13.8% | 0.1% |

Model-level cluster bootstrap (B=300, resample evaluator models) of the
Spearman difference, D1 size 3:

| difference | mean [95% CI] |
|---|---|
| (b) − (c) residualized double-fault | +0.604 [+0.552, +0.653] |
| (b) − (a) phi n_eff | +1.216 [+1.172, +1.268] |
| (b) − (d) −best acc | +0.771 [+0.716, +0.825] |
| (b) − (b0) pooled | +0.009 [+0.002, +0.015] |
| **(b) − (e) independent Condorcet** | **+0.011 [+0.005, +0.017]** |
| (b) − (f) phi copula | −0.002 [−0.004, −0.000] |

Same differences on D4 (14k items): (b)−(e) = +0.023 [+0.018, +0.029];
(b)−(f) = +0.006 [+0.004, +0.008]; (b)−(b0) = +0.003 [+0.001, +0.004].

Secondary: D1 circular (all 52 models, 22,100 subsets): (b) +0.828,
(e) +0.815, (c) +0.201, diff (b)−(e) +0.013 [+0.011, +0.016]. D1 size 4
(20,000 sampled subsets; gain>0 in 0.3%): (b) +0.880, (e) +0.872,
(c) +0.035, (a) −0.577.

## Verdict against the pre-registered criteria

1. **H1 formally CONFIRMED** on D1: (b) beats (c) by 0.60 rank points and
   (a) by 1.2, bootstrap CIs far from 0; (b) also beats the capability-only
   controls. Replicates on D2, D3, D4, size 4 and the circular protocol.
2. **But the win is not what the hypothesis claimed.** The independent
   heterogeneous Condorcet model (e), which uses NO dependence estimate at
   all, is within 0.01 of (b) on every Chen dataset. The dependence term
   (tetrachoric, phi, strata) contributes +0.01 to +0.02 rank points: real
   (CIs exclude 0), but an order of magnitude smaller than the gap to Kim's
   metric. Phi and tetrachoric as the latent R are indistinguishable
   (−0.002 to +0.006). Strata add +0.003 to +0.009.
3. So the correct reading of Kim 2026: residualizing a pairwise overlap
   statistic on best+mean accuracy throws away the information. Majority-vote
   gain over the best member is, to first order, a known **nonlinear
   function of the three member accuracies** (the Condorcet jury
   probability minus the max), which a linear rank residualization on
   (best, mean) cannot represent. Kim's "co-failure survives control, AUC
   0.597" and the +0.55 AUC we get for (c) are what remains after the
   linear control has removed most of the signal that a model-based
   predictor keeps. No estimator of rho is needed to beat it.
4. For fragment A's questions: (i) the Kish/phi n_eff of a subset
   ANTI-predicts gain (rho −0.42 on D1, −0.80 on D4) — low-correlation
   subsets are the ones that pair a strong model with weak, dissimilar
   ones, so n_eff is confounded with accuracy heterogeneity and is not a
   usable gain predictor on its own; (ii) with ~30 items per stratum the
   within-stratum tetrachoric matrices are far from PSD (max projection
   distortion 1.0–1.2 on D1–D3), i.e. the per-stratum latent estimate is
   mostly noise at Kim/Chen sample sizes and effectively collapses to
   independence after projection; on D4 (2,000+ items per stratum,
   distortion 0.000) the dependence term is where it helps most (+0.023);
   (iii) rho is indeed not constant across difficulty: phi 0.18–0.23 on the
   hard tercile vs 0.03–0.09 on medium/easy terciles in the Chen data, 0.09 →
   0.11 on D4; the tetrachoric goes negative on the easy terciles of D1–D3,
   an artefact of near-1 marginals with 30 items, not a finding.
5. Calibration: (b) is on the gain scale with mean signed error < 0.01
   accuracy points and R² 0.70–0.99, so the copula-Condorcet model is a
   usable *forecast* of gain, not just a ranking.

## Caveats

- Not Kim's data: Chen's MMLU-Pro slice is 200 items / 52 models, zero-shot,
  temperature 0, with later-2026 frontier models; gain>0 base rate 8.7% vs
  Kim's 9.98%. The qualitative Kim results (raw double-fault positive sign,
  residual weak) reproduce, so the comparison is fair in kind.
- Subsets overlap heavily; the seed percentiles and the model bootstrap
  are the only uncertainty statements, no per-subset p-values (as in Kim).
- D4 models are 2023–24 base models with 24–77% MMLU accuracy; gain>0 is
  almost never observed there (0.1%), so its AUC is near-degenerate; its
  Spearman is the informative number.
- The copula simulation uses 20,000 draws per stratum; Monte-Carlo noise on
  predicted gain is ~0.003, below the differences reported except the
  smallest ones ((b)−(f)), which should be read as "no difference".

## Bridge status

Alive, but re-aimed. "Tetrachoric per stratum beats residualized
double-fault" is true and uninteresting; "a heterogeneous Condorcet model
on the marginals alone beats Kim's residualized metric by 0.6 Spearman and
turns Kim's AUC 0.60 into 0.87, and the dependence estimate adds only
0.01–0.02 on top" is the finding. Next step if pursued: write it up as a
short comment on Kim 2026 once Kim's matrices are released (rerun is one
command), and ask whether any dependence estimator adds more than 0.02 at
Kim's sample size — our prior now is no.

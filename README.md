# tech-note-3 — The Control Ate the Signal

Companion code for the technical note **"The Control Ate the Signal"**
(alatip.github.io/the-control-ate-the-signal/): majority-vote gain over the
best member is a nonlinear function of the members' accuracies. A
heterogeneous-competence Condorcet model with no dependence term predicts
held-out gain at Spearman 0.80 where Kim 2026's residualized double-fault
(arXiv:2607.20768) reaches 0.20; a tetrachoric dependence estimate adds 0.01.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23133111.svg)](https://doi.org/10.5281/zenodo.23133111) — archived release v1.0. The note itself: [doi:10.5281/zenodo.23133020](https://doi.org/10.5281/zenodo.23133020).

Order of reading, which is also the order the files were written in:

1. `ALREADY-DONE.md` — prior-art check before anything was run.
2. `PREDICTION.md` — pre-registration: protocol, predictors, falsification
   criteria, honest prior. Written before the first number.
3. `run.py` — the whole experiment, numpy only.
4. `RESULTS.md` — findings, including the one deviation from the
   pre-registration (two controls added after a 2-seed smoke test).
5. `results/*.json` — every reported number.

## Run

```
pip install -r requirements.txt     # numpy only
./data/fetch.sh                     # ~7 MB of Chen 2026 matrices (CC-BY 4.0)
python3 run.py --dataset D1-chen-mmlupro --k 3          # ~2 min
python3 run.py --dataset D2-chen-mathhard --k 3
python3 run.py --dataset D3-chen-math500 --k 3
python3 run.py --dataset D1-chen-mmlupro --k 4 --boot 100
python3 run.py --dataset D1-chen-mmlupro --k 3 --circular   # all 52 models, ~12 min
python3 run.py --dataset D4-lb-mmlu --k 3               # needs the leaderboard cache, below
```

Each run writes `results/<dataset>_k<k>[_circular].json` and prints the
summary table from the note.

**D4, leaderboard pool.** Per-item correctness of 28 open base models on the
57 MMLU subjects from the archived Open LLM Leaderboard v1 (ungated). Fetch it
with `core/hfdata.py` from [tech-note-1](https://github.com/Alatip/tech-note-1)
(the data-source notes there explain which leaderboard dataset is ungated and
the alignment traps), then point `LEADERBOARD_BITS` at the resulting
`.cache/bits` directory.

`stats.py` is the estimator file from tech-note-1 (tetrachoric via numerically
integrated bivariate normal, Acklam inverse normal, vectorised solver); copied
so this repo is self-contained.

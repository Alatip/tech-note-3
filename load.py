"""Load correctness matrices (items x models, 0/1) from public sources.

Chen 2026 matrices: data/matrix_*.json from HF josefchen/co-failure-67-models.
Leaderboard v1: tech-note-1 style cache, see CEILING_CACHE below.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# Open LLM Leaderboard v1 per-item cache in tech-note-1 format (bits/<task>/<model>.json);
# fetch it with tech-note-1's core/hfdata.py, then point this env var at the bits/ directory.
CEILING_CACHE = os.environ.get("LEADERBOARD_BITS", os.path.join(HERE, ".cache", "bits"))


def chen(file, dataset):
    """Items x models matrix restricted to models with full coverage."""
    d = json.load(open(os.path.join(DATA, file)))
    items = [q for q, v in d.items() if v["dataset"] == dataset]
    models = sorted({m for q in items for m in d[q]["models"]})
    full = [m for m in models
            if all(d[q]["models"].get(m, {}).get("correct") is not None for q in items)]
    M = np.array([[int(bool(d[q]["models"][m]["correct"])) for m in full] for q in items], dtype=np.int8)
    return M, full, items


def leaderboard_mmlu():
    """28 base models x 14,042 MMLU items from the Open LLM Leaderboard v1 cache."""
    tasks = sorted(t for t in os.listdir(CEILING_CACHE) if t.startswith("hendrycksTest"))
    per_model = {}
    for t in tasks:
        for f in sorted(glob.glob(os.path.join(CEILING_CACHE, t, "*.json"))):
            j = json.load(open(f))
            per_model.setdefault(j["model"], {})[t] = np.asarray(j["acc"], dtype=np.int8)
    models = sorted(m for m, v in per_model.items() if len(v) == len(tasks))
    cols = [np.concatenate([per_model[m][t] for t in tasks]) for m in models]
    M = np.stack(cols, axis=1)
    items = [f"{t}:{i}" for t in tasks for i in range(len(per_model[models[0]][t]))]
    return M, models, items


DATASETS = {
    "D1-chen-mmlupro": lambda: chen("matrix_marketE2_final.json", "mmlu_pro"),
    "D2-chen-mathhard": lambda: chen("matrix_marketMH_final.json", "math_hard"),
    "D3-chen-math500": lambda: chen("matrix_marketE2_final.json", "math500"),
    "D4-lb-mmlu": leaderboard_mmlu,
}

if __name__ == "__main__":
    for k, fn in DATASETS.items():
        M, models, items = fn()
        acc = M.mean(0)
        print(f"{k}: {M.shape[0]} items x {M.shape[1]} models, acc min/med/max {acc.min():.2f} {np.median(acc):.2f} {acc.max():.2f}")

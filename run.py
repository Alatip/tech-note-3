"""Pre-registered test (see PREDICTION.md): does tetrachoric + correlated
Condorcet predict held-out majority-vote gain over the best member better than
Kim 2026's residualized double-fault and than phi-based n_eff?

python3 run.py --dataset D1-chen-mmlupro --k 3
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import Phi_inv, tetrachoric_vec, pair_margins  # noqa: E402
from load import DATASETS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_SIM = 20000


# ----------------------------------------------------------------- helpers
def rank(x):
    x = np.asarray(x, float)
    r = np.empty(len(x)); order = np.argsort(x, kind="mergesort"); r[order] = np.arange(len(x))
    # average ties
    sx = x[order]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, positive):
    positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank(score) + 1
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def majority_acc(M, subsets, chunk=4000):
    """M: items x models 0/1. subsets: (ns, k) int. Returns per-subset MV accuracy
    under the correct-count rule, even ties incorrect."""
    k = subsets.shape[1]
    out = np.empty(len(subsets))
    for s in range(0, len(subsets), chunk):
        S = subsets[s:s + chunk]
        cnt = M[:, S].sum(axis=2)            # items x ns
        out[s:s + chunk] = (cnt > k / 2).mean(axis=0)
    return out


def pair_means(P, subsets):
    """Mean of a pairwise (m x m) statistic over the pairs inside each subset."""
    k = subsets.shape[1]
    iu = list(itertools.combinations(range(k), 2))
    vals = np.stack([P[subsets[:, i], subsets[:, j]] for i, j in iu], axis=1)
    return np.nanmean(vals, axis=1)


def tetra_matrix(M):
    m = M.shape[1]
    p1, p2, p11 = pair_margins(M)
    r = tetrachoric_vec(p1, p2, p11)
    R = np.eye(m)
    iu = np.triu_indices(m, 1)
    r = np.where(np.isnan(r), 0.0, r)
    R[iu] = r; R[(iu[1], iu[0])] = r
    return R


def nearest_psd(R, eps=1e-6):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    A = (V * w) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / d[:, None] / d[None, :]
    return A, float(np.abs(A - R).max())


def simulate_copula(R, p, n, rng):
    """Gaussian-copula correctness draws with marginals p and latent correlation R."""
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, len(p))) @ L.T
    q = np.array([Phi_inv(float(v)) for v in p])
    return (Z < q[None, :]).astype(np.int8)


def residualize_ranks(y, *xs):
    Y = rank(y)
    X = np.column_stack([np.ones(len(Y))] + [rank(x) for x in xs])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


# ----------------------------------------------------------------- one seed
def run_seed(M, seed, k, circular, n_strata=3):
    rng = np.random.default_rng(seed)
    n_items, m = M.shape
    perm_i = rng.permutation(n_items)
    train_i, test_i = np.sort(perm_i[: n_items // 2]), np.sort(perm_i[n_items // 2:])
    if circular:
        definers = np.arange(m); evaluators = np.arange(m)
    else:
        perm_m = rng.permutation(m)
        definers, evaluators = np.sort(perm_m[: m // 2]), np.sort(perm_m[m // 2:])
    # difficulty from definers only (all items; definers are disjoint from evaluators)
    diff = M[:, definers].mean(1)
    # terciles computed on TRAIN items, applied to all
    cuts = np.quantile(diff[train_i], [i / n_strata for i in range(1, n_strata)])
    stratum = np.searchsorted(cuts, diff, side="right")
    E = M[:, evaluators]
    Etr, Ete = E[train_i], E[test_i]
    me = len(evaluators)
    subsets = np.array(list(itertools.combinations(range(me), k)), dtype=np.int32)
    if k >= 4 and len(subsets) > 20000:
        subsets = subsets[np.random.default_rng(0).choice(len(subsets), 20000, replace=False)]

    acc_tr = Etr.mean(0)
    best_pos = np.argmax(acc_tr[subsets], axis=1)
    best_idx = subsets[np.arange(len(subsets)), best_pos]
    # ---- target: held-out gain
    gain = majority_acc(Ete, subsets) - Ete.mean(0)[best_idx]
    out = {"gain": gain}

    # ---- (a) phi n_eff
    with np.errstate(invalid="ignore", divide="ignore"):
        Phi_mat = np.corrcoef(Etr.T)
    Phi_mat = np.where(np.isnan(Phi_mat), 0.0, Phi_mat); np.fill_diagonal(Phi_mat, np.nan)
    phibar = pair_means(Phi_mat, subsets)
    out["a_phi_neff"] = k / (1 + (k - 1) * phibar)

    # ---- (c) residualized double-fault (Kim)
    DF = ((1 - Etr.T.astype(float)) @ (1 - Etr.astype(float))) / len(train_i)
    np.fill_diagonal(DF, np.nan)
    df = pair_means(DF, subsets)
    best_acc = acc_tr[best_idx]; mean_acc = acc_tr[subsets].mean(1)
    out["c_resid_df"] = -residualize_ranks(df, best_acc, mean_acc)
    out["c_raw_df"] = -df
    out["d_neg_best"] = -best_acc
    out["d_mean_acc"] = mean_acc

    # ---- (b) tetrachoric + correlated Condorcet, per stratum and pooled
    def phi_matrix(Ms):
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(Ms.T)
        C = np.where(np.isnan(C), 0.0, C); np.fill_diagonal(C, 1.0)
        return C

    def copula_pred(strata_ids, corr="tetra"):
        pred_mv = np.zeros(len(subsets)); pred_best = np.zeros(len(subsets)); dist = 0.0
        w_total = 0
        for s in strata_ids:
            tr = train_i[stratum[train_i] == s]; te_n = (stratum[test_i] == s).sum()
            if len(tr) < 5 or te_n == 0:
                continue
            Ms = M[tr][:, evaluators]
            p = np.clip(Ms.mean(0), 0.5 / len(tr), 1 - 0.5 / len(tr))
            if corr == "tetra":
                R = tetra_matrix(Ms)
            elif corr == "phi":
                R = phi_matrix(Ms)
            else:
                R = np.eye(me)
            R, d = nearest_psd(R); dist = max(dist, d)
            sim = simulate_copula(R, p, N_SIM, rng)
            pred_mv += te_n * majority_acc(sim, subsets)
            pred_best += te_n * p[best_idx]
            w_total += te_n
        return pred_mv / w_total - pred_best / w_total, dist

    out["b_tetra_strata"], dist_s = copula_pred(range(n_strata))
    stratum_all = stratum.copy(); stratum[:] = 0
    out["b0_tetra_pooled"], dist_0 = copula_pred([0])
    stratum[:] = stratum_all
    # controls added after the 2-seed smoke test, before the main run (see RESULTS.md):
    # same marginals, independence (pure heterogeneous Condorcet) and phi as the latent R
    out["e_indep_condorcet"], _ = copula_pred(range(n_strata), corr="indep")
    out["f_phi_copula"], _ = copula_pred(range(n_strata), corr="phi")
    # descriptive: phi-bar vs tetrachoric-bar per stratum (all evaluator pairs)
    desc = []
    for s in range(n_strata):
        tr = train_i[stratum[train_i] == s]
        Ms = M[tr][:, evaluators]
        p1, p2, p11 = pair_margins(Ms)
        tet = tetrachoric_vec(p1, p2, p11)
        C = np.corrcoef(Ms.T); ph = C[np.triu_indices(me, 1)]
        desc.append({"stratum": s, "n_train_items": int(len(tr)), "mean_acc": float(Ms.mean()),
                     "phi_bar": float(np.nanmean(ph)), "tetra_bar": float(np.nanmean(tet))})
    meta = {"n_subsets": int(len(subsets)), "n_eval": int(me), "psd_distortion": max(dist_s, dist_0),
            "gain_pos_rate": float((gain > 0).mean()), "desc": desc, "subsets": subsets}
    return out, meta


PREDICTORS = ["b_tetra_strata", "b0_tetra_pooled", "e_indep_condorcet", "f_phi_copula", "c_resid_df", "a_phi_neff",
              "c_raw_df", "d_neg_best", "d_mean_acc"]


def evaluate(out):
    g = out["gain"]
    res = {}
    for p in PREDICTORS:
        res[p] = {"spearman": spearman(out[p], g), "auc": auc(out[p], g > 0)}
    b = out["b_tetra_strata"]
    ss_res = float(((g - b) ** 2).sum()); ss_tot = float(((g - g.mean()) ** 2).sum())
    res["b_tetra_strata"]["r2_calibrated"] = 1 - ss_res / ss_tot
    res["b_tetra_strata"]["mean_signed_error"] = float((b - g).mean())
    return res


def model_bootstrap(outs, metas, B=300, seed=123):
    """Resample evaluator models with replacement (dedup); keep subsets fully inside the
    sample; recompute spearman differences averaged over seeds."""
    rng = np.random.default_rng(seed)
    pairs = [("b_tetra_strata", "c_resid_df"), ("b_tetra_strata", "a_phi_neff"),
             ("b_tetra_strata", "b0_tetra_pooled"), ("b_tetra_strata", "d_neg_best"),
             ("b_tetra_strata", "e_indep_condorcet"), ("b_tetra_strata", "f_phi_copula")]
    diffs = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(B):
        acc = {key: [] for key in diffs}
        for out, meta in zip(outs, metas):
            me = meta["n_eval"]
            keep = np.unique(rng.integers(0, me, me))
            mask = np.isin(meta["subsets"], keep).all(axis=1)
            if mask.sum() < 10:
                continue
            g = out["gain"][mask]
            for a, b in pairs:
                acc[f"{a}-{b}"].append(spearman(out[a][mask], g) - spearman(out[b][mask], g))
        for key in diffs:
            diffs[key].append(float(np.nanmean(acc[key])))
    return {key: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            for key, v in diffs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D1-chen-mmlupro")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--circular", action="store_true")
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()
    M, models, items = DATASETS[args.dataset]()
    t0 = time.time()
    outs, metas, per_seed = [], [], []
    for seed in range(args.seeds):
        out, meta = run_seed(M, seed, args.k, args.circular)
        outs.append(out); metas.append(meta); per_seed.append(evaluate(out))
        print(f"seed {seed:2d} n_sub {meta['n_subsets']} gain>0 {meta['gain_pos_rate']:.3f} "
              f"rho b {per_seed[-1]['b_tetra_strata']['spearman']:+.3f} b0 {per_seed[-1]['b0_tetra_pooled']['spearman']:+.3f} "
              f"c {per_seed[-1]['c_resid_df']['spearman']:+.3f} a {per_seed[-1]['a_phi_neff']['spearman']:+.3f} "
              f"indep {per_seed[-1]['e_indep_condorcet']['spearman']:+.3f} phiCop {per_seed[-1]['f_phi_copula']['spearman']:+.3f}  psd-dist {meta['psd_distortion']:.3f}", flush=True)
    summary = {}
    for p in PREDICTORS:
        for met in ("spearman", "auc"):
            v = np.array([ps[p][met] for ps in per_seed])
            summary[f"{p}.{met}"] = {"mean": float(np.nanmean(v)), "p2.5": float(np.nanpercentile(v, 2.5)),
                                     "p97.5": float(np.nanpercentile(v, 97.5))}
    v = np.array([ps["b_tetra_strata"]["r2_calibrated"] for ps in per_seed])
    summary["b_tetra_strata.r2_calibrated"] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}
    summary["b_tetra_strata.mean_signed_error"] = float(np.mean([ps["b_tetra_strata"]["mean_signed_error"] for ps in per_seed]))
    boot = model_bootstrap(outs, metas, B=args.boot)
    desc = {}
    for s in range(3):
        rows = [m["desc"][s] for m in metas if len(m["desc"]) > s]
        desc[s] = {kk: float(np.mean([r[kk] for r in rows])) for kk in ("n_train_items", "mean_acc", "phi_bar", "tetra_bar")}
    result = {"dataset": args.dataset, "k": args.k, "seeds": args.seeds, "circular": args.circular,
              "n_models": len(models), "n_items": len(items), "n_eval": metas[0]["n_eval"], "n_subsets": metas[0]["n_subsets"],
              "gain_pos_rate": float(np.mean([m["gain_pos_rate"] for m in metas])),
              "summary": summary, "model_bootstrap_diff": boot, "strata_desc": desc,
              "psd_distortion_max": max(m["psd_distortion"] for m in metas), "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    name = f"{args.dataset}_k{args.k}{'_circular' if args.circular else ''}.json"
    json.dump(result, open(os.path.join(HERE, "results", name), "w"), indent=1)
    print("\n== summary", name)
    for p in PREDICTORS:
        s, a = summary[f"{p}.spearman"], summary[f"{p}.auc"]
        print(f"  {p:18s} rho {s['mean']:+.3f} [{s['p2.5']:+.3f},{s['p97.5']:+.3f}]  auc {a['mean']:.3f} [{a['p2.5']:.3f},{a['p97.5']:.3f}]")
    print(f"  b calibrated R2 {summary['b_tetra_strata.r2_calibrated']['mean']:+.3f}  mean signed err {summary['b_tetra_strata.mean_signed_error']:+.4f}")
    for key, v in boot.items():
        print(f"  boot diff {key:34s} {v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")
    for s, d in desc.items():
        print(f"  stratum {s}: n_train {d['n_train_items']:.0f} acc {d['mean_acc']:.2f} phi {d['phi_bar']:.3f} tetra {d['tetra_bar']:.3f}")
    print(f"  gain>0 rate {result['gain_pos_rate']:.3f}; psd distortion max {result['psd_distortion_max']:.3f}; {result['runtime_s']}s")


if __name__ == "__main__":
    main()

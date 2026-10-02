"""Copied verbatim from ~/projects/swarm-ceiling/polygon/core/stats.py (read-only
companion repo, Aziz Latipov, 2026) on 2026-10-02 so this experiment is
self-contained. numpy-only estimators: Phi, Phi_inv, bvn_upper, phi_coefficient,
tetrachoric, tetrachoric_vec, pair_margins, kish_neff."""
"""Estimators. No scipy — bivariate normal CDF is integrated here so the rig
stays dependency-light (numpy only) and every number is auditable."""
import math
import numpy as np

SQRT2 = math.sqrt(2.0)

def Phi(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, dtype=float) / SQRT2))

def phi_scalar(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2))

def Phi_inv(p):
    """Acklam's inverse normal CDF. Max relative error ~1.15e-9."""
    if not (0.0 < p < 1.0):
        raise ValueError("Phi_inv needs 0<p<1, got %r" % (p,))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bvn_upper(h, k, r, n=2000):
    """P(X > h, Y > k) for a standard bivariate normal with correlation r."""
    if r <= -0.999999:
        return max(0.0, 1.0 - phi_scalar(h) - phi_scalar(k))
    if r >= 0.999999:
        return 1.0 - phi_scalar(max(h, k))
    lim = 8.5
    if h >= lim:
        return 0.0
    lo, hi = h, lim
    step = (hi - lo) / n
    x = lo + (np.arange(n) + 0.5) * step
    w = np.exp(-x * x / 2.0) / math.sqrt(2 * math.pi)
    z = (k - r * x) / math.sqrt(1.0 - r * r)
    return float(np.sum(w * (1.0 - Phi(z))) * step)


def phi_coefficient(a, b):
    """Pearson correlation of two binary correctness vectors. Returns nan if
    either vector is constant (no information)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def tetrachoric(a=None, b=None, phi=None, p1=None, p2=None, tol=1e-7):
    """Latent bivariate-normal correlation behind two dichotomised variables.

    Either pass the raw vectors (a, b), or the summary (phi, p1, p2).
    This is the model-consistent estimator: the one-factor model in the note
    assumes a Gaussian latent, and dichotomising it attenuates phi.
    """
    if a is not None:
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        p1 = float(a.mean()); p2 = float(b.mean())
        p11 = float(np.mean((a > 0.5) & (b > 0.5)))
    else:
        if phi is None or p1 is None or p2 is None:
            raise ValueError("pass vectors, or all of phi/p1/p2")
        p11 = p1 * p2 + phi * math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    if min(p1, p2) <= 0 or max(p1, p2) >= 1:
        return float("nan")
    p11 = min(max(p11, 1e-9), min(p1, p2) - 1e-12)
    h = Phi_inv(1 - p1); k = Phi_inv(1 - p2)
    lo, hi = -0.999, 0.999
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if bvn_upper(h, k, mid) < p11:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def kish_neff(k, rho_bar):
    """Effective number of independent votes. k/(1+(k-1)*rho)."""
    return k / (1.0 + (k - 1) * rho_bar)


def pairwise(M, estimator="phi"):
    """M: (n_items, k_configs) binary. Returns (k,k) matrix, nan on diagonal."""
    M = np.asarray(M, dtype=float)
    k = M.shape[1]
    out = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            if estimator == "phi":
                v = phi_coefficient(M[:, i], M[:, j])
            elif estimator == "tetrachoric":
                v = tetrachoric(M[:, i], M[:, j])
            else:
                raise ValueError(estimator)
            out[i, j] = out[j, i] = v
    return out


def mean_pairwise(M, estimator="phi"):
    P = pairwise(M, estimator)
    iu = np.triu_indices(P.shape[0], 1)
    vals = P[iu]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def bootstrap_ci(M, fn, n_boot=1000, alpha=0.05, seed=0):
    """Item-level bootstrap. fn takes a matrix, returns a scalar."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(M[idx])
        except Exception:
            v = float("nan")
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals = np.sort(np.array(vals))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


# --------------------------------------------------------------------------
# Vectorised tetrachoric. ADDED, not substituted: `tetrachoric` above stays the
# reference implementation and every number it ever produced is unchanged. This
# solves many pairs at once because v004 needs 378 pairs x 57 subjects x a
# bootstrap, which is ~4e5 solves — hours at 3.7 ms each, seconds vectorised.
# tests/test_recovery.py asserts the two agree to 1e-6.
# --------------------------------------------------------------------------

# Phi() is np.vectorize(math.erf) — a Python-level loop, and it is what costs.
# So the vectorised path spends its budget on FEWER nodes rather than more:
# 64-node Gauss-Legendre matches the scalar path's 2000-node midpoint rule to
# 2.3e-7 (which is the midpoint rule's OWN error — GL is the more accurate of
# the two), at 30x fewer erf calls.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _bvn_upper_vec(h, k, r):
    """P(X>h, Y>k) for standard bivariate normals, elementwise over arrays."""
    h = np.asarray(h, float); k = np.asarray(k, float); r = np.asarray(r, float)
    lim = 8.5
    lo = np.minimum(h, lim)
    half = (lim - lo) / 2.0
    mid = (lim + lo) / 2.0
    x = mid[:, None] + half[:, None] * _GL_X[None, :]
    w = np.exp(-x * x / 2.0) / math.sqrt(2 * math.pi) * _GL_W[None, :] * half[:, None]
    rr = np.clip(r, -0.999999, 0.999999)[:, None]
    z = (k[:, None] - rr * x) / np.sqrt(1.0 - rr * rr)
    out = np.sum(w * (1.0 - Phi(z)), axis=1)
    return np.where(h >= lim, 0.0, out)


def tetrachoric_vec(p1, p2, p11, tol=1e-7):
    """Latent correlation from 2x2 margins, elementwise. Same bisection, same
    bounds and clamps as the scalar `tetrachoric`; nan where a margin is
    degenerate."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float); p11 = np.asarray(p11, float)
    ok = (np.minimum(p1, p2) > 0) & (np.maximum(p1, p2) < 1)
    q1 = np.where(ok, p1, 0.5); q2 = np.where(ok, p2, 0.5)
    t = np.clip(np.where(ok, p11, 0.25), 1e-9, None)
    t = np.minimum(t, np.minimum(q1, q2) - 1e-12)
    h = np.array([Phi_inv(1 - v) for v in q1])
    k = np.array([Phi_inv(1 - v) for v in q2])
    lo = np.full(q1.shape, -0.999); hi = np.full(q1.shape, 0.999)
    while np.max(hi - lo) > tol:
        mid = (lo + hi) / 2
        below = _bvn_upper_vec(h, k, mid) < t
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = (lo + hi) / 2
    # Near |r| = 1 the inner integrand turns into a near-step in x and 64 nodes
    # cannot resolve it — the fast and reference paths drifted apart by 1.7e-2
    # on a nested pair (p11 == min(p1,p2), a weak model right only where a
    # strong one also was). That regime is rare and it is exactly where the
    # estimator saturates, so it is handed back to the reference implementation
    # rather than approximated. Found by tests/test_recovery.py, not in review.
    hard = np.where(ok & (np.abs(out) > 0.95))[0]
    for i in hard:
        d = math.sqrt(q1[i] * (1 - q1[i]) * q2[i] * (1 - q2[i]))
        out[i] = tetrachoric(phi=(t[i] - q1[i] * q2[i]) / d, p1=float(q1[i]), p2=float(q2[i]))
    return np.where(ok, out, np.nan)


def pair_margins(M):
    """(p1, p2, p11) for every unordered pair of columns of a 0/1 matrix."""
    M = np.asarray(M, float)
    n, k = M.shape
    iu = np.triu_indices(k, 1)
    p = M.mean(0)
    p11 = (M.T @ M) / n
    return p[iu[0]], p[iu[1]], p11[iu]


def mean_tetrachoric(M):
    """Mean pairwise tetrachoric over all column pairs. Vectorised equivalent of
    mean_pairwise(M, 'tetrachoric')."""
    v = tetrachoric_vec(*pair_margins(M))
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if len(v) else float("nan")

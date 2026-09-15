"""Statistical tests used for the model comparisons.

  delong_roc_test(y, prob_a, prob_b) -> (auc_a, auc_b, p)   correlated ROCs
  bootstrap_auc_ci(y, prob)          -> (mean, lo, hi)
  paired_ttest(a, b)                 -> (t, p)              across seeds
  expected_calibration_error(y, prob)
  noninferiority_test(ref, ours, margin)
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# DeLong test, using the fast midrank formulation of Sun and Xu (2014).
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float); out[J] = T
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y_true, prob_a, prob_b):
    y = np.asarray(y_true).astype(int)
    order = (-y).argsort()                 # positives first
    m = int(y.sum())
    preds = np.vstack((np.asarray(prob_a), np.asarray(prob_b)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    L = np.array([[1.0, -1.0]])
    var = float(np.asarray(L.dot(cov).dot(L.T)).reshape(-1)[0])
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def bootstrap_auc_ci(y_true, prob, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int); p = np.asarray(prob)
    idx = np.arange(len(y)); aucs = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(aucs)), float(lo), float(hi)


def paired_ttest(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def expected_calibration_error(y_true, prob, n_bins=15):
    """Expected Calibration Error for a binary probability-of-positive.

    Confidence = probability assigned to the predicted class, i.e.
    max(p, 1-p). Predictions are grouped into `n_bins` equal-width confidence
    bins; ECE is the size-weighted mean gap between bin accuracy and bin mean
    confidence. Lower is better; zero is perfect calibration.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan")
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)         # confidence in predicted class
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def noninferiority_test(ref, ours, margin=0.02, alpha=0.05):
    """One-sided paired non-inferiority test on a metric measured across seeds.

    A non-significant superiority test does not establish equivalence, so the
    equivalence claim is tested directly against a margin fixed in advance.

    Hypotheses on the paired per-seed differences d = ref - ours:
        H0: mean(d) >= margin   (ours is inferior by at least `margin` AUROC)
        H1: mean(d) <  margin   (ours is non-inferior within `margin`)

    Non-inferiority is declared when the upper end of the one-sided (1-alpha)
    confidence interval for mean(d) lies below `margin`.

    Returns (mean_diff, ci_upper, p_value, is_noninferior); p_value is for the
    one-sided test that mean(d) < margin. At least two seeds are required.

    Parameter roles, spelled out because callers have shadowed these names:
    `ref` is the BASELINE being compared against; `ours` is the proposed model.
    Callers that keep a variable named `ref` for their own model (run_ablation.py
    does) must still pass the baseline first.
    """
    ref = np.asarray(ref, float); ours = np.asarray(ours, float)
    d = ref - ours
    k = len(d)
    if k < 2:
        return (float(np.mean(d)) if k else float("nan")), float("nan"), float("nan"), False
    mean_d = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = (sd / np.sqrt(k)) if sd > 0 else 1e-12
    tcrit = float(stats.t.ppf(1 - alpha, df=k - 1))          # one-sided critical value
    ci_upper = mean_d + tcrit * se
    t_stat = (mean_d - margin) / se
    p = float(stats.t.cdf(t_stat, df=k - 1))                 # P(mean_d < margin)
    return mean_d, float(ci_upper), p, bool(ci_upper < margin)

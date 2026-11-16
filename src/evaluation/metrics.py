"""Survey-weighted performance metrics, vectorised over replicate weights.

Every function takes ``y`` (0/1), ``p`` (predicted probability) and a weight matrix ``W`` of shape
(R, n): row 0 holds the full-sample expansion factors and rows 1..R-1 hold bootstrap replicate weights.
All estimators are Hajek-type ratios, so they estimate the population value of each metric.

Definitions (w_i = expansion factor, y_i = outcome, p_i = prediction, l_i = logit(p_i)):
  ROC-AUC   = sum_{i:y=1} sum_{j:y=0} w_i w_j [1(p_i>p_j) + 0.5*1(p_i=p_j)] / (W1*W0)
  PR-AUC    = average precision = sum_k (R_k - R_{k-1}) * P_k over distinct thresholds
  Brier     = sum w_i (p_i - y_i)^2 / sum w_i
  CITL      = a in logit P(y=1) = a + l_i   (offset model)   a>0: risk under-predicted, a<0: over-predicted
  slope     = b in logit P(y=1) = a' + b*l_i                 b<1: predictions too extreme, b>1: too timid
  O/E       = weighted observed prevalence / weighted mean prediction
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _as2d(W):
    W = np.asarray(W, dtype=float)
    return W[None, :] if W.ndim == 1 else W


def auc_ap(y, p, W, chunk: int = 200):
    """Weighted ROC-AUC and average precision for every weight row (tie-aware, identical to scikit-learn)."""
    y = np.asarray(y); p = np.asarray(p, dtype=float); W = _as2d(W)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order].astype(float), p[order]
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]      # last index of each tie group
    auc = np.empty(W.shape[0]); ap = np.empty(W.shape[0])
    for s in range(0, W.shape[0], chunk):
        Ws = W[s:s + chunk][:, order]
        tp = np.cumsum(Ws * ys, axis=1)[:, last]
        fp = np.cumsum(Ws * (1 - ys), axis=1)[:, last]
        P, N = tp[:, -1:], fp[:, -1:]
        tp0 = np.concatenate([np.zeros((tp.shape[0], 1)), tp[:, :-1]], axis=1)
        fp0 = np.concatenate([np.zeros((fp.shape[0], 1)), fp[:, :-1]], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            auc[s:s + chunk] = ((fp - fp0) * (tp + tp0) / 2).sum(1) / (P[:, 0] * N[:, 0])
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            ap[s:s + chunk] = ((tp - tp0) / P * prec).sum(1)
    return auc, ap


def brier(y, p, W):
    W = _as2d(W)
    return (W * (np.asarray(p) - np.asarray(y)) ** 2).sum(1) / W.sum(1)


def wmean(x, W):
    W = _as2d(W)
    return (W * np.asarray(x, dtype=float)).sum(1) / W.sum(1)


def citl(y, p, W, iters: int = 30, tol: float = 1e-9):
    """Calibration-in-the-large: intercept of the offset logistic model.

    Newton iterations (with clipped steps) run in parallel over the weight rows; the replicates start
    from the full-sample solution (row 0), which they reach in two or three steps.
    """
    y = np.asarray(y, dtype=float); l = logit(p); W = _as2d(W)

    def newton(Wm, a):
        for _ in range(iters):
            mu = expit(a[:, None] + l[None, :])
            step = np.clip((Wm * (y - mu)).sum(1) / np.maximum((Wm * mu * (1 - mu)).sum(1), 1e-12), -2.0, 2.0); a = a + step
            if np.max(np.abs(step)) < tol:
                break
        return a

    a0 = newton(W[:1], np.zeros(1))
    return a0 if W.shape[0] == 1 else np.r_[a0, newton(W[1:], np.full(W.shape[0] - 1, a0[0]))]


def calib_line(y, p, W, iters: int = 40, tol: float = 1e-9):
    """Logistic recalibration line logit P(y=1) = a + b*logit(p); returns (a, b) for every weight row."""
    y = np.asarray(y, dtype=float); l = logit(p); W = _as2d(W)

    def newton(Wm, a, b):
        for _ in range(iters):
            mu = expit(a[:, None] + b[:, None] * l[None, :]); r = Wm * (y - mu); v = Wm * mu * (1 - mu)
            g0, g1 = r.sum(1), (r * l).sum(1)
            h00, h01, h11 = v.sum(1), (v * l).sum(1), (v * l * l).sum(1)
            det = np.maximum(h00 * h11 - h01 ** 2, 1e-12)
            da = np.clip((h11 * g0 - h01 * g1) / det, -2.0, 2.0); db = np.clip((h00 * g1 - h01 * g0) / det, -0.5, 0.5)
            a, b = a + da, b + db
            if max(np.max(np.abs(da)), np.max(np.abs(db))) < tol:
                break
        return a, b

    # start at the perfectly calibrated line (a=0, b=1); steps are clipped, so the iteration cannot overshoot
    a0, b0 = newton(W[:1], np.zeros(1), np.ones(1))
    if W.shape[0] == 1:
        return a0, b0
    a, b = newton(W[1:], np.full(W.shape[0] - 1, a0[0]), np.full(W.shape[0] - 1, b0[0]))
    return np.r_[a0, a], np.r_[b0, b]


def threshold_metrics(y, p, W, thr: float):
    """Confusion-matrix metrics for the rule 'flag if p >= thr' (population-weighted)."""
    y = np.asarray(y, dtype=float); flag = (np.asarray(p) >= thr).astype(float); W = _as2d(W)
    tp = (W * flag * y).sum(1); fp = (W * flag * (1 - y)).sum(1)
    fn = (W * (1 - flag) * y).sum(1); tn = (W * (1 - flag) * (1 - y)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / (tp + fn); precision = tp / (tp + fp); specificity = tn / (tn + fp)
        f1 = 2 * precision * recall / (precision + recall)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, precision=precision, specificity=specificity,
                f1=f1, flag_rate=(tp + fp) / W.sum(1))


def all_metrics(y, p, W, thr: float | None = None) -> dict:
    """Every reported metric for every weight row. Keys map to arrays of length R."""
    W = _as2d(W)
    a, ap = auc_ap(y, p, W)
    _, slope = calib_line(y, p, W)
    prev, mp = wmean(y, W), wmean(p, W)
    out = dict(prev=prev, mean_pred=mp, oe=prev / mp, auc=a, prauc=ap, brier=brier(y, p, W), citl=citl(y, p, W), slope=slope)
    if thr is not None:
        t = threshold_metrics(y, p, W, thr)
        out.update({k: t[k] for k in ("recall", "precision", "specificity", "f1", "flag_rate", "tp", "fp", "fn", "tn")})
    return out


def summarize(m: dict, level: float = 0.95) -> dict:
    """Point estimate (row 0) and percentile interval (rows 1..) for each metric."""
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for k, v in m.items():
        out[k] = float(v[0])
        if len(v) > 1:
            r = v[1:][np.isfinite(v[1:])]
            out[k + "_lo"], out[k + "_hi"] = (float(np.percentile(r, lo)), float(np.percentile(r, hi))) if len(r) else (np.nan, np.nan)
            out[k + "_se"] = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    return out

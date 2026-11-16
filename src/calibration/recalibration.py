"""Model updating by recalibration (Steyerberg's hierarchy): intercept only, or intercept and slope.

Both are fitted with survey weights on an update sample and applied to later data:
  intercept only :  logit p' = a + logit p
  logistic       :  logit p' = a + b * logit p
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import calib_line, citl, expit, logit


def fit_intercept(y, p, w) -> float:
    return float(citl(y, p, w)[0])


def fit_logistic(y, p, w) -> tuple[float, float]:
    a, b = calib_line(y, p, w)
    return float(a[0]), float(b[0])


def apply(p, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return expit(a + b * logit(p))


def weighted_quantile(x, w, q: float) -> float:
    """Smallest value v such that the weighted share of x <= v is at least q."""
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(cw, q)])


def reliability_table(y, p, w, bins: int = 10):
    """Weighted mean prediction and observed rate in weighted-quantile bins of predicted risk."""
    import pandas as pd
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float); w = np.asarray(w, dtype=float)
    o = np.argsort(p, kind="stable"); cw = np.cumsum(w[o]) / w.sum()
    b = np.empty(len(p), dtype=int); b[o] = np.minimum((cw * bins).astype(int), bins - 1)
    d = pd.DataFrame({"b": b, "y": y, "p": p, "w": w})
    return d.groupby("b").apply(lambda g: pd.Series(dict(mean_pred=np.average(g.p, weights=g.w), observed=np.average(g.y, weights=g.w),
                                                         n=len(g), events=g.y.sum()))).reset_index(drop=True)

"""Design-based bootstrap for a stratified multistage survey (Rao-Wu rescaled bootstrap).

Rao and Wu (1988) / Rao, Wu and Yue (1992): inside every stratum h with n_h sampled PSUs, draw
m_h = n_h - 1 PSUs with replacement and rescale the expansion factor of each person in PSU j by
(n_h / (n_h - 1)) * r_hj, where r_hj is the number of times PSU j was drawn. The replicate weights are
then used to recompute any statistic; no rows are resampled, so every replicate keeps the full sample.

Assumptions and limits (documented in docs/methodology.md):
  * first-stage sampling fractions are treated as negligible (with-replacement approximation);
  * strata with a single PSU carry no variance information: they are merged into the smallest
    other stratum of the same year before resampling ("collapse"), and the number merged is logged;
  * the models are held fixed, so intervals describe test-sample uncertainty, not training variability;
  * when several years are pooled, a cluster code revisited in different years is one PSU, so the
    dependence created by the ENAHO panel component is kept inside the resampling unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_singletons(strata: pd.Series, psu: pd.Series) -> tuple[pd.Series, int]:
    """Merge strata that contain one PSU into the smallest other stratum; returns (new strata, n merged)."""
    strata = strata.astype(str).copy()
    n_psu = pd.DataFrame({"s": strata.values, "p": psu.values}).drop_duplicates().groupby("s").size()
    single = list(n_psu[n_psu < 2].index)
    if not single or len(n_psu) - len(single) < 1:
        return strata, 0
    target = n_psu[n_psu >= 2].idxmin()
    strata[strata.isin(single)] = target
    return strata, len(single)


def replicate_weights(w, strata, psu, n_rep: int, seed: int) -> np.ndarray:
    """Matrix (n_rep + 1, n): row 0 = original weights, rows 1.. = Rao-Wu replicate weights."""
    w = np.asarray(w, dtype=float)
    strata, _ = collapse_singletons(pd.Series(strata).reset_index(drop=True), pd.Series(psu).reset_index(drop=True))
    df = pd.DataFrame({"s": strata.values, "p": pd.Series(psu).astype(str).values})
    rng = np.random.default_rng(seed)
    W = np.empty((n_rep + 1, len(w))); W[0] = w
    factor = np.ones((n_rep, len(w)))
    for _, g in df.groupby("s", sort=True):
        codes, uniq = pd.factorize(g.p.values, sort=True)
        n_h = len(uniq)
        if n_h < 2:                       # only possible when the whole sample is one stratum with one PSU
            continue
        draws = rng.integers(0, n_h, size=(n_rep, n_h - 1))
        counts = np.zeros((n_rep, n_h))
        np.add.at(counts, (np.repeat(np.arange(n_rep), n_h - 1), draws.ravel()), 1.0)
        factor[:, g.index.values] = counts[:, codes] * n_h / (n_h - 1)
    W[1:] = w[None, :] * factor
    return W


def paired_diff(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> dict:
    """Difference a - b of two metric vectors computed with the same replicate weights."""
    d = a - b; r = d[1:][np.isfinite(d[1:])]
    lo, hi = np.percentile(r, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    # two-sided bootstrap p-value; undefined when the two vectors are identical by construction
    # (for example the ROC-AUC before and after a monotone recalibration)
    degenerate = bool(np.all(r == r[0])) if r.size else True
    p = float("nan") if degenerate else float(min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean())))
    return dict(diff=float(d[0]), lo=float(lo), hi=float(hi), se=float(np.std(r, ddof=1)), p_boot=p)


def taylor_ratio_se(y, w, strata, psu, domain=None, all_psu: pd.DataFrame | None = None) -> tuple[float, float]:
    """Linearised standard error of a weighted proportion (ultimate-cluster variance, domain-aware).

    ``all_psu``: optional frame with columns (strata, psu) listing every PSU of the full sample, so PSUs
    without domain members contribute zeros (correct domain estimation).
    """
    y = np.asarray(y, dtype=float); w = np.asarray(w, dtype=float)
    d = np.ones(len(y), dtype=bool) if domain is None else np.asarray(domain, dtype=bool)
    Wd = (w * d).sum(); m = (w * d * y).sum() / Wd
    z = pd.DataFrame({"s": np.asarray(strata), "p": np.asarray(psu), "z": w * d * (y - m) / Wd}).groupby(["s", "p"]).z.sum()
    if all_psu is not None:
        idx = pd.MultiIndex.from_frame(all_psu.drop_duplicates().rename(columns=dict(zip(all_psu.columns, ["s", "p"]))))
        z = z.reindex(idx).fillna(0.0)
    v = 0.0
    for _, s in z.groupby(level=0):
        n = len(s)
        if n > 1:
            v += n / (n - 1) * ((s - s.mean()) ** 2).sum()
    return float(m), float(np.sqrt(v))

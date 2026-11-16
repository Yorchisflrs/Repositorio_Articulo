"""SHAP utilities for tree models, permutation importance and family-level attributions.

Age-grade distortion is a deterministic function of age and years of schooling
(distortion = age - 6 - years), so SHAP shares credit among the three in a way that is not identified.
Family-level attributions (sum of SHAP values inside a family, then mean absolute value) are therefore
reported next to the per-feature values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.config import FAM_AREA, FAM_DIGITAL, FAM_HOUSEHOLD, FEATURES
from src.evaluation.metrics import auc_ap

FAMILIES = {
    "Age and schooling trajectory": ["age", "school_years", "overage"],
    "Other individual": ["female", "indig_tongue", "child_of_head"],
    "Household socioeconomic": FAM_HOUSEHOLD,
    "Digital access": FAM_DIGITAL,
    "Area": FAM_AREA,
}
assert sorted(sum(FAMILIES.values(), [])) == sorted(FEATURES)


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values on the log-odds scale (TreeExplainer, path-dependent)."""
    return np.asarray(shap.TreeExplainer(model).shap_values(X))


def mean_abs(sv: np.ndarray, w=None) -> pd.Series:
    return pd.Series(np.average(np.abs(sv), axis=0, weights=w), index=FEATURES)


def family_mean_abs(sv: np.ndarray, w=None) -> pd.Series:
    idx = {f: i for i, f in enumerate(FEATURES)}
    return pd.Series({fam: np.average(np.abs(sv[:, [idx[f] for f in cols]].sum(1)), weights=w) for fam, cols in FAMILIES.items()})


def permutation_importance(model, X: pd.DataFrame, y, w, groups: dict[str, list[str]], n_repeats: int, seed: int) -> pd.DataFrame:
    """Drop in survey-weighted ROC-AUC and PR-AUC when the columns of a group are permuted jointly."""
    rng = np.random.default_rng(seed)
    a0, p0 = auc_ap(y, model.predict_proba(X)[:, 1], w); rows = []
    for name, cols in groups.items():
        da, dp = [], []
        for _ in range(n_repeats):
            Xp = X.copy(); perm = rng.permutation(len(X)); Xp[cols] = X[cols].values[perm]
            a, p = auc_ap(y, model.predict_proba(Xp)[:, 1], w); da.append(a0[0] - a[0]); dp.append(p0[0] - p[0])
        rows.append(dict(feature=name, auc_drop=np.mean(da), auc_drop_sd=np.std(da, ddof=1), prauc_drop=np.mean(dp), prauc_drop_sd=np.std(dp, ddof=1)))
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)

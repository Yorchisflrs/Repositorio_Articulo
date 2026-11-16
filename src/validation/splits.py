"""Grouped and nested cross-validation.

Groups are cluster codes pooled across years, so every dwelling of a cluster (including panel dwellings
revisited in another year) stays in one fold. Hyper-parameters are chosen inside each outer training
fold (inner grouped CV, survey-weighted PR-AUC), so out-of-fold predictions never use their own labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import auc_ap
from src.modeling.models import expand_grid, fit_model, predict


def grouped_oof(name, params, X, y, groups, n_splits, train_w=None):
    """Out-of-fold predictions for a fixed configuration."""
    oof = np.zeros(len(y))
    for a, b in GroupKFold(n_splits).split(X, y, groups):
        m = fit_model(name, params, X.iloc[a], y[a], None if train_w is None else train_w[a])
        oof[b] = predict(m, X.iloc[b])
    return oof


def select_params(name, grid, X, y, w_eval, groups, n_splits, train_w=None):
    """Configuration with the highest survey-weighted PR-AUC of grouped out-of-fold predictions."""
    best = None
    for params in expand_grid(grid):
        oof = grouped_oof(name, params, X, y, groups, n_splits, train_w)
        score = auc_ap(y, oof, w_eval)[1][0]
        if best is None or score > best[0] + 1e-12:
            best = (score, params)
    return best[1], best[0]


def nested_cv(name, grid, X, y, w_eval, groups, outer=5, inner=3, train_w=None):
    """Nested grouped CV. Returns out-of-fold predictions and the configuration chosen in each outer fold."""
    oof = np.zeros(len(y)); chosen = []
    groups = np.asarray(groups)
    for a, b in GroupKFold(outer).split(X, y, groups):
        Xa, ya = X.iloc[a], y[a]
        params, _ = select_params(name, grid, Xa, ya, w_eval[a], groups[a], inner, None if train_w is None else train_w[a])
        chosen.append(params)
        m = fit_model(name, params, Xa, ya, None if train_w is None else train_w[a])
        oof[b] = predict(m, X.iloc[b])
    return oof, chosen

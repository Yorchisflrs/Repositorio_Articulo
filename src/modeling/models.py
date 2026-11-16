"""Model factories and a uniform fit/predict interface (with optional survey weights in training)."""
from __future__ import annotations

import itertools

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED

MODELS = ["LR", "RF", "XGB"]


def make_model(name: str, params: dict, seed: int = SEED):
    """LR: L2 logistic regression on standardised, median-imputed inputs. RF: random forest on
    median-imputed inputs. XGB: gradient-boosted trees (handles missing values natively)."""
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=params["C"], max_iter=2000))
    if name == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"],
                                                    max_features="sqrt", n_jobs=-1, random_state=seed))
    if name == "XGB":
        return xgb.XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                 random_state=seed, n_jobs=-1)
    raise ValueError(name)


def fit_model(name: str, params: dict, X, y, sample_weight=None, seed: int = SEED):
    """Fit; ``sample_weight`` (expansion factors) is rescaled to mean 1 so regularisation keeps its meaning."""
    m = make_model(name, params, seed)
    if sample_weight is None:
        return m.fit(X, y)
    w = np.asarray(sample_weight, dtype=float); w = w / w.mean()
    if name == "XGB":
        return m.fit(X, y, sample_weight=w)
    return m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]

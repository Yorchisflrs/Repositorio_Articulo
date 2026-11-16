"""Step 6 - E3: model updating (recalibration and retraining) and reliability curves.

A. Final-year design: every strategy is fitted on 2022-2024 and evaluated only on 2025.
B. Rolling-origin design: for each test year t the update uses data available before t
   (lagged recalibration on t-1; retraining on the last three years or on all previous years).
For t = 2020 the lagged recalibration uses out-of-fold predictions of 2019, and both retraining windows
coincide with the frozen model, so they are not repeated.

Outputs: results/metrics/E3_update_2025.csv, E3_update_2025_paired.csv, E3_rolling_origin.csv,
         results/metrics/reliability_curves.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import recalibration as rc  # noqa: E402
from src.config import CFG, FEATURES, MET_DIR, PROC_DIR, TEST_YEARS, TRAIN_YEARS  # noqa: E402
from src.evaluation.metrics import all_metrics  # noqa: E402
from src.evaluation.report import metric_row, rep_weights  # noqa: E402
from src.modeling.models import MODELS, fit_model, predict  # noqa: E402
from src.validation.bootstrap import paired_diff  # noqa: E402

d = pd.read_parquet(PROC_DIR / "analytic.parquet")
pr = pd.read_parquet(PROC_DIR / "predictions.parquet")
hp = json.load(open(MET_DIR / "hyperparameters.json"))
thr = json.load(open(MET_DIR / "thresholds.json"))["oof_weighted"]
w = d.factora07.values


def frozen_pred(model, idx):
    """Frozen-model prediction; out-of-fold inside the training years."""
    k = f"{model}_uw"; p = pr[f"p_{k}"].values.copy(); tr = d.year.isin(TRAIN_YEARS).values
    p[tr] = pr[f"oof_{k}"].values[tr]
    return p[idx]


def strategies(model, fit_idx, ev_idx, retrain_sets: dict):
    """Predictions on ev_idx for: frozen, intercept recalibration, logistic recalibration, retraining."""
    pf, pe = frozen_pred(model, fit_idx), frozen_pred(model, ev_idx)
    a = rc.fit_intercept(d.y.values[fit_idx], pf, w[fit_idx]); a2, b2 = rc.fit_logistic(d.y.values[fit_idx], pf, w[fit_idx])
    out = {"Frozen (2017-2019)": pe, "Intercept recalibration": rc.apply(pe, a), "Intercept + slope recalibration": rc.apply(pe, a2, b2)}
    for lab, idx in retrain_sets.items():
        m = fit_model(model, hp[f"{model}_uw"]["final"], d.loc[idx, FEATURES], d.y.values[idx])
        out[lab] = predict(m, d.loc[ev_idx, FEATURES])
    return out, dict(intercept=a, logistic_intercept=a2, logistic_slope=b2)


# ---- A. update on 2022-2024, evaluate on 2025
fit_years, ev_year = CFG["update"]["fit_years"], CFG["update"]["eval_year"]
fit_idx = np.where(d.year.isin(fit_years))[0]; ev_idx = np.where(d.year == ev_year)[0]
g = d.iloc[ev_idx]; W = rep_weights(g, seed_offset=TEST_YEARS.index(ev_year) + 1)
rows, paired, coefs = [], [], {}
for model in MODELS:
    preds, coefs[model] = strategies(model, fit_idx, ev_idx, {
        "Retrained on 2022-2024": fit_idx, "Retrained on 2017-2024": np.where(d.year < ev_year)[0]})
    base = all_metrics(g.y.values, preds["Frozen (2017-2019)"], W, thr[f"{model}_uw"])
    for lab, p in preds.items():
        rows.append(metric_row(g.y.values, p, W, thr[f"{model}_uw"], model=model, strategy=lab))
        if lab != "Frozen (2017-2019)":
            m = all_metrics(g.y.values, p, W, thr[f"{model}_uw"])
            for met in ("auc", "prauc", "brier", "citl", "slope"):
                r = paired_diff(m[met], base[met]); r.update(model=model, strategy=lab, metric=met); paired.append(r)
pd.DataFrame(rows).to_csv(MET_DIR / "E3_update_2025.csv", index=False)
pd.DataFrame(paired)[["model", "strategy", "metric", "diff", "lo", "hi", "se", "p_boot"]].to_csv(MET_DIR / "E3_update_2025_paired.csv", index=False)
json.dump(coefs, open(MET_DIR / "E3_recalibration_coefficients.json", "w"), indent=1)

# ---- B. rolling origin
roll = []
for t in TEST_YEARS:
    ev_idx = np.where(d.year == t)[0]; g = d.iloc[ev_idx]; W = rep_weights(g, seed_offset=TEST_YEARS.index(t) + 1)
    lag_idx = np.where(d.year == t - 1)[0]
    for model in MODELS:
        sets = {} if t == 2020 else {"Retrained, last three years": np.where(d.year.between(t - 3, t - 1))[0],
                                     "Retrained, all previous years": np.where(d.year < t)[0]}
        preds, _ = strategies(model, lag_idx, ev_idx, sets)
        names = {"Intercept recalibration": "Intercept recalibration on t-1", "Intercept + slope recalibration": "Intercept + slope recalibration on t-1"}
        for lab, p in preds.items():
            roll.append(metric_row(g.y.values, p, W, thr[f"{model}_uw"], year=t, model=model, strategy=names.get(lab, lab)))
    print("rolling", t, flush=True)
roll = pd.DataFrame(roll); roll.to_csv(MET_DIR / "E3_rolling_origin.csv", index=False)

# ---- reliability curves (weighted deciles) of the frozen models
rel = []
for lab, mask in [("CV 2017-2019", d.year.isin(TRAIN_YEARS))] + [(str(t), d.year == t) for t in TEST_YEARS]:
    idx = np.where(mask)[0]
    for model in MODELS:
        t_ = rc.reliability_table(d.y.values[idx], frozen_pred(model, idx), w[idx]); t_["test_set"] = lab; t_["model"] = model; rel.append(t_)
pd.concat(rel).to_csv(MET_DIR / "reliability_curves.csv", index=False)

a = pd.DataFrame(rows)
print(a[["model", "strategy", "auc", "auc_lo", "auc_hi", "prauc", "brier", "citl", "citl_lo", "citl_hi", "slope"]].round(3).to_string(index=False))
x = roll[roll.model == "XGB"]
print(x.pivot(index="year", columns="strategy", values="citl").round(2).to_string())
print(x.pivot(index="year", columns="strategy", values="auc").round(3).to_string())

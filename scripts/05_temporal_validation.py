"""Step 5 - E1 (in-period) and E2 (forward-in-time) evaluation with design-based intervals, plus shift diagnostics.

Inputs : data/processed/analytic.parquet, data/processed/predictions.parquet
Outputs: results/metrics/E1_E2_performance.csv      every metric, model and training-weight option, with 95% intervals
         results/metrics/E2_paired_differences.csv  paired contrasts (same replicate weights)
         results/metrics/E2_in_year_reference.csv   grouped CV inside each test year (what a fresh model reaches)
         results/metrics/E2_mode_2020.csv           2020 by interview mode
         results/metrics/shift_diagnostics.csv      prior shift, covariate shift (domain classifier, PSI)
         results/metrics/thresholds.json            flagging thresholds fixed on training years
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration.recalibration import weighted_quantile  # noqa: E402
from src.config import CFG, FEATURES, MET_DIR, PROC_DIR, TEST_YEARS, TRAIN_YEARS  # noqa: E402
from src.evaluation.metrics import all_metrics, auc_ap  # noqa: E402
from src.evaluation.report import metric_row, rep_weights  # noqa: E402
from src.modeling.models import MODELS, fit_model, predict  # noqa: E402
from src.validation.bootstrap import paired_diff  # noqa: E402
from src.validation.splits import grouped_oof  # noqa: E402

d = pd.read_parquet(PROC_DIR / "analytic.parquet")
pr = pd.read_parquet(PROC_DIR / "predictions.parquet")
hp = json.load(open(MET_DIR / "hyperparameters.json"))
KEYS = [f"{m}_{t}" for m in MODELS for t in ("uw", "w")]
tr = d.year.isin(TRAIN_YEARS).values

# flagging threshold: weighted 90th percentile of out-of-fold risk in the training years
thr = {k: weighted_quantile(pr.loc[tr, f"oof_{k}"].values, d.factora07.values[tr], CFG["flag_quantile"]) for k in KEYS}
thr_resub = {k: float(np.quantile(pr.loc[tr, f"p_{k}"].values, CFG["flag_quantile"])) for k in KEYS}   # original definition
json.dump(dict(oof_weighted=thr, resubstitution_unweighted=thr_resub), open(MET_DIR / "thresholds.json", "w"), indent=1)

rows, diffs, store = [], [], {}
sets = [("CV 2017-2019", d[tr], "oof_")] + [(str(y), d[d.year == y], "p_") for y in TEST_YEARS]
for i, (label, g, prefix) in enumerate(sets):
    W = rep_weights(g, seed_offset=i)
    yv = g.y.values
    for k in KEYS:
        p = pr.loc[g.index, prefix + k].values
        store[(label, k)] = all_metrics(yv, p, W, thr[k])
        m, wt = k.split("_")
        rows.append(metric_row(yv, p, W, thr[k], experiment="E1" if prefix == "oof_" else "E2", test_set=label, model=m,
                               training="weighted" if wt == "w" else "unweighted"))
    for a, b in [("XGB_uw", "LR_uw"), ("XGB_uw", "RF_uw"), ("RF_uw", "LR_uw"), ("LR_w", "LR_uw"), ("RF_w", "RF_uw"), ("XGB_w", "XGB_uw")]:
        for met in ("auc", "prauc", "brier", "citl", "slope", "recall"):
            r = paired_diff(store[(label, a)][met], store[(label, b)][met]); r.update(test_set=label, contrast=f"{a} - {b}", metric=met)
            diffs.append(r)
    print(label, "done", flush=True)
perf = pd.DataFrame(rows); perf.to_csv(MET_DIR / "E1_E2_performance.csv", index=False)
pd.DataFrame(diffs)[["test_set", "contrast", "metric", "diff", "lo", "hi", "se", "p_boot"]].to_csv(MET_DIR / "E2_paired_differences.csv", index=False)

# year-to-year change of the frozen XGBoost model relative to its cross-validated value is not paired (different samples);
# the in-year reference below separates "the year is harder to rank" from "the frozen model is stale".
ref = []
for y in TEST_YEARS:
    g = d[d.year == y].reset_index(drop=True); W = rep_weights(g, seed_offset=100 + y)
    for m in MODELS:
        k = f"{m}_uw"
        oof = grouped_oof(m, hp[k]["final"], g[FEATURES], g.y.values, g.conglome.values, CFG["cv"]["outer_folds"])
        fro = pr.loc[d.year == y, f"p_{k}"].values
        a_in, a_fr = all_metrics(g.y.values, oof, W), all_metrics(g.y.values, fro, W)
        for met in ("auc", "prauc"):
            r = paired_diff(a_fr[met], a_in[met])
            ref.append(dict(year=y, model=m, metric=met, in_year_cv=float(a_in[met][0]), in_year_lo=float(np.percentile(a_in[met][1:], 2.5)),
                            in_year_hi=float(np.percentile(a_in[met][1:], 97.5)), frozen=float(a_fr[met][0]),
                            frozen_minus_in_year=r["diff"], lo=r["lo"], hi=r["hi"]))
    print("in-year reference", y, flush=True)
pd.DataFrame(ref).to_csv(MET_DIR / "E2_in_year_reference.csv", index=False)

# 2020 by interview mode (telephone April-September, face-to-face September-December)
g20 = d[d.year == 2020]; mode = []
for lab, mask in (("telephone", g20.telephone == 1), ("face-to-face", g20.telephone == 0)):
    g = g20[mask]; W = rep_weights(g, seed_offset=200 + int(lab == "telephone"))
    for k in ("LR_uw", "RF_uw", "XGB_uw"):
        mode.append(metric_row(g.y.values, pr.loc[g.index, f"p_{k}"].values, W, thr[k], mode=lab, model=k.split("_")[0]))
pd.DataFrame(mode).to_csv(MET_DIR / "E2_mode_2020.csv", index=False)

# ---- shift diagnostics against the training period
def psi(a, wa, b, wb):
    """Population stability index between two weighted samples (deciles of the reference, or categories)."""
    a = np.nan_to_num(a, nan=-99.0); b = np.nan_to_num(b, nan=-99.0)
    if len(np.unique(a)) > 12:
        cuts = np.unique(np.quantile(a, np.linspace(0.1, 0.9, 9))); ia, ib = np.searchsorted(cuts, a), np.searchsorted(cuts, b)
    else:
        cats = np.unique(np.r_[a, b]); ia, ib = np.searchsorted(cats, a), np.searchsorted(cats, b)
    k = max(ia.max(), ib.max()) + 1
    pa = np.bincount(ia, weights=wa, minlength=k) / wa.sum(); pb = np.bincount(ib, weights=wb, minlength=k) / wb.sum()
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())

T = d[tr]; sh = []
dom_params = dict(max_depth=3, min_child_weight=20, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
for y in TEST_YEARS:
    g = d[d.year == y]
    both = pd.concat([T, g], ignore_index=True); z = np.r_[np.zeros(len(T)), np.ones(len(g))].astype(int)
    wz = np.r_[T.factora07.values / T.factora07.sum(), g.factora07.values / g.factora07.sum()]
    oof = grouped_oof("XGB", dom_params, both[FEATURES], z, both.conglome.values, 5)
    ps = {f: psi(T[f].values, T.factora07.values, g[f].values, g.factora07.values) for f in FEATURES}
    top = sorted(ps, key=ps.get, reverse=True)[:3]
    sh.append(dict(year=y, prevalence_train=float(np.average(T.y, weights=T.factora07)), prevalence_year=float(np.average(g.y, weights=g.factora07)),
                   prevalence_ratio=float(np.average(g.y, weights=g.factora07) / np.average(T.y, weights=T.factora07)),
                   domain_classifier_auc=float(auc_ap(z, oof, wz)[0][0]), psi_max=max(ps.values()), psi_mean=float(np.mean(list(ps.values()))),
                   psi_top3="; ".join(f"{f}={ps[f]:.3f}" for f in top), **{f"psi_{f}": v for f, v in ps.items()}))
pd.DataFrame(sh).to_csv(MET_DIR / "shift_diagnostics.csv", index=False)

show = perf[perf.training == "unweighted"]
print(show.pivot(index="test_set", columns="model", values="auc").round(3).to_string())
print(show.pivot(index="test_set", columns="model", values="citl").round(2).to_string())
print(pd.DataFrame(sh)[["year", "prevalence_ratio", "domain_classifier_auc", "psi_max", "psi_top3"]].round(3).to_string(index=False))

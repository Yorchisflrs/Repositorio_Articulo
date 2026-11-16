"""Step 9b - Prospective check on the ENAHO panel component: X(t) -> Y(t+1).

Persons are linked across consecutive years with the official person panel code (P201P, module 200),
never with dwelling codes alone. A link is valid when the code matches, sex is identical and age
increases by 0-2 years. Baseline: analytic-population adolescents aged 12-16 who ARE enrolled in year t.
Outcome: in year t+1 (interview April-December, household member, not absent) the person is not enrolled
and has not completed secondary school.

Limits: the cross-sectional files carry no longitudinal weights, so baseline expansion factors are used
and results describe the panel subsample; follow-up is restricted to dwellings kept in the panel.

Outputs: results/metrics/panel_linkage.csv, panel_followup_balance.csv, panel_prospective_performance.csv
         data/processed/panel_pairs.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES, MET_DIR, PROC_DIR, TRAIN_YEARS, YEARS  # noqa: E402
from src.evaluation.metrics import all_metrics, summarize  # noqa: E402
from src.evaluation.report import metric_row, rep_weights  # noqa: E402
from src.modeling.models import fit_model, predict  # noqa: E402
from src.validation.splits import grouped_oof  # noqa: E402

h = pd.read_parquet(PROC_DIR / "harmonized_10_18.parquet")
d = pd.read_parquet(PROC_DIR / "analytic.parquet")
pr = pd.read_parquet(PROC_DIR / "predictions.parquet")
hp = json.load(open(MET_DIR / "hyperparameters.json"))
for m in ("LR", "RF", "XGB"):      # status-model score at baseline: out-of-fold in training years, frozen afterwards
    d[f"score_{m}"] = np.where(d.year.isin(TRAIN_YEARS), pr[f"oof_{m}_uw"], pr[f"p_{m}_uw"])

link, pairs = [], []
for t in YEARS[:-1]:
    b = d[(d.year == t) & (d.y == 0) & (d.age <= 16)]
    f = h[h.year == t + 1][["p201p", "conglome", "p207", "p208a", "p204", "p205", "m300", "p301a", "p306", "mes"]].drop_duplicates("p201p", keep=False)
    m = b.merge(f, on="p201p", how="left", suffixes=("", "_f"), indicator=True)
    found = (m._merge == "both").values
    ok_id = found & (m.p207 == m.p207_f).values & (m.p208a_f - m.p208a).between(0, 2).values
    valid = ok_id & (m.m300_f == 1).values & m.p306_f.notna().values & (m.mes_f >= 4).values & (m.p204_f == 1).values & (m.p205_f != 1).values
    v = m[valid].copy(); v["y_next"] = ((v.p306_f == 2) & (v.p301a_f < 6)).astype(int); v["t"] = t
    pairs.append(v)
    link.append(dict(t=t, baseline_enrolled_12_16=len(b), code_found_next_year=int(found.sum()), share_found=float(found.mean()),
                     same_cluster=float((m.conglome == m.conglome_f)[found].mean()), sex_age_consistent=float(ok_id.sum() / max(found.sum(), 1)),
                     valid_followup=int(valid.sum()), not_enrolled_next_year=int(v.y_next.sum()), incidence=float(np.average(v.y_next, weights=v.factora07)),
                     completed_secondary_next_year=int((v.p301a_f >= 6).sum())))
    d.loc[b.index, "followed"] = valid.astype(float)
pairs = pd.concat(pairs, ignore_index=True)
pairs.to_parquet(PROC_DIR / "panel_pairs.parquet")
link = pd.DataFrame(link); link.to_csv(MET_DIR / "panel_linkage.csv", index=False)

# balance between followed and not-followed baselines (standardised differences, survey-weighted)
base = d[d.followed.notna()]
bal = []
for c in FEATURES + ["score_XGB"]:
    a, b_ = base[base.followed == 1], base[base.followed == 0]
    ma, mb = np.average(a[c].fillna(a[c].median()), weights=a.factora07), np.average(b_[c].fillna(b_[c].median()), weights=b_.factora07)
    sd = np.sqrt((a[c].var() + b_[c].var()) / 2)
    bal.append(dict(variable=c, followed=ma, not_followed=mb, std_diff=(ma - mb) / sd if sd > 0 else 0.0))
pd.DataFrame(bal).to_csv(MET_DIR / "panel_followup_balance.csv", index=False)

rows = []
periods = {"t in 2017-2019 (score out-of-fold)": pairs.t <= 2019, "t in 2020-2024 (frozen score, out of time)": pairs.t >= 2020, "All pairs 2017-2024": pairs.t >= 2017}
for pi, (plab, pm) in enumerate(periods.items()):
    g = pairs[pm].reset_index(drop=True); g["year"] = g.t; W = rep_weights(g, seed_offset=500 + pi)
    for m in ("LR", "RF", "XGB"):
        rows.append(metric_row(g.y_next.values, g[f"score_{m}"].values, W, period=plab, model=m, approach="Status score at t (no prospective training)"))
    s = summarize({k: v for k, v in all_metrics(g.y_next.values, (g.overage.values + 3) / 20, W).items() if k in ("auc", "prauc")})
    rows.append(dict(period=plab, model="Score", approach="Age-grade distortion alone", n=len(g), events=int(g.y_next.sum()), **s))
    Wu = np.vstack([np.ones(len(g)), W[1:] / g.factora07.values[None, :]])     # unweighted point estimate, same resampling
    rows.append(metric_row(g.y_next.values, g["score_XGB"].values, Wu, period=plab, model="XGB", approach="Status score at t, unweighted (sensitivity)"))

# prospective models: (a) temporal split, train on pairs with t<=2019 and test on t>=2020; (b) grouped CV over all pairs
P = pairs.copy(); P["year"] = P.t
trn, tst = P[P.t <= 2019], P[P.t >= 2020].reset_index(drop=True)
Wt = rep_weights(tst, seed_offset=510); Wall = rep_weights(P, seed_offset=511)
PSETS = {"Full": FEATURES, "Without schooling trajectory": [f for f in FEATURES if f not in ("school_years", "overage")],
         "Age and schooling trajectory only": ["age", "school_years", "overage"]}
for model, params in (("LR", {"C": 0.1}), ("XGB", dict(max_depth=2, min_child_weight=20, n_estimators=200, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0))):
    for lab, cols in PSETS.items():
        mfit = fit_model(model, params, trn[cols], trn.y_next.values)
        rows.append(metric_row(tst.y_next.values, predict(mfit, tst[cols]), Wt, period="t in 2020-2024 (frozen score, out of time)", model=model,
                               approach=f"Prospective model trained on pairs t<=2019: {lab}"))
        oof = grouped_oof(model, params, P[cols], P.y_next.values, P.conglome.values, 5)
        rows.append(metric_row(P.y_next.values, oof, Wall, period="All pairs 2017-2024", model=model, approach=f"Prospective model, grouped 5-fold CV: {lab}"))
res = pd.DataFrame(rows); res.to_csv(MET_DIR / "panel_prospective_performance.csv", index=False)
json.dump(dict(prospective_LR=dict(C=0.1), prospective_XGB="max_depth=2, 200 trees, learning rate 0.03 (fixed a priori; too few events to tune)",
               train_events=int(trn.y_next.sum()), test_events=int(tst.y_next.sum()), pairs=int(len(P)), events=int(P.y_next.sum())),
          open(MET_DIR / "panel_summary.json", "w"), indent=1)
print(link.round(3).to_string(index=False))
print(pd.DataFrame(bal).assign(a=lambda x: x.std_diff.abs()).sort_values("a", ascending=False).head(6).round(3).to_string(index=False))
print(res[["period", "model", "approach", "n", "events", "auc", "auc_lo", "auc_hi", "prauc", "prev"]].round(3).to_string(index=False))

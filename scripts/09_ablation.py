"""Step 9 - E6 ablation and the information-leakage analysis.

Part A (ablation): the model is refitted on 2017-2019 with predictor families removed (same
hyper-parameters as the full model) and evaluated on every test year, with paired differences.
Part B (leakage): age-grade distortion keeps growing while an adolescent is out of school, so part of its
predictive power may describe a spell of non-enrollment that already started. Item P303 ("enrolled last
year?") separates the two situations inside the cross-section:
   incident   = enrolled last year, not enrolled this year
   persistent = not enrolled last year, not enrolled this year
The status model is scored against each type, and an incident-only model is trained among adolescents
who were enrolled last year. P303 was not asked in the 2020 telephone interviews.

Outputs: results/metrics/E6_ablation.csv, E6_ablation_paired.csv, leakage_outcome_types.csv,
         leakage_incident_model.csv, leakage_composition.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FAM_DIGITAL, FEATURES, MET_DIR, PROC_DIR, TEST_YEARS, TRAIN_YEARS  # noqa: E402
from src.evaluation.metrics import all_metrics, summarize  # noqa: E402
from src.evaluation.report import metric_row, rep_weights  # noqa: E402
from src.modeling.models import fit_model, predict  # noqa: E402
from src.validation.bootstrap import paired_diff  # noqa: E402
from src.validation.splits import grouped_oof  # noqa: E402

d = pd.read_parquet(PROC_DIR / "analytic.parquet")
pr = pd.read_parquet(PROC_DIR / "predictions.parquet")
hp = json.load(open(MET_DIR / "hyperparameters.json"))
TRAJ = ["school_years", "overage"]; INDIV = ["age", "female", "indig_tongue", "child_of_head"] + TRAJ
SETS = {
    "Full model": FEATURES,
    "Without age-grade distortion": [f for f in FEATURES if f != "overage"],
    "Without schooling trajectory": [f for f in FEATURES if f not in TRAJ],
    "Without age and schooling trajectory": [f for f in FEATURES if f not in TRAJ + ["age"]],
    "Household and area only": [f for f in FEATURES if f not in INDIV],
    "Age and schooling trajectory only": ["age"] + TRAJ,
    "Without digital access": [f for f in FEATURES if f not in FAM_DIGITAL],
    "Full + household internet item (not comparable from 2023)": FEATURES + ["hh_internet_raw"],
    "Full + fixed internet connection": FEATURES + ["hh_internet_fixed"],
}
tr = d[d.year.isin(TRAIN_YEARS)]
Wy = {y: rep_weights(d[d.year == y], seed_offset=TEST_YEARS.index(y) + 1) for y in TEST_YEARS}

# ---- Part A
rows, paired = [], []
for model in ("XGB", "LR"):
    params = hp[f"{model}_uw"]["final"]; store = {}
    for lab, cols in SETS.items():
        m = fit_model(model, params, tr[cols], tr.y.values)
        for y in TEST_YEARS:
            g = d[d.year == y]; p = predict(m, g[cols]); store[(lab, y)] = all_metrics(g.y.values, p, Wy[y])
            rows.append(metric_row(g.y.values, p, Wy[y], model=model, predictor_set=lab, n_predictors=len(cols), year=y))
            if lab != "Full model":
                for met in ("auc", "prauc", "brier"):
                    r = paired_diff(store[(lab, y)][met], store[("Full model", y)][met]); r.update(model=model, predictor_set=lab, year=y, metric=met)
                    paired.append(r)
    # single-variable benchmark: age-grade distortion used directly as a score
    if model == "XGB":
        for y in TEST_YEARS:
            g = d[d.year == y]; s = summarize({k: v for k, v in all_metrics(g.y.values, (g.overage.values + 3) / 20, Wy[y]).items() if k in ("auc", "prauc")})
            rows.append(dict(model="Score", predictor_set="Age-grade distortion alone (no model)", n_predictors=1, year=y, n=len(g), events=int(g.y.sum()), **s))
    print("ablation", model, flush=True)
abl = pd.DataFrame(rows); abl.to_csv(MET_DIR / "E6_ablation.csv", index=False)
pd.DataFrame(paired)[["model", "predictor_set", "year", "metric", "diff", "lo", "hi", "se", "p_boot"]].to_csv(MET_DIR / "E6_ablation_paired.csv", index=False)

# ---- Part B: composition of the outcome and performance by type of non-enrollment
d["kind"] = np.select([(d.y == 1) & (d.p303 == 1), (d.y == 1) & (d.p303 == 2), (d.y == 1) & d.p303.isna()], ["incident", "persistent", "unknown"], "enrolled")
comp = d.groupby("year").apply(lambda g: pd.Series(dict(
    n=len(g), non_enrolled=int(g.y.sum()), incident=int((g.kind == "incident").sum()), persistent=int((g.kind == "persistent").sum()),
    p303_missing=int(g.p303.isna().sum()),
    weighted_share_persistent_among_known=float(g.factora07[g.kind == "persistent"].sum() / max(g.factora07[g.kind.isin(["incident", "persistent"])].sum(), 1e-9)),
    incidence_among_enrolled_last_year=float(np.average(g.y[g.p303 == 1], weights=g.factora07[g.p303 == 1])),
    median_overage_incident=float(g.overage[g.kind == "incident"].median()), median_overage_persistent=float(g.overage[g.kind == "persistent"].median()),
    median_overage_enrolled=float(g.overage[g.kind == "enrolled"].median())))).reset_index()
comp.to_csv(MET_DIR / "leakage_composition.csv", index=False)

types = []
known = d.p303.notna()
periods = {"CV 2017-2019": d.year.isin(TRAIN_YEARS), "2021": d.year == 2021, "2022-2025": d.year.between(2022, 2025), "2021-2025": d.year.between(2021, 2025)}
for pi, (plab, pm) in enumerate(periods.items()):
    g = d[pm & known]; W = rep_weights(g, seed_offset=400 + pi)
    for model in ("LR", "RF", "XGB"):
        p = np.where(d.year.isin(TRAIN_YEARS), pr[f"oof_{model}_uw"], pr[f"p_{model}_uw"])[(pm & known).values]
        for tlab, keep in (("All non-enrolled vs enrolled", g.kind.isin(["incident", "persistent", "enrolled"])),
                           ("Persistent vs enrolled", g.kind.isin(["persistent", "enrolled"])),
                           ("Incident vs enrolled", g.kind.isin(["incident", "enrolled"]))):
            mk = keep.values
            types.append(metric_row(g.y.values[mk], p[mk], W[:, mk], period=plab, model=model, contrast=tlab))
pd.DataFrame(types).to_csv(MET_DIR / "leakage_outcome_types.csv", index=False)

# ---- incident-only models: population = enrolled last year (P303 = 1); outcome = not enrolled this year
inc = d[d.p303 == 1].reset_index(drop=True); itr = inc[inc.year.isin(TRAIN_YEARS)]
irows = []
ISETS = {k: SETS[k] for k in ("Full model", "Without schooling trajectory", "Without age and schooling trajectory", "Age and schooling trajectory only", "Household and area only")}
for model in ("XGB", "LR"):
    params = hp[f"{model}_uw"]["final"]
    for lab, cols in ISETS.items():
        oof = grouped_oof(model, params, itr[cols], itr.y.values, itr.conglome.values, 5)
        Wt = rep_weights(itr, seed_offset=450)
        irows.append(metric_row(itr.y.values, oof, Wt, model=model, predictor_set=lab, test_set="CV 2017-2019", trained_on="incident outcome"))
        m = fit_model(model, params, itr[cols], itr.y.values)
        for plab, pm in (("2021", inc.year == 2021), ("2022-2025", inc.year.between(2022, 2025)), ("2021-2025", inc.year.between(2021, 2025))):
            g = inc[pm]; W = rep_weights(g, seed_offset=451 + len(plab))
            irows.append(metric_row(g.y.values, predict(m, g[cols]), W, model=model, predictor_set=lab, test_set=plab, trained_on="incident outcome"))
    print("incident", model, flush=True)
pd.DataFrame(irows).to_csv(MET_DIR / "leakage_incident_model.csv", index=False)

x = abl[abl.model == "XGB"].pivot(index="predictor_set", columns="year", values="auc").round(3)
print(x.to_string()); print(abl[abl.model == "Score"].pivot(index="predictor_set", columns="year", values="auc").round(3).to_string())
print(comp.round(3).to_string(index=False))
t = pd.DataFrame(types); print(t[t.model == "XGB"][["period", "contrast", "n", "events", "auc", "auc_lo", "auc_hi", "prauc", "prev"]].round(3).to_string(index=False))
i = pd.DataFrame(irows); print(i[["model", "predictor_set", "test_set", "n", "events", "auc", "auc_lo", "auc_hi", "prauc", "prev", "citl"]].round(3).to_string(index=False))

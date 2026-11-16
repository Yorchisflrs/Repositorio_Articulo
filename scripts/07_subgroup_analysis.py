"""Step 7 - E4: subgroup reliability of the frozen models, with design-based intervals.

Groups: area, poverty, mother tongue, sex, and their two-way intersections when the pooled test sample has
at least 50 non-enrolled adolescents (smaller cells are listed but not evaluated).
Three periods are audited: in-period (out-of-fold, 2017-2019), pandemic (2020-2021) and post-pandemic (2022-2025).
A fairness correction is also measured, not assumed: an intercept update (global or specific to
area x mother-tongue cells) fitted on 2022-2023 and evaluated on 2024-2025.

Outputs: results/metrics/E4_subgroups.csv, E4_subgroup_gaps.csv, E4_small_cells.csv, E4_group_recalibration.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import recalibration as rc  # noqa: E402
from src.config import CFG, MET_DIR, PROC_DIR, TRAIN_YEARS  # noqa: E402
from src.evaluation.metrics import all_metrics  # noqa: E402
from src.evaluation.report import metric_row, rep_weights  # noqa: E402
from src.modeling.models import MODELS  # noqa: E402
from src.validation.bootstrap import paired_diff  # noqa: E402

MIN_EVENTS = 50
d = pd.read_parquet(PROC_DIR / "analytic.parquet")
pr = pd.read_parquet(PROC_DIR / "predictions.parquet")
thr = json.load(open(MET_DIR / "thresholds.json"))["oof_weighted"]
tr = d.year.isin(TRAIN_YEARS).values
P = {m: np.where(tr, pr[f"oof_{m}_uw"].values, pr[f"p_{m}_uw"].values) for m in MODELS}   # OOF in training years

base = {"Urban": d.rural == 0, "Rural": d.rural == 1, "Non-poor": d.poor == 0, "Poor": d.poor == 1,
        "Spanish": d.indig_tongue == 0, "Indigenous tongue": d.indig_tongue == 1, "Male": d.female == 0, "Female": d.female == 1}
pairs = [("Urban", "Rural"), ("Non-poor", "Poor"), ("Spanish", "Indigenous tongue"), ("Male", "Female")]
groups = {"All": pd.Series(True, index=d.index), **base}
dims = [("Urban", "Rural"), ("Non-poor", "Poor"), ("Spanish", "Indigenous tongue"), ("Male", "Female")]
for i in range(len(dims)):
    for j in range(i + 1, len(dims)):
        for a in dims[i]:
            for b in dims[j]:
                groups[f"{a} x {b}"] = base[a] & base[b]

periods = {"In-period CV 2017-2019": d.year.isin(TRAIN_YEARS), "Pandemic 2020-2021": d.year.isin([2020, 2021]),
           "Post-pandemic 2022-2025": d.year.isin(CFG["subgroup_pool"])}
rows, gaps, small = [], [], []
for pi, (plab, pmask) in enumerate(periods.items()):
    g = d[pmask]; W = rep_weights(g, seed_offset=300 + pi); pos = {m: P[m][pmask.values] for m in MODELS}
    store = {}
    for glab, gm in groups.items():
        mk = gm[pmask].values; ev = int(g.y.values[mk].sum())
        if ev < MIN_EVENTS:
            small.append(dict(period=plab, group=glab, n=int(mk.sum()), events=ev)); continue
        for m in MODELS:
            store[(glab, m)] = all_metrics(g.y.values[mk], pos[m][mk], W[:, mk], thr[f"{m}_uw"])
            rows.append(metric_row(g.y.values[mk], pos[m][mk], W[:, mk], thr[f"{m}_uw"], period=plab, group=glab, model=m))
    for a, b in pairs:
        for m in MODELS:
            for met in ("auc", "prauc", "citl", "slope", "recall", "precision", "flag_rate", "oe"):
                r = paired_diff(store[(b, m)][met], store[(a, m)][met]); r.update(period=plab, contrast=f"{b} - {a}", model=m, metric=met)
                gaps.append(r)
    print(plab, "done", flush=True)
sub = pd.DataFrame(rows); sub.to_csv(MET_DIR / "E4_subgroups.csv", index=False)
pd.DataFrame(gaps)[["period", "contrast", "model", "metric", "diff", "lo", "hi", "se", "p_boot"]].to_csv(MET_DIR / "E4_subgroup_gaps.csv", index=False)
pd.DataFrame(small).to_csv(MET_DIR / "E4_small_cells.csv", index=False)

# ---- measured effect of an intercept correction (fit 2022-2023, evaluate 2024-2025)
fit = d.year.isin([2022, 2023]).values; ev = d.year.isin([2024, 2025]).values
cell = np.where(d.rural == 1, "Rural", "Urban") + " / " + np.where(d.indig_tongue == 1, "Indigenous", "Spanish")
g = d[ev]; W = rep_weights(g, seed_offset=350); rec = []
for m in MODELS:
    p = P[m]; a_glob = rc.fit_intercept(d.y.values[fit], p[fit], d.factora07.values[fit])
    a_cell = {c: rc.fit_intercept(d.y.values[fit & (cell == c)], p[fit & (cell == c)], d.factora07.values[fit & (cell == c)]) for c in np.unique(cell)}
    variants = {"Frozen": p[ev], "Global intercept": rc.apply(p[ev], a_glob),
                "Cell-specific intercept": rc.apply(p[ev], np.array([a_cell[c] for c in cell[ev]]))}
    for vlab, pv in variants.items():
        for glab in ["All", "Urban", "Rural", "Spanish", "Indigenous tongue"] + list(np.unique(cell)):
            mk = (cell[ev] == glab) if " / " in glab else groups[glab][ev].values
            if g.y.values[mk].sum() < 20:
                continue
            rec.append(metric_row(g.y.values[mk], pv[mk], W[:, mk], thr[f"{m}_uw"], model=m, variant=vlab, group=glab,
                                  intercept_global=a_glob, **{f"intercept_{k}": v for k, v in a_cell.items()}))
pd.DataFrame(rec).to_csv(MET_DIR / "E4_group_recalibration.csv", index=False)

s = sub[(sub.model == "XGB") & ~sub.group.str.contains(" x ")]
for plab in periods:
    print(plab); print(s[s.period == plab][["group", "n", "events", "prev", "mean_pred", "auc", "citl", "citl_lo", "citl_hi", "slope", "recall", "recall_lo", "recall_hi", "precision"]].round(3).to_string(index=False))
print(pd.DataFrame(small).to_string(index=False)[:600])

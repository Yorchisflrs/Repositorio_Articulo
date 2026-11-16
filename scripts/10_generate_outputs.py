"""Step 10 - tables, figures and manuscript numbers, all derived from the result files.

Nothing is computed from microdata here: the script only reads results/metrics and results/tables and
writes
  paper/generated/numbers.tex            macros with every number quoted in the manuscript
  paper/generated/table_*.tex            LaTeX tables included by the manuscript
  results/tables/T*_*.csv                the same tables as CSV
  results/tables/manuscript_numbers.csv  key, value, source file and row selector of every number
  results/figures/figure_0*.{pdf,png}    publication figures (600 dpi PNG and vector PDF)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURE_LABELS, FIG_DIR, MET_DIR, ROOT, TAB_DIR, TEST_YEARS, TRAIN_YEARS  # noqa: E402
from src.evaluation import outputs as o  # noqa: E402
from src.evaluation.outputs import ci, f, fs, pct, thousands  # noqa: E402

GEN = ROOT / "paper" / "generated"
R = o.Registry()
MODELS = ["LR", "RF", "XGB"]
SETS = ["CV 2017-2019"] + [str(y) for y in TEST_YEARS]
SETKEY = {"CV 2017-2019": "cv", **{str(y): str(y) for y in TEST_YEARS}}


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(MET_DIR / name)


def one(df: pd.DataFrame, **sel) -> pd.Series:
    m = np.ones(len(df), dtype=bool)
    for k, v in sel.items():
        m &= (df[k].astype(str) == str(v)).values
    if m.sum() != 1:
        raise ValueError(f"selector {sel} matched {m.sum()} rows")
    return df[m].iloc[0]


def sel_str(**sel) -> str:
    return "; ".join(f"{k}={v}" for k, v in sel.items())


def reg_metrics(prefix: str, row: pd.Series, src: str, selector: str, metrics, d: int = 3) -> None:
    """Register point estimates and intervals of the listed metrics under prefix.metric / prefix.metric.ci."""
    for met in metrics:
        if met not in row or pd.isna(row[met]):
            continue
        R.add(f"{prefix}.{met}", f(row[met], d), src, f"{selector}; column={met}")
        if met in ("prev", "flag_rate"):
            R.add(f"{prefix}.{met}.pct", pct(row[met], 1), src, f"{selector}; column={met}, in %")
        if met in ("oe", "citl", "slope", "recall", "precision"):
            R.add(f"{prefix}.{met}.2", f(row[met], 2), src, f"{selector}; column={met}, two decimals")
            if f"{met}_lo" in row and not pd.isna(row[f"{met}_lo"]):
                R.add(f"{prefix}.{met}.ci.2", ci(row[f"{met}_lo"], row[f"{met}_hi"], 2), src, f"{selector}; columns={met}_lo,{met}_hi, two decimals")
        if f"{met}_lo" in row and not pd.isna(row[f"{met}_lo"]):
            R.add(f"{prefix}.{met}.ci", ci(row[f"{met}_lo"], row[f"{met}_hi"], d), src, f"{selector}; columns={met}_lo,{met}_hi")


MET = ["auc", "prauc", "brier", "citl", "slope", "recall", "precision", "specificity", "flag_rate", "oe", "prev", "mean_pred"]

# =============================================================================================== data
flow = pd.read_csv(TAB_DIR / "sample_flow.csv")
prev = pd.read_csv(TAB_DIR / "prevalence_by_year.csv")
comp = rd("leakage_composition.csv")
val = json.load(open(MET_DIR / "data_validation.json"))
month = pd.read_csv(TAB_DIR / "prevalence_by_month.csv")
design = pd.read_csv(TAB_DIR / "design_structure.csv")
overlap = pd.read_csv(TAB_DIR / "panel_overlap.csv")
inet = pd.read_csv(TAB_DIR / "internet_item_by_year.csv")

R.add("n.total", thousands(val["analytic_n"]), "results/metrics/data_validation.json", "analytic_n")
R.add("ev.total", thousands(val["analytic_non_enrolled"]), "results/metrics/data_validation.json", "analytic_non_enrolled")
R.add("n.roster", thousands(flow.total.iloc[0]), "results/tables/sample_flow.csv", "row 1; column=total")
R.add("n.train", thousands(comp[comp.year.isin(TRAIN_YEARS)].n.sum()), "results/metrics/leakage_composition.csv", "years 2017-2019; sum of n")
R.add("ev.train", thousands(comp[comp.year.isin(TRAIN_YEARS)].non_enrolled.sum()), "results/metrics/leakage_composition.csv", "years 2017-2019; sum of non_enrolled")
R.add("n.test", thousands(comp[comp.year.isin(TEST_YEARS)].n.sum()), "results/metrics/leakage_composition.csv", "years 2020-2025; sum of n")
R.add("ev.test", thousands(comp[comp.year.isin(TEST_YEARS)].non_enrolled.sum()), "results/metrics/leakage_composition.csv", "years 2020-2025; sum of non_enrolled")
R.add("ev.test.min", thousands(comp[comp.year.isin(TEST_YEARS)].non_enrolled.min()), "results/metrics/leakage_composition.csv", "years 2020-2025; min of non_enrolled")
R.add("ev.test.max", thousands(comp[comp.year.isin(TEST_YEARS)].non_enrolled.max()), "results/metrics/leakage_composition.csv", "years 2020-2025; max of non_enrolled")
for mth, lab in ((1, "jan"), (2, "feb"), (3, "mar")):
    R.add(f"month.{lab}", f(one(month, mes=mth).pct_not_enrolled, 1), "results/tables/prevalence_by_month.csv", f"mes={mth}")
R.add("month.aprdec.min", f(month[month.mes >= 4].pct_not_enrolled.min(), 1), "results/tables/prevalence_by_month.csv", "mes>=4; min")
R.add("month.aprdec.max", f(month[month.mes >= 4].pct_not_enrolled.max(), 1), "results/tables/prevalence_by_month.csv", "mes>=4; max")
known = comp.incident.sum() + comp.persistent.sum()
R.add("p303.known", thousands(known), "results/metrics/leakage_composition.csv", "sum of incident + persistent")
R.add("p303.incident", thousands(comp.incident.sum()), "results/metrics/leakage_composition.csv", "sum of incident")
R.add("p303.persistent", thousands(comp.persistent.sum()), "results/metrics/leakage_composition.csv", "sum of persistent")
R.add("p303.incident.pct", f(100 * comp.incident.sum() / known, 0), "results/metrics/leakage_composition.csv", "incident / (incident + persistent), unweighted")
R.add("p303.persistent.pct", f(100 * comp.persistent.sum() / known, 0), "results/metrics/leakage_composition.csv", "persistent / (incident + persistent), unweighted")
x_ = comp[comp.year != 2020].weighted_share_persistent_among_known
R.add("p303.wshare.min", f(100 * x_.min(), 0), "results/metrics/leakage_composition.csv", "years except 2020; min of weighted_share_persistent_among_known, in %")
R.add("p303.wshare.max", f(100 * x_.max(), 0), "results/metrics/leakage_composition.csv", "years except 2020; max of weighted_share_persistent_among_known, in %")
R.add("p303.missing2020", thousands(one(comp, year=2020).p303_missing), "results/metrics/leakage_composition.csv", "year=2020; column=p303_missing")
R.add("incidence.min", f(100 * comp.incidence_among_enrolled_last_year[comp.year != 2020].min(), 1), "results/metrics/leakage_composition.csv", "years except 2020; min of incidence_among_enrolled_last_year")
R.add("incidence.max", f(100 * comp.incidence_among_enrolled_last_year[comp.year != 2020].max(), 1), "results/metrics/leakage_composition.csv", "years except 2020; max of incidence_among_enrolled_last_year")
R.add("design.psu.min", thousands(design.psu.min()), "results/tables/design_structure.csv", "min of psu")
R.add("design.psu.max", thousands(design.psu.max()), "results/tables/design_structure.csv", "max of psu")
R.add("design.deff.min", f(design.kish_deff_weights.min(), 1), "results/tables/design_structure.csv", "min of kish_deff_weights")
R.add("design.deff.max", f(design.kish_deff_weights.max(), 1), "results/tables/design_structure.csv", "max of kish_deff_weights")
for y in (2020, 2021, 2022, 2023):
    r = one(overlap, year=y)
    R.add(f"overlap.{y}.cluster", f(100 * r.share_cluster_in_training, 0), "results/tables/panel_overlap.csv", f"year={y}; column=share_cluster_in_training")
    R.add(f"overlap.{y}.person", f(100 * r.share_person_panel_code_in_training, 0), "results/tables/panel_overlap.csv", f"year={y}; column=share_person_panel_code_in_training")
for y in (2022, 2023):
    R.add(f"internet.{y}", f(one(inet, year=y).pct_internet_item_P1144, 1), "results/tables/internet_item_by_year.csv", f"year={y}; column=pct_internet_item_P1144")
R.add("missing.max", f(100 * val["max_missing_share_any_feature_year"], 1), "results/metrics/data_validation.json", "max_missing_share_any_feature_year")
R.add("internet.fixed.2023", f(one(inet, year=2023).pct_fixed_connection, 1), "results/tables/internet_item_by_year.csv", "year=2023; column=pct_fixed_connection")

nat = prev[prev.area == "National"]
for _, r in nat.iterrows():
    y = int(r.year)
    R.add(f"prev.{y}", pct(r.prev, 2), "results/tables/prevalence_by_year.csv", f"year={y}; area=National; column=prev")
    R.add(f"prev.{y}.ci", f"{pct(r.lo_boot, 2)}, {pct(r.hi_boot, 2)}", "results/tables/prevalence_by_year.csv", f"year={y}; area=National; columns=lo_boot,hi_boot")
for y in (2019, 2021, 2025):
    for area in ("Urban", "Rural"):
        r = one(prev, year=y, area=area)
        R.add(f"prev.{y}.{area.lower()}", pct(r.prev, 1), "results/tables/prevalence_by_year.csv", f"year={y}; area={area}; column=prev")

# ---- Table I: sample and outcome by year
t1 = nat[["year", "n", "non_enrolled", "prev", "lo_boot", "hi_boot"]].merge(
    comp[["year", "incident", "persistent", "weighted_share_persistent_among_known"]], on="year")
t1.to_csv(TAB_DIR / "T1_sample_by_year.csv", index=False)
body = [r"\begin{tabular}{@{}llrrcr@{}}", r"\toprule",
        r" & Year & $n$ & Events & Prevalence, \% [95\% CI] & Prior$^{\mathrm{a}}$ \\", r"\midrule"]
for i, r in t1.iterrows():
    y = int(r.year)
    role = "Train" if y == TRAIN_YEARS[0] else ("Test" if y == TEST_YEARS[0] else "")
    prior = f(100 * r.weighted_share_persistent_among_known, 0) + ("$^{\\mathrm{b}}$" if y == 2020 else "")
    body.append(f"{role} & {y} & {thousands(r.n)} & {int(r.non_enrolled)} & {pct(r.prev, 2)} [{pct(r.lo_boot, 2)}, {pct(r.hi_boot, 2)}] & {prior} \\\\")
    if y == TRAIN_YEARS[-1]:
        body.append(r"\midrule")
body += [r"\bottomrule", r"\end{tabular}"]
o.write_table(GEN / "table_sample.tex", body)

# =============================================================================================== E1 / E2
perf = rd("E1_E2_performance.csv")
pu = perf[perf.training == "unweighted"]
for m in MODELS:
    for s in SETS:
        r = one(pu, model=m, test_set=s)
        reg_metrics(f"perf.{m}.{SETKEY[s]}", r, "results/metrics/E1_E2_performance.csv", sel_str(training="unweighted", model=m, test_set=s), MET)
        rw = one(perf[perf.training == "weighted"], model=m, test_set=s)
        reg_metrics(f"perfw.{m}.{SETKEY[s]}", rw, "results/metrics/E1_E2_performance.csv", sel_str(training="weighted", model=m, test_set=s), ["auc", "prauc", "citl"])

cols = ["test_set", "model", "n", "events"] + [c for met in ("auc", "prauc", "brier", "citl", "slope", "recall", "precision", "flag_rate") for c in (met, met + "_lo", met + "_hi")]
pu[cols].to_csv(TAB_DIR / "T2_performance_by_year.csv", index=False)
body = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
        r"Evaluation set & Model & ROC-AUC [95\% CI] & PR-AUC [95\% CI] & Brier & CITL [95\% CI] & Slope [95\% CI] & Recall & Precision \\", r"\midrule"]
for s in SETS:
    for i, m in enumerate(MODELS):
        r = one(pu, model=m, test_set=s)
        lab = ("In-period, nested CV" if s.startswith("CV") else s) if i == 0 else ""
        body.append(f"{lab} & {m} & {f(r.auc)} [{ci(r.auc_lo, r.auc_hi)}] & {f(r.prauc)} [{ci(r.prauc_lo, r.prauc_hi)}] & {f(r.brier)} & "
                    f"{f(r.citl, 2)} [{ci(r.citl_lo, r.citl_hi, 2)}] & {f(r.slope, 2)} [{ci(r.slope_lo, r.slope_hi, 2)}] & {f(r.recall, 2)} & {f(r.precision, 2)} \\\\")
    if s != SETS[-1]:
        body.append(r"\addlinespace[1pt]")
body += [r"\bottomrule", r"\end{tabular}"]
o.write_table(GEN / "table_performance.tex", body)

pd_ = rd("E2_paired_differences.csv")
for contrast, key in (("XGB_uw - LR_uw", "xgb-lr"), ("XGB_uw - RF_uw", "xgb-rf"), ("XGB_w - XGB_uw", "xgbw-xgb"), ("LR_w - LR_uw", "lrw-lr"), ("RF_w - RF_uw", "rfw-rf")):
    sub = pd_[pd_.contrast == contrast]
    if sub.empty:
        continue
    for met in ("auc", "prauc", "citl"):
        x = sub[sub.metric == met]
        R.add(f"diff.{key}.{met}.min", fs(x["diff"].min()), "results/metrics/E2_paired_differences.csv", f"contrast={contrast}; metric={met}; min over evaluation sets")
        R.add(f"diff.{key}.{met}.max", fs(x["diff"].max()), "results/metrics/E2_paired_differences.csv", f"contrast={contrast}; metric={met}; max over evaluation sets")
        R.add(f"diff.{key}.{met}.absmax", f(x["diff"].abs().max()), "results/metrics/E2_paired_differences.csv", f"contrast={contrast}; metric={met}; max |diff| over evaluation sets")
        R.add(f"diff.{key}.{met}.nsig", str(int(((x.lo > 0) | (x.hi < 0)).sum())), "results/metrics/E2_paired_differences.csv", f"contrast={contrast}; metric={met}; sets whose interval excludes 0 (of {len(x)})")
        for s in SETS:
            r = one(x, test_set=s)
            R.add(f"diff.{key}.{met}.{SETKEY[s]}", fs(r["diff"]), "results/metrics/E2_paired_differences.csv", f"contrast={contrast}; metric={met}; test_set={s}")
            R.add(f"diff.{key}.{met}.{SETKEY[s]}.ci", ci(r.lo, r.hi), "results/metrics/E2_paired_differences.csv", f"contrast={contrast}; metric={met}; test_set={s}; columns=lo,hi")

iny = rd("E2_in_year_reference.csv")
for m in MODELS:
    for y in TEST_YEARS:
        r = one(iny, model=m, year=y, metric="auc")
        src, s = "results/metrics/E2_in_year_reference.csv", sel_str(model=m, year=y, metric="auc")
        R.add(f"inyear.{m}.{y}.auc", f(r.in_year_cv), src, s + "; column=in_year_cv")
        R.add(f"inyear.{m}.{y}.diff", fs(r.frozen_minus_in_year), src, s + "; column=frozen_minus_in_year")
        R.add(f"inyear.{m}.{y}.diff.ci", ci(r.lo, r.hi), src, s + "; columns=lo,hi")
x = iny[(iny.metric == "auc")]
R.add("inyear.absdiff.max", f(x.frozen_minus_in_year.abs().max()), "results/metrics/E2_in_year_reference.csv", "metric=auc; max |frozen_minus_in_year| over models and years")

mode = rd("E2_mode_2020.csv")
for md, key in (("telephone", "tel"), ("face-to-face", "f2f")):
    r = one(mode, mode=md, model="XGB")
    reg_metrics(f"mode.{key}", r, "results/metrics/E2_mode_2020.csv", sel_str(mode=md, model="XGB"), ["auc", "citl"])
    R.add(f"mode.{key}.n", thousands(r.n), "results/metrics/E2_mode_2020.csv", sel_str(mode=md, model="XGB") + "; column=n")

shift = rd("shift_diagnostics.csv")
for y in TEST_YEARS:
    r = one(shift, year=y)
    R.add(f"shift.{y}.prevratio", f(r.prevalence_ratio, 2), "results/metrics/shift_diagnostics.csv", f"year={y}; column=prevalence_ratio")
    R.add(f"shift.{y}.domauc", f(r.domain_classifier_auc, 2), "results/metrics/shift_diagnostics.csv", f"year={y}; column=domain_classifier_auc")
    R.add(f"shift.{y}.psimax", f(r.psi_max, 2), "results/metrics/shift_diagnostics.csv", f"year={y}; column=psi_max")
R.add("shift.domauc.min", f(shift.domain_classifier_auc.min(), 2), "results/metrics/shift_diagnostics.csv", "min of domain_classifier_auc")
R.add("shift.domauc.max", f(shift.domain_classifier_auc.max(), 2), "results/metrics/shift_diagnostics.csv", "max of domain_classifier_auc")
R.add("shift.psi.max", f(shift.psi_max.max(), 2), "results/metrics/shift_diagnostics.csv", "max of psi_max")

thr = json.load(open(MET_DIR / "thresholds.json"))
for m in MODELS:
    R.add(f"thr.{m}.oof", f(thr["oof_weighted"][f"{m}_uw"]), "results/metrics/thresholds.json", f"oof_weighted; {m}_uw")
    R.add(f"thr.{m}.resub", f(thr["resubstitution_unweighted"][f"{m}_uw"]), "results/metrics/thresholds.json", f"resubstitution_unweighted; {m}_uw")
hp = json.load(open(MET_DIR / "hyperparameters.json"))
R.add("hp.LR.C", str(hp["LR_uw"]["final"]["C"]), "results/metrics/hyperparameters.json", "LR_uw; final; C")
R.add("hp.RF.leaf", str(hp["RF_uw"]["final"]["min_samples_leaf"]), "results/metrics/hyperparameters.json", "RF_uw; final; min_samples_leaf")
R.add("hp.RF.trees", str(hp["RF_uw"]["final"]["n_estimators"]), "results/metrics/hyperparameters.json", "RF_uw; final; n_estimators")
R.add("hp.XGB.depth", str(hp["XGB_uw"]["final"]["max_depth"]), "results/metrics/hyperparameters.json", "XGB_uw; final; max_depth")
R.add("hp.XGB.trees", str(hp["XGB_uw"]["final"]["n_estimators"]), "results/metrics/hyperparameters.json", "XGB_uw; final; n_estimators")
R.add("hp.XGB.mcw", str(hp["XGB_uw"]["final"]["min_child_weight"]), "results/metrics/hyperparameters.json", "XGB_uw; final; min_child_weight")

# =============================================================================================== E3
upd = rd("E3_update_2025.csv"); updp = rd("E3_update_2025_paired.csv")
UKEY = {"Frozen (2017-2019)": "frozen", "Intercept recalibration": "int", "Intercept + slope recalibration": "intslope",
        "Retrained on 2022-2024": "re3", "Retrained on 2017-2024": "reall"}
for m in MODELS:
    for s, k in UKEY.items():
        r = one(upd, model=m, strategy=s)
        reg_metrics(f"upd.{m}.{k}", r, "results/metrics/E3_update_2025.csv", sel_str(model=m, strategy=s), ["auc", "prauc", "brier", "citl", "slope", "recall", "precision"])
        if k != "frozen":
            for met in ("auc", "prauc", "brier", "citl"):
                q = one(updp, model=m, strategy=s, metric=met)
                d = 4 if met == "brier" else 3
                R.add(f"upd.{m}.{k}.d{met}", fs(q["diff"], d), "results/metrics/E3_update_2025_paired.csv", sel_str(model=m, strategy=s, metric=met) + "; column=diff")
                R.add(f"upd.{m}.{k}.d{met}.ci", ci(q.lo, q.hi, d), "results/metrics/E3_update_2025_paired.csv", sel_str(model=m, strategy=s, metric=met) + "; columns=lo,hi")
roll = rd("E3_rolling_origin.csv")
RKEY = {"Frozen (2017-2019)": "frozen", "Intercept recalibration on t-1": "int", "Intercept + slope recalibration on t-1": "intslope",
        "Retrained, last three years": "re3", "Retrained, all previous years": "reall"}
for m in MODELS:
    for s, k in RKEY.items():
        for y in TEST_YEARS:
            x = roll[(roll.model == m) & (roll.strategy == s) & (roll.year == y)]
            if len(x) == 1 and not pd.isna(x.iloc[0].auc):
                reg_metrics(f"roll.{m}.{k}.{y}", x.iloc[0], "results/metrics/E3_rolling_origin.csv", sel_str(model=m, strategy=s, year=y), ["auc", "citl"])
x = roll[(roll.model == "XGB") & roll.year.between(2022, 2025)]
for s, k in RKEY.items():
    R.add(f"roll.XGB.{k}.mae2225", f(x[x.strategy == s].citl.abs().mean(), 2), "results/metrics/E3_rolling_origin.csv", f"model=XGB; strategy={s}; mean |citl| over 2022-2025")
x = roll[(roll.model == "XGB") & (roll.year >= 2021)]
for s, k in RKEY.items():
    R.add(f"roll.XGB.{k}.mae2125", f(x[x.strategy == s].citl.abs().mean(), 2), "results/metrics/E3_rolling_origin.csv", f"model=XGB; strategy={s}; mean |citl| over 2021-2025")
a = roll[(roll.model == "XGB") & (roll.strategy == "Retrained, all previous years") & (roll.year >= 2021)].set_index("year").auc
b = roll[(roll.model == "XGB") & (roll.strategy == "Frozen (2017-2019)") & (roll.year >= 2021)].set_index("year").auc
R.add("roll.XGB.reall.dauc.max", fs((a - b).max()), "results/metrics/E3_rolling_origin.csv", "model=XGB; retrained all previous minus frozen; max AUC difference 2021-2025")
R.add("roll.XGB.reall.dauc.min", fs((a - b).min()), "results/metrics/E3_rolling_origin.csv", "model=XGB; retrained all previous minus frozen; min AUC difference 2021-2025")

# ---- Table III: updating strategies evaluated on 2025 (XGBoost)
t3 = upd[upd.model == "XGB"][["strategy", "auc", "auc_lo", "auc_hi", "prauc", "brier", "citl", "citl_lo", "citl_hi", "slope", "slope_lo", "slope_hi"]]
t3 = t3.merge(updp[(updp.model == "XGB") & (updp.metric == "auc")][["strategy", "diff", "lo", "hi"]].rename(columns={"diff": "dauc", "lo": "dauc_lo", "hi": "dauc_hi"}), on="strategy", how="left")
t3.to_csv(TAB_DIR / "T3_updating_2025.csv", index=False)
ULAB = {"Frozen (2017-2019)": "Frozen", "Intercept recalibration": "Intercept update", "Intercept + slope recalibration": "Intercept and slope",
        "Retrained on 2022-2024": "Retrained, 2022--24", "Retrained on 2017-2024": "Retrained, 2017--24"}
body = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Strategy & ROC-AUC & $\Delta$AUC [95\% CI] & CITL [95\% CI] \\", r"\midrule"]
for _, r in t3.iterrows():
    dd = "--" if r.strategy.startswith("Frozen") or r.strategy.startswith("Intercept") else f"{fs(r.dauc)} [{ci(r.dauc_lo, r.dauc_hi)}]"
    body.append(f"{ULAB[r.strategy]} & {f(r.auc)} & {dd} & {f(r.citl, 2)} [{ci(r.citl_lo, r.citl_hi, 2)}] \\\\")
body += [r"\bottomrule", r"\end{tabular}"]
o.write_table(GEN / "table_updating.tex", body)

# =============================================================================================== E6: ablation and outcome validity
abl = rd("E6_ablation.csv"); ablp = rd("E6_ablation_paired.csv")
AKEY = {"Full model": "full", "Without age-grade distortion": "noover", "Without schooling trajectory": "notraj",
        "Without age and schooling trajectory": "noagetraj", "Household and area only": "hharea", "Age and schooling trajectory only": "agetraj",
        "Without digital access": "nodigital", "Full + household internet item (not comparable from 2023)": "plusinet", "Full + fixed internet connection": "plusfixed"}
for m in ("XGB", "LR"):
    for s_, k in AKEY.items():
        x = abl[(abl.model == m) & (abl.predictor_set == s_)]
        src, sl = "results/metrics/E6_ablation.csv", f"model={m}; predictor_set={s_}"
        R.add(f"abl.{m}.{k}.auc.min", f(x.auc.min()), src, sl + "; min of auc over 2020-2025")
        R.add(f"abl.{m}.{k}.auc.max", f(x.auc.max()), src, sl + "; max of auc over 2020-2025")
        R.add(f"abl.{m}.{k}.prauc.min", f(x.prauc.min()), src, sl + "; min of prauc over 2020-2025")
        R.add(f"abl.{m}.{k}.prauc.max", f(x.prauc.max()), src, sl + "; max of prauc over 2020-2025")
        if k != "full":
            for met in ("auc", "prauc"):
                q = ablp[(ablp.model == m) & (ablp.predictor_set == s_) & (ablp.metric == met)]
                src2, sl2 = "results/metrics/E6_ablation_paired.csv", f"model={m}; predictor_set={s_}; metric={met}"
                R.add(f"abl.{m}.{k}.d{met}.min", fs(q["diff"].min()), src2, sl2 + "; min of diff over 2020-2025")
                R.add(f"abl.{m}.{k}.d{met}.max", fs(q["diff"].max()), src2, sl2 + "; max of diff over 2020-2025")
                R.add(f"abl.{m}.{k}.d{met}.absmax", f(q["diff"].abs().max()), src2, sl2 + "; max of |diff| over 2020-2025")
                R.add(f"abl.{m}.{k}.d{met}.absmin", f(q["diff"].abs().min()), src2, sl2 + "; min of |diff| over 2020-2025")
                R.add(f"abl.{m}.{k}.d{met}.nsig", str(int(((q.lo > 0) | (q.hi < 0)).sum())), src2, sl2 + "; years whose interval excludes 0 (of 6)")
x = abl[abl.model == "Score"]
R.add("abl.score.auc.min", f(x.auc.min()), "results/metrics/E6_ablation.csv", "model=Score; min of auc over 2020-2025")
R.add("abl.score.auc.max", f(x.auc.max()), "results/metrics/E6_ablation.csv", "model=Score; max of auc over 2020-2025")
full_pr = abl[(abl.model == "XGB") & (abl.predictor_set == "Full model")].set_index("year").prauc
notraj_pr = abl[(abl.model == "XGB") & (abl.predictor_set == "Without schooling trajectory")].set_index("year").prauc
R.add("abl.XGB.notraj.prauc.ratio.min", f((notraj_pr / full_pr).min() * 100, 0), "results/metrics/E6_ablation.csv", "XGB; PR-AUC without schooling trajectory / full model; min over years, in %")
R.add("abl.XGB.notraj.prauc.ratio.max", f((notraj_pr / full_pr).max() * 100, 0), "results/metrics/E6_ablation.csv", "XGB; PR-AUC without schooling trajectory / full model; max over years, in %")

types = rd("leakage_outcome_types.csv")
TK = {"All non-enrolled vs enrolled": "all", "Persistent vs enrolled": "pers", "Incident vs enrolled": "inc"}
PK = {"CV 2017-2019": "cv", "2021": "2021", "2022-2025": "post", "2021-2025": "2125"}
for per, pk in PK.items():
    for m in MODELS:
        for c, ck in TK.items():
            r = one(types, period=per, model=m, contrast=c)
            reg_metrics(f"type.{pk}.{m}.{ck}", r, "results/metrics/leakage_outcome_types.csv", sel_str(period=per, model=m, contrast=c), ["auc", "prauc", "prev"])
            R.add(f"type.{pk}.{m}.{ck}.events", thousands(r.events), "results/metrics/leakage_outcome_types.csv", sel_str(period=per, model=m, contrast=c) + "; column=events")
inc = rd("leakage_incident_model.csv")
IK = {"Full model": "full", "Without schooling trajectory": "notraj", "Without age and schooling trajectory": "noagetraj",
      "Age and schooling trajectory only": "agetraj", "Household and area only": "hharea"}
for m in ("XGB", "LR"):
    for s_, k in IK.items():
        for per, pk in PK.items():
            r = one(inc, model=m, predictor_set=s_, test_set=per)
            reg_metrics(f"incmodel.{m}.{k}.{pk}", r, "results/metrics/leakage_incident_model.csv", sel_str(model=m, predictor_set=s_, test_set=per), ["auc", "prauc", "prev", "citl"])

link = rd("panel_linkage.csv"); pan = rd("panel_prospective_performance.csv"); psum = json.load(open(MET_DIR / "panel_summary.json")); bal = rd("panel_followup_balance.csv")
R.add("panel.pairs", thousands(psum["pairs"]), "results/metrics/panel_summary.json", "pairs")
R.add("panel.events", thousands(psum["events"]), "results/metrics/panel_summary.json", "events")
R.add("panel.events.train", thousands(psum["train_events"]), "results/metrics/panel_summary.json", "train_events")
R.add("panel.events.test", thousands(psum["test_events"]), "results/metrics/panel_summary.json", "test_events")
R.add("panel.found.min", f(100 * link.share_found.min(), 0), "results/metrics/panel_linkage.csv", "min of share_found")
R.add("panel.found.max", f(100 * link.share_found.max(), 0), "results/metrics/panel_linkage.csv", "max of share_found")
R.add("panel.consistent.min", f(100 * link.sex_age_consistent.min(), 1), "results/metrics/panel_linkage.csv", "min of sex_age_consistent")
R.add("panel.balance.max", f(bal.std_diff.abs().max(), 2), "results/metrics/panel_followup_balance.csv", "max |std_diff|")
QK = {"t in 2017-2019 (score out-of-fold)": "pre", "t in 2020-2024 (frozen score, out of time)": "oot", "All pairs 2017-2024": "all"}
for per, pk in QK.items():
    for m in MODELS:
        r = one(pan, period=per, model=m, approach="Status score at t (no prospective training)")
        reg_metrics(f"panel.{pk}.{m}.status", r, "results/metrics/panel_prospective_performance.csv", sel_str(period=per, model=m, approach="Status score at t (no prospective training)"), ["auc", "prauc", "prev", "citl"])
    r = one(pan, period=per, model="Score", approach="Age-grade distortion alone")
    reg_metrics(f"panel.{pk}.score", r, "results/metrics/panel_prospective_performance.csv", sel_str(period=per, model="Score", approach="Age-grade distortion alone"), ["auc", "prauc"])
for m in ("LR", "XGB"):
    for lab, k in (("Full", "full"), ("Without schooling trajectory", "notraj"), ("Age and schooling trajectory only", "agetraj")):
        a1 = f"Prospective model trained on pairs t<=2019: {lab}"; a2 = f"Prospective model, grouped 5-fold CV: {lab}"
        reg_metrics(f"panel.oot.{m}.prosp.{k}", one(pan, model=m, approach=a1), "results/metrics/panel_prospective_performance.csv", sel_str(model=m, approach=a1), ["auc", "prauc", "citl"])
        reg_metrics(f"panel.all.{m}.prospcv.{k}", one(pan, model=m, approach=a2), "results/metrics/panel_prospective_performance.csv", sel_str(model=m, approach=a2), ["auc", "prauc", "citl"])

# ---- Table V: what the score identifies (XGBoost)
rows5 = []


def t5(block, label, r, src):
    rows5.append(dict(block=block, outcome=label, events=int(r.events), prev=r.prev, auc=r.auc, auc_lo=r.auc_lo, auc_hi=r.auc_hi,
                      prauc=r.prauc, prauc_lo=r.prauc_lo, prauc_hi=r.prauc_hi, source=src))


for per, blk in (("CV 2017-2019", "In-period, 2017--2019"), ("2022-2025", "Frozen model, 2022--2025")):
    for c, lab in (("All non-enrolled vs enrolled", "Not enrolled (status)"), ("Persistent vs enrolled", "\\quad persistent"), ("Incident vs enrolled", "\\quad incident")):
        if per.startswith("CV") and c.startswith("All"):      # same sample as E1: reuse that row so the two tables agree to the last digit
            t5(blk, lab, one(pu, model="XGB", test_set="CV 2017-2019"), "E1_E2_performance.csv")
        else:
            t5(blk, lab, one(types, period=per, model="XGB", contrast=c), "leakage_outcome_types.csv")
t5("Frozen model, 2022--2025", "\\quad incident, own model$^{\\mathrm{a}}$", one(inc, model="XGB", predictor_set="Full model", test_set="2022-2025"), "leakage_incident_model.csv")
t5("Not enrolled next year (panel, baseline 2020--2024)", "\\quad status score", one(pan, period="t in 2020-2024 (frozen score, out of time)", model="XGB", approach="Status score at t (no prospective training)"), "panel_prospective_performance.csv")
t5("Not enrolled next year (panel, baseline 2020--2024)", "\\quad prospective model$^{\\mathrm{b}}$", one(pan, model="XGB", approach="Prospective model trained on pairs t<=2019: Full"), "panel_prospective_performance.csv")
t5df = pd.DataFrame(rows5); t5df.to_csv(TAB_DIR / "T5_outcome_validity.csv", index=False)
body = [r"\begin{tabular}{@{}lrrcc@{}}", r"\toprule", r"Outcome & Events & Prev. & ROC-AUC [95\% CI] & PR-AUC \\"]
last = None
for _, r in t5df.iterrows():
    if r.block != last:
        body += [r"\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{r.block}}}}} \\\\"]; last = r.block
    body.append(f"{r.outcome} & {thousands(r.events)} & {pct(r.prev, 1)} & {f(r.auc)} [{ci(r.auc_lo, r.auc_hi)}] & {f(r.prauc)} \\\\")
body += [r"\bottomrule", r"\end{tabular}"]
o.write_table(GEN / "table_outcome.tex", body)

# =============================================================================================== E4: subgroups
sub = rd("E4_subgroups.csv"); gaps = rd("E4_subgroup_gaps.csv"); small = rd("E4_small_cells.csv")
PERK = {"In-period CV 2017-2019": "cv", "Pandemic 2020-2021": "pan", "Post-pandemic 2022-2025": "post"}
GRP = ["All", "Urban", "Rural", "Non-poor", "Poor", "Spanish", "Indigenous tongue", "Male", "Female"]
GK = {"All": "all", "Urban": "urban", "Rural": "rural", "Non-poor": "nonpoor", "Poor": "poor", "Spanish": "spanish", "Indigenous tongue": "indig", "Male": "male", "Female": "female"}
CK = {"Rural - Urban": "rural-urban", "Poor - Non-poor": "poor-nonpoor", "Indigenous tongue - Spanish": "indig-spanish", "Female - Male": "female-male"}
for per, pk in PERK.items():
    for m in MODELS:
        for g in GRP:
            r = one(sub, period=per, model=m, group=g)
            reg_metrics(f"sub.{pk}.{m}.{GK[g]}", r, "results/metrics/E4_subgroups.csv", sel_str(period=per, model=m, group=g), ["auc", "citl", "slope", "recall", "precision", "flag_rate", "prev"])
            R.add(f"sub.{pk}.{m}.{GK[g]}.events", thousands(r.events), "results/metrics/E4_subgroups.csv", sel_str(period=per, model=m, group=g) + "; column=events")
        for c, ck in CK.items():
            for met in ("auc", "citl", "recall", "precision", "flag_rate"):
                q = one(gaps, period=per, model=m, contrast=c, metric=met)
                d_ = 2 if met in ("citl", "recall", "precision") else 3
                R.add(f"gap.{pk}.{m}.{ck}.{met}", fs(q["diff"], d_), "results/metrics/E4_subgroup_gaps.csv", sel_str(period=per, model=m, contrast=c, metric=met) + "; column=diff")
                R.add(f"gap.{pk}.{m}.{ck}.{met}.ci", ci(q.lo, q.hi, d_), "results/metrics/E4_subgroup_gaps.csv", sel_str(period=per, model=m, contrast=c, metric=met) + "; columns=lo,hi")
    q = gaps[(gaps.period == per) & (gaps.metric == "citl")]
    R.add(f"gap.{pk}.citl.nsig", str(int(((q.lo > 0) | (q.hi < 0)).sum())), "results/metrics/E4_subgroup_gaps.csv", f"period={per}; metric=citl; contrasts x models whose interval excludes 0 (of {len(q)})")
    q = gaps[(gaps.period == per) & (gaps.metric == "auc")]
    R.add(f"gap.{pk}.auc.nsig", str(int(((q.lo > 0) | (q.hi < 0)).sum())), "results/metrics/E4_subgroup_gaps.csv", f"period={per}; metric=auc; contrasts x models whose interval excludes 0 (of {len(q)})")
R.add("sub.groups.evaluated", str(sub[sub.period.str.startswith("Post")].group.nunique()), "results/metrics/E4_subgroups.csv", "period=Post-pandemic 2022-2025; number of groups evaluated")
R.add("sub.small.n", str(small.group.nunique()), "results/metrics/E4_small_cells.csv", "number of distinct groups with fewer than 50 events in some period")
x = sub[(sub.period == "Post-pandemic 2022-2025") & (sub.model == "XGB") & (sub.group != "All") & ~sub.group.str.contains(" x ")]
R.add("sub.post.XGB.auc.min", f(x.auc.min()), "results/metrics/E4_subgroups.csv", "period=Post-pandemic; model=XGB; eight marginal groups; min of auc")
R.add("sub.post.XGB.auc.max", f(x.auc.max()), "results/metrics/E4_subgroups.csv", "period=Post-pandemic; model=XGB; eight marginal groups; max of auc")

grec = rd("E4_group_recalibration.csv")
VK = {"Frozen": "frozen", "Global intercept": "global", "Cell-specific intercept": "cell"}
for m in MODELS:
    for v, vk in VK.items():
        for g in ("All", "Urban", "Rural", "Spanish", "Indigenous tongue"):
            x = grec[(grec.model == m) & (grec.variant == v) & (grec.group == g)]
            if len(x) == 1:
                reg_metrics(f"grec.{m}.{vk}.{GK[g]}", x.iloc[0], "results/metrics/E4_group_recalibration.csv", sel_str(model=m, variant=v, group=g), ["auc", "citl", "recall", "precision"])

# ---- Table IV: subgroup reliability of the frozen XGBoost model, 2022-2025
x = sub[(sub.period == "Post-pandemic 2022-2025") & (sub.model == "XGB") & sub.group.isin(GRP)].set_index("group").loc[GRP].reset_index()
x[["group", "n", "events", "prev", "auc", "auc_lo", "auc_hi", "citl", "citl_lo", "citl_hi", "slope", "recall", "recall_lo", "recall_hi", "precision", "precision_lo", "precision_hi", "flag_rate"]].to_csv(TAB_DIR / "T4_subgroups_2022_2025.csv", index=False)
body = [r"\begin{tabular}{@{}lrccc@{}}", r"\toprule", r"Group & Events & ROC-AUC & CITL [95\% CI] & Recall [95\% CI] \\", r"\midrule"]
for i, r in x.iterrows():
    lab = "Indigenous" if r.group == "Indigenous tongue" else r.group
    body.append(f"{lab} & {thousands(r.events)} & {f(r.auc)} & {f(r.citl, 2)} [{ci(r.citl_lo, r.citl_hi, 2)}] & {f(r.recall, 2)} [{ci(r.recall_lo, r.recall_hi, 2)}] \\\\")
    if r.group in ("All", "Rural", "Poor", "Indigenous tongue"):
        body.append(r"\addlinespace[1pt]")
body += [r"\bottomrule", r"\end{tabular}"]
o.write_table(GEN / "table_subgroups.tex", body)

# =============================================================================================== E5: explanations
fam = pd.read_csv(MET_DIR / "E5_shap_family.csv").set_index("family")
FK = {"Age and schooling trajectory": "traj", "Other individual": "indiv", "Household socioeconomic": "hh", "Digital access": "digital", "Area": "area"}
for fam_, k in FK.items():
    for col, ck in (("Training 2017-2019", "train"), ("Post-pandemic 2022-2025", "post")):
        R.add(f"shap.fam.{k}.{ck}", f(fam.loc[fam_, col], 2), "results/metrics/E5_shap_family.csv", f"family={fam_}; column={col}")
ssum = json.load(open(MET_DIR / "E5_shap_summary.json"))
R.add("shap.refits", str(ssum["n_refits"]), "results/metrics/E5_shap_summary.json", "n_refits")
R.add("shap.top1.share", f(100 * ssum["share_refits_same_top1"], 0), "results/metrics/E5_shap_summary.json", "share_refits_same_top1, in %")
R.add("shap.top5.share", f(100 * ssum["share_refits_same_top5_set"], 0), "results/metrics/E5_shap_summary.json", "share_refits_same_top5_set, in %")
R.add("shap.rho.min", f(ssum["spearman_rank_corr_min"], 2), "results/metrics/E5_shap_summary.json", "spearman_rank_corr_min")
R.add("shap.rho.trainpost", f(ssum["spearman_training_vs_post"], 2), "results/metrics/E5_shap_summary.json", "spearman_training_vs_post")
R.add("shap.rho.perm", f(ssum["spearman_shap_vs_permutation"], 2), "results/metrics/E5_shap_summary.json", "spearman_shap_vs_permutation")
stab = rd("E5_shap_stability.csv")
x = stab[stab.rank_original.between(3, 10)]
R.add("shap.rank.3to10.min", str(int(x.rank_min.min())), "results/metrics/E5_shap_stability.csv", "features ranked 3-10; min of rank_min")
R.add("shap.rank.3to10.max", str(int(x.rank_max.max())), "results/metrics/E5_shap_stability.csv", "features ranked 3-10; max of rank_max")
perm = rd("E5_permutation_importance.csv")
pf_ = perm[perm.level == "family"].set_index("feature")
for fam_, k in FK.items():
    R.add(f"perm.fam.{k}.auc", f(pf_.loc[fam_, "auc_drop"]), "results/metrics/E5_permutation_importance.csv", f"level=family; feature={fam_}; column=auc_drop")
dep = rd("E5_shap_dependence_overage.csv").set_index("overage")
for v in (0, 1, 2, 3):
    R.add(f"dep.overage.{v}.rate", pct(dep.loc[float(v), "observed_rate"], 1), "results/metrics/E5_shap_dependence_overage.csv", f"overage={v}; column=observed_rate, in %")
    R.add(f"dep.overage.{v}.share", pct(dep.loc[float(v), "share"], 0), "results/metrics/E5_shap_dependence_overage.csv", f"overage={v}; column=share, in %")

# =============================================================================================== comparison with the original project
leg = pd.read_csv(ROOT / "legacy" / "original_tables" / "E1_E3_performance.csv")
cmp_rows = []
for _, r in leg.iterrows():
    if r.exp.startswith("E1"):
        s_, what = "CV 2017-2019", "E1 in-period CV"
    elif r.exp.startswith("E2"):
        s_, what = r.exp.split("->")[1], "E2 frozen model"
    else:
        continue
    n = one(pu, model=r.model, test_set=s_)
    cmp_rows.append(dict(experiment=what, evaluation_set=s_, model=r.model,
                         auc_original=r.auc, auc_revised=n.auc, auc_ci_original=f"[{r.auc_lo:.3f}, {r.auc_hi:.3f}]" if not pd.isna(r.auc_lo) else "not reported",
                         auc_ci_revised=f"[{n.auc_lo:.3f}, {n.auc_hi:.3f}]", prauc_original=r.prauc, prauc_revised=n.prauc,
                         citl_original=r.citl, citl_revised=n.citl, citl_ci_revised=f"[{n.citl_lo:.3f}, {n.citl_hi:.3f}]",
                         slope_original=r.slope, slope_revised=n.slope))
for _, r in leg[leg.exp.str.startswith("E3")].iterrows():
    strat = "Retrained on 2022-2024" if "retrain" in r.exp else "Intercept recalibration"
    n = one(upd, model=r.model, strategy=strat)
    cmp_rows.append(dict(experiment="E3 " + strat + (" (original fitted on 2023-2024)" if "recalib" in r.exp else ""), evaluation_set="2025", model=r.model,
                         auc_original=r.auc, auc_revised=n.auc, auc_ci_original=f"[{r.auc_lo:.3f}, {r.auc_hi:.3f}]" if not pd.isna(r.auc_lo) else "not reported",
                         auc_ci_revised=f"[{n.auc_lo:.3f}, {n.auc_hi:.3f}]", prauc_original=r.prauc, prauc_revised=n.prauc,
                         citl_original=r.citl, citl_revised=n.citl, citl_ci_revised=f"[{n.citl_lo:.3f}, {n.citl_hi:.3f}]",
                         slope_original=r.slope, slope_revised=n.slope))
pd.DataFrame(cmp_rows).round(4).to_csv(TAB_DIR / "comparison_original_vs_revised.csv", index=False)

# =============================================================================================== aggregated prediction summaries (no person-level data)
from src.config import PROC_DIR  # noqa: E402
if (PROC_DIR / "predictions.parquet").exists() and (PROC_DIR / "analytic.parquet").exists():
    from src.calibration.recalibration import weighted_quantile  # noqa: E402
    a_ = pd.read_parquet(PROC_DIR / "analytic.parquet", columns=["year", "factora07", "y"])
    pr_ = pd.read_parquet(PROC_DIR / "predictions.parquet")
    rows_ = []
    for m in MODELS:
        p_ = np.where(a_.year.isin(TRAIN_YEARS), pr_[f"oof_{m}_uw"], pr_[f"p_{m}_uw"])
        for y in sorted(a_.year.unique()):
            k = (a_.year == y).values
            q = {f"q{int(100 * qq):02d}": float(weighted_quantile(p_[k], a_.factora07.values[k], qq)) for qq in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)}
            rows_.append(dict(model=m, year=int(y), prediction="out-of-fold" if y in TRAIN_YEARS else "frozen model", n=int(k.sum()),
                              mean_pred=float(np.average(p_[k], weights=a_.factora07.values[k])),
                              observed=float(np.average(a_.y.values[k], weights=a_.factora07.values[k])), **q))
    pd.DataFrame(rows_).to_csv(TAB_DIR / "prediction_distribution_by_year.csv", index=False)

# =============================================================================================== Figure 1
o.set_style()
XS = np.arange(len(SETS)); XL = ["CV"] + [f"'{str(y)[2:]}" for y in TEST_YEARS]


def shade_pandemic(ax):
    ax.axvspan(0.5, 2.5, color="#f1efe9", zorder=0, lw=0)


def fig_temporal():
    fig, axs = plt.subplots(1, 3, figsize=(o.COL2, 2.75), gridspec_kw=dict(wspace=0.30))
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.13), borderaxespad=0.0, columnspacing=1.0, handletextpad=0.5)
    # (a) discrimination of the frozen XGBoost model against a model fitted within each year
    ax = axs[0]; shade_pandemic(ax)
    x = pu[pu.model == "XGB"].set_index("test_set").loc[SETS]
    ax.errorbar(XS, x.auc, yerr=[x.auc - x.auc_lo, x.auc_hi - x.auc], color=o.AQUA, marker="D", ms=4, lw=1.4, capsize=2, elinewidth=0.9,
                label="Frozen model (trained 2017-2019)", zorder=3)
    r = iny[(iny.model == "XGB") & (iny.metric == "auc")].set_index("year").loc[TEST_YEARS]
    ax.errorbar(XS[1:] + 0.16, r.in_year_cv, yerr=[r.in_year_cv - r.in_year_lo, r.in_year_hi - r.in_year_cv], color=o.INK2, marker="o", mfc="white", ms=4,
                lw=0, capsize=2, elinewidth=0.9, label="Same-year model (grouped CV)", zorder=2)
    ax.set_ylim(0.76, 0.97); ax.set_ylabel("ROC-AUC"); ax.set_title("(a) Discrimination, XGBoost", loc="left")
    ax.legend(ncol=1, handlelength=1.6, **below)
    # (b) calibration-in-the-large of the three frozen models
    ax = axs[1]; shade_pandemic(ax); ax.axhline(0, color=o.INK2, lw=0.7)
    for k, m in enumerate(MODELS):
        x = pu[pu.model == m].set_index("test_set").loc[SETS]
        ax.errorbar(XS + (k - 1) * 0.2, x.citl, yerr=[x.citl - x.citl_lo, x.citl_hi - x.citl], color=o.MODEL_COLOR[m], marker=o.MODEL_MARKER[m], ms=3.6,
                    lw=0, capsize=1.6, elinewidth=0.9, label=o.MODEL_NAME[m])
    ax.set_ylim(-0.48, 0.72); ax.set_ylabel("Calibration-in-the-large (logit)"); ax.set_title("(b) Calibration, frozen models", loc="left")
    ax.legend(ncol=2, handlelength=1.0, **below)
    # (c) calibration under updating, rolling origin (each year uses only earlier years)
    ax = axs[2]; shade_pandemic(ax); ax.axhline(0, color=o.INK2, lw=0.7)
    style = {"Frozen (2017-2019)": (o.AQUA, "D", "-", "Frozen"), "Intercept recalibration on t-1": (o.VIOLET, "v", "--", "Intercept update on year t-1"),
             "Retrained, all previous years": (o.ORANGE, "s", "-.", "Retrained on all earlier years")}
    for s, (c, mk, ls, lab) in style.items():
        x = roll[(roll.model == "XGB") & (roll.strategy == s)].set_index("year").reindex(TEST_YEARS)
        ax.plot(XS[1:], x.citl, color=c, marker=mk, ls=ls, ms=4, lw=1.3, label=lab)
    ax.set_ylim(-0.48, 0.72); ax.set_title("(c) Calibration under updating, XGBoost", loc="left")
    ax.legend(ncol=1, handlelength=2.6, **below)
    for ax in axs:
        ax.set_xticks(XS); ax.set_xticklabels(XL); ax.set_xlim(-0.5, len(SETS) - 0.5); ax.grid(axis="x", visible=False)
        ax.text(1.5, ax.get_ylim()[1], "pandemic", ha="center", va="top", fontsize=6.5, color=o.INK2)
    axs[2].set_xlim(0.5, len(SETS) - 0.5)
    o.save(fig, FIG_DIR, "figure_01"); plt.close(fig)


fig_temporal()


def fig_subgroups():
    """Calibration-in-the-large of the frozen XGBoost model by group and period (forest plot)."""
    order = ["Urban", "Rural", "Non-poor", "Poor", "Spanish", "Indigenous tongue", "Male", "Female"]
    ypos = {g: -(i + 0.55 * (i // 2)) for i, g in enumerate(order)}
    per = [("In-period CV 2017-2019", "Training period, 2017-2019 (CV)", o.INK2, "o", 0.24), ("Pandemic 2020-2021", "2020-2021", o.ORANGE, "s", 0.0),
           ("Post-pandemic 2022-2025", "2022-2025", o.BLUE, "D", -0.24)]
    fig, ax = plt.subplots(figsize=(o.COL1, 3.05))
    ax.axvline(0, color=o.INK2, lw=0.7)
    for plab, leg, c, mk, dy in per:
        x = sub[(sub.period == plab) & (sub.model == "XGB")].set_index("group").loc[order]
        yy = np.array([ypos[g] for g in order]) + dy
        ax.errorbar(x.citl, yy, xerr=[x.citl - x.citl_lo, x.citl_hi - x.citl], color=c, marker=mk, ms=3.8, lw=0, elinewidth=1.0, capsize=1.8, label=leg,
                    mfc="white" if mk == "o" else c)
    ax.set_yticks([ypos[g] for g in order]); ax.set_yticklabels(["Indigenous\nmother tongue" if g == "Indigenous tongue" else g for g in order])
    ax.set_ylim(min(ypos.values()) - 0.6, 0.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calibration-in-the-large (logit)")
    lo, hi = ax.get_xlim(); ax.set_xlim(min(lo, -0.45), max(hi, 0.85))
    ax.text(ax.get_xlim()[0], 0.62, "risk over-predicted", ha="left", va="bottom", fontsize=6.5, color=o.INK2)
    ax.text(ax.get_xlim()[1], 0.62, "risk under-predicted", ha="right", va="bottom", fontsize=6.5, color=o.INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=3, handlelength=1.0, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    o.save(fig, FIG_DIR, "figure_02"); plt.close(fig)


fig_subgroups()


# =============================================================================================== supplementary figures
# Not in the manuscript (page limit); delivered with the results. Same palette and style as Figures 1-2.
YRS = list(range(2017, 2026))


def sup_prevalence():
    fig, ax = plt.subplots(figsize=(o.COL1, 2.3))
    ax.axvspan(2019.5, 2021.5, color="#f1efe9", zorder=0, lw=0)
    for area, c, mk in (("National", o.INK, "o"), ("Rural", o.ORANGE, "s"), ("Urban", o.BLUE, "D")):
        x = prev[prev.area == area].set_index("year").loc[YRS]
        ax.errorbar(x.index + {"National": 0, "Rural": 0.12, "Urban": -0.12}[area], 100 * x.prev, yerr=[100 * (x.prev - x.lo_boot), 100 * (x.hi_boot - x.prev)],
                    color=c, marker=mk, ms=3.5, lw=1.2, capsize=1.5, elinewidth=0.8, label=area)
    ax.set_ylabel("Not enrolled (%)"); ax.set_ylim(0, None); ax.set_xticks(YRS); ax.set_xticklabels([f"'{str(y)[2:]}" for y in YRS])
    ax.grid(axis="x", visible=False); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), handlelength=1.2)
    ax.set_title("Survey-weighted non-enrollment, ages 12-17, Apr-Dec", loc="left")
    o.save(fig, FIG_DIR, "figure_S01_prevalence"); plt.close(fig)


def sup_discrimination():
    fig, axs = plt.subplots(1, 2, figsize=(o.COL2, 2.4), gridspec_kw=dict(wspace=0.25))
    for ax, met, lab in ((axs[0], "auc", "ROC-AUC"), (axs[1], "prauc", "PR-AUC")):
        shade_pandemic(ax)
        for k, m in enumerate(MODELS):
            x = pu[pu.model == m].set_index("test_set").loc[SETS]
            ax.errorbar(XS + (k - 1) * 0.18, x[met], yerr=[x[met] - x[met + "_lo"], x[met + "_hi"] - x[met]], color=o.MODEL_COLOR[m],
                        marker=o.MODEL_MARKER[m], ms=3.5, lw=1.0, capsize=1.5, elinewidth=0.8, label=o.MODEL_NAME[m])
        ax.set_xticks(XS); ax.set_xticklabels(XL); ax.grid(axis="x", visible=False); ax.set_ylabel(lab); ax.set_title(f"{lab} of the frozen models", loc="left")
    axs[1].legend(loc="lower right", handlelength=1.2)
    o.save(fig, FIG_DIR, "figure_S02_discrimination_by_year"); plt.close(fig)


def sup_reliability():
    rel = rd("reliability_curves.csv")
    fig, axs = plt.subplots(1, 3, figsize=(o.COL2, 2.5), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, s_ in zip(axs, ("CV 2017-2019", "2020", "2025")):
        ax.plot([0, 0.6], [0, 0.6], color=o.INK2, lw=0.7, ls="--")
        for m in MODELS:
            x = rel[(rel.test_set == s_) & (rel.model == m)]
            ax.plot(x.mean_pred, x.observed, color=o.MODEL_COLOR[m], marker=o.MODEL_MARKER[m], ms=3.2, lw=1.1, label=o.MODEL_NAME[m])
        ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
        ax.set_title("Training period (CV)" if s_.startswith("CV") else s_, loc="left"); ax.set_xlabel("Mean predicted risk (decile)")
    axs[0].set_ylabel("Observed rate"); axs[0].legend(loc="upper left", handlelength=1.2)
    o.save(fig, FIG_DIR, "figure_S03_calibration_curves"); plt.close(fig)


def sup_model_comparison():
    fig, axs = plt.subplots(1, 2, figsize=(o.COL2, 2.3), gridspec_kw=dict(wspace=0.25))
    for ax, met, lab in ((axs[0], "auc", "Difference in ROC-AUC"), (axs[1], "citl", "Difference in CITL")):
        ax.axhline(0, color=o.INK2, lw=0.7); shade_pandemic(ax)
        for k, (cn, lb, c, mk) in enumerate((("XGB_uw - LR_uw", "XGBoost minus logistic regression", o.BLUE, "o"),
                                             ("XGB_uw - RF_uw", "XGBoost minus random forest", o.ORANGE, "s"),
                                             ("XGB_w - XGB_uw", "XGBoost weighted minus unweighted", o.VIOLET, "D"))):
            x = pd_[(pd_.contrast == cn) & (pd_.metric == met)].set_index("test_set").loc[SETS]
            ax.errorbar(XS + (k - 1) * 0.18, x["diff"], yerr=[x["diff"] - x.lo, x.hi - x["diff"]], color=c, marker=mk, ms=3.5, lw=0, capsize=1.5,
                        elinewidth=0.9, label=lb)
        ax.set_xticks(XS); ax.set_xticklabels(XL); ax.grid(axis="x", visible=False); ax.set_ylabel(lab); ax.set_title("Paired " + lab[0].lower() + lab[1:], loc="left")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=3, handlelength=1.0)
    o.save(fig, FIG_DIR, "figure_S04_model_comparison"); plt.close(fig)


def sup_shift():
    fig, axs = plt.subplots(1, 3, figsize=(o.COL2, 2.3), gridspec_kw=dict(wspace=0.4))
    x = shift.set_index("year").loc[TEST_YEARS]; xx = np.arange(len(TEST_YEARS))
    axs[0].bar(xx, x.prevalence_ratio, 0.6, color=o.BLUE); axs[0].axhline(1, color=o.INK2, lw=0.7)
    axs[0].set_title("Prevalence / training prevalence", loc="left")
    axs[1].bar(xx, x.domain_classifier_auc, 0.6, color=o.ORANGE); axs[1].axhline(0.5, color=o.INK2, lw=0.7, ls=":")
    axs[1].set_ylim(0.4, 0.8); axs[1].set_title("Domain-classifier ROC-AUC", loc="left")
    cols = [c for c in shift.columns if c.startswith("psi_") and c not in ("psi_max", "psi_mean", "psi_top3")]
    top = shift[cols].max().sort_values(ascending=False).index[:4]
    for k, c in enumerate(top):
        axs[2].plot(xx, x[c], marker="os^D"[k], ms=3.5, lw=1.1, color=[o.BLUE, o.ORANGE, o.AQUA, o.VIOLET][k], label=FEATURE_LABELS[c[4:]])
    axs[2].axhline(0.1, color=o.INK2, lw=0.7, ls=":"); axs[2].set_title("Population stability index", loc="left")
    axs[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=6.5, handlelength=1.2)
    for ax in axs:
        ax.set_xticks(xx); ax.set_xticklabels([f"'{str(y)[2:]}" for y in TEST_YEARS]); ax.grid(axis="x", visible=False)
    o.save(fig, FIG_DIR, "figure_S05_dataset_shift"); plt.close(fig)


def sup_shap():
    imp = pd.read_csv(MET_DIR / "E5_shap_importance.csv").set_index("feature")
    st = rd("E5_shap_stability.csv").set_index("feature")
    fig, axs = plt.subplots(1, 2, figsize=(o.COL2, 2.9), gridspec_kw=dict(wspace=0.75, width_ratios=[1.1, 1]))
    top = imp.head(12).index[::-1]; yy = np.arange(len(top))
    axs[0].barh(yy + 0.2, imp.loc[top, "Training 2017-2019"], 0.38, color=o.BLUE, label="2017-2019")
    axs[0].barh(yy - 0.2, imp.loc[top, "Post-pandemic 2022-2025"], 0.38, color=o.ORANGE, label="2022-2025")
    axs[0].errorbar(st.loc[top, "mean_abs_shap"], yy + 0.2, xerr=[st.loc[top, "mean_abs_shap"] - st.loc[top, "refit_p2.5"], st.loc[top, "refit_p97.5"] - st.loc[top, "mean_abs_shap"]],
                    fmt="none", ecolor=o.INK, elinewidth=0.7, capsize=1.2)
    axs[0].set_yticks(yy); axs[0].set_yticklabels([FEATURE_LABELS[f] for f in top]); axs[0].set_xscale("log"); axs[0].grid(axis="y", visible=False)
    axs[0].set_xlabel("Mean |SHAP| (log-odds, log scale)"); axs[0].legend(loc="lower right"); axs[0].set_title("SHAP importance (bars: 30 refits, 2017-2019)", loc="left")
    fm = fam[["Training 2017-2019", "Post-pandemic 2022-2025"]].sort_values("Training 2017-2019")
    yy = np.arange(len(fm))
    axs[1].barh(yy + 0.2, fm["Training 2017-2019"], 0.38, color=o.BLUE, label="2017-2019")
    axs[1].barh(yy - 0.2, fm["Post-pandemic 2022-2025"], 0.38, color=o.ORANGE, label="2022-2025")
    axs[1].set_yticks(yy); axs[1].set_yticklabels(fm.index); axs[1].grid(axis="y", visible=False); axs[1].set_xlabel("Mean |summed SHAP| (log-odds)")
    axs[1].set_title("By predictor family", loc="left")
    o.save(fig, FIG_DIR, "figure_S06_shap_summary"); plt.close(fig)


def sup_dependence():
    dp = rd("E5_shap_dependence_overage.csv")
    dp = dp[dp.n >= 30]
    fig, axs = plt.subplots(1, 2, figsize=(o.COL2 * 0.8, 2.3), gridspec_kw=dict(wspace=0.35))
    ax = axs[0]
    ax.fill_between(dp.overage, dp.shap_p10, dp.shap_p90, color=o.BLUE, alpha=0.18, lw=0, label="10th-90th percentile")
    ax.plot(dp.overage, dp.shap_mean, color=o.BLUE, marker="o", ms=3.5, lw=1.3, label="Mean")
    ax.axhline(0, color=o.INK2, lw=0.7); ax.set_xlabel("Age-grade distortion (years)"); ax.set_ylabel("SHAP value (log-odds)")
    ax.set_title("SHAP value of age-grade distortion", loc="left"); ax.legend(loc="upper left")
    ax = axs[1]
    ax.plot(dp.overage, 100 * dp.observed_rate, color=o.ORANGE, marker="s", ms=3.5, lw=1.3)
    ax.set_xlabel("Age-grade distortion (years)"); ax.set_ylabel("Not enrolled (%, weighted)"); ax.set_title("Observed non-enrollment", loc="left")
    fig.text(0.0, -0.06, "Training years 2017-2019; distortion values with at least 30 adolescents.", fontsize=6.5, color=o.INK2)
    o.save(fig, FIG_DIR, "figure_S07_shap_dependence_overage"); plt.close(fig)


def sup_updating():
    fig, axs = plt.subplots(1, 2, figsize=(o.COL2, 2.2), gridspec_kw=dict(wspace=0.55))
    order = ["Frozen (2017-2019)", "Intercept recalibration", "Intercept + slope recalibration", "Retrained on 2022-2024", "Retrained on 2017-2024"]
    for ax, met, lab in ((axs[0], "auc", "ROC-AUC, 2025"), (axs[1], "citl", "Calibration-in-the-large, 2025")):
        for k, m in enumerate(MODELS):
            x = upd[upd.model == m].set_index("strategy").loc[order]
            yy = np.arange(len(order))[::-1] + (k - 1) * 0.22
            ax.errorbar(x[met], yy, xerr=[x[met] - x[met + "_lo"], x[met + "_hi"] - x[met]], color=o.MODEL_COLOR[m], marker=o.MODEL_MARKER[m], ms=3.3,
                        lw=0, capsize=1.3, elinewidth=0.8, label=o.MODEL_NAME[m])
        ax.set_yticks(np.arange(len(order))[::-1]); ax.set_yticklabels([ULAB[s_].replace("--", "-") for s_ in order]); ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
    axs[1].axvline(0, color=o.INK2, lw=0.7); axs[1].legend(loc="upper center", bbox_to_anchor=(-0.3, -0.25), ncol=3, handlelength=1.0)
    axs[0].set_title("Updating strategies fitted on 2022-2024", loc="left")
    o.save(fig, FIG_DIR, "figure_S08_updating_2025"); plt.close(fig)


def sup_ablation():
    sets = ["Full model", "Without age-grade distortion", "Without digital access", "Age and schooling trajectory only", "Without schooling trajectory",
            "Without age and schooling trajectory", "Household and area only"]
    cols_ = [o.INK, o.AQUA, o.YELLOW, o.VIOLET, o.ORANGE, o.MAGENTA, o.BLUE]
    fig, axs = plt.subplots(1, 2, figsize=(o.COL2, 2.5), gridspec_kw=dict(wspace=0.25))
    xx = np.arange(len(TEST_YEARS))
    for ax, met, lab in ((axs[0], "auc", "ROC-AUC"), (axs[1], "prauc", "PR-AUC")):
        for k, s_ in enumerate(sets):
            x = abl[(abl.model == "XGB") & (abl.predictor_set == s_)].set_index("year").loc[TEST_YEARS]
            ax.plot(xx, x[met], marker="osD^v<>"[k], ms=3.2, lw=1.1, color=cols_[k], label=s_)
        sc = abl[abl.model == "Score"].set_index("year").loc[TEST_YEARS]
        ax.plot(xx, sc[met], color=o.INK2, ls=":", lw=1.1, label="Age-grade distortion alone (no model)")
        ax.set_xticks(xx); ax.set_xticklabels(TEST_YEARS); ax.grid(axis="x", visible=False); ax.set_ylabel(lab); ax.set_title(f"{lab}, XGBoost refitted on 2017-2019", loc="left")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=3, fontsize=6.5, handlelength=1.6)
    o.save(fig, FIG_DIR, "figure_S09_ablation"); plt.close(fig)


for fn in (sup_prevalence, sup_discrimination, sup_reliability, sup_model_comparison, sup_shift, sup_shap, sup_dependence, sup_updating, sup_ablation):
    fn()

# =============================================================================================== write the registry
R.write(GEN / "numbers.tex", TAB_DIR / "manuscript_numbers.csv", ROOT / "paper" / "manuscript.tex")
print(f"{len(R.rows)} numbers registered in {GEN / 'numbers.tex'}")

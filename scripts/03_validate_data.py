"""Step 3 - Data validation: coding by year, outcome checks, exclusions, design structure, internet item.

Inputs : data/processed/harmonized_10_18.parquet, data/processed/analytic.parquet, raw metadata
Outputs: results/tables/  value_labels_by_year.csv, raw_code_counts_by_year.csv, prevalence_by_year.csv,
                          prevalence_by_month.csv, prevalence_by_year_month.csv, exclusion_detail.csv,
                          internet_item_by_year.csv, interview_mode_2020.csv, design_structure.csv,
                          panel_overlap.csv, feature_summary_by_year.csv, reasons_not_enrolled.csv
         results/metrics/data_validation.json   (pass/fail of every automatic check)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES, MET_DIR, N_BOOT, PROC_DIR, RAW_DIR, SEED, TAB_DIR, TRAIN_YEARS, YEARS  # noqa: E402
from src.data.enaho import module_meta  # noqa: E402
from src.evaluation.metrics import wmean  # noqa: E402
from src.validation.bootstrap import collapse_singletons, replicate_weights, taylor_ratio_se  # noqa: E402

h = pd.read_parquet(PROC_DIR / "harmonized_10_18.parquet")
a = pd.read_parquet(PROC_DIR / "analytic.parquet")
checks = {}
wavg = lambda g, c: float(np.average(g[c], weights=g.factora07))

# ---- 1. value labels of the raw variables, every year (documents coding changes)
rows = []
for y in YEARS:
    for mod, vars_ in {"300": ["p300a", "p301a", "p306", "p303", "p305", "p313"], "200": ["p203", "p204", "p205", "p207"],
                       "100": ["p1142", "p1144", "p114b1", "panel", "tipoentrevista", "nbi1", "result"],
                       "612": ["p612n", "p612"], "sumaria": ["pobreza"]}.items():
        labels, values, _ = module_meta(RAW_DIR, y, mod)
        for v in vars_:
            vl = values.get(v, {})
            if v == "p612n":
                vl = {k: s for k, s in vl.items() if k in (7, 28) or "tab" in str(s).lower()}
            rows.append(dict(year=y, module=mod, variable=v, present=v in labels, label=labels.get(v, ""),
                             value_labels="; ".join(f"{int(k) if float(k).is_integer() else k}={s}" for k, s in sorted(vl.items()))))
lab = pd.DataFrame(rows); lab.to_csv(TAB_DIR / "value_labels_by_year.csv", index=False)
p306 = lab[lab.variable == "p306"].value_labels.str.lower().str.replace("í", "i")
checks["p306_coded_1_yes_2_no_every_year"] = bool(p306.str.contains("1=si").all() and p306.str.contains("2=no").all())

# ---- 2. observed codes of the key raw variables by year (ages 12-17)
t = h[h.p208a.between(12, 17)]
cc = pd.concat([t.groupby(["year", v], dropna=False).size().rename("n").reset_index().rename(columns={v: "code"}).assign(variable=v)
                for v in ["p306", "p301a", "p300a", "p203", "p204", "p205", "pobreza", "estrato", "dominio", "panel", "tipoentrevista"]])
cc.to_csv(TAB_DIR / "raw_code_counts_by_year.csv", index=False)
checks["p300a_codes_10_15_only_from_2020"] = bool(t[t.p300a >= 10].year.min() == 2020)
checks["p306_values_only_1_2"] = bool(set(t.p306.dropna().unique()) <= {1.0, 2.0})

# ---- 3. outcome: prevalence by year (two variance estimators), by month and by year-month
rows = []
frame = h[["year", "estrato", "conglome"]]
for y in YEARS:
    g = a[a.year == y]
    for area, m in [("National", g.y >= 0), ("Urban", g.rural == 0), ("Rural", g.rural == 1)]:
        s = g[m]
        est, se = taylor_ratio_se(g.y.values, g.factora07.values, g.estrato.values, g.conglome.values, domain=m.values,
                                  all_psu=frame[frame.year == y][["estrato", "conglome"]])
        W = replicate_weights(s.factora07.values, s.estrato.values, s.conglome.values, N_BOOT, SEED + y)
        bs = wmean(s.y.values, W)
        rows.append(dict(year=y, area=area, n=len(s), non_enrolled=int(s.y.sum()), prev=est, se_taylor=se, lo_taylor=est - 1.96 * se,
                         hi_taylor=est + 1.96 * se, lo_boot=float(np.percentile(bs[1:], 2.5)), hi_boot=float(np.percentile(bs[1:], 97.5))))
prev = pd.DataFrame(rows); prev.to_csv(TAB_DIR / "prevalence_by_year.csv", index=False)

pre = h[h.p208a.between(12, 17) & (h.p204 == 1) & (h.p205 != 1) & (h.p301a < 6) & h.p306.notna()]   # before the month filter
bm = pre.groupby("mes").apply(lambda g: pd.Series(dict(n=len(g), pct_not_enrolled=100 * wavg(g, "y")))).reset_index()
bm.to_csv(TAB_DIR / "prevalence_by_month.csv", index=False)
bym = pre.groupby(["year", "mes"]).apply(lambda g: pd.Series(dict(n=len(g), pct_not_enrolled=100 * wavg(g, "y")))).reset_index()
bym.to_csv(TAB_DIR / "prevalence_by_year_month.csv", index=False)
checks["analytic_n"] = int(len(a)); checks["analytic_non_enrolled"] = int(a.y.sum())
checks["month_artefact_jan_feb_above_50pct"] = bool((bm.set_index("mes").pct_not_enrolled[[1, 2]] > 50).all())
checks["april_december_between_3_and_5pct"] = bool(bm.set_index("mes").pct_not_enrolled[4:].between(3, 5).all())

# ---- 4. detail of the attainment exclusion
x = h[h.p208a.between(12, 17) & (h.p204 == 1) & (h.p205 != 1) & (h.m300 == 1)]
ex = pd.DataFrame({"completed_secondary_or_more (P301A 6-11)": x[x.p301a.between(6, 11)].groupby("year").size(),
                   "special_basic_education (P301A 12)": x[x.p301a == 12].groupby("year").size(),
                   "attainment_missing": x[x.p301a.isna()].groupby("year").size()}).fillna(0).astype(int)
ex.to_csv(TAB_DIR / "exclusion_detail.csv")

# ---- 5. household internet item (comparability break in 2023)
hh = h.drop_duplicates(["year", "conglome", "vivienda", "hogar"])
it = a.groupby("year").apply(lambda g: pd.Series(dict(
    pct_internet_item_P1144=100 * wavg(g, "hh_internet_raw"), pct_fixed_connection=100 * wavg(g, "hh_internet_fixed"),
    pct_mobile_only_among_connected=100 * float(((g.p114b1 != 1) & (g.hh_internet_raw == 1)).sum() / max((g.hh_internet_raw == 1).sum(), 1)) if g.name >= 2023 else np.nan,
    p114b1_missing_and_P1144_yes=int(((g.p114b1.isna()) & (g.hh_internet_raw == 1) & (g.name >= 2023)).sum()),
    pct_computer=100 * float(np.average(g.hh_computer.fillna(0), weights=g.factora07)), pct_mobile_phone=100 * wavg(g, "hh_mobile")))).reset_index()
it.to_csv(TAB_DIR / "internet_item_by_year.csv", index=False)
jump = it.set_index("year").pct_internet_item_P1144
checks["internet_item_jump_2022_2023_points"] = round(float(jump[2023] - jump[2022]), 1)
checks["fixed_connection_2022_vs_2023_points"] = round(float(it.set_index("year").pct_fixed_connection[2023] - jump[2022]), 1)

# ---- 6. interview mode (variable published in 2020 and 2021)
im = a[a.year.isin([2020, 2021])].groupby(["year", "mes"]).apply(lambda g: pd.Series(dict(n=len(g), pct_telephone=100 * g.telephone.mean()))).reset_index()
im.to_csv(TAB_DIR / "interview_mode_2020.csv", index=False)

# ---- 7. design structure: strata, PSUs, single-PSU strata, weights
rows = []
for y in YEARS:
    g = a[a.year == y]
    s1 = g.groupby("estrato").conglome.nunique(); s2 = g.groupby(["dept", "estrato"]).conglome.nunique()
    _, merged = collapse_singletons(g.dept + "_" + g.estrato.astype(str), g.conglome)
    rows.append(dict(year=y, n=len(g), psu=g.conglome.nunique(), strata_estrato=len(s1), min_psu_per_estrato=int(s1.min()),
                     strata_dept_x_estrato=len(s2), single_psu_strata_dept_x_estrato=int((s2 < 2).sum()), merged_by_collapse=merged,
                     persons_per_psu_mean=round(len(g) / g.conglome.nunique(), 2), weight_min=float(g.factora07.min()),
                     weight_max=float(g.factora07.max()), weight_cv=float(g.factora07.std() / g.factora07.mean()),
                     kish_deff_weights=float(1 + (g.factora07.std(ddof=0) / g.factora07.mean()) ** 2), weighted_population=float(g.factora07.sum())))
pd.DataFrame(rows).to_csv(TAB_DIR / "design_structure.csv", index=False)
checks["weights_all_positive"] = bool((a.factora07 > 0).all())
checks["missing_weights"] = int(a.factora07.isna().sum())

# ---- 8. overlap of clusters, households and persons between training and test years (panel component)
tr = a[a.year.isin(TRAIN_YEARS)]
keyh = lambda g: g.conglome + "_" + g.vivienda + "_" + g.hogar
rows = [dict(year=y, n=int((a.year == y).sum()),
             share_cluster_in_training=float(a[a.year == y].conglome.isin(set(tr.conglome)).mean()),
             share_household_code_in_training=float(keyh(a[a.year == y]).isin(set(keyh(tr))).mean()),
             share_person_panel_code_in_training=float(a[a.year == y].p201p.isin(set(tr.p201p)).mean())) for y in YEARS if y not in TRAIN_YEARS]
pd.DataFrame(rows).to_csv(TAB_DIR / "panel_overlap.csv", index=False)

# ---- 9. feature ranges and missingness by year
fs = a.groupby("year")[FEATURES].agg(["mean", "min", "max"]).round(4)
fs.columns = [f"{c}_{s}" for c, s in fs.columns]; fs.T.to_csv(TAB_DIR / "feature_summary_by_year.csv")
miss = a.groupby("year")[FEATURES].apply(lambda g: g.isna().mean()).round(4)
checks["max_missing_share_any_feature_year"] = float(miss.values.max())
checks["age_range_ok"] = bool(a.age.between(12, 17).all()); checks["overage_range"] = [float(a.overage.min()), float(a.overage.max())]
checks["school_years_range"] = [float(a.school_years.min()), float(a.school_years.max())]
checks["negative_overage_share"] = float((a.overage < 0).mean())
checks["head_age_below_15"] = int((a.head_age < 15).sum()); checks["hh_size_max"] = float(a.hh_size.max())
checks["n_kids_zero_or_missing"] = int((a.n_kids.fillna(0) < 1).sum())

# ---- 10. main reason for not being enrolled (P313), non-enrolled adolescents, by period
labels, values, _ = module_meta(RAW_DIR, 2019, "300")
ne = a[a.y == 1].assign(period=lambda g: np.select([g.year <= 2019, g.year <= 2021], ["2017-2019", "2020-2021"], "2022-2025"))
rs = (ne.groupby(["period", "p313"], dropna=False).factora07.sum() / ne.groupby("period").factora07.sum()).rename("share").reset_index()
rs["reason_2019_label"] = rs.p313.map(values.get("p313", {}))
rs.to_csv(TAB_DIR / "reasons_not_enrolled.csv", index=False)
checks["p313_answered_share_among_non_enrolled"] = float(ne.p313.notna().mean())
checks["enrolled_last_year_share_among_non_enrolled (P303=1)"] = float((ne.p303 == 1).mean())

json.dump(checks, open(MET_DIR / "data_validation.json", "w"), indent=1)
print(json.dumps(checks, indent=1))
print(prev[prev.area == "National"][["year", "n", "non_enrolled", "prev", "lo_taylor", "hi_taylor", "lo_boot", "hi_boot"]].round(4).to_string(index=False))
print(bm.round(1).T.to_string()); print(ex.T.to_string()); print(it.round(1).to_string(index=False))

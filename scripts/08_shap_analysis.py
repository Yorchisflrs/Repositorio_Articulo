"""Step 8 - E5: explanations of the frozen XGBoost model.

Global, temporal and subgroup SHAP importance; family-level attributions; stability over cluster-bootstrap
refits of the training data; permutation importance; dependence of the main predictor; predictor correlation.
Importance describes how the model uses a predictor; it is not a causal effect.

Outputs: results/metrics/E5_shap_importance.csv, E5_shap_family.csv, E5_shap_by_subgroup.csv,
         E5_shap_stability.csv, E5_permutation_importance.csv, E5_shap_dependence_overage.csv,
         E5_feature_correlation.csv, E5_shap_signs.csv;  data/processed/shap_frozen_xgb.npz (for figures)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, FEATURES, MET_DIR, MODEL_DIR, PROC_DIR, SEED, TEST_YEARS, TRAIN_YEARS  # noqa: E402
from src.explainability.shap_utils import FAMILIES, family_mean_abs, mean_abs, permutation_importance, shap_values  # noqa: E402
from src.modeling.models import fit_model  # noqa: E402

N_REFITS, N_PERM = 30, 30
d = pd.read_parquet(PROC_DIR / "analytic.parquet")
hp = json.load(open(MET_DIR / "hyperparameters.json"))["XGB_uw"]["final"]
model = pickle.load(open(MODEL_DIR / "XGB_uw.pkl", "rb"))
sv = shap_values(model, d[FEATURES]); w = d.factora07.values
tr = d.year.isin(TRAIN_YEARS).values; post = d.year.isin(CFG["subgroup_pool"]).values

# ---- importance by period and by year (survey-weighted mean |SHAP|)
cols = {"Training 2017-2019": tr, "Pandemic 2020-2021": d.year.isin([2020, 2021]).values, "Post-pandemic 2022-2025": post}
cols.update({str(y): (d.year == y).values for y in TEST_YEARS})
imp = pd.DataFrame({k: mean_abs(sv[m], w[m]) for k, m in cols.items()}).sort_values("Training 2017-2019", ascending=False)
imp.insert(0, "rank_training", range(1, len(imp) + 1))
imp["rank_post"] = imp["Post-pandemic 2022-2025"].rank(ascending=False).astype(int)
imp.to_csv(MET_DIR / "E5_shap_importance.csv", index_label="feature")
fam = pd.DataFrame({k: family_mean_abs(sv[m], w[m]) for k, m in cols.items()}); fam.to_csv(MET_DIR / "E5_shap_family.csv", index_label="family")

# ---- importance by subgroup (post-pandemic pool)
sg = {"Urban": d.rural == 0, "Rural": d.rural == 1, "Non-poor": d.poor == 0, "Poor": d.poor == 1, "Spanish": d.indig_tongue == 0,
      "Indigenous tongue": d.indig_tongue == 1, "Male": d.female == 0, "Female": d.female == 1}
pd.DataFrame({k: mean_abs(sv[post & m.values], w[post & m.values]) for k, m in sg.items()}).loc[imp.index].to_csv(MET_DIR / "E5_shap_by_subgroup.csv", index_label="feature")

# ---- direction of the attributions quoted in the text
rows = []
for f in FEATURES:
    x = d.loc[tr, f].fillna(d.loc[tr, f].median()).values; s = sv[tr, FEATURES.index(f)]
    hi = x > np.median(x) if len(np.unique(x)) > 2 else x == 1
    rows.append(dict(feature=f, mean_shap_high_or_1=s[hi].mean() if hi.any() else np.nan, mean_shap_low_or_0=s[~hi].mean() if (~hi).any() else np.nan,
                     spearman_value_shap=spearmanr(x, s)[0]))
pd.DataFrame(rows).to_csv(MET_DIR / "E5_shap_signs.csv", index=False)

# ---- stability: refit on cluster-bootstrap samples of the training years, explain a fixed sample
rng = np.random.default_rng(SEED); T = d[tr].reset_index(drop=True)
ref_idx = rng.choice(len(T), 5000, replace=False); Xref = T.loc[ref_idx, FEATURES]
base_imp = np.abs(shap_values(model, Xref)).mean(0); psu_idx = T.groupby("conglome").indices; psus = np.array(list(psu_idx))
B = []
for b in range(N_REFITS):
    take = np.concatenate([psu_idx[k] for k in rng.choice(psus, len(psus))])
    mb = fit_model("XGB", hp, T.loc[take, FEATURES], T.y.values[take], seed=SEED + 1 + b)
    B.append(np.abs(shap_values(mb, Xref)).mean(0))
B = np.array(B); ranks = (-B).argsort(1).argsort(1) + 1
stab = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": base_imp, "refit_mean": B.mean(0), "refit_p2.5": np.percentile(B, 2.5, 0),
                     "refit_p97.5": np.percentile(B, 97.5, 0), "rank_original": (-base_imp).argsort().argsort() + 1,
                     "rank_min": ranks.min(0), "rank_max": ranks.max(0)}).sort_values("rank_original")
stab.to_csv(MET_DIR / "E5_shap_stability.csv", index=False)
rho = [spearmanr(base_imp, b)[0] for b in B]
top5 = set(np.argsort(-base_imp)[:5])
summary = dict(n_refits=N_REFITS, spearman_rank_corr_min=float(min(rho)), spearman_rank_corr_mean=float(np.mean(rho)),
               share_refits_same_top5_set=float(np.mean([set(np.argsort(-b)[:5]) == top5 for b in B])),
               share_refits_same_top1=float(np.mean([np.argmax(b) == np.argmax(base_imp) for b in B])),
               spearman_training_vs_post=float(spearmanr(imp["Training 2017-2019"], imp["Post-pandemic 2022-2025"])[0]))

# ---- permutation importance on the post-pandemic pool (features and families)
P = d[post]
pi = permutation_importance(model, P[FEATURES], P.y.values, P.factora07.values, {f: [f] for f in FEATURES}, N_PERM, SEED)
pf = permutation_importance(model, P[FEATURES], P.y.values, P.factora07.values, FAMILIES, N_PERM, SEED + 1)
pd.concat([pi.assign(level="feature"), pf.assign(level="family")]).to_csv(MET_DIR / "E5_permutation_importance.csv", index=False)
summary["spearman_shap_vs_permutation"] = float(spearmanr(imp["Post-pandemic 2022-2025"].loc[pi.feature], pi.auc_drop)[0])

# ---- dependence of the main predictor and correlation among predictors
k = FEATURES.index("overage"); T2 = d[tr]
dep = T2.assign(shap=sv[tr, k]).groupby("overage").apply(lambda g: pd.Series(dict(
    n=len(g), share=g.factora07.sum() / T2.factora07.sum(), observed_rate=np.average(g.y, weights=g.factora07), shap_mean=g["shap"].mean(),
    shap_p10=g["shap"].quantile(0.1), shap_p90=g["shap"].quantile(0.9)))).reset_index()
dep.to_csv(MET_DIR / "E5_shap_dependence_overage.csv", index=False)
d.loc[tr, FEATURES].corr(method="spearman").round(3).to_csv(MET_DIR / "E5_feature_correlation.csv")
summary["overage_identity_share"] = float((d.overage == (d.age - 6 - d.school_years).clip(-2, 12)).mean())
json.dump(summary, open(MET_DIR / "E5_shap_summary.json", "w"), indent=1)
np.savez_compressed(PROC_DIR / "shap_frozen_xgb.npz", sv=sv.astype("float32"), train=tr, post=post)

print(imp[["rank_training", "Training 2017-2019", "Post-pandemic 2022-2025", "rank_post"]].head(10).round(3).to_string())
print(fam[["Training 2017-2019", "Post-pandemic 2022-2025"]].round(3).to_string())
print(json.dumps(summary, indent=1)); print(pf.round(4).to_string(index=False)); print(pi.head(6).round(4).to_string(index=False))
print(dep.round(3).to_string(index=False))

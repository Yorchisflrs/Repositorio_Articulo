"""Step 4 - E1: nested grouped cross-validation on 2017-2019 and final (frozen) models.

Each algorithm (LR, RF, XGB) is trained twice: without weights (original strategy) and with the
expansion factors as training weights. Hyper-parameters are selected by nested grouped CV; the final
configuration is selected by grouped CV on the whole training period (never on test years).

Outputs: data/processed/models/<ALG>_<uw|w>.pkl        frozen models
         data/processed/predictions.parquet             OOF (training years) and frozen predictions (all years)
         results/metrics/hyperparameters.json           configuration chosen per outer fold and final one
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, FEATURES, MET_DIR, MODEL_DIR, PROC_DIR, TRAIN_YEARS  # noqa: E402
from src.evaluation.metrics import auc_ap  # noqa: E402
from src.modeling.models import MODELS, fit_model, predict  # noqa: E402
from src.validation.splits import nested_cv, select_params  # noqa: E402

d = pd.read_parquet(PROC_DIR / "analytic.parquet")
tr_mask = d.year.isin(TRAIN_YEARS).values
tr = d[tr_mask].reset_index(drop=True)
X, y, w, groups = tr[FEATURES], tr.y.values, tr.factora07.values, tr.conglome.values
outer, inner = CFG["cv"]["outer_folds"], CFG["cv"]["inner_folds"]
pred = d[["year"]].copy(); hp = {}
for name in MODELS:
    for tag, tw in (("uw", None), ("w", w)):
        t0 = time.time(); key = f"{name}_{tag}"
        oof, chosen = nested_cv(name, CFG["grids"][name], X, y, w, groups, outer, inner, tw)
        final, score = select_params(name, CFG["grids"][name], X, y, w, groups, outer, tw)
        model = fit_model(name, final, X, y, tw)
        pickle.dump(model, open(MODEL_DIR / f"{key}.pkl", "wb"))
        pred[f"oof_{key}"] = np.nan; pred.loc[tr_mask, f"oof_{key}"] = oof
        pred[f"p_{key}"] = predict(model, d[FEATURES])
        a, ap = auc_ap(y, oof, w)
        hp[key] = dict(final=final, final_cv_prauc=score, chosen_per_outer_fold=chosen, nested_auc=float(a[0]), nested_prauc=float(ap[0]))
        print(f"{key}: nested AUC={a[0]:.4f} PR-AUC={ap[0]:.4f} final={final} ({time.time() - t0:.0f}s)", flush=True)
        pred.to_parquet(PROC_DIR / "predictions.parquet")
        json.dump(hp, open(MET_DIR / "hyperparameters.json", "w"), indent=1)
print("done")

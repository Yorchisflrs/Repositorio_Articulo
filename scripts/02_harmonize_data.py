"""Step 2 - Harmonise variables across 2017-2025 and build the analytic population.

Inputs : data/processed/persons_10_18.parquet
Outputs: data/processed/harmonized_10_18.parquet   (all persons 10-18 with harmonised variables)
         data/processed/analytic.parquet           (analytic population: 12-17, April-December, ...)
         results/tables/sample_flow.csv            (exclusions, per year)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES, PROC_DIR, TAB_DIR  # noqa: E402
from src.preprocessing.features import add_features, analytic_sample  # noqa: E402

d = add_features(pd.read_parquet(PROC_DIR / "persons_10_18.parquet"))
d.to_parquet(PROC_DIR / "harmonized_10_18.parquet")
s, flow = analytic_sample(d)
s["y"] = s.y.astype(int)
s = s.reset_index(drop=True)
s.to_parquet(PROC_DIR / "analytic.parquet")
flow["excluded_at_step"] = -flow.total.diff().fillna(0).astype(int)
flow.to_csv(TAB_DIR / "sample_flow.csv")
print(flow[["total", "excluded_at_step"]].to_string())
print("analytic:", len(s), "non-enrolled:", int(s.y.sum()))
print("missing share:", s[FEATURES].isna().mean()[lambda x: x > 0].round(4).to_dict())
print(s.groupby("year").y.agg(["size", "sum"]).T.to_string())

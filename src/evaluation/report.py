"""Helpers shared by the evaluation scripts: design identifiers, replicate weights and metric rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import N_BOOT, SEED
from src.evaluation.metrics import all_metrics, summarize
from src.validation.bootstrap import replicate_weights


def design_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(strata, psu) for the rows passed.

    One year: strata = ESTRATO, PSU = CONGLOME (INEI's variance specification).
    Several years pooled: a cluster code revisited in different years is a single PSU, assigned to the
    stratum of its first appearance, so repeated visits are resampled together.
    """
    if df.year.nunique() == 1:
        return df.estrato.astype(int).astype(str).values, df.conglome.values
    first = df.sort_values("year").drop_duplicates("conglome").set_index("conglome").estrato.astype(int).astype(str)
    return df.conglome.map(first).values, df.conglome.values


def rep_weights(df: pd.DataFrame, n_rep: int = N_BOOT, seed_offset: int = 0) -> np.ndarray:
    strata, psu = design_ids(df)
    return replicate_weights(df.factora07.values, strata, psu, n_rep, SEED + seed_offset)


def metric_row(y, p, W, thr=None, **labels) -> dict:
    """One tidy row: labels + n, events and every metric with its percentile interval."""
    y = np.asarray(y)
    row = dict(labels); row.update(n=int(len(y)), events=int(y.sum()))
    row.update(summarize(all_metrics(y, p, W, thr)))
    return row


def fmt_ci(x, lo, hi, d=3) -> str:
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

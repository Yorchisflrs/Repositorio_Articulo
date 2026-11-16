"""Harmonised variables and the analytic population.

Every transformation is listed with its ENAHO source in docs/data_dictionary.md. The coding follows the
official dictionaries of each year; changes between years are handled here and nowhere else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CFG

INDIGENOUS_CODES = [1, 2, 3, 10, 11, 12, 13, 14, 15]   # Quechua, Aymara, other native; 10-15 Amazonian (from 2020)
HEAD_EDUC_YEARS = {1: 0, 2: 0, 3: 3, 4: 6, 5: 8, 6: 11, 7: 13, 8: 14, 9: 14, 10: 16, 11: 18, 12: np.nan}


def years_of_schooling(p301a, p301b, p301c) -> np.ndarray:
    """Years approved. Primary grade is recorded in P301C and secondary year in P301B.

    P301A: 1 none, 2 pre-school, 3 incomplete primary, 4 complete primary, 5 incomplete secondary,
    6 complete secondary, 7-11 post-secondary, 12 special basic education (not mapped: NaN).
    """
    gp = pd.Series(p301c).fillna(0).clip(0, 5).values
    gs = pd.Series(p301b).fillna(0).clip(0, 4).values
    a = np.asarray(p301a, dtype=float)
    return np.select([a <= 2, a == 3, a == 4, a == 5, (a >= 6) & (a <= 11)], [0, gp, 6, 6 + gs, 11], np.nan)


def weighted_quintile(x: pd.Series, w: pd.Series) -> pd.Series:
    """Quintile (1-5) of x in the weighted distribution of the rows passed."""
    o = np.argsort(x.values, kind="stable"); cw = np.cumsum(w.values[o]) / w.sum()
    q = np.empty(len(x)); q[o] = np.minimum((cw * 5).astype(int) + 1, 5)
    return pd.Series(q, index=x.index)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Row-wise harmonised variables (no sample-dependent quantities)."""
    d = d.copy()
    d["y"] = np.where(d.p306.notna(), (d.p306 == 2).astype(float), np.nan)        # 1 = not enrolled this year
    d["age"] = d.p208a.astype(int)
    d["female"] = (d.p207 == 2).astype(int)
    d["indig_tongue"] = d.p300a.isin(INDIGENOUS_CODES).astype(int)
    d["school_years"] = years_of_schooling(d.p301a, d.p301b, d.p301c)
    d["overage"] = (d.age - 6 - d.school_years).clip(-2, 12)
    d["child_of_head"] = (d.p203 == 3).astype(int)
    d["hh_mobile"] = (d.p1142 == 1).astype(int)
    d["hh_computer"] = np.where(d.computer.isna(), np.nan, (d.computer == 1).astype(float))
    d["hh_internet_raw"] = (d.p1144 == 1).astype(int)
    # fixed connection: P1144 up to 2022, P114B1 from 2023 (missing by skip pattern = no internet = 0)
    d["hh_internet_fixed"] = np.where(d.year <= 2022, d.hh_internet_raw, (d.p114b1 == 1).astype(int))
    d["poor"] = d.pobreza.isin([1, 2]).astype(int)
    d["extreme_poor"] = (d.pobreza == 1).astype(int)
    d["pcexp"] = d.gashog2d / d.mieperho
    d["head_educ_years"] = d.head_educ.map(HEAD_EDUC_YEARS)
    d["head_female"] = (d.head_sex == 2).astype(int)
    d["hh_size"] = d.mieperho
    d["rural"] = (d.estrato >= 6).astype(int)
    d["inadequate_housing"] = (d.nbi1 == 1).astype(int)
    for k in range(1, 9):
        d[f"dom_{k}"] = (d.dominio == k).astype(int)
    d["dept"] = d.ubigeo.str[:2]
    d["psu"] = d.year.astype(str) + "_" + d.conglome
    d["hhid"] = d.psu + "_" + d.vivienda + "_" + d.hogar
    d["telephone"] = (d.tipoentrevista == 2).astype(int) if "tipoentrevista" in d else 0
    return d


def flow_masks(d: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Successive inclusion criteria of the analytic population (applied in this order)."""
    pop = CFG["population"]
    return [
        ("Listed in the household roster, aged 12-17", d.p208a.between(pop["age_min"], pop["age_max"])),
        ("Household member (P204 = 1)", d.p204 == 1),
        ("Not absent 30 days or more (P205 != 1)", d.p205 != 1),
        ("Education module answered (module 300 record)", d.m300 == 1),
        ("Attainment reported and secondary not completed (P301A < 6)", d.p301a < 6),
        ("Enrollment item answered (P306)", d.p306.notna()),
        ("Interviewed April-December", d.mes.between(pop["first_month"], pop["last_month"])),
    ]


def analytic_sample(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the inclusion criteria; returns (sample, flow table with counts per year)."""
    keep = pd.Series(True, index=d.index); rows = []
    for label, m in flow_masks(d):
        keep &= m.fillna(False)
        r = d[keep].groupby("year").size().to_dict(); r.update(step=label, total=int(keep.sum()))
        rows.append(r)
    s = d[keep].copy()
    # quintile inside the analytic sample of each year (sample-dependent, hence computed after filtering)
    s["exp_quintile"] = s.groupby("year", group_keys=False).apply(lambda g: weighted_quintile(g.pcexp, g.factora07))
    flow = pd.DataFrame(rows).set_index("step")
    return s, flow

"""Harmonisation rules and recalibration helpers on small hand-made cases."""
import numpy as np
import pandas as pd

from src.calibration import recalibration as rc
from src.evaluation.metrics import citl
from src.preprocessing.features import INDIGENOUS_CODES, weighted_quintile, years_of_schooling


def test_years_of_schooling_uses_grade_for_primary_and_year_for_secondary():
    #                 none  preschool  prim.inc(3rd)  prim.complete  sec.inc(2nd)  sec.complete  special
    p301a = [1, 2, 3, 4, 5, 6, 12]
    p301b = [np.nan, np.nan, np.nan, np.nan, 2, 5, np.nan]      # year (secondary)
    p301c = [np.nan, np.nan, 3, 6, np.nan, np.nan, np.nan]      # grade (primary)
    out = years_of_schooling(p301a, p301b, p301c)
    assert list(out[:6]) == [0, 0, 3, 6, 8, 11] and np.isnan(out[6])


def test_indigenous_codes_include_amazonian_languages_added_in_2020():
    assert {1, 2, 3, 10, 15} <= set(INDIGENOUS_CODES) and 4 not in INDIGENOUS_CODES     # 4 = Spanish


def test_weighted_quintiles_have_equal_weight():
    rng = np.random.default_rng(0); x = pd.Series(rng.lognormal(size=5000)); w = pd.Series(rng.gamma(2, 1, 5000))
    q = weighted_quintile(x, w); share = w.groupby(q).sum() / w.sum()
    assert set(q.unique()) == {1, 2, 3, 4, 5} and (share - 0.2).abs().max() < 0.01
    assert x[q == 1].max() <= x[q == 2].min()


def test_intercept_recalibration_removes_citl():
    rng = np.random.default_rng(3); p = rng.uniform(0.01, 0.3, 20000); y = (rng.random(len(p)) < p * 1.5).astype(int); w = rng.gamma(2, 1, len(p))
    a = rc.fit_intercept(y, p, w)
    assert a > 0 and abs(citl(y, rc.apply(p, a), w)[0]) < 1e-6


def test_weighted_quantile_threshold():
    x = np.arange(1, 101, dtype=float); w = np.ones(100)
    assert rc.weighted_quantile(x, w, 0.90) == 90.0

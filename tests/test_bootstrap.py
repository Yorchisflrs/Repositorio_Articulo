"""Properties of the Rao-Wu rescaled bootstrap and of the linearised variance."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import wmean
from src.validation.bootstrap import collapse_singletons, paired_diff, replicate_weights, taylor_ratio_se


def test_first_row_is_original_and_totals_are_unbiased(toy):
    W = replicate_weights(toy["w"], toy["strata"], toy["psu"], 800, 1)
    assert W.shape == (801, len(toy["w"])) and np.allclose(W[0], toy["w"])
    assert abs(W[1:].sum(1).mean() / toy["w"].sum() - 1) < 0.01
    assert (W >= 0).all()


def test_whole_psus_are_resampled(toy):
    W = replicate_weights(toy["w"], toy["strata"], toy["psu"], 20, 3)
    ratio = pd.DataFrame(np.round(W[1:] / toy["w"], 8).T).groupby(toy["psu"]).nunique()
    assert (ratio.values == 1).all()          # every person of a PSU gets the same factor in a replicate


def test_bootstrap_se_close_to_linearisation(toy):
    W = replicate_weights(toy["w"], toy["strata"], toy["psu"], 1500, 5)
    se_boot = wmean(toy["y"], W)[1:].std(ddof=1)
    _, se_lin = taylor_ratio_se(toy["y"], toy["w"], toy["strata"], toy["psu"])
    assert abs(se_boot / se_lin - 1) < 0.10


def test_single_psu_strata_are_collapsed():
    strata = pd.Series(["a"] * 6 + ["b"] * 2 + ["c"] * 6); psu = pd.Series([1, 1, 2, 2, 3, 3, 9, 9, 4, 4, 5, 5, 6, 6])
    new, merged = collapse_singletons(strata, psu)
    assert merged == 1 and "b" not in set(new)
    W = replicate_weights(np.ones(14), strata, psu, 50, 0)
    assert np.isfinite(W).all()


def test_paired_difference_is_zero_for_identical_inputs():
    a = np.random.default_rng(0).normal(size=200); r = paired_diff(a, a.copy())
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0

"""Qualitative statements of the manuscript checked against the result files.

Each test quotes the sentence it protects. If a re-run of the pipeline changes a result so that the sentence
is no longer true, the test fails and the text must be revised.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

MET = Path(__file__).resolve().parents[1] / "results" / "metrics"
needs = pytest.mark.skipif(not (MET / "E4_subgroups.csv").exists(), reason="result files not available")


def excludes_zero(df):
    return (df.lo > 0) | (df.hi < 0)


@needs
def test_frozen_model_matches_same_year_reference():
    """'the year was harder rather than the model stale' / 'the gap never exceeds ...'"""
    r = pd.read_csv(MET / "E2_in_year_reference.csv")
    x = r[(r.model == "XGB") & (r.metric == "auc") & (r.year == 2020)].iloc[0]
    assert x.lo < 0 < x.hi


@needs
def test_calibration_recovers_from_2022():
    """'From 2022 onward the intervals of CITL include zero for the three models in every year.'"""
    p = pd.read_csv(MET / "E1_E2_performance.csv")
    x = p[(p.training == "unweighted") & p.test_set.isin([str(y) for y in range(2022, 2026)])]
    assert ((x.citl_lo < 0) & (x.citl_hi > 0)).all()
    y = p[(p.training == "unweighted") & p.test_set.isin(["2020", "2021"])]
    assert (y.citl_lo > 0).all(), "the three models under-predict risk in 2020 and 2021"


@needs
def test_weighted_training_raises_citl_everywhere():
    """'raises CITL by ... in every set'"""
    d = pd.read_csv(MET / "E2_paired_differences.csv")
    x = d[(d.contrast == "XGB_w - XGB_uw") & (d.metric == "citl")]
    assert (x["diff"] > 0).all() and len(x) == 7


@needs
def test_frozen_model_best_calibrated_after_2021():
    """'a model that was left alone ended up better calibrated than one that followed it'"""
    r = pd.read_csv(MET / "E3_rolling_origin.csv")
    x = r[(r.model == "XGB") & r.year.between(2022, 2025)].groupby("strategy").citl.apply(lambda s: s.abs().mean())
    assert x.idxmin() == "Frozen (2017-2019)"


@needs
def test_subgroup_discrimination_similar_and_calibration_gaps():
    """'no paired gap in ROC-AUC has an interval that excludes zero' and the three calibration gaps."""
    g = pd.read_csv(MET / "E4_subgroup_gaps.csv")
    post = g[g.period == "Post-pandemic 2022-2025"]
    assert not excludes_zero(post[post.metric == "auc"]).any()
    for contrast in ("Rural - Urban", "Indigenous tongue - Spanish", "Female - Male"):
        x = post[(post.metric == "citl") & (post.contrast == contrast)]
        assert excludes_zero(x).all(), f"{contrast}: gap should exclude zero for LR, RF and XGB"
    x = post[(post.metric == "citl") & (post.contrast == "Poor - Non-poor") & (post.model == "XGB")]
    assert not excludes_zero(x).any(), "poor and non-poor are calibrated alike"
    pre = g[(g.period == "In-period CV 2017-2019") & (g.metric == "citl") & (g.model == "XGB")]
    assert not excludes_zero(pre).any(), "no calibration gap in the training period for XGB"


@needs
def test_shap_top_two_stable():
    """'age-grade distortion ranked first and age second every time'"""
    s = pd.read_csv(MET / "E5_shap_stability.csv").set_index("feature")
    assert s.loc["overage", "rank_min"] == s.loc["overage", "rank_max"] == 1
    assert s.loc["age", "rank_min"] == s.loc["age", "rank_max"] == 2


@needs
def test_incident_model_does_not_close_the_gap():
    """'A model trained directly on the incident outcome does no better'"""
    t = pd.read_csv(MET / "leakage_outcome_types.csv")
    i = pd.read_csv(MET / "leakage_incident_model.csv")
    frozen = t[(t.period == "2022-2025") & (t.model == "XGB") & (t.contrast == "Incident vs enrolled")].auc.iloc[0]
    own = i[(i.model == "XGB") & (i.predictor_set == "Full model") & (i.test_set == "2022-2025")].auc.iloc[0]
    assert own <= frozen + 0.01


@needs
def test_panel_followup_is_balanced():
    """Balance of followed and not-followed baselines (reported in the audit)."""
    b = pd.read_csv(MET / "panel_followup_balance.csv")
    assert b.std_diff.abs().max() < 0.1


@needs
def test_bootstrap_p_values_are_probabilities():
    for f in MET.glob("*.csv"):
        d = pd.read_csv(f)
        if "p_boot" in d:
            assert ((d.p_boot.dropna() >= 0) & (d.p_boot.dropna() <= 1)).all(), f.name


@needs
def test_threshold_from_out_of_fold_is_lower_than_resubstitution():
    t = json.load(open(MET / "thresholds.json"))
    for m in ("RF", "XGB"):
        assert t["oof_weighted"][f"{m}_uw"] < t["resubstitution_unweighted"][f"{m}_uw"]

"""The vectorised survey-weighted metrics must agree with scikit-learn and statsmodels."""
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.metrics import all_metrics, auc_ap, brier, calib_line, citl, logit, summarize, threshold_metrics


def test_auc_ap_brier_match_sklearn_with_ties(toy):
    W = np.vstack([toy["w"], np.ones_like(toy["w"])])
    a, ap = auc_ap(toy["y"], toy["p"], W)
    for k in range(2):
        assert abs(a[k] - roc_auc_score(toy["y"], toy["p"], sample_weight=W[k])) < 1e-10
        assert abs(ap[k] - average_precision_score(toy["y"], toy["p"], sample_weight=W[k])) < 1e-10
        assert abs(brier(toy["y"], toy["p"], W)[k] - brier_score_loss(toy["y"], toy["p"], sample_weight=W[k])) < 1e-12


def test_calibration_matches_glm(toy):
    y, p, w = toy["y"], toy["p"], toy["w"]; fw = w / w.mean()
    ref_citl = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), offset=logit(p), freq_weights=fw).fit().params[0]
    ref_line = sm.GLM(y, sm.add_constant(logit(p)), family=sm.families.Binomial(), freq_weights=fw).fit().params
    a, b = calib_line(y, p, w)
    assert abs(citl(y, p, w)[0] - ref_citl) < 1e-7 and abs(a[0] - ref_line[0]) < 1e-6 and abs(b[0] - ref_line[1]) < 1e-6


def test_citl_sign_convention():
    """Positive CITL means the model under-predicts risk; negative means it over-predicts."""
    rng = np.random.default_rng(1); p = rng.uniform(0.02, 0.2, 20000); w = np.ones(len(p))
    under = (rng.random(len(p)) < p * 1.6).astype(int); over = (rng.random(len(p)) < p * 0.6).astype(int)
    assert citl(under, p, w)[0] > 0.2 and citl(over, p, w)[0] < -0.2


def test_slope_convention():
    """Slope below 1 when predictions are too extreme, above 1 when they are too timid."""
    rng = np.random.default_rng(2); l = rng.normal(-3, 1.2, 40000); y = (rng.random(len(l)) < 1 / (1 + np.exp(-l))).astype(int); w = np.ones(len(l))
    extreme = 1 / (1 + np.exp(-(-3 + 2.0 * (l + 3)))); timid = 1 / (1 + np.exp(-(-3 + 0.5 * (l + 3))))
    assert calib_line(y, extreme, w)[1][0] < 0.7 and calib_line(y, timid, w)[1][0] > 1.5


def test_threshold_metrics_confusion_matrix(toy):
    y, p, w = toy["y"], toy["p"], toy["w"]; t = threshold_metrics(y, p, w, 0.1); f = p >= 0.1
    assert np.isclose(t["tp"][0], w[f & (y == 1)].sum()) and np.isclose(t["tn"][0], w[~f & (y == 0)].sum())
    assert np.isclose(t["recall"][0], w[f & (y == 1)].sum() / w[y == 1].sum())
    assert np.isclose(t["specificity"][0], w[~f & (y == 0)].sum() / w[y == 0].sum())
    assert np.isclose(t["tp"][0] + t["fp"][0] + t["fn"][0] + t["tn"][0], w.sum())


def test_summarize_shapes(toy):
    W = np.vstack([toy["w"]] * 5); s = summarize(all_metrics(toy["y"], toy["p"], W, 0.1))
    assert {"auc", "auc_lo", "auc_hi", "citl", "slope", "recall", "precision", "f1", "specificity"} <= set(s)

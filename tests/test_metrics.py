"""Tests for creditrisk.metrics, on synthetic data (no Kaggle download)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from creditrisk import metrics


def reference_gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    """The competition's reference implementation, copied for comparison."""

    gini_in_time = (
        base.loc[:, ["WEEK_NUM", "target", "score"]]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1)
        .tolist()
    )
    x = np.arange(len(gini_in_time))
    a, b = np.polyfit(x, gini_in_time, 1)
    res_std = np.std(gini_in_time - (a * x + b))
    return np.mean(gini_in_time) + w_fallingrate * min(0, a) + w_resstd * res_std


@pytest.fixture
def scored():
    """20 weeks x 500 applications, ~10% defaults, an informative score."""

    rng = np.random.default_rng(0)
    n = 20 * 500
    target = rng.random(n) < 0.10
    score = 1 / (1 + np.exp(-(rng.normal(size=n) + 1.5 * target - 2)))
    return pd.DataFrame({
        "WEEK_NUM": np.repeat(np.arange(20), 500),
        "target": target.astype(int),
        "score": score,
    })


def test_gini_of_perfect_and_reversed_scores():
    y = [0, 0, 1, 1]
    assert metrics.gini(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.gini(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(-1.0)


def test_matches_the_competition_reference(scored):
    assert metrics.gini_stability(scored) == pytest.approx(
        reference_gini_stability(scored))


def test_components_add_up(scored):
    parts = metrics.stability_components(metrics.weekly_gini(scored))
    expected = (parts["mean_gini"] + 88.0 * min(0.0, parts["slope"])
                - 0.5 * parts["residual_std"])
    assert parts["stability"] == pytest.approx(expected)


def test_decaying_model_is_penalised():
    stable = pd.Series(np.full(50, 0.5))
    decaying = pd.Series(np.linspace(0.6, 0.4, 50))  # same mean, falling
    assert metrics.stability_components(decaying)["stability"] < \
        metrics.stability_components(stable)["stability"]


def test_improving_model_gets_no_bonus():
    stable = pd.Series(np.full(50, 0.5))
    improving = pd.Series(np.linspace(0.4, 0.6, 50))
    assert metrics.stability_components(improving)["stability"] == \
        pytest.approx(metrics.stability_components(stable)["stability"])


def test_single_class_weeks_are_skipped(scored):
    scored.loc[scored["WEEK_NUM"] == 3, "target"] = 0
    assert 3 not in metrics.weekly_gini(scored).index


def test_hard_labels_understate_the_metric_and_warn(scored):
    """Scoring with predict() instead of predict_proba() deflates the metric."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")       # probabilities: no warning
        from_probabilities = metrics.gini_stability(scored)

    labels = scored.assign(score=(scored["score"] > 0.5).astype(int))
    with pytest.warns(UserWarning, match="predict_proba"):
        from_labels = metrics.gini_stability(labels)

    assert from_labels < 0.5 * from_probabilities

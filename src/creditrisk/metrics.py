"""Evaluation metrics for the Home Credit stability competition.

The competition does not rank models by a single AUC. It rewards models whose
discriminative power is *high and does not decay over time*:

1. For every week ``w`` of loan decisions, compute the Gini coefficient
   ``gini_w = 2 * AUC_w - 1``.
2. Fit a straight line ``gini_w ~ a * w + b`` through the weekly values.
3. Combine three terms::

       stability = mean(gini_w) + 88 * min(0, a) - 0.5 * std(residuals)

   * ``mean(gini_w)`` rewards ranking defaulters above non-defaulters.
   * ``88 * min(0, a)`` penalises a downward trend (a model that is getting
     worse); an upward trend earns no bonus.
   * ``0.5 * std(residuals)`` penalises week-to-week volatility.

AUC and Gini are *ranking* metrics: they need a continuous score per
application (for example ``model.predict_proba(X)[:, 1]``). Passing hard 0/1
class labels (``model.predict(X)``) collapses the ranking to a single
threshold and badly understates both metrics. Functions here warn when the
scores look like class labels.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FALLING_RATE_WEIGHT = 88.0
RESIDUAL_STD_WEIGHT = -0.5


def gini(y_true, y_score) -> float:
    """Gini coefficient, ``2 * AUC - 1``.

    0 means the scores carry no ranking information; 1 means every defaulter
    is scored above every non-defaulter.
    """

    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def warn_if_class_labels(y_score, name: str = "y_score") -> None:
    """Warn when ``y_score`` looks like hard class labels, not probabilities."""

    if pd.Series(y_score).nunique(dropna=True) <= 2:
        warnings.warn(
            f"{name} has at most two distinct values. AUC/Gini need a "
            "continuous score such as predict_proba(X)[:, 1]; hard labels "
            "from predict(X) understate the metric.",
            UserWarning,
            stacklevel=3,
        )


def weekly_gini(
    df: pd.DataFrame,
    week_col: str = "WEEK_NUM",
    target_col: str = "target",
    score_col: str = "score",
) -> pd.Series:
    """Gini coefficient of the scores within each week.

    Parameters
    ----------
    df :
        One row per loan application, with the week, the true label and the
        model score.
    week_col, target_col, score_col :
        Column names.

    Returns
    -------
    pd.Series
        Gini per week, indexed by week in increasing order. Weeks containing
        a single class are dropped, because AUC is undefined there.
    """

    warn_if_class_labels(df[score_col], score_col)

    values = {}
    for week, group in df.groupby(week_col, sort=True):
        if group[target_col].nunique() < 2:
            continue
        values[week] = gini(group[target_col], group[score_col])

    return pd.Series(values, name="gini").rename_axis(week_col)


def stability_components(gini_by_week: pd.Series) -> dict:
    """Break the stability metric into its three ingredients.

    Following the competition's reference implementation, the trend is
    fitted against the position of each week (0, 1, 2, ...), not the week
    number itself.

    Returns
    -------
    dict
        ``mean_gini``, ``slope`` (change in Gini per week), ``residual_std``
        and the combined ``stability`` score.
    """

    y = np.asarray(gini_by_week, dtype=float)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    residual_std = float(np.std(y - (slope * x + intercept)))
    mean_gini = float(np.mean(y))

    return {
        "mean_gini": mean_gini,
        "slope": float(slope),
        "residual_std": residual_std,
        "stability": (
            mean_gini
            + FALLING_RATE_WEIGHT * min(0.0, slope)
            + RESIDUAL_STD_WEIGHT * residual_std
        ),
    }


def gini_stability(
    df: pd.DataFrame,
    week_col: str = "WEEK_NUM",
    target_col: str = "target",
    score_col: str = "score",
) -> float:
    """The competition's stability metric (higher is better).

    See the module docstring for the definition.
    """

    by_week = weekly_gini(df, week_col, target_col, score_col)
    return stability_components(by_week)["stability"]

"""Feature engineering used in the notebooks, as reusable functions.

Column-name suffixes in the Home Credit data describe the kind of variable:
``P`` days past due (dpd), ``A`` amount, ``D`` date, ``M`` masked category,
``T``/``L`` unspecified transforms. For example
``avgdbddpdlast24m_3658932P`` is an average days-past-due over 24 months.
"""
from typing import Mapping, Optional

import numpy as np
import pandas as pd

# Weights chosen by the team in notebook 01. They are judgement calls that
# favour long-term repayment behaviour over the most recent months; they were
# not tuned against the target.
AVG_DPD_WEIGHTS = {
    "avgdbddpdlast24m_3658932P": 0.8,
    "avgdbddpdlast3m_4187120P": 0.2,
}
MAX_DPD_WEIGHTS = {
    "maxdbddpdtollast12m_3658940P": 0.997,
    "maxdbddpdtollast6m_4187119P": 0.002,
    "maxdbddpdlast1m_3658939P": 0.001,
}
FIRST_PAYMENT_DPD_WEIGHTS = {
    "posfpd30lastmonth_3976960P": 0.2,
    "posfstqpd30lastmonth_3976962P": 0.8,
}


def parse_risk_band(band: Optional[str]) -> float:
    """Midpoint, in percent, of a credit-bureau risk band.

    The bureau reports the probability of default as a text range such as
    ``'6% - 8%'``. We summarise it by its midpoint (7.0).

    Parameters
    ----------
    band :
        Text like ``'59% - 66%'``, or a missing value.

    Returns
    -------
    float
        The midpoint in percent, or NaN if ``band`` is missing.
    """

    if pd.isna(band):
        return np.nan

    low, high = (float(part.strip().rstrip("%")) for part in band.split(" - "))
    return 0.5 * (low + high)


def weighted_mean(df: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Row-wise weighted mean that ignores missing values.

    For each row, the weights of the columns that are present are rescaled to
    sum to one, so a row with only one observed column simply returns that
    value. Rows where every column is missing return NaN.

    Parameters
    ----------
    df :
        Frame containing every column named in ``weights``.
    weights :
        Mapping from column name to its weight.
    """

    columns = list(weights)
    w = np.array([weights[c] for c in columns], dtype=float)
    values = df[columns].to_numpy(dtype=float)
    present = ~np.isnan(values)

    weight_sum = (present * w).sum(axis=1)
    weighted_sum = np.nansum(values * w, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

    return pd.Series(result, index=df.index)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the team's engineered features to a merged application table.

    Adds, when their source columns are present:

    * ``cb_risk_mean``: midpoint of the credit-bureau risk band, 0 if absent.
    * ``with_cb_risk``: 1 if the bureau provided a risk band (only about 3.5%
      of applications have one, so the indicator itself is informative).
    * ``weighted_avg_dpd``, ``weighted_maxdbddpdlastxm``, ``weighted_posfpd``:
      weighted combinations of days-past-due columns over different horizons.

    Returns a copy; the input is not modified.
    """

    out = df.copy()

    if "riskassesment_302T" in out:
        risk = out["riskassesment_302T"].map(parse_risk_band)
        out["with_cb_risk"] = risk.notna().astype(int)
        out["cb_risk_mean"] = risk.fillna(0.0)

    for name, weights in [
        ("weighted_avg_dpd", AVG_DPD_WEIGHTS),
        ("weighted_maxdbddpdlastxm", MAX_DPD_WEIGHTS),
        ("weighted_posfpd", FIRST_PAYMENT_DPD_WEIGHTS),
    ]:
        if all(column in out for column in weights):
            out[name] = weighted_mean(out, weights)

    return out

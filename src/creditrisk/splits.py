"""Train / validation / test splits.

The notebooks split applications at random, so all three sets cover the same
weeks. That answers "how well does the model rank applications from the period
it was trained on?". The business question is different: a credit model is
trained on past loans and used on future ones. ``time_based_split`` trains on
the earliest weeks and holds out the latest, which is the honest test of
stability over time.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Columns that identify a row or its date. They must not be model features:
# a model given the decision date or week can memorise period-specific default
# rates, which cannot transfer to future weeks.
NON_FEATURE_COLUMNS = ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]

Split = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def random_split(
    df: pd.DataFrame,
    train_size: float = 0.6,
    target_col: str = "target",
    random_state: int = 1,
) -> Split:
    """Stratified random 60/20/20 split, as used in the notebooks.

    ``random_state`` is fixed so results can be reproduced.
    """

    train, rest = train_test_split(
        df, train_size=train_size, stratify=df[target_col],
        random_state=random_state)
    valid, test = train_test_split(
        rest, train_size=0.5, stratify=rest[target_col],
        random_state=random_state)
    return train, valid, test


def time_based_split(
    df: pd.DataFrame,
    week_col: str = "WEEK_NUM",
    train_size: float = 0.6,
    valid_size: float = 0.2,
) -> Split:
    """Split by time: earliest weeks train, latest weeks test.

    Whole weeks are assigned to one set, so no week is shared between sets.
    Sizes are approximate because weeks contain different numbers of
    applications.
    """

    weeks = np.sort(df[week_col].unique())
    counts = df[week_col].value_counts().reindex(weeks).to_numpy()
    cumulative = np.cumsum(counts) / counts.sum()

    train_weeks = weeks[cumulative <= train_size]
    valid_weeks = weeks[
        (cumulative > train_size) & (cumulative <= train_size + valid_size)]

    in_train = df[week_col].isin(train_weeks)
    in_valid = df[week_col].isin(valid_weeks)
    return df[in_train], df[in_valid], df[~in_train & ~in_valid]


def feature_columns(df: pd.DataFrame) -> list:
    """Columns of ``df`` that may be used as model inputs."""

    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

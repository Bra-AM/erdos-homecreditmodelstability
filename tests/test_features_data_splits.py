"""Tests for feature engineering, joins and splits, on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from creditrisk import data, features, splits


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
@pytest.mark.parametrize("band, expected", [
    ("6% - 8%", 7.0),
    ("59% - 66%", 62.5),
    ("67% - 100%", 83.5),
    ("1% - 1%", 1.0),
])
def test_parse_risk_band(band, expected):
    assert features.parse_risk_band(band) == expected


def test_parse_risk_band_missing():
    assert np.isnan(features.parse_risk_band(np.nan))


def test_weighted_mean_ignores_missing_values():
    df = pd.DataFrame({
        "a": [10.0, 10.0, np.nan, np.nan],
        "b": [20.0, np.nan, 20.0, np.nan],
        "c": [30.0, np.nan, np.nan, np.nan],
    })
    out = features.weighted_mean(df, {"a": 0.5, "b": 0.3, "c": 0.2})

    assert out[0] == pytest.approx(0.5 * 10 + 0.3 * 20 + 0.2 * 30)
    assert out[1] == pytest.approx(10.0)   # only a present
    assert out[2] == pytest.approx(20.0)   # two inputs missing: falls back to b
    assert np.isnan(out[3])                # nothing to average


def test_add_engineered_features():
    df = pd.DataFrame({
        "riskassesment_302T": ["6% - 8%", np.nan],
        "avgdbddpdlast24m_3658932P": [10.0, np.nan],
        "avgdbddpdlast3m_4187120P": [0.0, 5.0],
    })
    out = features.add_engineered_features(df)

    assert out["cb_risk_mean"].tolist() == [7.0, 0.0]
    assert out["with_cb_risk"].tolist() == [1, 0]
    assert out["weighted_avg_dpd"].tolist() == pytest.approx([8.0, 5.0])
    assert "weighted_posfpd" not in out      # source columns absent
    assert "cb_risk_mean" not in df          # input not modified


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def test_join_is_by_case_id_not_by_position():
    base = pd.DataFrame({"case_id": [1, 2, 3]})
    other = pd.DataFrame({"case_id": [3, 1], "income": [300.0, 100.0]})
    out = data.join_per_case(base, other)

    assert out["income"].tolist()[0] == 100.0
    assert np.isnan(out["income"].tolist()[1])
    assert out["income"].tolist()[2] == 300.0


def test_join_rejects_duplicated_cases():
    base = pd.DataFrame({"case_id": [1, 2]})
    other = pd.DataFrame({"case_id": [1, 1], "income": [1.0, 2.0]})
    with pytest.raises(pd.errors.MergeError):
        data.join_per_case(base, other)


def test_aggregate_and_applicant_rows():
    person = pd.DataFrame({
        "case_id": [1, 1, 2],
        "num_group1": [0, 1, 0],
        "mainoccupationinc_384A": [10.0, 50.0, 20.0],
    })
    agg = data.aggregate_per_case(
        person, {"income_max": ("mainoccupationinc_384A", "max")})
    assert agg["income_max"].tolist() == [50.0, 20.0]
    assert data.applicant_rows(person)["case_id"].tolist() == [1, 2]


def test_load_table_concatenates_chunks(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    pd.DataFrame({"case_id": [1], "x": [1]}).to_csv(
        folder / "train_static_0_0.csv", index=False)
    pd.DataFrame({"case_id": [2], "x": [2]}).to_csv(
        folder / "train_static_0_1.csv", index=False)
    pd.DataFrame({"case_id": [1, 2]}).to_csv(
        folder / "train_base.csv", index=False)

    assert data.load_table(tmp_path, "static_0")["case_id"].tolist() == [1, 2]
    assert len(data.load_table(tmp_path, "base")) == 2
    with pytest.raises(FileNotFoundError):
        data.load_table(tmp_path, "person_1")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
@pytest.fixture
def applications():
    rng = np.random.default_rng(1)
    n = 5000
    return pd.DataFrame({
        "case_id": np.arange(n),
        "WEEK_NUM": np.sort(rng.integers(0, 50, n)),
        "target": (rng.random(n) < 0.1).astype(int),
        "income": rng.normal(size=n),
    })


def test_time_based_split_keeps_weeks_apart_and_ordered(applications):
    train, valid, test = splits.time_based_split(applications)

    assert len(train) + len(valid) + len(test) == len(applications)
    assert train["WEEK_NUM"].max() < valid["WEEK_NUM"].min()
    assert valid["WEEK_NUM"].max() < test["WEEK_NUM"].min()
    assert len(train) / len(applications) == pytest.approx(0.6, abs=0.05)


def test_random_split_is_stratified_and_reproducible(applications):
    train, valid, test = splits.random_split(applications)
    again = splits.random_split(applications)[0]

    assert train["case_id"].tolist() == again["case_id"].tolist()
    rate = applications["target"].mean()
    for part in (train, valid, test):
        assert part["target"].mean() == pytest.approx(rate, abs=0.01)


def test_feature_columns_exclude_identifiers_and_dates(applications):
    assert splits.feature_columns(applications) == ["income"]

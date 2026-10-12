"""Loading and joining the Home Credit tables.

The competition data is a set of CSV tables keyed by ``case_id`` (one loan
application):

* depth 0 (``base``, ``static_0_*``, ``static_cb_0``): one row per case.
* depth 1 (``person_1``, ``applprev_1_*``, ...): several rows per case,
  numbered by ``num_group1``. Row ``num_group1 == 0`` of ``person_1`` is the
  applicant.
* depth 2: several rows per (case, ``num_group1``), numbered by ``num_group2``.

Depth > 0 tables must be aggregated to one row per case before joining.

Every join here is made **on ``case_id``**, so the result does not depend on
the row order of the tables.
"""
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import pandas as pd

KEY = "case_id"


def load_table(
    data_dir: Union[str, Path],
    name: str,
    split: str = "train",
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Read one logical table, concatenating its chunk files.

    Large tables are shipped in pieces (``train_static_0_0.csv``,
    ``train_static_0_1.csv``, ...). ``load_table(dir, 'static_0')`` returns
    them as a single frame.

    Parameters
    ----------
    data_dir :
        The ``csv_files`` directory of the Kaggle download.
    name :
        Table name without the split prefix or chunk suffix, e.g.
        ``'base'``, ``'static_0'``, ``'static_cb_0'``, ``'person_1'``.
    split :
        ``'train'`` or ``'test'``.
    columns :
        Columns to read. Reading only what is needed keeps memory manageable;
        the static table alone is over 1 GB.
    """

    folder = Path(data_dir) / split
    exact = folder / f"{split}_{name}.csv"
    paths = [exact] if exact.exists() else sorted(
        folder.glob(f"{split}_{name}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No files for table '{name}' in {folder}")

    usecols = None if columns is None else list(columns)
    frames = [pd.read_csv(p, usecols=usecols, low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True)


def aggregate_per_case(
    table: pd.DataFrame,
    aggregations: Mapping[str, tuple],
) -> pd.DataFrame:
    """Collapse a depth > 0 table to one row per ``case_id``.

    Parameters
    ----------
    table :
        A table with several rows per case.
    aggregations :
        ``{new_column: (source_column, function)}``, as accepted by pandas
        named aggregation, e.g. ``{'income_max': ('mainoccupationinc_384A',
        'max')}``.
    """

    return table.groupby(KEY).agg(**aggregations).reset_index()


def applicant_rows(person: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``person_1`` describing the applicant (``num_group1 == 0``)."""

    return person[person["num_group1"] == 0].drop(columns="num_group1")


def join_per_case(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``other`` onto ``base`` by ``case_id``.

    ``other`` must have at most one row per case; otherwise the join would
    duplicate applications and pandas raises ``MergeError``.
    """

    return base.merge(other, on=KEY, how="left", validate="one_to_one")

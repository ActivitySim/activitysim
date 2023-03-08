# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package without
# needing to import them (pytest will automatically discover them).
# https://docs.pytest.org/en/7.2.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from activitysim.core.workflow import State
from activitysim.core.workflow.checkpoint import INITIAL_CHECKPOINT_NAME, ParquetStore


def _person_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Income": [45, 88, 56, 15, 71],
            "Name": ["Andre", "Bruce", "Carol", "David", "Eugene"],
            "Age": [14, 25, 55, 8, 21],
            "WorkMode": ["Car", "Bus", "Car", "Car", "Walk"],
        },
        index=pd.Index([441, 445, 552, 556, 934], name="person_id"),
    )
    df["WorkMode"] = df["WorkMode"].astype("category")
    return df


@pytest.fixture
def person_df() -> pd.DataFrame:
    """
    Sample persons dataframe with dummy data.
    """
    return _person_df()


def _los_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Speed": {"Car": 60, "Bus": 20, "Walk": 3},
            "Cost": {"Car": 3.25, "Bus": 1.75, "Walk": 0},
        }
    )


@pytest.fixture
def los_df() -> pd.DataFrame:
    """
    Sample LOS dataframe with dummy data.
    """
    return _los_df()


@pytest.fixture
def los_messy_df() -> pd.DataFrame:
    """
    Sample LOS dataframe with messy data.
    """
    return _los_messy_df()


def _los_messy_df() -> pd.DataFrame:
    los_df = _los_df()
    los_df["int_first"] = [123, "text", 456.7]
    los_df["text_first"] = ["klondike", 5, 12.34]
    los_df["float_first"] = [456.7, "text", 555]
    return los_df


@pytest.fixture(scope="session")
def sample_parquet_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Generate sample parquet store for testing.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        PyTest's own temporary path fixture, the sample parquet store
        will be created in a temporary directory here.

    Returns
    -------
    Path
        Location of zip archive
    """
    t = tmp_path_factory.mktemp("core-workflow")

    s = t.joinpath("sample-1")
    s.joinpath("configs").mkdir(parents=True, exist_ok=True)
    s.joinpath("data").mkdir(exist_ok=True)

    state = State.make_default(s)
    state.checkpoint.add(INITIAL_CHECKPOINT_NAME)

    # a table to store
    person_df = _person_df()
    state.add_table("persons", person_df)
    state.checkpoint.add("init_persons")

    # a second table
    state.add_table("level_of_service", _los_df())
    state.checkpoint.add("init_los")

    # modify table
    person_df["status"] = [11, 22, 33, 44, 55]
    state.add_table("persons", person_df)
    state.checkpoint.add("mod_persons")

    # modify table messy
    state.add_table("level_of_service", _los_messy_df())
    state.checkpoint.add("mod_los")

    return state.checkpoint.store.filename


@pytest.fixture(scope="session")
def sample_parquet_zip(sample_parquet_store: Path) -> Path:
    """
    Copy the sample parquet store into a read-only Zip archive.

    Parameters
    ----------
    sample_parquet_store : Path
        Location of original ParquetStore files.

    Returns
    -------
    Path
        Location of zip archive
    """
    ps = ParquetStore(sample_parquet_store, mode="r")
    return ps.make_zip_archive(
        output_filename=ps.filename.parent.joinpath("samplepipeline")
    )

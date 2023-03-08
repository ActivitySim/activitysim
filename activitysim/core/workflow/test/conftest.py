from __future__ import annotations

import tempfile

import pandas as pd
import pytest

from activitysim.core.workflow import State
from activitysim.core.workflow.checkpoint import INITIAL_CHECKPOINT_NAME, ParquetStore

tempdir = tempfile.TemporaryDirectory()


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
    return _los_df()


@pytest.fixture(scope="session")
def sample_parquet_store(tmp_path_factory):
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

    return state.checkpoint.store.filename


@pytest.fixture(scope="session")
def sample_parquet_zip(sample_parquet_store):
    ps = ParquetStore(sample_parquet_store, mode="r")
    return ps.make_zip_archive(
        output_filename=ps.filename.parent.joinpath("samplepipeline")
    )

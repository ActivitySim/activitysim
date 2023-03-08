from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from activitysim.core import exceptions
from activitysim.core.workflow.checkpoint import GenericCheckpointStore, ParquetStore


def _test_parquet_store(store: GenericCheckpointStore, person_df, los_df, los_messy_df):
    assert isinstance(store, GenericCheckpointStore)
    assert store.list_checkpoint_names() == [
        "init",
        "init_persons",
        "init_los",
        "mod_persons",
        "mod_los",
    ]
    with pytest.raises(exceptions.TableNameNotFound):
        store.get_dataframe("missing-tablename", "init_persons")
    with pytest.raises(exceptions.CheckpointNameNotFoundError):
        store.get_dataframe("persons", "bad-checkpoint-name")
    pd.testing.assert_frame_equal(
        store.get_dataframe("persons", "init_persons"), person_df
    )
    pd.testing.assert_frame_equal(store.get_dataframe("persons", "init_los"), person_df)
    with pytest.raises(AssertionError, match="DataFrame shape mismatch"):
        pd.testing.assert_frame_equal(
            store.get_dataframe("persons", "mod_persons"), person_df
        )
    pd.testing.assert_frame_equal(
        store.get_dataframe("persons", "mod_persons"),
        person_df.assign(status=[11, 22, 33, 44, 55]),
    )
    # call for last checkpoint explicitly
    pd.testing.assert_frame_equal(
        store.get_dataframe("persons", "_"),
        person_df.assign(status=[11, 22, 33, 44, 55]),
    )
    # call for last checkpoint implicitly
    pd.testing.assert_frame_equal(
        store.get_dataframe("persons"),
        person_df.assign(status=[11, 22, 33, 44, 55]),
    )

    pd.testing.assert_frame_equal(
        store.get_dataframe("level_of_service", "init_los"),
        los_df,
    )
    pd.testing.assert_frame_equal(
        store.get_dataframe("level_of_service", "mod_persons"),
        los_df,
    )
    # messy data has mixed dtypes, falls back to pickle instead of parquet
    pd.testing.assert_frame_equal(
        store.get_dataframe("level_of_service"),
        los_messy_df,
    )


def test_parquet_store(sample_parquet_store: Path, person_df, los_df, los_messy_df):
    ps = ParquetStore(sample_parquet_store, mode="r")
    _test_parquet_store(ps, person_df, los_df, los_messy_df)


def test_parquet_store_zip(sample_parquet_zip: Path, person_df, los_df, los_messy_df):
    ps = ParquetStore(sample_parquet_zip, mode="r")
    _test_parquet_store(ps, person_df, los_df, los_messy_df)

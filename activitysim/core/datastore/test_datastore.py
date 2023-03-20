from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from sharrow.dataset import construct

from activitysim.core.datastore import new_store
from activitysim.core.datastore.parquet import ParquetStore
from activitysim.core.exceptions import ReadOnlyError


@pytest.fixture
def person_dataset() -> xr.Dataset:
    """
    Sample persons dataset with dummy data.
    """
    df = pd.DataFrame(
        {
            "Income": [45, 88, 56, 15, 71],
            "Name": ["Andre", "Bruce", "Carol", "David", "Eugene"],
            "Age": [14, 25, 55, 8, 21],
            "WorkMode": ["Car", "Bus", "Car", "Car", "Walk"],
            "household_id": [11, 11, 22, 22, 33],
        },
        index=pd.Index([441, 445, 552, 556, 934], name="person_id"),
    )
    df["WorkMode"] = df["WorkMode"].astype("category")
    return construct(df)


@pytest.fixture
def household_dataset() -> xr.Dataset:
    """
    Sample household dataset with dummy data.
    """
    df = pd.DataFrame(
        {
            "n_cars": [1, 2, 1],
        },
        index=pd.Index([11, 22, 33], name="household_id"),
    )
    return construct(df)


@pytest.fixture
def tours_dataset() -> xr.Dataset:
    """
    Sample tours dataset with dummy data.
    """
    df = pd.DataFrame(
        {
            "TourMode": ["Car", "Bus", "Car", "Car", "Walk"],
            "person_id": [441, 445, 552, 556, 934],
        },
        index=pd.Index([4411, 4451, 5521, 5561, 9341], name="tour_id"),
    )
    df["TourMode"] = df["TourMode"].astype("category")
    return construct(df)


@pytest.mark.parametrize("storage_format", ["parquet", "zarr", "hdf"])
def test_datasstore_checkpointing(tmp_path: Path, person_dataset, storage_format):

    tm = new_store(tmp_path, storage_format=storage_format)
    tm["persons"] = person_dataset
    tm.make_checkpoint("init_persons")

    person_dataset["DoubleAge"] = person_dataset["Age"] * 2
    tm.add_data("persons", person_dataset["DoubleAge"])
    tm.make_checkpoint("annot_persons")

    tm2 = new_store(tmp_path, storage_format=storage_format)
    tm2.restore_checkpoint("annot_persons")
    xr.testing.assert_equal(tm2.get_dataset("persons"), person_dataset)

    tm2.restore_checkpoint("init_persons")
    assert "DoubleAge" not in tm2.get_dataset("persons")

    tm_ro = new_store(tmp_path, mode="r", storage_format=storage_format)
    with pytest.raises(ReadOnlyError):
        tm_ro.make_checkpoint("will-fail")

    if storage_format != "hdf":
        for fmt in ["parquet", "zarr"]:
            tm_ro_x = new_store(tmp_path, mode="r", storage_format=fmt)
            tm_ro_x.restore_checkpoint("annot_persons")
            xr.testing.assert_equal(tm_ro_x.get_dataset("persons"), person_dataset)


def test_datasstore_relationships(
    tmp_path: Path, person_dataset, household_dataset, tours_dataset
):
    pth = tmp_path.joinpath("relations")

    if pth.exists():
        shutil.rmtree(pth)

    pth.mkdir(parents=True, exist_ok=True)
    tm = ParquetStore(directory=pth)

    tm["persons"] = person_dataset
    tm.make_checkpoint("init_persons")

    tm["households"] = household_dataset
    tm.add_relationship("persons.household_id @ households.household_id")
    tm.make_checkpoint("init_households")

    tm["tours"] = tours_dataset
    tm.add_relationship("tours.person_id @ persons.person_id")
    tm.make_checkpoint("init_tours")

    tm.digitize_relationships()
    assert tm.relationships_are_digitized

    tm.make_checkpoint("digitized")

    tm2 = ParquetStore(directory=pth, mode="r")
    tm2.read_metadata("*")
    tm2.restore_checkpoint("init_households")

    assert sorted(tm2.get_dataset("persons")) == [
        "Age",
        "Income",
        "Name",
        "WorkMode",
        "household_id",
    ]

    assert sorted(tm2.get_dataset("households")) == [
        "n_cars",
    ]

    tm2.restore_checkpoint("digitized")
    assert sorted(tm2.get_dataset("persons")) == [
        "Age",
        "Income",
        "Name",
        "WorkMode",
        "digitizedOffsethousehold_id_households_household_id",
        "household_id",
    ]

    double_age = tm2.get_dataset("persons")["Age"] * 2
    with pytest.raises(ReadOnlyError):
        tm2.add_data("persons", double_age.rename("doubleAge"))

    with pytest.raises(ReadOnlyError):
        tm2.make_checkpoint("age-x2")

    tm.add_data("persons", double_age.rename("doubleAge"))
    assert sorted(tm.get_dataset("persons")) == [
        "Age",
        "Income",
        "Name",
        "WorkMode",
        "digitizedOffsethousehold_id_households_household_id",
        "doubleAge",
        "household_id",
    ]

    tm.make_checkpoint("age-x2")
    tm2.read_metadata()
    tm2.restore_checkpoint("age-x2")

    person_restored = tm2.get_dataframe("persons")
    print(person_restored.WorkMode.dtype)
    assert person_restored.WorkMode.dtype == "category"

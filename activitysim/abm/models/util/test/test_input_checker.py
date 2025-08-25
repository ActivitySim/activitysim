# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import os.path

import pandas as pd
import pandas.testing as pdt
import pandera as pa
import pytest
import yaml

from activitysim.abm.models.input_checker import TABLE_STORE, validate_with_pandera


@pytest.fixture(scope="class")
def v_errors():
    v_errors = {}
    v_errors["households"] = []
    return v_errors


@pytest.fixture(scope="class")
def v_warnings():
    v_warnings = {}
    v_warnings["households"] = []
    return v_warnings


@pytest.fixture(scope="module")
def validation_settings():
    return {"method": "pandera", "class": "Household"}


@pytest.fixture(scope="module")
def households():
    return pd.DataFrame(
        data={
            "household_id": [1, 2, 3, 4],
            "home_zone_id": [0, 1, 2, 3],
            "income": [10000, 20000, 30000, 40000],
            "hhsize": [2, 2, 4, 5],
        }
    )


TABLE_STORE["households"] = pd.DataFrame(
    data={
        "household_id": [1, 2, 3, 4],
        "home_zone_id": [0, 1, 2, 3],
        "income": [10000, 20000, 30000, 40000],
        "hhsize": [2, 2, 4, 5],
    }
)


def test_passing_dataframe(households, v_errors, v_warnings, validation_settings):
    TABLE_STORE["households"] = households

    class input_checker:
        class Household(pa.DataFrameModel):
            household_id: int = pa.Field(unique=True, gt=0)
            home_zone_id: int = pa.Field(ge=0)
            hhsize: int = pa.Field(gt=0)
            income: int = pa.Field(ge=0, raise_warning=True)

        @pa.dataframe_check(name="Example setup of a passing error check.")
        def dummy_example(cls, households: pd.DataFrame):
            return (households.household_id > 0).all()

    returned_errors, returned_warnings = validate_with_pandera(
        input_checker, "households", validation_settings, v_errors, v_warnings
    )

    assert (
        len(returned_errors["households"]) == 0
    ), f"Expect no household errors, but got {returned_errors}"
    assert (
        len(returned_warnings["households"]) == 0
    ), f"Expect no household warnings, but got {returned_warnings}"


def test_error_dataframe(households, v_errors, v_warnings, validation_settings):
    TABLE_STORE["households"] = households

    class input_checker:
        class Household(pa.DataFrameModel):
            household_id: int = pa.Field(unique=True, gt=0)
            home_zone_id: int = pa.Field(ge=0)
            hhsize: int = pa.Field(gt=0)
            income: int = pa.Field(ge=0, raise_warning=True)
            bug1: int  # error here

    returned_errors, returned_warnings = validate_with_pandera(
        input_checker, "households", validation_settings, v_errors, v_warnings
    )

    assert (
        len(returned_errors["households"]) == 1
    ), f"Expected household error, but got {len(returned_errors['households'])}"
    assert (
        len(returned_warnings["households"]) == 0
    ), f"Expect no household warnings, but got {returned_warnings['households']}"


def test_warning_dataframe(households, v_errors, v_warnings, validation_settings):
    TABLE_STORE["households"] = households

    class input_checker:
        class Household(pa.DataFrameModel):
            household_id: int = pa.Field(unique=True, gt=0)
            home_zone_id: int = pa.Field(ge=0)
            hhsize: int = pa.Field(gt=0)
            income: int = pa.Field(ge=100000, raise_warning=True)  # warning here

    returned_errors, returned_warnings = validate_with_pandera(
        input_checker, "households", validation_settings, v_errors, v_warnings
    )

    assert (
        len(returned_errors["households"]) == 0
    ), f"Expect no household errors, but got {returned_errors['households']}"
    assert (
        len(returned_warnings["households"]) > 0
    ), f"Expected warnings, but got {len(returned_warnings['households'])}"


def test_custom_check_failure_dataframe(
    households, v_errors, v_warnings, validation_settings
):
    TABLE_STORE["households"] = households

    class input_checker:
        class Household(pa.DataFrameModel):
            household_id: int = pa.Field(unique=True, gt=0)
            home_zone_id: int = pa.Field(ge=0)
            hhsize: int = pa.Field(gt=0)
            income: int = pa.Field(ge=0, raise_warning=True)

            @pa.dataframe_check(name="Example setup of a failed error check.")
            def dummy_example(cls, households: pd.DataFrame):
                return False

    returned_errors, returned_warnings = validate_with_pandera(
        input_checker, "households", validation_settings, v_errors, v_warnings
    )

    assert (
        len(returned_errors["households"]) > 0
    ), f"Expected household errors, but got {returned_errors['households']}"
    assert (
        len(returned_warnings["households"]) == 0
    ), f"Expect no household warnings, but got {returned_warnings['households']}"

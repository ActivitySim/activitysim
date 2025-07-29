from __future__ import annotations

import os
import shutil
from importlib.resources import files
from pathlib import Path

import openmatrix
import pytest

import activitysim.abm  # noqa: F401
from activitysim.core import workflow


def example_path(dirname):
    resource = files("activitysim.examples.placeholder_sandag").joinpath(dirname)
    return str(resource)


def mtc_example_path(dirname):
    resource = files("activitysim.examples.prototype_mtc").joinpath(dirname)
    return str(resource)


def psrc_example_path(dirname):
    resource = files("activitysim.examples.placeholder_psrc").joinpath(dirname)
    return str(resource)


@pytest.fixture(scope="session")
def example_data_dir(tmp_path_factory) -> Path:
    """Fixture to provide the path to the example data directory."""
    td = tmp_path_factory.mktemp("skim-conflict-data")
    shutil.copytree(example_path("data_2"), td.joinpath("data_2"))
    shutil.copy(
        example_path(os.path.join("data_3", "maz_to_maz_bike.csv")),
        td.joinpath("data_2"),
    )

    # add extra skims to OMX to create a conflict

    with openmatrix.open_file(td.joinpath("data_2").joinpath("skims1.omx"), "a") as omx:
        for t in ["EA", "AM", "MD", "PM", "EV"]:
            # Create a new matrix for each time period
            omx.createMatrix(f"DISTBIKE__{t}", obj=omx["DISTBIKE"][:])

    return td.joinpath("data_2")


def test_skim_name_conflicts(example_data_dir, tmp_path_factory):
    # when sharrow is required, the run should fail due to conflicting skim names
    state = workflow.State.make_default(
        data_dir=example_data_dir,
        configs_dir=(
            example_path("configs_2_zone"),
            psrc_example_path("configs"),
        ),
        output_dir=tmp_path_factory.mktemp("out-fail"),
        settings={
            "households_sample_size": 20,
            "sharrow": "require",
            "disable_zarr": True,
        },
    )
    with pytest.raises(ValueError):
        state.run(
            [
                "initialize_landuse",
                "initialize_households",
            ]
        )


def test_skim_name_conflicts_no_sharrow(example_data_dir, tmp_path_factory):
    # when sharrow is disabled, the run should warn about conflicting skim names but not fail
    state = workflow.State.make_default(
        data_dir=example_data_dir,
        configs_dir=(
            example_path("configs_2_zone"),
            psrc_example_path("configs"),
        ),
        output_dir=tmp_path_factory.mktemp("out-pass"),
        settings={
            "households_sample_size": 20,
            "sharrow": False,
            "disable_zarr": True,
        },
    )
    # Run the beginning workflow with the modified settings, should only warn about the conflict
    with pytest.warns(
        UserWarning,
        match="some skims have both time-dependent and time-agnostic versions",
    ):
        state.run(
            [
                "initialize_landuse",
                "initialize_households",
            ]
        )


@pytest.mark.parametrize("solution", ["^DISTBIKE$", "^DISTBIKE__.+"])
def test_skim_name_conflicts_ok(example_data_dir, tmp_path_factory, solution):
    # when sharrow is required, and omx_ignore_patterns is set correctly,
    # the run should work without raising an error
    state = workflow.State.make_default(
        data_dir=example_data_dir,
        configs_dir=(
            example_path("configs_2_zone"),
            psrc_example_path("configs"),
        ),
        output_dir=tmp_path_factory.mktemp("out-solved"),
        settings={
            "households_sample_size": 20,
            "sharrow": "require",
            "disable_zarr": True,
            "omx_ignore_patterns": [solution],
        },
    )
    state.run(
        [
            "initialize_landuse",
            "initialize_households",
        ]
    )

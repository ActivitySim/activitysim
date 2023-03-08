from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import workflow
from activitysim.core.test import run_if_exists


def _test_sandag_xborder(sharrow=False, mp=True):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_sandag_xborder", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        if mp:
            regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        else:
            regress_trips_df = pd.read_csv(
                test_path("regress/final_trips_1_process.csv")
            )
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))
        # column ordering not important
        assert sorted(regress_trips_df.columns) == sorted(final_trips_df.columns)
        pdt.assert_frame_equal(
            final_trips_df[sorted(regress_trips_df.columns)],
            regress_trips_df[sorted(regress_trips_df.columns)],
        )

    file_path = os.path.join(os.path.dirname(__file__), "../simulation.py")

    run_args = [file_path]
    if sharrow:
        run_args += ["-c", test_path("configs_sharrow")]
    if not mp:
        run_args += ["-c", test_path("configs_single_process")]
    run_args += [
        "-c",
        test_path("configs"),
        "-c",
        example_path("configs"),
        "-d",
        example_path("data"),
        "-o",
        test_path("output"),
    ]

    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a"] + run_args, check=True)
    else:
        subprocess.run([sys.executable] + run_args, check=True)

    regress()


def test_sandag_xborder():
    _test_sandag_xborder(mp=False)


def test_sandag_xborder_mp():
    _test_sandag_xborder(mp=True)


def test_sandag_xborder_sharrow():
    _test_sandag_xborder(sharrow=True, mp=False)


EXPECTED_MODELS = [
    "initialize_landuse",
    "initialize_households",
    "initialize_tours",
    "initialize_los",
    "initialize_tvpb",
    "tour_scheduling_probabilistic",
    "tour_od_choice",
    "reassign_tour_purpose_by_poe",
    "tour_mode_choice_simulate",
    "stop_frequency",
    "trip_purpose",
    "trip_scheduling",
    "trip_destination",
    "trip_mode_choice",
    "write_trip_matrices",
    "write_tables",
]


@run_if_exists("prototype_sandag_xborder_reference_pipeline.zip")
def test_sandag_xborder_progressive():
    import activitysim.abm  # register components # noqa: F401

    state = workflow.create_example("prototype_sandag_xborder", temp=True)
    state.settings.multiprocess = False
    state.settings.num_processes = 1
    state.settings.households_sample_size = 10
    state.settings.chunk_size = 0
    state.settings.recode_pipeline_columns = False
    state.import_extensions("extensions")

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.sharrow == False

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent.joinpath(
                    "prototype_sandag_xborder_reference_pipeline.zip"
                ),
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> prototype_sandag_xborder {step_name}: ERROR")
            raise
        else:
            print(f"> prototype_sandag_xborder {step_name}: ok")


if __name__ == "__main__":
    test_sandag_xborder()

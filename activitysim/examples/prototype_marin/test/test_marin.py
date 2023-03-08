from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import test, workflow


def example_path(dirname):
    resource = os.path.join("examples", "prototype_marin", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def _test_path(dirname):
    return os.path.join(os.path.dirname(__file__), dirname)


def test_marin():
    def regress():
        regress_trips_df = pd.read_csv(_test_path("regress/final_tours.csv"))
        final_trips_df = pd.read_csv(_test_path("output/final_tours.csv"))

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        test.assert_frame_substantively_equal(
            final_trips_df, regress_trips_df, check_dtype=False
        )

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    subprocess.run(
        [
            "coverage",
            "run",
            "-a",
            file_path,
            "-c",
            _test_path("configs"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            _test_path("output"),
        ],
        check=True,
    )

    regress()


EXPECTED_MODELS = [
    "initialize_landuse",
    "initialize_households",
    "initialize_tours",
    "initialize_los",
    "initialize_tvpb",
    "tour_mode_choice_simulate",
    "write_data_dictionary",
    "track_skim_usage",
    "write_tables",
]


@test.run_if_exists("reference_pipeline.zip")
def test_marin_progressive():
    import activitysim.abm  # register components

    state = workflow.State.make_default(
        configs_dir=(
            _test_path("configs"),
            example_path("configs"),
        ),
        data_dir=(example_path("data"),),
        output_dir=_test_path("output"),
    )

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    assert state.settings.sharrow == False

    state.settings.trace_hh_id = 8268
    trace_validation_directory = Path(__file__).parent / "reference_trace.tar.gz"
    if trace_validation_directory.exists():
        state.tracing.validation_directory = trace_validation_directory

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent / "reference_pipeline.zip",
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> MARIN {step_name}: ERROR")
            raise
        else:
            print(f"> MARIN {step_name}: ok")


if __name__ == "__main__":
    test_marin()
    test_marin_progressive()

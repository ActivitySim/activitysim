from __future__ import annotations

import atexit
import importlib.resources

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from contextlib import ExitStack

import pandas as pd

from activitysim.core.test import assert_frame_substantively_equal


def _test_psrc(sharrow=False):
    def example_path(dirname):
        file_manager = ExitStack()
        atexit.register(file_manager.close)
        ref = importlib.resources.files("activitysim").joinpath(
            "examples", "placeholder_psrc", dirname
        )
        return file_manager.enter_context(importlib.resources.as_file(ref))

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        assert_frame_substantively_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if sharrow:
        run_args = ["-c", test_path("configs_sharrow")]
    else:
        run_args = []

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
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run([sys.executable, file_path] + run_args, check=True)

    regress()


def test_psrc():
    _test_psrc(sharrow=False)


def test_psrc_sharrow():
    _test_psrc(sharrow=True)


if __name__ == "__main__":
    test_psrc()

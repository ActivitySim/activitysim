# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def run_test_mtc(multiprocess=False, chunkless=False):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        pdt.assert_frame_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if multiprocess:
        run_args = [
            "-c",
            test_path("configs_mp"),
            "-c",
            example_path("configs_mp"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    elif chunkless:
        run_args = [
            "-c",
            test_path("configs_chunkless"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    else:
        run_args = [
            "-c",
            test_path("configs"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]

    subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)

    regress()


def test_mtc():
    run_test_mtc(multiprocess=False)


def test_mtc_chunkless():
    run_test_mtc(multiprocess=False, chunkless=True)


def test_mtc_mp():
    run_test_mtc(multiprocess=True)


if __name__ == "__main__":

    run_test_mtc(multiprocess=False)
    run_test_mtc(multiprocess=True)
    run_test_mtc(multiprocess=False, chunkless=True)

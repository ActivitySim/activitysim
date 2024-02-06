from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess

import pandas as pd
import pkg_resources

from activitysim.core.test._tools import assert_frame_substantively_equal


def run_test_semcog(multiprocess=False):
    def example_path(dirname):
        resource = os.path.join("examples", "production_semcog", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(
            test_path("regress/final_trips.csv"), dtype={"depart": int}
        )
        final_trips_df = pd.read_csv(
            test_path("output/final_trips.csv"), dtype={"depart": int}
        )
        assert_frame_substantively_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "../simulation.py")

    if multiprocess:
        subprocess.run(
            [
                "coverage",
                "run",
                "-a",
                file_path,
                "-c",
                test_path("configs_mp"),
                "-c",
                example_path("configs_mp"),
                "-c",
                example_path("configs"),
                "-d",
                example_path("data"),
                "--data_model",
                example_path("data_model"),
                "-o",
                test_path("output"),
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "coverage",
                "run",
                "-a",
                file_path,
                "-c",
                test_path("configs"),
                "-c",
                example_path("configs"),
                "-d",
                example_path("data"),
                "--data_model",
                example_path("data_model"),
                "-o",
                test_path("output"),
            ],
            check=True,
        )

    regress()


def test_semcog():
    run_test_semcog(multiprocess=False)


def test_semcog_mp():
    run_test_semcog(multiprocess=True)


if __name__ == "__main__":
    run_test_semcog(multiprocess=False)
    run_test_semcog(multiprocess=True)

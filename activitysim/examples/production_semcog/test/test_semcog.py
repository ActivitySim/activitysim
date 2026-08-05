from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import importlib.resources
import os
import subprocess

import pandas as pd

from activitysim.core.test._tools import assert_frame_substantively_equal


def run_test_semcog(multiprocess=False, use_explicit_error_terms=False):
    def example_path(dirname):
        resource = os.path.join("examples", "production_semcog", dirname)
        return str(importlib.resources.files("activitysim").joinpath(resource))

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress(use_explicit_error_terms=False):
        regress_trips_df = pd.read_csv(
            test_path(
                f"regress/final{'_eet' if use_explicit_error_terms else ''}_trips.csv"
            ),
            dtype={"depart": int},
        )
        final_trips_df = pd.read_csv(
            test_path("output/final_trips.csv"), dtype={"depart": int}
        )
        assert_frame_substantively_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "../simulation.py")

    test_config_files = []
    if use_explicit_error_terms:
        test_config_files = [
            "-c",
            test_path("configs_eet"),
        ]
    if multiprocess:
        subprocess.run(
            [
                "coverage",
                "run",
                "-a",
                file_path,
                *test_config_files,
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
                *test_config_files,
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

    regress(use_explicit_error_terms=use_explicit_error_terms)


def test_semcog():
    run_test_semcog(multiprocess=False)


def test_semcog_mp():
    run_test_semcog(multiprocess=True)


def test_semcog_eet():
    run_test_semcog(multiprocess=False, use_explicit_error_terms=True)


def test_semcog_mp_eet():
    run_test_semcog(multiprocess=True, use_explicit_error_terms=True)


if __name__ == "__main__":
    run_test_semcog(multiprocess=False)
    run_test_semcog(multiprocess=True)
    run_test_semcog(multiprocess=False, use_explicit_error_terms=True)
    run_test_semcog(multiprocess=True, use_explicit_error_terms=True)

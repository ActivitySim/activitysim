# ActivitySim
# See full license in LICENSE.txt.
import os
import shutil
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt
import pkg_resources
import pytest

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def example_path(dirname):
    resource = os.path.join("examples", "placeholder_sandag", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def mtc_example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def psrc_example_path(dirname):
    resource = os.path.join("examples", "placeholder_psrc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def build_data():
    shutil.copy(
        example_path(os.path.join("data_3", "maz_to_maz_bike.csv")),
        example_path(os.path.join("data_2", "maz_to_maz_bike.csv")),
    )


@pytest.fixture(scope="module")
def data():
    build_data()


def run_test(zone, multiprocess=False, sharrow=False):
    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress(zone):

        # ## regress tours
        if sharrow and os.path.isfile(
            test_path(f"regress/final_{zone}_zone_tours_sh.csv")
        ):
            regress_tours_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_tours_sh.csv")
            )
        else:
            regress_tours_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_tours.csv")
            )
        tours_df = pd.read_csv(test_path(f"output_{zone}/final_{zone}_zone_tours.csv"))
        tours_df.to_csv(
            test_path(f"regress/final_{zone}_zone_tours_last_run.csv"), index=False
        )
        print("regress tours")
        pdt.assert_frame_equal(
            tours_df, regress_tours_df, rtol=1e-03, check_dtype=False
        )

        # ## regress trips
        if sharrow and os.path.isfile(
            test_path(f"regress/final_{zone}_zone_trips_sh.csv")
        ):
            regress_trips_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_trips_sh.csv")
            )
        else:
            regress_trips_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_trips.csv")
            )
        trips_df = pd.read_csv(test_path(f"output_{zone}/final_{zone}_zone_trips.csv"))
        trips_df.to_csv(
            test_path(f"regress/final_{zone}_zone_trips_last_run.csv"), index=False
        )
        print("regress trips")
        pdt.assert_frame_equal(
            trips_df, regress_trips_df, rtol=1e-03, check_dtype=False
        )

        if zone == "2":
            # also test accessibility for the 2-zone system
            regress_accessibility_df = pd.read_csv(
                test_path(
                    f"regress/final_{zone}_zone_proto_disaggregate_accessibility.csv"
                )
            )
            final_accessibility_df = pd.read_csv(
                test_path(
                    f"output_{zone}/final_{zone}_zone_proto_disaggregate_accessibility.csv"
                )
            )
            final_accessibility_df = final_accessibility_df[
                [
                    c
                    for c in final_accessibility_df.columns
                    if not c.startswith("_original_")
                ]
            ]
            pdt.assert_frame_equal(final_accessibility_df, regress_accessibility_df)

    # run test
    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if zone == "2":
        base_configs = psrc_example_path("configs")
    else:
        base_configs = mtc_example_path("configs")

    run_args = [
        "-c",
        test_path(f"configs_{zone}_zone"),
        "-c",
        example_path(f"configs_{zone}_zone"),
        "-c",
        base_configs,
        "-d",
        example_path(f"data_{zone}"),
        "-o",
        test_path(f"output_{zone}"),
    ]

    if multiprocess:
        run_args = run_args + ["-s", "settings_mp.yaml"]

    if sharrow:
        run_args = ["-c", test_path(f"configs_{zone}_sharrow")] + run_args

    try:
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    except FileNotFoundError:
        subprocess.run([sys.executable, file_path] + run_args, check=True)
        from tempfile import TemporaryFile
        from time import sleep

        with TemporaryFile() as outputstream:
            env = os.environ.copy()
            pythonpath = env.pop("PYTHONPATH", None)
            process = subprocess.Popen(
                args=[sys.executable, file_path] + run_args,
                shell=True,
                stdout=outputstream,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env=env,
            )
            while process.poll() is None:
                where = outputstream.tell()
                lines = outputstream.read()
                if not lines:
                    # Adjust the sleep interval to your needs
                    sleep(0.25)
                    # make sure pointing to the last place we read
                    outputstream.seek(where)
                else:
                    # Windows adds an extra carriage return and then chokes on
                    # it when displaying (or, as it were, not displaying) the
                    # output.  So we give Windows a little helping hand.
                    print(lines.decode().replace("\r\n", "\n"), end="")

    regress(zone)


def test_1_zone(data):
    run_test(zone="1", multiprocess=False)


def test_1_zone_mp(data):
    run_test(zone="1", multiprocess=True)


def test_1_zone_sharrow(data):
    # Run both single and MP in one test function
    # guarantees that compile happens in single
    run_test(zone="1", multiprocess=False, sharrow=True)
    run_test(zone="1", multiprocess=True, sharrow=True)


def test_2_zone(data):
    run_test(zone="2", multiprocess=False)


def test_2_zone_mp(data):
    run_test(zone="2", multiprocess=True)


def test_2_zone_sharrow(data):
    # Run both single and MP in one test function
    # guarantees that compile happens in single
    run_test(zone="2", multiprocess=False, sharrow=True)
    run_test(zone="2", multiprocess=True, sharrow=True)


def test_3_zone(data):
    run_test(zone="3", multiprocess=False)


def test_3_zone_mp(data):
    run_test(zone="3", multiprocess=True)


def test_3_zone_sharrow(data):
    # Run both single and MP in one test function
    # guarantees that compile happens in single
    run_test(zone="3", multiprocess=False, sharrow=True)
    run_test(zone="3", multiprocess=True, sharrow=True)


if __name__ == "__main__":

    # call each test explicitly so we get a pass/fail for each
    build_data()
    run_test(zone="1", multiprocess=False)
    run_test(zone="1", multiprocess=True)
    run_test(zone="1", multiprocess=False, sharrow=True)
    run_test(zone="1", multiprocess=True, sharrow=True)

    run_test(zone="2", multiprocess=False)
    run_test(zone="2", multiprocess=True)
    run_test(zone="2", multiprocess=False, sharrow=True)
    run_test(zone="2", multiprocess=True, sharrow=True)

    run_test(zone="3", multiprocess=False)
    run_test(zone="3", multiprocess=True)
    run_test(zone="3", multiprocess=False, sharrow=True)
    run_test(zone="3", multiprocess=True, sharrow=True)

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def _test_prototype_mtc_extended(
    multiprocess=False, sharrow=False, shadow_pricing=True
):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc_extended", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def example_mtc_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_suffix = ""
        if shadow_pricing:
            regress_suffix += "-shadowpriced"
        if sharrow:
            regress_suffix += "-sharrow"

        output_dir = "output"
        regress_trips_df = pd.read_csv(
            test_path(f"regress/final_trips{regress_suffix}.csv")
        )
        final_trips_df = pd.read_csv(test_path(f"{output_dir}/final_trips.csv"))

        regress_vehicles_df = pd.read_csv(test_path("regress/final_vehicles.csv"))
        final_vehicles_df = pd.read_csv(test_path(f"{output_dir}/final_vehicles.csv"))

        regress_accessibility_df = pd.read_csv(
            test_path(
                f"regress/final_proto_disaggregate_accessibility{regress_suffix}.csv"
            )
        )
        final_accessibiliy_df = pd.read_csv(
            test_path(f"{output_dir}/final_proto_disaggregate_accessibility.csv")
        )
        # new transforms may add columns to final_accessibiliy_df, but that is
        # not a test breakage if the existing columns still match.
        final_accessibiliy_df = final_accessibiliy_df.drop(
            columns=[
                i
                for i in final_accessibiliy_df.columns
                if i not in regress_accessibility_df.columns
            ]
        )
        pdt.assert_frame_equal(
            final_accessibiliy_df, regress_accessibility_df, rtol=1.0e-4
        )

        pdt.assert_frame_equal(final_trips_df, regress_trips_df, rtol=1.0e-4)
        pdt.assert_frame_equal(final_vehicles_df, regress_vehicles_df, rtol=1.0e-4)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")
    shadowprice_configs = (
        [] if shadow_pricing else ["-c", test_path("no-shadow-pricing")]
    )
    if sharrow:
        sh_configs = ["-c", example_path("configs_sharrow")]
    else:
        sh_configs = []
    if multiprocess:
        mp_configs = [
            "-c",
            test_path("configs_mp"),
            "-c",
            example_path("configs_mp"),
        ]
    elif sharrow:
        mp_configs = [
            "-c",
            test_path("configs"),
        ]
    else:
        mp_configs = [
            "-c",
            test_path("configs"),
        ]
    run_args = (
        shadowprice_configs
        + sh_configs
        + mp_configs
        + [
            "-c",
            example_path("configs"),
            "-c",
            example_mtc_path("configs"),
            "-d",
            example_mtc_path("data"),
            "-o",
            test_path("output"),
        ]
    )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "activitysim", "run"] + run_args, check=True
        )

    regress()


def test_prototype_mtc_extended():
    _test_prototype_mtc_extended(
        multiprocess=False, sharrow=False, shadow_pricing=False
    )


def test_prototype_mtc_extended_sharrow():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=True, shadow_pricing=False)


def test_prototype_mtc_extended_mp():
    _test_prototype_mtc_extended(multiprocess=True, sharrow=False, shadow_pricing=False)


def test_prototype_mtc_extended_shadow_pricing():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=False, shadow_pricing=True)


def test_prototype_mtc_extended_sharrow_shadow_pricing():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=True, shadow_pricing=True)


def test_prototype_mtc_extended_mp_shadow_pricing():
    _test_prototype_mtc_extended(multiprocess=True, sharrow=False, shadow_pricing=True)


if __name__ == "__main__":

    test_prototype_mtc_extended()
    test_prototype_mtc_extended_sharrow()
    test_prototype_mtc_extended_mp()

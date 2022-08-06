# ActivitySim
# See full license in LICENSE.txt.

import os

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from .. import inject, los


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def add_canonical_dirs(configs_dir_name):

    configs_dir = os.path.join(os.path.dirname(__file__), f"los/{configs_dir_name}")
    inject.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), f"los/data")
    inject.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), f"output")
    inject.add_injectable("output_dir", output_dir)


def test_legacy_configs():

    add_canonical_dirs("configs_legacy_settings")

    with pytest.warns(FutureWarning):
        network_los = los.Network_LOS()

    assert network_los.setting("zone_system") == los.ONE_ZONE

    assert "z1_taz_skims.omx" in network_los.omx_file_names("taz")


def test_one_zone():

    add_canonical_dirs("configs_1z")

    network_los = los.Network_LOS()

    assert network_los.setting("zone_system") == los.ONE_ZONE

    assert "z1_taz_skims.omx" in network_los.omx_file_names("taz")

    network_los.load_data()

    # OMAZ, DMAZ, DIST, DISTBIKE
    # 23000,21000,1.89,1.89
    # 23000,22000,0.89,0.89
    # 23000,23000,0.19,0.19

    od_df = pd.DataFrame({"orig": [5, 23, 23, 23], "dest": [7, 20, 21, 22]})

    skim_dict = network_los.get_default_skim_dict()

    # skims should be the same as maz_to_maz distances in test data where 1 MAZ per TAZ
    # OMAZ, DMAZ, DIST, DISTBIKE
    # 1000, 2000, 0.24, 0.24
    # 23000,20000,2.55,2.55
    # 23000,21000,1.9,1.9
    # 23000,22000,0.62,0.62
    skims = skim_dict.wrap("orig", "dest")
    skims.set_df(od_df)
    pdt.assert_series_equal(
        skims["DIST"], pd.Series([0.4, 2.55, 1.9, 0.62]).astype(np.float32)
    )

    # OMAZ, DMAZ, DIST, DISTBIKE
    # 2000, 1000, 0.37, 0.37
    # 20000,23000,2.45,2.45
    # 21000,23000,1.89,1.89
    # 22000,23000,0.89,0.89

    skims = skim_dict.wrap("dest", "orig")
    skims.set_df(od_df)
    pdt.assert_series_equal(
        skims["DIST"], pd.Series([0.46, 2.45, 1.89, 0.89]).astype(np.float32)
    )


def test_two_zone():

    add_canonical_dirs("configs_2z")

    network_los = los.Network_LOS()

    assert network_los.setting("zone_system") == los.TWO_ZONE

    assert "z2_taz_skims.omx" in network_los.omx_file_names("taz")

    assert network_los.blend_distance_skim_name == "DIST"

    network_los.load_data()

    skim_dict = network_los.get_default_skim_dict()

    # skims should be the same as maz_to_maz distances when no blending
    od_df = pd.DataFrame(
        {
            "orig": [1000, 2000, 23000, 23000, 23000],
            "dest": [2000, 2000, 20000, 21000, 22000],
        }
    )
    # compare to distances from maz_to_maz table
    dist = pd.Series(network_los.get_mazpairs(od_df.orig, od_df.dest, "DIST")).astype(
        np.float32
    )
    # make sure we got the right values
    pdt.assert_series_equal(
        dist, pd.Series([0.24, 0.14, 2.55, 1.9, 0.62]).astype(np.float32)
    )

    skims = skim_dict.wrap("orig", "dest")
    skims.set_df(od_df)
    # assert no blending for DISTBIKE
    assert network_los.max_blend_distance.get("DISTBIKE", 0) == 0

    skim_dist = skims["DISTBIKE"]

    print(type(skims), type(skim_dist.iloc[0]))
    print(type(dist.iloc[0]))
    pdt.assert_series_equal(skim_dist, dist)

    # but should be different where maz-maz distance differs from skim backstop and blending desired
    # blending enabled for DIST
    assert network_los.max_blend_distance.get("DIST") > 0
    with pytest.raises(AssertionError) as excinfo:
        pdt.assert_series_equal(skims["DIST"], dist)


def test_three_zone():

    add_canonical_dirs("configs_3z")

    network_los = los.Network_LOS()

    assert network_los.setting("zone_system") == los.THREE_ZONE

    assert "z3_taz_skims.omx" in network_los.omx_file_names("taz")

    assert network_los.blend_distance_skim_name == "DIST"

    network_los.load_data()

    od_df = pd.DataFrame(
        {
            "orig": [1000, 2000, 23000, 23000, 23000],
            "dest": [2000, 2000, 20000, 21000, 22000],
        }
    )

    dist = network_los.get_mazpairs(od_df.orig, od_df.dest, "DIST").astype(np.float32)
    np.testing.assert_almost_equal(dist, [0.24, 0.14, 2.55, 1.9, 0.62])


def test_30_minute_windows():

    add_canonical_dirs("configs_test_misc")
    network_los = los.Network_LOS(los_settings_file_name="settings_30_min.yaml")

    assert network_los.skim_time_period_label(1) == "EA"
    assert network_los.skim_time_period_label(16) == "AM"
    assert network_los.skim_time_period_label(24) == "MD"
    assert network_los.skim_time_period_label(36) == "PM"
    assert network_los.skim_time_period_label(46) == "EV"

    np.testing.assert_array_equal(
        network_los.skim_time_period_label(pd.Series([1, 16, 24, 36, 46])),
        np.array(["EA", "AM", "MD", "PM", "EV"]),
    )


def test_60_minute_windows():

    add_canonical_dirs("configs_test_misc")
    network_los = los.Network_LOS(los_settings_file_name="settings_60_min.yaml")

    assert network_los.skim_time_period_label(1) == "EA"
    assert network_los.skim_time_period_label(8) == "AM"
    assert network_los.skim_time_period_label(12) == "MD"
    assert network_los.skim_time_period_label(18) == "PM"
    assert network_los.skim_time_period_label(23) == "EV"

    np.testing.assert_array_equal(
        network_los.skim_time_period_label(pd.Series([1, 8, 12, 18, 23])),
        np.array(["EA", "AM", "MD", "PM", "EV"]),
    )


def test_1_week_time_window():

    add_canonical_dirs("configs_test_misc")
    network_los = los.Network_LOS(los_settings_file_name="settings_1_week.yaml")

    assert network_los.skim_time_period_label(1) == "Sunday"
    assert network_los.skim_time_period_label(2) == "Monday"
    assert network_los.skim_time_period_label(3) == "Tuesday"
    assert network_los.skim_time_period_label(4) == "Wednesday"
    assert network_los.skim_time_period_label(5) == "Thursday"
    assert network_los.skim_time_period_label(6) == "Friday"
    assert network_los.skim_time_period_label(7) == "Saturday"

    weekly_series = network_los.skim_time_period_label(pd.Series([1, 2, 3, 4, 5, 6, 7]))

    np.testing.assert_array_equal(
        weekly_series,
        np.array(
            [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
        ),
    )


def test_skim_time_periods_future_warning():

    add_canonical_dirs("configs_test_misc")

    with pytest.warns(FutureWarning) as warning_test:
        network_los = los.Network_LOS(
            los_settings_file_name="settings_legacy_hours_key.yaml"
        )

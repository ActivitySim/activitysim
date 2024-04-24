# ActivitySim
# See full license in LICENSE.txt.
import os
from ast import literal_eval
import pandas as pd
import numpy as np
import pandas.testing as pdt

from activitysim.abm.models.util.school_escort_tours_trips import (
    create_bundle_attributes,
    create_child_escorting_stops,
    create_chauf_trip_table,
)


def test_create_bundle_attributes():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    inbound_input = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_inbound__input.pkl")
    )
    inbound_expected = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_inbound__output.pkl")
    )

    outbound_input = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_outbound_cond__input.pkl")
    )
    outbound_expected = pd.read_pickle(
        os.path.join(data_dir, "create_bundle_attributes_outbound_cond__output.pkl")
    )
    inbound_result = create_bundle_attributes(inbound_input)
    pdt.assert_frame_equal(inbound_result, inbound_expected, check_dtype=False)

    outbound_result = create_bundle_attributes(outbound_input)
    pdt.assert_frame_equal(outbound_result, outbound_expected, check_dtype=False)


def test_create_chauf_trip_table():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__input.pkl")
    )
    chauf_trip_bundles = create_chauf_trip_table(bundles.copy())

    chauf_trip_bundles_expected = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__output.pkl")
    )

    pdt.assert_frame_equal(chauf_trip_bundles, chauf_trip_bundles_expected)


def test_create_child_escorting_stops():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_child_escorting_stops__input.pkl")
    )

    escortee_trips = []
    for escortee_num in range(0, int(bundles.num_escortees.max()) + 1):
        escortee_bundles = create_child_escorting_stops(bundles.copy(), escortee_num)
        escortee_trips.append(escortee_bundles)

    escortee_trips = pd.concat(escortee_trips)

    escortee_trips_expected = pd.read_pickle(
        os.path.join(data_dir, "create_child_escorting_stops__output.pkl")
    )

    pdt.assert_frame_equal(escortee_trips, escortee_trips_expected)


if __name__ == "__main__":
    test_create_bundle_attributes()
    test_create_chauf_trip_table()
    test_create_child_escorting_stops()

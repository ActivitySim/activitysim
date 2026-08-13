from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
from ast import literal_eval

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import activitysim.abm.models.school_escorting as school_escorting
from activitysim.abm.models.school_escorting import (
    SchoolEscortSettings,
    assign_school_escort_bundle_ids,
    create_school_escorting_bundles_table,
    determine_escorting_participants,
)
from activitysim.abm.models.util import canonical_ids
from activitysim.abm.models.util.school_escort_tours_trips import (
    create_bundle_attributes,
    create_chauf_escort_trips,
    create_chauf_trip_table,
    create_child_escorting_stops,
    create_pure_school_escort_tours,
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


def _make_escorting_persons():
    """Create households with tied and untied participant rankings."""
    return pd.DataFrame(
        {
            "person_id": [101, 102, 103, 104, 201, 202, 203, 204],
            "household_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "ptype": [1, 1, 8, 8, 1, 4, 8, 8],
            "sex": [1, 1, 2, 2, 1, 2, 1, 1],
            "age": [40, 40, 9, 9, 45, 42, 7, 12],
            "is_student": [False, False, True, True, False, False, True, True],
            "cdap_activity": ["M"] * 8,
        }
    ).set_index("person_id")


def _participant_assignments(persons):
    """Return only the participant ID columns produced for test households."""
    choosers = pd.DataFrame({"household_id": [1, 2], "home_zone_id": [5, 6]}).set_index(
        "household_id"
    )
    model_settings = SchoolEscortSettings(ALTS="dummy")
    choosers, participant_columns = determine_escorting_participants(
        choosers, persons, model_settings
    )
    return choosers[participant_columns]


def test_determine_escorting_participants_order_independent():
    """Participant assignment is independent of input order and MP slicing."""
    persons = _make_escorting_persons()
    baseline = _participant_assignments(persons)

    candidates = [
        persons.iloc[::-1],
        persons.iloc[[3, 0, 2, 1, 7, 4, 6, 5]],
        persons[persons["household_id"] == 1],
        persons[persons["household_id"] == 2],
    ]
    for reordered in candidates:
        result = _participant_assignments(reordered)
        common = result.index.intersection(baseline.index)
        pdt.assert_frame_equal(
            result.loc[common].sort_index(), baseline.loc[common].sort_index()
        )


def test_determine_escorting_participants_ranking_and_tie_breaks():
    """Weights and ages rank first, with person ID resolving ties."""
    assignments = _participant_assignments(_make_escorting_persons())

    assert assignments.loc[1, "chauf_id1"] == 101
    assert assignments.loc[1, "chauf_id2"] == 102
    assert assignments.loc[1, "child_id1"] == 103
    assert assignments.loc[1, "child_id2"] == 104

    assert assignments.loc[2, "chauf_id1"] == 202
    assert assignments.loc[2, "chauf_id2"] == 201
    assert assignments.loc[2, "child_id1"] == 203
    assert assignments.loc[2, "child_id2"] == 204


def _make_escort_bundles_for_ids():
    """Create bundle rows whose input order should not affect their IDs."""
    direction_dtype = pd.CategoricalDtype(["outbound", "inbound"])
    bundles = pd.DataFrame(
        {
            "household_id": [100, 100, 100, 200, 200],
            "school_escort_direction": [
                "inbound",
                "inbound",
                "outbound",
                "inbound",
                "outbound",
            ],
            "bundle_num": [1, 2, 1, 1, 1],
        }
    )
    bundles["school_escort_direction"] = bundles["school_escort_direction"].astype(
        direction_dtype
    )
    return bundles


def test_assign_school_escort_bundle_ids_order_independent():
    """Semantic bundle keys produce stable IDs for arbitrarily ordered rows."""
    bundles = _make_escort_bundles_for_ids()
    baseline = assign_school_escort_bundle_ids(bundles)
    shuffled = assign_school_escort_bundle_ids(bundles.sample(frac=1, random_state=19))

    keys = ["household_id", "school_escort_direction", "bundle_num"]
    pdt.assert_series_equal(
        baseline.set_index(keys)["bundle_id"].sort_index(),
        shuffled.set_index(keys)["bundle_id"].sort_index(),
    )
    assert baseline["bundle_id"].is_unique


def test_assign_school_escort_bundle_ids_rejects_duplicate_keys():
    """Duplicate semantic bundle keys fail instead of using input order."""
    bundles = _make_escort_bundles_for_ids()
    bundles = pd.concat([bundles, bundles.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate school escort bundle keys"):
        assign_school_escort_bundle_ids(bundles)


@pytest.mark.parametrize("stage", ["outbound_cond", "inbound"])
def test_create_bundles_orders_tied_escortees_and_uses_int64(monkeypatch, stage):
    """Equal travel times use child number while large IDs remain int64."""
    monkeypatch.setattr(school_escorting, "NUM_ESCORTEES", 3)
    monkeypatch.setattr(school_escorting, "NUM_CHAPERONES", 2)

    choosers = pd.DataFrame(
        {
            "household_id": [100],
            "home_zone_id": [5],
            "nbundles": [1],
            "bundle1": [1],
            "bundle2": [1],
            "bundle3": [0],
            "child_id1": [3_000_000_003],
            "child_id2": [3_000_000_004],
            "child_id3": [0],
            "chauf1": [2],
            "chauf2": [2],
            "chauf3": [0],
            "chauf_id1": [3_000_000_001],
            "chauf_id2": [3_000_000_002],
            "time_home_to_school1": [10.0],
            "time_home_to_school2": [10.0],
            "time_home_to_school3": [99.0],
            "alt": [2],
            "Description": ["test"],
        }
    ).set_index("household_id")
    tours = pd.DataFrame(
        {
            "tour_id": [5_000_000_011, 5_000_000_031, 5_000_000_041],
            "person_id": [3_000_000_001, 3_000_000_003, 3_000_000_004],
            "tour_type": ["work", "school", "school"],
            "tour_num": [1, 1, 1],
            "tour_category": ["mandatory"] * 3,
            "start": [9, 8, 8],
            "end": [17, 15, 15],
            "destination": [30, 20, 21],
            "origin": [5, 5, 5],
        }
    ).set_index("tour_id")

    bundles = create_school_escorting_bundles_table(choosers, tours, stage)

    assert list(bundles["child_order"].iloc[0]) == [1, 2, 3]
    assert bundles["escortees"].iloc[0] == "3000000003_3000000004"
    assert bundles["chauf_num"].dtype == np.dtype("int64")
    assert bundles["chauf_id"].dtype == np.dtype("int64")


class _FakeState:
    """Provide the tours table used only for categorical dtypes."""

    def __init__(self, tours):
        self._tours = tours

    def get_dataframe(self, name):
        assert name == "tours"
        return self._tours


def _fake_set_tour_index(state, tours, is_school_escorting=False, **kwargs):
    """Assign order-sensitive IDs so a missing tie-break is observable."""
    tours["tour_id"] = tours["person_id"] * 100 + tours["tour_type_num"]
    tours.set_index("tour_id", inplace=True)
    return tours


def _make_pure_escort_bundles():
    """Create two tied pure-escort tours for one chauffeur."""
    return pd.DataFrame(
        {
            "bundle_id": [1001, 1002],
            "household_id": [100, 100],
            "chauf_id": [1001, 1001],
            "escort_type": ["pure_escort", "pure_escort"],
            "school_escort_direction": ["outbound", "outbound"],
            "home_zone_id": [5, 5],
            "school_destinations": ["20", "21"],
            "school_starts": ["8", "8"],
            "school_ends": ["15", "15"],
        }
    )


def _run_pure_escort(bundles, monkeypatch):
    """Create pure-escort tours with a minimal workflow state."""
    monkeypatch.setattr(canonical_ids, "set_tour_index", _fake_set_tour_index)
    tours_for_dtype = pd.DataFrame(
        {
            "tour_category": pd.Categorical(["non_mandatory"]),
            "tour_type": pd.Categorical(["escort"]),
        }
    )
    result = create_pure_school_escort_tours(
        _FakeState(tours_for_dtype), bundles.copy()
    )
    return result.reset_index().set_index("bundle_id")


def test_create_pure_school_escort_tours_order_independent(monkeypatch):
    """Bundle ID resolves tied start times independently of input order."""
    bundles = _make_pure_escort_bundles()
    baseline = _run_pure_escort(bundles, monkeypatch)
    reversed_input = _run_pure_escort(bundles.iloc[::-1], monkeypatch)

    columns = [
        "tour_id",
        "tour_num",
        "tour_type_num",
        "next_pure_escort_start",
    ]
    pdt.assert_frame_equal(
        reversed_input[columns].sort_index(), baseline[columns].sort_index()
    )
    assert baseline.loc[1001, "tour_num"] == 1
    assert baseline.loc[1002, "tour_num"] == 2


def test_create_chauf_escort_trips_uses_int64_tour_ids():
    """Chauffeur trips retain tour IDs larger than a signed 32-bit integer."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__input.pkl")
    )
    bundles["chauf_tour_id"] += 3_000_000_000

    trips = create_chauf_escort_trips(bundles)

    assert trips["tour_id"].dtype == np.dtype("int64")
    assert trips["tour_id"].min() > np.iinfo(np.int32).max


if __name__ == "__main__":
    test_create_bundle_attributes()
    test_create_chauf_trip_table()
    test_create_child_escorting_stops()

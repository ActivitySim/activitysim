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


def create_column_as_concatentated_list(bundles, col_dict):
    for col, data in col_dict.items():
        bundles[col] = (
            pd.concat(data, axis=1, ignore_index=False)
            .reindex(bundles.index)
            .agg(lambda row: row.dropna().tolist(), axis=1)
        )
    return bundles


def create_chauf_trip_table_optimized(bundles):
    bundles["dropoff"] = bundles["school_escort_direction"] == "outbound"
    bundles["person_id"] = bundles["chauf_id"]

    original_index = bundles.index
    bundles.reset_index(drop=True, inplace=True)

    participants = []
    school_escort_trip_num = []
    outbound = []
    purposes = []
    destinations = []

    for i in range(bundles["num_escortees"].max()):
        dropoff_mask = (bundles["dropoff"] == True) & (
            bundles["num_escortees"] >= (i + 1)
        )
        pickup_mask = (bundles["dropoff"] == False) & (
            bundles["num_escortees"] >= (i + 1)
        )
        participants.append(
            bundles.loc[dropoff_mask, "escortees"].str.split("_").str[i:].str.join("_")
        )
        participants.append(
            bundles.loc[pickup_mask, "escortees"]
            .str.split("_")
            .str[: i + 1]
            .str.join("_")
        )
        school_escort_trip_num.append(
            pd.Series(index=bundles.loc[dropoff_mask | pickup_mask].index, data=(i + 1))
        )

        outbound_flag = np.where(
            # is outbound trip
            (
                bundles.loc[dropoff_mask | pickup_mask, "school_escort_direction"]
                == "outbound"
            )
            # or chauf is going back to pick up the first child
            | (
                (i == 0)
                & (
                    bundles.loc[dropoff_mask | pickup_mask, "escort_type"]
                    == "pure_escort"
                )
                & (
                    bundles.loc[dropoff_mask | pickup_mask, "school_escort_direction"]
                    == "inbound"
                )
            ),
            True,
            # chauf is inbound and has already picked up a child or taken their mandatory tour
            False,
        )
        outbound.append(
            pd.Series(
                index=bundles.loc[dropoff_mask | pickup_mask].index, data=outbound_flag
            )
        )

        purposes.append(
            pd.Series(
                index=bundles.loc[dropoff_mask | pickup_mask].index, data="escort"
            )
        )
        destinations.append(
            bundles.loc[dropoff_mask | pickup_mask, "school_destinations"]
            .str.split("_")
            .str[i]
        )

    # adding trip home for inbound
    inbound_mask = bundles["dropoff"] == False
    outbound.append(pd.Series(index=bundles.loc[inbound_mask].index, data=False))
    school_escort_trip_num.append(bundles.loc[inbound_mask, "num_escortees"] + 1)
    purposes.append(pd.Series(index=bundles.loc[inbound_mask].index, data="home"))
    destinations.append(bundles.loc[inbound_mask, "home_zone_id"])
    # kids aren't in the car until after they are picked up, inserting empty car for first trip
    participants.insert(0, pd.Series(index=bundles.loc[inbound_mask].index, data=""))

    # adding trip to work
    to_work_mask = (bundles["dropoff"] == True) & (
        bundles["escort_type"] == "ride_share"
    )
    outbound.append(pd.Series(index=bundles.loc[to_work_mask].index, data=True))
    school_escort_trip_num.append(bundles.loc[to_work_mask, "num_escortees"] + 1)
    purposes.append(bundles.loc[to_work_mask, "first_mand_tour_purpose"])
    destinations.append(bundles.loc[to_work_mask, "first_mand_tour_dest"])
    # kids have already been dropped off
    participants.append(pd.Series(index=bundles.loc[to_work_mask].index, data=""))

    bundles = create_column_as_concatentated_list(
        bundles,
        {
            "destination": destinations,
            "escort_participants": participants,
            "school_escort_trip_num": school_escort_trip_num,
            "outbound": outbound,
            "purpose": purposes,
        },
    )

    bundles.drop(columns=["dropoff"], inplace=True)
    bundles["person_id"] = bundles["person_id"].fillna(-1).astype(int)

    bundles.index = original_index
    return bundles


def test_create_chauf_trip_table():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__input.pkl")
    )
    # chauf_trip_bundles = bundles.apply(lambda row: create_chauf_trip_table(row), axis=1)
    chauf_trip_bundles = create_chauf_trip_table_optimized(bundles.copy())

    chauf_trip_bundles_expected = pd.read_pickle(
        os.path.join(data_dir, "create_chauf_trip_table__output.pkl")
    )
    chauf_trip_bundles_expected = chauf_trip_bundles_expected.astype(
        chauf_trip_bundles.dtypes.to_dict()
    )

    pdt.assert_frame_equal(chauf_trip_bundles, chauf_trip_bundles_expected)


def create_child_escorting_stops_optimized(bundles, escortee_num):
    bundles["num_escortees"] = bundles["escortees"].str.split("_").str.len()

    # only want the escortee bundles where the escortee_num is less than the number of escortees
    if (escortee_num > (bundles["num_escortees"] - 1)).all():
        return bundles
    bundles = bundles[escortee_num <= (bundles["num_escortees"] - 1)]
    original_index = bundles.index
    bundles.reset_index(drop=True, inplace=True)

    # intializing variables
    bundles["dropoff"] = bundles["school_escort_direction"] == "outbound"
    bundles["person_id"] = bundles["escortees"].str.split("_").str[escortee_num]
    bundles["tour_id"] = bundles["school_tour_ids"].str.split("_").str[escortee_num]
    participants = []
    purposes = []
    destinations = []
    school_escort_trip_num = []

    # looping up through the current escorting destination
    for i in range(escortee_num + 1):
        # dropping off children
        dropoff_mask = (bundles["dropoff"] == True) & (
            bundles["num_escortees"] >= (i + 1)
        )
        participants.append(
            bundles.loc[dropoff_mask, "escortees"].str.split("_").str[i:].str.join("_")
        )
        destinations.append(
            bundles.loc[dropoff_mask, "school_destinations"].str.split("_").str[i]
        )
        purposes.append(
            pd.Series(
                index=bundles.loc[dropoff_mask].index,
                data=np.where(
                    bundles.loc[dropoff_mask, "person_id"]
                    == bundles.loc[dropoff_mask, "escortees"].str.split("_").str[i],
                    "school",
                    "escort",
                ),
            )
        )

        # picking up children:
        pickup_mask = (bundles["dropoff"] == False) & (
            bundles["num_escortees"] >= (escortee_num + i + 1)
        )
        participants.append(
            bundles.loc[pickup_mask, "escortees"]
            .str.split("_")
            .str[: escortee_num + i + 1]
            .str.join("_")
        )
        is_last_stop = i == (
            bundles.loc[pickup_mask, "escortees"].str.split("_").str.len() - 1
        )
        destinations.append(
            pd.Series(
                index=bundles.loc[pickup_mask].index,
                data=np.where(
                    is_last_stop,
                    bundles.loc[pickup_mask, "home_zone_id"],
                    bundles.loc[pickup_mask, "school_destinations"]
                    .str.split("_")
                    .str[escortee_num + i + 1],
                ),
            )
        )
        purposes.append(
            pd.Series(
                index=bundles.loc[pickup_mask].index,
                data=np.where(is_last_stop, "home", "escort"),
            )
        )

        school_escort_trip_num.append(
            pd.Series(index=bundles.loc[pickup_mask].index, data=(i + 1))
        )

    bundles = create_column_as_concatentated_list(
        bundles,
        {
            "escort_participants": participants,
            "school_escort_trip_num": school_escort_trip_num,
            "purpose": purposes,
            "destination": destinations,
        },
    )

    bundles.drop(columns=["dropoff"], inplace=True)
    bundles["person_id"] = bundles["person_id"].fillna(-1).astype(int)

    bundles.index = original_index
    return bundles


def test_create_child_escorting_stops():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bundles = pd.read_pickle(
        os.path.join(data_dir, "create_child_escorting_stops__input.pkl")
    )
    bundles = bundles[bundles.Alt == 15]

    escortee_trips = []
    for escortee_num in range(0, int(bundles.num_escortees.max()) + 1):
        # escortee_bundles = bundles.apply(
        #     lambda row: create_child_escorting_stops(row, escortee_num), axis=1
        # )
        escortee_bundles = create_child_escorting_stops_optimized(
            bundles.copy(), escortee_num
        )
        escortee_trips.append(escortee_bundles)

    escortee_trips = pd.concat(escortee_trips)
    escortee_trips = escortee_trips[escortee_trips.person_id > 0]

    escortee_trips_expected = pd.read_pickle(
        os.path.join(data_dir, "create_child_escorting_stops__output.pkl")
    )
    escortee_trips_expected = escortee_trips_expected[escortee_trips_expected.Alt == 15]
    escortee_trips_expected = escortee_trips_expected.astype(
        escortee_trips.dtypes.to_dict()
    )

    pdt.assert_frame_equal(escortee_trips, escortee_trips_expected)


if __name__ == "__main__":
    test_create_bundle_attributes()
    test_create_chauf_trip_table()
    # test_create_child_escorting_stops()

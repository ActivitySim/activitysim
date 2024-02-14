from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import tracing, workflow

# from .util import estimation

logger = logging.getLogger(__name__)


@workflow.step
def stop_frequency_university_parking(
    state: workflow.State, trips: pd.DataFrame, tours: pd.DataFrame
):
    """
    This model inserts parking trips on drive tours that include university parking as determined in the
    parking_location_choice_at_university model.  Parking trips are added to the trip table before
    and after groups of trips that are on campus zones.

    The main interface to this model is the stop_frequency_university_parking() function.
    This function is registered as a step in the example Pipeline.
    """

    trace_label = "stop_frequency_university_parking"
    model_settings_file_name = "stop_frequency_university_parking.yaml"

    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    parking_name = model_settings["PARKING_TRIP_NAME"]

    tours_with_parking = tours[tours["univ_parking_zone_id"].notna()]

    trip_choosers = trips[trips.tour_id.isin(tours_with_parking.index)]
    trips_without_parking = trips[~trips.tour_id.isin(tours_with_parking.index)]

    if len(trip_choosers) > 0:
        trip_choosers = pd.merge(
            trip_choosers.reset_index(),
            tours_with_parking["univ_parking_zone_id"].reset_index(),
            how="left",
            on="tour_id",
        )
        trip_choosers.set_index("trip_id", inplace=True, verify_integrity=True)

        # will duplicate first and last campus trips to "insert" parking trips.
        # this duplication also sets the depart times where the parking trip to
        # campus has the same depart as the original trip and the parking trip
        # from campus has a depart time matching the following trip
        trip_choosers["duplicates_needed"] = 1
        trip_choosers["park_before"] = False
        trip_choosers["park_after"] = False
        atwork_campus_subtours = trip_choosers["purpose"] == "atwork"

        # looking at each tour individually
        trips_grouped = trip_choosers.groupby(["tour_id"])["parked_at_university"]
        # to campus
        trip_choosers.loc[
            (trip_choosers["parked_at_university"] == True)
            & (trips_grouped.transform("shift", 1).fillna(False) == False),
            "park_before",
        ] = True
        # from campus
        trip_choosers.loc[
            (trip_choosers["parked_at_university"] == False)
            & (trips_grouped.transform("shift", 1).fillna(False) == True),
            "park_after",
        ] = True

        trip_choosers.loc[
            trip_choosers["park_before"] | trip_choosers["park_after"],
            "duplicates_needed",
        ] = 2

        # atwork subtours that came from a parent tour that is already parked on campus
        # do not need parking trips.  This assumes atwork subtours can not go back to
        # get car from parked location.
        parked_atwork_subtour_ids = trip_choosers.loc[
            (trip_choosers["purpose"] == "Work")
            & (trip_choosers["primary_purpose"] == "atwork")
            & (trip_choosers["parked_at_university"] == True),
            "tour_id",
        ]
        parked_atwork_trips = trip_choosers["tour_id"].isin(parked_atwork_subtour_ids)

        trip_choosers.loc[parked_atwork_trips, "park_before"] = False
        trip_choosers.loc[parked_atwork_trips, "park_after"] = False
        trip_choosers.loc[parked_atwork_trips, "duplicates_needed"] = 1

        logger.info(
            "creating %d parking trips",
            (trip_choosers["park_before"] | trip_choosers["park_after"]).sum(),
        )

        # duplicating trips in table
        trip_choosers = trip_choosers.reset_index()
        trip_choosers = trip_choosers.take(
            np.repeat(
                trip_choosers.index.values, trip_choosers.duplicates_needed.values
            )
        )
        trip_choosers = trip_choosers.reset_index(drop=True)

        # recompute fields dependent on trip_num sequence
        grouped = trip_choosers.groupby(["tour_id", "outbound"])
        trip_choosers["trip_num"] = grouped.cumcount() + 1
        trip_choosers["trip_count"] = trip_choosers["trip_num"] + grouped.cumcount(
            ascending=False
        )

        # first duplicatd trip is parking trip if going to campus
        park_to_campus = (trip_choosers["park_before"] == True) & (
            trip_choosers["park_before"].shift(-1) == True
        )

        # second duplicatd trip is parking trip if going away from campus
        park_from_campus = (trip_choosers["park_after"] == True) & (
            trip_choosers["park_after"].shift(1) == False
        )

        park_trips = park_to_campus | park_from_campus

        # check if parking_name is in the purpose category
        if not parking_name in trip_choosers.purpose.cat.categories:
            trip_choosers.purpose = trip_choosers.purpose.cat.add_categories(
                [parking_name]
            )
        trip_choosers.loc[park_trips, "purpose"] = parking_name
        trip_choosers.loc[park_trips, "destination_logsum"] = pd.NA
        trip_choosers.loc[park_trips, "destination"] = trip_choosers.loc[
            park_trips, "univ_parking_zone_id"
        ]
        trip_choosers.loc[park_trips, "original_school_taz"] = pd.NA

        # need to change subsequent origin for trips that are going to parking lot
        trip_choosers["last_destination"] = trip_choosers.groupby("tour_id")[
            "destination"
        ].transform("shift")
        trip_choosers["origin"] = np.where(
            trip_choosers["last_destination"].notna()
            & (trip_choosers["last_destination"] != trip_choosers["origin"]),
            trip_choosers["last_destination"],
            trip_choosers["origin"],
        )
        trip_choosers.drop(columns="last_destination", inplace=True)

        trip_choosers["parked_at_university"] = (
            trip_choosers.groupby("tour_id")["parked_at_university"]
            .transform("shift")
            .fillna(False)
        )

        # all atwork subtour trips are parked if parent tour is parked at work
        trip_choosers.loc[
            trip_choosers["tour_id"].isin(parked_atwork_subtour_ids),
            "parked_at_university",
        ] = True

        trips = pd.concat(
            [
                trip_choosers[trips.reset_index().columns],
                trips_without_parking.reset_index(),
            ],
            ignore_index=True,
        )

    else:
        trips.reset_index(inplace=True)

    trips["origin"] = trips["origin"].astype(int)
    trips["destination"] = trips["destination"].astype(int)
    trips["tour_includes_parking"] = np.where(
        trips["tour_id"].isin(tours_with_parking.index), 1, 0
    )

    # resetting trip_id's
    trips["trip_id_pre_parking"] = trips["trip_id"]

    # taken from stop_frequency.py
    # With 4 trips per leg originally, can have an additional 4 parking trips per leg
    # e.g. if trips 1 and 3 are on university, need parking to and from 1 and parking
    # to and from 3
    MAX_TRIPS_PER_LEG = 8
    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * MAX_TRIPS_PER_LEG) + trips.trip_num
    trips["trip_id"] = trips.tour_id * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num
    trips.sort_values(by="trip_id", inplace=True)

    trips.set_index("trip_id", inplace=True, verify_integrity=True)

    state.add_table("trips", trips)
    # since new trips were added inbetween other trips on the tour, the trip_id's changed
    # resetting random number generator for trips... does this have unintended consequences?
    state.get_rn_generator().drop_channel("trips")
    state.get_rn_generator().add_channel("trips", trips)

    tracing.print_summary(
        "stop_frequency_university_parking trip purposes",
        trips["purpose"],
        value_counts=True,
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(trips, label=trace_label, warn_if_empty=True)

from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import logit, los, tracing, workflow

# from .util import estimation

logger = logging.getLogger(__name__)


def closest_parking_zone_xwalk(univ_zones, parking_zones, network_los):
    """
    Create lookup table matching university zone to nearest parking location.
    If university zone has parking, that zone is selected.

    Parameters
    ----------
    univ_zones : pandas.Series
        zones to find the nearest parking location for
    parking_zones : pandas.Series
        zones with parking spaces
    network_los : Network_LOS object
        skim information

    Returns
    -------
    closest_parking_df : pandas.DataFrame
        index of university zone input parameter and a column closest_parking_zone
    """
    skim_dict = network_los.get_default_skim_dict()

    closest_zones = []
    for univ_zone in univ_zones.to_numpy():
        if univ_zone in parking_zones.to_numpy():
            # if zone has parking data, choose that zone
            closest_zones.append(univ_zone)
        else:
            # find nearest zone from distance skim
            parking_zone_idx = np.argmin(
                skim_dict.lookup(univ_zone, parking_zones.to_numpy(), "DIST")
            )
            parking_zone = parking_zones.to_numpy()[parking_zone_idx]
            closest_zones.append(parking_zone)

    closest_parking_df = pd.DataFrame(
        {"univ_zone": univ_zones, "closest_parking_zone": closest_zones}
    )
    closest_parking_df.set_index("univ_zone", inplace=True)

    return closest_parking_df


@workflow.step
def parking_location_choice_at_university(
    state: workflow.State,
    trips: pd.DataFrame,
    tours: pd.DataFrame,
    land_use: pd.DataFrame,
    network_los: los.Network_LOS,
):
    """
    This model selects a parking location for groups of trips that are on university campuses where
    the tour mode is auto.  Parking locations are sampled weighted by the number of parking spots.

    The main interface to this model is the parking_location_choice_at_university() function.
    This function is registered as a step in the example Pipeline.
    """

    trace_label = "parking_location_choice_at_university"
    model_settings_file_name = "parking_location_choice_at_university.yaml"
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)

    univ_codes_col = model_settings["LANDUSE_UNIV_CODE_COL_NAME"]
    univ_codes = model_settings["UNIV_CODES_THAT_REQUIRE_PARKING"]

    parking_spaces_col = model_settings["LANDUSE_PARKING_SPACES_COL_NAME"]
    parking_univ_code_col = model_settings["LANDUSE_PARKING_UNIV_CODE_COL_NAME"]

    parking_tour_modes = model_settings["TOUR_MODES_THAT_REQUIRE_PARKING"]
    nearest_lot_tour_purposes = model_settings["TOUR_PURPOSES_TO_NEAREST_LOT"]

    land_use_df = land_use

    # initialize univ parking columns
    trips["parked_at_university"] = False
    tours["univ_parking_zone_id"] = pd.NA

    all_univ_zones = land_use_df[land_use_df[univ_codes_col].isin(univ_codes)].index
    all_parking_zones = land_use_df[land_use_df[parking_spaces_col] > 0].index

    # grabbing all trips and tours that have a destination on a campus and selected tour mode
    trip_choosers = trips[trips["destination"].isin(all_univ_zones)]
    tour_choosers = tours[
        tours.index.isin(trip_choosers["tour_id"])
        & tours.tour_mode.isin(parking_tour_modes)
    ]

    # removing trips that did not have the right tour mode.  (Faster than merging tour mode first?)
    trip_choosers = trip_choosers[trip_choosers.tour_id.isin(tour_choosers.index)]
    trip_choosers.loc[trip_choosers["purpose"] != "Home", "parked_at_university"] = True

    logger.info("Running %s for %d tours", trace_label, len(tour_choosers))

    closest_parking_df = closest_parking_zone_xwalk(
        all_univ_zones, all_parking_zones, network_los
    )

    # Set parking locations for each university independently
    for univ_code in univ_codes:
        # selecting land use data
        univ_zones = land_use_df[land_use_df[univ_codes_col] == univ_code].reset_index()
        parking_univ_zones = land_use_df[
            land_use_df[parking_univ_code_col] == univ_code
        ].reset_index()

        if len(univ_zones) == 0:
            logger.info("No zones found for university code: %s", univ_code)
            continue

        if (len(parking_univ_zones) == 0) or (
            parking_univ_zones[parking_spaces_col].sum() == 0
        ):
            logger.info("No parking found for university code: %s", univ_code)
            continue

        # selecting tours that have trips attending this university's zone(s)
        univ_trip_choosers = trip_choosers[
            trip_choosers["destination"].isin(univ_zones.zone_id)
        ]
        parking_tours = tour_choosers.index.isin(univ_trip_choosers.tour_id)
        num_parking_tours = parking_tours.sum()

        # constructing probabilities based on the number of parking spaces
        # format is columns for each parking zone alternative and indexed by choosers
        # probabilities are the same for each row
        probs = (
            parking_univ_zones[parking_spaces_col]
            / parking_univ_zones[parking_spaces_col].sum()
        ).to_frame()
        probs.set_index(parking_univ_zones.zone_id, inplace=True)
        probs = probs.T
        probs = probs.loc[np.repeat(probs.index, num_parking_tours)]
        probs.set_index(tour_choosers[parking_tours].index, inplace=True)

        # making stable choices using ActivitySim's random number generator
        choices, rands = logit.make_choices(state, probs)
        choices = choices.map(pd.Series(probs.columns))
        tour_choosers.loc[parking_tours, "univ_parking_zone_id"] = choices

        # for tours that have purpose specified in model setting, set parking location to
        # nearest parking lot
        if nearest_lot_tour_purposes is not None:
            tours_nearest_lot = tour_choosers.primary_purpose.isin(
                nearest_lot_tour_purposes
            ) & tour_choosers.destination.isin(all_univ_zones)
            tour_choosers.loc[
                tours_nearest_lot, "univ_parking_zone_id"
            ] = tour_choosers.loc[tours_nearest_lot, "destination"].map(
                closest_parking_df["closest_parking_zone"]
            )

        logger.info(
            "Selected parking locations for %s tours for university with code: %s",
            num_parking_tours,
            univ_code,
        )

    # Overriding school_zone_id in persons table
    trips.loc[
        trips.index.isin(trip_choosers.index), "parked_at_university"
    ] = trip_choosers["parked_at_university"]
    tours.loc[
        tours.index.isin(tour_choosers.index), "univ_parking_zone_id"
    ] = tour_choosers["univ_parking_zone_id"]

    state.add_table("trips", trips)
    state.add_table("tours", tours)

    tracing.print_summary(
        "parking_location_choice_at_university zones",
        tours["univ_parking_zone_id"],
        value_counts=True,
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(tours, label=trace_label, warn_if_empty=True)

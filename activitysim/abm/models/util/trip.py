# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.canonical_ids import set_trip_index
from activitysim.core import config, inject
from activitysim.core.util import assign_in_place, reindex

logger = logging.getLogger(__name__)


def failed_trip_cohorts(trips, failed):

    # outbound trips in a tour with a failed outbound trip
    bad_outbound_trips = trips.outbound & (
        trips.tour_id.isin(trips.tour_id[failed & trips.outbound])
    )

    # inbound trips in a tour with a failed inbound trip
    bad_inbound_trips = ~trips.outbound & (
        trips.tour_id.isin(trips.tour_id[failed & ~trips.outbound])
    )

    bad_trips = bad_outbound_trips | bad_inbound_trips

    return bad_trips


def flag_failed_trip_leg_mates(trips_df, col_name):
    """
    set boolean flag column of specified name to identify failed trip leg_mates in place
    """

    failed_trip_leg_mates = (
        failed_trip_cohorts(trips_df, trips_df.failed) & ~trips_df.failed
    )
    trips_df.loc[failed_trip_leg_mates, col_name] = True

    # handle outbound and inbound legs independently
    # for ob in [True, False]:
    #     same_leg = (trips_df.outbound == ob)
    #     # tour_ids of all tours with a failed trip in this (outbound or inbound) leg direction
    #     bad_tours = trips_df.tour_id[trips_df.failed & same_leg].unique()
    #     # not-failed leg_mates of all failed trips in this (outbound or inbound) leg direction
    #     failed_trip_leg_mates = same_leg & (trips_df.tour_id.isin(bad_tours)) & ~trips_df.failed
    #     # set the flag column
    #     trips_df.loc[failed_trip_leg_mates, col_name] = True


def cleanup_failed_trips(trips):
    """
    drop failed trips and cleanup fields in leg_mates:

    trip_num        assign new ordinal trip num after failed trips are dropped
    trip_count      assign new count of trips in leg, sans failed trips
    first           update first flag as we may have dropped first trip (last trip can't fail)
    next_trip_id    assign id of next trip in leg after failed trips are dropped
    """

    if trips.failed.any():
        logger.warning(
            "cleanup_failed_trips dropping %s failed trips" % trips.failed.sum()
        )

        trips["patch"] = False
        flag_failed_trip_leg_mates(trips, "patch")

        # drop the original failures
        trips = trips[~trips.failed]

        # increasing trip_id order
        patch_trips = trips[trips.patch].sort_index()

        # recompute fields dependent on trip_num sequence
        grouped = patch_trips.groupby(["tour_id", "outbound"])
        patch_trips["trip_num"] = grouped.cumcount() + 1
        # FIXME - 'clever' hack to avoid regroup - implementation dependent optimization that could change
        patch_trips["trip_count"] = patch_trips["trip_num"] + grouped.cumcount(
            ascending=False
        )

        assign_in_place(trips, patch_trips[["trip_num", "trip_count"]])

        del trips["patch"]

    del trips["failed"]

    return trips


def generate_alternative_sizes(max_duration, max_trips):
    """
    Builds a lookup Numpy array pattern sizes based on the
    number of trips in the leg and the duration available
    to the leg.
    :param max_duration:
    :param max_trips:
    :return:
    """

    def np_shift(xs, n, fill_zero=True):
        if n >= 0:
            shift_array = np.concatenate((np.full(n, np.nan), xs[:-n]))
        else:
            shift_array = np.concatenate((xs[-n:], np.full(-n, np.nan)))
        return (
            np.nan_to_num(shift_array, np.nan).astype(int) if fill_zero else shift_array
        )

    levels = np.empty([max_trips, max_duration + max_trips])
    levels[0] = np.arange(1, max_duration + max_trips + 1)

    for level in np.arange(1, max_trips):
        levels[level] = np_shift(
            np.cumsum(np_shift(levels[level - 1], 1)), -1, fill_zero=False
        )

    return levels[:, : max_duration + 1].astype(int)


def get_time_windows(residual, level):
    """

    :param residual:
    :param level:
    :return:
    """
    ranges = []

    for a in np.arange(residual + 1):
        if level > 1:
            windows = get_time_windows(residual - a, level - 1)
            width_dim = len(windows.shape) - 1
            ranges.append(np.vstack([np.repeat(a, windows.shape[width_dim]), windows]))
        else:
            return np.arange(residual + 1)
    return np.concatenate(ranges, axis=1)


@inject.injectable()
def stop_frequency_alts():
    # alt file for building trips even though simulation is simple_simulate not interaction_simulate
    file_path = config.config_file_path("stop_frequency_alternatives.csv")
    df = pd.read_csv(file_path, comment="#")
    df.set_index("alt", inplace=True)
    return df


def initialize_from_tours(tours, stop_frequency_alts, addtl_tour_cols_to_preserve=None):
    """
    Instantiates a trips table based on tour-level attributes: stop frequency,
    tour origin, tour destination.
    """

    OUTBOUND_ALT = "out"
    assert OUTBOUND_ALT in stop_frequency_alts.columns

    # get the actual alternatives for each person - have to go back to the
    # stop_frequency_alts dataframe to get this - the stop_frequency choice
    # column has the index values for the chosen alternative

    trips = stop_frequency_alts.loc[tours.stop_frequency]

    # assign unique tour identifiers to trips. NOTE: shouldn't be
    # necessary in most cases but there are some edge cases (e.g.
    # tour mode choice logsums) where tour_id will not be unique
    unique_tours = tours.copy().reset_index()
    trips.index = unique_tours.index

    """

    ::

      tours.stop_frequency    =>    proto trips table
      ________________________________________________________
                stop_frequency      |                out  in
      tour_id                       |     tour_id
      954910          1out_1in      |     954910       1   1
      985824          0out_1in      |     985824       0   1
    """

    # reformat with the columns given below
    trips = trips.stack().reset_index()
    trips.columns = ["tour_temp_index", "direction", "trip_count"]

    # tours legs have one more trip than stop
    trips.trip_count += 1

    # prefer direction as boolean
    trips["outbound"] = trips.direction == OUTBOUND_ALT

    """
           tour_temp_index direction  trip_count  outbound
    0             0           out           2         True
    1             0            in           1        False
    2             1           out           2         True
    3             1            in           3        False
    """

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    trips = trips.take(np.repeat(trips.index.values, trips.trip_count.values))
    trips = trips.reset_index(drop=True)

    grouped = trips.groupby(["tour_temp_index", "outbound"])
    trips["trip_num"] = grouped.cumcount() + 1

    trips["person_id"] = reindex(unique_tours.person_id, trips.tour_temp_index)
    trips["household_id"] = reindex(unique_tours.household_id, trips.tour_temp_index)
    trips["primary_purpose"] = reindex(
        unique_tours.primary_purpose, trips.tour_temp_index
    )

    if addtl_tour_cols_to_preserve is None:
        addtl_tour_cols_to_preserve = []
    for col in addtl_tour_cols_to_preserve:
        trips[col] = reindex(unique_tours[col], trips.tour_temp_index)

    # reorder columns and drop 'direction'
    trips = trips[
        [
            "person_id",
            "household_id",
            "tour_temp_index",
            "primary_purpose",
            "trip_num",
            "outbound",
            "trip_count",
        ]
        + addtl_tour_cols_to_preserve
    ]

    """
      person_id  household_id  tour_temp_index  primary_purpose trip_num  outbound  trip_count
    0     32927         32927        0             work            1        True           2
    1     32927         32927        0             work            2        True           2
    2     32927         32927        0             work            1       False           2
    3     32927         32927        0             work            2       False           2
    4     33993         33993        1             univ            1        True           1
    5     33993         33993        1             univ            1       False           2
    6     33993         33993        1             univ            2       False           2

    """

    # previously in trip_destination.py
    tour_destination = reindex(unique_tours.destination, trips.tour_temp_index).astype(
        np.int64
    )
    tour_origin = reindex(unique_tours.origin, trips.tour_temp_index).astype(np.int64)
    trips["destination"] = np.where(trips.outbound, tour_destination, tour_origin)
    trips["origin"] = np.where(trips.outbound, tour_origin, tour_destination)
    trips["failed"] = False

    # replace temp tour identifier with tour_id
    trips["tour_id"] = reindex(unique_tours.tour_id, trips.tour_temp_index)

    # trip ids are generated based on unique combination of `tour_id`, `outbound`,
    # and `trip_num`. When pseudo-trips are generated from pseudo-tours for the
    # purposes of computing logsums, `tour_id` won't be unique on `outbound` and
    # `trip_num`, so we use `tour_temp_index` instead. this will only be the case
    # when generating temporary pseudo-trips which won't get saved as outputs.
    if (
        trips.groupby(["tour_id", "outbound", "trip_num"])["person_id"].count().max()
        > 1
    ):
        trip_index_tour_id = "tour_temp_index"
    else:
        trip_index_tour_id = "tour_id"

    set_trip_index(trips, trip_index_tour_id)
    del trips["tour_temp_index"]

    return trips

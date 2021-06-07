# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np

from activitysim.core.util import assign_in_place


logger = logging.getLogger(__name__)


def failed_trip_cohorts(trips, failed):

    # outbound trips in a tour with a failed outbound trip
    bad_outbound_trips = \
        trips.outbound & (trips.tour_id.isin(trips.tour_id[failed & trips.outbound]))

    # inbound trips in a tour with a failed inbound trip
    bad_inbound_trips = \
        ~trips.outbound & (trips.tour_id.isin(trips.tour_id[failed & ~trips.outbound]))

    bad_trips = bad_outbound_trips | bad_inbound_trips

    return bad_trips


def flag_failed_trip_leg_mates(trips_df, col_name):
    """
    set boolean flag column of specified name to identify failed trip leg_mates in place
    """

    failed_trip_leg_mates = failed_trip_cohorts(trips_df, trips_df.failed) & ~trips_df.failed
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
        logger.warning("cleanup_failed_trips dropping %s failed trips" % trips.failed.sum())

        trips['patch'] = False
        flag_failed_trip_leg_mates(trips, 'patch')

        # drop the original failures
        trips = trips[~trips.failed]

        # increasing trip_id order
        patch_trips = trips[trips.patch].sort_index()

        # recompute fields dependent on trip_num sequence
        grouped = patch_trips.groupby(['tour_id', 'outbound'])
        patch_trips['trip_num'] = grouped.cumcount() + 1
        # FIXME - 'clever' hack to avoid regroup - implementation dependent optimization that could change
        patch_trips['trip_count'] = patch_trips['trip_num'] + grouped.cumcount(ascending=False)

        assign_in_place(trips, patch_trips[['trip_num', 'trip_count']])

        del trips['patch']

    del trips['failed']

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
        return np.nan_to_num(shift_array, np.nan).astype(int) if fill_zero else shift_array

    levels = np.empty([max_trips, max_duration + max_trips])
    levels[0] = np.arange(1, max_duration + max_trips + 1)

    for level in np.arange(1, max_trips):
        levels[level] = np_shift(np.cumsum(np_shift(levels[level - 1], 1)), -1, fill_zero=False)

    return levels[:, :max_duration+1].astype(int)


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

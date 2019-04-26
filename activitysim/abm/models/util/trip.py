# ActivitySim
# See full license in LICENSE.txt.
from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

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
        patch_trips['trip_count'] = patch_trips['trip_num'] + grouped.cumcount(ascending=False)

        assign_in_place(trips, patch_trips[['trip_num', 'trip_count']])

        del trips['patch']

    del trips['failed']

    return trips

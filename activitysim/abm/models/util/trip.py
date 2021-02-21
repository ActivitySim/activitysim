# ActivitySim
# See full license in LICENSE.txt.
import logging
import pandas as pd
import numpy as np

from activitysim.core import config
from activitysim.core.util import assign_in_place, reindex


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


def initialize_from_tours(tours):

    stop_frequency_alts = pd.read_csv(
        config.config_file_path('stop_frequency_alternatives.csv'), comment='#')
    stop_frequency_alts.set_index('alt', inplace=True)

    MAX_TRIPS_PER_LEG = 4  # max number of trips per leg (inbound or outbound) of tour
    OUTBOUND_ALT = 'out'
    assert OUTBOUND_ALT in stop_frequency_alts.columns

    # get the actual alternatives for each person - have to go back to the
    # stop_frequency_alts dataframe to get this - the stop_frequency choice
    # column has the index values for the chosen alternative

    trips = stop_frequency_alts.loc[tours.stop_frequency]

    # assign tour ids to the index
    trips.index = tours.index

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
    trips.columns = ['tour_id', 'direction', 'trip_count']

    # tours legs have one more leg than stop
    trips.trip_count += 1

    # prefer direction as boolean
    trips['outbound'] = trips.direction == OUTBOUND_ALT

    """
           tour_id direction  trip_count  outbound
    0       954910       out           2      True
    1       954910        in           2     False
    2       985824       out           1      True
    3       985824        in           2     False
    """

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    trips = trips.take(np.repeat(trips.index.values, trips.trip_count.values))
    trips = trips.reset_index(drop=True)

    grouped = trips.groupby(['tour_id', 'outbound'])
    trips['trip_num'] = grouped.cumcount() + 1

    trips['person_id'] = reindex(tours.person_id, trips.tour_id)
    trips['household_id'] = reindex(tours.household_id, trips.tour_id)
    trips['primary_purpose'] = reindex(tours.primary_purpose, trips.tour_id)

    # reorder columns and drop 'direction'
    trips = trips[['person_id', 'household_id', 'tour_id', 'primary_purpose',
                   'trip_num', 'outbound', 'trip_count']]

    """
      person_id  household_id  tour_id  primary_purpose trip_num  outbound  trip_count
    0     32927         32927   954910             work        1      True           2
    1     32927         32927   954910             work        2      True           2
    2     32927         32927   954910             work        1     False           2
    3     32927         32927   954910             work        2     False           2
    4     33993         33993   985824             univ        1      True           1
    5     33993         33993   985824             univ        1     False           2
    6     33993         33993   985824             univ        2     False           2

    """

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * MAX_TRIPS_PER_LEG) + trips.trip_num
    trips['trip_id'] = trips.tour_id * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num

    trips.set_index('trip_id', inplace=True, verify_integrity=True)

    # copied from trip_destination.py L503-507
    tour_destination = reindex(tours.destination, trips.tour_id).astype(np.int64)
    tour_origin = reindex(tours.origin, trips.tour_id).astype(np.int64)
    trips['destination'] = np.where(trips.outbound, tour_destination, tour_origin)
    trips['origin'] = np.where(trips.outbound, tour_origin, tour_destination)
    trips['failed'] = False

    return trips

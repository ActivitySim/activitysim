# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import numpy as np

from activitysim.util import reindex

from activitysim import activitysim as asim
from activitysim import tracing
from activitysim import pipeline

logger = logging.getLogger(__name__)


@orca.step()
def create_simple_trips(tours, households, persons, trace_hh_id):
    """
    Create a simple trip table
    """

    logger.info("Running simple trips table creation with %d tours" % len(tours.index))

    tours_df = tours.to_frame()
    tours_df.reset_index(inplace=True)

    tours_df['household_id'] = reindex(persons.household_id, tours_df.person_id)
    tours_df['TAZ'] = reindex(households.TAZ, tours_df.household_id)

    # create inbound and outbound records
    trips = pd.concat([tours_df, tours_df], ignore_index=True)

    # unique index
    trips.reset_index(drop=True, inplace=True)
    # name index so tracing knows how to slice
    trips.index.name = 'trip_id'

    # first half are outbound, second half are inbound
    trips['INBOUND'] = np.repeat([False, True], len(trips.index)/2)

    # TRIPID for outbound trips = 1, inbound_trips = 2
    trips['trip_num'] = np.repeat([1, 2], len(trips.index)/2)

    # set key fields from tour fields: 'TAZ','destination','start','end'
    trips['OTAZ'] = trips.TAZ
    trips['OTAZ'][trips.INBOUND] = trips.destination[trips.INBOUND]

    trips['DTAZ'] = trips.destination
    trips['DTAZ'][trips.INBOUND] = trips.TAZ[trips.INBOUND]

    trips['start_trip'] = trips.start
    trips['start_trip'][trips.INBOUND] = trips.end[trips.INBOUND]

    trips['end_trip'] = trips.end
    trips['end_trip'][trips.INBOUND] = trips.start[trips.INBOUND]

    trip_columns = ['tour_id', 'INBOUND', 'trip_num', 'OTAZ', 'DTAZ', 'start_trip', 'end_trip']
    trips = trips[trip_columns]

    orca.add_table("trips", trips)

    tracing.register_traceable_table('trips', trips)
    pipeline.get_rn_generator().add_channel(trips, 'trips')

    if trace_hh_id:
        tracing.trace_df(trips,
                         label="trips",
                         warn_if_empty=True)

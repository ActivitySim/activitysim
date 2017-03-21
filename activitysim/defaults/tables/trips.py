# ActivitySim
# See full license in LICENSE.txt.

import logging

import orca
import numpy as np
import pandas as pd
from activitysim.util.reindex import reindex

logger = logging.getLogger(__name__)


@orca.injectable()
def trips_table(tours_merged):
    """
    Create a simple trip table for testing using the tour table.  Create
    trips for the outbound and inbound leg using the tour home and destination zone
    and the tour start and end time period.
    """

    logger.info("Running simple trips table creation with %d tours" % len(tours_merged.index))

    # create inbound and outbound records
    inbound_trips = tours_merged.to_frame().copy()  # tours, persons, households
    outbound_trips = tours_merged.to_frame().copy()
    inbound_trips['INBOUND'] = True
    outbound_trips['INBOUND'] = False
    trips = pd.concat([inbound_trips, outbound_trips])
    del inbound_trips, outbound_trips

    # set key fields from tour fields: 'TAZ','destination','start','end'
    trips['TRIPID'] = 1  # outbound
    trips['TRIPID'][trips.INBOUND] = 2

    trips['OTAZ'] = trips.TAZ
    trips['OTAZ'][trips.INBOUND] = trips.destination[trips.INBOUND]

    trips['DTAZ'] = trips.destination
    trips['DTAZ'][trips.INBOUND] = trips.TAZ[trips.INBOUND]

    trips['start_trip'] = trips.start
    trips['start_trip'][trips.INBOUND] = trips.end[trips.INBOUND]

    trips['end_trip'] = trips.start
    trips['end_trip'][trips.INBOUND] = trips.end[trips.INBOUND]

    trips['SEQID'] = range(1, len(trips)+1)

    # FIXME - set index
    trips.index = trips.SEQID

    orca.add_table("trips", trips)


@orca.column("trips")
def start_period(trips, settings):
    cats = pd.cut(trips.start_trip,
                  settings['time_periods']['hours'],
                  labels=settings['time_periods']['labels'])
    # cut returns labelled categories but we convert to str
    return cats.astype(str)

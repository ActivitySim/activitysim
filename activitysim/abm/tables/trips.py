# ActivitySim
# See full license in LICENSE.txt.

import logging

import orca
import pandas as pd

logger = logging.getLogger(__name__)


@orca.column("trips")
def start_period(trips, settings):
    cats = pd.cut(trips.start_trip,
                  settings['time_periods']['hours'],
                  labels=settings['time_periods']['labels'])
    # cut returns labelled categories but we convert to str
    return cats.astype(str)


@orca.table()
def trips_merged(trips, tours):
    return orca.merge_tables(trips.name, tables=[
        trips, tours])

orca.broadcast('tours', 'trips', cast_index=True, onto_on='tour_id')

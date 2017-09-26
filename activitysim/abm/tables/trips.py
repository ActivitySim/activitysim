# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd

from activitysim.core import inject


logger = logging.getLogger(__name__)


@inject.column('trips')
def start_period(trips, settings):
    cats = pd.cut(trips.start_trip,
                  settings['skim_time_periods']['hours'],
                  labels=settings['skim_time_periods']['labels'])
    # cut returns labelled categories but we convert to str
    return cats.astype(str)


@inject.table()
def trips_merged(trips, tours):
    return inject.merge_tables(trips.name, tables=[trips, tours])


inject.broadcast('tours', 'trips', cast_index=True, onto_on='tour_id')

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd


from activitysim.core import inject
from activitysim.core import timetable as tt

logger = logging.getLogger(__name__)


@inject.injectable(cache=True)
def tdd_alts(configs_dir):
    # right now this file just contains the start and end hour
    f = os.path.join(configs_dir, 'tour_departure_and_duration_alternatives.csv')
    df = pd.read_csv(f)

    df['duration'] = df.end - df.start

    return df


@inject.table()
def person_windows(persons, tdd_alts):

    df = tt.create_timetable_windows(persons, tdd_alts)

    inject.add_table('person_windows', df)

    return df


@inject.injectable()
def timetable(person_windows, tdd_alts):
    return tt.TimeTable(person_windows.to_frame(), tdd_alts, person_windows.name)

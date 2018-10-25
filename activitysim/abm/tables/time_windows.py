# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import config
from activitysim.core import timetable as tt

logger = logging.getLogger(__name__)


@inject.injectable(cache=True)
def tdd_alts():
    # right now this file just contains the start and end hour
    f = config.config_file_path('tour_departure_and_duration_alternatives.csv')
    df = pd.read_csv(f)

    df['duration'] = df.end - df.start

    # - NARROW
    df = df.astype(np.int8)

    return df


@inject.table()
def person_windows(persons, tdd_alts):

    df = tt.create_timetable_windows(persons, tdd_alts)

    inject.add_table('person_windows', df)

    return df


@inject.injectable()
def timetable(person_windows, tdd_alts):

    logging.debug('@inject timetable')
    return tt.TimeTable(person_windows.to_frame(), tdd_alts, person_windows.name)

# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd

from activitysim.core import pipeline

from activitysim.core import inject


logger = logging.getLogger(__name__)


@inject.table(cache=True)
def person_time_windows(persons):

    assert persons.index is not None

    df = pd.DataFrame(index=persons.index)

    return df


class TimeTable(object):
    """

    """
    def __init__(self, time_windows_df):

        self.df = time_windows_df


@inject.injectable(cache=False)
def timetable(person_time_windows):

    logger.debug("loading timetable")
    return TimeTable(person_time_windows)

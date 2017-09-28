# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import inject


logger = logging.getLogger(__name__)


@inject.table()
def person_time_windows(persons):

    assert persons.index is not None

    time_windows = config.setting('time_windows')

    # hdf5 store converts these to strs, se we conform
    time_window_cols = [str(w) for w in time_windows]

    UNSCHEDULED = 0

    df = pd.DataFrame(data=UNSCHEDULED,
                      index=persons.index,
                      columns=time_window_cols)

    inject.add_table('person_time_windows', df)

    return df


class TimeTable(object):
    """

    """
    def __init__(self, table_name, time_window_df):

        self.time_window_df = time_window_df
        self.idx = pd.Series(range(len(time_window_df.index)), index=time_window_df.index)
        self.table_name = table_name

        self.row_ix = pd.Series(range(len(time_window_df.index)), index=time_window_df.index)

        int_time_windows = [int(c) for c in time_window_df.columns.values]
        self.time_ix = pd.Series(range(len(time_window_df.columns)), index=int_time_windows)

    def replace_table(self):
        pipeline.replace_table(self.table_name, self.time_window_df)

    def set_availability(self, df, id_col='person_id', start_col='start', end_col='end'):

        windows = self.time_window_df.as_matrix()

        row_ixs = df[id_col].map(self.row_ix)

        starts = np.asanyarray(df[start_col].map(self.time_ix))
        ends = np.asanyarray(df[end_col].map(self.time_ix))
        durations = ends - starts + 1

        # - entire tour
        # flattened array of ranges of duration
        # ranges = np.asanyarray([range(d) for d in duration]).flatten()
        ranges = np.arange(sum(durations)) + np.repeat(durations - np.cumsum(durations), durations)
        i = np.repeat(row_ixs, durations)
        j = np.repeat(starts, durations) + ranges
        windows[i, j] = 1

        # - start window
        windows[row_ixs, starts] += 1

        # - end window
        windows[row_ixs, ends] += 2


@inject.injectable()
def timetable(person_time_windows):

    return TimeTable(person_time_windows.name, person_time_windows.to_frame())

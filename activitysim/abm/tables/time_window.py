# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import inject


logger = logging.getLogger(__name__)

C_EMPTY = '0'
C_END = '4'
C_START = '2'
C_MIDDLE = '7'
C_START_END = '6'

# C_EMPTY = '0'
# C_END = '1'
# C_START = '1'
# C_MIDDLE = '1'
# C_START_END = '1'

I_EMPTY = int(C_EMPTY)
I_END = int(C_END)
I_START = int(C_START)
I_MIDDLE = int(C_MIDDLE)
I_START_END = int(C_START_END)


@inject.injectable(cache=True)
def tdd_alts(configs_dir):
    # right now this file just contains the start and end hour
    f = os.path.join(configs_dir, 'tour_departure_and_duration_alternatives.csv')
    df = pd.read_csv(f)

    df['duration'] = df.end - df.start

    return df


@inject.injectable(cache=True)
def tdd_windows(tdd_alts):

    w_strings = [
        C_EMPTY * (row.start - 5) +
        (C_START + C_MIDDLE * (row.duration - 1) if row.duration > 0 else '') +
        (C_END if row.duration > 0 else C_START_END) +
        (C_EMPTY * (23 - row.end))
        for idx, row in tdd_alts.iterrows()]

    windows = np.asanyarray([list(r) for r in w_strings]).astype(int)

    df = pd.DataFrame(data=windows, index=tdd_alts.index)

    return df


@inject.injectable(cache=True)
def tdd_intersects(tdd_windows):

    intersects = \
        (tdd_windows == I_MIDDLE) * ~I_EMPTY + \
        (tdd_windows == I_START) * ~I_END + \
        (tdd_windows == I_END) * ~I_START + \
        (tdd_windows == I_START_END) * ~I_START_END

    return intersects


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

        self.person_windows_table_name = table_name

        self.person_windows_df = time_window_df
        self.person_windows = self.person_windows_df.as_matrix()

        # series to map person_id to time_window ordinal index
        self.row_ix = pd.Series(range(len(time_window_df.index)), index=time_window_df.index)

        int_time_windows = [int(c) for c in time_window_df.columns.values]
        self.time_ix = pd.Series(range(len(time_window_df.columns)), index=int_time_windows)

        self.tdd_intersects_df = inject.get_injectable('tdd_intersects')
        self.tdd_windows_df = inject.get_injectable('tdd_windows')

    def replace_table(self):

        # it appears that writing to numpy array person_windows writes through to person_windows_df
        # so no need to refresh pandas dataframe
        pipeline.replace_table(self.person_windows_table_name, self.person_windows_df)

    def tour_available(self, person_ids, tdds):
        """

        Parameters
        ----------
        person_ids : pandas Series

        tdds

        Returns
        -------
        available : pandas Series of bool
            with same index as person_ids.index (presumably tour_id, but we don't care)
        """

        assert len(person_ids) == len(tdds)

        # df with one tdd_intersect row for each row in df
        tour_intersect_masks = self.tdd_intersects_df.loc[tdds]

        # numpy array with one time window row for each row in df
        tour_intersect_masks = tour_intersect_masks.as_matrix()

        # row idxs of tour_df group rows in person_windows
        row_ixs = person_ids.map(self.row_ix)

        available = ~np.bitwise_and(self.person_windows[row_ixs], tour_intersect_masks).any(axis=1)
        available = pd.Series(available, index=person_ids.index)

        return available

    def assign(self, person_ids, tdds):

        assert len(person_ids) == len(tdds)

        # vectorization doesn't work duplicates
        assert len(person_ids.index) == len(np.unique(person_ids.values))

        # df with one time window row for each row in df (tour_num group of tour_df)
        tour_windows = self.tdd_windows_df.loc[tdds]

        # numpy array with one time window row for each row in df
        tour_windows = tour_windows.as_matrix()

        # row idxs of tour_df group rows in person_windows
        row_ixs = person_ids.map(self.row_ix)

        self.person_windows[row_ixs] = np.bitwise_or(self.person_windows[row_ixs], tour_windows)


@inject.injectable()
def timetable(person_time_windows):

    return TimeTable(person_time_windows.name, person_time_windows.to_frame())

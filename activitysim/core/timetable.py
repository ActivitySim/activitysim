# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core import config
from activitysim.core import pipeline

logger = logging.getLogger(__name__)


# time_window states
# coding for time_window states is chosen so that we can quickly assign a tour
# to person_windows with a bitwise_or of alt's tdd_window with existing person_windows
# e.g. bitwise_or of I_START with I_END equals I_START_END
I_EMPTY = 0x0
I_END = 0x4
I_START = 0x2
I_MIDDLE = 0x7
I_START_END = 0x6

I_BIT_SHIFT = 3

COLLISIONS = [
    [I_START, I_START],
    [I_END, I_END],
    [I_MIDDLE, I_MIDDLE],
    [I_START, I_MIDDLE], [I_MIDDLE, I_START],
    [I_END, I_MIDDLE], [I_MIDDLE, I_END],
    [I_START_END, I_MIDDLE], [I_MIDDLE, I_START_END],
]

COLLISION_LIST = [a + (b << I_BIT_SHIFT) for a, b in COLLISIONS]


# str versions of time windows states
C_EMPTY = str(I_EMPTY)
C_END = str(I_END)
C_START = str(I_START)
C_MIDDLE = str(I_MIDDLE)
C_START_END = str(I_START_END)


def create_person_time_windows(persons, tdd_alts):

    assert persons.index is not None

    # pad time windows at both ends of day
    time_windows = range(tdd_alts.start.min() - 1, tdd_alts.end.max() + 2)

    # hdf5 store converts these to strs, se we conform
    time_window_cols = [str(w) for w in time_windows]

    UNSCHEDULED = 0

    df = pd.DataFrame(data=UNSCHEDULED,
                      index=persons.index,
                      columns=time_window_cols)

    return df


class TimeTable(object):
    """
                     tdd time window states
    tdd_alts_df      tdd_window_states_df
    start  end      '0' '1' '2' '3' '4'...
    5      5    ==>  0   6   0   0   0 ...
    5      6    ==>  0   2   4   0   0 ...
    5      7    ==>  0   2   7   4   0 ...

    """

    def __init__(self, table_name, person_windows_df, tdd_alts_df):
        """

        Parameters
        ----------
        table_name
        time_window_df
        tdd_alts_df
        """

        self.person_windows_table_name = table_name

        self.person_windows_df = person_windows_df
        self.person_windows = self.person_windows_df.as_matrix()

        # series to map person_id to time_window ordinal index
        self.row_ix = pd.Series(range(len(person_windows_df.index)), index=person_windows_df.index)

        int_time_windows = [int(c) for c in person_windows_df.columns.values]
        self.time_ix = pd.Series(range(len(person_windows_df.columns)), index=int_time_windows)

        # - pre-compute time window states for every tdd_alt
        # convert tdd_alts_df start, end times to time_windows
        min_period = min(int_time_windows)
        max_period = max(int_time_windows)
        # construct with strings so we can create runs of strings using char * int
        w_strings = [
            C_EMPTY * (row.start - min_period) +
            (C_START + C_MIDDLE * (row.duration - 1) if row.duration > 0 else '') +
            (C_END if row.duration > 0 else C_START_END) +
            (C_EMPTY * (max_period - row.end))
            for idx, row in tdd_alts_df.iterrows()]
        windows = np.asanyarray([list(r) for r in w_strings]).astype(int)
        self.tdd_window_states_df = pd.DataFrame(data=windows, index=tdd_alts_df.index)
        # print "\ntdd_window_states_df\n", self.tdd_window_states_df

    def get_person_windows(self):

        # It appears that assignments into person_windows write through to underlying pandas table.
        # Because we set person_windows = person_windows_df.as_matrix, though as_matrix does not
        # document this feature.

        # so no need to refresh pandas dataframe, but if we had to it would go here

        return self.person_windows_df

    def replace_table(self):
        """
        Save or replace person_windows_df  DataFrame to pipeline with saved table name
        (specified when object instantiated.)

        This is a convenience function in case caller instantiates object in one context
        (e.g. dependency injection) where it knows the pipeline table name, but wants to
        checkpoint the table in another context where it does not know that name.
        """

        # get person_windows_df from bottleneck function in case updates to self.person_window
        # do not write through to pandas dataframe
        pipeline.replace_table(self.person_windows_table_name, self.get_person_windows())

    def tour_available(self, person_ids, tdds):
        """
        test whether person's time window allows tour with specific tdd alt's time window

        Parameters
        ----------
        person_ids : pandas Series
            series of person_ids indexed by tour_id
        tdds : pandas series
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        available : pandas Series of bool
            with same index as person_ids.index (presumably tour_id, but we don't care)
        """

        assert len(person_ids) == len(tdds)

        # numpy array with one tdd_window_states_df row for tdds
        tour_windows = self.tdd_window_states_df.loc[tdds].as_matrix()

        # numpy array with one person_windows row for each person
        row_ixs = person_ids.map(self.row_ix).values
        person_windows = self.person_windows[row_ixs]

        x = tour_windows + (person_windows << I_BIT_SHIFT)

        available = ~np.isin(x, COLLISION_LIST).any(axis=1)
        available = pd.Series(available, index=person_ids.index)

        return available

    def assign(self, person_ids, tdds):
        """
        Assign tours (represented by tdd allt ids) to persons

        Updates self.person_windows numpy array. Assignments will no 'take' outside this object
        until/unless replace_table() called or updated timetable retrieved by get_person_windows()

        Parameters
        ----------
        person_ids : pandas Series
            series of person_ids indexed by tour_id
        tdds : pandas series
            series of tdd_alt ids, index irrelevant
        """

        assert len(person_ids) == len(tdds)

        # vectorization doesn't work duplicates
        assert len(person_ids.index) == len(np.unique(person_ids.values))

        # df with one time window row for each row in df (tour_num group of tour_df)
        tour_windows = self.tdd_window_states_df.loc[tdds]

        # numpy array with one time window row for each row in df
        tour_windows = tour_windows.as_matrix()

        # row idxs of tour_df group rows in person_windows
        row_ixs = person_ids.map(self.row_ix).values

        self.person_windows[row_ixs] = np.bitwise_or(self.person_windows[row_ixs], tour_windows)

    def adjacent_window_run_length(self, person_ids, periods, before):
        """
        Return the number of adjacent periods before or after specified period
        that are available (not in the middle of another tour.)

        Internal DRY method to implement adjacent_window_before and adjacent_window_after

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant
        before : bool
            Specify desired run length is of adjacent window before (True) or after (False)
        Returns
        -------

        """

        assert len(person_ids) == len(periods)

        person_row_ixs = person_ids.map(self.row_ix).values
        time_col_ixs = periods.map(self.time_ix).values

        # ones for available windows
        available = (self.person_windows[person_row_ixs] != I_MIDDLE) * 1
        # padding periods not available
        available[:, 0] = 0
        available[:, -1] = 0

        # column idxs of windows
        num_rows, num_cols = available.shape
        window_idx = np.tile(np.arange(0, num_cols), num_rows).reshape(num_rows, num_cols)

        if before:
            # ones after specified time, zeroes before
            before_mask = (window_idx < time_col_ixs.reshape(num_rows, 1)) * 1
            # index of first unavailable window after time
            first_unavailable = np.where((1-available)*before_mask, window_idx, 0).max(axis=1)
            available_run_length = time_col_ixs - first_unavailable - 1
        else:
            # ones after specified time, zeroes before
            after_mask = (window_idx > time_col_ixs.reshape(num_rows, 1)) * 1
            # index of first unavailable window after time
            first_unavailable = \
                np.where((1 - available) * after_mask, window_idx, num_cols).min(axis=1)
            available_run_length = first_unavailable - time_col_ixs - 1

        return pd.Series(available_run_length, index=person_ids.index)

    def adjacent_window_before(self, person_ids, periods):
        """
        Return number of adjacent periods before specified period that are available
        (not in the middle of another tour.)

        Implements CTRAMP MTCTM1 macro @@getAdjWindowBeforeThisPeriodAlt
        Function name is kind of a misnomer, but parallels that used in mtctm1 UECs

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series int
            Number of adjacent windows indexed by person_ids.index
        """
        return self.adjacent_window_run_length(person_ids, periods, before=True)

    def adjacent_window_after(self, person_ids, periods):
        """
        Return number of adjacent periods after specified period that are available
        (not in the middle of another tour.)

        Implements CTRAMP MTCTM1 macro @@adjWindowAfterThisPeriodAlt
        Function name is kind of a misnomer, but parallels that used in mtctm1 UECs

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series int
            Number of adjacent windows indexed by person_ids.index
        """
        return self.adjacent_window_run_length(person_ids, periods, before=False)

    def window_in_states(self, person_ids, periods, states):
        """
        Return boolean array indicating whether specified time_window is in list of states.

        Internal DRY method to implement previous_tour_ends and previous_tour_begins

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant
        states : list of int
            presumably (e.g. I_EMPTY, I_START...)

        Returns
        -------
        pandas Series boolean
            indexed by person_ids.index
        """

        # row ixs of tour_df group rows in person_windows
        person_row_ixs = person_ids.map(self.row_ix).values

        # col ixs of periods in person_windows
        time_col_ixs = periods.map(self.time_ix).values

        window = self.person_windows[person_row_ixs, time_col_ixs]

        return pd.Series(np.isin(window, states), person_ids.index)

    def previous_tour_ends(self, person_ids, periods):
        """
        Does a previously scheduled tour end in the specified period?

        Implements CTRAMP @@prevTourEndsThisDeparturePeriodAlt

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series boolean
            indexed by person_ids.index
        """
        return self.window_in_states(person_ids, periods, [I_END, I_START_END])

    def previous_tour_begins(self, person_ids, periods):
        """
        Does a previously scheduled tour begin in the specified period?

        Implements CTRAMP @@prevTourBeginsThisArrivalPeriodAlt

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series boolean
            indexed by person_ids.index
        """

        return self.window_in_states(person_ids, periods, [I_START, I_START_END])

    def remaining_periods_available(self, person_ids, starts, ends):
        """
        Determine number of periods remaining available after the time window from starts to ends
        is hypothetically scheduled

        Implements CTRAMP @@remainingPeriodsAvailableAlt

        The start and end periods will always be available after scheduling, so ignore them.
        The periods between start and end must be currently unscheduled, so assume they will become
        unavailable after scheduling this window.

        Parameters
        ----------
        person_ids : pandas Series int
            series of person_ids indexed by tour_id
        starts : pandas series int
            series of tdd_alt ids, index irrelevant
        ends : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        available : pandas Series int
            number periods available indexed by person_ids.index
        """

        # row idxs of tour_df group rows in person_windows
        row_ixs = person_ids.map(self.row_ix).values

        available = (self.person_windows[row_ixs] != I_MIDDLE).sum(axis=1)

        # don't count time window padding at both ends of day
        available -= 2

        available -= np.clip((ends - starts - 1), a_min=0, a_max=None)
        available = pd.Series(available, index=person_ids.index)

        return available

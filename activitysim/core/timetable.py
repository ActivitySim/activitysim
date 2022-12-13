# ActivitySim
# See full license in LICENSE.txt.

import logging
from builtins import object, range

import numba as nb
import numpy as np
import pandas as pd

from activitysim.core import chunk, pipeline

logger = logging.getLogger(__name__)


# time_window states
# coding for time_window states is chosen so that we can quickly assign a tour
# to windows with a bitwise_or of alt's tdd footprint with existing windows
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
    [I_START, I_MIDDLE],
    [I_MIDDLE, I_START],
    [I_END, I_MIDDLE],
    [I_MIDDLE, I_END],
    [I_START_END, I_MIDDLE],
    [I_MIDDLE, I_START_END],
]

COLLISION_LIST = [a + (b << I_BIT_SHIFT) for a, b in COLLISIONS]
COLLISION_ARRAY = np.asarray(COLLISION_LIST)

# str versions of time windows period states
C_EMPTY = str(I_EMPTY)
C_END = str(I_END)
C_START = str(I_START)
C_MIDDLE = str(I_MIDDLE)
C_START_END = str(I_START_END)


@nb.njit
def _fast_tour_available(
    tdds,
    tdd_footprints,
    window_row_ids,
    window_row_ix__mapper,
    self_windows,
):
    """

    Parameters
    ----------
    tdds : array-like, shape (k)
    tdd_footprints : array-like, shape (c, t)
    window_row_ids : array-like, shape (k)
    window_row_ix__mapper : FastMapping._mapper
    self_windows : array-like

    Returns
    -------
    array of bool, shape (k)
    """
    out = np.ones_like(tdds, dtype=np.bool_)
    for k in range(tdds.shape[0]):
        tour_footprints = tdd_footprints[tdds[k]]  # -> shape (t)
        row_ix = window_row_ix__mapper[window_row_ids[k]]
        windows = self_windows[row_ix]
        x = tour_footprints + (windows << I_BIT_SHIFT)
        stop = False
        for j in range(COLLISION_ARRAY.size):
            for i in range(x.size):
                if x[i] == COLLISION_ARRAY[j]:
                    out[k] = False
                    stop = True
                    break
            if stop:
                break
    return out


@nb.njit
def _available_run_length(
    available,
    before,
    periods,
    time_ix_mapper,
):
    num_rows = available.shape[0]
    num_cols = available.shape[1]
    _time_col_ix_map = np.arange(num_cols)
    available_run_length = np.zeros(num_rows, dtype=np.int32)
    for row in range(num_rows):
        _time_col_ix = time_ix_mapper[periods[row]]  # scalar
        if before:
            mask = (_time_col_ix_map < _time_col_ix) * 1
            # index of first unavailable window after time
            first_unavailable = np.where(
                (1 - available[row]) * mask, _time_col_ix_map, 0
            ).max()
            available_run_length[row] = _time_col_ix - first_unavailable - 1
        else:
            # ones after specified time, zeroes before
            mask = (_time_col_ix_map > _time_col_ix) * 1
            # index of first unavailable window after time
            first_unavailable = np.where(
                (1 - available[row]) * mask, _time_col_ix_map, num_cols
            ).min()
            available_run_length[row] = first_unavailable - _time_col_ix - 1
    return available_run_length


@nb.njit
def _available_run_length_1(
    windows,
    window_row_mapper,
    time_ix_mapper,
    before,
    window_row_id,
    period,
):
    num_cols = windows.shape[1]
    _time_col_ix_map = np.arange(num_cols)
    available = np.ones(num_cols, dtype=np.int8)
    available[0] = 0
    available[-1] = 0

    window_row = windows[window_row_mapper[window_row_id], :]
    for j in range(1, num_cols - 1):
        if window_row[j] != I_MIDDLE:
            available[j] = 1
        else:
            available[j] = 0

    _time_col_ix = time_ix_mapper[period]  # scalar
    if before:
        mask = (_time_col_ix_map < _time_col_ix) * 1
        # index of first unavailable window after time
        first_unavailable = np.where((1 - available) * mask, _time_col_ix_map, 0).max()
        available_run_length = _time_col_ix - first_unavailable - 1
    else:
        # ones after specified time, zeroes before
        mask = (_time_col_ix_map > _time_col_ix) * 1
        # index of first unavailable window after time
        first_unavailable = np.where(
            (1 - available) * mask, _time_col_ix_map, num_cols
        ).min()
        available_run_length = first_unavailable - _time_col_ix - 1
    return available_run_length


@nb.njit
def _available_run_length_2(
    windows,
    window_row_mapper,
    time_ix_mapper,
    before,
    window_row_id_values,
    periods,
):
    num_rows = window_row_id_values.shape[0]
    num_cols = windows.shape[1]
    _time_col_ix_map = np.arange(num_cols)
    available_run_length = np.zeros(num_rows, dtype=np.int32)
    available = np.ones(num_cols, dtype=np.int8)
    available[0] = 0
    available[-1] = 0
    for row in range(num_rows):

        row_ix = window_row_mapper[window_row_id_values[row]]
        window_row = windows[row_ix]
        for j in range(1, num_cols - 1):
            if window_row[j] != I_MIDDLE:
                available[j] = 1
            else:
                available[j] = 0

        _time_col_ix = time_ix_mapper[periods[row]]  # scalar
        if before:
            mask = (_time_col_ix_map < _time_col_ix) * 1
            # index of first unavailable window after time
            first_unavailable = np.where(
                (1 - available) * mask, _time_col_ix_map, 0
            ).max()
            available_run_length[row] = _time_col_ix - first_unavailable - 1
        else:
            # ones after specified time, zeroes before
            mask = (_time_col_ix_map > _time_col_ix) * 1
            # index of first unavailable window after time
            first_unavailable = np.where(
                (1 - available) * mask, _time_col_ix_map, num_cols
            ).min()
            available_run_length[row] = first_unavailable - _time_col_ix - 1
    return available_run_length


def tour_map(persons, tours, tdd_alts, persons_id_col="person_id"):

    sigil = {
        "empty": "   ",
        "overlap": "+++",
        "work": "WWW",
        "school": "SSS",
        "escort": "esc",
        "shopping": "shp",
        "othmaint": "mnt",
        "othdiscr": "dsc",
        "eatout": "eat",
        "social": "soc",
        "eat": "eat",
        "business": "bus",
        "maint": "mnt",
    }

    sigil_type = "S3"

    # we can only map scheduled tours
    tours = tours[tours.tdd.notnull()]

    # convert tdd_alts_df start, end times to time_windows
    min_period = tdd_alts.start.min()
    max_period = tdd_alts.end.max()
    n_periods = max_period - min_period + 1
    n_persons = len(persons.index)

    agenda = np.array([sigil["empty"]] * (n_periods * n_persons), dtype=sigil_type)
    agenda = agenda.reshape(n_persons, n_periods)

    scheduled = np.zeros_like(agenda, dtype=int)
    row_ix_map = pd.Series(list(range(n_persons)), index=persons.index)

    # construct with strings so we can create runs of strings using char * int
    w_strings = [
        "0" * (row.start - min_period)
        + "1" * (row.duration + 1)
        + "0" * (max_period - row.end)
        for idx, row in tdd_alts.iterrows()
    ]

    window_periods = np.asanyarray([list(r) for r in w_strings]).astype(int)
    window_periods_df = pd.DataFrame(data=window_periods, index=tdd_alts.index)

    for keys, nth_tours in tours.groupby(["tour_type", "tour_type_num"], sort=True):

        tour_type = keys[0]
        tour_sigil = sigil[tour_type]

        # numpy array with one time window row for each row in nth_tours
        tour_windows = window_periods_df.loc[nth_tours.tdd].values

        # row idxs of tour_df group rows in windows
        row_ixs = nth_tours[persons_id_col].map(row_ix_map).values

        # count tours per period
        scheduled[row_ixs] += tour_windows

        # assign tour_char to agenda tour periods
        agenda[row_ixs] = np.where(tour_windows, tour_sigil, agenda[row_ixs])

    # show tour overlaps
    agenda = np.where(scheduled > 1, sigil["overlap"], agenda)

    # a = pd.Series([' '.join(a) for a in agenda], index=persons.index)
    a = pd.DataFrame(
        data=agenda, columns=[str(w) for w in range(min_period, max_period + 1)]
    )

    a.index = persons.index
    a.index.name = persons_id_col

    return a


def create_timetable_windows(rows, tdd_alts):
    """
    create an empty (all available) timetable with one window row per rows.index

    Parameters
    ----------
    rows - pd.DataFrame or Series
        all we care about is the index
    tdd_alts - pd.DataFrame
        We expect a start and end column, and create a timetable to accomodate all alts
        (with on window of padding at each end)

    so if start is 5 and end is 23, we return something like this:

             4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
    person_id
    30       0  0  0  0  0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    109      0  0  0  0  0  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0

    Returns
    -------
    pd.DataFrame indexed by rows.index, and one column of int8 for each time window (plus padding)

    """

    # want something with an index to
    assert rows.index is not None

    # pad windows at both ends of day
    windows = list(range(tdd_alts.start.min() - 1, tdd_alts.end.max() + 2))

    # hdf5 store converts these to strs, se we conform
    window_cols = [str(w) for w in windows]

    UNSCHEDULED = 0

    df = pd.DataFrame(
        data=UNSCHEDULED, index=rows.index, columns=window_cols, dtype=np.int8
    )

    return df


class TimeTable(object):
    """
    ::

      tdd_alts_df      tdd_footprints_df
      start  end      '0' '1' '2' '3' '4'...
      5      5    ==>  0   6   0   0   0 ...
      5      6    ==>  0   2   4   0   0 ...
      5      7    ==>  0   2   7   4   0 ...

    """

    def __init__(self, windows_df, tdd_alts_df, table_name=None):
        """

        Parameters
        ----------
        table_name
        time_window_df
        tdd_alts_df
        """

        self.windows_table_name = table_name

        self.windows_df = windows_df
        self.windows = self.windows_df.values
        self.checkpoint_df = None

        # series to map window row index value to window row's ordinal index
        from ..core.fast_mapping import FastMapping

        self.window_row_ix = FastMapping(
            pd.Series(list(range(len(windows_df.index))), index=windows_df.index)
        )

        int_time_periods = [int(c) for c in windows_df.columns.values]
        self.time_ix = FastMapping(
            pd.Series(list(range(len(windows_df.columns))), index=int_time_periods)
        )

        # - pre-compute window state footprints for every tdd_alt
        min_period = min(int_time_periods)
        max_period = max(int_time_periods)
        # construct with strings so we can create runs of strings using char * int
        w_strings = [
            C_EMPTY * (row.start - min_period)
            + (C_START + C_MIDDLE * (row.duration - 1) if row.duration > 0 else "")
            + (C_END if row.duration > 0 else C_START_END)
            + (C_EMPTY * (max_period - row.end))
            for idx, row in tdd_alts_df.iterrows()
        ]

        # we want range index so we can use raw numpy
        assert (tdd_alts_df.index == list(range(tdd_alts_df.shape[0]))).all()
        self.tdd_footprints = np.asanyarray([list(r) for r in w_strings]).astype(int)

    def begin_transaction(self, transaction_loggers):
        """
        begin a transaction for an estimator or list of estimators
        this permits rolling timetable back to the state at the start of the transaction
        so that timetables can be built for scheduling override choices
        """
        if not isinstance(transaction_loggers, list):
            transaction_loggers = [transaction_loggers]
        for transaction_logger in transaction_loggers:
            transaction_logger.log(
                "timetable.begin_transaction %s" % self.windows_table_name
            )
        self.checkpoint_df = self.windows_df.copy()
        self.transaction_loggers = transaction_loggers
        pass

    def rollback(self):
        assert self.checkpoint_df is not None
        for logger in self.transaction_loggers:
            logger.log("timetable.rollback %s" % self.windows_table_name)
        self.windows_df = self.checkpoint_df
        self.windows = self.windows_df.values
        self.checkpoint_df = None
        self.transaction_loggers = None

    def export_for_numba(self):
        return dict(
            tt_row_mapper=self.window_row_ix._mapper,
            tt_col_mapper=self.time_ix._mapper,
            tt_windows=self.windows,
        )

    def slice_windows_by_row_id(self, window_row_ids):
        """
        return windows array slice containing rows for specified window_row_ids
        (in window_row_ids order)
        """
        row_ixs = self.window_row_ix.apply_to(window_row_ids.values)
        windows = self.windows[row_ixs]

        return windows

    def slice_windows_by_row_id_and_period(self, window_row_ids, periods):

        # row ixs of tour_df group rows in windows
        row_ixs = self.window_row_ix.apply_to(window_row_ids)

        # col ixs of periods in windows
        time_col_ixs = self.time_ix.apply_to(periods)

        windows = self.windows[row_ixs, time_col_ixs]

        return windows

    def get_windows_df(self):

        # It appears that assignments into windows write through to underlying pandas table.
        # because we set windows = windows_df.values, and since all the columns are the same type
        # so no need to refresh pandas dataframe, but if we had to it would go here

        # assert (self.windows_df.values == self.windows).all()
        return self.windows_df

    def replace_table(self):
        """
        Save or replace windows_df  DataFrame to pipeline with saved table name
        (specified when object instantiated.)

        This is a convenience function in case caller instantiates object in one context
        (e.g. dependency injection) where it knows the pipeline table name, but wants to
        checkpoint the table in another context where it does not know that name.
        """

        assert self.windows_table_name is not None
        if self.checkpoint_df is not None:
            for logger in self.transaction_loggers.values():
                logger.log(
                    "Attempt to replace_table while in transaction: %s"
                    % self.windows_table_name,
                    level=logging.ERROR,
                )
            raise RuntimeError("Attempt to replace_table while in transaction")

        # get windows_df from bottleneck function in case updates to self.person_window
        # do not write through to pandas dataframe
        pipeline.replace_table(self.windows_table_name, self.get_windows_df())

    def tour_available(self, window_row_ids, tdds):
        """
        test whether time window allows tour with specific tdd alt's time window

        Parameters
        ----------
        window_row_ids : pandas Series
            series of window_row_ids indexed by tour_id
        tdds : pandas series
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        available : pandas Series of bool
            with same index as window_row_ids.index (presumably tour_id, but we don't care)
        """
        if isinstance(tdds, pd.Series):
            tdds = tdds.astype(np.int32).to_numpy()
        else:
            tdds = tdds.astype(np.int32)
        if isinstance(window_row_ids, pd.Series):
            window_row_ids = window_row_ids.astype(np.int64).to_numpy()
        else:
            window_row_ids = window_row_ids.astype(np.int64)

        try:
            available = _fast_tour_available(
                tdds,
                self.tdd_footprints,
                window_row_ids,
                self.window_row_ix._mapper,
                self.windows,
            )
        except KeyError:
            # key error messages here may not have enough detail to be useful,
            # so we'll also log the row ids and mapper to help diagnose errors.
            logger.error("KeyError in _fast_tour_available")
            logger.error(f"{window_row_ids=}")
            logger.error(f"{self.window_row_ix._mapper=}")
            raise

        return available

    def assign(self, window_row_ids, tdds):
        """
        Assign tours (represented by tdd alt ids) to persons

        Updates self.windows numpy array. Assignments will not 'take' outside this object
        until/unless replace_table called or updated timetable retrieved by get_windows_df

        Parameters
        ----------
        window_row_ids : pandas Series
            series of window_row_ids indexed by tour_id
        tdds : pandas series
            series of tdd_alt ids, index irrelevant
        """

        assert len(window_row_ids) == len(tdds)

        # vectorization doesn't work duplicates
        assert len(window_row_ids.index) == len(np.unique(window_row_ids.values))

        # numpy array with one time window row for each person tdd
        tour_footprints = self.tdd_footprints[tdds.values.astype(int)]

        # row idxs of windows to assign to
        row_ixs = self.window_row_ix.apply_to(window_row_ids)

        self.windows[row_ixs] = np.bitwise_or(self.windows[row_ixs], tour_footprints)

    def assign_subtour_mask(self, window_row_ids, tdds):
        """
        ::

          index     window_row_ids   tdds
          20973389  20973389           26
          44612864  44612864            3
          48954854  48954854            7

          tour footprints
          [[0 0 2 7 7 7 7 7 7 4 0 0 0 0 0 0 0 0 0 0 0]
          [0 2 7 7 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
          [0 2 7 7 7 7 7 7 4 0 0 0 0 0 0 0 0 0 0 0 0]]

          subtour_mask
          [[7 7 0 0 0 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7]
          [7 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
          [7 0 0 0 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7 7]]

        """

        # expect window_row_ids for every row
        assert len(window_row_ids) == len(self.window_row_ix)

        assert len(window_row_ids) == len(tdds)

        self.windows.fill(0)
        self.assign(window_row_ids, tdds)

        # numpy array with one time window row for each person tdd
        tour_footprints = self.tdd_footprints[tdds.values.astype(int)]

        # row idxs of windows to assign to
        row_ixs = self.window_row_ix.apply_to(window_row_ids)

        self.windows[row_ixs] = (tour_footprints == 0) * I_MIDDLE

    def assign_footprints(self, window_row_ids, footprints):
        """
        assign footprints for specified window_row_ids

        This method is used for initialization of joint_tour timetables based on the
        combined availability of the joint tour participants

        Parameters
        ----------
        window_row_ids : pandas Series
            series of window_row_ids index irrelevant, but we want to use map()
        footprints : numpy array
            with one row per window_row_id and one column per time period
        """

        assert len(window_row_ids) == footprints.shape[0]

        # require same number of periods in footprints
        assert self.windows.shape[1] == footprints.shape[1]

        # vectorization doesn't work with duplicate row_ids
        assert len(window_row_ids.values) == len(np.unique(window_row_ids.values))

        # row idxs of windows to assign to
        row_ixs = self.window_row_ix.apply_to(window_row_ids)

        self.windows[row_ixs] = np.bitwise_or(self.windows[row_ixs], footprints)

    def pairwise_available(self, window1_row_ids, window2_row_ids):

        available1 = (self.slice_windows_by_row_id(window1_row_ids) != I_MIDDLE) * 1
        available2 = (self.slice_windows_by_row_id(window2_row_ids) != I_MIDDLE) * 1

        return available1 * available2

    def individually_available(self, window_row_ids):

        return (self.slice_windows_by_row_id(window_row_ids) != I_MIDDLE) * 1

    def adjacent_window_run_length(self, window_row_ids, periods, before):
        """
        Return the number of adjacent periods before or after specified period
        that are available (not in the middle of another tour.)

        Internal DRY method to implement adjacent_window_before and adjacent_window_after

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant
        before : bool
            Specify desired run length is of adjacent window before (True) or after (False)
        """
        assert len(window_row_ids) == len(periods)

        trace_label = "tt.adjacent_window_run_length"
        with chunk.chunk_log(trace_label):
            available_run_length = _available_run_length_2(
                self.windows,
                self.window_row_ix._mapper,
                self.time_ix._mapper,
                before,
                window_row_ids.values,
                periods.to_numpy(),
            )

            chunk.log_df(trace_label, "available_run_length", available_run_length)

        return pd.Series(available_run_length, index=window_row_ids.index)

    def adjacent_window_before(self, window_row_ids, periods):
        """
        Return number of adjacent periods before specified period that are available
        (not in the middle of another tour.)

        Implements MTC TM1 macro @@getAdjWindowBeforeThisPeriodAlt
        Function name is kind of a misnomer, but parallels that used in MTC TM1 UECs

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series int
            Number of adjacent windows indexed by window_row_ids.index
        """
        return self.adjacent_window_run_length(window_row_ids, periods, before=True)

    def adjacent_window_after(self, window_row_ids, periods):
        """
        Return number of adjacent periods after specified period that are available
        (not in the middle of another tour.)

        Implements MTC TM1 macro @@adjWindowAfterThisPeriodAlt
        Function name is kind of a misnomer, but parallels that used in MTC TM1 UECs

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series int
            Number of adjacent windows indexed by window_row_ids.index
        """
        return self.adjacent_window_run_length(window_row_ids, periods, before=False)

    def window_periods_in_states(self, window_row_ids, periods, states):
        """
        Return boolean array indicating whether specified window periods are in list of states.

        Internal DRY method to implement previous_tour_ends and previous_tour_begins

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant (one period per window_row_id)
        states : list of int
            presumably (e.g. I_EMPTY, I_START...)

        Returns
        -------
        pandas Series boolean
            indexed by window_row_ids.index
        """

        assert len(window_row_ids) == len(periods)

        window = self.slice_windows_by_row_id_and_period(window_row_ids, periods)

        return pd.Series(np.isin(window, states), window_row_ids.index)

    def previous_tour_ends(self, window_row_ids, periods):
        """
        Does a previously scheduled tour end in the specified period?

        Implements MTC TM1 @@prevTourEndsThisDeparturePeriodAlt

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant (one period per window_row_id)

        Returns
        -------
        pandas Series boolean
            indexed by window_row_ids.index
        """
        return self.window_periods_in_states(
            window_row_ids, periods, [I_END, I_START_END]
        )

    def previous_tour_begins(self, window_row_ids, periods):
        """
        Does a previously scheduled tour begin in the specified period?

        Implements MTC TM1 @@prevTourBeginsThisArrivalPeriodAlt

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        periods : pandas series int
            series of tdd_alt ids, index irrelevant

        Returns
        -------
        pandas Series boolean
            indexed by window_row_ids.index
        """

        return self.window_periods_in_states(
            window_row_ids, periods, [I_START, I_START_END]
        )

    def remaining_periods_available(self, window_row_ids, starts, ends):
        """
        Determine number of periods remaining available after the time window from starts to ends
        is hypothetically scheduled

        Implements MTC TM1 @@remainingPeriodsAvailableAlt

        The start and end periods will always be available after scheduling, so ignore them.
        The periods between start and end must be currently unscheduled, so assume they will become
        unavailable after scheduling this window.

        Parameters
        ----------
        window_row_ids : pandas Series int
            series of window_row_ids indexed by tour_id
        starts : pandas series int
            series of tdd_alt ids, index irrelevant (one per window_row_id)
        ends : pandas series int
            series of tdd_alt ids, index irrelevant (one per window_row_id)

        Returns
        -------
        available : pandas Series int
            number periods available indexed by window_row_ids.index
        """
        result = tt_remaining_periods_available(self, window_row_ids, starts, ends)
        return result

    def max_time_block_available(self, window_row_ids):
        """
        determine the length of the maximum time block available in the persons day

        Parameters
        ----------
        window_row_ids: pandas.Series

        Returns
        -------
            pandas.Series with same index as window_row_ids, and integer max_run_length of
        """
        result = pd.Series(
            _max_time_blocks_available_1(
                self.window_row_ix._mapper, self.windows, np.asarray(window_row_ids)
            ),
            index=window_row_ids.index,
        )
        return result


@nb.njit
def _max_time_block_available_1(windows_row):
    """

    Parameters
    ----------
    tt_windows : array[int8], 1 dimension
        Array of currently scheduled stuff
    tt_row_mapper : numba.typed.Dict[int,int]
        Maps value in `window_row_id` to row position in `tt_windows`.
    window_row_id : int

    Returns
    -------

    """
    max_block_avail = 0
    current_block = 0
    for i in range(1, windows_row.size - 1):
        if windows_row[i] != I_MIDDLE:
            current_block += 1
        else:
            current_block = 0
        if current_block > max_block_avail:
            max_block_avail = current_block
    return max_block_avail


@nb.njit
def _max_time_blocks_available_1(tt_window_row_ix, tt_windows, window_row_ids):
    max_blocks = np.zeros(window_row_ids.size, dtype=np.uint8)
    # FIXME consider dedupe/redupe window_row_ids for performance
    # as this may be called for alts with lots of duplicates (e.g. trip scheduling time pressure calculations)
    for j in range(window_row_ids.size):
        max_blocks[j] = _max_time_block_available_1(
            tt_windows[tt_window_row_ix[window_row_ids[j]]]
        )
    return max_blocks


@nb.njit
def sharrow_tt_max_time_block_available(tt_windows, tt_row_mapper, window_row_id):
    return _max_time_block_available_1(tt_windows[tt_row_mapper[window_row_id]])


def tt_slice_windows_by_row_id(tt_window_row_ix, tt_windows, window_row_ids):
    """
    return windows array slice containing rows for specified window_row_ids
    (in window_row_ids order)
    """
    row_ixs = tt_window_row_ix.apply_to(window_row_ids.values)
    windows = tt_windows[row_ixs]

    return windows


@nb.njit
def _count_windows_that_are_not_middles(
    window_row_id,  # int
    windows_mapper,  # nb.Dict[int,int]
    windows,  # ndarray
):
    row_ix = windows_mapper[window_row_id]
    x = 0
    for i in range(windows.shape[1]):
        if windows[row_ix, i] != I_MIDDLE:
            x += 1
    return x


@nb.njit
def sharrow_tt_remaining_periods_available(
    tt_windows,  # ndarray
    tt_row_mapper,  # nb.Dict[int,int]
    window_row_id,  # int
    starter,  # int
    ender,  # int
):
    """
    Number of periods remaining available after hypothetical scheduling

    This is what's left after a new tour or trip from `starts` to
    `ends` is hypothetically scheduled.

    Implements MTC TM1 @@remainingPeriodsAvailableAlt

    The start and end periods will always be available after
    scheduling, so ignore them. The periods between start and end
    must be currently unscheduled, so assume they will become
    unavailable after scheduling this window.

    Parameters
    ----------
    tt_windows : array[int8], 2 dimensions
        Array of currently scheduled stuff
    tt_row_mapper : numba.typed.Dict[int,int]
        Maps value in the `window_row_ids` to row positions in `windows`.
    window_row_id : int
        An identifier for which window row to use.
    starter : int
        The starting period of the new tour that will block windows.
    ender : int
        The ending period of the new tour that will block windows.

    Returns
    -------
    int
    """
    available = _count_windows_that_are_not_middles(
        window_row_id,
        tt_row_mapper,
        tt_windows,
    )
    # don't count time window padding at both ends of day
    available -= 2
    this_block = ender - starter - 1
    if this_block > 0:
        available -= this_block
    return available


@nb.njit
def _remaining_periods_available(
    windows,  # ndarray
    windows_row_mapper,
    window_row_ids,  # ndarray[int]
    starts,  # ndarray[int]
    ends,  # ndarray[int]
):
    """
    Number of periods remaining available after hypothetical scheduling

    This is what's left after a new tour or trip from `starts` to
    `ends` is hypothetically scheduled.

    Implements MTC TM1 @@remainingPeriodsAvailableAlt

    The start and end periods will always be available after
    scheduling, so ignore them. The periods between start and end
    must be currently unscheduled, so assume they will become
    unavailable after scheduling this window.

    Parameters
    ----------
    windows_row_mapper : numba.typed.Dict[int,int]
        Maps value in the `window_row_ids` to row positions in `windows`.
    windows : array[int8], 2 dimensions
    window_row_ids : array[int], 1-dimension
    starts : array[int]
        A 1-dimension array the same shape as `window_row_ids` which
        gives the starting period of the new tour that will block windows.
    ends : array[int], 1-dimension
        A 1-dimension array the same shape as `window_row_ids` which
        gives the ending period of the new tour that will block windows.

    Returns
    -------
    array[int]
        A 1-dimension array the same shape as `window_row_ids` which
        gives the number of available periods remaining.
    """
    result = np.empty(window_row_ids.shape, dtype=np.int64)
    for i in range(window_row_ids.shape[0]):
        result[i] = sharrow_tt_remaining_periods_available(
            windows,
            windows_row_mapper,
            window_row_ids[i],
            starts[i],
            ends[i],
        )
    return result


def tt_remaining_periods_available(tt, window_row_ids, starts, ends):
    """
    Number of periods remaining available after hypothetical scheduling

    That is, what's left after something from starts to ends is
    hypothetically scheduled

    Implements MTC TM1 @@remainingPeriodsAvailableAlt

    The start and end periods will always be available after
    scheduling, so ignore them. The periods between start and end
    must be currently unscheduled, so assume they will become
    unavailable after scheduling this window.

    Parameters
    ----------
    tt : TimeTable
    window_row_ids : pandas.Series[int]
        series of window_row_ids indexed by tour_id
    starts : pandas.Series[int]
        series of tdd_alt ids, index irrelevant (one per window_row_id)
    ends : pandas.Series[int]
        series of tdd_alt ids, index irrelevant (one per window_row_id)

    Returns
    -------
    available : pandas Series int
        number periods available indexed by window_row_ids.index
    """

    result = _remaining_periods_available(
        tt.windows,
        tt.window_row_ix._mapper,
        window_row_ids.values,
        starts.values,
        ends.values,
    )
    if isinstance(window_row_ids, pd.Series):
        result = pd.Series(result, index=window_row_ids.index)
    return result


@nb.njit
def _window_period_in_states(
    windows,
    windows_row_mapper,
    windows_col_mapper,
    window_row_id,
    period,
    state1,
    state2,
):
    """
    Return boolean indicating whether specified window periods are in list of states.

    Internal DRY method to implement previous_tour_ends and previous_tour_begins

    Parameters
    ----------
    windows : array of int8, 2 dimensions
        Array of currently scheduled stuff
    windows_row_mapper : numba.typed.Dict[int,int]
        Maps value in the `window_row_ids` to row positions in `windows`.
    windows_col_mapper : numba.typed.Dict[int,int]
        Array of currently scheduled stuff
    window_row_id : int
        An identifier for which window row to use.
    period : int
        An identifier for which window col to use.
    state1, state2 : int
        presumably (e.g. I_EMPTY, I_START...)

    Returns
    -------
    bool
    """
    w = windows[windows_row_mapper[window_row_id], windows_col_mapper[period]]
    if w == state1 or w == state2:
        return True
    return False


@nb.njit
def _windows_periods_in_states(
    windows,
    windows_row_mapper,
    windows_col_mapper,
    window_row_ids,
    periods,
    state1,
    state2,
):
    result = np.empty(window_row_ids.shape, dtype=np.int8)
    for i in range(window_row_ids.shape[0]):
        result[i] = _window_period_in_states(
            windows,
            windows_row_mapper,
            windows_col_mapper,
            window_row_ids[i],
            periods[i],
            state1,
            state2,
        )
    return result


def tt_previous_tour_ends(tt, window_row_ids, periods):
    return _windows_periods_in_states(
        tt.windows,
        tt.window_row_ix._mapper,
        tt.time_ix._mapper,
        window_row_ids.values,
        periods.values,
        I_END,
        I_START_END,
    )


@nb.njit
def sharrow_tt_previous_tour_ends(
    tt_windows, tt_row_mapper, tt_col_mapper, window_row_id, period
):
    return _window_period_in_states(
        tt_windows,
        tt_row_mapper,
        tt_col_mapper,
        window_row_id,
        period,
        I_END,
        I_START_END,
    )


def tt_previous_tour_begins(tt, window_row_ids, periods):
    return _windows_periods_in_states(
        tt.windows,
        tt.window_row_ix._mapper,
        tt.time_ix._mapper,
        window_row_ids.values,
        periods.values,
        I_START,
        I_START_END,
    )


@nb.njit
def sharrow_tt_previous_tour_begins(
    tt_windows, tt_row_mapper, tt_col_mapper, window_row_id, period
):
    return _window_period_in_states(
        tt_windows,
        tt_row_mapper,
        tt_col_mapper,
        window_row_id,
        period,
        I_START,
        I_START_END,
    )


def tt_adjacent_window_before(tt, window_row_ids, periods):
    return _available_run_length_2(
        tt.windows,
        tt.window_row_ix._mapper,
        tt.time_ix._mapper,
        True,
        window_row_ids.values,
        periods.to_numpy(),
    )


@nb.njit
def sharrow_tt_adjacent_window_before(
    tt_windows, tt_row_mapper, tt_col_mapper, window_row_id, period
):
    return _available_run_length_1(
        tt_windows,
        tt_row_mapper,
        tt_col_mapper,
        True,
        window_row_id,
        period,
    )


def tt_adjacent_window_after(tt, window_row_ids, periods):
    return _available_run_length_2(
        tt.windows,
        tt.window_row_ix._mapper,
        tt.time_ix._mapper,
        False,
        window_row_ids.values,
        periods.to_numpy(),
    )


@nb.njit
def sharrow_tt_adjacent_window_after(
    tt_windows, tt_row_mapper, tt_col_mapper, window_row_id, period
):
    return _available_run_length_1(
        tt_windows,
        tt_row_mapper,
        tt_col_mapper,
        False,
        window_row_id,
        period,
    )

# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import timetable as tt

logger = logging.getLogger(__name__)


def get_previous_tour_by_tourid(current_tour_window_ids,
                                previous_tour_by_window_id,
                                alts):
    """
    Matches current tours with attributes of previous tours for the same
    person.  See the return value below for more information.

    Parameters
    ----------
    current_tour_window_ids : Series
        A Series of parent ids for the tours we're about make the choice for
        - index should match the tours DataFrame.
    previous_tour_by_window_id : Series
        A Series where the index is the parent (window) id and the value is the index
        of the alternatives of the scheduling.
    alts : DataFrame
        The alternatives of the scheduling.

    Returns
    -------
    prev_alts : DataFrame
        A DataFrame with an index matching the CURRENT tours we're making a
        decision for, but with columns from the PREVIOUS tour of the person
        associated with each of the CURRENT tours.  Columns listed in PREV_TOUR_COLUMNS
        from the alternatives will have "_previous" added as a suffix to keep
        differentiated from the current alternatives that will be part of the
        interaction.
    """

    PREV_TOUR_COLUMNS = ['start', 'end']

    previous_tour_by_tourid = \
        previous_tour_by_window_id.loc[current_tour_window_ids]

    previous_tour_by_tourid = alts.loc[previous_tour_by_tourid, PREV_TOUR_COLUMNS]

    previous_tour_by_tourid.index = current_tour_window_ids.index
    previous_tour_by_tourid.columns = [x+'_previous' for x in PREV_TOUR_COLUMNS]

    return previous_tour_by_tourid


def tdd_interaction_dataset(tours, alts, timetable, choice_column, window_id_col):
    """
    interaction_sample_simulate expects
    alts index same as choosers (e.g. tour_id)
    name of choice column in alts

    Parameters
    ----------
    tours : pandas DataFrame
        must have person_id column and index on tour_id
    alts : pandas DataFrame
        alts index must be timetable tdd id
    timetable : TimeTable object
    choice_column : str
        name of column to store alt index in alt_tdd DataFrame
        (since alt_tdd is duplicate index on person_id but unique on person_id,alt_id)

    Returns
    -------
    alt_tdd : pandas DataFrame
        columns: start, end , duration, <choice_column>
        index: tour_id


    """

    alts_ids = np.tile(alts.index, len(tours.index))
    tour_ids = np.repeat(tours.index, len(alts.index))
    window_row_id_ids = np.repeat(tours[window_id_col], len(alts.index))

    alt_tdd = alts.take(alts_ids).copy()
    alt_tdd.index = tour_ids
    alt_tdd[window_id_col] = window_row_id_ids
    alt_tdd[choice_column] = alts_ids

    # slice out all non-available tours
    available = timetable.tour_available(alt_tdd[window_id_col], alt_tdd[choice_column])

    assert available.any()

    alt_tdd = alt_tdd[available]

    # FIXME - don't need this any more after slicing
    del alt_tdd[window_id_col]

    return alt_tdd


def schedule_tours(tours, persons_merged,
                   alts, spec, constants,
                   timetable,
                   previous_tour, window_id_col,
                   chunk_size, tour_trace_label):
    """
    previous_tour stores values used to add columns that can be used in the spec
    which have to do with the previous tours per person.  Every column in the
    alternatives table is appended with the suffix "_previous" and made
    available.  So if your alternatives table has columns for start and end,
    then start_previous and end_previous will be set to the start and end of
    the most recent tour for a person.  The first time through,
    start_previous and end_previous are undefined, so make sure to protect
    with a tour_num >= 2 in the variable computation.

    Parameters
    ----------
    tours : DataFrame
        chunk of tours to schedule with unique window_id_col (person_id or parent_tour_id)
    persons_merged : DataFrame
        DataFrame of persons to be merged with tours containing attributes referenced
        by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent all possible time slots.
        tdd_interaction_dataset function will use timetable to filter them to omit
        unavailable alternatives
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
    constants : dict
        dict of model-specific constants for eval
    timetable : TimeTable
        timetable of timewidows for person (or subtour) with rows for tours[window_id_col]
    previous_tour: Series
        series with value of tdd_alt choice for last previous tour scheduled for
    window_id_col : str
        column name from tours that identifies 'owner' of this tour
        (person_id for non/mandatory tours or parent_tout_id for subtours)
    chunk_size
    tour_trace_label

    Returns
    -------

    """

    logger.info("%s schedule_tours running %d tour choices" % (tour_trace_label, len(tours)))

    # timetable can't handle multiple tours per person
    assert len(tours.index) == len(np.unique(tours.person_id.values))

    # merge persons into tours
    tours = pd.merge(tours, persons_merged, left_on='person_id', right_index=True)

    # merge previous tour columns
    tours = tours.join(
        get_previous_tour_by_tourid(tours[window_id_col], previous_tour, alts)
    )

    # build interaction dataset filtered to include only available tdd alts
    # dataframe columns start, end , duration, person_id, tdd
    # indexed (not unique) on tour_id
    choice_column = 'tdd'
    alt_tdd = tdd_interaction_dataset(tours, alts, timetable, choice_column, window_id_col)

    locals_d = {
        'tt': timetable
    }
    if constants is not None:
        locals_d.update(constants)

    choices = interaction_sample_simulate(
        tours,
        alt_tdd,
        spec,
        choice_column=choice_column,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=tour_trace_label
    )

    previous_tour.loc[tours[window_id_col]] = choices.values

    timetable.assign(tours[window_id_col], choices)

    return choices


def vectorize_tour_scheduling(tours, persons_merged, alts, spec,
                              constants={},
                              chunk_size=0, trace_label=None):
    """
    The purpose of this method is fairly straightforward - it takes tours
    and schedules them into time slots.  Alternatives should be specified so
    as to define those time slots (usually with start and end times).

    schedule_tours adds variables that can be used in the spec which have
    to do with the previous tours per person.  Every column in the
    alternatives table is appended with the suffix "_previous" and made
    available.  So if your alternatives table has columns for start and end,
    then start_previous and end_previous will be set to the start and end of
    the most recent tour for a person.  The first time through,
    start_previous and end_previous are undefined, so make sure to protect
    with a tour_num >= 2 in the variable computation.

    Parameters
    ----------
    tours : DataFrame
        DataFrame of tours containing tour attributes, as well as a person_id
        column to define the nth tour for each person.
    persons_merged : DataFrame
        DataFrame of persons containing attributes referenced by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent time slots.  Will be passed to
        interaction_simulate in batches for each nth tour.
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
        (or dict of specs keyed on tour_type if tour_types is not None)
    constants : dict
        dict of model-specific constants for eval
    Returns
    -------
    choices : Series
        A Series of choices where the index is the index of the tours
        DataFrame and the values are the index of the alts DataFrame.
    """

    if not trace_label:
        trace_label = 'vectorize_tour_scheduling'

    assert len(tours.index) > 0
    assert 'tour_num' in tours.columns
    assert 'tour_type' in tours.columns

    timetable = inject.get_injectable("timetable")
    choice_list = []

    # keep a series of the the most recent tours for each person
    # initialize with first trip from alts
    previous_tour_by_personid = pd.Series(alts.index[0], index=tours.person_id.unique())

    # no more than one tour per person per call to schedule_tours
    # tours must be scheduled in increasing trip_num order
    # second trip of type must be in group immediately following first
    # segregate scheduling by tour_type if multiple specs passed in dict keyed by tour_type

    for tour_num, nth_tours in tours.groupby('tour_num', sort=True):

        tour_trace_label = tracing.extend_trace_label(trace_label, 'tour_%s' % (tour_num,))

        if isinstance(spec, dict):

            for tour_type in spec:

                tour_trace_label = tracing.extend_trace_label(trace_label, tour_type)

                choices = \
                    schedule_tours(nth_tours[nth_tours.tour_type == tour_type],
                                   persons_merged, alts,
                                   spec[tour_type],
                                   constants, timetable,
                                   previous_tour_by_personid, 'person_id',
                                   chunk_size, tour_trace_label)

                choice_list.append(choices)

        else:

            choices = \
                schedule_tours(nth_tours,
                               persons_merged, alts,
                               spec,
                               constants, timetable,
                               previous_tour_by_personid, 'person_id',
                               chunk_size, tour_trace_label)

            choice_list.append(choices)

    choices = pd.concat(choice_list)

    # add the start, end, and duration from tdd_alts
    tdd = alts.loc[choices]
    tdd.index = choices.index
    # include the index of the choice in the tdd alts table
    tdd['tdd'] = choices

    timetable.replace_table()

    return tdd


def vectorize_subtour_scheduling(parent_tours, subtours, persons_merged, alts, spec,
                                 constants, chunk_size=0, trace_label=None):
    """
    Like vectorize_tour_scheduling but specifically for atwork subtours

    subtours have a few peculiarities necessitating separate treatment:

    Timetable has to be initialized to set all timeperiods outside parent tour footprint as
    unavailable. So atwork subtour timewindows are limited to the foorprint of the parent work
    tour. And parent_tour_id' column of tours is used instead of parent_id as timetable row_id.

    Parameters
    ----------
    parent_tours : DataFrame
        parent tours of the subtours (because we need to know the tdd of the parent tour to
        assign_subtour_mask of timetable indexed by parent_tour id
    subtours : DataFrame
        atwork subtours to schedule
    persons_merged : DataFrame
        DataFrame of persons containing attributes referenced by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent time slots.  Will be passed to
        interaction_simulate in batches for each nth tour.
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
        (all subtours share same spec regardless of subtour type)
    constants : dict
        dict of model-specific constants for eval    chunk_size
    trace_label

    Returns
    -------
    choices : Series
        A Series of choices where the index is the index of the subtours
        DataFrame and the values are the index of the alts DataFrame.
    """
    if not trace_label:
        trace_label = 'vectorize_non_mandatory_tour_scheduling'

    assert len(subtours.index) > 0
    assert 'tour_num' in subtours.columns
    assert 'tour_type' in subtours.columns

    # timetable with a window for each parent tour
    parent_tour_windows = tt.create_timetable_windows(parent_tours, alts)
    timetable = tt.TimeTable(parent_tour_windows, alts)

    # mask the periods outside parent tour footprint
    timetable.assign_subtour_mask(parent_tours.tour_id, parent_tours.tdd)

    print timetable.windows
    """
    [[7 7 7 0 0 0 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7]
     [7 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
     [7 7 7 7 7 0 0 0 0 0 0 0 0 0 0 7 7 7 7 7 7]
     [7 7 0 0 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7 7]]
    """

    choice_list = []

    # keep a series of the the most recent tours for each person
    # initialize with first trip from alts
    previous_tour_by_parent_tour_id = \
        pd.Series(alts.index[0], index=subtours['parent_tour_id'].unique())

    # no more than one tour per person per call to schedule_tours
    # tours must be scheduled in increasing trip_num order
    # second trip of type must be in group immediately following first
    # segregate scheduling by tour_type if multiple specs passed in dict keyed by tour_type

    for tour_num, nth_tours in subtours.groupby('tour_num', sort=True):

        tour_trace_label = tracing.extend_trace_label(trace_label, 'tour_%s' % (tour_num,))

        choices = \
            schedule_tours(nth_tours,
                           persons_merged, alts,
                           spec,
                           constants, timetable,
                           previous_tour_by_parent_tour_id, 'parent_tour_id',
                           chunk_size, tour_trace_label)

        choice_list.append(choices)

    choices = pd.concat(choice_list)

    # add the start, end, and duration from tdd_alts
    tdd = alts.loc[choices]
    tdd.index = choices.index
    # include the index of the choice in the tdd alts table
    tdd['tdd'] = choices

    # print "\nfinal timetable.windows\n", timetable.windows
    """
    [[7 7 7 0 0 0 0 2 7 7 4 7 7 7 7 7 7 7 7 7 7]
     [7 0 2 7 4 0 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
     [7 7 7 7 7 2 4 0 0 0 0 0 0 0 0 7 7 7 7 7 7]
     [7 7 0 2 7 7 4 0 0 7 7 7 7 7 7 7 7 7 7 7 7]]
    """

    return tdd

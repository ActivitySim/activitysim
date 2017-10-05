# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import logit

logger = logging.getLogger(__name__)


def get_previous_tour_by_tourid(current_tour_person_ids,
                                previous_tour_by_personid,
                                alts):
    """
    Matches current tours with attributes of previous tours for the same
    person.  See the return value below for more information.

    Parameters
    ----------
    current_tour_person_ids : Series
        A Series of person ids for the tours we're about make the choice for
        - index should match the tours DataFrame.
    previous_tour_by_personid : Series
        A Series where the index is the person id and the value is the index
        of the alternatives of the scheduling.
    alts : DataFrame
        The alternatives of the scheduling.

    Returns
    -------
    prev_alts : DataFrame
        A DataFrame with an index matching the CURRENT tours we're making a
        decision for, but with columns from the PREVIOUS tour of the person
        associated with each of the CURRENT tours.  Every column of the
        alternatives will have "_previous" added as a suffix to keep
        differentiated from the current alternatives that will be part of the
        interaction.
    """
    previous_tour_by_tourid = \
        previous_tour_by_personid.loc[current_tour_person_ids]

    previous_tour_by_tourid = alts.loc[previous_tour_by_tourid]

    previous_tour_by_tourid.index = current_tour_person_ids.index
    previous_tour_by_tourid.columns = [x+'_previous' for x in
                                       previous_tour_by_tourid.columns]

    return previous_tour_by_tourid


def tdd_interaction_dataset(tours, alts, timetable, choice_column):
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
    person_ids = np.repeat(tours['person_id'], len(alts.index))

    alt_tdd = alts.take(alts_ids).copy()
    alt_tdd.index = tour_ids
    alt_tdd['person_id'] = person_ids
    alt_tdd[choice_column] = alts_ids

    # slice out all non-available tours
    available = timetable.tour_available(alt_tdd.person_id, alt_tdd[choice_column])

    assert available.any()

    alt_tdd = alt_tdd[available]

    # FIXME - don't need this any more after slicing
    del alt_tdd['person_id']

    return alt_tdd


def vectorize_tour_scheduling(tours, persons_merged,
                              alts, spec, constants={},
                              chunk_size=0, trace_label=None):
    """
    The purpose of this method is fairly straightforward - it takes tours
    and schedules them into time slots.  Alternatives should be specified so
    as to define those time slots (usually with start and end times).

    The difficulty of doing this in Python is that subsequent tours are
    dependent on certain characteristics of previous tours for the same
    person.  This is a problem with Python's vectorization requirement,
    so this method does all the 1st tours, then all the 2nd tours, and so forth.

    This method also adds variables that can be used in the spec which have
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

    Returns
    -------
    choices : Series
        A Series of choices where the index is the index of the tours
        DataFrame and the values are the index of the alts DataFrame.
    """

    assert len(tours.index) > 0
    timetable = inject.get_injectable("timetable")

    # because this is Python, we have to vectorize everything by doing the
    # "nth" trip for each person in a for loop (in other words, because each
    # trip is dependent on the time windows left by the previous decision) -
    # hopefully this will work out ok!

    choices = []
    tour_trace_label = None

    # keep a series of the the most recent tours for each person
    previous_tour_by_personid = pd.Series(
        pd.Series(alts.index).iloc[0], index=tours.person_id.unique())

    for tour_num, nth_tours in tours.groupby('tour_num'):

        if trace_label:
            tour_trace_label = tracing.extend_trace_label(trace_label, 'tour_%s' % tour_num)
            logger.info("%s running %d #%d tour choices" % (trace_label, len(nth_tours), tour_num))

        nth_tours = pd.merge(nth_tours, persons_merged, left_on='person_id', right_index=True)

        nth_tours = nth_tours.join(
            get_previous_tour_by_tourid(nth_tours.person_id, previous_tour_by_personid, alts)
        )

        choice_column = 'tdd'
        alt_tdd = tdd_interaction_dataset(nth_tours, alts, timetable, choice_column=choice_column)

        """
        alt_tdd : pandas DataFrame
            columns: start, end , duration, person_id, tdd
            index: tour_id
        """

        nth_choices = interaction_sample_simulate(
            nth_tours,
            alt_tdd,
            spec,
            choice_column=choice_column,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=tour_trace_label
        )

        previous_tour_by_personid.loc[nth_tours.person_id] = nth_choices.values

        timetable.assign(nth_tours.person_id, nth_choices)

        choices.append(nth_choices)

    choices = pd.concat(choices)

    # add the start, end, and duration from tdd_alts
    tdd = alts.loc[choices]
    tdd.index = choices.index
    # include the index of the choice in the tdd alts table
    tdd['tdd'] = choices

    timetable.replace_table()

    return tdd

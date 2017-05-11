# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing

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


def vectorize_tour_scheduling(tours, alts, spec, constants={},
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

    max_num_trips = tours.groupby('person_id').size().max()

    if np.isnan(max_num_trips):
        s = pd.Series()
        s.index.name = 'tour_id'
        return s

    # because this is Python, we have to vectorize everything by doing the
    # "nth" trip for each person in a for loop (in other words, because each
    # trip is dependent on the time windows left by the previous decision) -
    # hopefully this will work out ok!

    choices = []

    # keep a series of the the most recent tours for each person
    previous_tour_by_personid = pd.Series(
        pd.Series(alts.index).iloc[0], index=tours.person_id.unique())

    for i in range(max_num_trips):

        # this reset_index / set_index stuff keeps the index as the tours
        # index rather that switching to person_id as the index which is
        # what happens when you groupby person_id
        index_name = tours.index.name or 'index'
        nth_tours = tours.reset_index().\
            groupby('person_id').nth(i).reset_index().set_index(index_name)

        nth_tours.index.name = 'tour_id'

        if trace_label:
            logger.info("%s running %d #%d tour choices" % (trace_label, len(nth_tours), i+1))

        # tour num can be set by the user, but if it isn't we set it here
        if "tour_num" not in nth_tours:
            nth_tours["tour_num"] = i+1

        nth_tours = nth_tours.join(get_previous_tour_by_tourid(
            nth_tours.person_id,
            previous_tour_by_personid,
            alts))

        tour_trace_label = tracing.extend_trace_label(trace_label, 'tour_%s' % i)

        nth_choices = asim.interaction_simulate(
            nth_tours,
            alts.copy(),
            spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=tour_trace_label
        )

        choices.append(nth_choices)

        previous_tour_by_personid.loc[nth_tours.person_id] = nth_choices.values

    choices = pd.concat(choices)

    # return the concatenated choices
    return choices

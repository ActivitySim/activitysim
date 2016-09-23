# ActivitySim
# See full license in LICENSE.txt.

import logging
import itertools

import numpy as np
import pandas as pd
from zbox import toolz as tz, gen


from ..activitysim import eval_variables
from .. import nl

from .. import tracing

logger = logging.getLogger(__name__)


def _run_cdap(
        people, hh_id_col, p_type_col,
        locals_d,
        trace_hh_id, trace_label):

    """
    implements core run_cdap functionality but without chunking of people df
    """

    # from documentation of cdapPersonArray in HouseholdCoordinatedDailyActivityPatternModel.java
    # Method reorders the persons in the household for use with the CDAP model,
    # which only explicitly models the interaction of five persons in a HH. Priority
    # in the reordering is first given to full time workers (up to two), then to
    # part time workers (up to two workers, of any type), then to children (youngest
    # to oldest, up to three). If the method is called for a household with less
    # than 5 people, the cdapPersonArray is the same as the person array.

    WORKER_PTYPES = [1, 2]
    CHILD_PTYPES = [6, 7, 8]

    CDAP_WORKER = 1
    CDAP_CHILD = 2
    CDAP_BACKFILL = 3
    CDAP_UNASSIGNED = 9
    people['cdap_person'] = CDAP_UNASSIGNED

    # choose up to 2 workers, preferring full over part, older over younger
    workers = people.loc[people[p_type_col].isin(WORKER_PTYPES),
                         ['household_id', 'ptype', 'age']]\
        .sort_values(by=['household_id', 'ptype', 'age'], ascending=[True, True, False])\
        .groupby(hh_id_col).head(2)
    # tag the selected workers
    people.loc[workers.index, 'cdap_person'] = CDAP_WORKER

    # choose up to 3, preferring youngest
    children = people.loc[people[p_type_col].isin(CHILD_PTYPES),
                          ['household_id', 'ptype', 'age']]\
        .sort_values(by=['household_id', 'age'], ascending=[True, True])\
        .groupby(hh_id_col).head(3)
    # tag the selected children
    people.loc[children.index, 'cdap_person'] = CDAP_CHILD

    # choose up to 5, preferring anyone already chosen
    others = people[['household_id', 'cdap_person']]\
        .sort_values(by=['household_id', 'cdap_person'], ascending=[True, True])\
        .groupby(hh_id_col).head(5)
    # tag the backfilled persons
    people.loc[others[others.cdap_person == CDAP_UNASSIGNED].index, 'cdap_person'] = CDAP_BACKFILL

    tracing.trace_df(people[['household_id', 'PERSONS', 'ptype', 'age', 'cdap_person']],
                     '%s.people' % trace_label,
                     transpose=False,
                     slicer='NONE')

    person_choices = pd.Series(['Home'] * len(people), index=people.index)

    if trace_hh_id:
        tracing.trace_df(person_choices, '%s.person_choices' % trace_label,
                         columns='choice')

    return person_choices


def hh_chunked_choosers(choosers):
    # generator to iterate over chooses in chunk_size chunks
    last_chooser = choosers['chunk_id'].max()
    i = 0
    while i <= last_chooser:
        yield i, choosers[choosers['chunk_id'] == i]
        i += 1


def run_cdap(
        people, hh_id_col, p_type_col,
        locals_d,
        chunk_size=0, trace_hh_id=None, trace_label=None):
    """
    Choose individual activity patterns for people.

    Parameters
    ----------
    people : pandas.DataFrame
        Table of people data. Must contain at least a household ID
        column and a categorization of person type.
    hh_id_col : str
        Name of the column in `people` that has their household ID.
    p_type_col : str
        Name of the column in `people` that contains the person type number.

    chunk_size: int
        chunk size or 0 for no chunking
    trace_hh_id : int
        hh_id to trace or None if no hh tracing
    trace_label : str
        label for tracing or None if no tracing

    Returns
    -------
    choices : pandas.Series
        Maps index of `people` to their activity pattern choice,
        where that choice is taken from the columns of specs
        (so it's important that the specs all refer to alternatives
        in the same way).

    """

    trace_label = (trace_label or 'cdap')

    if (chunk_size == 0) or (chunk_size >= len(people.index)):
        choices = _run_cdap(people, hh_id_col, p_type_col,
                            locals_d,
                            trace_hh_id, trace_label)
        return choices

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, people_chunk in hh_chunked_choosers(people):

        logger.info("run_cdap running hh_chunk =%s of size %d" % (i, len(people_chunk)))

        chunk_trace_label = "%s.chunk_%s" % (trace_label, i)

        choices = _run_cdap(people_chunk, hh_id_col, p_type_col,
                            locals_d,
                            trace_hh_id, chunk_trace_label)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    return choices

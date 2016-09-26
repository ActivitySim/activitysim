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

_hh_id_ = 'household_id'
_ptype_ = 'ptype'
_age_ = 'age'
_cdap_rank_ = 'cdap_rank'

WORKER_PTYPES = [1, 2]
CHILD_PTYPES = [6, 7, 8]

RANK_WORKER = 1
RANK_CHILD = 2
RANK_BACKFILL = 3
RANK_UNASSIGNED = 9


def assign_cdap_person_array(people, trace_hh_id=None, trace_label=None):
    # from documentation of cdapPersonArray in HouseholdCoordinatedDailyActivityPatternModel.java
    # Method reorders the persons in the household for use with the CDAP model,
    # which only explicitly models the interaction of five persons in a HH. Priority
    # in the reordering is first given to full time workers (up to two), then to
    # part time workers (up to two workers, of any type), then to children (youngest
    # to oldest, up to three). If the method is called for a household with less
    # than 5 people, the cdapPersonArray is the same as the person array.

    if _cdap_rank_ in people.columns:
        raise RuntimeError("assign_cdap_person_array: '%s' already in people columns" % _cdap_rank_)
    people[_cdap_rank_] = RANK_UNASSIGNED

    # choose up to 2 workers, preferring full over part, older over younger
    workers = people.loc[people[_ptype_].isin(WORKER_PTYPES),
                         [_hh_id_, _ptype_]]\
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])\
        .groupby(_hh_id_).head(2)
    # tag the selected workers
    people.loc[workers.index, _cdap_rank_] = RANK_WORKER
    del workers

    # choose up to 3, preferring youngest
    children = people.loc[people[_ptype_].isin(CHILD_PTYPES),
                          [_hh_id_, _ptype_, _age_]]\
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])\
        .groupby(_hh_id_).head(3)
    # tag the selected children
    people.loc[children.index, _cdap_rank_] = RANK_CHILD
    del children

    # choose up to 5, preferring anyone already chosen
    others = people[[_hh_id_, _cdap_rank_]]\
        .sort_values(by=[_hh_id_, _cdap_rank_], ascending=[True, True])\
        .groupby(_hh_id_).head(5)
    # tag the backfilled persons
    people.loc[others[others.cdap_rank == RANK_UNASSIGNED].index, _cdap_rank_] \
        = RANK_BACKFILL
    del others

    # assign person number in cdapPersonArray preference order
    # i.e. convert cdap_rank from category to index in order of category rank within household
    people[_cdap_rank_] = people\
        .sort_values(by=[_hh_id_, _cdap_rank_, _age_], ascending=[True, True, True])\
        .groupby(_hh_id_)[_hh_id_]\
        .rank(method='first', na_option='top')\
        .astype(int)

    # FIXME - possible workaround if above too big/slow
    # stackoverflow.com/questions/26720916/faster-way-to-rank-rows-in-subgroups-in-pandas-dataframe
    # Working with a big DataFrame (13 million lines), the method rank with groupby
    # maxed out my 8GB of RAM an it took a really long time. I found a workaround
    # less greedy in memory , that I put here just in case:
    # df.sort_values('value')
    # tmp = df.groupby('group').size()
    # rank = tmp.map(range)
    # rank =[item for sublist in rank for item in sublist]
    # df['rank'] = rank

    # tracing.trace_df(people[[_hh_id_, _ptype_, _age_, _cdap_rank_]],
    #                  '%s.cdap_person_array' % trace_label,
    #                  transpose=False,
    #                  slicer='NONE')

    if trace_hh_id:
        tracing.trace_df(people, '%s.cdap_rank' % trace_label,
                         columns=[_hh_id_, _ptype_, _age_, _cdap_rank_],
                         warn_if_empty=True)


def individual_utilities(
        people,
        cdap_indiv_spec,
        trace_hh_id=None, trace_label=None):
    """
    Calculate CDAP utilities for all individuals.

    Parameters
    ----------
    people : pandas.DataFrame
        DataFrame of individual people data.
    cdap_indiv_spec : pandas.DataFrame
        CDAP spec applied to individuals.

    Returns
    -------
    utilities : pandas.DataFrame
        Will have index of `people` and columns for each of the alternatives.

    """
    # calculate single person utilities
    #     evaluate variables from one_spec expressions
    #     multiply by one_spec alternative values
    indiv_vars = eval_variables(cdap_indiv_spec.index, people)
    indiv_utils = indiv_vars.dot(cdap_indiv_spec)

    if trace_hh_id:
        tracing.trace_df(indiv_vars, '%s.indiv_vals' % trace_label,
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)
        tracing.trace_df(indiv_utils, '%s.indiv_utils' % trace_label,
                         column_labels=['activity', 'person'],
                         warn_if_empty=False)

    return indiv_utils


def _run_cdap(
        people,
        cdap_indiv_spec,
        locals_d,
        trace_hh_id, trace_label):

    """
    implements core run_cdap functionality but without chunking of people df
    """

    assign_cdap_person_array(people, trace_hh_id, trace_label)

    # Calculate CDAP utilities for each individual.
    # ind_utils has index of `PERID` and a column for each alternative
    # e.g. three columns" Mandatory, NonMandatory, Home
    ind_utils = individual_utilities(people, cdap_indiv_spec, trace_hh_id, trace_label)

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
        people,
        cdap_indiv_spec,
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
        choices = _run_cdap(people,
                            cdap_indiv_spec,
                            locals_d,
                            trace_hh_id, trace_label)
        return choices

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, people_chunk in hh_chunked_choosers(people):

        logger.info("run_cdap running hh_chunk =%s of size %d" % (i, len(people_chunk)))

        chunk_trace_label = "%s.chunk_%s" % (trace_label, i)

        choices = _run_cdap(people_chunk,
                            locals_d,
                            trace_hh_id, chunk_trace_label)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    return choices

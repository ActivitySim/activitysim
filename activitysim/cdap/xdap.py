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

DUMP = False

_persons_index_ = 'PERID'
_hh_index_ = 'HHID'
_hh_size_ = 'PERSONS'

_hh_id_ = 'household_id'
_ptype_ = 'ptype'
_age_ = 'age'
_cdap_rank_ = 'cdap_rank'

MAX_HHSIZE = 4

WORKER_PTYPES = [1, 2]
CHILD_PTYPES = [6, 7, 8]

RANK_WORKER = 1
RANK_CHILD = 2
RANK_BACKFILL = 3
RANK_UNASSIGNED = 9


def set_hh_index(df):

    # index on household_id, not person_id
    df.set_index(_hh_id_, inplace=True)
    df.index.name = _hh_index_


def add_pn(col, pnum):
    """
    return the canonical column name for the indiv_util column or columns
    in merged hh_chooser df for individual with cdap_rank pnum

    e.g. M_p1, ptype_p2
    """
    if type(col) is str:
        return col if col == _hh_id_ else '%s_p%s' % (col, pnum)
    elif isinstance(col, (list, tuple)):
        return [c if c == _hh_id_ else '%s_p%s' % (c, pnum) for c in col]
    else:
        raise RuntimeError("add_pn col not list or str")


def add_interaction_column(choosers, p_tup):
    # add an interaction column in place to choosers df listing the ptypes of the persons in p_tup
    # for instance,
    # p_tup = (1,3) means persons with cdap_rank 1 and 3
    # for  p_tup = (1,3) dest column name will be 'p1_p3'
    # for a household where person 1 is part-time worker (ptype=2) and person 3 is infant (ptype 8)
    # the corresponding row value interaction code will be 28
    # We take advantage of the fact that interactions are symmetrical to simplify spec expressions:
    # We name the interaction_column in increasing pnum (cdap_rank) order (p1_p2 and not p3_p1)
    # And we format row values in increasing ptype order (28 and not 82)
    # This simplifies the spec expressions as we don't have to test for p1_p3 == 28 | p1_p3 == 82

    if p_tup != tuple(sorted(p_tup)):
        raise RuntimeError("add_interaction_column tuple not sorted" % p_tup)

    dest_col = '_'.join(['p%s' % pnum for pnum in p_tup])

    # build a string concatenating the ptypes of the persons in the order they appear in p_tup
    choosers[dest_col] = choosers[add_pn('ptype', p_tup[0])].astype(str)
    for pnum in p_tup[1:]:
        choosers[dest_col] = choosers[dest_col] \
                             + choosers[add_pn('ptype', pnum)].astype(str)

    # sort the list of ptypes so it is in increasing ptype order, then convert to int
    choosers[dest_col] = choosers[dest_col].apply(lambda x: ''.join(sorted(x))).astype(int)


def assign_cdap_person_array(persons, trace_hh_id=None, trace_label=None):
    # from documentation of cdapPersonArray in HouseholdCoordinatedDailyActivityPatternModel.java
    # Method reorders the persons in the household for use with the CDAP model,
    # which only explicitly models the interaction of five persons in a HH. Priority
    # in the reordering is first given to full time workers (up to two), then to
    # part time workers (up to two workers, of any type), then to children (youngest
    # to oldest, up to three). If the method is called for a household with less
    # than 5 people, the cdapPersonArray is the same as the person array.

    # For clarity, we use a name constant MAX_HHSIZE = 5
    # in principal other numbers should work, though this has not been tested and higher values
    # higher values might cause problems as the number of alternative patterns will be 3**MAX_HHSIZE

    if _cdap_rank_ in persons.columns:
        raise RuntimeError("assign_cdap_person_array: '%s' col already in persons df" % _cdap_rank_)
    persons[_cdap_rank_] = RANK_UNASSIGNED

    # choose up to 2 workers, preferring full over part, older over younger
    workers = persons.loc[persons[_ptype_].isin(WORKER_PTYPES), [_hh_id_, _ptype_]]\
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])\
        .groupby(_hh_id_).head(2)
    # tag the selected workers
    persons.loc[workers.index, _cdap_rank_] = RANK_WORKER
    del workers

    # choose up to 3, preferring youngest
    children = persons.loc[persons[_ptype_].isin(CHILD_PTYPES), [_hh_id_, _ptype_, _age_]]\
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])\
        .groupby(_hh_id_).head(3)
    # tag the selected children
    persons.loc[children.index, _cdap_rank_] = RANK_CHILD
    del children

    # choose up to MAX_HHSIZE, preferring anyone already chosen
    others = persons[[_hh_id_, _cdap_rank_]]\
        .sort_values(by=[_hh_id_, _cdap_rank_], ascending=[True, True])\
        .groupby(_hh_id_).head(MAX_HHSIZE)
    # tag the backfilled persons
    persons.loc[others[others.cdap_rank == RANK_UNASSIGNED].index, _cdap_rank_] \
        = RANK_BACKFILL
    del others

    # assign person number in cdapPersonArray preference order
    # i.e. convert cdap_rank from category to index in order of category rank within household
    persons[_cdap_rank_] = persons\
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

    if trace_hh_id:
        tracing.trace_df(persons, '%s.cdap_rank' % trace_label,
                         columns=[_hh_id_, _ptype_, _age_, _cdap_rank_],
                         warn_if_empty=True)


def individual_utilities(
        persons,
        cdap_indiv_spec,
        trace_hh_id=None, trace_label=None):
    """
    Calculate CDAP utilities for all individuals.

    Parameters
    ----------
    persons : pandas.DataFrame
        DataFrame of individual persons data.
    cdap_indiv_spec : pandas.DataFrame
        CDAP spec applied to individuals.

    Returns
    -------
    utilities : pandas.DataFrame
        Will have index of `persons` and columns for each of the alternatives.

    """

    # calculate single person utilities
    individual_vars = eval_variables(cdap_indiv_spec.index, persons)
    indiv_utils = individual_vars.dot(cdap_indiv_spec)

    # add columns from persons to facilitate building household interactions
    useful_columns = [_hh_id_, _ptype_, _cdap_rank_, _hh_size_]
    indiv_utils[useful_columns] = persons[useful_columns]

    if DUMP:
        tracing.trace_df(indiv_utils,
                         '%s.DUMP.indiv_utils' % trace_label,
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:
        tracing.trace_df(individual_vars, '%s.individual_vars' % trace_label,
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)
        tracing.trace_df(indiv_utils, '%s.indiv_utils' % trace_label,
                         column_labels=['activity', 'person'],
                         warn_if_empty=False)

    return indiv_utils


def hhsize1_utilities(indiv_utils, trace_hh_id, trace_label):
    """

    Parameters
    ----------
    indiv_utils : pandas.DataFrame
        individual utilities indexed by _persons_index_
    trace_hh_id
    trace_label

    Returns
    -------
    hhsize1_utils : pandas.DataFrame
        utilities for households with 1 person indexed by _hh_index_
    """

    hhsize1_utils = indiv_utils.loc[indiv_utils[_hh_size_] == 1, [_hh_id_, 'M', 'N', 'H']]

    # index on household_id, not person_id
    set_hh_index(hhsize1_utils)

    if DUMP:
        tracing.trace_df(hhsize1_utils,
                         '%s.DUMP.hhsize1_utils' % trace_label,
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:
        tracing.trace_df(hhsize1_utils, '%s.hhsize1_utils' % trace_label,
                         column_labels=['expression', 'household'],
                         warn_if_empty=False)

    return hhsize1_utils


def build_cdap_spec(cdap_interaction_coefficients, hhsize):

    expression_name = "Expression"

    alternatives = [''.join(tup) for tup in itertools.product('HMN', repeat=hhsize)]
    z = [0.0] * len(alternatives)

    spec = pd.DataFrame(columns=alternatives)
    spec.index.name = expression_name

    #   Expression   MM   MN   MH   NM   NN   NH   HM   HN   HH
    # 0       M_p1  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
    # 1       N_p1  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0
    for pnum in range(1, hhsize+1):
        for activity in ['M', 'N', 'H']:
            expression = add_pn(activity, pnum)
            spec.loc[expression] = [(1 if a[pnum - 1] == activity else '') for a in alternatives]

    # ignore rows whose cardinality exceeds hhsize
    relevant_rows = cdap_interaction_coefficients.cardinality <= hhsize

    for row in cdap_interaction_coefficients[relevant_rows].itertuples():

        if row.cardinality not in [2, 3]:
            raise RuntimeError("Bad row cardinality %d for %s" % (row.cardinality, row.slug))

        # for each set of interacting persons
        # e.g. for (1, 2), (1,3), (2,3) for a coefficient with cardinality 2 in hhsize 3
        for tup in itertools.combinations(range(1, hhsize+1), row.cardinality):

            # column named (e.g.) p1_p3 for an interaction between p1 and p3
            interaction_column = '_'.join(['p%s' % pnum for pnum in tup])

            # (e.g.) p1_p3==13 for an interaction between p1 and p3 of ptypes 1 and 3 (or 3 and1 )
            expression = "%s==%d" % (interaction_column, row.interaction_ptypes)

            # columns with names matching activity for each of the persons in tup
            # e.g. ['MMM', 'MMN', 'MMH'] for an interaction between p1 and p3 with activity 'M'
            alternative_columns = \
                filter(lambda alt: all([alt[p - 1] == row.activity for p in tup]), alternatives)

            # set the alternative_columns to the slug
            # setting with enlargement so this will add row if necessary, or update if it exists
            spec.loc[expression, alternative_columns] = row.slug

    return spec


def patch_cdap_spec_slugs(spec, cdap_interaction_coefficients):

    d = cdap_interaction_coefficients.set_index('slug')['coefficient'].to_dict()

    for c in spec.columns:
        spec[c] =\
            spec[c].map(lambda x: d.get(x, x or 0.0)).fillna(0)


def hh_choosers(indiv_utils, hhsize):

    merge_cols = [_hh_id_, _ptype_, 'M', 'N', 'H']

    if hhsize < MAX_HHSIZE:
        include_households = (indiv_utils[_hh_size_] == hhsize)
    else:
        include_households = (indiv_utils[_hh_size_] >= MAX_HHSIZE)

    choosers = indiv_utils.loc[include_households & (indiv_utils[_cdap_rank_] == 1), merge_cols]
    choosers.columns = add_pn(merge_cols, 1)  # add pn suffix

    for pnum in range(2, hhsize+1):

        rhs = indiv_utils.loc[include_households & (indiv_utils[_cdap_rank_] == pnum), merge_cols]
        rhs.columns = add_pn(merge_cols, pnum)  # add pn suffix

        choosers = pd.merge(left=choosers, right=rhs, on=_hh_id_)

    set_hh_index(choosers)

    # coerce utilities to float (merge apparently makes column type objects)
    for pnum in range(1, hhsize+1):
        pn_cols = add_pn(['M', 'N', 'H'], pnum)
        choosers[pn_cols] = choosers[pn_cols].astype(float)

    for i in range(2, hhsize+1):
        for tup in itertools.combinations(range(1, hhsize+1), i):
            add_interaction_column(choosers, tup)

    return choosers


def hh_utilities(indiv_utils, cdap_interaction_coefficients, hhsize,
                 trace_hh_id, trace_label):

    spec = build_cdap_spec(cdap_interaction_coefficients, hhsize=hhsize)

    if DUMP:
        tracing.trace_df(spec,
                         '%s.DUMP.hhsize%d_spec' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    patch_cdap_spec_slugs(spec, cdap_interaction_coefficients)

    choosers = hh_choosers(indiv_utils, hhsize=hhsize)

    vars = eval_variables(spec.index, choosers)

    utils = vars.dot(spec)

    if DUMP:

        tracing.trace_df(spec,
                         '%s.DUMP.hhsize%d_spec_patched' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(choosers,
                         '%s.DUMP.hhsize%d_choosers' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(vars,
                         '%s.DUMP.hhsize%d_vars' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(utils,
                         '%s.DUMP.hhsize%d_utils' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:

        # dump the generated spec
        tracing.trace_df(spec,
                         '%s.hhsize%d_spec' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(choosers,
                         '%s.hhsize%d_choosers' % (trace_label, hhsize),
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)
        tracing.trace_df(vars,
                         '%s.hhsize%d_vars' % (trace_label, hhsize),
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)
        tracing.trace_df(utils,
                         '%s.hhsize%d_utils' % (trace_label, hhsize),
                         column_labels=['expression', 'household'],
                         warn_if_empty=False)

    return utils


def _run_cdap(
        households,
        persons,
        cdap_indiv_spec,
        cdap_interaction_coefficients,
        locals_d,
        trace_hh_id, trace_label):

    """
    implements core run_cdap functionality but without chunking of persons df
    """

    assign_cdap_person_array(persons, trace_hh_id, trace_label)

    tracing.trace_df(persons[[_hh_id_, _ptype_, _age_, _cdap_rank_]],
                     '%s.cdap_person_array' % trace_label,
                     transpose=False,
                     slicer='NONE')

    # Calculate CDAP utilities for each individual.
    # ind_utils has index of `PERID` and a column for each alternative
    # e.g. three columns" Mandatory, NonMandatory, Home
    indiv_utils = individual_utilities(persons, cdap_indiv_spec, trace_hh_id, trace_label)

    hhsize1_utils = hhsize1_utilities(indiv_utils, trace_hh_id, trace_label)

    hhsize2_utils = hh_utilities(
        indiv_utils, cdap_interaction_coefficients, hhsize=2,
        trace_hh_id=trace_hh_id, trace_label=trace_label)

    hhsize3_utils = hh_utilities(
        indiv_utils, cdap_interaction_coefficients, hhsize=3,
        trace_hh_id=trace_hh_id, trace_label=trace_label)

    hhsize4_utils = hh_utilities(
        indiv_utils, cdap_interaction_coefficients, hhsize=4,
        trace_hh_id=trace_hh_id, trace_label=trace_label)

    hhsize5_utils = hh_utilities(
        indiv_utils, cdap_interaction_coefficients, hhsize=5,
        trace_hh_id=trace_hh_id, trace_label=trace_label)

    person_choices = pd.Series(['Home'] * len(persons), index=persons.index)

    if trace_hh_id:
        tracing.trace_df(person_choices, '%s.person_choices' % trace_label,
                         columns='choice')

    return person_choices


def chunked_hh_and_persons(households, persons):
    # generator to iterate over households and persons df in chunk_size chunks simultaneously
    last_chooser = households['chunk_id'].max()
    i = 0
    while i <= last_chooser:
        yield i, \
              persons[persons['chunk_id'] == i], \
              households[households['chunk_id'] == i]
        i += 1


def run_cdap(
        households,
        persons,
        cdap_indiv_spec,
        cdap_interaction_coefficients,
        locals_d,
        chunk_size=0, trace_hh_id=None, trace_label=None):
    """
    Choose individual activity patterns for persons.

    Parameters
    ----------
    persons : pandas.DataFrame
        Table of persons data. Must contain at least a household ID
        column and a categorization of person type.
    cdap_indiv_spec : pandas.DataFrame
        CDAP spec for individuals without taking any interactions into account.
    chunk_size: int
        chunk size or 0 for no chunking
    trace_hh_id : int
        hh_id to trace or None if no hh tracing
    trace_label : str
        label for tracing or None if no tracing

    Returns
    -------
    choices : pandas.Series
        Maps index of `persons` to their activity pattern choice,
        where that choice is taken from the columns of specs
        (so it's important that the specs all refer to alternatives
        in the same way).

    """

    trace_label = (trace_label or 'cdap')

    if (chunk_size == 0) or (chunk_size >= len(persons.index)):
        choices = _run_cdap(households,
                            persons,
                            cdap_indiv_spec,
                            cdap_interaction_coefficients,
                            locals_d,
                            trace_hh_id, trace_label)
        return choices

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, household_chunk, persons_chunk in chunked_hh_and_persons(households, persons):

        logger.info("run_cdap running hh_chunk = %s with %d households and %d persons"
                    % (i, len(household_chunk), len(persons_chunk)))

        chunk_trace_label = "%s.chunk_%s" % (trace_label, i)

        choices = _run_cdap(household_chunk,
                            persons_chunk,
                            cdap_indiv_spec,
                            cdap_interaction_coefficients,
                            locals_d,
                            trace_hh_id, chunk_trace_label)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    return choices

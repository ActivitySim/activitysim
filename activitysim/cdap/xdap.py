# ActivitySim
# See full license in LICENSE.txt.

import logging
import itertools
import time

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
_chunk_id_ = 'chunk_id'

MAX_HHSIZE = 5
MAX_INTERACTION_CARDINALITY = 3

WORKER_PTYPES = [1, 2]
CHILD_PTYPES = [6, 7, 8]


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


def assign_cdap_rank(persons, trace_hh_id=None, trace_label=None):
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

    # transient categories used to categorize persons in _cdap_rank_ before assigning final rank
    RANK_WORKER = 1
    RANK_CHILD = 2
    RANK_BACKFILL = 3
    RANK_UNASSIGNED = 9

    cdap_persons = persons[[_hh_id_, _hh_size_, _ptype_, _age_]]

    cdap_persons[_cdap_rank_] = RANK_UNASSIGNED

    # choose up to 2 workers, preferring full over part, older over younger
    workers = \
        cdap_persons.loc[cdap_persons[_ptype_].isin(WORKER_PTYPES), [_hh_id_, _ptype_]]\
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])\
        .groupby(_hh_id_).head(2)
    # tag the selected workers
    cdap_persons.loc[workers.index, _cdap_rank_] = RANK_WORKER
    del workers

    # choose up to 3, preferring youngest
    children = \
        cdap_persons.loc[cdap_persons[_ptype_].isin(CHILD_PTYPES), [_hh_id_, _ptype_, _age_]]\
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])\
        .groupby(_hh_id_).head(3)
    # tag the selected children
    cdap_persons.loc[children.index, _cdap_rank_] = RANK_CHILD
    del children

    # choose up to MAX_HHSIZE, preferring anyone already chosen
    others = \
        cdap_persons[[_hh_id_, _cdap_rank_]]\
        .sort_values(by=[_hh_id_, _cdap_rank_], ascending=[True, True])\
        .groupby(_hh_id_).head(MAX_HHSIZE)
    # tag the backfilled persons
    cdap_persons.loc[others[others.cdap_rank == RANK_UNASSIGNED].index, _cdap_rank_] \
        = RANK_BACKFILL
    del others

    # assign person number in cdapPersonArray preference order
    # i.e. convert cdap_rank from category to index in order of category rank within household
    cdap_persons[_cdap_rank_] = cdap_persons\
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

    if DUMP:
        tracing.trace_df(cdap_persons,
                         '%s.DUMP.cdap_person_array' % trace_label,
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:
        tracing.trace_df(cdap_persons, '%s.cdap_rank' % trace_label,
                         warn_if_empty=True)

    return cdap_persons[_cdap_rank_]


def individual_utilities(
        persons,
        cdap_indiv_spec,
        locals_d,
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
        plus some 'useful columns' [_hh_id_, _ptype_, _cdap_rank_, _hh_size_]

    """

    # calculate single person utilities
    individual_vars = eval_variables(cdap_indiv_spec.index, persons, locals_d)
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


def build_cdap_spec(cdap_interaction_coefficients, hhsize,
                    trace_hh_id, trace_label):

    if DUMP or trace_hh_id:
        tracing.trace_df(cdap_interaction_coefficients,
                         '%s.hhsize%d_cdap_interaction_coefficients' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    # cdap spec is same for all households of MAX_HHSIZE and greater
    hhsize = min(hhsize, MAX_HHSIZE)

    expression_name = "Expression"

    alternatives = [''.join(tup) for tup in itertools.product('HMN', repeat=hhsize)]

    spec = pd.DataFrame(columns=[expression_name] + alternatives)

    #   Expression   MM   MN   MH   NM   NN   NH   HM   HN   HH
    # 0       M_p1  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
    # 1       N_p1  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0
    for pnum in range(1, hhsize+1):
        for activity in ['M', 'N', 'H']:

            new_row_index = len(spec)
            spec.loc[new_row_index, expression_name] = add_pn(activity, pnum)

            # list of alternative columns where person pnum has expression activity
            # e.g. for M_p1 we want the columns where activity M is in position p1
            alternative_columns = filter(lambda alt: alt[pnum - 1] == activity, alternatives)
            spec.loc[new_row_index, alternative_columns] = 1

    # ignore rows whose cardinality exceeds hhsize
    relevant_rows = cdap_interaction_coefficients.cardinality <= hhsize

    for row in cdap_interaction_coefficients[relevant_rows].itertuples():

        # if it is a wildcard all_people interaction
        if not row.interaction_ptypes:

            # if the interaction includes all household members
            # then slug is the name of the alternative column (e.g. HHHH)
            # FIXME - should we be doing this for greater than HH_MAXSIZE households?
            if row.slug in alternatives:
                spec.loc[len(spec), [expression_name, row.slug]] = ['1', row.slug]

            continue

        if row.cardinality not in range(1, MAX_INTERACTION_CARDINALITY+1):
            raise RuntimeError("Bad row cardinality %d for %s" % (row.cardinality, row.slug))

        # for each set of interacting persons
        # e.g. for (1, 2), (1,3), (2,3) for a coefficient with cardinality 2 in hhsize 3
        for tup in itertools.combinations(range(1, hhsize+1), row.cardinality):

            if row.cardinality == 1:
                interaction_column = "ptype_p%d" % tup[0]
            else:
                # column named (e.g.) p1_p3 for an interaction between p1 and p3
                interaction_column = '_'.join(['p%s' % pnum for pnum in tup])

            # (e.g.) p1_p3==13 for an interaction between p1 and p3 of ptypes 1 and 3 (or 3 and1 )
            expression = "%s==%s" % (interaction_column, row.interaction_ptypes)

            # columns with names matching activity for each of the persons in tup
            # e.g. ['MMM', 'MMN', 'MMH'] for an interaction between p1 and p3 with activity 'M'
            alternative_columns = \
                filter(lambda alt: all([alt[p - 1] == row.activity for p in tup]), alternatives)

            existing_row_index = (spec[expression_name] == expression)
            if (existing_row_index).any():
                spec.loc[existing_row_index, alternative_columns] = row.slug
                spec.loc[existing_row_index, expression_name] = expression
            else:
                new_row_index = len(spec)
                spec.loc[new_row_index, alternative_columns] = row.slug
                spec.loc[new_row_index, expression_name] = expression

    spec.set_index(expression_name, inplace=True)

    if DUMP or trace_hh_id:
        tracing.trace_df(spec,
                         '%s.hhsize%d_spec' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    # replace slug with coefficient
    d = cdap_interaction_coefficients.set_index('slug')['coefficient'].to_dict()
    for c in spec.columns:
        spec[c] =\
            spec[c].map(lambda x: d.get(x, x or 0.0)).fillna(0)

    if DUMP or trace_hh_id:
        tracing.trace_df(spec,
                         '%s.hhsize%d_spec_patched' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    return spec


def hh_choosers(indiv_utils, hhsize):

    merge_cols = [_hh_id_, _ptype_, 'M', 'N', 'H']

    if hhsize > MAX_HHSIZE:
        raise RuntimeError("hh_choosers hhsize > MAX_HHSIZE")

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

    for i in range(2, min(hhsize, MAX_INTERACTION_CARDINALITY)+1):
        for tup in itertools.combinations(range(1, hhsize+1), i):
            add_interaction_column(choosers, tup)

    return choosers


def hh_utilities(indiv_utils, cdap_interaction_coefficients, hhsize,
                 trace_hh_id, trace_label):

    choosers = hh_choosers(indiv_utils, hhsize=hhsize)

    # only trace if trace_hh_id is in this hhsize (build_cdap_spec can't tell if that is the case)
    if trace_hh_id not in choosers.index:
        trace_hh_id = None

    spec = build_cdap_spec(cdap_interaction_coefficients, hhsize,
                           trace_hh_id, trace_label)

    vars = eval_variables(spec.index, choosers)

    utils = vars.dot(spec)

    if DUMP:

        tracing.trace_df(choosers,
                         '%s.DUMP.hhsize%d_choosers' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(vars,
                         '%s.DUMP.hhsize%d_vars' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:

        tracing.trace_df(choosers,
                         '%s.hhsize%d_choosers' % (trace_label, hhsize),
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)
        tracing.trace_df(vars,
                         '%s.hhsize%d_vars' % (trace_label, hhsize),
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)

    return utils


def household_activity_choices(indiv_utils, cdap_interaction_coefficients, hhsize,
                               trace_hh_id, trace_label):
    """

    Parameters
    ----------
    indiv_utils
    cdap_interaction_coefficients
    hhsize

    Returns
    -------
    choices : pandas.Series
        the chosen cdap activity pattern for each household represented as a string (e.g. 'MNH')
        with same index (_hh_index_) as utils

    """

    # calculate household utilities for each activity pattern alternative (e.g. 'MMM', 'MNH', ...)
    # utils will be a dataframe indexed by _hh_index_

    if hhsize == 1:
        # extract the individual utilities for individuals from hhsize 1 households
        utils = indiv_utils.loc[indiv_utils[_hh_size_] == 1, [_hh_id_, 'M', 'N', 'H']]
        # index on household_id, not person_id
        set_hh_index(utils)
    else:
        utils = hh_utilities(
            indiv_utils, cdap_interaction_coefficients, hhsize=hhsize,
            trace_hh_id=trace_hh_id, trace_label=trace_label)

    probs = nl.utils_to_probs(utils, trace_label=trace_label)

    # select an activity pattern alternative for each household based on probability
    # result is a series indexed on _hh_index_ with the (0 based) index of the column from probs
    idx_choices = nl.make_choices(probs, trace_label=trace_label)

    # convert choice expressed as index into alternative name from util column label
    choices = pd.Series(utils.columns[idx_choices].values, index=utils.index)

    if DUMP:
        tracing.trace_df(utils,
                         '%s.DUMP.hhsize%d_utils' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(probs,
                         '%s.DUMP.hhsize%d_probs' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(choices,
                         '%s.DUMP.hhsize%d_activity_choices' % (trace_label, hhsize),
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:
        tracing.trace_df(utils, '%s.hhsize%d_utils' % (trace_label, hhsize),
                         column_labels=['expression', 'household'],
                         warn_if_empty=False)

        tracing.trace_df(probs, '%s.hhsize%d_probs' % (trace_label, hhsize),
                         column_labels=['expression', 'household'],
                         warn_if_empty=False)

        tracing.trace_df(choices, '%s.hhsize%d_activity_choices' % (trace_label, hhsize),
                         column_labels=['expression', 'household'],
                         warn_if_empty=False)

    return choices


def unpack_cdap_indiv_activity_choices(indiv_utils, hh_choices,
                                       trace_hh_id, trace_label):
    # return a list of alternatives chosen indexed by _person_idx_

    # indiv_utils has index of `persons` and columns for each of the alternatives.
    # plus some 'useful columns' [_hh_id_, _ptype_, _cdap_rank_, _hh_size_]

    cdap_indivs = indiv_utils[_cdap_rank_] <= MAX_HHSIZE

    indiv_activity = pd.merge(
        left=indiv_utils.loc[cdap_indivs, [_hh_id_, _cdap_rank_]],
        right=hh_choices.to_frame(name='hh_choices'),
        left_on=_hh_id_,
        right_index=True
    )

    # resulting dataframe has columns _hh_id_,_cdap_rank_, hh_choices indexed on _persons_index_

    indiv_activity["cdap_activity"] = ''

    # for each cdap_rank (1..5)
    for i in range(MAX_HHSIZE):
        pnum_i = (indiv_activity[_cdap_rank_] == i+1)
        indiv_activity.loc[pnum_i, ["cdap_activity"]] = indiv_activity[pnum_i]['hh_choices'].str[i]

    cdap_indiv_activity_choices = indiv_activity['cdap_activity']

    if DUMP:
        tracing.trace_df(cdap_indiv_activity_choices,
                         '%s.DUMP.cdap_indiv_activity_choices' % trace_label,
                         transpose=False,
                         slicer='NONE')

    return cdap_indiv_activity_choices


def extra_hh_member_choices(indiv_utils, cdap_fixed_relative_proportions, locals_d,
                            trace_hh_id, trace_label):

    # return a list of alternatives chosen indexed by _person_idx_

    # indiv_utils has index of `persons` and columns for each of the alternatives.
    # plus some 'useful columns' [_hh_id_, _ptype_, _cdap_rank_, _hh_size_]

    print "CONSTANTS: ", locals_d

    USE_FIXED_PROPORTIONS = True

    if USE_FIXED_PROPORTIONS:
        # relative probabilities depend only on ptype
        choosers = indiv_utils[indiv_utils[_cdap_rank_] > MAX_HHSIZE][[_hh_id_, _ptype_]]

        # eval the expression file
        model_design = eval_variables(cdap_fixed_relative_proportions.index, choosers, locals_d)

        # cdap_fixed_relative_proportions computes relative proportions by ptype, not utilities
        proportions = model_design.dot(cdap_fixed_relative_proportions)

        # convert relative proportions to probability
        probs = proportions.div(proportions.sum(axis=1), axis=0)

    else:
        # select only the utility columns
        utils = indiv_utils[indiv_utils[_cdap_rank_] > MAX_HHSIZE][['M', 'N', 'H']]

        # convert utilities to probabilities
        probs = nl.utils_to_probs(utils, trace_label=None)

    # select an activity pattern alternative for each person based on probability
    # result is a series indexed on _person_index_ with the (0 based) index of the column from probs
    idx_choices = nl.make_choices(probs, trace_label=trace_label)

    # convert choice from column index to activity name
    choices = pd.Series(probs.columns[idx_choices].values, index=probs.index)

    if DUMP:

        if USE_FIXED_PROPORTIONS:
            tracing.trace_df(proportions,
                             '%s.DUMP.extra_proportions' % trace_label,
                             transpose=False,
                             slicer='NONE')
        else:
            tracing.trace_df(utils,
                             '%s.DUMP.extra_utils' % trace_label,
                             transpose=False,
                             slicer='NONE')

        tracing.trace_df(probs,
                         '%s.DUMP.extra_probs' % trace_label,
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(choices,
                         '%s.DUMP.extra_choices' % trace_label,
                         transpose=False,
                         slicer='NONE')

    if trace_hh_id:

        if USE_FIXED_PROPORTIONS:
            tracing.trace_df(proportions, '%s.extra_hh_member_choices_proportions' % trace_label,
                             column_labels=['expression', 'person'],
                             warn_if_empty=False)
        else:
            tracing.trace_df(utils, '%s.extra_hh_member_choices_utils' % trace_label,
                             column_labels=['expression', 'person'],
                             warn_if_empty=False)

        tracing.trace_df(probs, '%s.extra_hh_member_choices_probs' % trace_label,
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)

        tracing.trace_df(choices, '%s.extra_hh_member_choices_choices' % trace_label,
                         column_labels=['expression', 'person'],
                         warn_if_empty=False)

    return choices


def print_elapsed_time(msg, t0=None):
    t1 = time.time()
    if DUMP:
        print "%s : %s" % (msg, t1 - (t0 or t1))
    return t1


def _run_cdap(
        persons,
        cdap_indiv_spec,
        cdap_interaction_coefficients,
        cdap_fixed_relative_proportions,
        locals_d,
        trace_hh_id, trace_label):

    """
    implements core run_cdap functionality but without chunking of persons df
    """

    t0 = print_elapsed_time("_run_cdap")

    persons[_cdap_rank_] = assign_cdap_rank(persons, trace_hh_id, trace_label)

    t0 = print_elapsed_time("assign_cdap_rank", t0)

    # Calculate CDAP utilities for each individual.
    # ind_utils has index of `PERID` and a column for each alternative
    # e.g. three columns 'M' (Mandatory), 'N' (NonMandatory), 'H' (Home)
    indiv_utils = individual_utilities(persons, cdap_indiv_spec, locals_d,
                                       trace_hh_id, trace_label)

    t0 = print_elapsed_time("individual_utilities", t0)

    hh_choices_list = []

    for hhsize in range(1, MAX_HHSIZE+1):

        choices = household_activity_choices(
            indiv_utils, cdap_interaction_coefficients, hhsize=hhsize,
            trace_hh_id=trace_hh_id, trace_label=trace_label)

        hh_choices_list.append(choices)

        t0 = print_elapsed_time("hhsize%d_utils" % hhsize, t0)

    # concat all the resulting Series
    hh_activity_choices = pd.concat(hh_choices_list)

    cdap_person_choices \
        = unpack_cdap_indiv_activity_choices(indiv_utils, hh_activity_choices,
                                             trace_hh_id, trace_label)

    extra_person_choices \
        = extra_hh_member_choices(indiv_utils, cdap_fixed_relative_proportions, locals_d,
                                  trace_hh_id, trace_label)

    person_choices = pd.concat([cdap_person_choices, extra_person_choices])

    if DUMP:
        tracing.trace_df(hh_activity_choices,
                         '%s.DUMP.hh_activity_choices' % trace_label,
                         transpose=False,
                         slicer='NONE')

        tracing.trace_df(person_choices,
                         '%s.DUMP.person_choices' % trace_label,
                         transpose=False,
                         slicer='NONE')

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
        persons,
        cdap_indiv_spec,
        cdap_interaction_coefficients,
        cdap_fixed_relative_proportions,
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
        choices = _run_cdap(persons,
                            cdap_indiv_spec,
                            cdap_interaction_coefficients,
                            cdap_fixed_relative_proportions,
                            locals_d,
                            trace_hh_id, trace_label)
        return choices

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, persons_chunk in hh_chunked_choosers(persons):

        logger.info("run_cdap running hh_chunk = %s with %d persons"
                    % (i, len(persons_chunk)))

        chunk_trace_label = "%s.chunk_%s" % (trace_label, i)

        choices = _run_cdap(persons_chunk,
                            cdap_indiv_spec,
                            cdap_interaction_coefficients,
                            cdap_fixed_relative_proportions,
                            locals_d,
                            trace_hh_id, chunk_trace_label)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    return choices

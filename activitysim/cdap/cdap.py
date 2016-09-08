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


def make_interactions(people, hh_id_col, p_type_col):
    """
    Make two Pandas DataFrames associating people IDs with two
    and three person interactions they have within their households.

    Interactions are strings of numbers representing the makeup
    of the interaction, e.g. '12' or '341'.

    Note that for two-person interactions the interaction string is ordered
    with the person from the index in the first position of the string
    and some other person in the second position. In contrast,
    the interaction strings for three-person interactions are not ordered.
    The person from the index may be in any position of the string.

    Parameters
    ----------
    people : pandas.DataFrame
        Table of people data. Must contain at least a household ID
        column and a categorization of person type.
    hh_id_col : str
        Name of the column in `people` that has their household ID.
    p_type_col : str
        Name of the column in `people` that contains the person type number.

    Returns
    -------
    two_interaction : pandas.DataFrame
        Interactions between two people. Index will be person IDs taken
        from the index of `people`.
        The table will have one column called `interaction`.
    three_interaction : pandas.DataFrame
        Interactions between three people. Index will be person IDs taken
        from the index of `people`.
        The table will have one column called `interaction`.

    """
    two_fmt = '{}{}'.format
    three_fmt = '{}{}{}'.format
    two = []
    three = []
    two_perm_cache = {}
    three_combo_cache = {}

    for hh_id, df in people.groupby(hh_id_col, sort=False):
        hh_size = len(df)

        # skip households with only one person
        if hh_size == 1:
            continue

        ptypes = df[p_type_col].values
        hh_idx = df.index.values

        if hh_size in two_perm_cache:
            two_perms = two_perm_cache[hh_size]
        else:
            two_perms = list(itertools.permutations(np.arange(hh_size), 2))
            two_perm_cache[hh_size] = two_perms

        two.extend(
            (hh_idx[pA], two_fmt(*ptypes[[pA, pB]])) for pA, pB in two_perms)

        # now skip households with two people
        if hh_size == 2:
            continue

        if hh_size in three_combo_cache:
            three_combos = three_combo_cache[hh_size]
        else:
            three_combos = list(itertools.combinations(np.arange(hh_size), 3))
            three_combo_cache[hh_size] = three_combos

        three.extend(
            (hh_idx[p], three_fmt(*ptypes.take(idx)))
            for idx in three_combos
            for p in idx)

    if two:
        two_idx, two_val = zip(*two)
    else:
        two_idx, two_val = [], []

    if three:
        three_idx, three_val = zip(*three)
    else:
        three_idx, three_val = [], []

    return (
        pd.DataFrame({'interaction': two_val}, index=two_idx),
        pd.DataFrame({'interaction': three_val}, index=three_idx))


def individual_utilities(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
        trace_hh_id=None, trace_label=None):
    """
    Calculate CDAP utilities for all individuals.

    Parameters
    ----------
    people : pandas.DataFrame
        DataFrame of individual people data.
    hh_id_col : str
        Name of the column in `people` that has their household ID.
    p_type_col : str
        Name of the column in `people` that contains the person type number.
    one_spec : pandas.DataFrame
        CDAP spec applied to individuals.
    two_spec : pandas.DataFrame
        CDAP spec applied to interactions between two people.
    three_spec : pandas.DataFrame
        CDAP spec applied to interactions between three people.

    Returns
    -------
    utilities : pandas.DataFrame
        Will have index of `people` and columns for each of the alternatives.

    """
    # calculate single person utilities
    #     evaluate variables from one_spec expressions
    #     multiply by one_spec alternative values
    one_vars = eval_variables(one_spec.index, people)
    one_utils = one_vars.dot(one_spec)

    # make two- and three-person interactions
    two_int, three_int = make_interactions(people, hh_id_col, p_type_col)

    # calculate two-interaction utilities
    #     evaluate variables from two_spec expressions
    #     multiply by two_spec alternative values
    #     groupby person and sum
    two_vars = eval_variables(two_spec.index, two_int)
    two_utils = two_vars.dot(two_spec).groupby(level=0).sum()

    # calculate three-interaction utilities
    #     evaluate variables from three_spec expressions
    #     multiply by three_spec alternative values
    #     groupby person and sum
    three_vars = eval_variables(three_spec.index, three_int)
    three_utils = three_vars.dot(three_spec).groupby(level=0).sum()

    # add one-, two-, and three-person utilities
    utils = one_utils.add(
        two_utils, fill_value=0).add(three_utils, fill_value=0)

    if trace_hh_id:
        tracing.trace_cdap_ind_utils(one_vars, '%s.ind_utils.one_vars' % trace_label)
        tracing.trace_cdap_ind_utils(one_utils, '%s.ind_utils.one_utils' % trace_label)

        tracing.trace_cdap_ind_utils(two_vars, '%s.ind_utils.two_vars' % trace_label)
        tracing.trace_cdap_ind_utils(two_utils, '%s.ind_utils.two_utils' % trace_label)

        tracing.trace_cdap_ind_utils(three_vars, '%s.ind_utils.three_vars' % trace_label)
        tracing.trace_cdap_ind_utils(three_utils, '%s.ind_util.three_utils' % trace_label)

    return utils


def initial_household_utilities(utilities, people, hh_id_col):
    """
    Create initial household utilities by grouping and summing utilities
    from individual household members.

    Parameters
    ----------
    utilities : pandas.DataFrame
        Should have the index of `people` and columns for each alternative.
    people : pandas.DataFrame
        DataFrame of individual people data.
    hh_id_col : str
        Name of the column in `people` that has their household ID.

    Returns
    -------
    hh_util : dict of pandas.Series
        Keys will be household IDs and values will be Series
        mapping alternative choices to their utility.

    """
    hh_util = {}

    alts = utilities.columns
    combo_cache = {}

    for hh_id, df in people.groupby(hh_id_col, sort=False):
        hh_size = len(df)
        utils = utilities.loc[df.index].as_matrix()

        if hh_size in combo_cache:
            ncombos, combos, flat_combos, tiled = combo_cache[hh_size]
        else:
            combos = list(itertools.product(alts, repeat=hh_size))
            flat_combos = list(
                tz.concat(itertools.product(range(len(alts)), repeat=hh_size)))
            ncombos = len(combos)
            tiled = np.tile(np.arange(hh_size), ncombos)
            combo_cache[hh_size] = (ncombos, combos, flat_combos, tiled)

        u = utils[tiled, flat_combos].reshape((ncombos, hh_size)).sum(axis=1)

        hh_util[hh_id] = pd.Series(u, index=combos)

    return hh_util


def apply_final_rules(hh_util, people, hh_id_col, final_rules):
    """
    Final rules can be used to set the utility values for certain
    household alternatives. Often they are set to zero to reflect
    the unavailability of certain alternatives to certain types of people.

    This modifies the `hh_util` data inplace.

    Parameters
    ----------
    hh_util : dict of pandas.Series
        Keys will be household IDs and values will be Series
        mapping alternative choices to their utility.
    people : pandas.DataFrame
        DataFrame of individual people data.
    hh_id_col : str
        Name of the column in `people` that has their household ID.
    final_rules : pandas.DataFrame
        This table must have an index of expressions that can be used
        to filter the `people` table. It must have two columns:
        the first must have the name of the alternative to which the rule
        applies, and the second must have the value of the utility for that
        alternative. The names of the columns is not important, but
        the order is.

    """
    rule_mask = eval_variables(final_rules.index, people)

    if not rule_mask.as_matrix().any():
        # if the rules don't apply to anyone then return now
        return

    alt_match_cache = {}

    for hh_id, df in people.groupby(hh_id_col, sort=False):
        mask = rule_mask.loc[df.index]
        if not mask.as_matrix().any():
            # if the mask doesn't apply to anyone in this household
            # carry on to the next household
            continue

        utils = hh_util[hh_id]
        hh_size = len(df)

        for exp, row in final_rules.iterrows():
            m = mask[exp].as_matrix()
            if not m.any():
                # if this sub-mask doesn't apply to anyone here then
                # carry on to the next rule
                continue

            # this crazy business combines three things to figure out
            # which household alternatives need to be modified by this rule.
            # the three things are:
            # - the mask of people for whom the rule expression is true (m)
            # - the individual alternative to which the rule applies
            #   (alt)
            # - the alternative combinations for the household
            #   (utils.index)
            alt = row.iloc[0]
            key = (alt, hh_size)
            if key in alt_match_cache:
                alt_match = alt_match_cache[key]
            else:
                alt_match = np.array(utils.index.values.tolist()) == alt
                alt_match_cache[key] = alt_match

            app = np.any(np.bitwise_and(alt_match, m), axis=1)

            utils[app] = row.iloc[1]


def apply_all_people(hh_util, all_people):
    """
    Apply utility adjustments to household alternatives.

    This modifies the `hh_util` data inplace.

    Parameters
    ----------
    hh_util : dict of pandas.Series
        Keys will be household IDs and values will be Series
        mapping alternative choices to their utility.
    all_people : pandas.DataFrame
        Adjustments to household alternatives, with alternatives in the
        index and the adjustment values in the first column.
        Index should be household alternatives in the form of tuples
        containing individual alternatives, e.g.
        ('Mandatory', 'Mandatory', 'Mandatory'), where 'Mandatory' is
        one of the alternatives available to individual household members.
        Note that these may also be expressed as Python code to save space,
        so the previous could also be written as ('Mandatory',) * 3.

    """

    # FIXME - don't modify all_people in place
    all_people = all_people.copy(deep=True)

    # evaluate all the expressions in the all_people index
    all_people.index = [eval(x) for x in all_people.index]

    # FIXME - pathological knowledge of column position?
    all_people = all_people.iloc[:, 0]

    matching_idx = {}

    for hh in hh_util.values():
        l = len(hh)
        if l in matching_idx:
            matching = matching_idx[l]
        else:
            matching = hh.index.intersection(all_people.index)
            matching_idx[l] = matching

        hh.loc[matching] += all_people.loc[matching]


def make_household_choices(hh_util, trace_hh_id=None, trace_label=None):
    """
    Decide on the activity pattern for each household.

    Parameters
    ----------
    hh_util : dict of pandas.Series
        Keys will be household IDs and values will be Series
        mapping alternative choices to their utility.
        The series index values are tuples e.g. ('NonMandatory', 'Mandatory', 'Home')
        The length of the tuple in the index depends on the number of hh members

    In other words, hh_utils looks something like this:
    hh_util =  {
        237824: pandas.Series(
            (Mandatory, Mandatory)       :   6.39426
            (Mandatory, NonMandatory)    :   5.09445
            (Mandatory, Home)            :    4.6314
            (NonMandatory, Mandatory)    :   4.48733
            (NonMandatory, NonMandatory) :   3.18751
            (NonMandatory, Home)         :   2.72446
            (Home, Mandatory)            :   4.76486
            (Home, NonMandatory)         :   3.46505
        ),
        237825: pandas.Series(
            (Mandatory)                  :   6.39426
            (NonMandatory)               :   5.09445
            (Home)                       :    4.6314
        )
    }

    Returns
    -------
    choices : pandas.Series
        Maps household ID to chosen alternative, where the alternative
        is a tuple of individual utilities.

    """

    # we want to segregate hh_utils by number of alternatives
    unique_alternative_counts = tz.unique(tz.map(len, hh_util.values()))

    choices_list = []
    for alternative_count in unique_alternative_counts:

        # filter hh_util to segregate series with this number of alternatives
        hh_util_n = tz.valfilter(lambda x: len(x) == alternative_count, hh_util)

        if trace_hh_id and trace_hh_id in hh_util_n:
            tracing.info(message="household %s in %s.make_household_choices with %s alternatives"
                         % (trace_hh_id, trace_label, alternative_count))

        # compose a function to convert hh_util dict subset to a df of alternative utilities
        # input dict should be filtered to contain only series with same number of alternatives
        # pd.DataFrame.from_dict returns a df with a column for each hh_id
        #                        and utility values indexed by alternative list tuples
        # pd.DataFrame.transpose takes that df and transposes it to return a df
        #                        indexed by hh_id with utilities of the alternatives in the columns

        standard_hh_util_df = tz.compose(
            pd.DataFrame.transpose,
            pd.DataFrame.from_dict)

        df = standard_hh_util_df(hh_util_n)

        # resulting df has one row per hh_id
        assert len(df.index) == len(hh_util_n)

        # resulting df has one column per alternative
        assert len(df.columns) == alternative_count

        if trace_hh_id:
            tracing.trace_df(df, '%s.hh_choices.utilities' % trace_label,
                             slicer='HHID', column_labels=['alternative', 'utility'])

        probs = nl.utils_to_probs(df, trace_label=trace_label)
        if trace_hh_id:
            tracing.trace_df(probs, '%s.hh_choices.probs' % trace_label,
                             slicer='HHID', column_labels=['alternative', 'probability'])

        choices = pd.Series(
            df.columns[nl.make_choices(probs, trace_label)].values,
            index=df.index)

        choices_list.append(choices)

    # concat all the resulting Series
    choices = pd.concat(choices_list)

    return choices


def household_choices_to_people(hh_choices, people):
    """
    Map household choices to people so that we know the activity pattern
    for individuals.

    Parameters
    ----------
    hh_choices : pandas.Series
        Maps household ID to chosen alternative, where the alternative
        is a tuple of individual utilities.
    people : pandas.DataFrame
        DataFrame of individual people data.

    Returns
    -------
    choices : pandas.Series
        Maps index of `people` to their activity pattern choice.

    """
    return pd.Series(
        gen(tz.concat(hh_choices.values)), index=people.index)


def _run_cdap(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
        final_rules, all_people, trace_hh_id, trace_label):

    """
    implements core run_cdap functionality but without chunking of people df
    """

    ind_utils = individual_utilities(
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec, trace_hh_id, trace_label)

    if trace_hh_id:
        tracing.trace_cdap_ind_utils(ind_utils, '%s.ind_utils' % trace_label)

    hh_utils = initial_household_utilities(ind_utils, people, hh_id_col)

    if trace_hh_id:
        tracing.trace_cdap_hh_utils(hh_utils, '%s.hh_utils' % trace_label)

    apply_final_rules(hh_utils, people, hh_id_col, final_rules)

    if trace_hh_id:
        tracing.trace_cdap_hh_utils(hh_utils, '%s.hh_utils.apply_final_rules' % trace_label)

    apply_all_people(hh_utils, all_people)

    if trace_hh_id:
        tracing.trace_cdap_hh_utils(hh_utils, '%s.hh_utils.apply_all_people' % trace_label)

    hh_choices = make_household_choices(hh_utils, trace_hh_id, trace_label)

    if trace_hh_id:
        tracing.trace_df(hh_choices, '%s.hh_choices' % trace_label,
                         slicer='HHID', columns='choice')

    person_choices = household_choices_to_people(hh_choices, people)

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
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
        final_rules, all_people,
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
    one_spec : pandas.DataFrame
        CDAP spec applied to individuals.
    two_spec : pandas.DataFrame
        CDAP spec applied to interactions between two people.
    three_spec : pandas.DataFrame
        CDAP spec applied to interactions between three people.
    final_rules : pandas.DataFrame
        This table must have an index of expressions that can be used
        to filter the `people` table. It must have two columns:
        the first must have the name of the alternative to which the rule
        applies, and the second must have the value of the utility for that
        alternative. The names of the columns is not important, but
        the order is.
    all_people : pandas.DataFrame
        Adjustments to household alternatives, with alternatives in the
        index and the adjustment values in the first column.
        Index should be household alternatives in the form of tuples
        containing individual alternatives, e.g.
        ('Mandatory', 'Mandatory', 'Mandatory'), where 'Mandatory' is
        one of the alternatives available to individual household members.
        Note that these may also be expressed as Python code to save space,
        so the previous could also be written as ('Mandatory',) * 3.
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
        choices = _run_cdap(people, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
                            final_rules, all_people, trace_hh_id, trace_label)
        return choices

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for i, people_chunk in hh_chunked_choosers(people):

        logger.info("run_cdap running hh_chunk =%s of size %d" % (i, len(people_chunk)))

        chunk_trace_label = "%s.chunk_%s" % (trace_label, i)

        choices = _run_cdap(people_chunk, hh_id_col, p_type_col, one_spec, two_spec, three_spec,
                            final_rules, all_people, trace_hh_id, chunk_trace_label)

        choices_list.append(choices)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    choices = pd.concat(choices_list)

    return choices

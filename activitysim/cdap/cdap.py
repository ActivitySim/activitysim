import itertools

import numpy as np
import pandas as pd
from zbox import toolz as tz, gen

from ..activitysim import eval_variables
from .. import mnl


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

    for hh, df in people.groupby(hh_id_col, sort=False):
        # skip households with only one person
        if len(df) == 1:
            continue

        ptypes = df[p_type_col]

        for pA, pB in itertools.permutations(df.index, 2):
            two.append((pA, two_fmt(*ptypes[[pA, pB]])))

        # now skip households with two people
        if len(df) == 2:
            continue

        for idx in itertools.combinations(df.index, 3):
            combo = three_fmt(*ptypes[list(idx)])
            three.extend((p, combo) for p in idx)

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
        people, hh_id_col, p_type_col, one_spec, two_spec, three_spec):
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
    two_spec : pandas.DataFrame
    three_spec : pandas.DataFrame

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

    for hh_id, df in people.groupby(hh_id_col, sort=False):
        utils = utilities.loc[df.index]
        hh = []

        for combo in itertools.product(alts, repeat=len(df)):
            hh.append(
                (combo, utils.lookup(df.index, combo).sum()))

        idx, u = zip(*hh)
        hh_util[hh_id] = pd.Series(u, index=idx)

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

    for hh_id, df in people.groupby(hh_id_col, sort=False):
        mask = rule_mask.loc[df.index]
        utils = hh_util[hh_id]

        for exp, row in final_rules.iterrows():
            m = mask[exp].as_matrix()

            # this crazy business combines three things to figure out
            # which household alternatives need to be modified by this rule.
            # the three things are:
            # - the mask of people for whom the rule expression is true (m)
            # - the individual alternative to which the rule applies
            #   (row.iloc[0])
            # - the alternative combinations for the household (combo)
            app = [
                ((np.array([row.iloc[0]] * len(utils.index[0])) == combo) & m
                 ).any()
                for combo in utils.index]

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
    # evaluate all the expressions in the all_people index
    all_people.index = [eval(x) for x in all_people.index]
    all_people = all_people.icol(0)

    matching_idx = {}

    for hh in hh_util.values():
        l = len(hh)
        if l in matching_idx:
            matching = matching_idx[l]
        else:
            matching = hh.index.intersection(all_people.index)
            matching_idx[l] = matching

        hh.loc[matching] += all_people.loc[matching]


def make_household_choices(hh_util):
    """
    Decide on the activity pattern for each household.

    Parameters
    ----------
    hh_util : dict of pandas.Series
        Keys will be household IDs and values will be Series
        mapping alternative choices to their utility.

    Returns
    -------
    choices : pandas.Series
        Maps household ID to chosen alternative, where the alternative
        is a tuple of individual utilities.

    """
    # convert hh_util dict to a few DFs with alternatives in the columns
    # and household IDs in the index
    df_func = tz.compose(
        pd.DataFrame.transpose,
        pd.DataFrame.from_dict)
    grouped_by_size = (
        tz.valfilter(lambda x: len(x) == l, hh_util)
        for l in tz.unique(tz.map(len, hh_util.values())))
    dfs = tz.map(df_func, grouped_by_size)

    # go over all the DFs and do utils_to_probs and make_choices
    choices = (
        pd.Series(
            df.columns[mnl.make_choices(mnl.utils_to_probs(df))].values,
            index=df.index)
        for df in dfs)

    # concat all the resulting Series
    return pd.concat(choices)


def household_choices_to_people(hh_choices, people, hh_id_col):
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
    hh_id_col : str
        Name of the column in `people` that has their household ID.

    Returns
    -------
    choices : pandas.Series
        Maps index of `people` to their activity pattern choice.

    """
    return pd.Series(
        gen(tz.concat(hh_choices.values)), index=people.index)

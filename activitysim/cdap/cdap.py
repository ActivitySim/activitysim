import itertools

import numpy as np
import pandas as pd

from ..activitysim import eval_variables


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

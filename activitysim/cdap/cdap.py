import itertools

import pandas as pd


def make_interactions(people, hh_id_col, p_type_col):
    """
    Make two Pandas series associating people IDs with two
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
    two_interaction : pandas.Series
        Interactions between two people. Index will be person IDs taken
        from the index of `people`.
    three_interaction : pandas.Series
        Interactions between three people. Index will be person IDs taken
        from the index of `people`.

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
        pd.Series(two_val, index=two_idx),
        pd.Series(three_val, index=three_idx))

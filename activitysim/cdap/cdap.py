def make_interactions(people, hh_id_col, p_type_col):
    """
    Make two Pandas series associating people IDs with two
    and three person interactions they have within their households.

    Interactions are strings of numbers representing the makeup
    of the interaction, e.g. '12' or '341'.

    Parameters
    ----------
    people : pandas.DataFrame
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
    pass

from __future__ import annotations

import pandas as pd


def simple_table_join(
    left: pd.DataFrame, right: pd.DataFrame, left_on: str
) -> pd.DataFrame:
    """
    A simple table join.

    The left table should usually have a many-to-one (or a one-to-one)
    relationship with the right table (so, exactly one row on the right table
    matches each row in the left).  This is not enforced and the code can
    still work with many-to-many join, but ActivitySim by convention includes
    only many-to-one joins.

    This function mostly mirrors the usual pandas `join`, except when there
    are duplicate column names in the right-side table, in which case those
    duplciate columns are silently dropped instead of getting renamed.

    Parameters
    ----------
    left, right : DataFrame
    left_on : str
        The name of the column of the left

    Returns
    -------
    DataFrame
    """
    # all the column names in both left and right
    intersection = set(left.columns).intersection(right.columns)
    intersection.discard(left_on)  # intersection is ok if it's the join key

    # duplicate column names in the right-side table are ignored.
    right = right.drop(intersection, axis=1)

    return pd.merge(
        left,
        right,
        left_on=left_on,
        right_index=True,
    )

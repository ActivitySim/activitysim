import os
import psutil
import gc

from operator import itemgetter

import numpy as np
import pandas as pd

from zbox import toolz as tz


def memory_info():
    gc.collect()
    process = psutil.Process(os.getpid())
    bytes = process.memory_info().rss
    mb = (bytes / (1024 * 1024.0))
    gb = (bytes / (1024 * 1024 * 1024.0))
    return "memory_info: %s MB (%s GB)" % (int(mb), round(gb, 2))


def left_merge_on_index_and_col(left_df, right_df, join_col, target_col):
    """
    like pandas left merge, but join on both index and a specified join_col

    FIXME - for now return a series of ov values from specified right_df target_col

    Parameters
    ----------
    left_df : pandas DataFrame
        index name assumed to be same as that of right_df
    right_df : pandas DataFrame
        index name assumed to be same as that of left_df
    join_col : str
        name of column to join on (in addition to index values)
        should have same name in both dataframes
    target_col : str
        name of column from right_df whose joined values should be returned as series

    Returns
    -------
    target_series : pandas Series
        series of target_col values with same index as left_df
        i.e. values joined to left_df from right_df with index of left_df
    """
    assert left_df.index.name == right_df.index.name

    # want to know name previous index column will have after reset_index
    idx_col = right_df.index.name

    # SELECT target_col FROM full_sample LEFT JOIN unique_sample on idx_col, join_col
    merged = \
        pd.merge(
            left_df[[join_col]].reset_index(),
            right_df[[join_col, target_col]].reset_index(),
            on=[idx_col, join_col],
            how="left")

    merged.set_index(idx_col, inplace=True)

    return merged[target_col]


def reindex(series1, series2):
    """
    This reindexes the first series by the second series.  This is an extremely
    common operation that does not appear to  be in Pandas at this time.
    If anyone knows of an easier way to do this in Pandas, please inform the
    UrbanSim developers.

    The canonical example would be a parcel series which has an index which is
    parcel_ids and a value which you want to fetch, let's say it's land_area.
    Another dataset, let's say of buildings has a series which indicate the
    parcel_ids that the buildings are located on, but which does not have
    land_area.  If you pass parcels.land_area as the first series and
    buildings.parcel_id as the second series, this function returns a series
    which is indexed by buildings and has land_area as values and can be
    added to the buildings dataset.

    In short, this is a join on to a different table using a foreign key
    stored in the current table, but with only one attribute rather than
    for a full dataset.

    This is very similar to the pandas "loc" function or "reindex" function,
    but neither of those functions return the series indexed on the current
    table.  In both of those cases, the series would be indexed on the foreign
    table and would require a second step to change the index.

    Parameters
    ----------
    series1, series2 : pandas.Series

    Returns
    -------
    reindexed : pandas.Series

    """

    # turns out the merge is much faster than the .loc below
    df = pd.merge(series2.to_frame(name='left'),
                  series1.to_frame(name='right'),
                  left_on="left",
                  right_index=True,
                  how="left")
    return df.right

    # return pd.Series(series1.loc[series2.values].values, index=series2.index)


def other_than(groups, bools):
    """
    Construct a Series that has booleans indicating the presence of
    something- or someone-else with a certain property within a group.

    Parameters
    ----------
    groups : pandas.Series
        A column with the same index as `bools` that defines the grouping
        of `bools`. The `bools` Series will be used to index `groups` and
        then the grouped values will be counted.
    bools : pandas.Series
        A boolean Series indicating where the property of interest is present.
        Should have the same index as `groups`.

    Returns
    -------
    others : pandas.Series
        A boolean Series with the same index as `groups` and `bools`
        indicating whether there is something- or something-else within
        a group with some property (as indicated by `bools`).

    """
    counts = groups[bools].value_counts()
    merge_col = groups.to_frame(name='right')
    pipeline = tz.compose(
        tz.curry(pd.Series.fillna, value=False),
        itemgetter('left'),
        tz.curry(
            pd.DataFrame.merge, right=merge_col, how='right', left_index=True,
            right_on='right'),
        tz.curry(pd.Series.to_frame, name='left'))
    gt0 = pipeline(counts > 0)
    gt1 = pipeline(counts > 1)

    return gt1.where(bools, other=gt0)


def quick_loc_df(loc_list, target_df, attribute):
    """
    faster replacement for target_df.loc[loc_list][attribute]

    pandas DataFrame.loc[] indexing doesn't scale for large arrays (e.g. > 1,000,000 elements)

    Parameters
    ----------
    loc_list : list-like (numpy.ndarray, pandas.Int64Index, or pandas.Series)
    target_df : pandas.DataFrame containing column named attribute
    attribute : name of column from loc_list to return

    Returns
    -------
        pandas.Series
    """

    left_on = "left"

    if isinstance(loc_list, pd.Int64Index):
        left_df = pd.DataFrame({left_on: loc_list.values})
    elif isinstance(loc_list, pd.Series):
        left_df = loc_list.to_frame(name=left_on)
    elif isinstance(loc_list, np.ndarray):
        left_df = pd.DataFrame({left_on: loc_list})
    else:
        raise RuntimeError("quick_loc_df loc_list of unexpected type %s" % type(loc_list))

    df = pd.merge(left_df,
                  target_df[[attribute]],
                  left_on=left_on,
                  right_index=True,
                  how="left")

    # regression test
    # assert list(df[attribute]) == list(target_df.loc[loc_list][attribute])

    return df[attribute]


def quick_loc_series(loc_list, target_series):
    """
    faster replacement for target_series.loc[loc_list]

    pandas Series.loc[] indexing doesn't scale for large arrays (e.g. > 1,000,000 elements)

    Parameters
    ----------
    loc_list : list-like (numpy.ndarray, pandas.Int64Index, or pandas.Series)
    target_series : pandas.Series

    Returns
    -------
        pandas.Series
    """

    left_on = "left"

    if isinstance(loc_list, pd.Int64Index):
        left_df = pd.DataFrame({left_on: loc_list.values})
    elif isinstance(loc_list, pd.Series):
        left_df = loc_list.to_frame(name=left_on)
    elif isinstance(loc_list, np.ndarray):
        left_df = pd.DataFrame({left_on: loc_list})
    else:
        raise RuntimeError("quick_loc_series loc_list of unexpected type %s" % type(loc_list))

    df = pd.merge(left_df,
                  target_series.to_frame(name='right'),
                  left_on=left_on,
                  right_index=True,
                  how="left")

    # regression test
    # assert list(df.right) == list(target_series.loc[loc_list])

    return df.right

# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import inject, tracing

logger = logging.getLogger(__name__)


def rle(a):
    """
    Compute run lengths of values in rows of a two dimensional ndarry of ints.

    We assume the first and last columns are buffer columns
    (because this is the case for time windows)  and so don't include them in results.


    Return arrays giving row_id, start_pos, run_length, and value of each run of any length.

    Parameters
    ----------
    a : numpy.ndarray of int shape(n, <num_time_periods_in_a_day>)


        The input array would normally only have values of 0 or 1 to detect overlapping
        time period availability but we don't assume this, and will detect and report
        runs of any values. (Might prove useful in future?...)

    Returns
    -------
    row_id : numpy.ndarray int shape(<num_runs>)
    start_pos : numpy.ndarray int shape(<num_runs>)
    run_length : numpy.ndarray int shape(<num_runs>)
    run_val : numpy.ndarray int shape(<num_runs>)
    """

    # note timewindows have a beginning and end of day padding columns that we must ignore
    # a = [[1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    #      [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    #      [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    #      [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]])

    a = np.asarray(a)

    # row element is different from the one before
    changed = np.array(a[..., :-1] != a[..., 1:])

    # first and last real columns always considered different from padding
    changed[..., 0] = True
    changed[..., -1] = True

    # array([[ True, False, False,  True,  True, False, False,  True,  True],
    #        [ True, False,  True, False,  True, False,  True,  True,  True],
    #        [ True,  True,  True, False,  True, False,  True,  True,  True],
    #        [ True, False, False, False, False, False,  True,  True,  True]])

    # indices of change points (row_index, col_index)
    i = np.where(changed)
    # ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    #  [0, 3, 4, 7, 8, 0, 2, 4, 6, 7, 8, 0, 1, 2, 4, 6, 7, 8, 0, 6, 7, 8])

    row_id = i[0][1:]
    row_changed, run_length = np.diff(i)
    # row_id      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    # row_changed [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # run_length  [3, 1, 3, 1,-8, 2, 2, 2, 1, 1,-8, 1, 1, 2, 2, 1, 1,-8, 6, 1, 1]

    # start position of run in changed array + 1 to get pos in input array
    start_pos = np.cumsum(np.append(0, run_length))[:-1] + 1
    # start_pos   [1, 4, 5, 8, 9, 1, 3, 5, 7, 8, 9, 1, 2, 3, 5, 7, 8, 9, 1, 7, 8]

    # drop bogus negative run length when row changes, we want to drop them
    real_rows = np.where(1 - row_changed)[0]
    row_id = row_id[real_rows]
    run_length = run_length[real_rows]
    start_pos = start_pos[real_rows]

    # index into original array to get run_val
    run_val = a[(row_id, start_pos)]

    # real_rows  [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20]
    # row_id     [0, 0, 0, 0, 1, 1, 1, 1, 1,  2,  2,  2,  2,  2,  2,  3,  3,  3]
    # run_length [3, 1, 3, 1, 2, 2, 2, 1, 1,  1,  1,  2,  2,  1,  1,  6,  1,  1]
    # start_pos  [1, 4, 5, 8, 1, 3, 5, 7, 8,  1,  2,  3,  5,  7,  8,  1,  7,  8]
    # run_val    [1, 0, 1, 0, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  1,  0,  1]

    return row_id, start_pos, run_length, run_val


def p2p_time_window_overlap(p1_ids, p2_ids):
    """

    Parameters
    ----------
    p1_ids
    p2_ids

    Returns
    -------

    """

    timetable = inject.get_injectable("timetable")

    assert len(p1_ids) == len(p2_ids)
    # if series, ought to have same index
    assert (p1_ids.index == p2_ids.index).all()

    # ndarray with one row per p2p and one column per time period
    # array value of 1 where overlapping free periods and 0 elsewhere
    available = timetable.pairwise_available(p1_ids, p2_ids)

    row_ids, start_pos, run_length, run_val = rle(available)

    # rle returns all runs, but we only care about runs of available (run_val == 1)
    target_rows = np.where(run_val == 1)
    row_ids = row_ids[target_rows]
    run_length = run_length[target_rows]

    df = pd.DataFrame({"row_ids": row_ids, "run_length": run_length})

    # groupby index of row_ids match the numpy row indexes of timetable.pairwise_available ndarray
    # but there may be missing values of any no-overlap persons pairs
    max_overlap = df.groupby("row_ids").run_length.max()
    # fill in any missing values to align with input arrays
    input_row_ids = np.arange(len(p1_ids))
    max_overlap = max_overlap.reindex(input_row_ids).fillna(0)

    # FIXME should we return series or ndarray?
    max_overlap.index = p1_ids.index

    return max_overlap


def person_pairs(persons):

    p = persons[["household_id", "adult"]].reset_index()
    p2p = pd.merge(p, p, left_on="household_id", right_on="household_id", how="outer")

    # we desire well known non-contingent column names
    p2p.rename(
        columns={
            "%s_x" % persons.index.name: "person1",
            "%s_y" % persons.index.name: "person2",
        },
        inplace=True,
    )

    p2p = p2p[p2p.person1 < p2p.person2]

    # index is meaningless, but might as well be tidy
    p2p.reset_index(drop=True, inplace=True)

    p2p["p2p_type"] = (p2p.adult_x * 1 + p2p.adult_y * 1).map(
        {0: "cc", 1: "ac", 2: "aa"}
    )

    p2p = p2p[["household_id", "person1", "person2", "p2p_type"]]

    return p2p


def hh_time_window_overlap(households, persons):

    p2p = person_pairs(persons)

    p2p["max_overlap"] = p2p_time_window_overlap(p2p.person1, p2p.person2)

    hh_overlap = (
        p2p.groupby(["household_id", "p2p_type"])
        .max_overlap.max()
        .unstack(level=-1, fill_value=0)
    )

    # fill in missing households (in case there were no overlaps)
    hh_overlap = hh_overlap.reindex(households.index).fillna(0).astype(np.int8)

    # make sure we have all p2p_types (if there were none to unstack, then column will be missing)
    for c in ["aa", "cc", "ac"]:
        if c not in hh_overlap.columns:
            hh_overlap[c] = 0

    return hh_overlap


def person_time_window_overlap(persons):

    p2p = person_pairs(persons)

    p2p["max_overlap"] = p2p_time_window_overlap(p2p.person1, p2p.person2)

    p_overlap = (
        pd.concat(
            [
                p2p[["person1", "p2p_type", "max_overlap"]].rename(
                    columns={"person1": "person_id"}
                ),
                p2p[["person2", "p2p_type", "max_overlap"]].rename(
                    columns={"person2": "person_id"}
                ),
            ]
        )
        .groupby(["person_id", "p2p_type"])
        .max_overlap.max()
    )

    # unstack to create columns for each p2p_type (aa, cc, and ac)
    p_overlap = p_overlap.unstack(level=-1, fill_value=0)

    # make sure we have columns for all p2p_types (in case there were none of a p2ptype to unstack)
    for c in ["aa", "cc", "ac"]:
        if c not in p_overlap.columns:
            p_overlap[c] = 0

    # fill in missing households (in case there were persons with no overlaps)
    p_overlap = p_overlap.reindex(persons.index).fillna(0).astype(np.int8)

    return p_overlap


def person_max_window(persons):

    timetable = inject.get_injectable("timetable")

    # ndarray with one row per person and one column per time period
    # array value of 1 where free periods and 0 elsewhere
    s = pd.Series(persons.index.values, index=persons.index)
    available = timetable.individually_available(s)

    row_ids, start_pos, run_length, run_val = rle(available)

    # rle returns all runs, but we only care about runs of available (run_val == 1)
    target_rows = np.where(run_val == 1)
    row_ids = row_ids[target_rows]
    run_length = run_length[target_rows]

    df = pd.DataFrame({"row_ids": row_ids, "run_length": run_length})

    # groupby index of row_ids match the numpy row indexes of timetable.pairwise_available ndarray
    # but there may be missing values of any no-overlap persons pairs
    max_overlap = df.groupby("row_ids").run_length.max()
    # fill in any missing values to align with input arrays
    input_row_ids = np.arange(persons.shape[0])
    max_window = max_overlap.reindex(input_row_ids).fillna(0)

    # FIXME should we return series or ndarray?
    max_window.index = persons.index

    return max_window

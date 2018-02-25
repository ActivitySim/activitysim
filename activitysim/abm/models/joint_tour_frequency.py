# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from activitysim.core.util import assign_in_place
from .util import expressions
from .util.tour_frequency import process_joint_tours

logger = logging.getLogger(__name__)


@inject.injectable()
def joint_tour_frequency_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'joint_tour_frequency.csv')


@inject.injectable()
def joint_tour_frequency_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'joint_tour_frequency.yaml')


@inject.injectable()
def joint_tour_frequency_alternatives(configs_dir):
    # alt file for building tours even though simulation is simple_simulate not interaction_simulate
    f = os.path.join(configs_dir, 'joint_tour_frequency_alternatives.csv')
    df = pd.read_csv(f, comment='#')
    df.set_index('alt', inplace=True)
    return df


def person_pairs(persons):

    p = persons[['household_id', 'adult']].reset_index()
    p2p = pd.merge(p, p, left_on='household_id', right_on='household_id', how='outer')

    p2p = p2p[p2p.PERID_x < p2p.PERID_y]

    p2p['p2p_type'] = (p2p.adult_x * 1 + p2p.adult_y * 1).map({0: 'cc', 1: 'ac', 2: 'aa'})

    p2p = p2p[['household_id', 'PERID_x', 'PERID_y', 'p2p_type']]

    return p2p


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


def time_window_overlap(households, persons):

    timetable = inject.get_injectable("timetable")

    p2p = person_pairs(persons)\

    # we want p2p index to be zero-based series to match row_ids returned by rle
    p2p.reset_index(drop=True, inplace=True)

    # ndarray with one row per p2p and one column per time period
    # array value of 1 where overlapping free periods and 0 elsewhere
    available = timetable.pairwise_available(p2p.PERID_x, p2p.PERID_y)

    row_ids, start_pos, run_length, run_val = rle(available)

    # rle returns all runs, but we only care about runs of available (run_val == 1)
    target_rows = np.where(run_val == 1)
    row_ids = row_ids[target_rows]
    run_length = run_length[target_rows]

    df = pd.DataFrame({'row_ids': row_ids, 'run_length': run_length})

    p2p['max_run_len'] = df.groupby('row_ids').run_length.max()
    p2p.max_run_len.fillna(0, inplace=True)

    hh_time_window_overlap = \
        p2p.groupby(['household_id', 'p2p_type']).max_run_len.max().unstack(level=-1, fill_value=0)

    # fill in missing households (in case there were no overlaps)
    hh_time_window_overlap = hh_time_window_overlap.reindex(households.index).fillna(0)

    hh_time_window_overlap.rename(
        columns={'aa': 'time_window_overlap_adult',
                 'cc': 'time_window_overlap_child',
                 'ac': 'time_window_overlap_adult_child'},
        inplace=True
    )

    return hh_time_window_overlap


@inject.step()
def joint_tour_frequency(
        households, persons,
        joint_tour_frequency_spec,
        joint_tour_frequency_settings,
        joint_tour_frequency_alternatives,
        configs_dir,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """
    trace_label = 'joint_tour_frequency'

    households = households.to_frame()
    persons = persons.to_frame()

    multi_person_households = households[households.hhsize > 1].copy()

    logger.info("Running joint_tour_frequency with %d multi-person households" %
                multi_person_households.shape[0])

    macro_settings = joint_tour_frequency_settings.get('joint_tour_frequency_macros', None)

    hh_time_window_overlap = time_window_overlap(multi_person_households, persons)
    assign_in_place(multi_person_households, hh_time_window_overlap)

    if macro_settings:
        expressions.assign_columns(
            df=multi_person_households,
            model_settings=macro_settings,
            trace_label=trace_label)

    nest_spec = config.get_logit_model_settings(joint_tour_frequency_settings)
    constants = config.get_model_constants(joint_tour_frequency_settings)

    choices = simulate.simple_simulate(
        multi_person_households,
        spec=joint_tour_frequency_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='joint_tour_frequency')

    # convert indexes to alternative names
    choices = pd.Series(joint_tour_frequency_spec.columns[choices.values], index=choices.index)

    # add joint_tour_frequency column to households
    # reindex since we are working with a subset of households
    households['joint_tour_frequency'] = choices.reindex(households.index)

    # - remember this as it is needed by subsequent joint_tour model steps
    assign_in_place(households, hh_time_window_overlap.reindex(households.index).fillna(0))
    pipeline.replace_table("households", households)

    # - create atwork_subtours based on atwork_subtour_frequency choice names
    multi_person_households = households[households.hhsize > 1]
    assert not multi_person_households.joint_tour_frequency.isnull().any()

    joint_tours = process_joint_tours(multi_person_households, joint_tour_frequency_alternatives)
    pipeline.replace_table('joint_tours', joint_tours)
    tracing.register_traceable_table('joint_tours', joint_tours)

    pipeline.get_rn_generator().add_channel(joint_tours, 'joint_tours')

    tracing.print_summary('joint_tour_frequency', households.joint_tour_frequency,
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(multi_person_households,
                         label="joint_tour_frequency.households",
                         warn_if_empty=True)

        tracing.trace_df(joint_tours,
                         label="joint_tour_frequency.joint_tours",
                         slicer='household_id',
                         warn_if_empty=True)

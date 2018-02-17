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

from .util import expressions
from activitysim.core.util import assign_in_place

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

    # real_rows  [ 0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20]
    # row_id     [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
    # run_length [3, 1, 3, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 6, 1, 1]
    # start_pos  [1, 4, 5, 8, 1, 3, 5, 7, 8, 1, 2, 3, 5, 7, 8, 1, 7, 8]

    # we now have row_id, run_length, and start_pos of all runs
    # but we only care about runs of target_run_val

    # index into original array to get run_val
    run_val = a[(row_id, start_pos)]

    return row_id, start_pos, run_length, run_val


def assign_time_window_overlap(households, persons):

    timetable = inject.get_injectable("timetable")

    p2p = person_pairs(persons).reset_index(drop=True)

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

    households['time_window_overlap_adult'] = hh_time_window_overlap['aa']
    households['time_window_overlap_child'] = hh_time_window_overlap['cc']
    households['time_window_overlap_adult_child'] = hh_time_window_overlap['ac']


@inject.step()
def joint_tour_frequency(households, persons,
                         joint_tour_frequency_spec,
                         joint_tour_frequency_settings,
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

    multi_person_households = households[households.PERSONS > 1].copy()

    logger.info("Running joint_tour_frequency with %d multi-person households" %
                len(multi_person_households))

    macro_settings = joint_tour_frequency_settings.get('joint_tour_frequency_macros', None)

    assign_time_window_overlap(multi_person_households, persons)

    if macro_settings:
        expressions.assign_columns(
            df=multi_person_households,
            model_settings=macro_settings,
            configs_dir=configs_dir,
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

    tracing.print_summary('joint_tour_frequency', choices, value_counts=True)

    multi_person_households['joint_tour_frequency'] = choices
    # tracing.trace_df(multi_person_households, '%s.DUMP.multi_person_households' %
    #                  trace_label, transpose=False, slicer='NONE')

    # reindex since we are working with a subset of households
    choices = choices.reindex(households.index)

    # add joint_tour_frequency column to households
    households['joint_tour_frequency'] = choices
    pipeline.replace_table("households", households)

    # - create atwork_subtours based on atwork_subtour_frequency choice names
    multi_person_households = households[households.PERSONS > 1]
    assert not multi_person_households.joint_tour_frequency.isnull().any()

    # alts = inject.get_injectable('joint_tour_frequency_alts')
    # joint_tours = process_joint_tours(multi_person_households, alts)
    #
    # tours = pipeline.extend_table("joint_tours", joint_tours)
    # tracing.register_traceable_table('joint_tours', joint_tours)
    # pipeline.get_rn_generator().add_channel(joint_tours, 'joint_tours')

    if trace_hh_id:
        trace_columns = ['joint_tour_frequency']
        tracing.trace_df(multi_person_households,
                         label="joint_tour_frequency.households",
                         # columns=trace_columns,
                         warn_if_empty=True)

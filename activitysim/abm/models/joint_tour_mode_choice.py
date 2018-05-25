# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core.util import force_garbage_collect

from activitysim.core.util import assign_in_place

from .util.mode import get_segment_and_unstack
from .util.mode import mode_choice_simulate
from .util.mode import annotate_preprocessors


logger = logging.getLogger(__name__)


@inject.injectable()
def joint_tour_mode_choice_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'joint_tour_mode_choice.yaml')


@inject.step()
def joint_tour_mode_choice(
        tours,
        persons_merged,
        tour_mode_choice_spec,
        joint_tour_mode_choice_settings,
        tour_mode_choice_settings,
        skim_dict, skim_stack,
        chunk_size,
        trace_hh_id):
    """
    Tour mode choice simulate
    """

    trace_label = 'joint_tour_mode_choice'

    tours = tours.to_frame()
    joint_tours = tours[tours.tour_category == 'joint']

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    persons_merged = persons_merged.to_frame()

    nest_spec = config.get_logit_model_settings(tour_mode_choice_settings)
    constants = config.get_model_constants(tour_mode_choice_settings)

    logger.info("Running joint_tour_mode_choice with %d tours" % joint_tours.shape[0])

    tracing.print_summary('%s tour_type' % trace_label, joint_tours.tour_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(tour_mode_choice_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    joint_tours_merged = pd.merge(joint_tours, persons_merged, left_on='person_id',
                                  right_index=True, how='left')

    # setup skim keys
    orig_col_name = 'TAZ'
    dest_col_name = 'destination'
    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col_name, right_key=dest_col_name,
                                             skim_key='out_period')
    dot_skim_stack_wrapper = skim_stack.wrap(left_key=dest_col_name, right_key=orig_col_name,
                                             skim_key='in_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
    }

    locals_dict = {
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name
    }
    locals_dict.update(constants)

    annotations = annotate_preprocessors(
        joint_tours_merged, locals_dict, skims,
        joint_tour_mode_choice_settings, trace_label)

    choices_list = []

    for tour_type, segment in joint_tours_merged.groupby('tour_type'):

        # if tour_type != 'work':
        #     continue

        logger.info("joint_tour_mode_choice tour_type '%s' (%s tours)" %
                    (tour_type, len(segment.index), ))

        # named index so tracing knows how to slice
        assert segment.index.name == 'tour_id'

        spec = get_segment_and_unstack(tour_mode_choice_spec, tour_type)

        if trace_hh_id:
            tracing.trace_df(spec, tracing.extend_trace_label(trace_label, 'spec.%s' % tour_type),
                             slicer='NONE', transpose=False)

        choices = mode_choice_simulate(
            segment,
            skims=skims,
            spec=spec,
            constants=constants,
            nest_spec=nest_spec,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_type),
            trace_choice_name='joint_tour_mode_choice')

        tracing.print_summary('joint_tour_mode_choice %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        force_garbage_collect()

    choices = pd.concat(choices_list)

    tracing.print_summary('joint_tour_mode_choice all tour type choices',
                          choices, value_counts=True)

    joint_tours['mode'] = choices

    assign_in_place(tours, joint_tours[['mode']])
    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(joint_tours,
                         label=tracing.extend_trace_label(trace_label, 'mode'),
                         slicer='tour_id',
                         index_label='tour_id')

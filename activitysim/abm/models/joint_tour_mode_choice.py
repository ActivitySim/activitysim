# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core.util import force_garbage_collect

from .util import expressions

from .util.mode import get_segment_and_unstack

from .mode_choice import _mode_choice_simulate

logger = logging.getLogger(__name__)


@inject.injectable()
def joint_tour_mode_choice_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'joint_tour_mode_choice.yaml')


@inject.step()
def joint_tour_mode_choice(
        joint_tours,
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

    joint_tours_df = joint_tours.to_frame()
    persons_merged = persons_merged.to_frame()

    nest_spec = config.get_logit_model_settings(tour_mode_choice_settings)
    constants = config.get_model_constants(tour_mode_choice_settings)

    logger.info("Running joint_tour_mode_choice with %d tours" % joint_tours_df.shape[0])

    tracing.print_summary('%s tour_type' % trace_label, joint_tours_df.tour_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(tour_mode_choice_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    # - run preprocessor to annotate choosers
    preprocessor_settings = joint_tour_mode_choice_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=joint_tours_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    joint_tours_merged = pd.merge(joint_tours_df, persons_merged, left_on='person_id',
                                  right_index=True, how='left')

    # setup skim keys
    odt_skim_stack_wrapper = skim_stack.wrap(left_key='TAZ', right_key='destination',
                                             skim_key="out_period")
    dot_skim_stack_wrapper = skim_stack.wrap(left_key='destination', right_key='TAZ',
                                             skim_key="in_period")
    od_skims = skim_dict.wrap('TAZ', 'destination')

    choices_list = []

    for tour_type, segment in joint_tours_merged.groupby('tour_type'):

        # if tour_type != 'work':
        #     continue

        logger.info("joint_tour_mode_choice tour_type '%s' (%s tours)" %
                    (tour_type, len(segment.index), ))

        # named index so tracing knows how to slice
        assert segment.index.name == 'joint_tour_id'

        spec = get_segment_and_unstack(tour_mode_choice_spec, tour_type)

        if trace_hh_id:
            tracing.trace_df(spec, tracing.extend_trace_label(trace_label, 'spec.%s' % tour_type),
                             slicer='NONE', transpose=False)

        choices = _mode_choice_simulate(
            segment,
            odt_skim_stack_wrapper=odt_skim_stack_wrapper,
            dot_skim_stack_wrapper=dot_skim_stack_wrapper,
            od_skim_stack_wrapper=od_skims,
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

    if preprocessor_settings:
        # if we annotated joint_tours, then we want a fresh copy before replace_table
        joint_tours_df = joint_tours.to_frame()

    # replace_table rather than add_columns as we want table for tracing.
    joint_tours_df['mode'] = choices
    pipeline.replace_table("joint_tours", joint_tours_df)

    if trace_hh_id:
        tracing.trace_df(joint_tours_df,
                         label=tracing.extend_trace_label(trace_label, 'mode'),
                         slicer='tour_id',
                         index_label='tour_id',
                         warn_if_empty=True)

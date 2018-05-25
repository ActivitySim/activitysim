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


@inject.step()
def atwork_subtour_mode_choice(
        tours,
        persons_merged,
        tour_mode_choice_spec,
        tour_mode_choice_settings,
        skim_dict, skim_stack,
        chunk_size,
        trace_hh_id):
    """
    At-work subtour mode choice simulate
    """

    trace_label = 'atwork_subtour_mode_choice'

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == 'atwork']

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    subtours_merged = \
        pd.merge(subtours, persons_merged.to_frame(),
                 left_on='person_id', right_index=True, how='left')

    nest_spec = config.get_logit_model_settings(tour_mode_choice_settings)
    constants = config.get_model_constants(tour_mode_choice_settings)

    logger.info("Running %s with %d subtours" % (trace_label, subtours_merged.shape[0]))

    tracing.print_summary('%s tour_type' % trace_label,
                          subtours_merged.tour_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(tour_mode_choice_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    # setup skim keys
    orig_col_name = 'workplace_taz'
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
        subtours_merged, locals_dict, skims,
        tour_mode_choice_settings, trace_label)

    spec = get_segment_and_unstack(tour_mode_choice_spec, segment='atwork')

    if trace_hh_id:
        tracing.trace_df(spec, tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    choices = mode_choice_simulate(
        subtours_merged,
        skims=skims,
        spec=spec,
        constants=constants,
        nest_spec=nest_spec,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='tour_mode_choice')

    tracing.print_summary('%s choices' % trace_label, choices, value_counts=True)

    assign_in_place(tours, choices.to_frame('mode'))
    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(tours[tours.tour_category == 'atwork'],
                         label=tracing.extend_trace_label(trace_label, 'mode'),
                         slicer='tour_id',
                         index_label='tour_id')

    force_garbage_collect()

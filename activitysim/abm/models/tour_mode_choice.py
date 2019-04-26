# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core.mem import force_garbage_collect
from activitysim.core.util import assign_in_place

from .util.mode import tour_mode_choice_spec
from .util.mode import run_tour_mode_choice_simulate

logger = logging.getLogger(__name__)

"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@inject.step()
def tour_mode_choice_simulate(tours, persons_merged,
                              skim_dict, skim_stack,
                              chunk_size,
                              trace_hh_id):
    """
    Tour mode choice simulate
    """
    trace_label = 'tour_mode_choice'
    model_settings = config.read_model_settings('tour_mode_choice.yaml')

    spec = tour_mode_choice_spec(model_settings)

    primary_tours = tours.to_frame()

    assert not (primary_tours.tour_category == 'atwork').any()

    persons_merged = persons_merged.to_frame()

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    logger.info("Running %s with %d tours" % (trace_label, primary_tours.shape[0]))

    tracing.print_summary('tour_types',
                          primary_tours.tour_type, value_counts=True)

    primary_tours_merged = pd.merge(primary_tours, persons_merged, left_on='person_id',
                                    right_index=True, how='left', suffixes=('', '_r'))

    # setup skim keys
    orig_col_name = 'TAZ'
    dest_col_name = 'destination'
    out_time_col_name = 'start'
    in_time_col_name = 'end'
    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col_name, right_key=dest_col_name,
                                             skim_key='out_period')
    dot_skim_stack_wrapper = skim_stack.wrap(left_key=dest_col_name, right_key=orig_col_name,
                                             skim_key='in_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name,
        'out_time_col_name': out_time_col_name,
        'in_time_col_name': in_time_col_name
    }

    choices_list = []
    for tour_type, segment in primary_tours_merged.groupby('tour_type'):

        logger.info("tour_mode_choice_simulate tour_type '%s' (%s tours)" %
                    (tour_type, len(segment.index), ))

        # name index so tracing knows how to slice
        assert segment.index.name == 'tour_id'

        choices = run_tour_mode_choice_simulate(
            segment,
            spec, tour_type, model_settings,
            skims=skims,
            constants=constants,
            nest_spec=nest_spec,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_type),
            trace_choice_name='tour_mode_choice')

        tracing.print_summary('tour_mode_choice_simulate %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        force_garbage_collect()

    choices = pd.concat(choices_list)

    tracing.print_summary('tour_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    # so we can trace with annotations
    primary_tours['tour_mode'] = choices

    # but only keep mode choice col
    all_tours = tours.to_frame()
    # uncomment to save annotations to table
    # assign_in_place(all_tours, annotations)
    assign_in_place(all_tours, choices.to_frame('tour_mode'))

    pipeline.replace_table("tours", all_tours)

    if trace_hh_id:
        tracing.trace_df(primary_tours,
                         label=tracing.extend_trace_label(trace_label, 'tour_mode'),
                         slicer='tour_id',
                         index_label='tour_id',
                         warn_if_empty=True)

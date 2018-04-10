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
from .util.overlap import hh_time_window_overlap
from .util.tour_frequency import process_joint_tours

logger = logging.getLogger(__name__)



def get_stop_frequency_spec(tour_type):

    configs_dir = inject.get_injectable('configs_dir')
    file_name = 'stop_frequency_%s.csv' % tour_type
    return simulate.read_model_spec(configs_dir, file_name)


@inject.injectable()
def stop_frequency_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'stop_frequency.yaml')


# @inject.injectable()
# def stop_frequency_alternatives(configs_dir):
#     # alt file for building trips even though simulation is simple_simulate not interaction_simulate
#     f = os.path.join(configs_dir, 'stop_frequency_alternatives.csv')
#     df = pd.read_csv(f, comment='#')
#     df.set_index('alt', inplace=True)
#     return df


@inject.step()
def stop_frequency(
        tours, tours_merged,
        stop_frequency_settings,
        skim_dict, skim_stack,
        chunk_size,
        trace_hh_id):
    """
    stop frequency
    """

    trace_label = 'stop_frequency'

    tours = tours.to_frame()

    tours_merged = tours_merged.to_frame()
    tours_merged = tours_merged[tours_merged.tour_type == 'work']

    nest_spec = config.get_logit_model_settings(stop_frequency_settings)
    constants = config.get_model_constants(stop_frequency_settings)

    # - run preprocessor to annotate tours_merged
    preprocessor_settings = stop_frequency_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        # FIXME tours should maybe have a permanent origin field?
        tours_merged['origin'] = \
            tours_merged.workplace_taz.where((tours_merged.tour_category == 'subtour'),
                                             tours_merged.TAZ)

        od_skim_stack_wrapper = skim_dict.wrap('origin', 'destination')
        skims = [od_skim_stack_wrapper]

        locals_dict = {
            "od_skims": od_skim_stack_wrapper
        }
        if constants is not None:
            locals_dict.update(constants)

        simulate.add_skims(tours_merged, skims)

        # this should be pre-slice as some expressions may count tours by type
        annotations = expressions.compute_columns(
            df=tours_merged,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

        #print annotations

        assign_in_place(tours_merged, annotations)

    choices_list = []
    for segment_type, choosers in tours_merged.groupby('segment_type'):

        spec = get_stop_frequency_spec(segment_type)

        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name='stops')

        # convert indexes to alternative names
        choices = pd.Series(spec.columns[choices.values], index=choices.index)

        tracing.print_summary('tour_mode_choice_simulate %s choices' % segment_type,
                              choices, value_counts=True)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    #bug reindex since we ran model on a subset of households
    choices = choices.reindex(tours_merged.index)

    assign_in_place(tours, choices.to_frame('stop_frequency'))
    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label="stop_frequency",
                         slicer='person_id',
                         columns=None,
                         warn_if_empty=True)

        tracing.trace_df(annotations,
                         label="stop_frequency.annotations",
                         columns=None,
                         warn_if_empty=True)

        tracing.trace_df(tours_merged,
                         label="stop_frequency.tours_merged",
                         slicer='person_id',
                         columns=None,
                         warn_if_empty=True)

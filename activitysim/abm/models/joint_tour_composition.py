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


from .util.overlap import hh_time_window_overlap


logger = logging.getLogger(__name__)


@inject.injectable()
def joint_tour_composition_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'joint_tour_composition.csv')


def add_null_results(trace_label, tours):
    logger.info("Skipping %s: add_null_results" % trace_label)
    tours['composition'] = np.nan
    pipeline.replace_table("tours", tours)


@inject.step()
def joint_tour_composition(
        tours, households, persons,
        joint_tour_composition_spec,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the makeup of the travel party (adults, children, or mixed).
    """
    trace_label = 'joint_tour_composition'

    model_settings = config.read_model_settings('joint_tour_composition.yaml')

    tours = tours.to_frame()
    joint_tours = tours[tours.tour_category == 'joint']

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        add_null_results(trace_label, tours)
        return

    # - only interested in households with joint_tours
    households = households.to_frame()
    households = households[households.num_hh_joint_tours > 0]

    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(households.index)]

    logger.info("Running joint_tour_composition with %d joint tours" % joint_tours.shape[0])

    # - run preprocessor
    preprocessor_settings = model_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_dict = {
            'persons': persons,
            'hh_time_window_overlap': hh_time_window_overlap
        }

        expressions.assign_columns(
            df=households,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    joint_tours_merged = pd.merge(joint_tours, households,
                                  left_on='household_id', right_index=True, how='left')

    # - simple_simulate

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choices = simulate.simple_simulate(
        choosers=joint_tours_merged,
        spec=joint_tour_composition_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='composition')

    # convert indexes to alternative names
    choices = pd.Series(joint_tour_composition_spec.columns[choices.values], index=choices.index)

    # add composition column to tours
    joint_tours['composition'] = choices

    assign_in_place(tours, joint_tours[['composition']])
    pipeline.replace_table("tours", tours)

    tracing.print_summary('joint_tour_composition', joint_tours.composition,
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(joint_tours,
                         label="joint_tour_composition.joint_tours",
                         slicer='household_id')

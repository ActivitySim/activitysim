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
from activitysim.core.util import reindex


logger = logging.getLogger(__name__)


@inject.injectable()
def joint_party_composition_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'joint_party_composition.csv')


@inject.injectable()
def joint_party_composition_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'joint_party_composition.yaml')


def tour_person_count(exp, tours, persons):
    return reindex(persons.query(exp).groupby('household_id').size(), tours.household_id).fillna(0)


@inject.step()
def joint_party_composition(
        joint_tours, households,
        joint_party_composition_spec,
        joint_party_composition_settings,
        configs_dir,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """
    trace_label = 'joint_tour_party_composition'

    joint_tours = joint_tours.to_frame()
    households = households.to_frame()

    joint_tours_merged = pd.merge(joint_tours, households,
                                  left_on='household_id', right_index=True, how='left')

    logger.info("Running joint_party_composition with %d joint tours" %
                joint_tours.shape[0])

    macro_settings = joint_party_composition_settings.get('joint_party_composition_macros', None)

    macro_helpers = {'tour_person_count': tour_person_count}

    if macro_settings:
        expressions.assign_columns(
            df=joint_tours_merged,
            model_settings=macro_settings,
            locals_dict=macro_helpers,
            trace_label=trace_label)

    nest_spec = config.get_logit_model_settings(joint_party_composition_settings)
    constants = config.get_model_constants(joint_party_composition_settings)

    choices = simulate.simple_simulate(
        joint_tours_merged,
        spec=joint_party_composition_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='composition')

    # convert indexes to alternative names
    choices = pd.Series(joint_party_composition_spec.columns[choices.values], index=choices.index)

    # add joint_tour_frequency column to households
    # reindex since we are working with a subset of households
    joint_tours['composition'] = choices.reindex(joint_tours.index)
    pipeline.replace_table("joint_tours", joint_tours)

    tracing.print_summary('joint_party_composition', joint_tours.composition,
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(joint_tours,
                         label="joint_party_composition.joint_tours",
                         slicer='household_id',
                         warn_if_empty=True)

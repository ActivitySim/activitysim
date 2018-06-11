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
    This model predicts the frequency of making fully joint trips (see the
    alternatives above).
    """
    trace_label = 'joint_tour_frequency'

    # - only interested in households with more than one cdap travel_active person
    households = households.to_frame()
    multi_person_households = households[households.num_travel_active > 1].copy()

    # - only interested in persons in multi_person_households
    # FIXME - gratuitous pathological efficiency move, just let yaml specify persons?
    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(multi_person_households.index)]

    logger.info("Running joint_tour_frequency with %d multi-person households" %
                multi_person_households.shape[0])

    # - preprocessor
    preprocessor_settings = joint_tour_frequency_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_dict = {
            'persons': persons,
            'hh_time_window_overlap': hh_time_window_overlap
        }

        expressions.assign_columns(
            df=multi_person_households,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    # - simple_simulate

    nest_spec = config.get_logit_model_settings(joint_tour_frequency_settings)
    constants = config.get_model_constants(joint_tour_frequency_settings)

    choices = simulate.simple_simulate(
        choosers=multi_person_households,
        spec=joint_tour_frequency_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='joint_tour_frequency')

    # convert indexes to alternative names
    choices = pd.Series(joint_tour_frequency_spec.columns[choices.values], index=choices.index)

    # - create joint_tours based on joint_tour_frequency choices

    # - we need a person_id in order to generate the tour index (and for register_traceable_table)
    # - but we don't know the tour participants yet
    # - so we arbitrarily choose the first person in the household
    # - to be point person for the purpose of generating an index and setting origin
    temp_point_persons = persons.loc[persons.PNUM == 1]
    temp_point_persons['person_id'] = temp_point_persons.index
    temp_point_persons = temp_point_persons.set_index('household_id')
    temp_point_persons = temp_point_persons[['person_id', 'home_taz']]

    joint_tours = \
        process_joint_tours(choices, joint_tour_frequency_alternatives, temp_point_persons)

    tours = pipeline.extend_table("tours", joint_tours)

    tracing.register_traceable_table('tours', joint_tours)
    pipeline.get_rn_generator().add_channel(joint_tours, 'tours')

    # - annotate households
    # add joint_tour_frequency and num_hh_joint_tours columns to households
    # reindex since we ran model on a subset of households
    households['joint_tour_frequency'] = choices.reindex(households.index)

    households['num_hh_joint_tours'] = joint_tours.groupby('household_id').size().\
        reindex(households.index).fillna(0).astype(np.int8)

    pipeline.replace_table("households", households)

    tracing.print_summary('joint_tour_frequency', households.joint_tour_frequency,
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(households,
                         label="joint_tour_frequency.households")

        tracing.trace_df(joint_tours,
                         label="joint_tour_frequency.joint_tours",
                         slicer='household_id')

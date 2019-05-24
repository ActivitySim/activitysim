# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util import expressions
from .util.overlap import hh_time_window_overlap
from .util.tour_frequency import process_joint_tours

logger = logging.getLogger(__name__)


@inject.step()
def joint_tour_frequency(
        households, persons,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the frequency of making fully joint trips (see the
    alternatives above).
    """
    trace_label = 'joint_tour_frequency'
    model_settings = config.read_model_settings('joint_tour_frequency.yaml')
    model_spec = simulate.read_model_spec(file_name='joint_tour_frequency.csv')

    alternatives = simulate.read_model_alts(
        config.config_file_path('joint_tour_frequency_alternatives.csv'), set_index='alt')

    # - only interested in households with more than one cdap travel_active person and
    # - at least one non-preschooler
    households = households.to_frame()
    multi_person_households = households[households.participates_in_jtf_model].copy()

    # - only interested in persons in multi_person_households
    # FIXME - gratuitous pathological efficiency move, just let yaml specify persons?
    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(multi_person_households.index)]

    logger.info("Running joint_tour_frequency with %d multi-person households" %
                multi_person_households.shape[0])

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
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

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choices = simulate.simple_simulate(
        choosers=multi_person_households,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='joint_tour_frequency')

    # convert indexes to alternative names
    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)

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
        process_joint_tours(choices, alternatives, temp_point_persons)

    tours = pipeline.extend_table("tours", joint_tours)

    tracing.register_traceable_table('tours', joint_tours)
    pipeline.get_rn_generator().add_channel('tours', joint_tours)

    # - annotate households
    # add joint_tour_frequency and num_hh_joint_tours columns to households
    # reindex since we ran model on a subset of households
    households['joint_tour_frequency'] = choices.reindex(households.index).fillna('').astype(str)

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

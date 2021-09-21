# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import expressions

from activitysim.core import assign
from activitysim.core import los

from activitysim.core.util import assign_in_place

from .util.mode import mode_choice_simulate
from .util import estimation


logger = logging.getLogger(__name__)


@inject.step()
def vehicle_choice(
        households,
        households_merged,
        vehicles,
        chunk_size,
        trace_hh_id):
    """
    
    """
    trace_label = 'vehicle_choice'
    model_settings_file_name = 'vehicle_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    choice_column_name = 'vehicle_type'

    estimator = estimation.manager.begin_estimation('vehicle_type')

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients_df, trace_label)
    
    constants = config.get_model_constants(model_settings)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients_df)

    # merge vehicles onto households, index will be vehicle_id
    choosers = inject.merge_tables('households_merged', ['households_merged', 'vehicles'])

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        if constants is not None:
            locals_dict.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    logger.info("Running %s with %d households", trace_label, len(choosers))

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='vehicle_type',
        estimator=estimator)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'households', 'vehicle_choice')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    vehicles['vehicle_type'] = choices

    # - annotate households table
    expressions.assign_columns(
        df=households,
        model_settings=model_settings.get('annotate_households'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
    pipeline.replace_table("households", households)

    # - annotate persons table
    households = persons.to_frame()
    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_households'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
    pipeline.replace_table("persons", persons)

    tracing.print_summary('vehicle_choice', households.vehicle_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(households,
                         label='vehicle_choice',
                         warn_if_empty=True)

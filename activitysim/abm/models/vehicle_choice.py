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
from activitysim.core import logit
from activitysim.core import assign
from activitysim.core import los

from activitysim.core.util import assign_in_place

from .util.mode import mode_choice_simulate
from .util import estimation


logger = logging.getLogger(__name__)


@inject.step()
def vehicle_choice(
        persons,
        households,
        vehicles,
        vehicles_merged,
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
    choosers = vehicles_merged.to_frame()

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

    # run logit choices
    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='vehicle_type',
        estimator=estimator)

    if isinstance(choices, pd.Series):
        choices = choices.to_frame('choice')

    choices.rename(columns={'logsum': logsum_column_name,
                            'choice': choice_column_name},
                   inplace=True)

    alts = model_spec.columns
    choices[choice_column_name] = \
        choices[choice_column_name].map(dict(list(zip(list(range(len(alts))), alts))))

    # append probabilistic attributes to veh types
    probs_spec_file = model_settings.get("PROBS_SPEC", None)
    if probs_spec_file is not None:

        # name of first column must be "vehicle_type"
        probs_spec = pd.read_csv(
            config.config_file_path(probs_spec_file), comment='#')

        # left join vehicles to probs
        choosers = pd.merge(
            choices.reset_index(), probs_spec,
            on=choice_column_name,
            how='left').set_index('vehicle_id')
        del choosers[choice_column_name]

        # probs should sum to 1 with residual probs resulting in choice of 'fail'
        chooser_probs = choosers.div(choosers.sum(axis=1), axis=0).fillna(0)
        chooser_probs['fail'] = 1 - chooser_probs.sum(axis=1).clip(0, 1)

        # make probabilistic choices
        prob_choices, rands = logit.make_choices(chooser_probs, trace_label=trace_label, trace_choosers=choosers)

        # convert alt choice index to vehicle type attribute
        prob_choices = chooser_probs.columns[prob_choices.values].to_series(index=prob_choices.index)
        failed = (prob_choices == chooser_probs.columns.get_loc('fail'))
        prob_choices = prob_choices.where(~failed, "NOT CHOSEN")

        # add new attribute to logit choice vehicle types
        choices[choice_column_name] = choices[choice_column_name] + '_' + prob_choices

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'households', 'vehicle_choice')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # update vehicles table
    vehicles = vehicles.to_frame()
    assign_in_place(vehicles, choices)
    pipeline.replace_table("vehicles", vehicles)

    # - annotate households table
    households = households.to_frame()
    expressions.assign_columns(
        df=households,
        model_settings=model_settings.get('annotate_households'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
    pipeline.replace_table("households", households)

    # - annotate persons table
    persons = persons.to_frame()
    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_households'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
    pipeline.replace_table("persons", persons)

    tracing.print_summary('vehicle_choice', vehicles.vehicle_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(vehicles,
                         label='vehicle_choice',
                         warn_if_empty=True)

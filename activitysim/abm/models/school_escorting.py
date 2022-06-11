# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import expressions
from activitysim.core import los

import pandas as pd
import numpy as np
import warnings

from .util import estimation


logger = logging.getLogger(__name__)


def determine_escorting_paricipants(choosers, persons, model_settings):
    """
    Determining which persons correspond to chauffer 1..n and escortee 1..n.
    Chauffers are those with the highest weight given by:
     weight = 100 * person type +  10*gender + 1*(age > 25)
    and escortees are selected youngest to oldest.
    """

    num_chaperones = model_settings['NUM_CHAPERONES']
    num_escortees = model_settings['NUM_ESCORTEES']

    # is this cut correct?
    escortees = persons[persons.is_student & (persons.age < 16)]
    households_with_escortees = escortees['household_id']

    persontype_weight = 100
    gender_weight = 10
    age_weight = 1

    # can we move all of these to a config file?
    chaperones = persons[
        (persons.age > 18) & persons.household_id.isin(households_with_escortees)]

    chaperones['chaperone_weight'] = (
        (persontype_weight * chaperones['ptype'])
        + (gender_weight * np.where(chaperones['sex'] == 1, 1, 0))
        + (age_weight * np.where(chaperones['age'] > 25, 1, 0))
    )

    chaperones['chaperone_num'] = chaperones.sort_values(
        'chaperone_weight', ascending=False).groupby('household_id').cumcount() + 1
    escortees['escortee_num'] = escortees.sort_values(
        'age', ascending=True).groupby('household_id').cumcount() + 1

    participant_columns = []
    for i in range(1, num_chaperones+1):
        choosers['chauf_id'+str(i)] = chaperones[chaperones['chaperone_num'] == i] \
            .reset_index() \
            .set_index('household_id') \
            .reindex(choosers.index)['person_id']
        participant_columns.append('chauf_id'+str(i))
    for i in range(1, num_escortees+1):
        choosers['child_id'+str(i)] = escortees[escortees['escortee_num'] == i] \
            .reset_index() \
            .set_index('household_id') \
            .reindex(choosers.index)['person_id']
        participant_columns.append('child_id'+str(i))

    return choosers, participant_columns


def construct_alternatives(choosers, model_settings):
    """
    Constructing alternatives for school escorting
    """
    # FIXME: just reading alts right now instead of constructing
    alts = simulate.read_model_alts(model_settings['ALTS'], set_index='Alt')
    return alts


def run_school_escorting(choosers, households, model_settings, alts, trace_label, chunk_size, trace_hh_id):

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)
    locals_dict = {}
    locals_dict.update(constants)

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # for stage in ['outbound', 'inbound', 'outbound_cond']:
    for stage in ['outbound', 'inbound']:
        stage_trace_label = trace_label + '_' + stage
        estimator = estimation.manager.begin_estimation('school_escorting_' + stage)

        model_spec_raw = simulate.read_model_spec(file_name=model_settings[stage.upper() + '_SPEC'])
        coefficients_df = simulate.read_model_coefficients(file_name=model_settings[stage.upper() + '_COEFFICIENTS'])
        model_spec = simulate.eval_coefficients(model_spec_raw, coefficients_df, estimator)

        locals_dict.update(coefficients_df)

        logger.info("Running %s with %d households", stage_trace_label, len(choosers))

        preprocessor_settings = model_settings.get('preprocessor_' + stage, None)
        if preprocessor_settings:
            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=stage_trace_label)

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        choosers.to_csv('school_escorting_choosers_' + stage + '.csv')

        log_alt_losers = config.setting('log_alt_losers', False)

        choices = interaction_simulate(
            choosers=choosers,
            alternatives=alts,
            spec=model_spec,
            log_alt_losers=log_alt_losers,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            trace_label=stage_trace_label,
            trace_choice_name='school_escorting_' + 'stage',
            estimator=estimator)

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(choices, 'households', 'school_escorting_' + stage)
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # no need to reindex as we used all households
        escorting_choice = 'school_escorting_' + stage
        households[escorting_choice] = choices
        # also adding to choosers table
        choosers[escorting_choice] = choices

        stage_alts = alts.copy()


        # should this tracing be done for every step? - I think so...
        tracing.print_summary(escorting_choice, households[escorting_choice], value_counts=True)

        if trace_hh_id:
            tracing.trace_df(households,
                             label=escorting_choice,
                             warn_if_empty=True)

    return households


def add_prev_choices_to_choosers(choosers, choices, alts, stage):
    # adding choice details to chooser table
    escorting_choice = 'school_escorting_' + stage
    choosers[escorting_choice] = choices

    stage_alts = alts.copy()
    stage_alts.columns = stage_alts.columns + '_' + stage

    choosers = choosers.reset_index().merge(
        stage_alts,
        how='left',
        left_on=escorting_choice,
        right_on=stage_alts.index.name).set_index('household_id')

    return choosers


@inject.step()
def school_escorting(households,
                     households_merged,
                     persons,
                     chunk_size,
                     trace_hh_id):
    """
    The school escorting model determines whether children are dropped-off at or
    picked-up from school, simultaneously with the driver responsible for
    chauffeuring the children, which children are bundled together on half-tours,
    and the type of tour (pure escort versus rideshare).
    """
    trace_label = 'school_escorting_simulate'
    model_settings_file_name = 'school_escorting.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    # model_spec_raw = simulate.read_model_spec(file_name=model_settings['OUTBOUND_SPEC'])
    # coefficients_df = simulate.read_model_coefficients(file_name=model_settings['OUTBOUND_COEFFICIENTS'])
    # model_spec = simulate.eval_coefficients(model_spec_raw, coefficients_df, estimator)

    # nest_spec = config.get_logit_model_settings(model_settings)
    # constants = config.get_model_constants(model_settings)

    persons = persons.to_frame()
    households = households.to_frame()
    households_merged = households_merged.to_frame()

    alts = construct_alternatives(households_merged, model_settings)

    households_merged, participant_columns = determine_escorting_paricipants(
        households_merged, persons, model_settings)

    # households = run_school_escorting(choosers, households, model_settings, alts, trace_label, chunk_size, trace_hh_id)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)
    locals_dict = {}
    locals_dict.update(constants)

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    school_escorting_stages = ['outbound', 'inbound', 'outbound_cond']
    # school_escorting_stages = ['outbound', 'inbound']
    for stage_num, stage in enumerate(school_escorting_stages):
        stage_trace_label = trace_label + '_' + stage
        estimator = estimation.manager.begin_estimation('school_escorting_' + stage)

        model_spec_raw = simulate.read_model_spec(file_name=model_settings[stage.upper() + '_SPEC'])
        coefficients_df = simulate.read_model_coefficients(file_name=model_settings[stage.upper() + '_COEFFICIENTS'])
        model_spec = simulate.eval_coefficients(model_spec_raw, coefficients_df, estimator)

        # reduce memory by limiting columns if selected columns are supplied
        chooser_columns = model_settings.get('SIMULATE_CHOOSER_COLUMNS', None)
        if chooser_columns is not None:
            chooser_columns = chooser_columns + participant_columns
            choosers = households_merged[chooser_columns]
        else:
            choosers = households_merged

        # add previous data to stage
        if stage_num >= 1:
            choosers = add_prev_choices_to_choosers(
                choosers, choices, alts, school_escorting_stages[stage_num-1])

        choosers.to_csv('school_escorting_choosers_' + stage + '.csv')

        locals_dict.update(coefficients_df)

        logger.info("Running %s with %d households", stage_trace_label, len(choosers))

        preprocessor_settings = model_settings.get('preprocessor_' + stage, None)
        if preprocessor_settings:
            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=stage_trace_label)

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        choosers.to_csv('school_escorting_choosers_' + stage + '.csv')

        log_alt_losers = config.setting('log_alt_losers', False)

        choices = interaction_simulate(
            choosers=choosers,
            alternatives=alts,
            spec=model_spec,
            log_alt_losers=log_alt_losers,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            trace_label=stage_trace_label,
            trace_choice_name='school_escorting_' + 'stage',
            estimator=estimator)

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(choices, 'households', 'school_escorting_' + stage)
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # no need to reindex as we used all households
        escorting_choice = 'school_escorting_' + stage
        households[escorting_choice] = choices

        # should this tracing be done for every step? - I think so...
        tracing.print_summary(escorting_choice, households[escorting_choice], value_counts=True)

        if trace_hh_id:
            tracing.trace_df(households,
                             label=escorting_choice,
                             warn_if_empty=True)

    pipeline.replace_table("households", households)

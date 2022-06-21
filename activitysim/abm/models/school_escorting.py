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

from activitysim.core.util import reindex

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
    escortees = persons[persons.is_student & (persons.age < 16) & (persons.cdap_activity == 'M')]
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


def determine_child_order_specific_attributes(row):
    child_order = row['child_order']
    first_child_num = str(child_order[0] + 1)

    # origin is just the home zone id
    row['origin'] = row['school_origin_child' + first_child_num]
    row['destination'] = row['school_destination_child' + first_child_num]
    row['start'] = row['school_start_child' + first_child_num]

    time_home_to_school = row['time_home_to_school' + first_child_num]
    # FIXME hardcoded mins per time bin
    row['end'] = row['start'] + int(time_home_to_school / 30)

    return row


def add_school_escorting_type_to_tours_table(escort_bundles, tours):
    school_tour = ((tours.tour_type == 'school') & (tours.tour_num == 1))

    for direction in ['outbound', 'inbound']:
        for escort_type in ['ride_share', 'pure_escort']:
            bundles = escort_bundles[
                (escort_bundles.direction == direction)
                & (escort_bundles.escort_type == escort_type)
            ]
            for child_num in range(1,4):
                i = str(child_num)
                filter = (school_tour & tours['person_id'].isin(bundles['bundle_child' + i]))
                tours.loc[filter, 'school_esc_' + direction] = escort_type

    return tours


def create_pure_escort_tours(bundles, tours):
    # creating home to school tour for chauffers making pure escort tours
    # ride share tours are already created since they go off the mandatory tour

    # FIXME: can I just move all of this logic to a csv and annotate??
    pe_tours = bundles[bundles['escort_type'] == 'pure_escort']

    pe_tours = pe_tours.apply(lambda row: determine_child_order_specific_attributes(row), axis=1)

    pe_tours['person_id'] = pe_tours['chauf_id']
    pe_tours['tour_type_count'] = 1
    pe_tours['tour_type_num'] = 1
    pe_tours['tour_num'] = 1
    pe_tours['tour_category'] = 'non_mandatory'
    pe_tours['number_of_participants'] = 1
    pe_tours['tour_type'] = 'escort'
    # FIXME will have to calculate all these after
    pe_tours['tour_count'] = 1
    pe_tours['tdd'] = -1
    pe_tours['duration'] = pe_tours['end'] - pe_tours['start']
    pe_tours['duration'] = pe_tours['end'] - pe_tours['start']
    pe_tours['school_esc_outbound'] = np.where(pe_tours['direction'] == 'outbound', 'pure_escort', pd.NA)
    pe_tours['school_esc_inbound'] = np.where(pe_tours['direction'] == 'inbound', 'pure_escort', pd.NA)

    pe_tours = pe_tours[tours.columns]

    return pe_tours


def create_school_escorting_bundles_table(choosers, tours, stage):
    choosers.to_csv('school_escorting_tour_choosers_' + stage + '.csv')

    # making a table of bundles
    choosers = choosers.reset_index()
    choosers = choosers.loc[choosers.index.repeat(choosers['nbundles'])]

    bundles = pd.DataFrame()
    bundles.index = choosers.index
    bundles['household_id'] = choosers['household_id']
    bundles['direction'] = 'outbound' if 'outbound' in stage else 'inbound'
    bundles['bundle_num'] = choosers.groupby('household_id').cumcount() + 1

    # initialize values
    bundles['chauf_type_num'] = 0
    # bundles['first_school_start'] = 999
    # bundles['first_school_start'] = -999

    # getting bundle school start times and locations
    school_tours = tours[(tours.tour_type == 'school') & (tours.tour_num == 1)]

    school_starts = school_tours.set_index('person_id').start
    school_ends = school_tours.set_index('person_id').end
    school_destinations = school_tours.set_index('person_id').destination
    school_origins = school_tours.set_index('person_id').origin

    for child_num in range(1,4):
        i = str(child_num)
        bundles['bundle_child' + i] = np.where(choosers['bundle' + i] == bundles['bundle_num'], choosers['child_id' + i], -1)
        bundles['chauf_type_num'] = np.where((choosers['bundle' + i] == bundles['bundle_num']), choosers['chauf' + i], bundles['chauf_type_num'])
        bundles['time_home_to_school' + i] = np.where((choosers['bundle' + i] == bundles['bundle_num']), choosers['time_home_to_school' + i], np.NaN)

        bundles['school_destination_child' + i] = reindex(school_destinations, bundles['bundle_child' + i])
        bundles['school_origin_child' + i] = reindex(school_origins, bundles['bundle_child' + i])
        bundles['school_start_child' + i] = reindex(school_starts, bundles['bundle_child' + i])
        bundles['school_end_child' + i] = reindex(school_ends, bundles['bundle_child' + i])

        # bundles['first_school_start'] = np.where(bundles['school_start_child' + i] < bundles['first_school_start'], bundles['school_start_child' + i], bundles['first_school_start'])
        # bundles['first_school_end'] = np.where(bundles['school_end_child' + i] > bundles['first_school_end'], bundles['school_end_child' + i], bundles['first_school_end'])

    # FIXME assumes only two chauffeurs
    bundles['chauf_id'] = np.where(bundles['chauf_type_num'] <= 2, choosers['chauf_id1'], choosers['chauf_id2']).astype(int)
    bundles['chauf_num'] = np.where(bundles['chauf_type_num'] <= 2, 1, 2)
    bundles['escort_type'] = np.where(bundles['chauf_type_num'].isin([1,3]), 'ride_share', 'pure_escort')

    school_dist_cols = ['dist_home_to_school' + str(i) for i in range(1,4)]
    bundles['child_order'] = list(bundles[school_dist_cols].values.argsort())

    # getting chauffer mandatory times
    bundles['first_mand_tour_start_time'] = reindex(tours[(tours.tour_type == 'work') & (tours.tour_num == 1)].set_index('person_id').start, bundles['chauf_id'])
    bundles['first_mand_tour_end_time'] = reindex(tours[(tours.tour_type == 'work') & (tours.tour_num == 1)].set_index('person_id').end, bundles['chauf_id'])

    bundles['Alt'] = choosers['Alt']
    bundles['Description'] = choosers['Description']

    return bundles


@inject.step()
def school_escorting(households,
                     households_merged,
                     persons,
                     tours,
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
    tours = tours.to_frame()

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
    escort_bundles = []
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

        if stage_num >= 1:
            choosers['Alt'] = choices
            choosers = choosers.join(alts, how='left', on='Alt')
            bundles = create_school_escorting_bundles_table(choosers[choosers['Alt'] > 1], tours, stage)
            escort_bundles.append(bundles)

    escort_bundles = pd.concat(escort_bundles)
    escort_bundles.sort_values(by=['household_id', 'direction'], ascending=[True, False], inplace=True)
    escort_bundles.to_csv('escort_bundles.csv')

    tours = add_school_escorting_type_to_tours_table(escort_bundles, tours)

    # FIXME should this be called after non-mandatory tour creation?
    pure_escort_tours = create_pure_escort_tours(escort_bundles, tours)
    pure_escort_tours.to_csv('pure_escort_tours.csv')

    pipeline.replace_table("households", households)
    pipeline.replace_table("tours", tours)

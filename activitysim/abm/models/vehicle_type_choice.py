# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np
import itertools
import os

from activitysim.core.interaction_simulate import interaction_simulate
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
from .util import estimation


logger = logging.getLogger(__name__)


def read_vehicle_type_data(model_settings):
    # This is essentially a re-implementation of core/config/base_settings_file_path
    # Should we make this a little more general and move it there?
    # Also called in vehicle allocation model
    file_name = model_settings.get('VEHICLE_TYPE_DATA_FILE')

    if not file_name.lower().endswith('.csv'):
        file_name = '%s.csv' % (file_name, )

    configs_dir = inject.get_injectable('configs_dir')
    configs_dir = [configs_dir] if isinstance(configs_dir, str) else configs_dir

    for dir in configs_dir:
        file_path = os.path.join(dir, file_name)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)

    raise RuntimeError("base_settings_file %s not found" % file_name)


def get_combinatorial_vehicle_alternatives(alts_cats_dict, model_settings):
    alts_fname = model_settings.get('ALTS')
    print(alts_cats_dict)
    cat_cols = list(alts_cats_dict.keys())  # e.g. fuel type, body type, age
    alts_long = pd.DataFrame(
        list(itertools.product(*alts_cats_dict.values())),
        columns=alts_cats_dict.keys()).astype(str)
    alts_wide = pd.get_dummies(alts_long)  # rows will sum to num_cats

    # store alts in primary configs dir
    configs_dirs = inject.get_injectable("configs_dir")
    configs_dirs = [configs_dirs] if isinstance(configs_dirs, str) else configs_dirs

    alts_wide.to_csv(os.path.join(configs_dirs[0], alts_fname), index=True)
    alts_wide = pd.concat([alts_wide,alts_long], axis=1)
    return alts_wide, alts_long


def append_probabilistic_vehtype_type_choices(chooser, probs_spec_file):
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

    return choices


def annotate_vehicle_type_choice(model_settings, trace_label):
    # - annotate households table
    if model_settings.get('annotate_households'):
        households = inject.get_table('households').to_frame()
        expressions.assign_columns(
            df=households,
            model_settings=model_settings.get('annotate_households'),
            trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
        pipeline.replace_table("households", households)

    # - annotate persons table
    if model_settings.get('annotate_persons'):
        persons = inject.get_table('persons').to_frame()
        expressions.assign_columns(
            df=persons,
            model_settings=model_settings.get('annotate_persons'),
            trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))
        pipeline.replace_table("persons", persons)


def iterate_vehicle_type_choice(
        vehicles_merged,
        model_settings,
        model_spec,
        locals_dict,
        estimator,
        chunk_size,
        trace_label):

    # - model settings
    choice_column_name = model_settings.get("CHOICE_COL", 'vehicle_type')
    nest_spec = config.get_logit_model_settings(model_settings)
    vehicle_type_data_file = model_settings.get('VEHICLE_TYPE_DATA_FILE', None)
    probs_spec_file = model_settings.get("PROBS_SPEC", None)
    alts_cats_dict = model_settings.get('combinatorial_alts', False)

    # adding vehicle type data to be available to locals_dict regardless of option
    if vehicle_type_data_file:
        vehicle_type_data = pd.read_csv(
            config.config_file_path(vehicle_type_data_file), comment='#')
        scenario_year = model_settings.get('SCENARIO_YEAR')

        vehicle_type_data['age'] = (1 + scenario_year - vehicle_type_data['vehicle_year']).astype(str)
        # vehicle_type_data.set_index([body_type_col, fuel_type_col, 'age'], inplace=True)
        locals_dict.update({'vehicle_type_data': vehicle_type_data})

    # - Preparing alternatives
    # create alts on-the-fly as cartesian product of categorical values
    if alts_cats_dict:
        # do not include fuel types as alternatives if probability file is supplied
        if probs_spec_file:
            del alts_cats_dict['fuel_type']
        alts_wide, alts_long = get_combinatorial_vehicle_alternatives(alts_cats_dict, model_settings)

        # merge vehicle type data to alternatives if data is provided
        if vehicle_type_data_file and (probs_spec_file is None):

            alts_wide = pd.merge(
                alts_wide, vehicle_type_data,
                how='left',
                on=['body_type', 'fuel_type', 'age'],
                indicator=True)

            # checking to make sure all alternatives have data
            missing_alts = alts_wide.loc[alts_wide._merge == 'left_only',
                ['body_type', 'fuel_type', 'age']]
            assert len(missing_alts) == 0, \
                f"missing vehicle data for alternatives:\n {missing_alts}"
            alts_wide.drop(columns='_merge', inplace=True)

        # converting age to integer to allow interactions in utilities
        alts_wide['age'] = alts_wide['age'].astype(int)

    # - preparing choosers for iterating
    vehicles_merged = vehicles_merged.to_frame()
    vehicles_merged['already_owned_veh'] = ''
    logger.info("Running %s with %d vehicles", trace_label, len(vehicles_merged))
    all_choosers = []
    all_choices = []

    # - Selecting each vehicle in the household sequentially
    # This is necessary to determine and use utility terms that include other
    #   household vehicles
    for veh_num in range(1, vehicles_merged.vehicle_num.max()+1):
        print("veh_num: ", veh_num)
        # merge vehicles onto households, index will be vehicle_id
        choosers = vehicles_merged

        # - preprocessor
        # running preprocessor on entire vehicle table to enumerate vehicle types
        # already owned by the household
        preprocessor_settings = model_settings.get('preprocessor', None)
        if preprocessor_settings:
            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=trace_label)

        # only make choices for vehicles that have not been selected yet
        choosers = choosers[vehicles_merged['vehicle_num'] == veh_num]
        logger.info("Running %s with %d households", trace_label, len(choosers))

        choosers.to_csv(f'choosers_{veh_num}.csv')

        # if there were so many alts that they had to be created programmatically,
        # by combining categorical variables, then the utility expressions should make
        # use of interaction terms to accommodate alt-specific coefficients and constants
        if alts_cats_dict:
            log_alt_losers = config.setting('log_alt_losers', False)
            choices = interaction_simulate(
                choosers=choosers,
                alternatives=alts_wide,
                spec=model_spec,
                log_alt_losers=log_alt_losers,
                locals_d=locals_dict,
                chunk_size=chunk_size,
                trace_label=trace_label,
                trace_choice_name='vehicle_type',
                estimator=estimator)

        # otherwise, "simple simulation" should suffice, with a model spec that enumerates
        # each alternative as a distinct column in the .csv
        else:
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

        choices.rename(columns={'choice': choice_column_name}, inplace=True)

        if alts_cats_dict:
            alts = alts_long[alts_long.columns].apply(
                lambda row: '_'.join(row.values.astype(str)), axis=1).values
        else:
            alts = model_spec.columns
        choices[choice_column_name] = \
            choices[choice_column_name].map(dict(list(zip(list(range(len(alts))), alts))))

        print(choices)

        # STEP II: append probabilistic vehicle type attributes
        if probs_spec_file is not None:
            choices = append_probabilistic_vehtype_type_choices(chooser, probs_spec_file)

        vehicles_merged.loc[choices.index, 'already_owned_veh'] = choices[choice_column_name]
        all_choices.append(choices)
        all_choosers.append(choosers)

    all_choices = pd.concat(all_choices)
    all_choosers = pd.concat(all_choosers)
    return all_choices, all_choosers


@inject.step()
def vehicle_type_choice(
        persons,
        households,
        vehicles,
        vehicles_merged,
        chunk_size,
        trace_hh_id):
    """Assigns a vehicle type to each vehicle in the `vehicles` table.

    If a dictionary of "combinatorial alts" is not specified in
    vehicle_type_choice.yaml config file, then the model specification .csv file
    should contain one column of coefficients for each distinct alternative. This
    format corresponds to ActivitySim's :func:`activitysim.core.simulate.simple_simulate`
    format. Otherwise, this model will construct a table of alternatives, at run time,
    based on all possible combinations of values of the categorical variables enumerated
    as "combinatorial_alts" in the .yaml config. In this case, the model leverages
    ActivitySim's :func:`activitysim.core.interaction_simulate` model design, in which
    the model specification .csv has only one column of coefficients, and the utility
    expressions can turn coefficients on or off based on attributes of either
    the chooser _or_ the alternative.

    As an optional second step, the user may also specify a "PROBS_SPEC" .csv file in
    the main .yaml config, corresponding to a lookup table of additional vehicle
    attributes and probabilities to be sampled and assigned to vehicles after the logit
    choices have been made. The rows of the "PROBS_SPEC" file must be indexed on the
    vehicle type choices assigned in the logit model. These additional attributes are
    concatenated with the selected alternative from the logit model to form a single
    vehicle type name to be stored in the `vehicles` table as the column specified as
    "CHOICE_COL" in the .yaml config.

    The user may also augment the `households` or `persons` tables with new vehicle
    type-based fields specified via expressions in "annotate_households_vehicle_type.csv"
    and "annotate_persons_vehicle_type.csv", respectively.


    Parameters
    ----------
    persons : orca.DataFrameWrapper
    households : orca.DataFrameWrapper
    vehicles : orca.DataFrameWrapper
    vehicles_merged : orca.DataFrameWrapper
    chunk_size : orca.injectable
    trace_hh_id : orca.injectable

    Returns
    -------

    """
    trace_label = 'vehicle_type_choice'
    model_settings_file_name = 'vehicle_type_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    estimator = estimation.manager.begin_estimation('vehicle_type')

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    constants = config.get_model_constants(model_settings)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients_df)

    choices, choosers = iterate_vehicle_type_choice(
        vehicles_merged,
        model_settings,
        model_spec,
        locals_dict,
        estimator,
        chunk_size,
        trace_label)


    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

        # FIXME #interaction_simulate_estimation_requires_chooser_id_in_df_column
        #  shuold we do it here or have interaction_simulate do it?
        # chooser index must be duplicated in column or it will be omitted from interaction_dataset
        # estimation requires that chooser_id is either in index or a column of interaction_dataset
        # so it can be reformatted (melted) and indexed by chooser_id and alt_id
        assert choosers.index.name == 'vehicle_id'
        assert 'vehicle_id' not in choosers.columns
        choosers['vehicle_id'] = choosers.index

        # FIXME set_alt_id - do we need this for interaction_simulate estimation bundle tables?
        estimator.set_alt_id('alt_id')
        estimator.set_chooser_id(choosers.index.name)

        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'vehicles', 'vehicle_type_choice')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # update vehicles table
    vehicles = vehicles.to_frame()
    assign_in_place(vehicles, choices)
    pipeline.replace_table("vehicles", vehicles)

    annotate_vehicle_type_choice(model_settings, trace_label)

    tracing.print_summary('vehicle_type_choice', vehicles.vehicle_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(vehicles,
                         label='vehicle_type_choice',
                         warn_if_empty=True)

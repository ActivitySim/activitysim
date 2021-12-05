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

from .util.mode import mode_choice_simulate
from .util import estimation


logger = logging.getLogger(__name__)


@inject.step()
def vehicle_type_choice(
        persons,
        households,
        vehicles,
        vehicles_merged,
        chunk_size,
        trace_hh_id):
    """Assigns vehicle type to each vehicle
    """
    trace_label = 'vehicle_choice'
    model_settings_file_name = 'vehicle_type_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    # load/create alts on-the-fly as cartesian product of categorical values
    alts_cats_dict = model_settings.get('combinatorial_alts', False)
    if alts_cats_dict:
        alts_fname = model_settings.get('ALTS')
        try:
            alts_wide = config.config_file_path(alts_fname)
        except:
            cat_cols = list(alts_cats_dict.keys())  # e.g. fuel type, body type, age
            num_cats = len(cat_cols)
            alts_long = pd.DataFrame(
                list(itertools.product(*alts_cats_dict.values())),
                columns=alts_cats_dict.keys())
            alts_wide = pd.get_dummies(alts_long)  # rows will sum to num_cats

            # store alts in primary configs dir
            configs_dirs = inject.get_injectable("configs_dir")
            configs_dirs = [configs_dirs] if isinstance(configs_dirs, str) else configs_dirs
            alts_wide.to_csv(os.path.join(configs_dirs[0], alts_fname), index=False)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    choice_column_name = 'vehicle_type'

    estimator = estimation.manager.begin_estimation('vehicle_type')

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

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

    # STEP I. run logit choices
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
    else:
        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=model_spec,
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

    if alts_cats_dict:
        alts = alts_long[alts_long.columns].apply(
            lambda row: '_'.join(row.values.astype(str)), axis=1).values
    else:
        alts = model_spec.columns
    choices[choice_column_name] = \
        choices[choice_column_name].map(dict(list(zip(list(range(len(alts))), alts))))

    # STEP II: append probabilistic vehicle type attributes
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

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from activitysim.cli import run
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import chunk
from activitysim.core import simulate
from activitysim.core import logit
from activitysim.abm.models.util import estimation
from activitysim.core import expressions
from activitysim.core.util import assign_in_place


def run_trip_mode_choice(do_these_purposes=None, choose_individual_max_utility=True):

    """open pipeline and load stuff for mode choice dev assuming model has been run and pipeline.h5 exists"""
    resume_after = "trip_scheduling"
    model_name = "trip_mode_choice"
    chunk_size = 0  # test_mtc means no chunking

    pipeline.open_pipeline(resume_after)
    # preload any bulky injectables (e.g. skims) not in pipeline
    inject.get_injectable('preload_injectables', None)
    pipeline._PIPELINE.rng().begin_step(model_name)
    step_name = model_name
    args = {}
    checkpoint = pipeline.intermediate_checkpoint(model_name)
    inject.set_step_args(args)

    trips = inject.get_table('trips')
    tours_merged = inject.get_table('tours_merged')
    network_los = inject.get_injectable('network_los')

    trace_label = 'trip_mode_choice'
    model_settings_file_name = 'trip_mode_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    mode_column_name = 'trip_mode'

    trips_df = trips.to_frame()
    print("Running with %d trips", trips_df.shape[0])

    tours_merged = tours_merged.to_frame()
    tours_merged = tours_merged[model_settings['TOURS_MERGED_CHOOSER_COLUMNS']]

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips_df,
        tours_merged,
        left_on='tour_id',
        right_index=True,
        how="left")
    assert trips_merged.index.equals(trips.index)

    # setup skim keys
    assert ('trip_period' not in trips_merged)
    trips_merged['trip_period'] = network_los.skim_time_period_label(trips_merged.depart)

    orig_col = 'origin'
    dest_col = 'destination'

    constants = {}
    constants.update(config.get_model_constants(model_settings))
    constants.update({
        'ORIGIN': orig_col,
        'DESTINATION': dest_col
    })

    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col,
                                               dim3_key='trip_period')
    dot_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col, dest_key=orig_col,
                                               dim3_key='trip_period')
    od_skim_wrapper = skim_dict.wrap('origin', 'destination')

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    nest_specs = config.get_logit_model_settings(model_settings)

    estimator = estimation.manager.begin_estimation('trip_mode_choice')

    choices_list = []
    for primary_purpose, trips_segment in trips_merged.groupby('primary_purpose'):

        if (do_these_purposes is not None) and (primary_purpose not in do_these_purposes):
            continue

        print("trip_mode_choice tour_type '%s' (%s trips)" %
              (primary_purpose, len(trips_segment.index), ))

        # name index so tracing knows how to slice
        assert trips_segment.index.name == 'trip_id'

        coefficients = simulate.get_segment_coefficients(model_settings, primary_purpose)

        locals_dict = {}
        locals_dict.update(constants)
        locals_dict.update(coefficients)

        segment_trace_label = tracing.extend_trace_label(trace_label, primary_purpose)

        expressions.annotate_preprocessors(
            trips_segment, locals_dict, skims,
            model_settings, segment_trace_label)

        locals_dict.update(skims)

        ################ Replace wrapper function
        #     choices = mode_choice_simulate(...)
        spec=simulate.eval_coefficients(model_spec, coefficients, estimator)
        nest_spec = simulate.eval_nest_coefficients(nest_specs, coefficients, segment_trace_label)
        choices = simulate.simple_simulate(
            choosers=trips_segment,
            spec=spec,
            nest_spec=nest_spec,
            skims=skims,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            want_logsums=logsum_column_name is not None,
            trace_label=segment_trace_label,
            trace_choice_name='trip_mode_choice',
            estimator=estimator,
            trace_column_names=None,
            choose_individual_max_utility=choose_individual_max_utility)
        # for consistency, always return dataframe, whether or not logsums were requested
        if isinstance(choices, pd.Series):
            choices = choices.to_frame('choice')
        choices.rename(columns={'logsum': logsum_column_name,
                                'choice': mode_column_name},
                       inplace=True)
        if not choose_individual_max_utility:
            alts = spec.columns
            choices[mode_column_name] = choices[mode_column_name].map(dict(list(zip(list(range(len(alts))), alts))))
        ################
        choices_list.append(choices)
    choices_df_asim = pd.concat(choices_list)

    # update trips table with choices (and potionally logssums)
    trips_df = trips.to_frame()

    if (do_these_purposes is not None):
        trips_df  = trips_df.loc[trips_df.primary_purpose.isin(do_these_purposes)]

    assign_in_place(trips_df, choices_df_asim)
    assert not trips_df[mode_column_name].isnull().any()

    finalise = True
    if finalise:
        inject.set_step_args(None)
        #
        pipeline._PIPELINE.rng().end_step(model_name)
        pipeline.add_checkpoint(model_name)
        if not pipeline.intermediate_checkpoint():
            pipeline.add_checkpoint(pipeline.FINAL_CHECKPOINT_NAME)

        pipeline.close_pipeline()

    print("Done")

    return trips_df




def eval_nl_dev(choosers, spec, nest_spec, locals_d, custom_chooser, estimator,
                log_alt_losers=False,
                want_logsums=False, trace_label=None,
                trace_choice_name=None, trace_column_names=None):

    trace_label = tracing.extend_trace_label(trace_label, 'eval_nl')
    assert trace_label
    have_trace_targets = tracing.has_trace_targets(choosers)

    logit.validate_nest_spec(nest_spec, trace_label)
    raw_utilities = simulate.eval_utilities(spec, choosers, locals_d,
                                            log_alt_losers=log_alt_losers,
                                            trace_label=trace_label, have_trace_targets=have_trace_targets,
                                            estimator=estimator, trace_column_names=trace_column_names)
    # exponentiated utilities of leaves and nests
    nested_exp_utilities = simulate.compute_nested_exp_utilities(raw_utilities, nest_spec)
    nested_utils = simulate.compute_nested_utilities(raw_utilities, nest_spec)
    # probabilities of alternatives relative to siblings sharing the same nest
    nested_probabilities = simulate.compute_nested_probabilities(nested_exp_utilities, nest_spec,
                                                                 trace_label=trace_label)
    if want_logsums:
        # logsum of nest root
        logsums = pd.Series(np.log(nested_exp_utilities.root), index=choosers.index)
    # global (flattened) leaf probabilities based on relative nest coefficients (in spec order)
    base_probabilities = simulate.compute_base_probabilities(nested_probabilities, nest_spec, spec)
    # note base_probabilities could all be zero since we allowed all probs for nests to be zero
    # check here to print a clear message but make_choices will raise error if probs don't sum to 1
    BAD_PROB_THRESHOLD = 0.001
    no_choices = (base_probabilities.sum(axis=1) - 1).abs() > BAD_PROB_THRESHOLD
    if no_choices.any():
        print("BAD")
    choices, rands = logit.make_choices(base_probabilities, trace_label=trace_label)
    if want_logsums:
        choices = choices.to_frame('choice')
        choices['logsum'] = logsums
    return choices, raw_utilities, nested_exp_utilities, nested_utils, nested_probabilities, base_probabilities


def simple_simulate_dev(choosers, spec, nest_spec,
                        skims=None, locals_d=None,
                        chunk_size=0, custom_chooser=None,
                        log_alt_losers=False,
                        want_logsums=False,
                        estimator=None,
                        trace_label=None, trace_choice_name=None, trace_column_names=None):
    trace_label = tracing.extend_trace_label(trace_label, 'simple_simulate')
    assert len(choosers) > 0
    result_list = []
    # segment by person type and pick the right spec for each person type
    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(choosers, chunk_size, trace_label):
        # the following replaces choices = _simple_simulate(...)
        if skims is not None:
            simulate.set_skim_wrapper_targets(choosers, skims)

        # only do this for nested, logit is straight forward
        assert nest_spec is not None
        choices, raw_utilities, nested_exp_utilities, nested_utils, \
            nested_probs, base_probs = eval_nl_dev(choosers, spec, nest_spec, locals_d, custom_chooser,
                                log_alt_losers=log_alt_losers,
                                want_logsums=want_logsums, estimator=estimator, trace_label=trace_label,
                                trace_choice_name=trace_choice_name, trace_column_names=trace_column_names)

        result_list.append(choices)
        chunk.log_df(trace_label, f'result_list', result_list)

    if len(result_list) > 1:
        choices = pd.concat(result_list)
    assert len(choices.index == len(choosers.index))
    return choices, raw_utilities, nested_exp_utilities, nested_utils, nested_probs, base_probs


def get_stuff(do_these_purposes=None):
    #do_these_purposes=['escort']
    """open pipeline and load stuff for mode choice dev assuming model has been run and pipeline.h5 exists"""
    resume_after = "trip_scheduling"
    model_name = "trip_mode_choice"
    chunk_size = 0  # test_mtc means no chunking

    pipeline.open_pipeline(resume_after)
    # preload any bulky injectables (e.g. skims) not in pipeline
    inject.get_injectable('preload_injectables', None)
    pipeline._PIPELINE.rng().begin_step(model_name)
    step_name = model_name
    args = {}
    checkpoint = pipeline.intermediate_checkpoint(model_name)
    inject.set_step_args(args)

    trips = inject.get_table('trips')
    tours_merged = inject.get_table('tours_merged')
    network_los = inject.get_injectable('network_los')

    trace_label = 'trip_mode_choice'
    model_settings_file_name = 'trip_mode_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    mode_column_name = 'trip_mode'

    trips_df = trips.to_frame()
    print("Running with %d trips", trips_df.shape[0])

    tours_merged = tours_merged.to_frame()
    tours_merged = tours_merged[model_settings['TOURS_MERGED_CHOOSER_COLUMNS']]

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips_df,
        tours_merged,
        left_on='tour_id',
        right_index=True,
        how="left")
    assert trips_merged.index.equals(trips.index)

    # setup skim keys
    assert ('trip_period' not in trips_merged)
    trips_merged['trip_period'] = network_los.skim_time_period_label(trips_merged.depart)

    orig_col = 'origin'
    dest_col = 'destination'

    constants = {}
    constants.update(config.get_model_constants(model_settings))
    constants.update({
        'ORIGIN': orig_col,
        'DESTINATION': dest_col
    })

    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col, dest_key=dest_col,
                                               dim3_key='trip_period')
    dot_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col, dest_key=orig_col,
                                               dim3_key='trip_period')
    od_skim_wrapper = skim_dict.wrap('origin', 'destination')

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    nest_specs = config.get_logit_model_settings(model_settings)

    estimator = estimation.manager.begin_estimation('trip_mode_choice')

    choices_list = []
    raw_util_list = []
    nest_list = []
    nu_list = []
    nest_spec_list = []
    nested_probs_list = []
    base_probs_list = []

    for primary_purpose, trips_segment in trips_merged.groupby('primary_purpose'):

        if (do_these_purposes is not None) and (primary_purpose not in do_these_purposes):
            continue

        print("trip_mode_choice tour_type '%s' (%s trips)" %
              (primary_purpose, len(trips_segment.index), ))

        # name index so tracing knows how to slice
        assert trips_segment.index.name == 'trip_id'

        coefficients = simulate.get_segment_coefficients(model_settings, primary_purpose)

        locals_dict = {}
        locals_dict.update(constants)
        locals_dict.update(coefficients)

        segment_trace_label = tracing.extend_trace_label(trace_label, primary_purpose)

        expressions.annotate_preprocessors(
            trips_segment, locals_dict, skims,
            model_settings, segment_trace_label)

        locals_dict.update(skims)

        ################ Replace wrapper function
        #     choices = mode_choice_simulate(...)
        spec=simulate.eval_coefficients(model_spec, coefficients, estimator)
        nest_spec = simulate.eval_nest_coefficients(nest_specs, coefficients, segment_trace_label)
        choices, raw_utilities, nested_exp_utilities, nested_utils, nested_probs, base_probs = simple_simulate_dev(
            choosers=trips_segment,
            spec=spec,
            nest_spec=nest_spec,
            skims=skims,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            want_logsums=logsum_column_name is not None,
            trace_label=segment_trace_label,
            trace_choice_name='trip_mode_choice',
            estimator=estimator,
            trace_column_names=None)
        # for consistency, always return dataframe, whether or not logsums were requested
        if isinstance(choices, pd.Series):
            choices = choices.to_frame('choice')
        choices.rename(columns={'logsum': logsum_column_name,
                                'choice': mode_column_name},
                       inplace=True)
        alts = spec.columns
        choices[mode_column_name] = choices[mode_column_name].map(dict(list(zip(list(range(len(alts))), alts))))
        ################
        choices_list.append(choices)
        raw_util_list.append(raw_utilities)
        nest_list.append(nested_exp_utilities)
        nu_list.append(nested_utils)
        nest_spec_list.append(nest_spec)
        nested_probs_list.append(nested_probs)
        base_probs_list.append(base_probs)


    choices_df_asim = pd.concat(choices_list)

    # update trips table with choices (and potionally logssums)
    trips_df = trips.to_frame()

    if (do_these_purposes is not None):
        trips_df  = trips_df.loc[trips_df.primary_purpose.isin(do_these_purposes)]

    assign_in_place(trips_df, choices_df_asim)
    assert not trips_df[mode_column_name].isnull().any()

    finalise = True
    if finalise:
        inject.set_step_args(None)
        #
        pipeline._PIPELINE.rng().end_step(model_name)
        pipeline.add_checkpoint(model_name)
        if not pipeline.intermediate_checkpoint():
            pipeline.add_checkpoint(pipeline.FINAL_CHECKPOINT_NAME)

        pipeline.close_pipeline()

    print("Done")

    return trips_df, raw_util_list, nest_list, nu_list, nest_spec_list, nested_probs_list, base_probs_list
# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core.assign import evaluate_constants

from mode import _mode_choice_spec
from mode import get_segment_and_unstack
from mode import tour_mode_choice_coeffecients_spec


from . import expressions


logger = logging.getLogger(__name__)


def get_omnibus_logsum_spec(logsum_settings, selector, configs_dir, want_tracing):

    spec_file_name = logsum_settings['SPEC']
    coeffs_file_name = logsum_settings['COEFFS']

    logsums_spec_df = simulate.read_model_spec(configs_dir, spec_file_name)

    # - slice by selector
    assert selector in logsums_spec_df
    logsums_spec_df = logsums_spec_df[logsums_spec_df[selector] > 0]

    # - drop all selector columns (all columns before Alternative - Expression was set to index)
    for c in logsums_spec_df.columns:
        if c == 'Alternative':
            break
        del logsums_spec_df[c]

    # read coefficients
    with open(os.path.join(configs_dir, coeffs_file_name)) as f:
        logsums_coeffs = pd.read_csv(f, index_col='Expression')

    spec = _mode_choice_spec(logsums_spec_df,
                             logsums_coeffs,
                             logsum_settings,
                             trace_spec=want_tracing,
                             trace_label='logsum_spec')

    return spec


def get_logsum_spec(logsum_settings, selector, tour_purpose, configs_dir, want_tracing):
    """

    Parameters
    ----------
    logsum_settings
    selector - str
        one of nontour, joint, atwork
    segment - str
        one of eatout escort othdiscr othmaint school shopping social university work atwork
    configs_dir
    want_tracing

    Returns
    -------

    """

    omnibus_logsum_spec = \
        get_omnibus_logsum_spec(logsum_settings, selector, configs_dir, want_tracing)
    logsum_spec = get_segment_and_unstack(omnibus_logsum_spec, segment=tour_purpose)

    if want_tracing:
        trace_label = 'get_logsum_spec_%s_%s' % (selector, tour_purpose)
        tracing.trace_df(logsum_spec,
                         trace_label,
                         slicer='NONE', transpose=False)

    return logsum_spec


def filter_chooser_columns(choosers, logsum_settings, model_settings):

    chooser_columns = logsum_settings.get('LOGSUM_CHOOSER_COLUMNS', [])

    if 'CHOOSER_ORIG_COL_NAME' in model_settings:
        chooser_columns.append(model_settings['CHOOSER_ORIG_COL_NAME'])

    missing_columns = [c for c in chooser_columns if c not in choosers]
    if missing_columns:
        logger.info("filter_chooser_columns missing_columns %s" % missing_columns)

    # ignore any columns not appearing in choosers df
    chooser_columns = [c for c in chooser_columns if c in choosers]

    choosers = choosers[chooser_columns]
    return choosers


def compute_logsums(choosers,
                    logsum_spec, tour_purpose,
                    logsum_settings, model_settings,
                    skim_dict, skim_stack,
                    chunk_size, trace_hh_id, trace_label):
    """

    Parameters
    ----------
    choosers
    logsum_spec
    logsum_settings
    model_settings
    skim_dict
    skim_stack
    chunk_size
    trace_hh_id
    trace_label

    Returns
    -------
    logsums: pandas series
        computed logsums with same index as choosers
    """

    trace_label = tracing.extend_trace_label(trace_label, 'compute_logsums')

    # compute_logsums needs to know name of dest column in interaction_sample
    orig_col_name = model_settings['CHOOSER_ORIG_COL_NAME']
    dest_col_name = model_settings['ALT_DEST_COL_NAME']

    # FIXME - are we ok with altering choosers (so caller doesn't have to set these)?
    assert ('in_period' not in choosers) and ('out_period' not in choosers)
    choosers['in_period'] = expressions.skim_time_period_label(model_settings['IN_PERIOD'])
    choosers['out_period'] = expressions.skim_time_period_label(model_settings['OUT_PERIOD'])

    nest_spec = config.get_logit_model_settings(logsum_settings)
    constants = config.get_model_constants(logsum_settings)

    omnibus_coefficient_spec = tour_mode_choice_coeffecients_spec(logsum_settings)
    locals_dict = evaluate_constants(omnibus_coefficient_spec[tour_purpose], constants=constants)
    locals_dict.update(constants)

    logger.info("Running compute_logsums with %d choosers" % choosers.shape[0])

    if trace_hh_id:
        tracing.trace_df(logsum_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    # setup skim keys
    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col_name, right_key=dest_col_name,
                                             skim_key='out_period')
    dot_skim_stack_wrapper = skim_stack.wrap(left_key=dest_col_name, right_key=orig_col_name,
                                             skim_key='in_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name
    }
    if constants is not None:
        locals_dict.update(skims)

    # - run preprocessor to annotate choosers
    preprocessor_settings = logsum_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        simulate.add_skims(choosers, skims)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return logsums

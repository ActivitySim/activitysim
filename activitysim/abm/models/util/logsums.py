# ActivitySim
# See full license in LICENSE.txt.
from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core.assign import evaluate_constants

from .mode import tour_mode_choice_spec
from .mode import tour_mode_choice_coeffecients_spec


from . import expressions


logger = logging.getLogger(__name__)


def get_logsum_spec(model_settings):

    return tour_mode_choice_spec(model_settings)


def get_coeffecients_spec(model_settings):
    return tour_mode_choice_coeffecients_spec(model_settings)


def filter_chooser_columns(choosers, logsum_settings, model_settings):

    chooser_columns = logsum_settings.get('LOGSUM_CHOOSER_COLUMNS', [])

    if 'CHOOSER_ORIG_COL_NAME' in model_settings:
        chooser_columns.append(model_settings['CHOOSER_ORIG_COL_NAME'])

    missing_columns = [c for c in chooser_columns if c not in choosers]
    if missing_columns:
        logger.debug("logsum.filter_chooser_columns missing_columns %s" % missing_columns)

    # ignore any columns not appearing in choosers df
    chooser_columns = [c for c in chooser_columns if c in choosers]

    choosers = choosers[chooser_columns]
    return choosers


def compute_logsums(choosers,
                    tour_purpose,
                    logsum_settings, model_settings,
                    skim_dict, skim_stack,
                    chunk_size, trace_hh_id, trace_label):
    """

    Parameters
    ----------
    choosers
    tour_purpose
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

    logsum_spec = get_logsum_spec(logsum_settings)

    omnibus_coefficient_spec = get_coeffecients_spec(logsum_settings)
    coefficient_spec = omnibus_coefficient_spec[tour_purpose]

    # compute_logsums needs to know name of dest column in interaction_sample
    orig_col_name = model_settings['CHOOSER_ORIG_COL_NAME']
    dest_col_name = model_settings['ALT_DEST_COL_NAME']

    # FIXME - are we ok with altering choosers (so caller doesn't have to set these)?
    assert ('in_period' not in choosers) and ('out_period' not in choosers)
    choosers['in_period'] = expressions.skim_time_period_label(model_settings['IN_PERIOD'])
    choosers['out_period'] = expressions.skim_time_period_label(model_settings['OUT_PERIOD'])

    assert ('duration' not in choosers)
    choosers['duration'] = model_settings['IN_PERIOD'] - model_settings['OUT_PERIOD']

    nest_spec = config.get_logit_model_settings(logsum_settings)
    constants = config.get_model_constants(logsum_settings)

    logger.debug("Running compute_logsums with %d choosers" % choosers.shape[0])

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

    locals_dict = evaluate_constants(coefficient_spec, constants=constants)
    locals_dict.update(constants)
    locals_dict.update(skims)

    # - run preprocessor to annotate choosers
    # allow specification of alternate preprocessor for nontour choosers
    preprocessor = model_settings.get('LOGSUM_PREPROCESSOR', 'preprocessor')
    preprocessor_settings = logsum_settings[preprocessor]

    if preprocessor_settings:

        simulate.set_skim_wrapper_targets(choosers, skims)

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

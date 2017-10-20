# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config


logger = logging.getLogger(__name__)


def mode_choice_logsums_spec(configs_dir, dest_type):
    DEST_TO_TOUR_SPEC_TYPE = \
        {'university': 'university',
         'highschool': 'school',
         'gradeschool': 'school',
         'work': 'work'}

    # tour_spec_type is just name for labelling specs which differ by dest, not tour_type
    tour_spec_type = DEST_TO_TOUR_SPEC_TYPE.get(dest_type)
    spec = simulate.read_model_spec(configs_dir, 'logsums_spec_%s.csv' % tour_spec_type)
    return spec


def compute_logsums(choosers, logsum_spec, logsum_settings,
                    skim_dict, skim_stack, chooser_col_name, alt_col_name,
                    chunk_size, trace_hh_id, trace_label):
    """

    Parameters
    ----------
    choosers
    logsum_spec
    logsum_settings
    skim_dict
    skim_stack
    alt_col_name
    chunk_size
    trace_hh_id
    trace_label

    Returns
    -------
    logsums: pandas series
        computed logsums with same index as choosers
    """

    trace_label = tracing.extend_trace_label(trace_label, 'compute_logsums')

    nest_spec = config.get_logit_model_settings(logsum_settings)
    constants = config.get_model_constants(logsum_settings)

    logger.info("Running compute_logsums with %d choosers" % len(choosers.index))

    if trace_hh_id:
        tracing.trace_df(logsum_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    # setup skim keys
    odt_skim_stack_wrapper = skim_stack.wrap(left_key=chooser_col_name, right_key=alt_col_name,
                                             skim_key="out_period")
    dot_skim_stack_wrapper = skim_stack.wrap(left_key=alt_col_name, right_key=chooser_col_name,
                                             skim_key="in_period")
    od_skim_stack_wrapper = skim_dict.wrap(chooser_col_name, alt_col_name)

    skims = [odt_skim_stack_wrapper, dot_skim_stack_wrapper, od_skim_stack_wrapper]

    locals_d = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper
    }
    if constants is not None:
        locals_d.update(constants)

    logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return logsums

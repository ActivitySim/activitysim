# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd
import orca

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core import pipeline
from activitysim.core.simulate import read_model_spec
from activitysim.core.util import reindex

from .mode import get_segment_and_unstack


logger = logging.getLogger(__name__)

DUMP = False


def time_period_label(hour):

    time_periods = config.setting('time_periods')

    bin = np.digitize([hour % 24], time_periods['hours'])[0] - 1

    return time_periods['labels'][bin]


def compute_logsums(choosers, logsum_spec, logsum_settings,
                    skim_dict, skim_stack, alt_col_name,
                    chunk_size, trace_hh_id):

    trace_label = trace_hh_id and 'compute_logsums'

    nest_spec = config.get_logit_model_settings(logsum_settings)
    constants = config.get_model_constants(logsum_settings)

    logger.info("Running compute_logsums with %d choosers" % len(choosers.index))

    if trace_hh_id:
        tracing.trace_df(logsum_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    # setup skim keys
    odt_skim_stack_wrapper = skim_stack.wrap(left_key='TAZ', right_key=alt_col_name,
                                             skim_key="out_period")
    dot_skim_stack_wrapper = skim_stack.wrap(left_key=alt_col_name, right_key='TAZ',
                                             skim_key="in_period")
    od_skim_stack_wrapper = skim_dict.wrap('TAZ', alt_col_name)

    skims = [odt_skim_stack_wrapper, dot_skim_stack_wrapper, od_skim_stack_wrapper]

    locals_d = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper
    }
    if constants is not None:
        locals_d.update(constants)

    logsums = asim.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return logsums


@orca.step()
def workplace_location_logsums(persons_merged,
                               land_use,
                               skim_dict, skim_stack,
                               workplace_location_sample,
                               configs_dir,
                               chunk_size,
                               trace_hh_id):

    logsums_spec = read_model_spec(configs_dir, 'workplace_location_logsums.csv')
    workplace_location_settings = config.read_model_settings(configs_dir, 'workplace_location.yaml')

    alt_col_name = workplace_location_settings["ALT_COL_NAME"]

    # FIXME - just using settings from tour_mode_choice
    logsum_settings = config.read_model_settings(configs_dir, 'tour_mode_choice.yaml')

    persons_merged = persons_merged.to_frame()
    workplace_location_sample = workplace_location_sample.to_frame()

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = workplace_location_settings['LOGSUM_CHOOSER_COLUMNS']
    persons_merged = persons_merged[chooser_columns]

    choosers = pd.merge(workplace_location_sample,
                        persons_merged,
                        left_index=True,
                        right_index=True,
                        how="left")

    choosers['in_period'] = time_period_label(workplace_location_settings['IN_PERIOD'])
    choosers['out_period'] = time_period_label(workplace_location_settings['OUT_PERIOD'])

    # FIXME - should do this in expression file?
    choosers['dest_topology'] = reindex(land_use.TOPOLOGY, choosers[alt_col_name])
    choosers['dest_density_index'] = reindex(land_use.density_index, choosers[alt_col_name])

    tracing.dump_df(DUMP, persons_merged, 'workplace_location_logsums', 'persons_merged')
    tracing.dump_df(DUMP, choosers, 'workplace_location_logsums', 'choosers')

    logsums = compute_logsums(
        choosers, logsums_spec, logsum_settings,
        skim_dict, skim_stack, alt_col_name, chunk_size, trace_hh_id)

    choosers['logsum'] = logsums
    orca.add_table('workplace_location_choosers', choosers)

    workplace_location_sample['logsum'] = np.asanyarray(logsums)
    orca.add_table('workplace_location_logsums', workplace_location_sample)

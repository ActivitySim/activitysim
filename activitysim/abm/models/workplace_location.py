# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import numpy as np
import orca

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.core.util import reindex
from .util.logsums import compute_logsums

logger = logging.getLogger(__name__)


DUMP = False

"""
The workplace location model predicts the zones in which various people will
work.

for now we generate a workplace location for everyone -
presumably it will not get used in downstream models for everyone -
it should depend on CDAP and mandatory tour generation as to whether
it gets used
"""


def time_period_label(hour):
    time_periods = config.setting('time_periods')
    bin = np.digitize([hour % 24], time_periods['hours'])[0] - 1
    return time_periods['labels'][bin]


@orca.injectable()
def workplace_location_sample_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'workplace_location_sample.csv')


@orca.injectable()
def workplace_location_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'workplace_location.yaml')


@orca.step()
def workplace_location_sample(persons_merged,
                              workplace_location_sample_spec,
                              workplace_location_settings,
                              skim_dict,
                              destination_size_terms,
                              chunk_size,
                              trace_hh_id):
    """
    build a table of workers * all zones in order to select a sample of alternative work locations.
    """

    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()

    constants = config.get_model_constants(workplace_location_settings)

    sample_size = workplace_location_settings["SAMPLE_SIZE"]
    alt_col_name = workplace_location_settings["ALT_COL_NAME"]

    logger.info("Running workplace_location_sample with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", "TAZ_r")

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = workplace_location_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    choices = interaction_sample(
        choosers,
        alternatives,
        sample_size=sample_size,
        alt_col_name=alt_col_name,
        spec=workplace_location_sample_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_hh_id and 'workplace_location_sample',
        trace_choice_name='workplace_location')

    orca.add_table('workplace_location_sample', choices)


@orca.step()
def workplace_location_logsums(persons_merged,
                               land_use,
                               skim_dict, skim_stack,
                               workplace_location_sample,
                               configs_dir,
                               chunk_size,
                               trace_hh_id):
    """
    add logsum column to workplace_location_sample

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in workplace_location_sample, and computing the logsum of all the utilities
    """

    logsums_spec = simulate.read_model_spec(configs_dir, 'workplace_location_logsums.csv')
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

    # workplace_location_sample['logsum'] = np.asanyarray(logsums)
    # orca.add_table('workplace_location_logsums', workplace_location_sample)

    # add_column series should have an index matching the table to which it is being added
    # logsums does, since workplace_location_sample was on left side of merge creating choosers
    orca.add_column("workplace_location_sample", "logsum", logsums)


@orca.injectable()
def workplace_location_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'workplace_location.csv')


@orca.injectable()
def workplace_location_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'workplace_location.yaml')


@orca.step()
def workplace_location_simulate(persons_merged,
                                workplace_location_sample,
                                workplace_location_spec,
                                workplace_location_settings,
                                skim_dict,
                                destination_size_terms,
                                chunk_size,
                                trace_hh_id):

    """
    Workplace location model on workplace_location_sample annotated with mode_choice logsum
    to select a work_taz from sample alternatives
    """

    alt_col_name = workplace_location_settings["ALT_COL_NAME"]

    # for now I'm going to generate a workplace location for everyone -
    # presumably it will not get used in downstream models for everyone -
    # it should depend on CDAP and mandatory tour generation as to whether
    # it gets used
    choosers = persons_merged.to_frame()

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge additional alt columns into alt sample list
    workplace_location_sample = workplace_location_sample.to_frame()
    destination_size_terms = destination_size_terms.to_frame()
    alternatives = \
        pd.merge(workplace_location_sample, destination_size_terms,
                 left_on=alt_col_name, right_index=True, how="left")

    tracing.dump_df(DUMP, alternatives, 'workplace_location_simulate', 'alternatives')

    constants = config.get_model_constants(workplace_location_settings)

    sample_size = workplace_location_settings["SAMPLE_SIZE"]

    logger.info("Running workplace_location_simulate with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", alt_col_name)

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = workplace_location_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    tracing.dump_df(DUMP, choosers, 'workplace_location_simulate', 'choosers')

    choices = interaction_sample_simulate(
        choosers,
        alternatives,
        spec=workplace_location_spec,
        choice_column=alt_col_name,
        skims=skims,
        locals_d=locals_d,
        sample_size=sample_size,
        chunk_size=chunk_size,
        trace_label=trace_hh_id and 'workplace_location',
        trace_choice_name='workplace_location')

    # FIXME - no need to reindex since we didn't slice choosers
    # choices = choices.reindex(persons_merged.index)

    tracing.print_summary('workplace_taz', choices, describe=True)

    orca.add_column("persons", "workplace_taz", choices)

    pipeline.add_dependent_columns("persons", "persons_workplace")

    if trace_hh_id:
        trace_columns = ['workplace_taz'] + orca.get_table('persons_workplace').columns
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="workplace_location",
                         columns=trace_columns,
                         warn_if_empty=True)

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import orca

from activitysim.core import simulate as asim
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core import pipeline

logger = logging.getLogger(__name__)


DUMP = True


@orca.injectable()
def workplace_location_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'workplace_location.csv')


@orca.injectable()
def workplace_location_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'workplace_location.yaml')


@orca.step()
def workplace_location_simulate(persons_merged,
                                workplace_location_logsums,
                                workplace_location_spec,
                                workplace_location_settings,
                                skim_dict,
                                destination_size_terms,
                                chunk_size,
                                trace_hh_id):

    """
    The workplace location model predicts the zones in which various people will
    work.
    """

    alt_col_name = workplace_location_settings["ALT_COL_NAME"]

    # for now I'm going to generate a workplace location for everyone -
    # presumably it will not get used in downstream models for everyone -
    # it should depend on CDAP and mandatory tour generation as to whether
    # it gets used
    choosers = persons_merged.to_frame()

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge additional alt columns into alt sample list
    workplace_location_logsums = workplace_location_logsums.to_frame()
    destination_size_terms = destination_size_terms.to_frame()
    alternatives = \
        pd.merge(workplace_location_logsums, destination_size_terms,
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

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca

from activitysim.core.simulate import read_model_spec
from activitysim.core.interaction_sample import interaction_sample
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core import pipeline

logger = logging.getLogger(__name__)


@orca.injectable()
def workplace_location_sample_spec(configs_dir):
    return read_model_spec(configs_dir, 'workplace_location_sample.csv')


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
    The workplace location model predicts the zones in which various people will
    work.
    """

    # for now I'm going to generate a workplace location for everyone -
    # presumably it will not get used in downstream models for everyone -
    # it should depend on CDAP and mandatory tour generation as to whether
    # it gets used
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

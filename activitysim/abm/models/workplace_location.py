# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core import pipeline

logger = logging.getLogger(__name__)


@orca.injectable()
def workplace_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'workplace_location.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def workplace_location_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'workplace_location.yaml')


@orca.step()
def workplace_location_simulate(persons_merged,
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

    # for now I'm going to generate a workplace location for everyone -
    # presumably it will not get used in downstream models for everyone -
    # it should depend on CDAP and mandatory tour generation as to whether
    # it gets used
    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()

    constants = config.get_model_constants(workplace_location_settings)

    sample_size = workplace_location_settings["SAMPLE_SIZE"]

    logger.info("Running workplace_location_simulate with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", "TAZ_r")

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    # FIXME - HACK - only include columns actually used in spec (which we pathologically know)
    choosers = choosers[["income_segment", "TAZ", "mode_choice_logsums"]]

    choices = asim.interaction_simulate(
        choosers,
        alternatives,
        spec=workplace_location_spec,
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

    # FIXME - test prng repeatability
    r = pipeline.get_rn_generator().random_for_df(choices)
    orca.add_column("persons", "work_location_rand", [item for sublist in r for item in sublist])

    if trace_hh_id:
        trace_columns = ['workplace_taz'] + orca.get_table('persons_workplace').columns
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="workplace_location",
                         columns=trace_columns,
                         warn_if_empty=True)

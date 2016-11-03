# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca

from activitysim import activitysim as asim
from activitysim import tracing
from .util.misc import add_dependent_columns
from .util.misc import read_model_settings, get_model_constants

logger = logging.getLogger(__name__)


@orca.injectable()
def workplace_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'workplace_location.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def workplace_location_settings(configs_dir):
    return read_model_settings(configs_dir, 'school_location.yaml')


@orca.step()
def workplace_location_simulate(set_random_seed,
                                persons_merged,
                                workplace_location_spec,
                                workplace_location_settings,
                                skims,
                                destination_size_terms,
                                chunk_size,
                                trace_hh_id):

    """
    The workplace location model predicts the zones in which various people will
    work.
    """

    # for now I'm going to generate a workplace location for everyone -
    # presumably it will not get used in downstream models for everyone -
    # it should depend on CDAP and mandatory tour generation as to whethrer
    # it gets used
    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()

    constants = get_model_constants(workplace_location_settings)

    logger.info("Running workplace_location_simulate with %d persons" % len(choosers))

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims.set_keys("TAZ", "TAZ_r")

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
        sample_size=50,
        chunk_size=chunk_size,
        trace_label=trace_hh_id and 'workplace_location',
        trace_choice_name='workplace_location')

    # FIXME - no need to reindex?
    choices = choices.reindex(persons_merged.index)

    logger.info("%s workplace_taz choices min: %s max: %s" %
                (len(choices.index), choices.min(), choices.max()))

    tracing.print_summary('workplace_taz', choices, describe=True)

    orca.add_column("persons", "workplace_taz", choices)

    add_dependent_columns("persons", "persons_workplace")

    if trace_hh_id:
        trace_columns = ['workplace_taz'] + orca.get_table('persons_workplace').columns
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="workplace_location",
                         columns=trace_columns,
                         warn_if_empty=True)

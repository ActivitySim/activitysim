# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import numpy as np

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config

logger = logging.getLogger(__name__)


@orca.injectable()
def school_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'school_location.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def school_location_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'school_location.yaml')


@orca.step()
def school_location_simulate(persons_merged,
                             school_location_spec,
                             school_location_settings,
                             skim_dict,
                             destination_size_terms,
                             chunk_size,
                             trace_hh_id):

    """
    The school location model predicts the zones in which various people will
    go to school.
    """

    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()

    constants = config.get_model_constants(school_location_settings)

    sample_size = school_location_settings["SAMPLE_SIZE"]

    logger.info("Running school_location_simulate with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", "TAZ_r")

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    choices_list = []
    for school_type in ['university', 'highschool', 'gradeschool']:

        locals_d['segment'] = school_type

        choosers_segment = choosers[choosers["is_" + school_type]]

        # FIXME - no point in considering impossible alternatives
        alternatives_segment = alternatives[alternatives[school_type] > 0]

        logger.info("school_type %s:  %s persons %s alternatives" %
                    (school_type, len(choosers_segment), len(alternatives_segment)))

        if len(choosers_segment.index) > 0:

            choices = asim.interaction_simulate(
                choosers_segment,
                alternatives_segment,
                spec=school_location_spec[[school_type]],
                skims=skims,
                locals_d=locals_d,
                sample_size=sample_size,
                chunk_size=chunk_size,
                trace_label='school_location.%s' % school_type,
                trace_choice_name='school_location')

            choices_list.append(choices)

    choices = pd.concat(choices_list)

    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    choices = choices.reindex(persons_merged.index).fillna(-1)

    tracing.print_summary('school_taz', choices, describe=True)

    orca.add_column("persons", "school_taz", choices)

    pipeline.add_dependent_columns("persons", "persons_school")

    # FIXME - test prng repeatability
    r = pipeline.get_rn_generator().random_for_df(choices)
    orca.add_column("persons", "school_location_rand", [item for sublist in r for item in sublist])

    if trace_hh_id:
        trace_columns = ['school_taz'] + orca.get_table('persons_school').columns
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="school_location",
                         columns=trace_columns,
                         warn_if_empty=True)

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import numpy as np

from activitysim import activitysim as asim
from activitysim import tracing

from .util.misc import read_model_settings, get_logit_model_settings, get_model_constants

logger = logging.getLogger(__name__)


@orca.table()
def destination_choice_spec(configs_dir):
    f = os.path.join(configs_dir, 'destination_choice.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def destination_choice_settings(configs_dir):
    return read_model_settings(configs_dir, 'destination_choice.yaml')


@orca.step()
def destination_choice(set_random_seed,
                       non_mandatory_tours_merged,
                       skims,
                       destination_choice_spec,
                       destination_choice_settings,
                       destination_size_terms,
                       chunk_size,
                       trace_hh_id):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    # choosers are tours - in a sense tours are choosing their destination
    choosers = non_mandatory_tours_merged.to_frame()
    alternatives = destination_size_terms.to_frame()
    spec = destination_choice_spec.to_frame()

    constants = get_model_constants(destination_choice_settings)

    if trace_hh_id:
        # register non_mandatory_tours so we can slice utilities
        tracing.register_tours(choosers, trace_hh_id)

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    skims.set_keys("TAZ", "TAZ_r")
    # the skims will be available under the name "skims" for any @ expressions

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    logger.info("Running destination_choice  with %d non_mandatory_tours" % len(choosers.index))

    choices_list = []
    # segment by trip type and pick the right spec for each person type
    for name, segment in choosers.groupby('tour_type'):

        # FIXME - there are two options here escort with kids and without
        kludge_name = name
        if name == "escort":
            logging.error("destination_choice escort not implemented - running shopping instead")
            kludge_name = "shopping"

        # the segment is now available to switch between size terms
        locals_d['segment'] = kludge_name

        # FIXME - no point in considering impossible alternatives
        alternatives_segment = alternatives[alternatives[kludge_name] > 0]

        logger.info("Running segment '%s' of %d tours %d alternatives" %
                    (name, len(segment), len(alternatives_segment)))

        # name index so tracing knows how to slice
        segment.index.name = 'tour_id'

        choices = asim.interaction_simulate(segment,
                                            alternatives_segment,
                                            spec[[kludge_name]],
                                            skims=skims,
                                            locals_d=locals_d,
                                            sample_size=50,
                                            chunk_size=chunk_size,
                                            trace_label='destination.%s' % name)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    # FIXME - can there be null destinations?
    if choices.isnull().any():
        logger.error("destination_choice had %s null destinations" % choices.isnull().sum())
        assert choices.isnull().sum() == 0

    tracing.print_summary('destination', choices, describe=True)

    # every trip now has a destination which is the index from the
    # alternatives table - in this case it's the destination taz
    orca.add_column("non_mandatory_tours", "destination", choices)

    if trace_hh_id:
        tracing.trace_df(orca.get_table('non_mandatory_tours').to_frame(),
                         label="destination",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)

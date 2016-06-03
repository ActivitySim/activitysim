# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import numpy as np

from activitysim import activitysim as asim
from activitysim.defaults import tracing

logger = logging.getLogger(__name__)


@orca.table()
def destination_choice_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', 'destination_choice.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def destination_choice(set_random_seed,
                       non_mandatory_tours_merged,
                       skims,
                       destination_choice_spec,
                       destination_size_terms,
                       chunk_size):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    # choosers are tours - in a sense tours are choosing their destination
    choosers = non_mandatory_tours_merged.to_frame()
    alternatives = destination_size_terms.to_frame()
    spec = destination_choice_spec.to_frame()

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    skims.set_keys("TAZ", "TAZ_r")
    # the skims will be available under the name "skims" for any @ expressions
    locals_d = {"skims": skims}

    logger.info("%s destination_choice choosers" % len(choosers.index))

    choices_list = []
    # segment by trip type and pick the right spec for each person type
    for name, segment in choosers.groupby('tour_type'):

        # FIXME - there are two options here escort with kids and without
        if name == "escort":
            logger.error("destination_choice escort not implemented - running shopping instead")
            name = "shopping"

        # the segment is now available to switch between size terms
        locals_d['segment'] = name

        logger.info("Running segment '%s' of size %d" % (name, len(segment)))

        choices = asim.interaction_simulate(segment,
                                            alternatives,
                                            spec[[name]],
                                            skims=skims,
                                            locals_d=locals_d,
                                            sample_size=50,
                                            chunk_size=chunk_size)

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


@orca.step()
def patch_mandatory_tour_destination(mandatory_tours_merged):

    """
    Patch destination column of mandatory tours with school or workplace taz
    to conform to non-mandatory tours naming so that computed columns in the tours
    table can use destination for any tour type.
    """

    mandatory_tours_merged['destination'] = \
        np.where(mandatory_tours_merged['tour_type'] == 'school',
                 mandatory_tours_merged['school_taz'],
                 mandatory_tours_merged['workplace_taz'])

    orca.add_column("mandatory_tours", "destination", mandatory_tours_merged.destination)

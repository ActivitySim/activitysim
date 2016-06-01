# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import trace
from .util.misc import add_dependent_columns
from activitysim.util import reindex
from .util.non_mandatory_tour_frequency import process_non_mandatory_tours


logger = logging.getLogger(__name__)


@orca.injectable()
def non_mandatory_tour_frequency_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "non_mandatory_tour_frequency.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.table()
def non_mandatory_tour_frequency_alts(configs_dir):
    f = os.path.join(configs_dir, "configs",
                     "non_mandatory_tour_frequency_alternatives.csv")
    return pd.read_csv(f)


@orca.column("non_mandatory_tour_frequency_alts")
def tot_tours(non_mandatory_tour_frequency_alts):
    return non_mandatory_tour_frequency_alts.local.sum(axis=1)


@orca.step()
def non_mandatory_tour_frequency(set_random_seed,
                                 persons_merged,
                                 non_mandatory_tour_frequency_alts,
                                 non_mandatory_tour_frequency_spec,
                                 chunk_size):

    """
    This model predicts the frequency of making non-mandatory trips
    (alternatives for this model come from a separate csv file which is
    configured by the user) - these trips include escort, shopping, othmaint,
    othdiscr, eatout, and social trips in various combination.
    """

    choosers = persons_merged.to_frame()

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity.isin(['Mandatory',
                                                     'NonMandatory'])]

    logger.info("%d persons run for non-mandatory tour model" % len(choosers))

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for name, segment in choosers.groupby('ptype_cat'):

        logger.info("Running segment '%s' of size %d" % (name, len(segment)))

        choices = asim.interaction_simulate(
            segment,
            non_mandatory_tour_frequency_alts.to_frame(),
            # notice that we pick the column for the
            # segment for each segment we run
            non_mandatory_tour_frequency_spec[[name]],
            sample_size=50,
            chunk_size=chunk_size)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    trace.print_summary('non_mandatory_tour_frequency', choices, value_counts=True)

    # FIXME - no need to reindex?
    orca.add_column("persons", "non_mandatory_tour_frequency", choices)

    add_dependent_columns("persons", "persons_nmtf")


"""
We have now generated non-mandatory tours, but they are attributes of the
person table - this function creates a "tours" table which
has one row per tour that has been generated (and the person id it is
associated with)
"""


@orca.table(cache=True)
def non_mandatory_tours(persons,
                        non_mandatory_tour_frequency_alts):

    return process_non_mandatory_tours(
        persons.non_mandatory_tour_frequency.dropna(),
        non_mandatory_tour_frequency_alts.local
    )


"""
This is where I'm currently putting computed columns for non_mandatory_tours
- there's an argument this should go in the tables directory in tours.py
"""


@orca.column("non_mandatory_tours")
def destination_in_cbd(non_mandatory_tours, land_use, settings):
    # protection until filled in by destination choice model
    if "destination" not in non_mandatory_tours.columns:
        return pd.Series(False, index=non_mandatory_tours.index)

    s = reindex(land_use.area_type, non_mandatory_tours.destination)
    return s < settings['cbd_threshold']

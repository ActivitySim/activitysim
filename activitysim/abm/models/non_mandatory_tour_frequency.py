# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import orca
import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core.tracing import print_elapsed_time
from activitysim.core import pipeline
from activitysim.core import config

from activitysim.core.util import reindex


from .util.tour_frequency import process_non_mandatory_tours

logger = logging.getLogger(__name__)


@orca.injectable()
def non_mandatory_tour_frequency_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'non_mandatory_tour_frequency.yaml')


@orca.injectable()
def non_mandatory_tour_frequency_spec(configs_dir):
    f = os.path.join(configs_dir, 'non_mandatory_tour_frequency.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.table()
def non_mandatory_tour_frequency_alts(configs_dir):
    f = os.path.join(configs_dir, 'non_mandatory_tour_frequency_alternatives.csv')
    return pd.read_csv(f)


@orca.column("non_mandatory_tour_frequency_alts")
def tot_tours(non_mandatory_tour_frequency_alts):
    return non_mandatory_tour_frequency_alts.local.sum(axis=1)


@orca.step()
def non_mandatory_tour_frequency(persons_merged,
                                 non_mandatory_tour_frequency_alts,
                                 non_mandatory_tour_frequency_spec,
                                 non_mandatory_tour_frequency_settings,
                                 chunk_size,
                                 trace_hh_id):

    """
    This model predicts the frequency of making non-mandatory trips
    (alternatives for this model come from a separate csv file which is
    configured by the user) - these trips include escort, shopping, othmaint,
    othdiscr, eatout, and social trips in various combination.
    """

    t0 = print_elapsed_time()

    choosers = persons_merged.to_frame()
    alts = non_mandatory_tour_frequency_alts.to_frame()

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity.isin(['M', 'N'])]

    logger.info("Running non_mandatory_tour_frequency with %d persons" % len(choosers))

    constants = config.get_model_constants(non_mandatory_tour_frequency_settings)

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for name, segment in choosers.groupby('ptype_cat'):

        logger.info("Running segment '%s' of size %d" % (name, len(segment)))

        choices = asim.interaction_simulate(
            segment,
            alts,
            # notice that we pick the column for the segment for each segment we run
            spec=non_mandatory_tour_frequency_spec[[name]],
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=trace_hh_id and 'non_mandatory_tour_frequency.%s' % name,
            trace_choice_name='non_mandatory_tour_frequency')

        choices_list.append(choices)

        t0 = print_elapsed_time("non_mandatory_tour_frequency.%s" % name, t0)

        # FIXME - force garbage collection
        # mem = asim.memory_info()
        # logger.info('memory_info ptype %s, %s' % (name, mem))

    choices = pd.concat(choices_list)

    # FIXME - no need to reindex?
    orca.add_column("persons", "non_mandatory_tour_frequency", choices)

    create_non_mandatory_tours_table()

    pipeline.add_dependent_columns("persons", "persons_nmtf")

    if trace_hh_id:
        trace_columns = ['non_mandatory_tour_frequency']
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="non_mandatory_tour_frequency",
                         columns=trace_columns,
                         warn_if_empty=True)

"""
We have now generated non-mandatory tours, but they are attributes of the
person table - this function creates a "tours" table which
has one row per tour that has been generated (and the person id it is
associated with)
"""


def create_non_mandatory_tours_table():

    persons = orca.get_table('persons')
    non_mandatory_tour_frequency_alts = orca.get_table('non_mandatory_tour_frequency_alts')

    df = process_non_mandatory_tours(
        persons.non_mandatory_tour_frequency.dropna(),
        non_mandatory_tour_frequency_alts.local
    )

    orca.add_table("non_mandatory_tours", df)
    tracing.register_traceable_table('non_mandatory_tours', df)
    pipeline.get_rn_generator().add_channel(df, 'tours')


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

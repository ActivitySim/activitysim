# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core.simulate import read_model_spec
from activitysim.core.interaction_simulate import interaction_simulate

from activitysim.core import tracing
from activitysim.core.tracing import print_elapsed_time
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from activitysim.core.util import reindex

from .util.tour_frequency import process_non_mandatory_tours

logger = logging.getLogger(__name__)


@inject.injectable()
def non_mandatory_tour_frequency_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'non_mandatory_tour_frequency.yaml')


@inject.injectable()
def non_mandatory_tour_frequency_spec(configs_dir):
    return read_model_spec(configs_dir, 'non_mandatory_tour_frequency.csv')


@inject.table()
def non_mandatory_tour_frequency_alts(configs_dir):
    f = os.path.join(configs_dir, 'non_mandatory_tour_frequency_alternatives.csv')
    return pd.read_csv(f)


@inject.column("non_mandatory_tour_frequency_alts")
def tot_tours(non_mandatory_tour_frequency_alts):
    return non_mandatory_tour_frequency_alts.local.sum(axis=1)


@inject.step()
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

        choices = interaction_simulate(
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
        # mem = memory_info()
        # logger.info('memory_info ptype %s, %s' % (name, mem))

    choices = pd.concat(choices_list)

    # FIXME - no need to reindex?
    inject.add_column("persons", "non_mandatory_tour_frequency", choices)

    create_non_mandatory_tours_table()

    pipeline.add_dependent_columns("persons", "persons_nmtf")

    if trace_hh_id:
        trace_columns = ['non_mandatory_tour_frequency']
        tracing.trace_df(inject.get_table('persons_merged').to_frame(),
                         label="non_mandatory_tour_frequency",
                         columns=trace_columns,
                         warn_if_empty=True)


def create_non_mandatory_tours_table():
    """
    We have now generated non-mandatory tours, but they are attributes of the
    person table - this function creates a "tours" table which
    has one row per tour that has been generated (and the person id it is
    associated with)
    """

    persons = inject.get_table('persons')
    non_mandatory_tour_frequency_alts = inject.get_table('non_mandatory_tour_frequency_alts')

    df = process_non_mandatory_tours(
        persons.non_mandatory_tour_frequency.dropna(),
        non_mandatory_tour_frequency_alts.local
    )

    pipeline.extend_table("tours", df)
    tracing.register_traceable_table('tours', df)
    pipeline.get_rn_generator().add_channel(df, 'tours')

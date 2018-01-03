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


@inject.injectable()
def non_mandatory_tour_frequency_alts(configs_dir):
    f = os.path.join(configs_dir, 'non_mandatory_tour_frequency_alternatives.csv')
    df = pd.read_csv(f)
    return df


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

    non_mandatory_tour_frequency_alts['tot_tours'] = non_mandatory_tour_frequency_alts.sum(axis=1)

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
            non_mandatory_tour_frequency_alts,
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

    tracing.print_summary('non_mandatory_tour_frequency', choices, value_counts=True)

    # FIXME - no need to reindex?
    # FIXME - how about the persons not processed
    inject.add_column("persons", "non_mandatory_tour_frequency", choices)

    create_non_mandatory_tours()

    # add non_mandatory_tour-dependent columns (e.g. tour counts) to persons
    pipeline.add_dependent_columns("persons", "persons_nmtf")

    if trace_hh_id:
        trace_columns = ['non_mandatory_tour_frequency']
        tracing.trace_df(inject.get_table('persons_merged').to_frame(),
                         label="non_mandatory_tour_frequency",
                         columns=trace_columns,
                         warn_if_empty=True)


def create_non_mandatory_tours():
    """
    We have now generated non-mandatory tours, but they are attributes of the person table
    Now we create a "tours" table which has one row per tour that has been generated
    (and the person id it is associated with)
    """

    persons = inject.get_table('persons')
    alts = inject.get_injectable('non_mandatory_tour_frequency_alts')

    df = process_non_mandatory_tours(
        persons.non_mandatory_tour_frequency.dropna(),
        alts
    )

    pipeline.extend_table("tours", df)
    tracing.register_traceable_table('tours', df)
    pipeline.get_rn_generator().add_channel(df, 'tours')

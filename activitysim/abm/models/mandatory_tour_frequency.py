# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util.tour_frequency import process_mandatory_tours
from .util import expressions

logger = logging.getLogger(__name__)


@inject.injectable()
def mandatory_tour_frequency_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'mandatory_tour_frequency.csv')


@inject.injectable()
def mandatory_tour_frequency_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'mandatory_tour_frequency.yaml')


@inject.injectable()
def mandatory_tour_frequency_alternatives(configs_dir):
    # alt file for building tours even though simulation is simple_simulate not interaction_simulate
    f = os.path.join(configs_dir, 'mandatory_tour_frequency_alternatives.csv')
    df = pd.read_csv(f, comment='#')
    df.set_index('alt', inplace=True)
    return df


@inject.step()
def mandatory_tour_frequency(persons_merged,
                             mandatory_tour_frequency_spec,
                             mandatory_tour_frequency_settings,
                             trace_hh_id):
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """

    choosers = persons_merged.to_frame()
    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity == 'M']
    logger.info("Running mandatory_tour_frequency with %d persons" % len(choosers))

    nest_spec = config.get_logit_model_settings(mandatory_tour_frequency_settings)
    constants = config.get_model_constants(mandatory_tour_frequency_settings)

    choices = asim.simple_simulate(
        choosers,
        spec=mandatory_tour_frequency_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_hh_id and 'mandatory_tour_frequency',
        trace_choice_name='mandatory_tour_frequency')

    # convert indexes to alternative names
    choices = pd.Series(
        mandatory_tour_frequency_spec.columns[choices.values],
        index=choices.index).reindex(persons_merged.local.index)

    tracing.print_summary('mandatory_tour_frequency', choices, value_counts=True)

    inject.add_column("persons", "mandatory_tour_frequency", choices)

    create_mandatory_tours()

    # add mandatory_tour-dependent columns (e.g. tour counts) to persons
    pipeline.add_dependent_columns("persons", "persons_mtf")

    if trace_hh_id:
        trace_columns = ['mandatory_tour_frequency']
        tracing.trace_df(inject.get_table('persons_merged').to_frame(),
                         label="mandatory_tour_frequency",
                         columns=trace_columns,
                         warn_if_empty=True)


"""
This reprocesses the choice of index of the mandatory tour frequency
alternatives into an actual dataframe of tours.  Ending format is
the same as got non_mandatory_tours except trip types are "work" and "school"
"""


def create_mandatory_tours():

    # FIXME - move this to body?

    persons = inject.get_table('persons')
    configs_dir = inject.get_injectable('configs_dir')

    persons = persons.to_frame(columns=["mandatory_tour_frequency",
                                        "is_worker", "school_taz", "workplace_taz"])
    persons = persons[~persons.mandatory_tour_frequency.isnull()]

    tour_frequency_alternatives = inject.get_injectable('mandatory_tour_frequency_alternatives')

    tours = process_mandatory_tours(persons, tour_frequency_alternatives)

    expressions.assign_columns(
        df=tours,
        model_settings='annotate_tours_with_dest',
        configs_dir=configs_dir,
        trace_label='create_mandatory_tours')

    pipeline.extend_table("tours", tours)
    tracing.register_traceable_table('tours', tours)
    pipeline.get_rn_generator().add_channel(tours, 'tours')

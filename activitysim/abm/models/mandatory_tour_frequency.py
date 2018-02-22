# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util.tour_frequency import process_mandatory_tours
from .util import expressions

logger = logging.getLogger(__name__)


@inject.injectable()
def mandatory_tour_frequency_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'mandatory_tour_frequency.csv')


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
def mandatory_tour_frequency(persons, persons_merged,
                             mandatory_tour_frequency_spec,
                             mandatory_tour_frequency_settings,
                             mandatory_tour_frequency_alternatives,
                             chunk_size,
                             trace_hh_id):
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """

    trace_label = 'mandatory_tour_frequency'

    choosers = persons_merged.to_frame()
    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity == 'M']
    logger.info("Running mandatory_tour_frequency with %d persons" % len(choosers))

    nest_spec = config.get_logit_model_settings(mandatory_tour_frequency_settings)
    constants = config.get_model_constants(mandatory_tour_frequency_settings)

    choices = simulate.simple_simulate(
        choosers,
        spec=mandatory_tour_frequency_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='mandatory_tour_frequency')

    # convert indexes to alternative names
    choices = pd.Series(
        mandatory_tour_frequency_spec.columns[choices.values],
        index=choices.index).reindex(persons_merged.local.index)

    persons = persons.to_frame()

    # need to reindex as we only handled persons with cdap_activity == 'M'
    persons['mandatory_tour_frequency'] = choices.reindex(persons.index)

    """
    This reprocesses the choice of index of the mandatory tour frequency
    alternatives into an actual dataframe of tours.  Ending format is
    the same as got non_mandatory_tours except trip types are "work" and "school"
    """
    mandatory_tours = process_mandatory_tours(
        persons=persons[~persons.mandatory_tour_frequency.isnull()],
        mandatory_tour_frequency_alts=mandatory_tour_frequency_alternatives
    )

    expressions.assign_columns(
        df=mandatory_tours,
        model_settings='annotate_tours_with_dest',
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_tours_with_dest'))

    tours = pipeline.extend_table("tours", mandatory_tours)
    tracing.register_traceable_table('tours', tours)
    pipeline.get_rn_generator().add_channel(mandatory_tours, 'tours')

    expressions.assign_columns(
        df=persons,
        model_settings=mandatory_tour_frequency_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)

    tracing.print_summary('mandatory_tour_frequency', persons.mandatory_tour_frequency,
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(mandatory_tours,
                         label="mandatory_tour_frequency.mandatory_tours",
                         warn_if_empty=True)

        tracing.trace_df(inject.get_table('persons').to_frame(),
                         label="mandatory_tour_frequency.persons",
                         warn_if_empty=True)

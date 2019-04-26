# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

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


def add_null_results(trace_label, mandatory_tour_frequency_settings):
    logger.info("Skipping %s: add_null_results", trace_label)

    persons = inject.get_table('persons').to_frame()
    persons['mandatory_tour_frequency'] = ''

    tours = pd.DataFrame()
    tours['tour_category'] = None
    tours['tour_type'] = None
    tours['person_id'] = None
    tours.index.name = 'tour_id'
    pipeline.replace_table("tours", tours)

    expressions.assign_columns(
        df=persons,
        model_settings=mandatory_tour_frequency_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)


@inject.step()
def mandatory_tour_frequency(persons_merged,
                             chunk_size,
                             trace_hh_id):
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """
    trace_label = 'mandatory_tour_frequency'

    model_settings = config.read_model_settings('mandatory_tour_frequency.yaml')
    model_spec = simulate.read_model_spec(file_name='mandatory_tour_frequency.csv')
    alternatives = simulate.read_model_alts(
        config.config_file_path('mandatory_tour_frequency_alternatives.csv'), set_index='alt')

    choosers = persons_merged.to_frame()
    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity == 'M']
    logger.info("Running mandatory_tour_frequency with %d persons", len(choosers))

    # - if no mandatory tours
    if choosers.shape[0] == 0:
        add_null_results(trace_label, model_settings)
        return

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_dict = {}

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='mandatory_tour_frequency')

    # convert indexes to alternative names
    choices = pd.Series(
        model_spec.columns[choices.values],
        index=choices.index).reindex(persons_merged.local.index)

    # - create mandatory tours
    """
    This reprocesses the choice of index of the mandatory tour frequency
    alternatives into an actual dataframe of tours.  Ending format is
    the same as got non_mandatory_tours except trip types are "work" and "school"
    """
    choosers['mandatory_tour_frequency'] = choices
    mandatory_tours = process_mandatory_tours(
        persons=choosers,
        mandatory_tour_frequency_alts=alternatives
    )

    tours = pipeline.extend_table("tours", mandatory_tours)
    tracing.register_traceable_table('tours', mandatory_tours)
    pipeline.get_rn_generator().add_channel('tours', mandatory_tours)

    # - annotate persons
    persons = inject.get_table('persons').to_frame()

    # need to reindex as we only handled persons with cdap_activity == 'M'
    persons['mandatory_tour_frequency'] = choices.reindex(persons.index).fillna('').astype(str)

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)

    tracing.print_summary('mandatory_tour_frequency', persons.mandatory_tour_frequency,
                          value_counts=True)

    if trace_hh_id:
        tracing.trace_df(mandatory_tours,
                         label="mandatory_tour_frequency.mandatory_tours",
                         warn_if_empty=True)

        tracing.trace_df(persons,
                         label="mandatory_tour_frequency.persons",
                         warn_if_empty=True)

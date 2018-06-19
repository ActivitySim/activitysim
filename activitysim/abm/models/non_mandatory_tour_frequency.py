# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core.simulate import read_model_spec
from activitysim.core.interaction_simulate import interaction_simulate

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util import expressions
from .util.overlap import person_max_window

from activitysim.abm.tables.constants import PTYPE_NAME

from .util.tour_frequency import process_non_mandatory_tours

logger = logging.getLogger(__name__)


@inject.injectable()
def non_mandatory_tour_frequency_spec(configs_dir):
    return read_model_spec(configs_dir, 'non_mandatory_tour_frequency.csv')


@inject.injectable()
def non_mandatory_tour_frequency_alts(configs_dir):
    f = os.path.join(configs_dir, 'non_mandatory_tour_frequency_alternatives.csv')
    df = pd.read_csv(f, comment='#')
    return df


@inject.step()
def non_mandatory_tour_frequency(persons, persons_merged,
                                 non_mandatory_tour_frequency_alts,
                                 non_mandatory_tour_frequency_spec,
                                 chunk_size,
                                 trace_hh_id):

    """
    This model predicts the frequency of making non-mandatory trips
    (alternatives for this model come from a separate csv file which is
    configured by the user) - these trips include escort, shopping, othmaint,
    othdiscr, eatout, and social trips in various combination.
    """

    trace_label = 'non_mandatory_tour_frequency'
    model_settings = config.read_model_settings('non_mandatory_tour_frequency.yaml')

    choosers = persons_merged.to_frame()

    # FIXME kind of tacky both that we know to add this here and del it below
    non_mandatory_tour_frequency_alts['tot_tours'] = non_mandatory_tour_frequency_alts.sum(axis=1)

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_dict = {
            'person_max_window': person_max_window
        }

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity.isin(['M', 'N'])]

    logger.info("Running non_mandatory_tour_frequency with %d persons" % len(choosers))

    constants = config.get_model_constants(model_settings)

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for ptype, segment in choosers.groupby('ptype'):

        name = PTYPE_NAME[ptype]

        # pick the spec column for the segment
        spec = non_mandatory_tour_frequency_spec[[name]]

        # drop any zero-valued rows
        spec = spec[spec[name] != 0]

        logger.info("Running segment '%s' of size %d" % (name, len(segment)))

        choices = interaction_simulate(
            segment,
            non_mandatory_tour_frequency_alts,
            spec=spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label='non_mandatory_tour_frequency.%s' % name,
            trace_choice_name='non_mandatory_tour_frequency')

        choices_list.append(choices)

        # FIXME - force garbage collection?
        # force_garbage_collect()

    choices = pd.concat(choices_list)

    persons = persons.to_frame()

    # need to reindex as we only handled persons with cdap_activity in ['M', 'N']
    persons['non_mandatory_tour_frequency'] = choices.reindex(persons.index)

    """
    We have now generated non-mandatory tours, but they are attributes of the person table
    Now we create a "tours" table which has one row per tour that has been generated
    (and the person id it is associated with)
    """
    del non_mandatory_tour_frequency_alts['tot_tours']  # del tot_tours column we added above
    non_mandatory_tours = process_non_mandatory_tours(
        persons[~persons.mandatory_tour_frequency.isnull()],
        non_mandatory_tour_frequency_alts,
    )

    tours = pipeline.extend_table("tours", non_mandatory_tours)
    tracing.register_traceable_table('tours', tours)
    pipeline.get_rn_generator().add_channel(non_mandatory_tours, 'tours')

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=trace_label)

    pipeline.replace_table("persons", persons)

    tracing.print_summary('non_mandatory_tour_frequency',
                          persons.non_mandatory_tour_frequency, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(non_mandatory_tours,
                         label="non_mandatory_tour_frequency.non_mandatory_tours",
                         warn_if_empty=True)

        tracing.trace_df(choosers,
                         label="non_mandatory_tour_frequency.choosers",
                         warn_if_empty=True)

        tracing.trace_df(persons,
                         label="non_mandatory_tour_frequency.annotated_persons",
                         warn_if_empty=True)

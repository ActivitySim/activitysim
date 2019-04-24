# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import numpy as np
import pandas as pd

from activitysim.core.interaction_simulate import interaction_simulate

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import simulate
from activitysim.core import logit

from .util import expressions
from .util.overlap import person_max_window

from activitysim.abm.tables.constants import PTYPE_NAME

from .util.tour_frequency import process_non_mandatory_tours

logger = logging.getLogger(__name__)


def extension_probs():
    f = config.config_file_path('non_mandatory_tour_frequency_extension_probs.csv')
    df = pd.read_csv(f, comment='#')

    # convert cum probs to individual probs
    df['2_tours'] = df['2_tours'] - df['1_tours']
    df['1_tours'] = df['1_tours'] - df['0_tours']

    return df


def extend_tour_counts(persons, tour_counts, alternatives, trace_hh_id, trace_label):
    """
    extend tour counts based on a probability table

    counts can only be extended if original count is between 1 and 4
    and tours can only be extended if their count is at the max possible
    (e.g. 2 for escort, 1 otherwise) so escort might be increased to 3 or 4
    and other tour types might be increased to 2 or 3

    Parameters
    ----------
    persons: pandas dataframe
        (need this for join columns)
    tour_counts: pandas dataframe
        one row per person, once column per tour_type
    alternatives
        alternatives from nmtv interaction_simulate
        only need this to know max possible frequency for a tour type
    trace_hh_id
    trace_label

    Returns
    -------
    extended tour_counts


    tour_counts looks like this:
               escort  shopping  othmaint  othdiscr    eatout    social
    parent_id
    2588676         2         0         0         1         1         0
    2588677         0         1         0         1         0         0

    """

    assert tour_counts.index.name == persons.index.name

    PROBABILITY_COLUMNS = ['0_tours', '1_tours', '2_tours']
    JOIN_COLUMNS = ['ptype', 'has_mandatory_tour', 'has_joint_tour']
    TOUR_TYPE_COL = 'nonmandatory_tour_type'

    probs_spec = extension_probs()
    persons = persons[JOIN_COLUMNS]

    # only extend if there are 1 - 4 non_mandatory tours to start with
    extend_tour_counts = tour_counts.sum(axis=1).between(1, 4)
    if not extend_tour_counts.any():
        return tour_counts

    have_trace_targets = trace_hh_id and tracing.has_trace_targets(extend_tour_counts)

    for i, tour_type in enumerate(alternatives.columns):

        i_tour_type = i + 1  # (probs_spec nonmandatory_tour_type column is 1-based)
        tour_type_trace_label = tracing.extend_trace_label(trace_label, tour_type)

        # - only extend tour if frequency is max possible frequency for this tour type
        tour_type_is_maxed = \
            extend_tour_counts & (tour_counts[tour_type] == alternatives[tour_type].max())
        maxed_tour_count_idx = tour_counts.index[tour_type_is_maxed]

        if len(maxed_tour_count_idx) == 0:
            continue

        # - get extension probs for tour_type
        choosers = pd.merge(
            persons.loc[maxed_tour_count_idx],
            probs_spec[probs_spec[TOUR_TYPE_COL] == i_tour_type],
            on=JOIN_COLUMNS,
            how='left'
        ).set_index(maxed_tour_count_idx)
        assert choosers.index.name == tour_counts.index.name

        # - random choice of extension magnituce based on relative probs
        choices, rands = logit.make_choices(
            choosers[PROBABILITY_COLUMNS],
            trace_label=tour_type_trace_label,
            trace_choosers=choosers)

        # - extend tour_count (0-based prob alternative choice equals magnitude of extension)
        if choices.any():
            tour_counts.loc[choices.index, tour_type] += choices

        if have_trace_targets:
            tracing.trace_df(choices,
                             tracing.extend_trace_label(tour_type_trace_label, 'choices'),
                             columns=[None, 'choice'])
            tracing.trace_df(rands,
                             tracing.extend_trace_label(tour_type_trace_label, 'rands'),
                             columns=[None, 'rand'])

    return tour_counts


@inject.step()
def non_mandatory_tour_frequency(persons, persons_merged,
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
    model_spec = simulate.read_model_spec(file_name='non_mandatory_tour_frequency.csv')

    alternatives = simulate.read_model_alts(
        config.config_file_path('non_mandatory_tour_frequency_alternatives.csv'),
        set_index=None)

    choosers = persons_merged.to_frame()

    # FIXME kind of tacky both that we know to add this here and del it below
    # 'tot_tours' is used in model_spec expressions
    alternatives['tot_tours'] = alternatives.sum(axis=1)

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
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

    logger.info("Running non_mandatory_tour_frequency with %d persons", len(choosers))

    constants = config.get_model_constants(model_settings)

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for ptype, segment in choosers.groupby('ptype'):

        name = PTYPE_NAME[ptype]

        # pick the spec column for the segment
        spec = model_spec[[name]]

        # drop any zero-valued rows
        spec = spec[spec[name] != 0]

        logger.info("Running segment '%s' of size %d", name, len(segment))

        choices = interaction_simulate(
            segment,
            alternatives,
            spec=spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label='non_mandatory_tour_frequency.%s' % name,
            trace_choice_name='non_mandatory_tour_frequency')

        choices_list.append(choices)

        # FIXME - force garbage collection?
        # force_garbage_collect()

    choices = pd.concat(choices_list)

    del alternatives['tot_tours']  # del tot_tours column we added above

    # - add non_mandatory_tour_frequency column to persons
    persons = persons.to_frame()
    # need to reindex as we only handled persons with cdap_activity in ['M', 'N']
    # (we expect there to be an alt with no tours - which we can use to backfill non-travelers)
    no_tours_alt = (alternatives.sum(axis=1) == 0).index[0]
    persons['non_mandatory_tour_frequency'] = \
        choices.reindex(persons.index).fillna(no_tours_alt).astype(np.int8)

    """
    We have now generated non-mandatory tours, but they are attributes of the person table
    Now we create a "tours" table which has one row per tour that has been generated
    (and the person id it is associated with)
    """

    # - get counts of each of the alternatives (so we can extend)
    # (choices is just the index values for the chosen alts)
    """
               escort  shopping  othmaint  othdiscr    eatout    social
    parent_id
    2588676         2         0         0         1         1         0
    2588677         0         1         0         1         0         0
    """
    tour_counts = alternatives.loc[choices]
    tour_counts.index = choices.index  # assign person ids to the index

    prev_tour_count = tour_counts.sum().sum()

    # - extend_tour_counts
    tour_counts = extend_tour_counts(choosers, tour_counts, alternatives,
                                     trace_hh_id,
                                     tracing.extend_trace_label(trace_label, 'extend_tour_counts'))

    extended_tour_count = tour_counts.sum().sum()

    logging.info("extend_tour_counts increased nmtf tour count by %s from %s to %s" %
                 (extended_tour_count - prev_tour_count, prev_tour_count, extended_tour_count))

    # - create the non_mandatory tours
    non_mandatory_tours = process_non_mandatory_tours(persons, tour_counts)
    assert len(non_mandatory_tours) == extended_tour_count

    pipeline.extend_table("tours", non_mandatory_tours)

    tracing.register_traceable_table('tours', non_mandatory_tours)
    pipeline.get_rn_generator().add_channel('tours', non_mandatory_tours)

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

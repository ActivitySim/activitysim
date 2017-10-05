# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline


from .util.vectorize_tour_scheduling import vectorize_tour_scheduling


logger = logging.getLogger(__name__)


@inject.injectable()
def tdd_non_mandatory_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_nonmandatory.csv')


@inject.injectable()
def non_mandatory_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'non_mandatory_tour_scheduling.yaml')


@inject.step()
def non_mandatory_tour_scheduling(tours,
                                  persons_merged,
                                  tdd_alts,
                                  tdd_non_mandatory_spec,
                                  non_mandatory_tour_scheduling_settings,
                                  chunk_size,
                                  trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for non-mandatory tours
    """

    tours = tours.to_frame()
    persons_merged = persons_merged.to_frame()

    non_mandatory_tours = tours[~tours.mandatory]

    logger.info("Running non_mandatory_tour_scheduling with %d tours" % len(tours))

    constants = config.get_model_constants(non_mandatory_tour_scheduling_settings)

    tdd_choices = vectorize_tour_scheduling(non_mandatory_tours, persons_merged,
                                            tdd_alts, tdd_non_mandatory_spec,
                                            constants=constants,
                                            chunk_size=chunk_size,
                                            trace_label='non_mandatory_tour_scheduling')

    # add tdd_choices columns to tours
    for c in tdd_choices.columns:
        tours.loc[tdd_choices.index, c] = tdd_choices[c]

    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(non_mandatory_tours,
                         label="non_mandatory_tour_scheduling",
                         slicer='person_id',
                         index_label='tour_id',
                         columns=None,
                         warn_if_empty=True)

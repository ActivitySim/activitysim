# ActivitySim
# See full license in LICENSE.txt.


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
def mandatory_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'mandatory_tour_scheduling.yaml')


@inject.injectable()
def tdd_work_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_work.csv')


@inject.injectable()
def tdd_school_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_school.csv')


# I think it's easier to do this in one model so you can merge the two
# resulting series together right away
@inject.step()
def mandatory_tour_scheduling(tours,
                              persons_merged,
                              tdd_alts,
                              tdd_school_spec,
                              tdd_work_spec,
                              mandatory_tour_scheduling_settings,
                              chunk_size,
                              trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for mandatory tours
    """

    tours = tours.to_frame()
    persons_merged = persons_merged.to_frame()
    mandatory_tours = tours[tours.mandatory]

    constants = config.get_model_constants(mandatory_tour_scheduling_settings)

    # - school tours
    school_tours = mandatory_tours[mandatory_tours.tour_type == "school"]
    logger.info("Running mandatory_tour_scheduling school_tours with %d tours" % len(school_tours))
    school_choices = vectorize_tour_scheduling(
        school_tours, persons_merged,
        tdd_alts, tdd_school_spec,
        constants=constants,
        chunk_size=chunk_size,
        trace_label='mandatory_tour_scheduling.school')

    # - work tours
    work_tours = mandatory_tours[mandatory_tours.tour_type == "work"]
    logger.info("Running %d work tour scheduling choices" % len(work_tours))
    work_choices = vectorize_tour_scheduling(
        work_tours, persons_merged,
        tdd_alts, tdd_work_spec,
        constants=constants,
        chunk_size=chunk_size,
        trace_label='mandatory_tour_scheduling.work')

    tdd_choices = pd.concat([school_choices, work_choices])

    # add tdd_choices columns to tours
    for c in tdd_choices.columns:
        tours.loc[tdd_choices.index, c] = tdd_choices[c]

    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(mandatory_tours,
                         label="mandatory_tour_scheduling",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)

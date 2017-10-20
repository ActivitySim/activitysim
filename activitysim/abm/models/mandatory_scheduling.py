# ActivitySim
# See full license in LICENSE.txt.


import logging

import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt

from .util.vectorize_tour_scheduling import vectorize_tour_scheduling

logger = logging.getLogger(__name__)

DUMP = True


@inject.injectable()
def mandatory_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'mandatory_tour_scheduling.yaml')


@inject.injectable()
def tdd_work_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_work.csv')


@inject.injectable()
def tdd_school_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_school.csv')


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

    trace_label = 'mandatory_tour_scheduling'
    constants = config.get_model_constants(mandatory_tour_scheduling_settings)

    logger.info("Running mandatory_tour_scheduling with %d tours" % len(tours))
    tdd_choices = vectorize_tour_scheduling(
        tours, persons_merged,
        tdd_alts,
        spec={'work': tdd_work_spec, 'school': tdd_school_spec},
        constants=constants,
        chunk_size=chunk_size,
        trace_label=trace_label)

    # add tdd_choices columns to tours
    for c in tdd_choices.columns:
        tours.loc[tdd_choices.index, c] = tdd_choices[c]

    pipeline.replace_table("tours", tours)

    mandatory_tours = tours[tours.mandatory]

    tracing.dump_df(DUMP,
                    tt.tour_map(persons_merged, mandatory_tours, tdd_alts),
                    trace_label, 'tour_map')

    if trace_hh_id:
        tracing.trace_df(mandatory_tours,
                         label="mandatory_tour_scheduling",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)

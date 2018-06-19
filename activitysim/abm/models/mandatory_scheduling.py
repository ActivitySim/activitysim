# ActivitySim
# See full license in LICENSE.txt.


import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt

from .util import expressions
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling
from activitysim.core.util import assign_in_place


logger = logging.getLogger(__name__)

DUMP = False


@inject.injectable()
def tour_scheduling_work_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'tour_scheduling_work.csv')


@inject.injectable()
def tour_scheduling_school_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'tour_scheduling_school.csv')


@inject.step()
def mandatory_tour_scheduling(tours,
                              persons_merged,
                              tdd_alts,
                              tour_scheduling_work_spec,
                              tour_scheduling_school_spec,
                              chunk_size,
                              trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for mandatory tours
    """
    trace_label = 'mandatory_tour_scheduling'
    model_settings = config.read_model_settings('mandatory_tour_scheduling.yaml')

    tours = tours.to_frame()
    persons_merged = persons_merged.to_frame()
    mandatory_tours = tours[tours.tour_category == 'mandatory']

    # - if no mandatory_tours
    if mandatory_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    model_constants = config.get_model_constants(model_settings)

    logger.info("Running mandatory_tour_scheduling with %d tours" % len(tours))
    tdd_choices = vectorize_tour_scheduling(
        mandatory_tours, persons_merged,
        tdd_alts,
        spec={'work': tour_scheduling_work_spec, 'school': tour_scheduling_school_spec},
        constants=model_constants,
        chunk_size=chunk_size,
        trace_label=trace_label)

    assign_in_place(tours, tdd_choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    mandatory_tours = tours[tours.tour_category == 'mandatory']

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

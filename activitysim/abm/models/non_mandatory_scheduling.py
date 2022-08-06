# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate
from activitysim.core import timetable as tt
from activitysim.core import tracing
from activitysim.core.util import assign_in_place

from .util import estimation
from .util.tour_scheduling import run_tour_scheduling
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling

logger = logging.getLogger(__name__)
DUMP = False


@inject.step()
def non_mandatory_tour_scheduling(
    tours, persons_merged, tdd_alts, chunk_size, trace_hh_id
):
    """
    This model predicts the departure time and duration of each activity for non-mandatory tours
    """

    model_name = "non_mandatory_tour_scheduling"
    trace_label = model_name

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    # - if no mandatory_tours
    if non_mandatory_tours.shape[0] == 0:
        tracing.no_results(model_name)
        return

    tour_segment_col = None

    choices = run_tour_scheduling(
        model_name,
        non_mandatory_tours,
        persons_merged,
        tdd_alts,
        tour_segment_col,
        chunk_size,
        trace_hh_id,
    )

    assign_in_place(tours, choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    tracing.dump_df(
        DUMP,
        tt.tour_map(persons_merged, non_mandatory_tours, tdd_alts),
        trace_label,
        "tour_map",
    )

    if trace_hh_id:
        tracing.trace_df(
            non_mandatory_tours,
            label=trace_label,
            slicer="person_id",
            index_label="tour_id",
            columns=None,
            warn_if_empty=True,
        )

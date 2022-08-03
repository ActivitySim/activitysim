# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate
from activitysim.core import timetable as tt
from activitysim.core import tracing
from activitysim.core.util import assign_in_place, reindex

from .util import estimation
from .util import vectorize_tour_scheduling as vts
from .util.tour_scheduling import run_tour_scheduling

logger = logging.getLogger(__name__)

DUMP = False


@inject.step()
def mandatory_tour_scheduling(tours, persons_merged, tdd_alts, chunk_size, trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for mandatory tours
    """

    model_name = "mandatory_tour_scheduling"
    trace_label = model_name

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    mandatory_tours = tours[tours.tour_category == "mandatory"]

    # - if no mandatory_tours
    if mandatory_tours.shape[0] == 0:
        tracing.no_results(model_name)
        return

    # - add tour segmentation column
    # mtctm1 segments mandatory_scheduling spec by tour_type
    # (i.e. there are different specs for work and school tour_types)
    # mtctm1 logsum coefficients are segmented by primary_purpose
    # (i.e. there are different logsum coefficients for work, school, univ primary_purposes
    # for simplicity managing these different segmentation schemes,
    # we conflate them by segmenting tour processing to align with primary_purpose
    tour_segment_col = "mandatory_tour_seg"
    assert tour_segment_col not in mandatory_tours
    is_university_tour = (mandatory_tours.tour_type == "school") & reindex(
        persons_merged.is_university, mandatory_tours.person_id
    )
    mandatory_tours[tour_segment_col] = mandatory_tours.tour_type.where(
        ~is_university_tour, "univ"
    )

    choices = run_tour_scheduling(
        model_name,
        mandatory_tours,
        persons_merged,
        tdd_alts,
        tour_segment_col,
        chunk_size,
        trace_hh_id,
    )

    assign_in_place(tours, choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    mandatory_tours = tours[tours.tour_category == "mandatory"]

    tracing.dump_df(
        DUMP,
        tt.tour_map(persons_merged, mandatory_tours, tdd_alts),
        trace_label,
        "tour_map",
    )

    if trace_hh_id:
        tracing.trace_df(
            mandatory_tours,
            label=trace_label,
            slicer="person_id",
            index_label="tour",
            columns=None,
            warn_if_empty=True,
        )

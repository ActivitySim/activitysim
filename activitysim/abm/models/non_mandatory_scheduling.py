# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util.tour_scheduling import run_tour_scheduling
from activitysim.core import timetable as tt
from activitysim.core import tracing, workflow
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)
DUMP = False


@workflow.step
def non_mandatory_tour_scheduling(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    tdd_alts: pd.DataFrame,
) -> None:
    """
    This model predicts the departure time and duration of each activity for non-mandatory tours
    """

    model_name = "non_mandatory_tour_scheduling"
    trace_label = model_name
    trace_hh_id = state.settings.trace_hh_id
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    # - if no mandatory_tours
    if non_mandatory_tours.shape[0] == 0:
        tracing.no_results(model_name)
        return

    tour_segment_col = None

    choices = run_tour_scheduling(
        state,
        model_name,
        non_mandatory_tours,
        persons_merged,
        tdd_alts,
        tour_segment_col,
    )

    assign_in_place(
        tours, choices, state.settings.downcast_int, state.settings.downcast_float
    )
    state.add_table("tours", tours)

    # updated df for tracing
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    state.tracing.dump_df(
        DUMP,
        tt.tour_map(persons_merged, non_mandatory_tours, tdd_alts),
        trace_label,
        "tour_map",
    )

    if trace_hh_id:
        state.tracing.trace_df(
            non_mandatory_tours,
            label=trace_label,
            slicer="person_id",
            index_label="tour_id",
            columns=None,
            warn_if_empty=True,
        )

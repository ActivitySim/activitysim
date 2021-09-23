# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate
from activitysim.core import timetable as tt
from activitysim.core import tracing
from activitysim.core.util import assign_in_place

from .util import estimation
from .util.vectorize_tour_scheduling import vectorize_subtour_scheduling

logger = logging.getLogger(__name__)

DUMP = False


@inject.step()
def atwork_subtour_scheduling(
    tours, persons_merged, tdd_alts, skim_dict, chunk_size, trace_hh_id
):
    """
    This model predicts the departure time and duration of each activity for at work subtours tours
    """

    trace_label = "atwork_subtour_scheduling"
    model_settings_file_name = "tour_scheduling_atwork.yaml"

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == "atwork"]

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("atwork_subtour_scheduling")

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    persons_merged = persons_merged.to_frame()

    logger.info("Running %s with %d tours", trace_label, len(subtours))

    # preprocessor
    constants = config.get_model_constants(model_settings)
    od_skim_wrapper = skim_dict.wrap("origin", "destination")
    skims = {
        "od_skims": od_skim_wrapper,
    }
    expressions.annotate_preprocessors(
        subtours, constants, skims, model_settings, trace_label
    )

    # parent_tours table with columns ['tour_id', 'tdd'] index = tour_id
    parent_tour_ids = subtours.parent_tour_id.astype(np.int64).unique()
    parent_tours = pd.DataFrame({"tour_id": parent_tour_ids}, index=parent_tour_ids)
    parent_tours = parent_tours.merge(tours[["tdd"]], left_index=True, right_index=True)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        # we don't need to update timetable because subtours are scheduled inside work trip windows

    choices = vectorize_subtour_scheduling(
        parent_tours,
        subtours,
        persons_merged,
        tdd_alts,
        model_spec,
        model_settings,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_label=trace_label,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "tours", "tdd")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # choices are tdd alternative ids
    # we want to add start, end, and duration columns to tours, which we have in tdd_alts table
    tdd_choices = pd.merge(
        choices.to_frame("tdd"), tdd_alts, left_on=["tdd"], right_index=True, how="left"
    )

    assign_in_place(tours, tdd_choices)
    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(
            tours[tours.tour_category == "atwork"],
            label="atwork_subtour_scheduling",
            slicer="person_id",
            index_label="tour_id",
            columns=None,
        )

    if DUMP:
        subtours = tours[tours.tour_category == "atwork"]
        parent_tours = tours[tours.index.isin(subtours.parent_tour_id)]

        tracing.dump_df(DUMP, subtours, trace_label, "sub_tours")
        tracing.dump_df(DUMP, parent_tours, trace_label, "parent_tours")

        parent_tours["parent_tour_id"] = parent_tours.index
        subtours = pd.concat([parent_tours, subtours])
        tracing.dump_df(
            DUMP,
            tt.tour_map(
                parent_tours, subtours, tdd_alts, persons_id_col="parent_tour_id"
            ),
            trace_label,
            "tour_map",
        )

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.tour_frequency import process_atwork_subtours
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)

logger = logging.getLogger(__name__)


def add_null_results(whale, trace_label, tours):
    logger.info("Skipping %s: add_null_results", trace_label)
    tours["atwork_subtour_frequency"] = np.nan
    whale.add_table("tours", tours)


@workflow.step
def atwork_subtour_frequency(
    whale: workflow.Whale,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
):
    """
    This model predicts the frequency of making at-work subtour tours
    (alternatives for this model come from a separate csv file which is
    configured by the user).
    """

    trace_label = "atwork_subtour_frequency"
    model_settings_file_name = "atwork_subtour_frequency.yaml"
    trace_hh_id = whale.settings.trace_hh_id
    work_tours = tours[tours.tour_type == "work"]

    # - if no work_tours
    if len(work_tours) == 0:
        add_null_results(whale, trace_label, tours)
        return

    model_settings = whale.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(whale, "atwork_subtour_frequency")

    model_spec = whale.filesystem.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = whale.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        whale, model_spec, coefficients_df, estimator
    )

    alternatives = simulate.read_model_alts(
        whale, "atwork_subtour_frequency_alternatives.csv", set_index="alt"
    )

    # merge persons into work_tours
    work_tours = pd.merge(
        work_tours, persons_merged, left_on="person_id", right_index=True
    )

    logger.info("Running atwork_subtour_frequency with %d work tours", len(work_tours))

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        expressions.assign_columns(
            whale,
            df=work_tours,
            model_settings=preprocessor_settings,
            trace_label=trace_label,
        )

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(work_tours)

    choices = simulate.simple_simulate(
        whale,
        choosers=work_tours,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="atwork_subtour_frequency",
        estimator=estimator,
    )

    # convert indexes to alternative names
    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "tours", "atwork_subtour_frequency"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # add atwork_subtour_frequency column to tours
    # reindex since we are working with a subset of tours
    tours["atwork_subtour_frequency"] = choices.reindex(tours.index)
    whale.add_table("tours", tours)

    # - create atwork_subtours based on atwork_subtour_frequency choice names
    work_tours = tours[tours.tour_type == "work"]
    assert not work_tours.atwork_subtour_frequency.isnull().any()

    subtours = process_atwork_subtours(whale, work_tours, alternatives)

    tours = whale.extend_table("tours", subtours)

    whale.tracing.register_traceable_table("tours", subtours)
    whale.get_rn_generator().add_channel("tours", subtours)

    tracing.print_summary(
        "atwork_subtour_frequency", tours.atwork_subtour_frequency, value_counts=True
    )

    if trace_hh_id:
        whale.tracing.trace_df(tours, label="atwork_subtour_frequency.tours")

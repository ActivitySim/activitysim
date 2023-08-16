# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)

logger = logging.getLogger(__name__)


@workflow.step
def free_parking(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings_file_name: str = "free_parking.yaml",
    model_settings: dict[str, Any] = workflow.from_yaml("free_parking.yaml"),
    trace_label: str = "free_parking",
) -> None:
    """
    Determine for each person whether they have free parking available at work.

    Parameters
    ----------
    state : workflow.State
    persons_merged : DataFrame
    persons : DataFrame
    model_settings_file_name : str
        This filename is used to write settings files in estimation mode.
    model_settings : dict
        The settings used in this model component.
    trace_label : str

    Returns
    -------

    """

    choosers = pd.DataFrame(persons_merged)
    choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "free_parking")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    model_spec = state.filesystem.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="free_parking_at_work",
        estimator=estimator,
    )

    free_parking_alt = model_settings["FREE_PARKING_ALT"]
    choices = choices == free_parking_alt

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "free_parking_at_work"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["free_parking_at_work"] = (
        choices.reindex(persons.index).fillna(0).astype(bool)
    )

    state.add_table("persons", persons)

    tracing.print_summary(
        "free_parking", persons.free_parking_at_work, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

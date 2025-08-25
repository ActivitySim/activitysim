# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger("activitysim")


class TelecommuteStatusSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `telecommute_status` component.
    """

    TELECOMMUTE_ALT: int
    """Value that specifies if the worker is telecommuting on the simulation day."""

    CHOOSER_FILTER_COLUMN_NAME: str = "is_worker"
    """Column name in the dataframe to represent worker."""


@workflow.step
def telecommute_status(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: TelecommuteStatusSettings | None = None,
    model_settings_file_name: str = "telecommute_status.yaml",
    trace_label: str = "telecommute_status",
) -> None:
    """
    This model predicts whether a person (worker) telecommutes on the simulation day.
    The output from this model is TRUE (if telecommutes) or FALSE (if does not telecommute).
    """
    if model_settings is None:
        model_settings = TelecommuteStatusSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = persons_merged
    chooser_filter_column_name = model_settings.CHOOSER_FILTER_COLUMN_NAME
    choosers = choosers[(choosers[chooser_filter_column_name])]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "telecommute_status")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    expressions.annotate_preprocessors(
        state,
        df=choosers,
        locals_dict=constants,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )

    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
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
        trace_choice_name="is_telecommuting",
        estimator=estimator,
        compute_settings=model_settings.compute_settings,
    )

    telecommute_alt = model_settings.TELECOMMUTE_ALT
    choices = choices == telecommute_alt

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "persons", "is_telecommuting")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["is_telecommuting"] = choices.reindex(persons.index).fillna(0).astype(bool)

    state.add_table("persons", persons)

    tracing.print_summary(
        "telecommute_status", persons.is_telecommuting, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

    expressions.annotate_tables(
        state,
        locals_dict=constants,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )

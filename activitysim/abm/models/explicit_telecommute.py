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
from activitysim.core.configuration.logit import LogitComponentSettings, TourLocationComponentSettings

logger = logging.getLogger("activitysim")


class ExplicitTelecommuteSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `explicit_telecommute` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    TELECOMMUTE_ALT: int
    """The code for telecommuting"""

    CHOOSER_FILTER_COLUMN_NAME: str = "is_worker"
    """Column name in the dataframe to represent worker."""



@workflow.step
def explicit_telecommute(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: ExplicitTelecommuteSettings | None = None,
    model_settings_file_name: str = "explicit_telecommute.yaml",
    trace_label: str = "explicit_telecommute",
) -> None:
    """
    This model predicts whether a person (worker) telecommutes on the simulation day.
    The output from this model is TRUE (if telecommutes) or FALSE (works away from home).
    The workplace location choice is overridden for workers who telecommute
    and set to -1.
    """
    if model_settings is None:
        model_settings = ExplicitTelecommuteSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    workplace_model_settings = TourLocationComponentSettings.read_settings_file(
        state.filesystem,
        "workplace_location.yaml",
    )

    choosers = persons_merged
    chooser_filter_column_name = model_settings.CHOOSER_FILTER_COLUMN_NAME
    choosers = choosers[(choosers[chooser_filter_column_name])]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "explicit_telecommute")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.preprocessor
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
        choices = estimator.get_survey_values(
            choices, 
            "persons", 
            "is_telecommuting"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["is_telecommuting"] = (
        choices.reindex(persons.index).fillna(0).astype(bool)
    ) 
    # save original workplace_zone_id values to a new variable out_of_home_work_location
    # setting workplace_zone_id to home_zone_id if person is telecommuting on simulation day
    workplace_location = workplace_model_settings.DEST_CHOICE_COLUMN_NAME
    home_zone = workplace_model_settings.CHOOSER_ORIG_COL_NAME
    if workplace_location in persons.columns:
        persons['out_of_home_work_location'] = persons[workplace_location]
        persons[workplace_location] = np.where(
            persons.is_telecommuting == True, 
            persons[home_zone], 
            persons[workplace_location]
        )

    state.add_table("persons", persons)

    tracing.print_summary(
        "explicit_telecommute", 
        persons.is_telecommuting, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

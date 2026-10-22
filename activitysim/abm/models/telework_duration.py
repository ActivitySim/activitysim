# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from pydantic import model_validator

from activitysim.core import (
    config,
    estimation,
    expressions,
    logit,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger("activitysim")


class TeleworkDurationSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `telework_duration` component.
    """

    CHOICE_MODEL: Literal["PROBABILISTIC", "MNL"] = "PROBABILISTIC"
    """Choice model type to use for telework duration."""

    CHOOSER_FILTER_COLUMN_NAME: str = "has_in_home_work_activity"
    """Column name in chooser table to represent workers with in-home work activity on the simulation day."""

    DURATION_CATEGORY_COLUMN_NAME: str = "telework_duration_category"
    """Persons column for the chosen telework duration category."""

    DURATION_HOURS_COLUMN_NAME: str = "telework_duration_hours"
    """Persons column for telework duration in hours."""

    ALTS: str = "telework_duration_alts.csv"
    """Alternatives file with duration category to hour mapping."""

    ALT_NAME_COLUMN: str = "alt"
    """Alternatives file column containing category names."""

    ALT_DURATION_COLUMN: str = "duration_hours"
    """Alternatives file column containing duration values in hours."""

    SPEC: str | None = None
    """MNL utility specification file. Required only when CHOICE_MODEL is MNL."""

    COEFFICIENTS: str | None = None
    """MNL coefficients file. Required only when CHOICE_MODEL is MNL."""

    LOGIT_TYPE: Literal["MNL", "NL"] = "MNL"
    """Logit type when running MNL mode."""

    NESTS: dict | None = None
    """Nest settings for NL mode, if ever used."""

    PROBS_SPEC: str | None = None
    """Probabilistic choice lookup table. Required only when CHOICE_MODEL is PROBABILISTIC."""

    PROBS_JOIN_COLS: list[str] | None = None
    """Columns to join choosers to probability table."""

    CONSTANTS: dict = {}
    """Named constants usable in preprocessors and expressions."""

    preprocessor: dict | list[dict] | None = None
    """Chooser preprocessor settings."""

    @model_validator(mode="after")
    def validate_choice_specific_files(self):
        if self.CHOICE_MODEL == "MNL":
            if not self.SPEC:
                raise ValueError("SPEC is required when CHOICE_MODEL is 'MNL'.")
            if not self.COEFFICIENTS:
                raise ValueError("COEFFICIENTS is required when CHOICE_MODEL is 'MNL'.")
        elif self.CHOICE_MODEL == "PROBABILISTIC":
            if not self.PROBS_SPEC:
                raise ValueError(
                    "PROBS_SPEC is required when CHOICE_MODEL is 'PROBABILISTIC'."
                )
        return self


def _load_alternatives(state: workflow.State, model_settings: TeleworkDurationSettings):
    alts = simulate.read_model_alts(state, model_settings.ALTS, set_index=None)

    alt_name_col = model_settings.ALT_NAME_COLUMN
    alt_duration_col = model_settings.ALT_DURATION_COLUMN
    if alt_name_col not in alts.columns or alt_duration_col not in alts.columns:
        raise RuntimeError(
            "telework_duration alternatives file must include "
            f"'{alt_name_col}' and '{alt_duration_col}' columns"
        )

    alts = alts[[alt_name_col, alt_duration_col]].copy()
    alts[alt_name_col] = alts[alt_name_col].astype(str)
    alts = alts.drop_duplicates(subset=[alt_name_col])
    return alts


def _simulate_probabilistic(
    state: workflow.State,
    choosers: pd.DataFrame,
    model_settings: TeleworkDurationSettings,
    trace_label: str,
) -> pd.Series:
    probs = pd.read_csv(
        state.filesystem.get_config_file_path(model_settings.PROBS_SPEC), comment="#"
    )
    probs_join_cols = model_settings.PROBS_JOIN_COLS or []

    if probs_join_cols:
        chooser_probs = pd.merge(
            choosers.reset_index(),
            probs,
            on=probs_join_cols,
            how="left",
        ).set_index(choosers.index.name)
    else:
        if probs.shape[0] != 1:
            raise RuntimeError(
                "telework_duration probabilistic mode requires a single-row PROBS_SPEC "
                "when PROBS_JOIN_COLS is not provided"
            )
        chooser_probs = pd.concat([probs] * len(choosers), ignore_index=True)
        chooser_probs.index = choosers.index

    prob_cols = [c for c in probs.columns if c not in probs_join_cols]
    if not prob_cols:
        raise RuntimeError(
            "telework_duration probabilistic mode found no probability columns"
        )

    chooser_probs = chooser_probs[prob_cols].fillna(0)
    row_sums = chooser_probs.sum(axis=1)
    if (row_sums <= 0).any():
        raise RuntimeError(
            "telework_duration probabilistic mode found choosers with no positive "
            "probability mass"
        )

    chooser_probs = chooser_probs.div(row_sums, axis=0)

    choices, _ = logit.make_choices(
        state,
        chooser_probs,
        trace_label=trace_label,
        trace_choosers=choosers,
    )

    category_choices = pd.Series(prob_cols).loc[choices].astype(str)
    category_choices.index = choices.index

    return category_choices


@workflow.step
def telework_duration(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: TeleworkDurationSettings | None = None,
    model_settings_file_name: str = "telework_duration.yaml",
    trace_label: str = "telework_duration",
) -> None:
    """
    Simulate daily in-home work duration for workers with in-home work activity.

    This model applies only to workers where `has_in_home_work_activity` is True
    from the telework arrangement model. It supports either:
    - probabilistic sampling from `PROBS_SPEC`, or
    - MNL simulation from `SPEC` and `COEFFICIENTS`.
    """

    if model_settings is None:
        model_settings = TeleworkDurationSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    chooser_filter_col = model_settings.CHOOSER_FILTER_COLUMN_NAME

    choosers = persons_merged[persons_merged[chooser_filter_col]]

    logger.info("Running %s with %d persons", trace_label, len(choosers))

    category_col = model_settings.DURATION_CATEGORY_COLUMN_NAME
    duration_col = model_settings.DURATION_HOURS_COLUMN_NAME
    alts = _load_alternatives(state, model_settings)
    category_dtype = pd.api.types.CategoricalDtype(
        categories=alts[model_settings.ALT_NAME_COLUMN].tolist() + [""],
        ordered=False,
    )

    # Default values for non-eligible persons.
    persons[category_col] = pd.Series(
        pd.Categorical([""] * len(persons), dtype=category_dtype),
        index=persons.index,
    )
    persons[duration_col] = 0.0

    if choosers.empty:
        state.add_table("persons", persons)
        tracing.print_summary(category_col, persons[category_col], value_counts=True)
        tracing.print_summary(duration_col, persons[duration_col], value_counts=True)
        return

    estimator = estimation.manager.begin_estimation(state, "telework_duration")
    constants = config.get_model_constants(model_settings)

    expressions.annotate_preprocessors(
        state,
        df=choosers,
        locals_dict=constants,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )

    choice_model = model_settings.CHOICE_MODEL

    if choice_model == "MNL":
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

        raw_choices = simulate.simple_simulate(
            state,
            choosers=choosers,
            spec=model_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            trace_label=trace_label,
            trace_choice_name=category_col,
            estimator=estimator,
            compute_settings=model_settings.compute_settings,
        )
        category_choices = pd.Series(
            model_spec.columns[raw_choices.values], index=raw_choices.index
        ).astype(category_dtype)
    else:
        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings, tag="PROBS_SPEC")
            estimator.write_choosers(choosers)
        category_choices = _simulate_probabilistic(
            state,
            choosers,
            model_settings,
            trace_label,
        ).astype(category_dtype)

    alt_to_duration = alts.set_index(model_settings.ALT_NAME_COLUMN)[
        model_settings.ALT_DURATION_COLUMN
    ]

    if estimator:
        estimator.write_choices(category_choices)
        category_choices = estimator.get_survey_values(
            category_choices,
            "persons",
            category_col,
        )
        category_choices = category_choices.astype(category_dtype)
        estimator.write_override_choices(category_choices)
        estimator.end_estimation()

    duration_choices = category_choices.map(alt_to_duration).fillna(0.0).astype(float)

    persons.loc[category_choices.index, category_col] = category_choices
    persons.loc[duration_choices.index, duration_col] = duration_choices

    state.add_table("persons", persons)

    tracing.print_summary(category_col, persons[category_col], value_counts=True)
    tracing.print_summary(duration_col, persons[duration_col], value_counts=True)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

    expressions.annotate_tables(
        state,
        locals_dict=constants,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )

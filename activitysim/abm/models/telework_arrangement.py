# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger("activitysim")


class TeleworkArrangementSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `telework_arrangement` component.
    """

    CHOOSER_FILTER_COLUMN_NAME: str = "is_worker"
    """Column name in the dataframe to represent worker."""

    HAS_IN_HOME_WORK_ACTIVITY_ALT: int = 0
    """The alternative index for having in-home work activity on the simulation day."""


@workflow.step
def telework_arrangement(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: TeleworkArrangementSettings | None = None,
    model_settings_file_name: str = "telework_arrangement.yaml",
    trace_label: str = "telework_arrangement",
) -> None:
    """
    This model predicts the telework arrangement on the simulation day for all workers.
    The alternatives are whether or not a worker has in-home telework activities on the simulation day:
    The result is a new column in the persons table, "has_in_home_work_activity": True or False

    Parameters
    ----------
    state : workflow.State
    persons_merged : DataFrame
        This represents the 'choosers' table for this component.
    persons : DataFrame
        The original persons table is referenced so the telework arrangement column
        can be appended to it.
    model_settings : TeleworkArrangementSettings, optional
        The settings used in this model component.  If not provided, they are
        loaded out of the configs directory YAML file referenced by
        the `model_settings_file_name` argument.
    model_settings_file_name : str, default "telework_arrangement.yaml"
        This is where model setting are found if `model_settings` is not given
        explicitly.  The same filename is also used to write settings files to
        the estimation data bundle in estimation mode.
    trace_label : str, default "telework_arrangement"
        This label is used for various tracing purposes.
    """

    if model_settings is None:
        model_settings = TeleworkArrangementSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    chooser_filter_column_name = model_settings.CHOOSER_FILTER_COLUMN_NAME
    choosers = persons_merged[persons_merged[chooser_filter_column_name]]

    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "telework_arrangement")

    constants = config.get_model_constants(model_settings)

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
        trace_choice_name="telework_arrangement",
        estimator=estimator,
        compute_settings=model_settings.compute_settings,
    )

    has_in_home_work_activity_alt = model_settings.HAS_IN_HOME_WORK_ACTIVITY_ALT
    choices = choices == has_in_home_work_activity_alt

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices,
            "persons",
            "has_in_home_work_activity",
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["has_in_home_work_activity"] = (
        choices.reindex(persons.index).fillna(0).astype(bool)
    )

    state.add_table("persons", persons)

    tracing.print_summary(
        "telework_arrangement.has_in_home_work_activity",
        persons.has_in_home_work_activity,
        value_counts=True,
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

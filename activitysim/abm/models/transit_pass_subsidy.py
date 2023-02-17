# ActivitySim
# See full license in LICENSE.txt.
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

logger = logging.getLogger("activitysim")


@workflow.step
def transit_pass_subsidy(
    whale: workflow.Whale,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    trace_hh_id,
):
    """
    Transit pass subsidy model.
    """

    trace_label = "transit_pass_subsidy"
    model_settings_file_name = "transit_pass_subsidy.yaml"

    choosers = persons_merged
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = whale.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(whale, "transit_pass_subsidy")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            whale,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    model_spec = whale.filesystem.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = whale.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        whale, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        whale,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="transit_pass_subsidy",
        estimator=estimator,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "transit_pass_subsidy"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["transit_pass_subsidy"] = choices.reindex(persons.index)

    whale.add_table("persons", persons)

    tracing.print_summary(
        "transit_pass_subsidy", persons.transit_pass_subsidy, value_counts=True
    )

    if trace_hh_id:
        whale.trace_df(persons, label=trace_label, warn_if_empty=True)

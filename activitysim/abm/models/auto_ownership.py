# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, inject, pipeline, simulate, tracing

from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def auto_ownership_simulate(households, households_merged, chunk_size, trace_hh_id):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """
    trace_label = "auto_ownership_simulate"
    model_settings_file_name = "auto_ownership.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    estimator = estimation.manager.begin_estimation("auto_ownership")

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choosers = households_merged.to_frame()

    logger.info("Running %s with %d households", trace_label, len(choosers))

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    log_alt_losers = config.setting("log_alt_losers", False)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="auto_ownership",
        log_alt_losers=log_alt_losers,
        estimator=estimator,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "households", "auto_ownership")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    households = households.to_frame()

    # no need to reindex as we used all households
    households["auto_ownership"] = choices

    pipeline.replace_table("households", households)

    tracing.print_summary(
        "auto_ownership", households.auto_ownership, value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(households, label="auto_ownership", warn_if_empty=True)

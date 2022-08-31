# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.abm.models.util import estimation
from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

logger = logging.getLogger("activitysim")


@inject.step()
def telecommute_frequency(persons_merged, persons, chunk_size, trace_hh_id):
    """
    This model predicts the frequency of telecommute for a person (worker) who
    does not works from home. The alternatives of this model are 'No Telecommute',
    '1 day per week', '2 to 3 days per week' and '4 days per week'. This model
    reflects the choices of people who prefer a combination of working from home and
    office during a week.
    """

    trace_label = "telecommute_frequency"
    model_settings_file_name = "telecommute_frequency.yaml"

    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.workplace_zone_id > -1]

    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("telecommute_frequency")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="telecommute_frequency",
        estimator=estimator,
    )

    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "telecommute_frequency"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons = persons.to_frame()
    persons["telecommute_frequency"] = (
        choices.reindex(persons.index).fillna("").astype(str)
    )

    pipeline.replace_table("persons", persons)

    tracing.print_summary(
        "telecommute_frequency", persons.telecommute_frequency, value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

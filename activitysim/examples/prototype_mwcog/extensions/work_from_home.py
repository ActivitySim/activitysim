# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core import expressions

from activitysim.abm.models.util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def work_from_home(persons_merged, persons, chunk_size, trace_hh_id):
    """
    This model predicts the whether a person (worker) works from home. The output
    from this model is TRUE (if works from home) or FALSE (works away from home).
    The workplace location choice is overridden for workers who work from home
    and set to -1.

    The main interface to the work from home model is the work_from_home() function.
    This function is registered as an orca step in the example Pipeline.
    """

    trace_label = "work_from_home"
    model_settings_file_name = "work_from_home.yaml"

    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("work_from_home")

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
        trace_choice_name="work_from_home",
        estimator=estimator,
    )

    work_from_home_alt = model_settings["WORK_FROM_HOME_ALT"]
    choices = choices == work_from_home_alt

    dest_choice_column_name = model_settings["DEST_CHOICE_COLUMN_NAME"]
    print(dest_choice_column_name)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "persons", "work_from_home")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons = persons.to_frame()
    persons["work_from_home"] = choices.reindex(persons.index).fillna(0).astype(bool)
    persons[dest_choice_column_name] = np.where(
        persons.work_from_home is True, -1, persons[dest_choice_column_name]
    )

    pipeline.replace_table("persons", persons)

    tracing.print_summary("work_from_home", persons.work_from_home, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

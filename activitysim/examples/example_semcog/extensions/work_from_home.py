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

logger = logging.getLogger("activitysim")


@inject.step()
def work_from_home(
        persons_merged, persons,
        chunk_size, trace_hh_id):
    """
    This model predicts whether a person (worker) works from home. The output
    from this model is TRUE (if works from home) or FALSE (works away from home).
    The workplace location choice is overridden for workers who work from home
    and set to -1.

    """

    trace_label = 'work_from_home'
    model_settings_file_name = 'work_from_home.yaml'

    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation('work_from_home')

    constants = config.get_model_constants(model_settings)
    work_from_home_alt = model_settings['WORK_FROM_HOME_ALT']

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df)
        estimator.write_choosers(choosers)

    # - iterative what-if if specified
    iterations = model_settings.get('WORK_FROM_HOME_ITERATIONS', 1)
    iterations_coefficient_constant = model_settings.get('WORK_FROM_HOME_COEFFICIENT_CONSTANT', None)
    iterations_target_percent = model_settings.get('WORK_FROM_HOME_TARGET_PERCENT', None)
    iterations_target_percent_tolerance = model_settings.get('WORK_FROM_HOME_TARGET_PERCENT_TOLERANCE', None)

    for iteration in range(iterations):

        logger.info("Running %s with %d persons iteration %d", trace_label, len(choosers), iteration)

        # re-read spec to reset substitution
        model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
        model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=model_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name='work_from_home',
            estimator=estimator)

        if iterations_target_percent is not None:
            current_percent = ((choices == work_from_home_alt).sum() / len(choices))
            logger.info("Running %s iteration %i current percent %f target percent %f",
                        trace_label, iteration, current_percent, iterations_target_percent)

            if current_percent <= (iterations_target_percent +
                                   iterations_target_percent_tolerance
                                   ) and current_percent >= (iterations_target_percent -
                                                             iterations_target_percent_tolerance):
                logger.info("Running %s iteration %i converged with coefficient %f", trace_label, iteration,
                            coefficients_df.value[iterations_coefficient_constant])
                break

            else:
                new_value = np.log(iterations_target_percent /
                                   np.maximum(current_percent, 0.0001)
                                   ) + coefficients_df.value[iterations_coefficient_constant]
                coefficients_df.value[iterations_coefficient_constant] = new_value
                logger.info("Running %s iteration %i new coefficient for next iteration %f",
                            trace_label, iteration, new_value)
                iteration = iteration + 1

    choices = (choices == work_from_home_alt)

    dest_choice_column_name = model_settings['DEST_CHOICE_COLUMN_NAME']

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'persons', 'work_from_home')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons = persons.to_frame()
    persons['work_from_home'] = choices.reindex(persons.index).fillna(0).astype(bool)
    persons[dest_choice_column_name] = np.where(persons.work_from_home is True, -1, persons[dest_choice_column_name])

    pipeline.replace_table("persons", persons)

    tracing.print_summary('work_from_home', persons.work_from_home, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(persons,
                         label=trace_label,
                         warn_if_empty=True)

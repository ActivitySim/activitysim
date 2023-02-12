# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.abm.models.util import estimation
from activitysim.abm.models.util.overlap import hh_time_window_overlap
from activitysim.core import config, expressions, simulate, tracing, workflow

logger = logging.getLogger(__name__)


def add_null_results(whale, trace_label, tours):
    logger.info("Skipping %s: add_null_results" % trace_label)
    tours["composition"] = ""
    whale.add_table("tours", tours)


@workflow.step
def joint_tour_composition(
    whale: workflow.Whale, tours, households, persons, chunk_size
):
    """
    This model predicts the makeup of the travel party (adults, children, or mixed).
    """
    trace_label = "joint_tour_composition"
    model_settings_file_name = "joint_tour_composition.yaml"

    joint_tours = tours[tours.tour_category == "joint"]

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        add_null_results(whale, trace_label, tours)
        return

    model_settings = whale.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(whale, "joint_tour_composition")

    # - only interested in households with joint_tours
    households = households.to_frame()
    households = households[households.num_hh_joint_tours > 0]

    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(households.index)]

    logger.info(
        "Running joint_tour_composition with %d joint tours" % joint_tours.shape[0]
    )

    # - run preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        locals_dict = {
            "persons": persons,
            "hh_time_window_overlap": hh_time_window_overlap,
        }

        expressions.assign_columns(
            whale,
            df=households,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    joint_tours_merged = pd.merge(
        joint_tours, households, left_on="household_id", right_index=True, how="left"
    )

    # - simple_simulate
    model_spec = whale.filesystem.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = whale.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        whale, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(joint_tours_merged)

    choices = simulate.simple_simulate(
        whale,
        choosers=joint_tours_merged,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="composition",
        estimator=estimator,
    )

    # convert indexes to alternative names
    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "tours", "composition")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # add composition column to tours for tracing
    joint_tours["composition"] = choices

    # reindex since we ran model on a subset of households
    tours["composition"] = choices.reindex(tours.index).fillna("").astype(str)
    whale.add_table("tours", tours)

    tracing.print_summary(
        "joint_tour_composition", joint_tours.composition, value_counts=True
    )

    if whale.settings.trace_hh_id:
        tracing.trace_df(
            joint_tours,
            label="joint_tour_composition.joint_tours",
            slicer="household_id",
        )

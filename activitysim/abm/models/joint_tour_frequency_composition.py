from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.overlap import hh_time_window_overlap
from activitysim.abm.models.util.tour_frequency import (
    JointTourFreqCompSettings,
    process_joint_tours_frequency_composition,
)
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.interaction_simulate import interaction_simulate

logger = logging.getLogger(__name__)


@workflow.step
def joint_tour_frequency_composition(
    state: workflow.State,
    households_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: JointTourFreqCompSettings | None = None,
    model_settings_file_name: str = "joint_tour_frequency_composition.yaml",
    trace_label: str = "joint_tour_frequency_composition",
) -> None:
    """
    This model predicts the frequency and composition of fully joint tours.
    """
    if model_settings is None:
        model_settings = JointTourFreqCompSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    # FIXME setting index as "alt" causes crash in estimation mode...
    alts = simulate.read_model_alts(
        state, "joint_tour_frequency_composition_alternatives.csv", set_index=None
    )
    alts.index = alts["alt"].values

    # - only interested in households with more than one cdap travel_active person and
    # - at least one non-preschooler
    choosers = households_merged[households_merged.participates_in_jtf_model].copy()

    # - only interested in persons in choosers households
    persons = persons[persons.household_id.isin(choosers.index)]

    logger.info("Running %s with %d households", trace_label, len(choosers))

    # alt preprocessor
    alt_preprocessor_settings = model_settings.ALTS_PREPROCESSOR
    if alt_preprocessor_settings:
        locals_dict = {}

        alts = alts.copy()

        expressions.assign_columns(
            state,
            df=alts,
            model_settings=alt_preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    # - preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_dict = {
            "persons": persons,
            "hh_time_window_overlap": lambda *x: hh_time_window_overlap(state, *x),
        }

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    estimator = estimation.manager.begin_estimation(
        state, "joint_tour_frequency_composition"
    )

    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

        assert choosers.index.name == "household_id"
        assert "household_id" not in choosers.columns
        choosers["household_id"] = choosers.index

        estimator.set_chooser_id(choosers.index.name)

        # FIXME set_alt_id - do we need this for interaction_simulate estimation bundle tables?
        estimator.set_alt_id("alt_id")

    # The choice value 'joint_tour_frequency_composition' assigned by interaction_simulate
    # is the index value of the chosen alternative in the alternatives table.
    choices = interaction_simulate(
        state,
        choosers=choosers,
        alternatives=alts,
        spec=model_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name=trace_label,
        estimator=estimator,
        explicit_chunk_size=0,
        compute_settings=model_settings.compute_settings,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "joint_tour_frequency_composition"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # add joint tour frequency composition column to household table
    households_merged["joint_tour_frequency_composition"] = choices.reindex(
        households_merged.index
    ).fillna(0)

    # - create joint_tours based on choices

    # - we need a person_id in order to generate the tour index (and for register_traceable_table)
    # - but we don't know the tour participants yet
    # - so we arbitrarily choose the first person in the household
    # - to be point person for the purpose of generating an index and setting origin
    # FIXME: not all models are guaranteed to have PNUM
    temp_point_persons = persons.loc[persons.PNUM == 1]
    temp_point_persons["person_id"] = temp_point_persons.index
    temp_point_persons = temp_point_persons.set_index("household_id")
    temp_point_persons = temp_point_persons[["person_id", "home_zone_id"]]

    # create a tours table of tour_category "joint" and different tour_types (e.g. shopping, eat)
    # and add the composition column (adults or children or mixed) to the tour

    # Choices
    # hhid	choice
    # 11111	1
    # 22222	2
    # 33333	3

    # Alts
    # alt	purpose1	purpose2	party1	party2	eat	shop
    # 1	    5	        0	        3	    0	    1	0
    # 2	    5	        6	        1	    3	    1	1
    # 3	    6	        0	        1	    0	    0	1

    # Joint Tours
    # hhid	type	category	composition
    # 11111	eat	    joint	    mixed
    # 22222	eat	    joint	    adults
    # 22222	shop	joint	    mixed
    # 33333	shop	joint	    adults

    joint_tours = process_joint_tours_frequency_composition(
        state, choices, alts, temp_point_persons
    )

    tours = state.extend_table("tours", joint_tours)

    state.tracing.register_traceable_table("tours", joint_tours)
    state.get_rn_generator().add_channel("tours", joint_tours)

    # we expect there to be an alt with no tours - which we can use to backfill non-travelers
    no_tours_alt = 0
    # keep memory usage down by downcasting
    households_merged["joint_tour_frequency_composition"] = pd.to_numeric(
        choices.reindex(households_merged.index).fillna(no_tours_alt),
        downcast="integer",
    )

    households_merged["num_hh_joint_tours"] = (
        joint_tours.groupby("household_id")
        .size()
        .reindex(households_merged.index)
        .fillna(0)
        .astype(np.int8)
    )

    state.add_table("households", households_merged)

    tracing.print_summary(
        "joint_tour_frequency_composition",
        households_merged.joint_tour_frequency_composition,
        value_counts=True,
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(
            households_merged, label="joint_tour_frequency_composition.households"
        )

        state.tracing.trace_df(
            joint_tours,
            label="joint_tour_frequency_composition.joint_tours",
            slicer="household_id",
        )

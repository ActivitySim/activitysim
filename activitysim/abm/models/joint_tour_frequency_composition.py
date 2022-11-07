# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
import os
from activitysim.core.interaction_simulate import interaction_simulate

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import expressions

from .util import estimation

from .util.overlap import hh_time_window_overlap
from .util.tour_frequency import process_joint_tours_frequency_composition

logger = logging.getLogger(__name__)


@inject.step()
def joint_tour_frequency_composition(
    households_merged, persons, chunk_size, trace_hh_id
):
    """
    This model predicts the frequency and composition of fully joint tours.
    """

    trace_label = "joint_tour_frequency_composition"
    model_settings_file_name = "joint_tour_frequency_composition.yaml"

    model_settings = config.read_model_settings(model_settings_file_name)

    alt_tdd = simulate.read_model_alts(
        "joint_tour_frequency_composition_alternatives.csv", set_index="alt"
    )

    # - only interested in households with more than one cdap travel_active person and
    # - at least one non-preschooler
    households_merged = households_merged.to_frame()
    choosers = households_merged[households_merged.participates_in_jtf_model].copy()

    # - only interested in persons in choosers households
    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(choosers.index)]

    logger.info("Running %s with %d households", trace_label, len(choosers))

    # alt preprocessor
    alt_preprocessor_settings = model_settings.get("ALTS_PREPROCESSOR", None)
    if alt_preprocessor_settings:

        locals_dict = {}

        alt_tdd = alt_tdd.copy()

        expressions.assign_columns(
            df=alt_tdd,
            model_settings=alt_preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_dict = {
            "persons": persons,
            "hh_time_window_overlap": hh_time_window_overlap,
        }

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    estimator = estimation.manager.begin_estimation("joint_tour_frequency_composition")

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)
        estimator.write_alternatives(alts)

        assert choosers.index.name == "household_id"
        assert "household_id" not in choosers.columns
        choosers["household_id"] = choosers.index

        estimator.set_chooser_id(choosers.index.name)

    # The choice value 'joint_tour_frequency_composition' assigned by interaction_simulate
    # is the index value of the chosen alternative in the alternatives table.
    choices = interaction_simulate(
        choosers=choosers,
        alternatives=alt_tdd,
        spec=model_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name=trace_label,
        estimator=estimator,
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
        choices, alt_tdd, temp_point_persons
    )

    tours = pipeline.extend_table("tours", joint_tours)

    tracing.register_traceable_table("tours", joint_tours)
    pipeline.get_rn_generator().add_channel("tours", joint_tours)

    # we expect there to be an alt with no tours - which we can use to backfill non-travelers
    no_tours_alt = 0
    households_merged["joint_tour_frequency_composition"] = (
        choices.reindex(households_merged.index).fillna(no_tours_alt).astype(str)
    )

    households_merged["num_hh_joint_tours"] = (
        joint_tours.groupby("household_id")
        .size()
        .reindex(households_merged.index)
        .fillna(0)
        .astype(np.int8)
    )

    pipeline.replace_table("households", households_merged)

    tracing.print_summary(
        "joint_tour_frequency_composition",
        households_merged.joint_tour_frequency_composition,
        value_counts=True,
    )

    if trace_hh_id:
        tracing.trace_df(
            households_merged, label="joint_tour_frequency_composition.households"
        )

        tracing.trace_df(
            joint_tours,
            label="joint_tour_frequency_composition.joint_tours",
            slicer="household_id",
        )

# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

from .util import estimation
from .util.overlap import hh_time_window_overlap
from .util.tour_frequency import process_joint_tours

logger = logging.getLogger(__name__)


@inject.step()
def joint_tour_frequency(households, persons, chunk_size, trace_hh_id):
    """
    This model predicts the frequency of making fully joint trips (see the
    alternatives above).
    """
    trace_label = "joint_tour_frequency"
    model_settings_file_name = "joint_tour_frequency.yaml"

    estimator = estimation.manager.begin_estimation("joint_tour_frequency")

    model_settings = config.read_model_settings(model_settings_file_name)

    alternatives = simulate.read_model_alts(
        "joint_tour_frequency_alternatives.csv", set_index="alt"
    )

    # - only interested in households with more than one cdap travel_active person and
    # - at least one non-preschooler
    households = households.to_frame()
    multi_person_households = households[households.participates_in_jtf_model].copy()

    # - only interested in persons in multi_person_households
    # FIXME - gratuitous pathological efficiency move, just let yaml specify persons?
    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(multi_person_households.index)]

    logger.info(
        "Running joint_tour_frequency with %d multi-person households"
        % multi_person_households.shape[0]
    )

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_dict = {
            "persons": persons,
            "hh_time_window_overlap": hh_time_window_overlap,
        }

        expressions.assign_columns(
            df=multi_person_households,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(multi_person_households)

    choices = simulate.simple_simulate(
        choosers=multi_person_households,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="joint_tour_frequency",
        estimator=estimator,
    )

    # convert indexes to alternative names
    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "joint_tour_frequency"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # - create joint_tours based on joint_tour_frequency choices

    # - we need a person_id in order to generate the tour index (and for register_traceable_table)
    # - but we don't know the tour participants yet
    # - so we arbitrarily choose the first person in the household
    # - to be point person for the purpose of generating an index and setting origin
    temp_point_persons = persons.loc[persons.PNUM == 1]
    temp_point_persons["person_id"] = temp_point_persons.index
    temp_point_persons = temp_point_persons.set_index("household_id")
    temp_point_persons = temp_point_persons[["person_id", "home_zone_id"]]

    joint_tours = process_joint_tours(choices, alternatives, temp_point_persons)

    tours = pipeline.extend_table("tours", joint_tours)

    tracing.register_traceable_table("tours", joint_tours)
    pipeline.get_rn_generator().add_channel("tours", joint_tours)

    # - annotate households

    # we expect there to be an alt with no tours - which we can use to backfill non-travelers
    no_tours_alt = (alternatives.sum(axis=1) == 0).index[0]
    households["joint_tour_frequency"] = (
        choices.reindex(households.index).fillna(no_tours_alt).astype(str)
    )

    households["num_hh_joint_tours"] = (
        joint_tours.groupby("household_id")
        .size()
        .reindex(households.index)
        .fillna(0)
        .astype(np.int8)
    )

    pipeline.replace_table("households", households)

    tracing.print_summary(
        "joint_tour_frequency", households.joint_tour_frequency, value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(households, label="joint_tour_frequency.households")

        tracing.trace_df(
            joint_tours, label="joint_tour_frequency.joint_tours", slicer="household_id"
        )

    if estimator:
        survey_tours = estimation.manager.get_survey_table("tours")
        survey_tours = survey_tours[survey_tours.tour_category == "joint"]

        print(f"len(survey_tours) {len(survey_tours)}")
        print(f"len(joint_tours) {len(joint_tours)}")

        different = False
        survey_tours_not_in_tours = survey_tours[
            ~survey_tours.index.isin(joint_tours.index)
        ]
        if len(survey_tours_not_in_tours) > 0:
            print(f"survey_tours_not_in_tours\n{survey_tours_not_in_tours}")
            different = True
        tours_not_in_survey_tours = joint_tours[
            ~joint_tours.index.isin(survey_tours.index)
        ]
        if len(survey_tours_not_in_tours) > 0:
            print(f"tours_not_in_survey_tours\n{tours_not_in_survey_tours}")
            different = True
        assert not different

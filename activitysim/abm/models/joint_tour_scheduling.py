# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util.vectorize_tour_scheduling import (
    TourSchedulingSettings,
    vectorize_joint_tour_scheduling,
)
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.core.util import assign_in_place, reindex

logger = logging.getLogger(__name__)


# class JointTourSchedulingSettings(LogitComponentSettings, extra="forbid"):
#     """
#     Settings for the `joint_tour_scheduling` component.
#     """
#
#     preprocessor: PreprocessorSettings | None = None
#     """Setting for the preprocessor."""
#
#     sharrow_skip: bool = False
#     """Setting to skip sharrow"""
#


@workflow.step
def joint_tour_scheduling(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    tdd_alts: pd.DataFrame,
    model_settings: TourSchedulingSettings | None = None,
    model_settings_file_name: str = "joint_tour_scheduling.yaml",
    trace_label: str = "joint_tour_scheduling",
) -> None:
    """
    This model predicts the departure time and duration of each joint tour
    """

    if model_settings is None:
        model_settings = TourSchedulingSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    trace_hh_id = state.settings.trace_hh_id
    joint_tours = tours[tours.tour_category == "joint"]

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    # use state.get_dataframe as this won't exist if there are no joint_tours
    joint_tour_participants = state.get_dataframe("joint_tour_participants")

    logger.info("Running %s with %d joint tours", trace_label, joint_tours.shape[0])

    # it may seem peculiar that we are concerned with persons rather than households
    # but every joint tour is (somewhat arbitrarily) assigned a "primary person"
    # some of whose characteristics are used in the spec
    # and we get household attributes along with person attributes in persons_merged
    persons_merged = persons_merged[persons_merged.num_hh_joint_tours > 0]

    # since a households joint tours each potentially different participants
    # they may also have different joint tour masks (free time of all participants)
    # so we have to either chunk processing by joint_tour_num and build timetable by household
    # or build timetables by unique joint_tour

    constants = config.get_model_constants(model_settings)

    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            state,
            df=joint_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    timetable = state.get_injectable("timetable")

    estimator = estimation.manager.begin_estimation(state, "joint_tour_scheduling")

    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        timetable.begin_transaction(estimator)

    choices = vectorize_joint_tour_scheduling(
        state,
        joint_tours,
        joint_tour_participants,
        persons_merged,
        tdd_alts,
        timetable,
        spec=model_spec,
        model_settings=model_settings,
        estimator=estimator,
        chunk_size=state.settings.chunk_size,
        trace_label=trace_label,
        compute_settings=model_settings.compute_settings,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "tours", "tdd")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

        # update timetable to reflect the override choices (assign tours in tour_num order)
        timetable.rollback()
        for tour_num, nth_tours in joint_tours.groupby("tour_num", sort=True):
            nth_participants = joint_tour_participants[
                joint_tour_participants.tour_id.isin(nth_tours.index)
            ]

            estimator.log(
                "assign timetable for %s participants in %s tours with tour_num %s"
                % (len(nth_participants), len(nth_tours), tour_num)
            )
            # - update timetables of all joint tour participants
            timetable.assign(
                nth_participants.person_id, reindex(choices, nth_participants.tour_id)
            )

    timetable.replace_table(state)

    # choices are tdd alternative ids
    # we want to add start, end, and duration columns to tours, which we have in tdd_alts table
    choices = pd.merge(
        choices.to_frame("tdd"), tdd_alts, left_on=["tdd"], right_index=True, how="left"
    )

    assign_in_place(
        tours, choices, state.settings.downcast_int, state.settings.downcast_float
    )
    state.add_table("tours", tours)

    # updated df for tracing
    joint_tours = tours[tours.tour_category == "joint"]

    if trace_hh_id:
        state.tracing.trace_df(
            joint_tours, label="joint_tour_scheduling", slicer="household_id"
        )

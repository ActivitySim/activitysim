# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing
from activitysim.core.util import assign_in_place, reindex

from .util import estimation
from .util.vectorize_tour_scheduling import vectorize_joint_tour_scheduling

logger = logging.getLogger(__name__)


@inject.step()
def joint_tour_scheduling(tours, persons_merged, tdd_alts, chunk_size, trace_hh_id):
    """
    This model predicts the departure time and duration of each joint tour
    """
    trace_label = "joint_tour_scheduling"

    model_settings_file_name = "joint_tour_scheduling.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    tours = tours.to_frame()
    joint_tours = tours[tours.tour_category == "joint"]

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    # use inject.get_table as this won't exist if there are no joint_tours
    joint_tour_participants = inject.get_table("joint_tour_participants").to_frame()

    persons_merged = persons_merged.to_frame()

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
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=joint_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    timetable = inject.get_injectable("timetable")

    estimator = estimation.manager.begin_estimation("joint_tour_scheduling")

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        timetable.begin_transaction(estimator)

    choices = vectorize_joint_tour_scheduling(
        joint_tours,
        joint_tour_participants,
        persons_merged,
        tdd_alts,
        timetable,
        spec=model_spec,
        model_settings=model_settings,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_label=trace_label,
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

    timetable.replace_table()

    # choices are tdd alternative ids
    # we want to add start, end, and duration columns to tours, which we have in tdd_alts table
    choices = pd.merge(
        choices.to_frame("tdd"), tdd_alts, left_on=["tdd"], right_index=True, how="left"
    )

    assign_in_place(tours, choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    joint_tours = tours[tours.tour_category == "joint"]

    if trace_hh_id:
        tracing.trace_df(
            joint_tours, label="joint_tour_scheduling", slicer="household_id"
        )

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util import vectorize_tour_scheduling as vts
from activitysim.core import config, estimation, expressions, simulate, workflow

from .vectorize_tour_scheduling import TourModeComponentSettings, TourSchedulingSettings

logger = logging.getLogger(__name__)


def run_tour_scheduling(
    state: workflow.State,
    model_name: str,
    chooser_tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    tdd_alts: pd.DataFrame,
    tour_segment_col: str,
):
    trace_label = model_name
    model_settings_file_name = f"{model_name}.yaml"

    model_settings = TourSchedulingSettings.read_settings_file(
        state.filesystem,
        model_settings_file_name,
        mandatory=False,
    )

    if model_settings.LOGSUM_SETTINGS:
        logsum_settings = TourModeComponentSettings.read_settings_file(
            state.filesystem,
            str(model_settings.LOGSUM_SETTINGS),
            mandatory=False,
        )
        logsum_columns = logsum_settings.LOGSUM_CHOOSER_COLUMNS
    else:
        logsum_columns = []

    # - filter chooser columns for both logsums and simulate
    model_columns = model_settings.SIMULATE_CHOOSER_COLUMNS
    chooser_columns = logsum_columns + [
        c for c in model_columns if c not in logsum_columns
    ]

    persons_merged = expressions.filter_chooser_columns(persons_merged, chooser_columns)

    timetable = state.get_injectable("timetable")

    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_d = {"tt": timetable.attach_state(state)}
        locals_d.update(config.get_model_constants(model_settings))

        expressions.assign_columns(
            state,
            df=chooser_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    estimators = {}
    if model_settings.TOUR_SPEC_SEGMENTS:
        # load segmented specs
        spec_segment_settings = model_settings.SPEC_SEGMENTS
        specs = {}
        compute_settings = {}
        for spec_segment_name, spec_settings in spec_segment_settings.items():
            bundle_name = f"{model_name}_{spec_segment_name}"

            # estimator for this tour_segment
            estimator = estimation.manager.begin_estimation(
                state, model_name=bundle_name, bundle_name=bundle_name
            )

            spec_file_name = spec_settings.SPEC
            model_spec = state.filesystem.read_model_spec(file_name=spec_file_name)
            coefficients_df = state.filesystem.read_model_coefficients(spec_settings)
            specs[spec_segment_name] = simulate.eval_coefficients(
                state, model_spec, coefficients_df, estimator
            )
            compute_settings[
                spec_segment_name
            ] = spec_settings.compute_settings.subcomponent_settings(spec_segment_name)

            if estimator:
                estimators[spec_segment_name] = estimator  # add to local list
                estimator.write_model_settings(model_settings, model_settings_file_name)
                estimator.write_spec(spec_settings)
                estimator.write_coefficients(coefficients_df, spec_settings)

        # - spec dict segmented by primary_purpose
        tour_segment_settings = model_settings.TOUR_SPEC_SEGMENTS
        tour_segments = {}
        for tour_segment_name, spec_segment_name in tour_segment_settings.items():
            tour_segments[tour_segment_name] = {}
            tour_segments[tour_segment_name]["spec_segment_name"] = spec_segment_name
            tour_segments[tour_segment_name]["spec"] = specs[spec_segment_name]
            tour_segments[tour_segment_name]["compute_settings"] = compute_settings[
                spec_segment_name
            ]
            tour_segments[tour_segment_name]["estimator"] = estimators.get(
                spec_segment_name
            )

        # default tour_segment_col to 'tour_type' if segmented spec and tour_segment_col not specified
        if tour_segment_col is None and tour_segments:
            tour_segment_col = "tour_type"

    else:
        # unsegmented spec
        assert (
            not model_settings.SPEC_SEGMENTS
        ), f"model_settings.SPEC_SEGMENTS should be omitted not {model_settings.SPEC_SEGMENTS!r}"
        assert not model_settings.TOUR_SPEC_SEGMENTS
        assert tour_segment_col is None

        estimator = estimation.manager.begin_estimation(state, model_name)

        spec_file_name = model_settings.SPEC
        model_spec = state.filesystem.read_model_spec(file_name=spec_file_name)
        coefficients_df = state.filesystem.read_model_coefficients(model_settings)
        model_spec = simulate.eval_coefficients(
            state, model_spec, coefficients_df, estimator
        )

        if estimator:
            estimators[None] = estimator  # add to local list
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)

        # - non_mandatory tour scheduling is not segmented by tour type
        tour_segments = {
            "spec": model_spec,
            "estimator": estimator,
            "compute_settings": model_settings.compute_settings,
        }

    if estimators:
        timetable.begin_transaction(list(estimators.values()))

    logger.info(f"Running {model_name} with %d tours", len(chooser_tours))
    choices = vts.vectorize_tour_scheduling(
        state,
        chooser_tours,
        persons_merged,
        tdd_alts,
        timetable,
        tour_segments=tour_segments,
        tour_segment_col=tour_segment_col,
        model_settings=model_settings,
        chunk_size=state.settings.chunk_size,
        trace_label=trace_label,
    )

    if estimators:
        # overrride choices for all estimators
        choices_list = []
        for spec_segment_name, estimator in estimators.items():
            if spec_segment_name:
                model_choices = choices[(chooser_tours.tour_type == spec_segment_name)]
            else:
                model_choices = choices

            estimator.write_choices(model_choices)
            override_choices = estimator.get_survey_values(
                model_choices, "tours", "tdd"
            )
            estimator.write_override_choices(override_choices)

            choices_list.append(override_choices)
            estimator.end_estimation()
        choices = pd.concat(choices_list)

        # update timetable to reflect the override choices (assign tours in tour_num order)
        timetable.rollback()
        for tour_num, nth_tours in chooser_tours.groupby("tour_num", sort=True):
            timetable.assign(
                window_row_ids=nth_tours["person_id"],
                tdds=choices.reindex(nth_tours.index),
            )

    timetable.replace_table(state)

    # choices are tdd alternative ids
    # we want to add start, end, and duration columns to tours, which we have in tdd_alts table
    choices = pd.merge(
        choices.to_frame("tdd"), tdd_alts, left_on=["tdd"], right_index=True, how="left"
    )

    return choices

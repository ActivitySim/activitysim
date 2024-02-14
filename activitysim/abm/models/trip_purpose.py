# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from activitysim.abm.models.util.school_escort_tours_trips import (
    split_out_school_escorting_trips,
)
from activitysim.core import (
    chunk,
    config,
    estimation,
    expressions,
    logit,
    simulate,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


PROBS_JOIN_COLUMNS = ["primary_purpose", "outbound", "person_type"]


def map_coefficients(spec, coefficients):
    if isinstance(coefficients, pd.DataFrame):
        assert "value" in coefficients.columns
        coefficients = coefficients["value"].to_dict()

    assert isinstance(
        coefficients, dict
    ), "map_coefficients doesn't grok type of coefficients: %s" % (type(coefficients))

    for c in spec.columns:
        if c == simulate.SPEC_LABEL_NAME:
            continue
        spec[c] = spec[c].map(coefficients).astype(np.float32)

    assert not spec.isnull().any()

    return spec


def choose_intermediate_trip_purpose(
    state: workflow.State,
    trips,
    probs_spec,
    estimator,
    probs_join_cols,
    use_depart_time,
    trace_hh_id,
    trace_label,
    *,
    chunk_sizer: chunk.ChunkSizer,
):
    """
    chose purpose for intermediate trips based on probs_spec
    which assigns relative weights (summing to 1) to the possible purpose choices

    Returns
    -------
    purpose: pandas.Series of purpose (str) indexed by trip_id
    """

    non_purpose_cols = probs_join_cols.copy()
    if use_depart_time:
        non_purpose_cols += ["depart_range_start", "depart_range_end"]
    purpose_cols = [c for c in probs_spec.columns if c not in non_purpose_cols]

    num_trips = len(trips.index)
    have_trace_targets = trace_hh_id and state.tracing.has_trace_targets(trips)

    # probs should sum to 1 across rows
    sum_probs = probs_spec[purpose_cols].sum(axis=1)
    probs_spec[purpose_cols] = probs_spec[purpose_cols].div(sum_probs, axis=0)

    # left join trips to probs (there may be multiple rows per trip for multiple depart ranges)
    choosers = pd.merge(
        trips.reset_index(), probs_spec, on=probs_join_cols, how="left"
    ).set_index("trip_id")
    chunk_sizer.log_df(trace_label, "choosers", choosers)

    if use_depart_time:
        # select the matching depart range (this should result on in exactly one chooser row per trip)
        chooser_probs = (choosers.start >= choosers["depart_range_start"]) & (
            choosers.start <= choosers["depart_range_end"]
        )

        # if we failed to match a row in probs_spec
        if chooser_probs.sum() < num_trips:
            # this can happen if the spec doesn't have probs for the trips matching a trip's probs_join_cols
            missing_trip_ids = trips.index[
                ~trips.index.isin(choosers.index[chooser_probs])
            ].values
            unmatched_choosers = choosers[choosers.index.isin(missing_trip_ids)]
            unmatched_choosers = unmatched_choosers[
                ["person_id", "start"] + non_purpose_cols
            ]

            # join to persons for better diagnostics
            persons = state.get_dataframe("persons")
            persons_cols = [
                "age",
                "is_worker",
                "is_student",
                "is_gradeschool",
                "is_highschool",
                "is_university",
            ]
            unmatched_choosers = pd.merge(
                unmatched_choosers,
                persons[[col for col in persons_cols if col in persons.columns]],
                left_on="person_id",
                right_index=True,
                how="left",
            )

            file_name = "%s.UNMATCHED_PROBS" % trace_label
            logger.error(
                "{} {} of {} intermediate trips could not be matched to probs based on join columns  {}".format(
                    trace_label, len(unmatched_choosers), len(choosers), probs_join_cols
                )
            )
            logger.info(
                f"Writing {len(unmatched_choosers)} unmatched choosers to {file_name}"
            )
            state.tracing.write_csv(
                unmatched_choosers, file_name=file_name, transpose=False
            )
            raise RuntimeError(
                "Some trips could not be matched to probs based on join columns %s."
                % probs_join_cols
            )

        # select the matching depart range (this should result on in exactly one chooser row per trip)
        choosers = choosers[chooser_probs]

    # choosers should now match trips row for row
    assert choosers.index.identical(trips.index)

    if estimator:
        probs_cols = list(probs_spec.columns)
        print(choosers[probs_cols])
        estimator.write_table(choosers[probs_cols], "probs", append=True)

    choices, rands = logit.make_choices(
        state, choosers[purpose_cols], trace_label=trace_label, trace_choosers=choosers
    )

    if have_trace_targets:
        state.tracing.trace_df(
            choices, "%s.choices" % trace_label, columns=[None, "trip_purpose"]
        )
        state.tracing.trace_df(rands, "%s.rands" % trace_label, columns=[None, "rand"])

    choices = choices.map(pd.Series(purpose_cols))
    # expand the purpose categorical
    for p in purpose_cols:
        if not p in trips.primary_purpose.cat.categories:
            trips.primary_purpose = trips.primary_purpose.cat.add_categories([p])
    choices = choices.astype(trips["primary_purpose"].dtype)
    return choices


class TripPurposeSettings(PydanticReadable):
    probs_join_cols: list[str] = ["primary_purpose", "outbound", "person_type"]
    PROBS_SPEC: str = "trip_purpose_probs.csv"
    preprocessor: PreprocessorSettings | None = None
    use_depart_time: bool = True
    CONSTANTS: dict[str, Any] = {}


def run_trip_purpose(
    state: workflow.State,
    trips_df: pd.DataFrame,
    estimator,
    model_settings: TripPurposeSettings | None = None,
    model_settings_file_name: str = "trip_purpose.yaml",
    trace_label: str = "trip_purpose",
):
    """
    trip purpose - main functionality separated from model step so it can be called iteratively

    For each intermediate stop on a tour (i.e. trip other than the last trip outbound or inbound)
    each trip is assigned a purpose based on an observed frequency distribution

    The distribution should always be segmented by tour purpose and tour direction. By default it is also
    segmented by person type. The join columns can be overwritten using the "probs_join_cols" parameter in
    the model settings. The model will attempt to segment by trip depart time as well if necessary
    and depart time ranges are specified in the probability lookup table.

    Returns
    -------
    purpose: pandas.Series of purpose (str) indexed by trip_id
    """

    # uniform across trip_purpose
    chunk_tag = "trip_purpose"

    if model_settings is None:
        model_settings = TripPurposeSettings.read_settings_file(
            state.filesystem, model_settings_file_name
        )

    probs_join_cols = model_settings.probs_join_cols

    spec_file_name = model_settings.PROBS_SPEC
    probs_spec = pd.read_csv(
        state.filesystem.get_config_file_path(spec_file_name), comment="#"
    )
    # FIXME for now, not really doing estimation for probabilistic model - just overwriting choices
    # besides, it isn't clear that named coefficients would be helpful if we had some form of estimation
    # coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    # probs_spec = map_coefficients(probs_spec, coefficients_df)

    if estimator:
        estimator.write_spec(model_settings, tag="PROBS_SPEC")
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # estimator.write_coefficients(coefficients_df, model_settings)

    result_list = []

    # add home to purpose categorical
    # check if parking_name is in the purpose category
    if not "home" in trips_df.primary_purpose.cat.categories:
        trips_df.primary_purpose = trips_df.primary_purpose.cat.add_categories(["home"])

    # - last trip of outbound tour gets primary_purpose
    last_trip = trips_df.trip_num == trips_df.trip_count
    purpose = trips_df.primary_purpose[last_trip & trips_df.outbound]
    print(purpose.value_counts(dropna=False))
    result_list.append(purpose)
    logger.info("assign purpose to %s last outbound trips", purpose.shape[0])

    # - last trip of inbound tour gets home (or work for atwork subtours)
    purpose = trips_df.primary_purpose[last_trip & ~trips_df.outbound]
    print(purpose.value_counts(dropna=False))
    purpose = pd.Series(
        np.where(purpose == "atwork", "work", "home"), index=purpose.index
    ).astype(trips_df.primary_purpose.dtype)
    print(purpose.value_counts(dropna=False))
    result_list.append(purpose)
    logger.info("assign purpose to %s last inbound trips", purpose.shape[0])

    # - intermediate stops (non-last trips) purpose assigned by probability table
    trips_df = trips_df[~last_trip]
    logger.info("assign purpose to %s intermediate trips", trips_df.shape[0])

    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_dict = config.get_model_constants(model_settings)
        expressions.assign_columns(
            state,
            df=trips_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    use_depart_time = model_settings.use_depart_time

    for (
        _i,
        trips_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers(state, trips_df, chunk_tag, trace_label):
        choices = choose_intermediate_trip_purpose(
            state,
            trips_chunk,
            probs_spec,
            estimator,
            probs_join_cols=probs_join_cols,
            use_depart_time=use_depart_time,
            trace_hh_id=state.settings.trace_hh_id,
            trace_label=chunk_trace_label,
            chunk_sizer=chunk_sizer,
        )
        print(choices.value_counts(dropna=False))
        result_list.append(choices)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


@workflow.step
def trip_purpose(state: workflow.State, trips: pd.DataFrame) -> None:
    """
    trip purpose model step - calls run_trip_purpose to run the actual model

    adds purpose column to trips
    """
    trace_label = "trip_purpose"

    trips_df = trips

    if state.is_table("school_escort_trips"):
        school_escort_trips = state.get_dataframe("school_escort_trips")
        # separate out school escorting trips to exclude them from the model and estimation data bundle
        trips_df, se_trips_df, full_trips_index = split_out_school_escorting_trips(
            trips_df, school_escort_trips
        )

    estimator = estimation.manager.begin_estimation(state, "trip_purpose")
    if estimator:
        chooser_cols_for_estimation = [
            "person_id",
            "household_id",
            "tour_id",
            "trip_num",
        ]
        estimator.write_choosers(trips_df[chooser_cols_for_estimation])

    choices = run_trip_purpose(
        state,
        trips_df,
        estimator,
        trace_label=trace_label,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "trips", "purpose"
        )  # override choices
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    trips_df["purpose"] = choices

    if state.is_table("school_escort_trips"):
        # setting purpose for school escort trips
        se_trips_df["purpose"] = reindex(school_escort_trips.purpose, se_trips_df.index)
        # merge trips back together preserving index order
        trips_df = pd.concat([trips_df, se_trips_df])
        trips_df = trips_df.reindex(full_trips_index)

    # we should have assigned a purpose to all trips
    assert not trips_df.purpose.isnull().any()

    state.add_table("trips", trips_df)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(
            trips_df,
            label=trace_label,
            slicer="trip_id",
            index_label="trip_id",
            warn_if_empty=True,
        )

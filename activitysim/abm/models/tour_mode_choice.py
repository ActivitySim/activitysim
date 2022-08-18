# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
from orca import orca

from activitysim.core import (
    config,
    expressions,
    inject,
    logit,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.pathbuilder import TransitVirtualPathBuilder
from activitysim.core.util import assign_in_place, reindex

from .util import estimation, trip
from .util.mode import run_tour_mode_choice_simulate

logger = logging.getLogger(__name__)

"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


def get_alts_from_segmented_nested_logit(model_settings, segment_name, trace_label):
    """Infer alts from logit spec

    Parameters
    ----------
    model_settings : dict
    segment_column_name : str
    trace_label : str

    Returns
    -------
    list
    """

    nest_spec = config.get_logit_model_settings(model_settings)
    coefficients = simulate.get_segment_coefficients(model_settings, segment_name)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)
    tour_mode_alts = []
    for nest in logit.each_nest(nest_spec):
        if nest.is_leaf:
            tour_mode_alts.append(nest.name)

    return tour_mode_alts


def create_logsum_trips(tours, segment_column_name, model_settings, trace_label):
    """
    Construct table of trips from half-tours (1 inbound, 1 outbound) for each tour-mode.

    Parameters
    ----------
    tours : pandas.DataFrame
    segment_column_name : str
        column in tours table used for segmenting model spec
    model_settings : dict
    trace_label : str

    Returns
    -------
    pandas.DataFrame
        Table of trips: 2 per tour, with O/D and purpose inherited from tour
    """
    stop_frequency_alts = inject.get_injectable("stop_frequency_alts")
    stop_freq = "0out_0in"  # no intermediate stops
    tours["stop_frequency"] = stop_freq
    tours["primary_purpose"] = tours["tour_purpose"]
    trips = trip.initialize_from_tours(tours, stop_frequency_alts)
    trips["stop_frequency"] = stop_freq
    outbound = trips["outbound"]
    trips["depart"] = reindex(tours.start, trips.tour_id)
    trips.loc[~outbound, "depart"] = reindex(tours.end, trips.loc[~outbound, "tour_id"])

    # actual segment doesn't matter. just need to grab one
    # to get a set of coefficients from the spec
    segment_name = tours.iloc[0][segment_column_name]
    tour_mode_alts = get_alts_from_segmented_nested_logit(
        model_settings, segment_name, trace_label
    )

    # repeat rows from the trips table iterating over tour mode
    logsum_trips = pd.DataFrame()
    for tour_mode in tour_mode_alts:
        trips["tour_mode"] = tour_mode
        logsum_trips = pd.concat((logsum_trips, trips), ignore_index=True)
    assert len(logsum_trips) == len(trips) * len(tour_mode_alts)
    logsum_trips.index.name = "trip_id"

    return logsum_trips


def append_tour_leg_trip_mode_choice_logsums(tours):
    """Creates trip mode choice logsum column in tours table for each tour mode and leg

    Parameters
    ----------
    tours : pd.DataFrame

    Returns
    -------
    tours : pd.DataFrame
        Adds two * n_modes logsum columns to each tour row, e.g. "logsum_DRIVE_outbound"
    """
    trips = inject.get_table("trips").to_frame()
    trip_dir_mode_logsums = trips.pivot(
        index="tour_id",
        columns=["tour_mode", "outbound"],
        values="trip_mode_choice_logsum",
    )
    new_cols = [
        "_".join(["logsum", mode, "outbound" if outbound else "inbound"])
        for mode, outbound in trip_dir_mode_logsums.columns
    ]
    trip_dir_mode_logsums.columns = new_cols
    trip_dir_mode_logsums.reindex(tours.index)
    tours = pd.merge(tours, trip_dir_mode_logsums, left_index=True, right_index=True)

    return tours


def get_trip_mc_logsums_for_all_modes(
    tours, segment_column_name, model_settings, trace_label
):
    """Creates pseudo-trips from tours and runs trip mode choice to get logsums

    Parameters
    ----------
    tours : pandas.DataFrame
    segment_column_name : str
        column in tours table used for segmenting model spec
    model_settings : dict
    trace_label : str

    Returns
    -------
    tours : pd.DataFrame
        Adds two * n_modes logsum columns to each tour row, e.g. "logsum_DRIVE_outbound"
    """

    # create pseudo-trips from tours for all tour modes
    logsum_trips = create_logsum_trips(
        tours, segment_column_name, model_settings, trace_label
    )

    # temporarily register trips in the pipeline
    pipeline.replace_table("trips", logsum_trips)
    tracing.register_traceable_table("trips", logsum_trips)
    pipeline.get_rn_generator().add_channel("trips", logsum_trips)

    # run trip mode choice on pseudo-trips. use orca instead of pipeline to
    # execute the step because pipeline can only handle one open step at a time
    orca.run(["trip_mode_choice"])

    # add trip mode choice logsums as new cols in tours
    tours = append_tour_leg_trip_mode_choice_logsums(tours)

    # de-register logsum trips table
    pipeline.get_rn_generator().drop_channel("trips")
    tracing.deregister_traceable_table("trips")

    return tours


@inject.step()
def tour_mode_choice_simulate(
    tours, persons_merged, network_los, chunk_size, trace_hh_id
):
    """
    Tour mode choice simulate
    """
    trace_label = "tour_mode_choice"
    model_settings_file_name = "tour_mode_choice.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME")
    mode_column_name = "tour_mode"
    segment_column_name = "tour_purpose"

    primary_tours = tours.to_frame()
    assert not (primary_tours.tour_category == "atwork").any()

    logger.info("Running %s with %d tours" % (trace_label, primary_tours.shape[0]))

    tracing.print_summary("tour_types", primary_tours.tour_type, value_counts=True)

    persons_merged = persons_merged.to_frame()
    primary_tours_merged = pd.merge(
        primary_tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
        suffixes=("", "_r"),
    )

    constants = {}
    # model_constants can appear in expressions
    constants.update(config.get_model_constants(model_settings))

    skim_dict = network_los.get_default_skim_dict()

    # setup skim keys
    orig_col_name = "home_zone_id"
    dest_col_name = "destination"

    out_time_col_name = "start"
    in_time_col_name = "end"
    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="out_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="in_period"
    )
    odr_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="in_period"
    )
    dor_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="out_period"
    )
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,  # dot return skims for e.g. TNC bridge return fare
        "dor_skims": dor_skim_stack_wrapper,  # odt return skims for e.g. TNC bridge return fare
        "od_skims": od_skim_stack_wrapper,
        "orig_col_name": orig_col_name,
        "dest_col_name": dest_col_name,
        "out_time_col_name": out_time_col_name,
        "in_time_col_name": in_time_col_name,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?

        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col_name,
            dest_key=dest_col_name,
            tod_key="out_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col_name,
            dest_key=orig_col_name,
            tod_key="in_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

        # TVPB constants can appear in expressions
        if model_settings.get("use_TVPB_constants", True):
            constants.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

    # don't create estimation data bundle if trip mode choice is being called
    # from another model step (i.e. tour mode choice logsum creation)
    if pipeline.get_rn_generator().step_name != "tour_mode_choice_simulate":
        estimator = None
    else:
        estimator = estimation.manager.begin_estimation("tour_mode_choice")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # (run_tour_mode_choice_simulate writes choosers post-annotation)

    # FIXME should normalize handling of tour_type and tour_purpose
    # mtctm1 school tour_type includes univ, which has different coefficients from elementary and HS
    # we should either add this column when tours created or add univ to tour_types
    not_university = (primary_tours_merged.tour_type != "school") | ~(
        primary_tours_merged.is_university.astype(bool)
        if "is_university" in primary_tours_merged.columns
        else False
    )
    primary_tours_merged["tour_purpose"] = primary_tours_merged.tour_type.where(
        not_university, "univ"
    )

    # if trip logsums are used, run trip mode choice and append the logsums
    if model_settings.get("COMPUTE_TRIP_MODE_CHOICE_LOGSUMS", False):
        primary_tours_merged = get_trip_mc_logsums_for_all_modes(
            primary_tours_merged, segment_column_name, model_settings, trace_label
        )

    choices_list = []
    for tour_purpose, tours_segment in primary_tours_merged.groupby(
        segment_column_name
    ):

        logger.info(
            "tour_mode_choice_simulate tour_type '%s' (%s tours)"
            % (
                tour_purpose,
                len(tours_segment.index),
            )
        )

        if network_los.zone_system == los.THREE_ZONE:
            tvpb_logsum_odt.extend_trace_label(tour_purpose)
            tvpb_logsum_dot.extend_trace_label(tour_purpose)

        # name index so tracing knows how to slice
        assert tours_segment.index.name == "tour_id"

        choices_df = run_tour_mode_choice_simulate(
            tours_segment,
            tour_purpose,
            model_settings,
            mode_column_name=mode_column_name,
            logsum_column_name=logsum_column_name,
            network_los=network_los,
            skims=skims,
            constants=constants,
            estimator=estimator,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
            trace_choice_name="tour_mode_choice",
        )

        tracing.print_summary(
            "tour_mode_choice_simulate %s choices_df" % tour_purpose,
            choices_df.tour_mode,
            value_counts=True,
        )

        choices_list.append(choices_df)

    choices_df = pd.concat(choices_list)

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:

        tvpb_mode_path_types = model_settings.get("tvpb_mode_path_types")
        if tvpb_mode_path_types is not None:
            for mode, path_types in tvpb_mode_path_types.items():

                for direction, skim in zip(
                    ["od", "do"], [tvpb_logsum_odt, tvpb_logsum_dot]
                ):

                    path_type = path_types[direction]
                    skim_cache = skim.cache[path_type]

                    print(f"mode {mode} direction {direction} path_type {path_type}")

                    for c in skim_cache:

                        dest_col = f"{direction}_{c}"

                        if dest_col not in choices_df:
                            choices_df[dest_col] = (
                                np.nan
                                if pd.api.types.is_numeric_dtype(skim_cache[c])
                                else ""
                            )
                        choices_df[dest_col].where(
                            choices_df.tour_mode != mode, skim_cache[c], inplace=True
                        )

    if estimator:
        estimator.write_choices(choices_df.tour_mode)
        choices_df.tour_mode = estimator.get_survey_values(
            choices_df.tour_mode, "tours", "tour_mode"
        )
        estimator.write_override_choices(choices_df.tour_mode)
        estimator.end_estimation()

    tracing.print_summary(
        "tour_mode_choice_simulate all tour type choices",
        choices_df.tour_mode,
        value_counts=True,
    )

    # so we can trace with annotations
    assign_in_place(primary_tours, choices_df)

    # update tours table with mode choice (and optionally logsums)
    all_tours = tours.to_frame()
    assign_in_place(all_tours, choices_df)

    pipeline.replace_table("tours", all_tours)

    # - annotate tours table
    if model_settings.get("annotate_tours"):
        tours = inject.get_table("tours").to_frame()
        expressions.assign_columns(
            df=tours,
            model_settings=model_settings.get("annotate_tours"),
            trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
        )
        pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(
            primary_tours,
            label=tracing.extend_trace_label(trace_label, mode_column_name),
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )

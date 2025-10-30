# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util import (
    school_escort_tours_trips,
    trip,
    logsums,
    park_and_ride_capacity,
)
from activitysim.abm.models.util.mode import run_tour_mode_choice_simulate
from activitysim.core import (
    config,
    estimation,
    logit,
    los,
    simulate,
    tracing,
    workflow,
    expressions,
)
from activitysim.core.configuration.logit import TourModeComponentSettings
from activitysim.abm.models.park_and_ride_lot_choice import (
    ParkAndRideLotChoiceSettings,
    run_park_and_ride_lot_choice,
)
from activitysim.core.util import assign_in_place, reindex

logger = logging.getLogger(__name__)

"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


def get_alts_from_segmented_nested_logit(
    state: workflow.State,
    model_settings: TourModeComponentSettings,
    segment_name: str,
    trace_label: str,
):
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
    coefficients = state.filesystem.get_segment_coefficients(
        model_settings, segment_name
    )
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)
    tour_mode_alts = []
    for nest in logit.each_nest(nest_spec):
        if nest.is_leaf:
            tour_mode_alts.append(nest.name)

    return tour_mode_alts


def create_logsum_trips(
    state: workflow.State,
    tours: pd.DataFrame,
    segment_column_name: str,
    model_settings: TourModeComponentSettings,
    trace_label: str,
):
    """
    Construct table of trips from half-tours (1 inbound, 1 outbound) for each tour-mode.

    Parameters
    ----------
    tours : pandas.DataFrame
    segment_column_name : str
        column in tours table used for segmenting model spec
    model_settings : TourModeComponentSettings
    trace_label : str

    Returns
    -------
    pandas.DataFrame
        Table of trips: 2 per tour, with O/D and purpose inherited from tour
    """
    stop_frequency_alts = state.get_injectable("stop_frequency_alts")
    stop_freq_cat_type = pd.api.types.CategoricalDtype(
        stop_frequency_alts.index.tolist() + [""], ordered=False
    )
    stop_freq = "0out_0in"  # no intermediate stops
    tours["stop_frequency"] = stop_freq
    tours["stop_frequency"] = tours["stop_frequency"].astype(stop_freq_cat_type)
    tours["primary_purpose"] = tours["tour_purpose"]
    trips = trip.initialize_from_tours(state, tours, stop_frequency_alts)
    trips["stop_frequency"] = stop_freq

    outbound = trips["outbound"]
    trips["depart"] = reindex(tours.start, trips.tour_id)
    trips.loc[~outbound, "depart"] = reindex(tours.end, trips.loc[~outbound, "tour_id"])

    # actual segment doesn't matter. just need to grab one
    # to get a set of coefficients from the spec
    segment_name = tours.iloc[0][segment_column_name]
    tour_mode_alts = get_alts_from_segmented_nested_logit(
        state, model_settings, segment_name, trace_label
    )

    mode_cat_type = pd.api.types.CategoricalDtype(tour_mode_alts + [""], ordered=False)
    # repeat rows from the trips table iterating over tour mode
    logsum_trips = pd.DataFrame()
    for tour_mode in tour_mode_alts:
        trips["tour_mode"] = tour_mode
        trips["tour_mode"] = trips["tour_mode"].astype(mode_cat_type)
        logsum_trips = pd.concat((logsum_trips, trips), ignore_index=True)
    assert len(logsum_trips) == len(trips) * len(tour_mode_alts)
    logsum_trips.index.name = "trip_id"

    return logsum_trips


def append_tour_leg_trip_mode_choice_logsums(state: workflow.State, tours):
    """Creates trip mode choice logsum column in tours table for each tour mode and leg

    Parameters
    ----------
    tours : pd.DataFrame

    Returns
    -------
    tours : pd.DataFrame
        Adds two * n_modes logsum columns to each tour row, e.g. "logsum_DRIVE_outbound"
    """
    trips = state.get_dataframe("trips")
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
    state: workflow.State,
    tours: pd.DataFrame,
    segment_column_name: str,
    model_settings: TourModeComponentSettings,
    trace_label: str,
):
    """Creates pseudo-trips from tours and runs trip mode choice to get logsums

    Parameters
    ----------
    tours : pandas.DataFrame
    segment_column_name : str
        column in tours table used for segmenting model spec
    model_settings : TourModeComponentSettings
    trace_label : str

    Returns
    -------
    tours : pd.DataFrame
        Adds two * n_modes logsum columns to each tour row, e.g. "logsum_DRIVE_outbound"
    """

    # create pseudo-trips from tours for all tour modes
    logsum_trips = create_logsum_trips(
        state, tours, segment_column_name, model_settings, trace_label
    )

    # temporarily register trips in the pipeline
    state.add_table("trips", logsum_trips)
    state.tracing.register_traceable_table("trips", logsum_trips)
    state.get_rn_generator().add_channel("trips", logsum_trips)

    # run trip mode choice on pseudo-trips. use a direct call instead of pipeline to
    # execute the step because pipeline can only handle one open step at a time
    from .trip_mode_choice import trip_mode_choice

    trip_mode_choice(state, logsum_trips, state.get("network_los"))

    # add trip mode choice logsums as new cols in tours
    tours = append_tour_leg_trip_mode_choice_logsums(state, tours)

    # de-register logsum trips table
    state.get_rn_generator().drop_channel("trips")
    state.tracing.deregister_traceable_table("trips")

    return tours


@workflow.step
def tour_mode_choice_simulate(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: TourModeComponentSettings | None = None,
    model_settings_file_name: str = "tour_mode_choice.yaml",
    trace_label: str = "tour_mode_choice",
) -> None:
    """
    Tour mode choice simulate
    """
    if model_settings is None:
        model_settings = TourModeComponentSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    logsum_column_name = model_settings.MODE_CHOICE_LOGSUM_COLUMN_NAME
    mode_column_name = "tour_mode"
    segment_column_name = "tour_purpose"

    primary_tours = tours
    assert not (primary_tours.tour_category == "atwork").any()

    logger.info("Running %s with %d tours" % (trace_label, primary_tours.shape[0]))

    tracing.print_summary("tour_types", primary_tours.tour_type, value_counts=True)

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
    constants.update(model_settings.CONSTANTS)

    # setup skim keys
    skims = logsums.setup_skims(
        network_los,
        primary_tours_merged,
        add_periods=False,
        include_pnr_skims=("pnr_zone_id" in primary_tours_merged.columns),
        trace_label=trace_label,
    )

    # TVPB constants can appear in expressions
    if (network_los.zone_system == los.THREE_ZONE) & model_settings.use_TVPB_constants:
        constants.update(
            network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
        )

    # don't create estimation data bundle if trip mode choice is being called
    # from another model step (i.e. tour mode choice logsum creation)
    if state.get_rn_generator().step_name != "tour_mode_choice_simulate":
        estimator = None
    else:
        estimator = estimation.manager.begin_estimation(state, "tour_mode_choice")
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
    if model_settings.COMPUTE_TRIP_MODE_CHOICE_LOGSUMS:
        primary_tours_merged = get_trip_mc_logsums_for_all_modes(
            state,
            primary_tours_merged,
            segment_column_name,
            model_settings,
            trace_label,
        )

    max_iterations = 1  # default number of times to run tour mode choice if park-and-ride is not capacitated

    # if park-and-ride is included, need to check whether we iterate and how many times
    if "pnr_zone_id" in primary_tours_merged.columns:
        # read the park-and-ride lot choice model settings
        pnr_model_settings = ParkAndRideLotChoiceSettings.read_settings_file(
            state.filesystem,
            "park_and_ride_lot_choice.yaml",
        )
        # only iterate if enabled and not in estimation mode
        if (
            pnr_model_settings.ITERATE_WITH_TOUR_MODE_CHOICE
            and (pnr_model_settings.MAX_ITERATIONS > 0)
            and (estimator is None)
        ):
            max_iterations = pnr_model_settings.MAX_ITERATIONS

            # create pnr capacity helper class
            pnr_capacity_cls = park_and_ride_capacity.ParkAndRideCapacity(
                state, pnr_model_settings
            )

    choosers = primary_tours_merged
    final_choices = None
    base_trace_label = trace_label

    # iterating tour mode choice with park-and-ride lot choice
    # the first iteration includes all tours
    # subsequent iterations only includes tours that selected a capacitated park-and-ride lot
    for i in range(max_iterations):
        if max_iterations > 1:
            trace_label = tracing.extend_trace_label(base_trace_label, f"i{i}")

        choices_list = []
        for tour_purpose, tours_segment in choosers.groupby(
            segment_column_name, observed=True
        ):
            logger.info(
                "tour_mode_choice_simulate tour_type '%s' (%s tours)"
                % (
                    tour_purpose,
                    len(tours_segment.index),
                )
            )

            if network_los.zone_system == los.THREE_ZONE:
                skims["tvpb_logsum_odt"].extend_trace_label(tour_purpose)
                skims["tvpb_logsum_dot"].extend_trace_label(tour_purpose)

            # name index so tracing knows how to slice
            assert tours_segment.index.name == "tour_id"

            choices_df = run_tour_mode_choice_simulate(
                state,
                tours_segment,
                tour_purpose,
                model_settings,
                mode_column_name=mode_column_name,
                logsum_column_name=logsum_column_name,
                network_los=network_los,
                skims=skims,
                constants=constants,
                estimator=estimator,
                trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
                trace_choice_name="tour_mode_choice",
            )

            tracing.print_summary(
                "tour_mode_choice_simulate %s choices_df" % tour_purpose,
                choices_df.tour_mode,
                value_counts=True,
            )

            choices_list.append(choices_df)

        choices_i = pd.concat(choices_list)
        if final_choices is None:
            final_choices = choices_i.copy()
        else:
            # need to just update the existing choices with the new ones decided during this iteration
            final_choices.loc[choices_i.index] = choices_i

        if (max_iterations > 1) and (i < max_iterations - 1):
            # need to update the park-and-ride lot capacities and select new choosers
            pnr_capacity_cls.iteration = i
            # grabbing pnr_zone_id to calculate capacities
            choices_i["pnr_zone_id"] = choosers["pnr_zone_id"].reindex(choices_i.index)
            # grabbing start time to help determine which tours need to get resimulated
            choices_i["start"] = choosers["start"].reindex(choices_i.index)
            pnr_capacity_cls.set_choices(choices_i)
            choosers = pnr_capacity_cls.select_new_choosers(state, choosers)
            if choosers.empty:
                logger.info(
                    f"finished tour mode choice iterations at iteration {i} because all park-and-ride demand was met"
                )
                if pnr_capacity_cls.num_processes > 1:
                    # need to have this subprocess check-in still to satisfy barrier in synchronize_choices
                    for j in range(i, max_iterations):
                        dummy_choices = pd.DataFrame(columns=choices_i.columns)
                        pnr_capacity_cls.set_choices(dummy_choices)
                break
            choosers["pnr_zone_id"] = run_park_and_ride_lot_choice(
                state,
                choosers=choosers,
                land_use=state.get_dataframe("land_use"),
                network_los=network_los,
                model_settings=pnr_capacity_cls.model_settings,
                choosers_dest_col_name="destination",
                choosers_origin_col_name="home_zone_id",
                estimator=estimator,
                pnr_capacity_cls=pnr_capacity_cls,
                trace_label=trace_label,
            )
            # drop out_period and in_period from choosers since they get generated in run_tour_mode_choice_simulate
            # they are added since we are not sending a copy of the choosers to the above pnr lot choice call
            choosers.drop(
                columns=["out_period", "in_period"], errors="ignore", inplace=True
            )

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:
        tvpb_mode_path_types = model_settings.tvpb_mode_path_types
        if tvpb_mode_path_types is not None:
            for mode, path_types in tvpb_mode_path_types.items():
                for direction, skim in zip(
                    ["od", "do"], [skims["tvpb_logsum_odt"], skims["tvpb_logsum_dot"]]
                ):
                    path_type = path_types[direction]
                    skim_cache = skim.cache[path_type]

                    print(f"mode {mode} direction {direction} path_type {path_type}")

                    for c in skim_cache:
                        dest_col = f"{direction}_{c}"

                        if dest_col not in final_choices:
                            final_choices[dest_col] = (
                                np.nan
                                if pd.api.types.is_numeric_dtype(skim_cache[c])
                                else ""
                            )
                        final_choices[dest_col].where(
                            final_choices.tour_mode != mode, skim_cache[c], inplace=True
                        )

    if estimator:
        estimator.write_choices(final_choices.tour_mode)
        final_choices.tour_mode = estimator.get_survey_values(
            final_choices.tour_mode, "tours", "tour_mode"
        )
        estimator.write_override_choices(final_choices.tour_mode)
        estimator.end_estimation()

    tracing.print_summary(
        "tour_mode_choice_simulate all tour type choices",
        final_choices.tour_mode,
        value_counts=True,
    )

    # so we can trace with annotations
    assign_in_place(
        primary_tours,
        final_choices,
        state.settings.downcast_int,
        state.settings.downcast_float,
    )

    # update tours table with mode choice (and optionally logsums)
    all_tours = tours
    assign_in_place(
        all_tours,
        final_choices,
        state.settings.downcast_int,
        state.settings.downcast_float,
    )

    if (
        state.is_table("school_escort_tours")
        & model_settings.FORCE_ESCORTEE_CHAUFFEUR_MODE_MATCH
    ):
        all_tours = (
            school_escort_tours_trips.force_escortee_tour_modes_to_match_chauffeur(
                state, all_tours
            )
        )

    state.add_table("tours", all_tours)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(
            primary_tours,
            label=tracing.extend_trace_label(trace_label, mode_column_name),
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )

    expressions.annotate_tables(
        state,
        locals_dict=constants,
        skims=skims,
        model_settings=model_settings,
        trace_label=trace_label,
    )

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    expressions,
    logit,
    los,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.tracing import print_elapsed_time
from activitysim.core.util import assign_in_place, drop_unused_columns

logger = logging.getLogger(__name__)

NO_DESTINATION = -1


def wrap_skims(state: workflow.State, model_settings: ParkingLocationSettings):
    """
    wrap skims of trip destination using origin, dest column names from model settings.
    Various of these are used by destination_sample, compute_logsums, and destination_simulate
    so we create them all here with canonical names.

    Note that compute_logsums aliases their names so it can use the same equations to compute
    logsums from origin to alt_dest, and from alt_dest to primarly destination

    odt_skims - SkimStackWrapper: trip origin, trip alt_dest, time_of_day
    dot_skims - SkimStackWrapper: trip alt_dest, trip origin, time_of_day
    dpt_skims - SkimStackWrapper: trip alt_dest, trip primary_dest, time_of_day
    pdt_skims - SkimStackWrapper: trip primary_dest,trip alt_dest, time_of_day
    od_skims - SkimDictWrapper: trip origin, trip alt_dest
    dp_skims - SkimDictWrapper: trip alt_dest, trip primary_dest

    Parameters
    ----------
    model_settings

    Returns
    -------
        dict containing skims, keyed by canonical names relative to tour orientation
    """

    network_los = state.get_injectable("network_los")
    skim_dict = network_los.get_default_skim_dict()

    origin = model_settings.TRIP_ORIGIN
    park_zone = model_settings.ALT_DEST_COL_NAME
    destination = model_settings.TRIP_DESTINATION
    time_period = model_settings.TRIP_DEPARTURE_PERIOD

    skims = {
        "odt_skims": skim_dict.wrap_3d(
            orig_key=origin, dest_key=destination, dim3_key=time_period
        ),
        "dot_skims": skim_dict.wrap_3d(
            orig_key=destination, dest_key=origin, dim3_key=time_period
        ),
        "opt_skims": skim_dict.wrap_3d(
            orig_key=origin, dest_key=park_zone, dim3_key=time_period
        ),
        "pdt_skims": skim_dict.wrap_3d(
            orig_key=park_zone, dest_key=destination, dim3_key=time_period
        ),
        "od_skims": skim_dict.wrap(origin, destination),
        "do_skims": skim_dict.wrap(destination, origin),
        "op_skims": skim_dict.wrap(origin, park_zone),
        "pd_skims": skim_dict.wrap(park_zone, destination),
    }

    return skims


def get_spec_for_segment(
    state: workflow.State, model_settings: ParkingLocationSettings, segment: str
):
    omnibus_spec = state.filesystem.read_model_spec(
        file_name=model_settings.SPECIFICATION
    )

    spec = omnibus_spec[[segment]]

    # might as well ignore any spec rows with 0 utility
    spec = spec[spec.iloc[:, 0] != 0]
    assert spec.shape[0] > 0

    return spec


def parking_destination_simulate(
    state: workflow.State,
    segment_name,
    trips,
    destination_sample,
    model_settings: ParkingLocationSettings,
    skims,
    locals_dict,
    chunk_size,
    trace_hh_id,
    trace_label,
):
    """
    Chose destination from destination_sample (with od_logsum and dp_logsum columns added)


    Returns
    -------
    choices - pandas.Series
        destination alt chosen
    """
    trace_label = tracing.extend_trace_label(
        trace_label, "parking_destination_simulate"
    )

    spec = get_spec_for_segment(state, model_settings, segment_name)

    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    spec = simulate.eval_coefficients(state, spec, coefficients_df, None)

    alt_dest_col_name = model_settings.ALT_DEST_COL_NAME

    logger.info("Running parking_destination_simulate with %d trips", len(trips))

    parking_locations = interaction_sample_simulate(
        state,
        choosers=trips,
        alternatives=destination_sample,
        spec=spec,
        choice_column=alt_dest_col_name,
        want_logsums=False,
        allow_zero_probs=True,
        zero_prob_choice_val=NO_DESTINATION,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="parking_loc",
        explicit_chunk_size=model_settings.explicit_chunk,
    )

    # drop any failed zero_prob destinations
    if (parking_locations == NO_DESTINATION).any():
        logger.debug(
            "dropping %s failed parking locations",
            (parking_locations == NO_DESTINATION).sum(),
        )
        parking_locations = parking_locations[parking_locations != NO_DESTINATION]

    return parking_locations


def choose_parking_location(
    state: workflow.State,
    segment_name,
    trips,
    alternatives,
    model_settings: ParkingLocationSettings,
    want_sample_table,
    skims,
    chunk_size,
    trace_hh_id,
    trace_label,
):
    logger.info("choose_parking_location %s with %d trips", trace_label, trips.shape[0])

    t0 = print_elapsed_time()

    alt_dest_col_name = model_settings.ALT_DEST_COL_NAME

    # remove trips and alts columns that are not used in spec
    locals_dict = state.get_global_constants()
    locals_dict.update(config.get_model_constants(model_settings))
    locals_dict.update(skims)
    locals_dict["timeframe"] = "trip"
    locals_dict["PARKING"] = skims["op_skims"].dest_key

    spec = get_spec_for_segment(state, model_settings, segment_name)
    trips = drop_unused_columns(
        trips,
        spec,
        locals_dict,
        custom_chooser=None,
        additional_columns=model_settings.compute_settings.protect_columns,
    )
    alternatives = drop_unused_columns(
        alternatives,
        spec,
        locals_dict,
        custom_chooser=None,
        additional_columns=model_settings.compute_settings.protect_columns,
    )

    destination_sample = logit.interaction_dataset(
        state, trips, alternatives, alt_index_id=alt_dest_col_name
    )
    destination_sample.index = np.repeat(trips.index.values, len(alternatives))
    destination_sample.index.name = trips.index.name

    destinations = parking_destination_simulate(
        state,
        segment_name=segment_name,
        trips=trips,
        destination_sample=destination_sample,
        model_settings=model_settings,
        skims=skims,
        locals_dict=locals_dict,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    if want_sample_table:
        # FIXME - sample_table
        destination_sample.set_index(
            model_settings.ALT_DEST_COL_NAME, append=True, inplace=True
        )
    else:
        destination_sample = None

    t0 = print_elapsed_time("%s.parking_location_simulate" % trace_label, t0)

    return destinations, destination_sample


def run_parking_destination(
    state: workflow.State,
    model_settings: ParkingLocationSettings,
    trips,
    land_use,
    chunk_size,
    trace_hh_id,
    trace_label,
    fail_some_trips_for_testing=False,
):
    chooser_filter_column = model_settings.CHOOSER_FILTER_COLUMN_NAME
    chooser_segment_column = model_settings.CHOOSER_SEGMENT_COLUMN_NAME

    parking_location_column_name = model_settings.ALT_DEST_COL_NAME
    sample_table_name = model_settings.DEST_CHOICE_SAMPLE_TABLE_NAME
    want_sample_table = (
        state.settings.want_dest_choice_sample_tables and sample_table_name is not None
    )

    choosers = trips[trips[chooser_filter_column]]
    choosers = choosers.sort_index()

    # Placeholder for trips without a parking choice
    trips[parking_location_column_name] = -1

    skims = wrap_skims(state, model_settings)

    alt_column_filter_name = model_settings.ALTERNATIVE_FILTER_COLUMN_NAME
    alternatives = land_use[land_use[alt_column_filter_name]]
    alternatives.index.name = parking_location_column_name

    choices_list = []
    sample_list = []
    for segment_name, chooser_segment in choosers.groupby(chooser_segment_column):
        if chooser_segment.shape[0] == 0:
            logger.info(
                "%s skipping segment %s: no choosers", trace_label, segment_name
            )
            continue

        choices, destination_sample = choose_parking_location(
            state,
            segment_name,
            chooser_segment,
            alternatives,
            model_settings,
            want_sample_table,
            skims,
            chunk_size,
            trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, segment_name),
        )

        choices_list.append(choices)
        if want_sample_table:
            assert destination_sample is not None
            sample_list.append(destination_sample)

    if len(choices_list) > 0:
        parking_df = pd.concat(choices_list)

        if fail_some_trips_for_testing:
            parking_df = parking_df.drop(parking_df.index[0])

        assign_in_place(
            trips,
            parking_df.to_frame(parking_location_column_name),
            state.settings.downcast_int,
            state.settings.downcast_float,
        )
        trips[parking_location_column_name] = trips[
            parking_location_column_name
        ].fillna(-1)
    else:
        trips[parking_location_column_name] = -1

    save_sample_df = pd.concat(sample_list) if len(sample_list) > 0 else None

    return trips[parking_location_column_name], save_sample_df


class ParkingLocationSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `parking_location` component.
    """

    SPECIFICATION: Path | None = None
    SPEC: None = None
    """The school escort model does not use this setting, see `SPECIFICATION`."""

    PREPROCESSOR: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    ALT_DEST_COL_NAME: str = "parking_zone"
    """Parking destination column name."""

    TRIP_DEPARTURE_PERIOD: str = "stop_period"
    """Trip departure time period."""

    PARKING_LOCATION_SAMPLE_TABLE_NAME: str | None = None

    TRIP_ORIGIN: str = "origin"
    TRIP_DESTINATION: str = "destination"

    CHOOSER_FILTER_COLUMN_NAME: str
    """A boolean column to filter choosers.

    If this column evaluates as True the row will be kept.
    """

    CHOOSER_SEGMENT_COLUMN_NAME: str

    DEST_CHOICE_SAMPLE_TABLE_NAME: str | None = None

    ALTERNATIVE_FILTER_COLUMN_NAME: str

    SEGMENTS: list[str] | None = None

    AUTO_MODES: list[str]
    """List of auto modes that use parking. AUTO_MODES are used in write_trip_matrices to make sure
    parking locations are accurately represented in the output trip matrices."""

    explicit_chunk: float = 0
    """
    If > 0, use this chunk size instead of adaptive chunking.
    If less than 1, use this fraction of the total number of rows.
    """


@workflow.step
def parking_location(
    state: workflow.State,
    trips: pd.DataFrame,
    trips_merged: pd.DataFrame,
    land_use: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: ParkingLocationSettings | None = None,
    model_settings_file_name: str = "parking_location_choice.yaml",
    trace_label: str = "parking_location",
) -> None:
    """
    Given a set of trips, each trip needs to have a parking location if
    it is eligible for remote parking.
    """

    if model_settings is None:
        model_settings = ParkingLocationSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    trace_hh_id = state.settings.trace_hh_id
    alt_destination_col_name = model_settings.ALT_DEST_COL_NAME

    preprocessor_settings = model_settings.PREPROCESSOR

    trips_df = trips
    trips_merged_df = trips_merged
    land_use_df = land_use

    proposed_trip_departure_period = model_settings.TRIP_DEPARTURE_PERIOD
    # TODO: the number of skim time periods should be more readily available than this
    n_skim_time_periods = np.unique(
        network_los.los_settings.skim_time_periods.labels
    ).size
    if trips_merged_df[proposed_trip_departure_period].max() > n_skim_time_periods:
        # max proposed_trip_departure_period is out of range,
        # it is most likely the high-resolution time period, we need the skim-level time period
        if "trip_period" not in trips_merged_df:
            # TODO: resolve this to the skim time period index not the label, it will be faster
            trips_merged_df["trip_period"] = network_los.skim_time_period_label(
                trips_merged_df[proposed_trip_departure_period], as_cat=True
            )
        model_settings.TRIP_DEPARTURE_PERIOD = "trip_period"

    locals_dict = {"network_los": network_los}

    constants = config.get_model_constants(model_settings)

    if constants is not None:
        locals_dict.update(constants)

    if preprocessor_settings:
        expressions.assign_columns(
            state,
            df=trips_merged_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    parking_locations, save_sample_df = run_parking_destination(
        state,
        model_settings,
        trips_merged_df,
        land_use_df,
        chunk_size=state.settings.chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    assign_in_place(
        trips_df,
        parking_locations.to_frame(alt_destination_col_name),
        state.settings.downcast_int,
        state.settings.downcast_float,
    )

    state.add_table("trips", trips_df)

    if trace_hh_id:
        state.tracing.trace_df(
            trips_df,
            label=trace_label,
            slicer="trip_id",
            index_label="trip_id",
            warn_if_empty=True,
        )

    if save_sample_df is not None:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(
            trips_df[trips_df.trip_num < trips_df.trip_count]
        )

        sample_table_name = model_settings.PARKING_LOCATION_SAMPLE_TABLE_NAME
        assert sample_table_name is not None

        logger.info(f"adding {len(save_sample_df)} samples to {sample_table_name}")

        # lest they try to put tour samples into the same table
        if state.is_table(sample_table_name):
            raise RuntimeError("sample table %s already exists" % sample_table_name)
        state.extend_table(sample_table_name, save_sample_df)

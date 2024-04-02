# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import openmatrix as omx
import pandas as pd

from activitysim.core import config, expressions, los, workflow
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.abm.models.parking_location_choice import ParkingLocationSettings

logger = logging.getLogger(__name__)


class MatrixTableSettings(PydanticReadable):
    name: str
    data_field: str


class MatrixSettings(PydanticReadable):
    file_name: Path
    tables: list[MatrixTableSettings] = []
    is_tap: bool = False


class WriteTripMatricesSettings(PydanticReadable):
    """
    Settings for the `write_trip_matrices` component.
    """

    SAVE_TRIPS_TABLE: bool = False
    """Save trip tables"""

    HH_EXPANSION_WEIGHT_COL: str = "sample_rate"
    """Column represents the sampling rate of households"""

    MATRICES: list[MatrixSettings] = []

    CONSTANTS: dict[str, Any] = {}

    preprocessor: PreprocessorSettings | None = None


@workflow.step(copy_tables=["trips"])
def write_trip_matrices(
    state: workflow.State,
    network_los: los.Network_LOS,
    trips: pd.DataFrame,
    model_settings: WriteTripMatricesSettings | None = None,
    model_settings_file_name: str = "write_trip_matrices.yaml",
) -> None:
    """
    Write trip matrices step.

    Adds boolean columns to local trips table via annotation expressions,
    then aggregates trip counts and writes OD matrices to OMX.  Save annotated
    trips table to pipeline if desired.

    Writes taz trip tables for one and two zone system.  Writes taz and tap
    trip tables for three zone system.  Add ``is_tap:True`` to the settings file
    to identify an output matrix as tap level trips as opposed to taz level trips.

    For one zone system, uses the land use table for the set of possible tazs.
    For two zone system, uses the taz skim zone names for the set of possible tazs.
    For three zone system, uses the taz skim zone names for the set of possible tazs
    and uses the tap skim zone names for the set of possible taps.

    """

    if trips is None:
        # this step is a NOP if there is no trips table
        # this might legitimately happen if they comment out some steps to debug but still want write_tables
        # this saves them the hassle of remembering to comment out this step
        logger.warning(
            "write_trip_matrices returning empty-handed because there is no trips table"
        )
        return

    if model_settings is None:
        model_settings = WriteTripMatricesSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    trips_df = annotate_trips(state, trips, network_los, model_settings)

    if model_settings.SAVE_TRIPS_TABLE:
        state.add_table("trips", trips_df)

    if "parking_location" in state.settings.models:
        parking_settings = ParkingLocationSettings.read_settings_file(
            state.filesystem,
            "parking_location_choice.yaml",
        )
        parking_taz_col_name = parking_settings.ALT_DEST_COL_NAME
        if ~(trips_df["trip_mode"].isin(parking_settings.AUTO_MODES)).any():
            logger.warning(
                f"Parking location choice model is enabled, but none of {parking_settings.AUTO_MODES} auto modes found in trips table."
                "See AUTO_MODES setting in parking_location_choice.yaml."
            )

        if parking_taz_col_name in trips_df:
            trips_df["true_origin"] = trips_df["origin"]
            trips_df["true_destination"] = trips_df["destination"]

            # Get origin parking zone if vehicle not parked at origin
            trips_df["origin_parking_zone"] = np.where(
                (trips_df["tour_id"] == trips_df["tour_id"].shift(1))
                & trips_df["trip_mode"].isin(parking_settings.AUTO_MODES),
                trips_df[parking_taz_col_name].shift(1),
                -1,
            )

            trips_df.loc[trips_df[parking_taz_col_name] > 0, "destination"] = trips_df[
                parking_taz_col_name
            ]
            trips_df.loc[trips_df["origin_parking_zone"] > 0, "origin"] = trips_df[
                "origin_parking_zone"
            ]

    # write matrices by zone system type
    if network_los.zone_system == los.ONE_ZONE:  # taz trips written to taz matrices
        logger.info("aggregating trips one zone...")
        aggregate_trips = trips_df.groupby(["origin", "destination"], sort=False).sum(
            numeric_only=True
        )

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL
        aggregate_weight = (
            trips_df[["origin", "destination", hh_weight_col]]
            .groupby(["origin", "destination"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("origin")
        dest_vals = aggregate_trips.index.get_level_values("destination")

        # use the land use table for the set of possible tazs
        land_use = state.get_dataframe("land_use")
        zone_index = land_use.index
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        try:
            zone_labels = land_use[f"_original_{land_use.index.name}"]
        except KeyError:
            zone_labels = land_use.index

        write_matrices(
            state, aggregate_trips, zone_labels, orig_index, dest_index, model_settings
        )

    elif network_los.zone_system == los.TWO_ZONE:  # maz trips written to taz matrices
        logger.info("aggregating trips two zone...")
        trips_df["otaz"] = (
            state.get_dataframe("land_use").reindex(trips_df["origin"]).TAZ.tolist()
        )
        trips_df["dtaz"] = (
            state.get_dataframe("land_use")
            .reindex(trips_df["destination"])
            .TAZ.tolist()
        )
        aggregate_trips = trips_df.groupby(["otaz", "dtaz"], sort=False).sum(
            numeric_only=True
        )

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL
        aggregate_weight = (
            trips_df[["otaz", "dtaz", hh_weight_col]]
            .groupby(["otaz", "dtaz"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("otaz")
        dest_vals = aggregate_trips.index.get_level_values("dtaz")

        try:
            land_use_taz = state.get_dataframe("land_use_taz")
        except (KeyError, RuntimeError):
            pass  # table missing, ignore
        else:
            if "_original_TAZ" in land_use_taz.columns:
                orig_vals = orig_vals.map(land_use_taz["_original_TAZ"])
                dest_vals = dest_vals.map(land_use_taz["_original_TAZ"])

        zone_index = pd.Index(network_los.get_tazs(state), name="TAZ")
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        write_matrices(
            state, aggregate_trips, zone_index, orig_index, dest_index, model_settings
        )

    elif (
        network_los.zone_system == los.THREE_ZONE
    ):  # maz trips written to taz and tap matrices
        logger.info("aggregating trips three zone taz...")
        trips_df["otaz"] = (
            state.get_dataframe("land_use").reindex(trips_df["origin"]).TAZ.tolist()
        )
        trips_df["dtaz"] = (
            state.get_dataframe("land_use")
            .reindex(trips_df["destination"])
            .TAZ.tolist()
        )
        aggregate_trips = trips_df.groupby(["otaz", "dtaz"], sort=False).sum(
            numeric_only=True
        )

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL
        aggregate_weight = (
            trips_df[["otaz", "dtaz", hh_weight_col]]
            .groupby(["otaz", "dtaz"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("otaz")
        dest_vals = aggregate_trips.index.get_level_values("dtaz")

        try:
            land_use_taz = state.get_dataframe("land_use_taz")
        except (KeyError, RuntimeError):
            pass  # table missing, ignore
        else:
            if "_original_TAZ" in land_use_taz.columns:
                orig_vals = orig_vals.map(land_use_taz["_original_TAZ"])
                dest_vals = dest_vals.map(land_use_taz["_original_TAZ"])

        zone_index = pd.Index(network_los.get_tazs(state), name="TAZ")
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        write_matrices(
            state, aggregate_trips, zone_index, orig_index, dest_index, model_settings
        )

        logger.info("aggregating trips three zone tap...")
        aggregate_trips = trips_df.groupby(["btap", "atap"], sort=False).sum(
            numeric_only=True
        )

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL
        aggregate_weight = (
            trips_df[["btap", "atap", hh_weight_col]]
            .groupby(["btap", "atap"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("btap")
        dest_vals = aggregate_trips.index.get_level_values("atap")

        zone_index = pd.Index(network_los.get_taps(), name="TAP")
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        write_matrices(
            state,
            aggregate_trips,
            zone_index,
            orig_index,
            dest_index,
            model_settings,
            True,
        )

    if "parking_location" in state.settings.models:
        # Set trip origin and destination to be the actual location the person is and not where their vehicle is parked
        trips_df["origin"] = trips_df["true_origin"]
        trips_df["destination"] = trips_df["true_destination"]
        del trips_df["true_origin"], trips_df["true_destination"]
        if (
            network_los.zone_system == los.TWO_ZONE
            or network_los.zone_system == los.THREE_ZONE
        ):
            trips_df["otaz"] = (
                state.get_table("land_use").reindex(trips_df["origin"]).TAZ.tolist()
            )
            trips_df["dtaz"] = (
                state.get_table("land_use")
                .reindex(trips_df["destination"])
                .TAZ.tolist()
            )


def annotate_trips(
    state: workflow.State,
    trips: pd.DataFrame,
    network_los,
    model_settings: WriteTripMatricesSettings,
):
    """
    Add columns to local trips table. The annotator has
    access to the origin/destination skims and everything
    defined in the model settings CONSTANTS.

    Pipeline tables can also be accessed by listing them under
    TABLES in the preprocessor settings.
    """

    trips_df = trips

    trace_label = "trip_matrices"

    skim_dict = network_los.get_default_skim_dict()

    # setup skim keys
    if "trip_period" not in trips_df:
        trips_df["trip_period"] = network_los.skim_time_period_label(trips_df.depart)
    od_skim_wrapper = skim_dict.wrap("origin", "destination")
    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key="origin", dest_key="destination", dim3_key="trip_period"
    )
    skims = {"od_skims": od_skim_wrapper, "odt_skims": odt_skim_stack_wrapper}

    locals_dict = {}
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_dict.update(constants)

    expressions.annotate_preprocessors(
        state, trips_df, locals_dict, skims, model_settings, trace_label
    )

    if not np.issubdtype(trips_df["trip_period"].dtype, np.integer):
        if hasattr(skim_dict, "map_time_periods_from_series"):
            trip_period_idx = skim_dict.map_time_periods_from_series(
                trips_df["trip_period"]
            )
            if trip_period_idx is not None:
                trips_df["trip_period"] = trip_period_idx

    # Data will be expanded by an expansion weight column from
    # the households pipeline table, if specified in the model settings.
    hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL

    if hh_weight_col and hh_weight_col not in trips_df:
        logger.info("adding '%s' from households to trips table" % hh_weight_col)
        household_weights = state.get_dataframe("households")[hh_weight_col]
        trips_df[hh_weight_col] = trips_df.household_id.map(household_weights)

    return trips_df


def write_matrices(
    state: workflow.State,
    aggregate_trips,
    zone_index,
    orig_index,
    dest_index,
    model_settings: WriteTripMatricesSettings,
    is_tap=False,
):
    """
    Write aggregated trips to OMX format.

    The MATRICES setting lists the new OMX files to write.
    Each file can contain any number of 'tables', each specified by a
    table key ('name') and a trips table column ('data_field') to use
    for aggregated counts.

    Any data type may be used for columns added in the annotation phase,
    but the table 'data_field's must be summable types: ints, floats, bools.
    """

    matrix_settings = model_settings.MATRICES

    if not matrix_settings:
        logger.error("Missing MATRICES setting in write_trip_matrices.yaml")

    for matrix in matrix_settings:
        matrix_is_tap = matrix.is_tap

        if matrix_is_tap == is_tap:  # only write tap matrices to tap matrix files
            filename = str(matrix.file_name)
            filepath = state.get_output_file_path(filename)
            logger.info("opening %s" % filepath)
            file = omx.open_file(str(filepath), "w")  # possibly overwrite existing file
            table_settings = matrix.tables

            for table in table_settings:
                table_name = table.name
                col = table.data_field

                if col not in aggregate_trips:
                    logger.error(f"missing {col} column in aggregate_trips DataFrame")
                    return

                hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL
                if hh_weight_col:
                    aggregate_trips[col] = (
                        aggregate_trips[col] / aggregate_trips[hh_weight_col]
                    )

                data = np.zeros((len(zone_index), len(zone_index)))
                data[orig_index, dest_index] = aggregate_trips[col]
                logger.debug(
                    "writing %s sum %0.2f" % (table_name, aggregate_trips[col].sum())
                )
                file[table_name] = data  # write to file

            # include the index-to-zone map in the file
            logger.info(
                "adding %s mapping for %s zones to %s"
                % (zone_index.name, zone_index.size, filename)
            )
            file.create_mapping(zone_index.name, zone_index.to_numpy())

            logger.info("closing %s" % filepath)
            file.close()

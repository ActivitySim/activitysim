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
    """Name of the core in the omx output file"""
    data_field: str
    """Column in the trips table to aggregate"""
    origin: str = "origin"
    """Column in the trips table representing the from zone"""
    destination: str = "destination"
    """Column in the trips table representing the to zone"""


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

        # use the land use table for the set of possible tazs
        land_use = state.get_dataframe("land_use")
        try:
            zone_labels = land_use[f"_original_{land_use.index.name}"]
        except KeyError:
            zone_labels = land_use.index

        write_matrices(
            state=state,
            trips_df=trips_df,
            zone_index=zone_labels,  # Series or Index; used for mapping and shape
            model_settings=model_settings,
            is_tap=False,
        )

    elif network_los.zone_system == los.TWO_ZONE:  # maz trips written to taz matrices
        logger.info("aggregating trips two zone...")

        # TAZ domain for output
        zone_index = pd.Index(network_los.get_tazs(state), name="TAZ")

        # If original TAZ labels are provided, use them in the OMX mapping
        try:
            land_use_taz = state.get_dataframe("land_use_taz")
        except (KeyError, RuntimeError):
            pass  # table missing, ignore
        else:
            if "_original_TAZ" in land_use_taz.columns:
                zone_index = pd.Series(
                    land_use_taz["_original_TAZ"]
                    .reindex(zone_index)
                    .fillna(zone_index)
                    .values,
                    index=pd.Index(zone_index, name="TAZ"),
                    name="TAZ",
                )

        write_matrices(
            state=state,
            trips_df=trips_df,
            zone_index=zone_index,
            model_settings=model_settings,
            is_tap=False,
        )

    elif (
        network_los.zone_system == los.THREE_ZONE
    ):  # maz trips written to taz and tap matrices
        logger.info("aggregating trips three zone taz...")

        # TAZ domain for output
        zone_index = pd.Index(network_los.get_tazs(state), name="TAZ")

        try:
            land_use_taz = state.get_dataframe("land_use_taz")
        except (KeyError, RuntimeError):
            pass  # table missing, ignore
        else:
            if "_original_TAZ" in land_use_taz.columns:
                zone_index = pd.Series(
                    land_use_taz["_original_TAZ"]
                    .reindex(zone_index)
                    .fillna(zone_index)
                    .values,
                    index=pd.Index(zone_index, name="TAZ"),
                    name="TAZ",
                )

        write_matrices(
            state=state,
            trips_df=trips_df,
            zone_index=zone_index,
            model_settings=model_settings,
            is_tap=False,
        )

        logger.info("aggregating trips three zone tap...")

        tap_index = pd.Index(network_los.get_taps(), name="TAP")

        write_matrices(
            state=state,
            trips_df=trips_df,
            zone_index=tap_index,
            model_settings=model_settings,
            is_tap=True,
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
    trips_df: pd.DataFrame,
    zone_index: pd.Index | pd.Series,
    model_settings: WriteTripMatricesSettings,
    is_tap: bool = False,
):
    """
    Write aggregated trips to OMX format per table using table-specific origin/destination
    columns and allow repeated table names to accumulate into the same matrix.
    """

    matrix_settings = model_settings.MATRICES

    if not matrix_settings:
        logger.error("Missing MATRICES setting in write_trip_matrices.yaml")
        return

    # Normalize zone index used for placement (pos_index) and mapping labels (map_labels)
    if isinstance(zone_index, pd.Series):
        pos_index = zone_index.index
        map_labels = zone_index
    else:
        pos_index = zone_index
        map_labels = zone_index

    n_zones = len(pos_index)
    pos_index = pd.Index(pos_index, name=pos_index.name or "ZONE")

    hh_weight_col = model_settings.HH_EXPANSION_WEIGHT_COL

    # For each output file, accumulate table_name -> matrix
    for matrix in matrix_settings:
        if matrix.is_tap != is_tap:
            continue

        filename = str(matrix.file_name)
        filepath = state.get_output_file_path(filename)
        logger.info(f"opening {filepath}")
        f = omx.open_file(str(filepath), "w")  # overwrite file each run

        # Accumulator for datasets in this file
        datasets: dict[str, np.ndarray] = {}

        for table in matrix.tables:
            table_name = table.name
            col = table.data_field

            if col not in trips_df.columns:
                logger.warning(
                    f"Skipping table {table_name}: missing column {col} in trips"
                )
                continue

            # Effective origin/destination columns
            ocol = table.origin or ("btap" if is_tap else "origin")
            dcol = table.destination or ("atap" if is_tap else "destination")

            if ocol not in trips_df.columns or dcol not in trips_df.columns:
                logger.warning(
                    f"Skipping table {table_name}: missing origin/destination columns "
                    f"{ocol}/{dcol} in trips"
                )
                continue

            # Build a working frame with needed columns
            work = trips_df[[ocol, dcol, col]].copy()
            if hh_weight_col and hh_weight_col in trips_df.columns:
                work[hh_weight_col] = trips_df[hh_weight_col]

            # Map to zone domain if needed (TAZ/TAP domain is pos_index)
            # if values are already in domain, keep; else try MAZ->TAZ mapping for TAZ outputs
            def to_domain_vals(series: pd.Series) -> pd.Series:
                # already in domain?
                in_domain = pd.Series(series.isin(pos_index).values, index=series.index)
                if in_domain.all() or is_tap:
                    return series
                # try MAZ -> TAZ using land_use while preserving the original index
                try:
                    lu = state.get_dataframe("land_use")
                    if "TAZ" in lu.columns:
                        mapped = series.map(lu["TAZ"])
                        # if mapping produced any non-nulls, use it
                        if mapped.notna().any():
                            return mapped
                except Exception:
                    pass
                return series  # fallback; may drop later if not in domain

            work["_o"] = to_domain_vals(work[ocol])
            work["_d"] = to_domain_vals(work[dcol])

            # Drop rows with either missing origin/destination
            work = work.dropna(subset=["_o", "_d"])

            # Aggregate by OD
            if hh_weight_col and hh_weight_col in work.columns:
                grouped_sum = work.groupby(["_o", "_d"], sort=False)[col].sum()
                mean_w = work.groupby(["_o", "_d"], sort=False)[hh_weight_col].mean()
                vals = (grouped_sum / mean_w.replace(0, np.nan)).fillna(0.0)
            else:
                vals = work.groupby(["_o", "_d"], sort=False)[col].sum()

            if vals.empty:
                continue

            # Map OD labels to positional indices
            o_vals = vals.index.get_level_values(0)
            d_vals = vals.index.get_level_values(1)

            oi = pos_index.get_indexer(o_vals)
            di = pos_index.get_indexer(d_vals)

            mask = (oi != -1) & (di != -1)
            if not np.any(mask):
                logger.warning(
                    f"No valid OD pairs for table {table_name} in domain; skipping."
                )
                continue

            oi = oi[mask]
            di = di[mask]
            v = vals.to_numpy()[mask]

            # Accumulate into dataset matrix
            data = datasets.get(table_name)
            if data is None:
                data = np.zeros((n_zones, n_zones), dtype=float)
                datasets[table_name] = data
            data[oi, di] += v

            logger.debug(f"accumulated {table_name} sum {v.sum():0.2f}")

        # Write all accumulated datasets for this file
        for table_name, data in datasets.items():
            f[table_name] = data

        # include the index-to-zone map in the file
        logger.info(
            "adding %s mapping for %s zones to %s"
            % (map_labels.name or "ZONE", len(map_labels), filename)
        )
        f.create_mapping(map_labels.name or "ZONE", pd.Index(map_labels).to_numpy())

        logger.info("closing %s" % filepath)
        f.close()

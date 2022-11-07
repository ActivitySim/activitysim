# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import openmatrix as omx
import pandas as pd

from activitysim.core import config, expressions, inject, los, pipeline

logger = logging.getLogger(__name__)


@inject.step()
def write_trip_matrices(network_los):
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

    trips = inject.get_table("trips", None)
    if trips is None:
        # this step is a NOP if there is no trips table
        # this might legitimately happen if they comment out some steps to debug but still want write_tables
        # this saves them the hassle of remembering to comment out this step
        logger.warning(
            f"write_trip_matrices returning empty-handed because there is no trips table"
        )
        return

    model_settings = config.read_model_settings("write_trip_matrices.yaml")
    trips_df = annotate_trips(trips, network_los, model_settings)

    if bool(model_settings.get("SAVE_TRIPS_TABLE")):
        pipeline.replace_table("trips", trips_df)

    if "parking_location" in config.setting("models"):
        parking_settings = config.read_model_settings("parking_location_choice.yaml")
        parking_taz_col_name = parking_settings["ALT_DEST_COL_NAME"]
        if parking_taz_col_name in trips_df:
            trips_df.loc[trips_df[parking_taz_col_name] > 0, "destination"] = trips_df[
                parking_taz_col_name
            ]
        # Also need address the return trip

    # write matrices by zone system type
    if network_los.zone_system == los.ONE_ZONE:  # taz trips written to taz matrices
        logger.info("aggregating trips one zone...")
        aggregate_trips = trips_df.groupby(["origin", "destination"], sort=False).sum()

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.get("HH_EXPANSION_WEIGHT_COL")
        aggregate_weight = (
            trips_df[["origin", "destination", hh_weight_col]]
            .groupby(["origin", "destination"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("origin")
        dest_vals = aggregate_trips.index.get_level_values("destination")

        # use the land use table for the set of possible tazs
        zone_index = pipeline.get_table("land_use").index
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        write_matrices(
            aggregate_trips, zone_index, orig_index, dest_index, model_settings
        )

    elif network_los.zone_system == los.TWO_ZONE:  # maz trips written to taz matrices
        logger.info("aggregating trips two zone...")
        trips_df["otaz"] = (
            pipeline.get_table("land_use").reindex(trips_df["origin"]).TAZ.tolist()
        )
        trips_df["dtaz"] = (
            pipeline.get_table("land_use").reindex(trips_df["destination"]).TAZ.tolist()
        )
        aggregate_trips = trips_df.groupby(["otaz", "dtaz"], sort=False).sum()

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.get("HH_EXPANSION_WEIGHT_COL")
        aggregate_weight = (
            trips_df[["otaz", "dtaz", hh_weight_col]]
            .groupby(["otaz", "dtaz"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("otaz")
        dest_vals = aggregate_trips.index.get_level_values("dtaz")

        zone_index = pd.Index(network_los.get_tazs(), name="TAZ")
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        write_matrices(
            aggregate_trips, zone_index, orig_index, dest_index, model_settings
        )

    elif (
        network_los.zone_system == los.THREE_ZONE
    ):  # maz trips written to taz and tap matrices

        logger.info("aggregating trips three zone taz...")
        trips_df["otaz"] = (
            pipeline.get_table("land_use").reindex(trips_df["origin"]).TAZ.tolist()
        )
        trips_df["dtaz"] = (
            pipeline.get_table("land_use").reindex(trips_df["destination"]).TAZ.tolist()
        )
        aggregate_trips = trips_df.groupby(["otaz", "dtaz"], sort=False).sum()

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.get("HH_EXPANSION_WEIGHT_COL")
        aggregate_weight = (
            trips_df[["otaz", "dtaz", hh_weight_col]]
            .groupby(["otaz", "dtaz"], sort=False)
            .mean()
        )
        aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

        orig_vals = aggregate_trips.index.get_level_values("otaz")
        dest_vals = aggregate_trips.index.get_level_values("dtaz")

        zone_index = pd.Index(network_los.get_tazs(), name="TAZ")
        assert all(zone in zone_index for zone in orig_vals)
        assert all(zone in zone_index for zone in dest_vals)

        _, orig_index = zone_index.reindex(orig_vals)
        _, dest_index = zone_index.reindex(dest_vals)

        write_matrices(
            aggregate_trips, zone_index, orig_index, dest_index, model_settings
        )

        logger.info("aggregating trips three zone tap...")
        aggregate_trips = trips_df.groupby(["btap", "atap"], sort=False).sum()

        # use the average household weight for all trips in the origin destination pair
        hh_weight_col = model_settings.get("HH_EXPANSION_WEIGHT_COL")
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
            aggregate_trips, zone_index, orig_index, dest_index, model_settings, True
        )


def annotate_trips(trips, network_los, model_settings):
    """
    Add columns to local trips table. The annotator has
    access to the origin/destination skims and everything
    defined in the model settings CONSTANTS.

    Pipeline tables can also be accessed by listing them under
    TABLES in the preprocessor settings.
    """

    trips_df = trips.to_frame()

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
        trips_df, locals_dict, skims, model_settings, trace_label
    )

    # Data will be expanded by an expansion weight column from
    # the households pipeline table, if specified in the model settings.
    hh_weight_col = model_settings.get("HH_EXPANSION_WEIGHT_COL")

    if hh_weight_col and hh_weight_col not in trips_df:
        logger.info("adding '%s' from households to trips table" % hh_weight_col)
        household_weights = pipeline.get_table("households")[hh_weight_col]
        trips_df[hh_weight_col] = trips_df.household_id.map(household_weights)

    return trips_df


def write_matrices(
    aggregate_trips, zone_index, orig_index, dest_index, model_settings, is_tap=False
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

    matrix_settings = model_settings.get("MATRICES")

    if not matrix_settings:
        logger.error("Missing MATRICES setting in write_trip_matrices.yaml")

    for matrix in matrix_settings:
        matrix_is_tap = matrix.get("is_tap", False)

        if matrix_is_tap == is_tap:  # only write tap matrices to tap matrix files
            filename = matrix.get("file_name")
            filepath = config.output_file_path(filename)
            logger.info("opening %s" % filepath)
            file = omx.open_file(filepath, "w")  # possibly overwrite existing file
            table_settings = matrix.get("tables")

            for table in table_settings:
                table_name = table.get("name")
                col = table.get("data_field")

                if col not in aggregate_trips:
                    logger.error(f"missing {col} column in aggregate_trips DataFrame")
                    return

                hh_weight_col = model_settings.get("HH_EXPANSION_WEIGHT_COL")
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

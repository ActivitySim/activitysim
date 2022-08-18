# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import assign, chunk, config, inject, los, mem, pipeline, tracing
from activitysim.core.pathbuilder import TransitVirtualPathBuilder

logger = logging.getLogger(__name__)


def compute_accessibilities_for_zones(
    accessibility_df,
    land_use_df,
    assignment_spec,
    constants,
    network_los,
    trace_od,
    trace_label,
):

    orig_zones = accessibility_df.index.values
    dest_zones = land_use_df.index.values

    orig_zone_count = len(orig_zones)
    dest_zone_count = len(dest_zones)

    logger.info(
        "Running %s with %d orig zones %d dest zones"
        % (trace_label, orig_zone_count, dest_zone_count)
    )

    # create OD dataframe
    od_df = pd.DataFrame(
        data={
            "orig": np.repeat(orig_zones, dest_zone_count),
            "dest": np.tile(dest_zones, orig_zone_count),
        }
    )

    if trace_od:
        trace_orig, trace_dest = trace_od
        trace_od_rows = (od_df.orig == trace_orig) & (od_df.dest == trace_dest)
    else:
        trace_od_rows = None

    # merge land_use_columns into od_df
    logger.info(f"{trace_label}: merge land_use_columns into od_df")
    od_df = pd.merge(od_df, land_use_df, left_on="dest", right_index=True).sort_index()
    chunk.log_df(trace_label, "od_df", od_df)

    locals_d = {
        "log": np.log,
        "exp": np.exp,
        "network_los": network_los,
    }
    locals_d.update(constants)

    skim_dict = network_los.get_default_skim_dict()
    locals_d["skim_od"] = skim_dict.wrap("orig", "dest").set_df(od_df)
    locals_d["skim_do"] = skim_dict.wrap("dest", "orig").set_df(od_df)

    if network_los.zone_system == los.THREE_ZONE:
        locals_d["tvpb"] = network_los.tvpb

    logger.info(f"{trace_label}: assign.assign_variables")
    results, trace_results, trace_assigned_locals = assign.assign_variables(
        assignment_spec,
        od_df,
        locals_d,
        trace_rows=trace_od_rows,
        trace_label=trace_label,
        chunk_log=True,
    )

    chunk.log_df(trace_label, "results", results)
    logger.info(f"{trace_label}: have results")

    # accessibility_df = accessibility_df.copy()
    for column in results.columns:
        data = np.asanyarray(results[column])
        data.shape = (orig_zone_count, dest_zone_count)  # (o,d)
        accessibility_df[column] = np.log(np.sum(data, axis=1) + 1)

    if trace_od:

        if not trace_od_rows.any():
            logger.warning(
                f"trace_od not found origin = {trace_orig}, dest = {trace_dest}"
            )
        else:

            # add OD columns to trace results
            df = pd.concat([od_df[trace_od_rows], trace_results], axis=1)

            # dump the trace results table (with _temp variables) to aid debugging
            tracing.trace_df(
                df,
                label="accessibility",
                index_label="skim_offset",
                slicer="NONE",
                warn_if_empty=True,
            )

            if trace_assigned_locals:
                tracing.write_csv(
                    trace_assigned_locals, file_name="accessibility_locals"
                )

    return accessibility_df


@inject.step()
def compute_accessibility(land_use, accessibility, network_los, chunk_size, trace_od):

    """
    Compute accessibility for each zone in land use file using expressions from accessibility_spec

    The actual results depend on the expressions in accessibility_spec, but this is initially
    intended to permit implementation of the mtc accessibility calculation as implemented by
    Accessibility.job

    Compute measures of accessibility used by the automobile ownership model.
    The accessibility measure first multiplies an employment variable by a mode-specific decay
    function.  The product reflects the difficulty of accessing the activities the farther
    (in terms of round-trip travel time) the jobs are from the location in question. The products
    to each destination zone are next summed over each origin zone, and the logarithm of the
    product mutes large differences.  The decay function on the walk accessibility measure is
    steeper than automobile or transit.  The minimum accessibility is zero.
    """

    trace_label = "compute_accessibility"
    model_settings = config.read_model_settings("accessibility.yaml")
    assignment_spec = assign.read_assignment_spec(
        config.config_file_path("accessibility.csv")
    )

    accessibility_df = accessibility.to_frame()
    if len(accessibility_df.columns) > 0:
        logger.warning(
            f"accessibility table is not empty. Columns:{list(accessibility_df.columns)}"
        )
        raise RuntimeError(f"accessibility table is not empty.")

    constants = config.get_model_constants(model_settings)

    # only include the land_use columns needed by spec, as specified by land_use_columns model_setting
    land_use_columns = model_settings.get("land_use_columns", [])
    land_use_df = land_use.to_frame()
    land_use_df = land_use_df[land_use_columns]

    logger.info(
        f"Running {trace_label} with {len(accessibility_df.index)} orig zones {len(land_use_df)} dest zones"
    )

    accessibilities_list = []

    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        accessibility_df, chunk_size, trace_label
    ):

        accessibilities = compute_accessibilities_for_zones(
            chooser_chunk,
            land_use_df,
            assignment_spec,
            constants,
            network_los,
            trace_od,
            trace_label,
        )
        accessibilities_list.append(accessibilities)

    accessibility_df = pd.concat(accessibilities_list)

    logger.info(f"{trace_label} computed accessibilities {accessibility_df.shape}")

    # - write table to pipeline
    pipeline.replace_table("accessibility", accessibility_df)

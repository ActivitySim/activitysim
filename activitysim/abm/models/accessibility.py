# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from typing import Any

import numba as nb
import numpy as np
import pandas as pd

from activitysim.core import assign, chunk, los, workflow
from activitysim.core.configuration.base import PydanticReadable

logger = logging.getLogger(__name__)


class AccessibilitySettings(PydanticReadable):
    """
    Settings for aggregate accessibility component.
    """

    CONSTANTS: dict[str, Any] = {}

    land_use_columns: list[str] = []
    """Only include the these columns in the computational tables.

    This setting joins land use columns to the accessibility destinations.

    Memory usage is reduced by only listing the minimum columns needed by
    the SPEC, and nothing extra.
    """

    land_use_columns_orig: list[str] = []
    """Join these land use columns to the origin zones.

    This setting joins land use columns to the accessibility origins.
    To disambiguate from the destination land use columns, the names of the
    columns added will be prepended with 'landuse_orig_'.

    Memory usage is reduced by only listing the minimum columns needed by
    the SPEC, and nothing extra.

    .. versionadded:: 1.3
    """

    SPEC: str = "accessibility.csv"
    """Filename for the accessibility specification (csv) file."""

    explicit_chunk: float = 0
    """
    If > 0, use this chunk size instead of adaptive chunking.
    If less than 1, use this fraction of the total number of rows.
    """


@nb.njit
def _accumulate_accessibility(arr, orig_zone_count, dest_zone_count):
    assert arr.size == orig_zone_count * dest_zone_count
    assert arr.ndim == 1
    i = 0
    result = np.empty((orig_zone_count,), dtype=arr.dtype)
    for o in range(orig_zone_count):
        x = 0
        for d in range(dest_zone_count):
            x += arr[i]
            i += 1
        result[o] = np.log1p(x)
    return result


def compute_accessibilities_for_zones(
    state: workflow.State,
    accessibility_df: pd.DataFrame,
    land_use_df: pd.DataFrame,
    orig_land_use_df: pd.DataFrame | None,
    assignment_spec: dict,
    constants: dict,
    network_los: los.Network_LOS,
    trace_label: str,
    chunk_sizer: chunk.ChunkSizer,
):
    """
    Compute accessibility for each zone in land use file using expressions from accessibility_spec.

    Parameters
    ----------
    state : workflow.State
    accessibility_df : pd.DataFrame
    land_use_df : pd.DataFrame
    orig_land_use_df : pd.DataFrame | None
    assignment_spec : dict
    constants : dict
    network_los : los.Network_LOS
    trace_label : str
    chunk_sizer : chunk.ChunkSizer

    Returns
    -------
    accessibility_df : pd.DataFrame
        The accessibility_df is updated in place.
    """
    orig_zones = accessibility_df.index.values
    dest_zones = land_use_df.index.values

    orig_zone_count = len(orig_zones)
    dest_zone_count = len(dest_zones)

    logger.info(
        "Running %s with %d orig zones %d dest zones"
        % (trace_label, orig_zone_count, dest_zone_count)
    )

    # create OD dataframe
    od_data = {
        "orig": np.repeat(orig_zones, dest_zone_count),
        "dest": np.tile(dest_zones, orig_zone_count),
    }
    # previously, the land use was added to the dataframe via pd.merge
    # but the merge is expensive and unnecessary as we can just tile.
    logger.debug(f"{trace_label}: tiling land_use_columns into od_data")
    for c in land_use_df.columns:
        od_data[c] = np.tile(land_use_df[c].to_numpy(), orig_zone_count)
    if orig_land_use_df is not None:
        logger.debug(f"{trace_label}: repeating orig_land_use_columns into od_data")
        for c in orig_land_use_df:
            od_data[f"landuse_orig_{c}"] = np.repeat(
                orig_land_use_df[c], dest_zone_count
            )
    logger.debug(f"{trace_label}: converting od_data to DataFrame")
    od_df = pd.DataFrame(od_data)
    logger.debug(f"{trace_label}: dropping od_data")
    del od_data
    logger.debug(f"{trace_label}: dropping od_data complete")

    trace_od = state.settings.trace_od
    if trace_od:
        trace_orig, trace_dest = trace_od
        trace_od_rows = (od_df.orig == trace_orig) & (od_df.dest == trace_dest)
    else:
        trace_od_rows = None

    chunk_sizer.log_df(trace_label, "od_df", od_df)

    locals_d = {
        "log": np.log,
        "exp": np.exp,
        "network_los": network_los,
    }
    locals_d.update(constants)

    skim_dict = network_los.get_default_skim_dict()
    # FIXME: because od_df is so huge, next two lines use a fair bit of memory
    locals_d["skim_od"] = skim_dict.wrap("orig", "dest").set_df(od_df)
    locals_d["skim_do"] = skim_dict.wrap("dest", "orig").set_df(od_df)

    if network_los.zone_system == los.THREE_ZONE:
        locals_d["tvpb"] = network_los.tvpb

    logger.info(f"{trace_label}: assign.assign_variables")
    results, trace_results, trace_assigned_locals = assign.assign_variables(
        state,
        assignment_spec,
        od_df,
        locals_d,
        trace_rows=trace_od_rows,
        trace_label=trace_label,
        chunk_log=chunk_sizer,
    )

    chunk_sizer.log_df(trace_label, "results", results)
    logger.info(f"{trace_label}: have results")

    # accessibility_df = accessibility_df.copy()
    accessibility_new_columns = {}
    for column in results.columns:
        logger.info(f"{trace_label}: aggregating column {column}")
        accessibility_new_columns[column] = _accumulate_accessibility(
            results[column].to_numpy(), orig_zone_count, dest_zone_count
        )
    logger.info(f"{trace_label}: completed aggregating")
    accessibility_df = accessibility_df.assign(**accessibility_new_columns)
    logger.info(f"{trace_label}: completed aggregating info df")

    if trace_od:
        if not trace_od_rows.any():
            logger.warning(
                f"trace_od not found origin = {trace_orig}, dest = {trace_dest}"
            )
        else:
            # add OD columns to trace results
            df = pd.concat([od_df[trace_od_rows], trace_results], axis=1)

            # dump the trace results table (with _temp variables) to aid debugging
            state.tracing.trace_df(
                df,
                label="accessibility",
                index_label="skim_offset",
                slicer="NONE",
                warn_if_empty=True,
            )

            if trace_assigned_locals:
                state.tracing.write_csv(
                    trace_assigned_locals, file_name="accessibility_locals"
                )

    return accessibility_df


@workflow.step
def compute_accessibility(
    state: workflow.State,
    land_use: pd.DataFrame,
    accessibility: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: AccessibilitySettings | None = None,
    model_settings_file_name: str = "accessibility.yaml",
    trace_label: str = "compute_accessibility",
    output_table_name: str = "accessibility",
) -> None:
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
    if model_settings is None:
        model_settings = AccessibilitySettings.read_settings_file(
            state.filesystem, model_settings_file_name
        )

    assignment_spec = assign.read_assignment_spec(
        state.filesystem.get_config_file_path(model_settings.SPEC)
    )

    accessibility_df = accessibility
    if len(accessibility_df.columns) > 0:
        logger.warning(
            f"accessibility table is not empty. "
            f"Columns:{list(accessibility_df.columns)}"
        )
        raise RuntimeError("accessibility table is not empty.")

    constants = model_settings.CONSTANTS

    # only include the land_use columns needed by spec,
    # as specified by land_use_columns model_setting
    land_use_columns = model_settings.land_use_columns
    land_use_df = land_use
    land_use_df = land_use_df[land_use_columns]

    if model_settings.land_use_columns_orig:
        orig_land_use_df = land_use[model_settings.land_use_columns_orig]
    else:
        orig_land_use_df = None

    logger.info(
        f"Running {trace_label} with {len(accessibility_df.index)} orig zones "
        f"{len(land_use_df)} dest zones"
    )

    accessibilities_list = []
    explicit_chunk_size = model_settings.explicit_chunk

    for (
        _i,
        chooser_chunk,
        _chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers(
        state, accessibility_df, trace_label, explicit_chunk_size=explicit_chunk_size
    ):
        if orig_land_use_df is not None:
            orig_land_use_df_chunk = orig_land_use_df.loc[chooser_chunk.index]
        else:
            orig_land_use_df_chunk = None
        accessibilities = compute_accessibilities_for_zones(
            state,
            chooser_chunk,
            land_use_df,
            orig_land_use_df_chunk,
            assignment_spec,
            constants,
            network_los,
            trace_label,
            chunk_sizer,
        )
        accessibilities_list.append(accessibilities)

    accessibility_df = pd.concat(accessibilities_list)

    logger.info(f"{trace_label} computed accessibilities {accessibility_df.shape}")

    # - write table to pipeline
    state.add_table(output_table_name, accessibility_df)

# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
import numpy as np

from activitysim.core import assign
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import mem

from activitysim.core import los
from activitysim.core.pathbuilder import TransitVirtualPathBuilder

logger = logging.getLogger(__name__)


@inject.step()
def compute_accessibility(land_use, accessibility, network_los, trace_od):

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

    trace_label = 'compute_accessibility'
    model_settings = config.read_model_settings('accessibility.yaml')
    assignment_spec = assign.read_assignment_spec(config.config_file_path('accessibility.csv'))

    accessibility_df = accessibility.to_frame()
    if len(accessibility_df.columns) > 0:
        logger.warning(f"accessibility table is not empty. Columns:{list(accessibility_df.columns)}")
        raise RuntimeError(f"accessibility table is not empty.")

    constants = config.get_model_constants(model_settings)

    land_use_columns = model_settings.get('land_use_columns', [])
    land_use_df = land_use.to_frame()
    land_use_df = land_use_df[land_use_columns]

    # don't assume they are the same: accessibility may be sliced if we are multiprocessing
    orig_zones = accessibility_df.index.values
    dest_zones = land_use_df.index.values

    orig_zone_count = len(orig_zones)
    dest_zone_count = len(dest_zones)

    logger.info("Running %s with %d orig zones %d dest zones" %
                (trace_label, orig_zone_count, dest_zone_count))

    # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'orig': np.repeat(orig_zones, dest_zone_count),
            'dest': np.tile(dest_zones, orig_zone_count)
        }
    )

    if trace_od:
        trace_orig, trace_dest = trace_od
        trace_od_rows = (od_df.orig == trace_orig) & (od_df.dest == trace_dest)
    else:
        trace_od_rows = None

    # merge land_use_columns into od_df
    od_df = pd.merge(od_df, land_use_df, left_on='dest', right_index=True).sort_index()

    locals_d = {
        'log': np.log,
        'exp': np.exp,
        'network_los': network_los,
    }

    skim_dict = network_los.get_default_skim_dict()
    locals_d['skim_od'] = skim_dict.wrap('orig', 'dest').set_df(od_df)
    locals_d['skim_do'] = skim_dict.wrap('dest', 'orig').set_df(od_df)

    if network_los.zone_system == los.THREE_ZONE:
        locals_d['tvpb'] = TransitVirtualPathBuilder(network_los)

    if constants is not None:
        locals_d.update(constants)

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(assignment_spec, od_df, locals_d, trace_rows=trace_od_rows, trace_label=trace_label)

    for column in results.columns:
        data = np.asanyarray(results[column])
        data.shape = (orig_zone_count, dest_zone_count)  # (o,d)
        accessibility_df[column] = np.log(np.sum(data, axis=1) + 1)

    logger.info("{trace_label} added {len(results.columns} columns")

    # - write table to pipeline
    pipeline.replace_table("accessibility", accessibility_df)

    if trace_od:

        if not trace_od_rows.any():
            logger.warning(f"trace_od not found origin = {trace_orig}, dest = {trace_dest}")
        else:

            # add OD columns to trace results
            df = pd.concat([od_df[trace_od_rows], trace_results], axis=1)

            # dump the trace results table (with _temp variables) to aid debugging
            tracing.trace_df(df,
                             label='accessibility',
                             index_label='skim_offset',
                             slicer='NONE',
                             warn_if_empty=True)

            if trace_assigned_locals:
                tracing.write_csv(trace_assigned_locals, file_name="accessibility_locals")

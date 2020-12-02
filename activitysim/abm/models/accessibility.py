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


# class AccessibilitySkims(object):
#     """
#     Wrapper for skim arrays to facilitate use of skims by accessibility model
#
#     Parameters
#     ----------
#     skims : 2D array
#     omx: open omx file object
#         this is only used to load skims on demand that were not preloaded
#     length: int
#         number of zones in skim to return in skim matrix
#         in case the skims contain additional external zones that should be trimmed out so skim
#         array is correct shape to match (flattened) O-D tiled columns in the od dataframe
#     transpose: bool
#         whether to transpose the matrix before flattening. (i.e. act as a D-O instead of O-D skim)
#     """
#
#     def __init__(self, skim_dict, orig_zones, dest_zones, transpose=False):
#
#         logger.info(f"init AccessibilitySkims with {len(dest_zones)} dest zones {len(orig_zones)} orig zones")
#
#         assert len(orig_zones) <= len(dest_zones)
#         assert np.isin(orig_zones, dest_zones).all()
#         assert len(np.unique(orig_zones)) == len(orig_zones)
#         assert len(np.unique(dest_zones)) == len(dest_zones)
#
#         self.skim_dict = skim_dict
#         self.transpose = transpose
#
#         num_skim_zones = skim_dict.get_skim_info('omx_shape')[0]
#         if num_skim_zones == len(orig_zones) and skim_dict.offset_mapper.offset_series is None:
#             # no slicing required because whatever the offset_int, the skim data aligns with zone list
#             self.map_data = False
#         else:
#
#             logger.debug("AccessibilitySkims - applying offset_mapper")
#
#             skim_index = list(range(num_skim_zones))
#             orig_map = skim_dict.offset_mapper.map(orig_zones)
#             dest_map = skim_dict.offset_mapper.map(dest_zones)
#
#             # (we might be sliced multiprocessing)
#             # assert np.isin(skim_index, orig_map).all()
#
#             out_of_bounds = ~np.isin(skim_index, dest_map)
#             # if out_of_bounds.any():
#             #    print(f"{(out_of_bounds).sum()} skim zones not in dest_map")
#             #    print(f"dest_zones {dest_zones}")
#             #    print(f"dest_map {dest_map}")
#             #    print(f"skim_index {skim_index}")
#             assert not out_of_bounds.any(), \
#                 f"AccessibilitySkims {(out_of_bounds).sum()} skim zones not in dest_map: {np.ix_(out_of_bounds)[0]}"
#
#             self.map_data = True
#             self.orig_map = orig_map
#             self.dest_map = dest_map
#
#     def __getitem__(self, key):
#         """
#         accessor to return flattened skim array with specified key
#         flattened array will have length length*length and will match tiled OD df used by assign
#
#         this allows the skim array to be accessed from expressions as
#         skim['DISTANCE'] or skim[('SOVTOLL_TIME', 'MD')]
#         """
#
#         data = self.skim_dict.get(key).data
#
#         if self.transpose:
#             data = data.transpose()
#
#         if self.map_data:
#             # slice skim to include only orig rows and dest columns
#             # 2-d boolean slicing in numpy is a bit tricky
#             # data = data[orig_map, dest_map]          # <- WRONG!
#             # data = data[orig_map, :][:, dest_map]    # <- RIGHT
#             # data = data[np.ix_(orig_map, dest_map)]  # <- ALSO RIGHT
#
#             data = data[self.orig_map, :][:, self.dest_map]
#
#         return data.flatten()


@inject.step()
def compute_accessibility(accessibility, network_los, land_use, trace_od):

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

    logger.info("Running %s with %d dest zones" % (trace_label, len(accessibility_df)))

    constants = config.get_model_constants(model_settings)

    land_use_columns = model_settings.get('land_use_columns', [])
    land_use_df = land_use.to_frame()
    land_use_df = land_use_df[land_use_columns]

    # don't assume they are the same: accessibility may be sliced if we are multiprocessing
    orig_zones = accessibility_df.index.values
    dest_zones = land_use_df.index.values

    orig_zone_count = len(orig_zones)
    dest_zone_count = len(dest_zones)

    logger.info("Running %s with %d dest zones %d orig zones" %
                (trace_label, dest_zone_count, orig_zone_count))

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
        = assign.assign_variables(assignment_spec, od_df, locals_d, trace_rows=trace_od_rows)

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

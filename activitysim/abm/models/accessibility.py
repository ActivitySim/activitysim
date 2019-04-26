# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import pandas as pd
import numpy as np

from activitysim.core import assign
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline


logger = logging.getLogger(__name__)


class AccessibilitySkims(object):
    """
    Wrapper for skim arrays to facilitate use of skims by accessibility model

    Parameters
    ----------
    skims : 2D array
    omx: open omx file object
        this is only used to load skims on demand that were not preloaded
    length: int
        number of zones in skim to return in skim matrix
        in case the skims contain additional external zones that should be trimmed out so skim
        array is correct shape to match (flattened) O-D tiled columns in the od dataframe
    transpose: bool
        whether to transpose the matrix before flattening. (i.e. act as a D-O instead of O-D skim)
    """

    def __init__(self, skim_dict, orig_zones, dest_zones, transpose=False):

        omx_shape = skim_dict.skim_info['omx_shape']
        logger.info("init AccessibilitySkims with %d dest zones %d orig zones omx_shape %s" %
                    (len(dest_zones), len(orig_zones), omx_shape, ))

        assert len(orig_zones) <= len(dest_zones)
        assert np.isin(orig_zones, dest_zones).all()
        assert len(np.unique(orig_zones)) == len(orig_zones)
        assert len(np.unique(dest_zones)) == len(dest_zones)

        self.skim_dict = skim_dict
        self.transpose = transpose

        if omx_shape[0] == len(orig_zones):
            # no slicing required
            self.slice_map = None
        else:
            # 2-d boolean slicing in numpy is a bit tricky
            # data = data[orig_map, dest_map]          # <- WRONG!
            # data = data[orig_map, :][:, dest_map]    # <- RIGHT
            # data = data[np.ix_(orig_map, dest_map)]  # <- ALSO RIGHT

            skim_index = list(range(omx_shape[0]))
            orig_map = np.isin(skim_index, skim_dict.offset_mapper.map(orig_zones))
            dest_map = np.isin(skim_index, skim_dict.offset_mapper.map(dest_zones))

            if not dest_map.all():
                # not using the whole skim matrix
                logger.info("%s skim zones not in dest_map: %s" %
                            ((~dest_map).sum(), np.ix_(~dest_map)))

            self.slice_map = np.ix_(orig_map, dest_map)

    def __getitem__(self, key):
        """
        accessor to return flattened skim array with specified key
        flattened array will have length length*length and will match tiled OD df used by assign

        this allows the skim array to be accessed from expressions as
        skim['DISTANCE'] or skim[('SOVTOLL_TIME', 'MD')]
        """

        data = self.skim_dict.get(key).data

        if self.transpose:
            data = data.transpose()

        if self.slice_map is not None:
            # slice skim to include only orig rows and dest columns
            # 2-d boolean slicing in numpy is a bit tricky - see explanation in __init__
            data = data[self.slice_map]

        return data.flatten()


@inject.step()
def compute_accessibility(accessibility, skim_dict, land_use, trace_od):

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

    # #bug
    #
    # land_use_df = land_use_df[land_use_df.index % 2 == 1]
    # accessibility_df = accessibility_df[accessibility_df.index.isin(land_use_df.index)].head(5)
    #
    # print "land_use_df", land_use_df.index
    # print "accessibility_df", accessibility_df.index
    # #bug

    orig_zones = accessibility_df.index.values
    dest_zones = land_use_df.index.values

    orig_zone_count = len(orig_zones)
    dest_zone_count = len(dest_zones)

    logger.info("Running %s with %d dest zones %d orig zones" %
                (trace_label, dest_zone_count, orig_zone_count))

    # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'orig': np.repeat(np.asanyarray(accessibility_df.index), dest_zone_count),
            'dest': np.tile(np.asanyarray(land_use_df.index), orig_zone_count)
        }
    )

    if trace_od:
        trace_orig, trace_dest = trace_od
        trace_od_rows = (od_df.orig == trace_orig) & (od_df.dest == trace_dest)
    else:
        trace_od_rows = None

    # merge land_use_columns into od_df
    land_use_df = land_use_df[land_use_columns]
    od_df = pd.merge(od_df, land_use_df, left_on='dest', right_index=True).sort_index()

    locals_d = {
        'log': np.log,
        'exp': np.exp,
        'skim_od': AccessibilitySkims(skim_dict, orig_zones, dest_zones),
        'skim_do': AccessibilitySkims(skim_dict, orig_zones, dest_zones, transpose=True)
    }
    if constants is not None:
        locals_d.update(constants)

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(assignment_spec, od_df, locals_d, trace_rows=trace_od_rows)

    for column in results.columns:
        data = np.asanyarray(results[column])
        data.shape = (orig_zone_count, dest_zone_count)
        accessibility_df[column] = np.log(np.sum(data, axis=1) + 1)

    # - write table to pipeline
    pipeline.replace_table("accessibility", accessibility_df)

    if trace_od:

        if not trace_od_rows.any():
            logger.warning("trace_od not found origin = %s, dest = %s" % (trace_orig, trace_dest))
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

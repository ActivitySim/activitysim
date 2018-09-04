# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

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

    def __init__(self, skim_dict, dest_zones, orig_zones, transpose=False):

        assert skim_dict.skim_data.shape[0] == len(dest_zones)
        assert len(orig_zones) <= len(dest_zones)
        assert np.isin(orig_zones, dest_zones).all()
        assert len(np.unique(orig_zones)) == len(orig_zones)
        assert len(np.unique(dest_zones)) == len(dest_zones)

        self.skim_dict = skim_dict
        self.transpose = transpose
        self.orig_map = None

        if len(orig_zones) < len(dest_zones):
            self.orig_map = np.isin(dest_zones, orig_zones)
        else:
            self.orig_map = None

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

        if self.orig_map is not None:
            data = data[self.orig_map, :]

        return data.flatten()


@inject.injectable()
def accessibility_spec(configs_dir):
    f = os.path.join(configs_dir, 'accessibility.csv')
    return assign.read_assignment_spec(f)


@inject.step()
def compute_accessibility(accessibility, accessibility_spec,
                          skim_dict, land_use, trace_od):

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

    logger.info("Running compute_accessibility")
    model_settings = config.read_model_settings('accessibility.yaml')

    accessibility_df = accessibility.to_frame()

    constants = config.get_model_constants(model_settings)
    land_use_columns = model_settings.get('land_use_columns', [])

    land_use_df = land_use.to_frame()

    orig_zones = accessibility_df.index.values
    dest_zones = land_use_df.index.values

    orig_zone_count = len(orig_zones)
    dest_zone_count = len(dest_zones)

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
        'skim_od': AccessibilitySkims(skim_dict, dest_zones, orig_zones),
        'skim_do': AccessibilitySkims(skim_dict, dest_zones, orig_zones, transpose=True)
    }
    if constants is not None:
        locals_d.update(constants)

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(accessibility_spec, od_df, locals_d, trace_rows=trace_od_rows)

    for column in results.columns:
        data = np.asanyarray(results[column])
        data.shape = (orig_zone_count, dest_zone_count)
        accessibility_df[column] = np.log(np.sum(data, axis=1) + 1)

    # - write table to pipeline
    pipeline.replace_table("accessibility", accessibility_df)

    if trace_od:

        if not trace_od_rows.any():
            logger.warn("trace_od not found origin = %s, dest = %s" % (trace_orig, trace_dest))
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

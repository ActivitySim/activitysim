# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim.core import assign
from activitysim.core import tracing
from activitysim.core import config


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

    def __init__(self, skim_dict, omx, length, transpose=False):
        self.skim_dict = skim_dict
        self.omx = omx
        self.length = length
        self.transpose = transpose

    def __getitem__(self, key):
        """
        accessor to return flattened skim array with specified key
        flattened array will have length length*length and will match tiled OD df used by assign

        this allows the skim array to be accessed from expressions as
        skim['DISTANCE'] or skim[('SOVTOLL_TIME', 'MD')]
        """
        try:
            data = self.skim_dict.get(key).data
        except KeyError:
            omx_key = '__'.join(key)
            logger.info("AccessibilitySkims loading %s from omx as %s" % (key, omx_key,))
            data = self.omx[omx_key]

        data = data[:self.length, :self.length]

        if self.transpose:
            return data.transpose().flatten()
        else:
            return data.flatten()


@orca.injectable()
def accessibility_spec(configs_dir):
    f = os.path.join(configs_dir, 'accessibility.csv')
    return assign.read_assignment_spec(f)


@orca.injectable()
def accessibility_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'accessibility.yaml')


@orca.step()
def compute_accessibility(settings, accessibility_spec,
                          accessibility_settings,
                          skim_dict, omx_file, land_use, trace_od):

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

    constants = config.get_model_constants(accessibility_settings)
    land_use_columns = accessibility_settings.get('land_use_columns', [])

    land_use_df = land_use.to_frame()

    zone_count = len(land_use_df.index)

    # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'orig': np.repeat(np.asanyarray(land_use_df.index), zone_count),
            'dest': np.tile(np.asanyarray(land_use_df.index), zone_count)
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
        'skim_od': AccessibilitySkims(skim_dict, omx_file, zone_count),
        'skim_do': AccessibilitySkims(skim_dict, omx_file, zone_count, transpose=True)
    }
    if constants is not None:
        locals_d.update(constants)

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(accessibility_spec, od_df, locals_d, trace_rows=trace_od_rows)
    accessibility_df = pd.DataFrame(index=land_use.index)
    for column in results.columns:
        data = np.asanyarray(results[column])
        data.shape = (zone_count, zone_count)
        accessibility_df[column] = np.log(np.sum(data, axis=1) + 1)

        orca.add_column("accessibility", column, accessibility_df[column])

    if trace_od:

        if not trace_od_rows.any():
            logger.warn("trace_od not found origin = %s, dest = %s" % (trace_orig, trace_dest))
        else:

            # add OD columns to trace results
            df = pd.concat([od_df[trace_od_rows], trace_results], axis=1)

            # dump the trace results table (with _temp variables) to aid debugging
            # note that this is not the same as the orca-injected accessibility table
            # FIXME - should we name this differently and also dump the updated accessibility table?
            tracing.trace_df(df,
                             label='accessibility',
                             index_label='skim_offset',
                             slicer='NONE',
                             warn_if_empty=True)

            if trace_assigned_locals:
                tracing.write_locals(df, file_name="accessibility_locals")

        tracing.trace_df(orca.get_table('persons_merged').to_frame(), "persons_merged",
                         warn_if_empty=True)

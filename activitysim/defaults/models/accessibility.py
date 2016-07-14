# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim import asim_eval as asim_eval
from activitysim import tracing


class AccessibilitySkims(object):

    def __init__(self, skims, omx, length, transpose=False):
        self.skims = skims
        self.omx = omx
        self.length = length
        self.transpose = transpose

    def __getitem__(self, key):

        try:
            data = self.skims.get_skim(key).data
        except KeyError:
            omx_key = '__'.join(key)
            tracing.info(__name__,
                         message="AccessibilitySkims loading %s from omx as %s" % (key, omx_key,))
            data = self.omx[omx_key]

        data = data[:self.length, :self.length]

        if self.transpose:
            return data.transpose().flatten()
        else:
            return data.flatten()

    def get_from_omx(self, key, v):
        # treat this as a callback - override depending on how you store skims in the omx file
        #
        # from activitysim import skim as askim
        # from types import MethodType
        # askim.Skims3D.get_from_omx = MethodType(get_from_omx, None, askim.Skims3D)

        omx_key = key + '__' + v
        # print "my_get_from_omx - key: '%s' v: '%s', omx_key: '%s'" % (key, v, omx_key)
        return self.omx[omx_key]


@orca.injectable()
def accessibility_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "accessibility.csv")
    return asim_eval.read_assignment_spec(f)


@orca.step()
def compute_accessibility(settings, accessibility_spec, skims, omx_file, land_use, trace_od):

    tracing.info(__name__,
                 "Running compute_accessibility")

    settings_locals = settings.get('locals_accessibility', None)

    if not settings_locals:
        tracing.error(__name__, "no locals_accessibility settings")
        return

    land_use_df = land_use.to_frame()

    zone_count = len(land_use_df.index)

    # create OD dataframe
    od_df = pd.DataFrame(
        data={
            'orig': np.repeat(np.asanyarray(land_use_df.index), zone_count),
            'dest': np.tile(np.asanyarray(land_use_df.index), zone_count)
        }
    )

    # merge land_use_columns into od_df
    land_use_columns = settings_locals.get('land_use_columns', [])
    land_use_df = land_use_df[land_use_columns]
    od_df = pd.merge(od_df, land_use_df, left_on='dest', right_index=True).sort_index()

    locals_d = asim_eval.assign_variables_locals(settings_locals)
    locals_d['skim'] = AccessibilitySkims(skims, omx_file, zone_count)
    locals_d['skim_t'] = AccessibilitySkims(skims, omx_file, zone_count, transpose=True)

    result = asim_eval.assign_variables(accessibility_spec, od_df, locals_d)

    accessibility_df = pd.DataFrame(index=land_use.index)
    for column in result.columns:
        data = np.asanyarray(result[column])
        data.shape = (zone_count, zone_count)
        accessibility_df[column] = np.log(np.sum(data, axis=1) + 1)

        orca.add_column("accessibility", column, accessibility_df[column])

    if trace_od:

        tracing.info(__name__,
                     "trace origin = %s, dest = %s" % (trace_od[0], trace_od[1]))

        for key, value in settings_locals.iteritems():
            tracing.info(__name__,
                         message="SETTING: %s = %s" % (key, value))

        o, d = trace_od
        df = pd.concat([od_df, result], axis=1)[(od_df.orig == o) & (od_df.dest == d)]
        for column in df.columns:
            tracing.info(__name__,
                         message="RESULT: %s = %s" % (column, df[column].iloc[0]))

        tracing.trace_df(df,
                         label='accessibility.result',
                         index_label='skim_offset',
                         slicer='NONE')

        tracing.trace_df(orca.get_table('accessibility').to_frame(),
                         label="accessibility",
                         column_labels=['label', 'orig_taz', 'dest_taz'])

        tracing.trace_df(orca.get_table('accessibility').to_frame(), "accessibility.full",
                         slicer='NONE', transpose=False)

        tracing.trace_df(orca.get_table('persons_merged').to_frame(), "persons_merged")

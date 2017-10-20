# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import assign
from activitysim.core import inject

from activitysim.core.util import assign_in_place
from activitysim.core import util


def local_utilities():

    utility_dict = {
        'pd': pd,
        'np': np,
        'reindex': util.reindex,
        'setting': config.setting,
        'skim_time_period_label': skim_time_period_label
    }

    return utility_dict


def read_model_spec(configs_dir, model_spec_name):

    model_spec_file_name = os.path.join(configs_dir, model_spec_name)
    return assign.read_assignment_spec(model_spec_file_name)


def compute_columns(df, model_settings, configs_dir, trace_label=None):

    if isinstance(model_settings, str):
        model_settings_name = model_settings
        model_settings = config.read_model_settings(configs_dir, '%s.yaml' % model_settings)
        assert model_settings, "Found no model settings for %s" % model_settings_name
    else:
        model_settings_name = 'dict'

    assert 'DF' in model_settings, \
        "Expected to find 'DF' in %s" % model_settings_name

    df_name = model_settings.get('DF')
    helper_table_names = model_settings.get('TABLES', [])
    expressions_spec_name = model_settings.get('SPEC', model_settings_name)

    assert expressions_spec_name is not None, \
        "Expected to find 'SPEC' in %s" % model_settings_name

    if trace_label is None:
        trace_label = expressions_spec_name

    if not expressions_spec_name.endswith(".csv"):
        expressions_spec_name = '%s.csv' % expressions_spec_name

    expressions_spec = read_model_spec(configs_dir, expressions_spec_name)

    tables = {t: inject.get_table(t).to_frame() for t in helper_table_names}

    # if df was passed in, df might be a slice, or any other table, but DF is it's local alias
    assert df_name not in tables, "Did not expect to find df '%s' in TABLES" % df_name
    tables[df_name] = df

    locals_dict = local_utilities()
    locals_dict.update(tables)

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(expressions_spec,
                                  df,
                                  locals_dict,
                                  trace_rows=tracing.trace_targets(df))

    if trace_results is not None:
        tracing.trace_df(trace_results,
                         label=trace_label,
                         slicer='NONE',
                         warn_if_empty=True)

    if trace_assigned_locals:
        tracing.write_csv(trace_assigned_locals, file_name="%s_locals" % trace_label)

    return results


def assign_columns(df, model_settings, configs_dir=None, trace_label=None):

    results = compute_columns(df, model_settings, configs_dir, trace_label)
    assign_in_place(df, results)


# ##################################################################################################
# helpers
# ##################################################################################################

def skim_time_period_label(time):

    skim_time_periods = config.setting('skim_time_periods')

    # FIXME - eventually test and use np version always?
    if np.isscalar(time):
        bin = np.digitize([time % 24], skim_time_periods['hours'])[0] - 1
        return skim_time_periods['labels'][bin]

    return pd.cut(time, skim_time_periods['hours'], labels=skim_time_periods['labels']).astype(str)

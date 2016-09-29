# ActivitySim
# See full license in LICENSE.txt.

import os

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import tracing

from .util.misc import read_model_settings, get_model_constants
from activitysim.cdap import xdap


@orca.injectable()
def cdap_settings(configs_dir):
    return read_model_settings(configs_dir, 'cdap.yaml')


@orca.injectable()
def cdap_indiv_spec(configs_dir):
    f = os.path.join(configs_dir, 'cdap_indiv_and_hhsize1.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_interaction_coefficients(configs_dir):

    activity_column_name = 'activity'
    ptypes_column_name = 'interaction_ptypes'
    coefficient_column_name = 'coefficient'

    f = os.path.join(configs_dir, 'cdap_interaction_coefficients.csv')

    coefficients = pd.read_csv(f, comment='#')

    coefficients['cardinality'] = coefficients[ptypes_column_name].astype(str).str.len()

    coefficients['slug'] = \
        coefficients[activity_column_name] * coefficients['cardinality'] \
        + coefficients[ptypes_column_name].astype(str)

    return coefficients


@orca.step()
def xdap_simulate(households, persons_merged,
                  cdap_settings,
                  cdap_indiv_spec,
                  cdap_interaction_coefficients,
                  hh_chunk_size, trace_hh_id):
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """

    households_df = households.to_frame()
    persons_df = persons_merged.to_frame()

    constants = get_model_constants(cdap_settings)

    tracing.info(__name__,
                 "Running xdap_simulate with %d households and %d persons"
                 % (len(households_df.index), len(persons_df.index)))

    choices = xdap.run_cdap(households=households_df,
                            persons=persons_df,
                            cdap_indiv_spec=cdap_indiv_spec,
                            cdap_interaction_coefficients=cdap_interaction_coefficients,
                            locals_d=constants,
                            chunk_size=hh_chunk_size,
                            trace_hh_id=trace_hh_id,
                            trace_label='xdap')

    choices = choices.reindex(persons_merged.index)

    tracing.print_summary('xdap_activity', choices, value_counts=True)

    orca.add_column("persons", "xdap_activity", choices)

    if trace_hh_id:
        trace_columns = ['xdap_activity']
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="xdap",
                         columns=trace_columns,
                         warn_if_empty=True)

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
    """
    canonical model settings file to permit definition of local constants for by
    cdap_indiv_spec and cdap_fixed_relative_proportions
    """

    return read_model_settings(configs_dir, 'cdap.yaml')


@orca.injectable()
def cdap_indiv_spec(configs_dir):
    """
    spec to compute the activity utilities for each individual hh member
    with no interactions with other household members taken into account
    """

    f = os.path.join(configs_dir, 'cdap_indiv_and_hhsize1.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_interaction_coefficients(configs_dir):
    """
    The input cdap_interaction_coefficients.csv file has three columns:

    activity
        A single character activity type name (M, N, or H)

    interaction_ptypes
        List of ptypes in the interaction (in order of increasing ptype)
        Stars (***) instead of ptypes means the interaction applies to all ptypes in that size hh

    coefficient
        The coefficient to apply for all hh interactions for this activity and set of ptypes

    --------------------------------------------------------
    cdap_interaction_coefficients.csv
    --------------------------------------------------------
    activity,interaction_ptypes,coefficient
    # 2-way interactions,,
    H,11,1.626
    H,12,0.7407
    [...]
    # 3-way interactions,,
    H,124,0.9573
    H,122,0.9573
    # cdap_final_rules,,
    M,5,-999
    [...]
    # cdap_all_people,,
    M,***,-0.0671
    N,***,-0.3653
    [...]
    --------------------------------------------------------

    To facilitate building the spec for a given hh ssize, we add two additional columns:

    cardinality
        the number of persons in the interaction (e.g. 3 for a 3-way interaction)

    slug
        a human friendly efficient name so we can dump a readable spec trace file for debugging
        this slug is then replaced with the numerical coefficient value prior to evaluation
    """

    activity_column_name = 'activity'
    ptypes_column_name = 'interaction_ptypes'
    coefficient_column_name = 'coefficient'

    f = os.path.join(configs_dir, 'cdap_interaction_coefficients.csv')

    coefficients = pd.read_csv(f, comment='#')

    coefficients['cardinality'] = coefficients[ptypes_column_name].astype(str).str.len()

    wildcards = coefficients.interaction_ptypes == coefficients.cardinality.map(lambda x: x*'*')
    coefficients.loc[wildcards, ptypes_column_name] = ''

    coefficients['slug'] = \
        coefficients[activity_column_name] * coefficients['cardinality'] \
        + coefficients[ptypes_column_name].astype(str)

    return coefficients


@orca.injectable()
def cdap_fixed_relative_proportions(configs_dir):
    """
    spec to compute/specify the relative proportions of each activity (M, N, H)
    that should be used to choose activities for additional household members
    not handled by CDAP

    This spec is handled much like an activitysim logit utility spec,
    EXCEPT that the values computed are relative proportions, not utilities
    (i.e. values are not exponentiated before being normalized to probabilities summing to 1.0)
    """
    f = os.path.join(configs_dir, 'cdap_fixed_relative_proportions.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def xdap_simulate(persons_merged,
                  cdap_settings,
                  cdap_indiv_spec,
                  cdap_interaction_coefficients,
                  cdap_fixed_relative_proportions,
                  hh_chunk_size, trace_hh_id):
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """

    persons_df = persons_merged.to_frame()

    constants = get_model_constants(cdap_settings)

    tracing.info(__name__,
                 "Running xdap_simulate with %d persons" % len(persons_df.index))

    choices = xdap.run_cdap(persons=persons_df,
                            cdap_indiv_spec=cdap_indiv_spec,
                            cdap_interaction_coefficients=cdap_interaction_coefficients,
                            cdap_fixed_relative_proportions=cdap_fixed_relative_proportions,
                            locals_d=constants,
                            chunk_size=hh_chunk_size,
                            trace_hh_id=trace_hh_id,
                            trace_label='xdap')

    choices = choices.reindex(persons_merged.index)

    tracing.print_summary('cdap_activity', choices, value_counts=True)

    orca.add_column("persons", "xdap_activity", choices)

    if trace_hh_id:
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="xdap",
                         columns=['cdap_activity'],
                         warn_if_empty=True)

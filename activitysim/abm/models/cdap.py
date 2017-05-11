# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config


from .util.cdap import run_cdap

logger = logging.getLogger(__name__)


@orca.injectable()
def cdap_settings(configs_dir):
    """
    canonical model settings file to permit definition of local constants for by
    cdap_indiv_spec and cdap_fixed_relative_proportions
    """

    return config.read_model_settings(configs_dir, 'cdap.yaml')


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
    Rules and coefficients for generating interaction specs for different household sizes
    """
    f = os.path.join(configs_dir, 'cdap_interaction_coefficients.csv')
    return pd.read_csv(f, comment='#')


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
def cdap_simulate(persons_merged,
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

    constants = config.get_model_constants(cdap_settings)

    logger.info("Running cdap_simulate with %d persons" % len(persons_df.index))

    choices = run_cdap(persons=persons_df,
                       cdap_indiv_spec=cdap_indiv_spec,
                       cdap_interaction_coefficients=cdap_interaction_coefficients,
                       cdap_fixed_relative_proportions=cdap_fixed_relative_proportions,
                       locals_d=constants,
                       chunk_size=hh_chunk_size,
                       trace_hh_id=trace_hh_id,
                       trace_label='cdap')

    tracing.print_summary('cdap_activity', choices.cdap_activity, value_counts=True)

    print pd.crosstab(persons_df.ptype, choices.cdap_activity, margins=True)

    choices = choices.reindex(persons_merged.index)
    orca.add_column("persons", "cdap_activity", choices.cdap_activity)
    orca.add_column("persons", "cdap_rank", choices.cdap_rank)

    pipeline.add_dependent_columns("persons", "persons_cdap")
    pipeline.add_dependent_columns("households", "households_cdap")

    if trace_hh_id:

        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="cdap",
                         columns=['ptype', 'cdap_rank', 'cdap_activity'],
                         warn_if_empty=True)

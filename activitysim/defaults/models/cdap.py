# ActivitySim
# See full license in LICENSE.txt.

import os

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import tracing
from activitysim.cdap import cdap

from .util.misc import read_model_settings
from activitysim.cdap import xdap


@orca.injectable()
def cdap_1_person_spec(configs_dir):
    f = os.path.join(configs_dir, 'cdap_1_person.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_2_person_spec(configs_dir):
    f = os.path.join(configs_dir, 'cdap_2_person.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_3_person_spec(configs_dir):
    f = os.path.join(configs_dir, 'cdap_3_person.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_final_rules(configs_dir):
    f = os.path.join(configs_dir, 'cdap_final_rules.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_all_people(configs_dir):
    f = os.path.join(configs_dir, 'cdap_all_people.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def cdap_simulate(set_random_seed, persons_merged,
                  cdap_1_person_spec, cdap_2_person_spec, cdap_3_person_spec,
                  cdap_final_rules, cdap_all_people,
                  hh_chunk_size, trace_hh_id):
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """

    choosers = persons_merged.to_frame()

    tracing.info(__name__,
                 "Running cdap_simulate with %d persons" % len(choosers.index))

    choices = cdap.run_cdap(choosers,
                            "household_id",
                            "ptype",
                            cdap_1_person_spec,
                            cdap_2_person_spec,
                            cdap_3_person_spec,
                            cdap_final_rules,
                            cdap_all_people,
                            hh_chunk_size,
                            trace_hh_id,
                            trace_label='cdap')

    choices = choices.reindex(persons_merged.index)

    tracing.print_summary('cdap_activity', choices, value_counts=True)

    orca.add_column("persons", "cdap_activity", choices)

    if trace_hh_id:
        trace_columns = ['cdap_activity']
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="cdap",
                         columns=trace_columns,
                         warn_if_empty=True)


@orca.injectable()
def cdap_settings(configs_dir):
    return read_model_settings(configs_dir, 'auto_ownership.yaml')


@orca.step()
def xdap_simulate(persons_merged,
                  hh_chunk_size, trace_hh_id):
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """

    choosers = persons_merged.to_frame()

    tracing.info(__name__,
                 "Running xdap_simulate with %d persons" % len(choosers.index))

    choices = xdap.run_cdap(people=choosers,
                            hh_id_col="household_id",
                            p_type_col="ptype",
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

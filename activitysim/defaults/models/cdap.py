import os

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim.cdap import cdap

"""
CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
high-level activity pattern for each person, in a coordinated way with other
members of a person's household.

Because Python requires vectorization of computation, there are some specialized
routines in the cdap directory of activitysim for this purpose.  This module
simply applies those utilities using the simulation framework.
"""


@orca.injectable()
def cdap_1_person_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "cdap_1_person.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_2_person_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "cdap_2_person.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_3_person_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "cdap_3_person.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_final_rules(configs_dir):
    f = os.path.join(configs_dir, 'configs', "cdap_final_rules.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def cdap_all_people(configs_dir):
    f = os.path.join(configs_dir, 'configs', "cdap_all_people.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def cdap_simulate(set_random_seed, persons_merged,
                  cdap_1_person_spec, cdap_2_person_spec, cdap_3_person_spec,
                  cdap_final_rules, cdap_all_people):

    choices = cdap.run_cdap(persons_merged.to_frame(),
                            "household_id",
                            "ptype",
                            cdap_1_person_spec,
                            cdap_2_person_spec,
                            cdap_3_person_spec,
                            cdap_final_rules,
                            cdap_all_people)

    print "Choices:\n", choices.value_counts()
    orca.add_column("persons", "cdap_activity", choices)

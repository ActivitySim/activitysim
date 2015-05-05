import os
import orca

from activitysim import activitysim as asim
from .util.misc import add_dependent_columns


"""
The workplace location model predicts the zones in which various people will
work.
"""


@orca.injectable()
def workplace_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "workplace_location.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def workplace_location_simulate(set_random_seed,
                                persons_merged,
                                workplace_location_spec,
                                skims,
                                destination_size_terms):

    # for now I'm going to generate a workplace location for everyone -
    # presumably it will not get used in downstream models for everyone -
    # it should depend on CDAP and mandatory tour generation as to whethrer
    # it gets used
    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    skims.set_keys("TAZ", "TAZ_r")
    # the skims will be available under the name "skims" for any @ expressions
    locals_d = {"skims": skims}

    choices, _ = asim.interaction_simulate(choosers,
                                           alternatives,
                                           workplace_location_spec,
                                           skims=skims,
                                           locals_d=locals_d,
                                           sample_size=50)

    choices = choices.reindex(persons_merged.index)

    print "Describe of choices:\n", choices.describe()
    orca.add_column("persons", "workplace_taz", choices)

    add_dependent_columns("persons", "persons_workplace")

import os
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim


"""
The workplace location model predicts the zones in which various people will
work.
"""


@sim.injectable()
def workplace_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "workplace_location.csv")
    return asim.read_model_spec(f).fillna(0)


@sim.model()
def workplace_location_simulate(set_random_seed,
                                persons_merged,
                                workplace_location_spec,
                                skims,
                                destination_size_terms):

    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.employed_cat.isin(["full", "part"])]
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
                                           locals_d=locals_d)

    choices = choices.reindex(persons_merged.index).fillna(-1).astype('int')

    print "Describe of choices:\n", choices.describe()
    sim.add_column("persons", "workplace_taz", choices)

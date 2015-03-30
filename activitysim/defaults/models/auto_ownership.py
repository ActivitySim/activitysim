import os
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim

"""
Auto ownership is a standard model which predicts how many cars a household
with given characteristics owns
"""

# this is the max number of cars allowable in the auto ownership model
MAX_NUM_CARS = 5


@sim.injectable()
def auto_ownership_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "auto_ownership.csv")
    return asim.read_model_spec(f).fillna(0)


@sim.model()
def auto_ownership_simulate(set_random_seed, households_merged,
                            auto_ownership_spec):
    choices, _ = asim.simple_simulate(
        households_merged.to_frame(), auto_ownership_spec)

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)

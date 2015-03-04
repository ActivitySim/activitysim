import os
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim

"""
Auto ownership is a standard model which predicts how many cars a household
with given characteristics owns
"""

# this is the max number of cars allowable in the auto ownership model
MAX_NUM_CARS = 5


@sim.table()
def auto_alts():
    # alts can't be integers directly as they're used in expressions
    return asim.identity_matrix(["cars%d" % i for i in range(MAX_NUM_CARS)])


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join('configs', "auto_ownership.csv")
    # FIXME should read in all variables and comment out ones not used
    return asim.read_model_spec(f).head(4*26)


@sim.model()
def auto_ownership_simulate(households_merged,
                            auto_alts,
                            auto_ownership_spec):

    choices, _ = asim.simple_simulate(households_merged.to_frame(), 
                                      auto_alts.to_frame(),
                                      auto_ownership_spec,
                                      mult_by_alt_col=True)

    # map these back to integers - this is the actual number of cars chosen
    car_map = dict([("cars%d" % i, i) for i in range(MAX_NUM_CARS)])
    choices = choices.map(car_map)

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)
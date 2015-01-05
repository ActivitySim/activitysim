import urbansim.sim.simulation as sim
import os
from activitysim import activitysim as asim


@sim.table()
def auto_alts():
    return asim.identity_matrix(["cars%d" % i for i in range(5)])


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join('configs', "auto_ownership_coeffs.csv")
    return asim.read_model_spec(f).head(4*26)


@sim.model()
def auto_ownership_simulate(households,
                            auto_alts,
                            auto_ownership_spec,
                            land_use,
                            accessibility):

    choosers = sim.merge_tables(households.name, tables=[households,
                                                         land_use,
                                                         accessibility])
    alternatives = auto_alts.to_frame()

    choices, model_design = \
        asim.simple_simulate(choosers, alternatives, auto_ownership_spec)

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)

    return model_design

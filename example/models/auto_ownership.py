"""
Auto ownership is a standard model which predicts how many cars a household
with given characteristics owns
"""

# this is the max number of cars allowable in the auto ownership model
MAX_NUM_CARS = 5


@sim.table()
def auto_alts():
    return asim.identity_matrix(["cars%d" % i for i in range(MAX_NUM_CARS)])


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join('configs', "auto_ownership.csv")
    # FIXME should read in all variables and comment out ones not used
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
        asim.simple_simulate(choosers, alternatives, auto_ownership_spec,
                             mult_by_alt_col=True)

    # map these back to integers
    choices = choices.map(dict([("cars%d" % i, i)
                                for i in range(MAX_NUM_CARS)]))

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)

    return model_design

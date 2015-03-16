import urbansim.sim.simulation as sim
from activitysim import activitysim as asim
import os
import pandas as pd

"""
Given the tour generation from the above, each tour needs to have a
destination, so in this case tours are the choosers (with the associated
person that's making the tour)
"""


@sim.table()
def destination_choice_size_terms(configs_dir):
    f = os.path.join(configs_dir, 'configs',
                     'destination_choice_size_terms.csv')
    return pd.read_csv(f)


@sim.table()
def destination_choice_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs',
                     'destination_choice_alternatives_sample.csv')
    # FIXME not using all the variables yet
    return asim.read_model_spec(f, stack=False).head(5)


@sim.model()
def destination_choice(non_mandatory_tours_merged,
                       zones,
                       skims,
                       destination_choice_spec):

    # choosers are tours - in a sense tours are choosing their destination
    choosers = non_mandatory_tours_merged.to_frame()

    # FIXME these models don't have size terms at the moment

    # FIXME these models don't use stratified sampling - we're just making
    # FIXME the choice with the sampling model

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    skims.set_keys("TAZ", "TAZ_r")
    # the skims will be available under the name "skims" for any @ expressions
    locals_d = {"skims": skims}

    choices_list = []
    # segment by trip type and pick the right spec for each person type
    for name, segment in choosers.groupby('tour_type'):

        # FIXME - there are two options here escort with kids and without
        if name == "escort":
            continue

        print "Running segment '%s' of size %d" % (name, len(segment))

        choices, _ = \
            asim.simple_simulate(segment,
                                 zones.to_frame(),
                                 destination_choice_spec[name],
                                 skims=skims,
                                 locals_d=locals_d,
                                 mult_by_alt_col=False,
                                 sample_size=50)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    print "Choices:\n", choices.describe()
    # every trip now has a destination which is the index from the
    # alternatives table - in this case it's the destination taz
    sim.add_column("non_mandatory_tours", "destination", choices)

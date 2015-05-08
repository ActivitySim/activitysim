import os

import orca
import pandas as pd

from activitysim import activitysim as asim

"""
Given the tour generation from the above, each tour needs to have a
destination, so in this case tours are the choosers (with the associated
person that's making the tour)
"""


@orca.table()
def destination_choice_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', 'destination_choice.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def destination_choice(set_random_seed,
                       non_mandatory_tours_merged,
                       skims,
                       destination_choice_spec,
                       destination_size_terms):

    # choosers are tours - in a sense tours are choosing their destination
    choosers = non_mandatory_tours_merged.to_frame()
    alternatives = destination_size_terms.to_frame()
    spec = destination_choice_spec.to_frame()

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
            # FIXME just run one of the other models for now
            name = "shopping"

        # the segment is now available to switch between size terms
        locals_d['segment'] = name

        print "Running segment '%s' of size %d" % (name, len(segment))

        choices, _ = asim.interaction_simulate(segment,
                                               alternatives,
                                               spec[[name]],
                                               skims=skims,
                                               locals_d=locals_d,
                                               sample_size=50)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    print "Choices:\n", choices.describe()
    # every trip now has a destination which is the index from the
    # alternatives table - in this case it's the destination taz
    orca.add_column("non_mandatory_tours", "destination", choices)

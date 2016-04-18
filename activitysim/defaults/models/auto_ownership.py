# ActivitySim
# See full license in LICENSE.txt.

import os
import orca

from activitysim import activitysim as asim
from .util.misc import add_dependent_columns


@orca.injectable()
def auto_ownership_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "auto_ownership.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def auto_ownership_simulate(set_random_seed, households_merged,
                            auto_ownership_spec):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """

    choices, _ = asim.simple_simulate(
        households_merged.to_frame(), auto_ownership_spec)

    print "Choices:\n", choices.value_counts()

    orca.add_column("households", "auto_ownership", choices)

    add_dependent_columns("households", "households_autoown")

# ActivitySim
# See full license in LICENSE.txt.

import os

import orca
import pandas as pd

from activitysim import activitysim as asim
from .util.misc import add_dependent_columns


@orca.table()
def school_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "school_location.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def school_location_simulate(set_random_seed,
                             persons_merged,
                             school_location_spec,
                             skims,
                             destination_size_terms):

    """
    The school location model predicts the zones in which various people will
    go to school.
    """

    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()
    spec = school_location_spec.to_frame()

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    skims.set_keys("TAZ", "TAZ_r")
    # the skims will be available under the name "skims" for any @ expressions
    locals_d = {"skims": skims}

    choices_list = []
    for school_type in ['university', 'highschool', 'gradeschool']:

        locals_d['segment'] = school_type

        choosers_segment = choosers[choosers["is_" + school_type]]

        choices, _ = asim.interaction_simulate(choosers_segment,
                                               alternatives,
                                               spec[[school_type]],
                                               skims=skims,
                                               locals_d=locals_d,
                                               sample_size=50)
        choices_list.append(choices)

    choices = pd.concat(choices_list)

    # this fillna is necessary to avoid a downstream crash and might be a bit
    # wrong logically.  The issue here is that there is a small but non-zero
    # chance to choose a school trip even if not of the school type (because
    # of -999 rather than outright removal of alternative availability). -
    # this fills in the location for those uncommon circumstances,
    # so at least it runs
    choices = choices.reindex(persons_merged.index).fillna(-1)

    print "Describe of choices:\n", choices.describe()

    orca.add_column("persons", "school_taz", choices)

    add_dependent_columns("persons", "persons_school")

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import numpy as np

from activitysim import activitysim as asim
from activitysim import tracing
from .util.misc import add_dependent_columns


logger = logging.getLogger(__name__)


@orca.table()
def school_location_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "school_location.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def school_location_simulate(set_random_seed,
                             persons_merged,
                             school_location_spec,
                             skims,
                             destination_size_terms,
                             chunk_size,
                             trace_hh_id):

    """
    The school location model predicts the zones in which various people will
    go to school.
    """

    choosers = persons_merged.to_frame()
    alternatives = destination_size_terms.to_frame()
    spec = school_location_spec.to_frame()

    tracing.info(__name__,
                 "Running school_location_simulate with %d persons" % len(choosers))

    # set the keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    skims.set_keys("TAZ", "TAZ_r")
    # the skims will be available under the name "skims" for any @ expressions
    locals_d = {"skims": skims}

    choices_list = []
    for school_type in ['university', 'highschool', 'gradeschool']:

        tracing.info(__name__, "Running school_type %s" % school_type)

        locals_d['segment'] = school_type

        choosers_segment = choosers[choosers["is_" + school_type]]

        choices = asim.interaction_simulate(
            choosers_segment,
            alternatives,
            spec=spec[[school_type]],
            skims=skims,
            locals_d=locals_d,
            sample_size=50,
            chunk_size=chunk_size,
            trace_label='school_location.%s' % school_type,
            trace_choice_name='school_location')

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    # this fillna is necessary to avoid a downstream crash and might be a bit
    # wrong logically.  The issue here is that there is a small but non-zero
    # chance to choose a school trip even if not of the school type (because
    # of -999 rather than outright removal of alternative availability). -
    # this fills in the location for those uncommon circumstances,
    # so at least it runs
    if np.isnan(choices).any():
        if trace_hh_id:
            tracing.trace_nan_values(choices, 'school_location.choices')
        logger.warn("Converting %s nan school_taz choices to -1" %
                    (np.isnan(choices).sum(), len(choices.index)))
    choices = choices.reindex(persons_merged.index).fillna(-1)

    tracing.info(__name__, "%s school_taz choices min: %s max: %s" %
                 (len(choices.index), choices.min(), choices.max()))

    tracing.print_summary('school_taz', choices, describe=True)

    orca.add_column("persons", "school_taz", choices)
    add_dependent_columns("persons", "persons_school")

    if trace_hh_id:
        trace_columns = ['school_taz'] + orca.get_table('persons_school').columns
        tracing.trace_df(orca.get_table('persons_merged').to_frame(),
                         label="school_location",
                         columns=trace_columns)

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import tracing
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling


logger = logging.getLogger(__name__)


@orca.table()
def tdd_non_mandatory_spec(configs_dir):
    f = os.path.join(configs_dir, 'tour_departure_and_duration_nonmandatory.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def non_mandatory_scheduling(set_random_seed,
                             non_mandatory_tours_merged,
                             tdd_alts,
                             tdd_non_mandatory_spec,
                             chunk_size,
                             trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for
    non-mandatory tours
    """

    tours = non_mandatory_tours_merged.to_frame()

    tracing.info(__name__,
                 "Running non_mandatory_scheduling with %d tours" % len(tours))

    spec = tdd_non_mandatory_spec.to_frame()
    alts = tdd_alts.to_frame()

    choices = vectorize_tour_scheduling(tours, alts, spec, chunk_size,
                                        trace_label='non_mandatory_scheduling')

    tracing.print_summary('non_mandatory_scheduling tour_departure_and_duration',
                          choices, describe=True)

    orca.add_column(
        "non_mandatory_tours", "tour_departure_and_duration", choices)

    if trace_hh_id:
        tracing.trace_df(orca.get_table('non_mandatory_tours').to_frame(),
                         label="non_mandatory_tours",
                         slicer='person_id',
                         index_label='tour_id',
                         columns=None,
                         warn=True)

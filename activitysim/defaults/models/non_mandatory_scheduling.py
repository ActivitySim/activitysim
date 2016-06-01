# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import trace
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling


logger = logging.getLogger(__name__)


@orca.table()
def tdd_non_mandatory_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs',
                     'tour_departure_and_duration_nonmandatory.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def non_mandatory_scheduling(set_random_seed,
                             non_mandatory_tours_merged,
                             tdd_alts,
                             tdd_non_mandatory_spec,
                             chunk_size):
    """
    This model predicts the departure time and duration of each activity for
    non-mandatory tours
    """

    tours = non_mandatory_tours_merged.to_frame()

    logger.info("Running %d non-mandatory tour scheduling choices" % len(tours))

    spec = tdd_non_mandatory_spec.to_frame()
    alts = tdd_alts.to_frame()

    choices = vectorize_tour_scheduling(tours, alts, spec, chunk_size)

    trace.print_summary('non_mandatory_scheduling tour_departure_and_duration',
                        choices, describe=True)

    orca.add_column(
        "non_mandatory_tours", "tour_departure_and_duration", choices)

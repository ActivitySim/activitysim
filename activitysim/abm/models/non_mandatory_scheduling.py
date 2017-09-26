# ActivitySim
# See full license in LICENSE.txt.

import os
import logging


from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject


from .util.vectorize_tour_scheduling import vectorize_tour_scheduling


logger = logging.getLogger(__name__)


@inject.injectable()
def tdd_non_mandatory_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_nonmandatory.csv')


@inject.injectable()
def non_mandatory_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'non_mandatory_tour_scheduling.yaml')


@inject.step()
def non_mandatory_tour_scheduling(non_mandatory_tours_merged,
                                  tdd_alts,
                                  tdd_non_mandatory_spec,
                                  non_mandatory_tour_scheduling_settings,
                                  chunk_size,
                                  trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for
    non-mandatory tours
    """

    tours = non_mandatory_tours_merged.to_frame()

    logger.info("Running non_mandatory_tour_scheduling with %d tours" % len(tours))

    constants = config.get_model_constants(non_mandatory_tour_scheduling_settings)

    alts = tdd_alts.to_frame()

    choices = vectorize_tour_scheduling(tours, alts, tdd_non_mandatory_spec,
                                        constants=constants,
                                        chunk_size=chunk_size,
                                        trace_label='non_mandatory_tour_scheduling')

    tracing.print_summary('non_mandatory_tour_scheduling tour_departure_and_duration',
                          choices, describe=True)

    inject.add_column("non_mandatory_tours", "tour_departure_and_duration", choices)

    if trace_hh_id:
        tracing.trace_df(inject.get_table('non_mandatory_tours').to_frame(),
                         label="non_mandatory_tours",
                         slicer='person_id',
                         index_label='tour_id',
                         columns=None,
                         warn_if_empty=True)

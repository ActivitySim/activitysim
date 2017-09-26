# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject

from .util.vectorize_tour_scheduling import vectorize_tour_scheduling

logger = logging.getLogger(__name__)


@inject.injectable()
def mandatory_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'mandatory_tour_scheduling.yaml')


@inject.table()
def tdd_alts(configs_dir):
    # right now this file just contains the start and end hour
    f = os.path.join(configs_dir, 'tour_departure_and_duration_alternatives.csv')
    return pd.read_csv(f)


# used to have duration in the actual alternative csv file,
# but this is probably better as a computed column like this
@inject.column("tdd_alts")
def duration(tdd_alts):
    return tdd_alts.end - tdd_alts.start


@inject.injectable()
def tdd_work_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_work.csv')


@inject.injectable()
def tdd_school_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_departure_and_duration_school.csv')


# I think it's easier to do this in one model so you can merge the two
# resulting series together right away
@inject.step()
def mandatory_tour_scheduling(mandatory_tours_merged,
                              tdd_alts,
                              tdd_school_spec,
                              tdd_work_spec,
                              mandatory_tour_scheduling_settings,
                              chunk_size,
                              trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for
    mandatory tours
    """

    tours = mandatory_tours_merged.to_frame()
    alts = tdd_alts.to_frame()

    constants = config.get_model_constants(mandatory_tour_scheduling_settings)

    school_tours = tours[tours.tour_type == "school"]

    logger.info("Running mandatory_tour_scheduling school_tours with %d tours" % len(school_tours))

    school_choices = vectorize_tour_scheduling(
        school_tours, alts, tdd_school_spec,
        constants=constants,
        chunk_size=chunk_size,
        trace_label='mandatory_tour_scheduling.school')

    work_tours = tours[tours.tour_type == "work"]

    logger.info("Running %d work tour scheduling choices" % len(work_tours))

    work_choices = vectorize_tour_scheduling(
        work_tours, alts, tdd_work_spec,
        constants=constants,
        chunk_size=chunk_size,
        trace_label='mandatory_tour_scheduling.work')

    choices = pd.concat([school_choices, work_choices])

    tracing.print_summary('mandatory_tour_scheduling tour_departure_and_duration',
                          choices, describe=True)

    inject.add_column("mandatory_tours", "tour_departure_and_duration", choices)

    if trace_hh_id:
        tracing.trace_df(inject.get_table('mandatory_tours').to_frame(),
                         label="mandatory_tours",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)

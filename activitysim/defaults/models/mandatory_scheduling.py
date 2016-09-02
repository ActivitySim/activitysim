# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import tracing
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling

from .util.misc import read_model_settings, get_model_constants

logger = logging.getLogger(__name__)


@orca.injectable()
def mandatory_scheduling_settings(configs_dir):
    return read_model_settings(configs_dir, 'mandatory_scheduling.yaml')


@orca.table()
def tdd_alts(configs_dir):
    # right now this file just contains the start and end hour
    f = os.path.join(configs_dir, 'tour_departure_and_duration_alternatives.csv')
    return pd.read_csv(f)


# used to have duration in the actual alternative csv file,
# but this is probably better as a computed column like this
@orca.column("tdd_alts")
def duration(tdd_alts):
    return tdd_alts.end - tdd_alts.start


@orca.table()
def tdd_work_spec(configs_dir):
    f = os.path.join(configs_dir, 'tour_departure_and_duration_work.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.table()
def tdd_school_spec(configs_dir):
    f = os.path.join(configs_dir, 'tour_departure_and_duration_school.csv')
    return asim.read_model_spec(f).fillna(0)


# I think it's easier to do this in one model so you can merge the two
# resulting series together right away
@orca.step()
def mandatory_scheduling(set_random_seed,
                         mandatory_tours_merged,
                         tdd_alts,
                         tdd_school_spec,
                         tdd_work_spec,
                         mandatory_scheduling_settings,
                         chunk_size,
                         trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for
    mandatory tours
    """

    tours = mandatory_tours_merged.to_frame()
    alts = tdd_alts.to_frame()

    if trace_hh_id:
        # register non_mandatory_tours so we can slice utilities
        tracing.register_tours(tours, trace_hh_id)

    constants = get_model_constants(mandatory_scheduling_settings)

    school_spec = tdd_school_spec.to_frame()
    school_tours = tours[tours.tour_type == "school"]

    tracing.info(__name__,
                 "Running mandatory_scheduling school_tours with %d tours" % len(school_tours))

    school_choices = vectorize_tour_scheduling(
        school_tours, alts, school_spec, constants, chunk_size,
        trace_label='mandatory_scheduling.school')

    work_spec = tdd_work_spec.to_frame()
    work_tours = tours[tours.tour_type == "work"]

    tracing.info(__name__, "Running %d work tour scheduling choices" % len(work_tours))

    work_choices = vectorize_tour_scheduling(
        work_tours, alts, work_spec, constants, chunk_size,
        trace_label='mandatory_scheduling.work')

    choices = pd.concat([school_choices, work_choices])

    tracing.print_summary('mandatory_scheduling tour_departure_and_duration',
                          choices, describe=True)

    orca.add_column(
        "mandatory_tours", "tour_departure_and_duration", choices)

    if trace_hh_id:
        tracing.trace_df(orca.get_table('mandatory_tours').to_frame(),
                         label="mandatory_tours",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn=True)

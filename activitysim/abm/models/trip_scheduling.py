# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import logit
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import chunk
from activitysim.core import pipeline

from activitysim.core.util import assign_in_place
from .util import expressions
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)

"""
StopDepartArrivePeriodModel

StopDepartArriveProportions.csv
tourpurp,isInbound,interval,trip,p1,p2,p3,p4,p5...p40

"""


def trip_scheduling_probs(configs_dir):

    f = os.path.join(configs_dir, 'trip_purpose_probs.csv')
    df = pd.read_csv(f, comment='#')
    return df


@inject.step()
def trip_scheduling(
        trips,
        tours,
        configs_dir,
        chunk_size,
        trace_hh_id):

    trace_label = "trip_scheduling"

    model_settings = config.read_model_settings(configs_dir, 'trip_scheduling.yaml')
    probs_spec = trip_scheduling_probs(configs_dir)

    trips_df = trips.to_frame()
    tours = tours.to_frame()

    result_list = []

    trips_df['tour_hour'] = np.where(
        trips_df.outbound,
        reindex(tours.start, trips_df.tour_id),
        reindex(tours.end, trips_df.tour_id))

    print trips_df
    bug

    # - last trip of outbound tour
    purpose = trips_df.primary_purpose[trips_df['last'] & trips_df.outbound]
    result_list.append(purpose)
    logger.info("assign purpose to %s last outbound trips" % purpose.shape[0])

    # - last trip of inbound tour gets home (or work for atwork subtours)
    purpose = trips_df.primary_purpose[trips_df['last'] & ~trips_df.outbound]
    purpose = pd.Series(np.where(purpose == 'atwork', 'Work', 'Home'), index=purpose.index)
    result_list.append(purpose)
    logger.info("assign purpose to %s last inbound trips" % purpose.shape[0])

    # - intermediate stops (non-last trips) purpose assigned by probability table

    trips_df = trips_df[~trips_df['last']]
    logger.info("assign purpose to %s intermediate trips" % trips_df.shape[0])

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.util import reindex
from activitysim.core.util import assign_in_place

from activitysim.abm.models.trip_purpose import run_trip_purpose
from activitysim.abm.models.trip_destination import run_trip_destination

logger = logging.getLogger(__name__)


def run_trip_purpose_and_destination(
        trips_df,
        tours_merged_df,
        configs_dir,
        chunk_size,
        trace_hh_id,
        trace_label):

    choices = run_trip_purpose(
        trips_df,
        configs_dir=configs_dir,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=tracing.extend_trace_label(trace_label, 'purpose')
    )

    trips_df['purpose'] = choices

    trips_df = run_trip_destination(
        trips_df,
        tours_merged_df,
        configs_dir, chunk_size, trace_hh_id,
        trace_label=tracing.extend_trace_label(trace_label, 'destination'))

    return trips_df


@inject.step()
def trip_purpose_and_destination(
        trips,
        tours_merged,
        configs_dir,
        chunk_size,
        trace_hh_id):

    trace_label = "trip_purpose_and_destination"

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    results = []
    i = 0
    MAX_ITERATIONS = 5
    RESULT_COLUMNS = ['purpose', 'destination', 'origin', 'bad']

    while True:

        i += 1

        trips_df = run_trip_purpose_and_destination(
            trips_df,
            tours_merged_df,
            configs_dir,
            chunk_size,
            trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, "i%s" % i))

        num_bad_trips = trips_df.bad.sum()

        # if there were no bad trips, we are done
        if num_bad_trips == 0:
            results.append(trips_df[RESULT_COLUMNS])
            break

        logger.warn("%s %s failed trips in iteration %s" % (trace_label, num_bad_trips, i))

        # if max iterations reached, add remaining trips to results and give up
        if i >= MAX_ITERATIONS:
            logger.warn("%s too many iterations %s" % (trace_label, i))
            results.append(trips_df[RESULT_COLUMNS])
            break

        # otherwise, if any trips failed, then their leg-mates trips must also fail
        for ob in [True, False]:
            same_leg = (trips_df.outbound == ob)
            bad_tours = trips_df.tour_id[trips_df.bad & same_leg].unique()
            bad_trip_leg_mates = same_leg & (trips_df.tour_id.isin(bad_tours))
            trips_df.loc[bad_trip_leg_mates, 'bad'] = True

        num_kootie_trips = trips_df.bad.sum() - num_bad_trips
        if num_kootie_trips:
            logger.warn("%s %s kootie trips in iteration %s" % (trace_label, num_kootie_trips, i))

        # add the good trips to results
        results.append(trips_df[~trips_df.bad][RESULT_COLUMNS])

        # and keep the bad ones to retry
        trips_df = trips_df[trips_df.bad]
        tours_merged_df = tours_merged_df[tours_merged_df.index.isin(trips_df.tour_id)]

    # - assign result columns to trips
    results = pd.concat(results)
    trips_df = trips.to_frame()
    assign_in_place(trips_df, results)
    pipeline.replace_table("trips", trips_df)

    logger.info("%s %s failed trips" % (trace_label, trips_df.bad.sum()))

    if trips_df.bad.any():
        file_name = "%s_bad_trips" % trace_label
        logger.warn("%s writing %s failed trips to %s" %
                    (trace_label, trips_df.bad.sum(), file_name))
        tracing.write_csv(trips_df[trips_df.bad], file_name=file_name)

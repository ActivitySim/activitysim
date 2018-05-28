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

    assert not trips_df.empty

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


def fail_bad_trip_leg_mates(trips_df):
    """
    fail bad trip leg_mates in place
    """

    # otherwise, if any trips failed, then their leg-mates trips must also fail
    for ob in [True, False]:
        same_leg = (trips_df.outbound == ob)
        bad_tours = trips_df.tour_id[trips_df.bad & same_leg].unique()
        bad_trip_leg_mates = same_leg & (trips_df.tour_id.isin(bad_tours))
        trips_df.loc[bad_trip_leg_mates, 'bad'] = True


@inject.step()
def trip_purpose_and_destination(
        trips,
        tours_merged,
        configs_dir,
        chunk_size,
        trace_hh_id):

    trace_label = "trip_purpose_and_destination"
    model_settings = config.read_model_settings(configs_dir, 'trip_purpose_and_destination.yaml')

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    # if trip_destination has been run before, keep only bad trips (and leg_mates) to retry
    if 'bad' in trips_df:
        logger.info('trip_destination has already been run. Rerunning failed trips')
        fail_bad_trip_leg_mates(trips_df)
        trips_df = trips_df[trips_df.bad]
        tours_merged_df = tours_merged_df[tours_merged_df.index.isin(trips_df.tour_id)]

    if trips_df.empty:
        logger.info("%s - no trips. Nothing to do.")
        return

    results = []
    i = 0
    MAX_ITERATIONS = model_settings.get('max_iterations', 5)
    RESULT_COLUMNS = ['purpose', 'destination', 'origin', 'bad']

    while True:

        i += 1

        for c in RESULT_COLUMNS:
            del trips_df[c]

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
        file_name = "%s_failed_trips_%s" % (trace_label, i)
        logger.info("writing failed trips to %s" % file_name)
        tracing.write_csv(trips_df[trips_df.bad], file_name=file_name)

        # if max iterations reached, add remaining trips to results and give up
        # note that we do this BEFORE failing leg_mates so resulting trip legs are complete
        if i >= MAX_ITERATIONS:
            logger.warn("%s too many iterations %s" % (trace_label, i))
            results.append(trips_df[RESULT_COLUMNS])
            break

        # otherwise, if any trips failed, then their leg-mates trips must also fail
        fail_bad_trip_leg_mates(trips_df)

        num_leg_mates = trips_df.bad.sum() - num_bad_trips
        if num_leg_mates:
            logger.warn("%s %s failed trip leg_mates in iteration %s" %
                        (trace_label, num_leg_mates, i))

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

    if trips_df.bad.any():
        logger.warn("%s %s failed trips" % (trace_label, trips_df.bad.sum()))

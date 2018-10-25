# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.util import assign_in_place

from activitysim.abm.models.trip_purpose import run_trip_purpose
from activitysim.abm.models.trip_destination import run_trip_destination

from activitysim.abm.models.util.trip import flag_failed_trip_leg_mates
from activitysim.abm.models.util.trip import cleanup_failed_trips


logger = logging.getLogger(__name__)


def run_trip_purpose_and_destination(
        trips_df,
        tours_merged_df,
        chunk_size,
        trace_hh_id,
        trace_label):

    assert not trips_df.empty

    choices = run_trip_purpose(
        trips_df,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=tracing.extend_trace_label(trace_label, 'purpose')
    )

    trips_df['purpose'] = choices

    trips_df = run_trip_destination(
        trips_df,
        tours_merged_df,
        chunk_size, trace_hh_id,
        trace_label=tracing.extend_trace_label(trace_label, 'destination'))

    return trips_df


@inject.step()
def trip_purpose_and_destination(
        trips,
        tours_merged,
        chunk_size,
        trace_hh_id):

    trace_label = "trip_purpose_and_destination"
    model_settings = config.read_model_settings('trip_purpose_and_destination.yaml')

    MAX_ITERATIONS = model_settings.get('MAX_ITERATIONS', 5)

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    if trips_df.empty:
        logger.info("%s - no trips. Nothing to do." % trace_label)
        return

    # FIXME could allow MAX_ITERATIONS=0 to allow for cleanup-only run
    # in which case, we would need to drop bad trips, WITHOUT failing bad_trip leg_mates
    assert (MAX_ITERATIONS > 0)

    # if trip_destination has been run before, keep only failed trips (and leg_mates) to retry
    if 'destination' in trips_df:
        if trips_df.failed.any():
            logger.info('trip_destination has already been run. Rerunning failed trips')
            flag_failed_trip_leg_mates(trips_df, 'failed')
            trips_df = trips_df[trips_df.failed]
            tours_merged_df = tours_merged_df[tours_merged_df.index.isin(trips_df.tour_id)]
            logger.info('Rerunning %s failed trips and leg-mates' % trips_df.shape[0])
        else:
            # no failed trips from prior run of trip_destination
            logger.info("%s - no failed trips from prior model run." % trace_label)
            del trips_df['failed']
            pipeline.replace_table("trips", trips_df)
            return

    results = []
    i = 0
    RESULT_COLUMNS = ['purpose', 'destination', 'origin', 'failed']
    while True:

        i += 1

        for c in RESULT_COLUMNS:
            if c in trips_df:
                del trips_df[c]

        trips_df = run_trip_purpose_and_destination(
            trips_df,
            tours_merged_df,
            chunk_size,
            trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, "i%s" % i))

        num_failed_trips = trips_df.failed.sum()

        # if there were no failed trips, we are done
        if num_failed_trips == 0:
            results.append(trips_df[RESULT_COLUMNS])
            break

        logger.warning("%s %s failed trips in iteration %s" % (trace_label, num_failed_trips, i))
        file_name = "%s_i%s_failed_trips" % (trace_label, i)
        logger.info("writing failed trips to %s" % file_name)
        tracing.write_csv(trips_df[trips_df.failed], file_name=file_name, transpose=False)

        # if max iterations reached, add remaining trips to results and give up
        # note that we do this BEFORE failing leg_mates so resulting trip legs are complete
        if i >= MAX_ITERATIONS:
            logger.warning("%s too many iterations %s" % (trace_label, i))
            results.append(trips_df[RESULT_COLUMNS])
            break

        # otherwise, if any trips failed, then their leg-mates trips must also fail
        flag_failed_trip_leg_mates(trips_df, 'failed')

        # add the good trips to results
        results.append(trips_df[~trips_df.failed][RESULT_COLUMNS])

        # and keep the failed ones to retry
        trips_df = trips_df[trips_df.failed]
        tours_merged_df = tours_merged_df[tours_merged_df.index.isin(trips_df.tour_id)]

    # - assign result columns to trips
    results = pd.concat(results)

    logger.info("%s %s failed trips after %s iterations" % (trace_label, results.failed.sum(), i))

    trips_df = trips.to_frame()
    assign_in_place(trips_df, results)

    trips_df = cleanup_failed_trips(trips_df)

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(trips_df,
                         label=trace_label,
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)

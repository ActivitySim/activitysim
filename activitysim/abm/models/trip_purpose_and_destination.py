# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.abm.models.trip_destination import run_trip_destination
from activitysim.abm.models.trip_purpose import run_trip_purpose
from activitysim.abm.models.util.trip import (
    cleanup_failed_trips,
    flag_failed_trip_leg_mates,
)
from activitysim.core import config, inject, pipeline, tracing
from activitysim.core.util import assign_in_place

from .util import estimation

logger = logging.getLogger(__name__)


def run_trip_purpose_and_destination(
    trips_df, tours_merged_df, chunk_size, trace_hh_id, trace_label
):

    assert not trips_df.empty

    choices = run_trip_purpose(
        trips_df,
        estimator=None,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=tracing.extend_trace_label(trace_label, "purpose"),
    )

    trips_df["purpose"] = choices

    trips_df, save_sample_df = run_trip_destination(
        trips_df,
        tours_merged_df,
        estimator=None,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=tracing.extend_trace_label(trace_label, "destination"),
    )

    return trips_df, save_sample_df


@inject.step()
def trip_purpose_and_destination(trips, tours_merged, chunk_size, trace_hh_id):

    trace_label = "trip_purpose_and_destination"
    model_settings = config.read_model_settings("trip_purpose_and_destination.yaml")

    # for consistency, read sample_table_name setting from trip_destination settings file
    trip_destination_model_settings = config.read_model_settings(
        "trip_destination.yaml"
    )
    sample_table_name = trip_destination_model_settings.get(
        "DEST_CHOICE_SAMPLE_TABLE_NAME"
    )
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    MAX_ITERATIONS = model_settings.get("MAX_ITERATIONS", 5)

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    if trips_df.empty:
        logger.info("%s - no trips. Nothing to do." % trace_label)
        return

    # FIXME could allow MAX_ITERATIONS=0 to allow for cleanup-only run
    # in which case, we would need to drop bad trips, WITHOUT failing bad_trip leg_mates
    assert MAX_ITERATIONS > 0

    # if trip_destination has been run before, keep only failed trips (and leg_mates) to retry
    if "destination" in trips_df:

        if "failed" not in trips_df.columns:
            # trip_destination model cleaned up any failed trips
            logger.info("%s - no failed column from prior model run." % trace_label)
            return

        elif not trips_df.failed.any():
            # 'failed' column but no failed trips from prior run of trip_destination
            logger.info("%s - no failed trips from prior model run." % trace_label)
            trips_df.drop(columns="failed", inplace=True)
            pipeline.replace_table("trips", trips_df)
            return

        else:
            logger.info("trip_destination has already been run. Rerunning failed trips")
            flag_failed_trip_leg_mates(trips_df, "failed")
            trips_df = trips_df[trips_df.failed]
            tours_merged_df = tours_merged_df[
                tours_merged_df.index.isin(trips_df.tour_id)
            ]
            logger.info("Rerunning %s failed trips and leg-mates" % trips_df.shape[0])

            # drop any previously saved samples of failed trips
            if want_sample_table and pipeline.is_table(sample_table_name):
                logger.info("Dropping any previously saved samples of failed trips")
                save_sample_df = pipeline.get_table(sample_table_name)
                save_sample_df.drop(trips_df.index, level="trip_id", inplace=True)
                pipeline.replace_table(sample_table_name, save_sample_df)
                del save_sample_df

    # if we estimated trip_destination, there should have been no failed trips
    # if we didn't, but it is enabled, it is probably a configuration error
    # if we just estimated trip_purpose, it isn't clear what they are trying to do , nor how to handle it
    assert not (
        estimation.manager.begin_estimation("trip_purpose")
        or estimation.manager.begin_estimation("trip_destination")
    )

    processed_trips = []
    save_samples = []
    i = 0
    TRIP_RESULT_COLUMNS = ["purpose", "destination", "origin", "failed"]
    while True:

        i += 1

        for c in TRIP_RESULT_COLUMNS:
            if c in trips_df:
                del trips_df[c]

        trips_df, save_sample_df = run_trip_purpose_and_destination(
            trips_df,
            tours_merged_df,
            chunk_size=chunk_size,
            trace_hh_id=trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, "i%s" % i),
        )

        # # if testing, make sure at least one trip fails
        if (
            config.setting("testing_fail_trip_destination", False)
            and (i == 1)
            and not trips_df.failed.any()
        ):
            fail_o = trips_df[trips_df.trip_num < trips_df.trip_count].origin.max()
            trips_df.failed = (trips_df.origin == fail_o) & (
                trips_df.trip_num < trips_df.trip_count
            )

        num_failed_trips = trips_df.failed.sum()

        # if there were no failed trips, we are done
        if num_failed_trips == 0:
            processed_trips.append(trips_df[TRIP_RESULT_COLUMNS])
            if save_sample_df is not None:
                save_samples.append(save_sample_df)
            break

        logger.warning(
            "%s %s failed trips in iteration %s" % (trace_label, num_failed_trips, i)
        )
        file_name = "%s_i%s_failed_trips" % (trace_label, i)
        logger.info("writing failed trips to %s" % file_name)
        tracing.write_csv(
            trips_df[trips_df.failed], file_name=file_name, transpose=False
        )

        # if max iterations reached, add remaining trips to processed_trips and give up
        # note that we do this BEFORE failing leg_mates so resulting trip legs are complete
        if i >= MAX_ITERATIONS:
            logger.warning("%s too many iterations %s" % (trace_label, i))
            processed_trips.append(trips_df[TRIP_RESULT_COLUMNS])
            if save_sample_df is not None:
                save_sample_df.drop(
                    trips_df[trips_df.failed].index, level="trip_id", inplace=True
                )
                save_samples.append(save_sample_df)
            break

        # otherwise, if any trips failed, then their leg-mates trips must also fail
        flag_failed_trip_leg_mates(trips_df, "failed")

        # add the good trips to processed_trips
        processed_trips.append(trips_df[~trips_df.failed][TRIP_RESULT_COLUMNS])

        # and keep the failed ones to retry
        trips_df = trips_df[trips_df.failed]
        tours_merged_df = tours_merged_df[tours_merged_df.index.isin(trips_df.tour_id)]

        #  add trip samples of processed_trips to processed_samples
        if save_sample_df is not None:
            # drop failed trip samples
            save_sample_df.drop(trips_df.index, level="trip_id", inplace=True)
            save_samples.append(save_sample_df)

    # - assign result columns to trips
    processed_trips = pd.concat(processed_trips)

    if len(save_samples) > 0:
        save_sample_df = pd.concat(save_samples)
        logger.info(
            "adding %s samples to %s" % (len(save_sample_df), sample_table_name)
        )
        pipeline.extend_table(sample_table_name, save_sample_df)

    logger.info(
        "%s %s failed trips after %s iterations"
        % (trace_label, processed_trips.failed.sum(), i)
    )

    trips_df = trips.to_frame()
    assign_in_place(trips_df, processed_trips)

    trips_df = cleanup_failed_trips(trips_df)

    pipeline.replace_table("trips", trips_df)

    # check to make sure we wrote sample file if requestsd
    if want_sample_table and len(trips_df) > 0:
        assert pipeline.is_table(sample_table_name)
        # since we have saved samples for all successful trips
        # once we discard failed trips, we should samples for all trips
        save_sample_df = pipeline.get_table(sample_table_name)
        # expect samples only for intermediate trip destinatinos
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(
            trips_df[trips_df.trip_num < trips_df.trip_count]
        )
        del save_sample_df

    if trace_hh_id:
        tracing.trace_df(
            trips_df,
            label=trace_label,
            slicer="trip_id",
            index_label="trip_id",
            warn_if_empty=True,
        )

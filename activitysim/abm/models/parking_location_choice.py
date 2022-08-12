# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    expressions,
    inject,
    logit,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.tracing import print_elapsed_time
from activitysim.core.util import assign_in_place

from .util import estimation

logger = logging.getLogger(__name__)

NO_DESTINATION = -1


def wrap_skims(model_settings):
    """
    wrap skims of trip destination using origin, dest column names from model settings.
    Various of these are used by destination_sample, compute_logsums, and destination_simulate
    so we create them all here with canonical names.

    Note that compute_logsums aliases their names so it can use the same equations to compute
    logsums from origin to alt_dest, and from alt_dest to primarly destination

    odt_skims - SkimStackWrapper: trip origin, trip alt_dest, time_of_day
    dot_skims - SkimStackWrapper: trip alt_dest, trip origin, time_of_day
    dpt_skims - SkimStackWrapper: trip alt_dest, trip primary_dest, time_of_day
    pdt_skims - SkimStackWrapper: trip primary_dest,trip alt_dest, time_of_day
    od_skims - SkimDictWrapper: trip origin, trip alt_dest
    dp_skims - SkimDictWrapper: trip alt_dest, trip primary_dest

    Parameters
    ----------
    model_settings

    Returns
    -------
        dict containing skims, keyed by canonical names relative to tour orientation
    """

    network_los = inject.get_injectable("network_los")
    skim_dict = network_los.get_default_skim_dict()

    origin = model_settings["TRIP_ORIGIN"]
    park_zone = model_settings["ALT_DEST_COL_NAME"]
    destination = model_settings["TRIP_DESTINATION"]
    time_period = model_settings["TRIP_DEPARTURE_PERIOD"]

    skims = {
        "odt_skims": skim_dict.wrap_3d(
            orig_key=origin, dest_key=destination, dim3_key=time_period
        ),
        "dot_skims": skim_dict.wrap_3d(
            orig_key=destination, dest_key=origin, dim3_key=time_period
        ),
        "opt_skims": skim_dict.wrap_3d(
            orig_key=origin, dest_key=park_zone, dim3_key=time_period
        ),
        "pdt_skims": skim_dict.wrap_3d(
            orig_key=park_zone, dest_key=destination, dim3_key=time_period
        ),
        "od_skims": skim_dict.wrap(origin, destination),
        "do_skims": skim_dict.wrap(destination, origin),
        "op_skims": skim_dict.wrap(origin, park_zone),
        "pd_skims": skim_dict.wrap(park_zone, destination),
    }

    return skims


def get_spec_for_segment(model_settings, spec_name, segment):

    omnibus_spec = simulate.read_model_spec(file_name=model_settings[spec_name])

    spec = omnibus_spec[[segment]]

    # might as well ignore any spec rows with 0 utility
    spec = spec[spec.iloc[:, 0] != 0]
    assert spec.shape[0] > 0

    return spec


def parking_destination_simulate(
    segment_name,
    trips,
    destination_sample,
    model_settings,
    skims,
    chunk_size,
    trace_hh_id,
    trace_label,
):
    """
    Chose destination from destination_sample (with od_logsum and dp_logsum columns added)


    Returns
    -------
    choices - pandas.Series
        destination alt chosen
    """
    trace_label = tracing.extend_trace_label(trace_label, "trip_destination_simulate")

    spec = get_spec_for_segment(model_settings, "SPECIFICATION", segment_name)

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running trip_destination_simulate with %d trips", len(trips))

    locals_dict = config.get_model_constants(model_settings).copy()
    locals_dict.update(skims)

    parking_locations = interaction_sample_simulate(
        choosers=trips,
        alternatives=destination_sample,
        spec=spec,
        choice_column=alt_dest_col_name,
        want_logsums=False,
        allow_zero_probs=True,
        zero_prob_choice_val=NO_DESTINATION,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="parking_loc",
    )

    # drop any failed zero_prob destinations
    if (parking_locations == NO_DESTINATION).any():
        logger.debug(
            "dropping %s failed parking locations",
            (parking_locations == NO_DESTINATION).sum(),
        )
        parking_locations = parking_locations[parking_locations != NO_DESTINATION]

    return parking_locations


def choose_parking_location(
    segment_name,
    trips,
    alternatives,
    model_settings,
    want_sample_table,
    skims,
    chunk_size,
    trace_hh_id,
    trace_label,
):

    logger.info("choose_parking_location %s with %d trips", trace_label, trips.shape[0])

    t0 = print_elapsed_time()

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]
    destination_sample = logit.interaction_dataset(
        trips, alternatives, alt_index_id=alt_dest_col_name
    )
    destination_sample.index = np.repeat(trips.index.values, len(alternatives))
    destination_sample.index.name = trips.index.name
    destination_sample = destination_sample[[alt_dest_col_name]].copy()

    # # - trip_destination_simulate
    destinations = parking_destination_simulate(
        segment_name=segment_name,
        trips=trips,
        destination_sample=destination_sample,
        model_settings=model_settings,
        skims=skims,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    if want_sample_table:
        # FIXME - sample_table
        destination_sample.set_index(
            model_settings["ALT_DEST_COL_NAME"], append=True, inplace=True
        )
    else:
        destination_sample = None

    t0 = print_elapsed_time("%s.parking_location_simulate" % trace_label, t0)

    return destinations, destination_sample


def run_parking_destination(
    model_settings,
    trips,
    land_use,
    chunk_size,
    trace_hh_id,
    trace_label,
    fail_some_trips_for_testing=False,
):

    chooser_filter_column = model_settings.get("CHOOSER_FILTER_COLUMN_NAME")
    chooser_segment_column = model_settings.get("CHOOSER_SEGMENT_COLUMN_NAME")

    parking_location_column_name = model_settings["ALT_DEST_COL_NAME"]
    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    choosers = trips[trips[chooser_filter_column]]
    choosers = choosers.sort_index()

    # Placeholder for trips without a parking choice
    trips[parking_location_column_name] = -1

    skims = wrap_skims(model_settings)

    alt_column_filter_name = model_settings.get("ALTERNATIVE_FILTER_COLUMN_NAME")
    alternatives = land_use[land_use[alt_column_filter_name]]

    # don't need size terms in alternatives, just TAZ index
    alternatives = alternatives.drop(alternatives.columns, axis=1)
    alternatives.index.name = parking_location_column_name

    choices_list = []
    sample_list = []
    for segment_name, chooser_segment in choosers.groupby(chooser_segment_column):
        if chooser_segment.shape[0] == 0:
            logger.info(
                "%s skipping segment %s: no choosers", trace_label, segment_name
            )
            continue

        choices, destination_sample = choose_parking_location(
            segment_name,
            chooser_segment,
            alternatives,
            model_settings,
            want_sample_table,
            skims,
            chunk_size,
            trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, segment_name),
        )

        choices_list.append(choices)
        if want_sample_table:
            assert destination_sample is not None
            sample_list.append(destination_sample)

    if len(choices_list) > 0:
        parking_df = pd.concat(choices_list)

        if fail_some_trips_for_testing:
            parking_df = parking_df.drop(parking_df.index[0])

        assign_in_place(trips, parking_df.to_frame(parking_location_column_name))
        trips[parking_location_column_name] = trips[
            parking_location_column_name
        ].fillna(-1)
    else:
        trips[parking_location_column_name] = -1

    save_sample_df = pd.concat(sample_list) if len(sample_list) > 0 else None

    return trips[parking_location_column_name], save_sample_df


@inject.step()
def parking_location(
    trips, trips_merged, land_use, network_los, chunk_size, trace_hh_id
):
    """
    Given a set of trips, each trip needs to have a parking location if
    it is eligible for remote parking.
    """

    trace_label = "parking_location"
    model_settings = config.read_model_settings("parking_location_choice.yaml")
    alt_destination_col_name = model_settings["ALT_DEST_COL_NAME"]

    preprocessor_settings = model_settings.get("PREPROCESSOR", None)

    trips_df = trips.to_frame()
    trips_merged_df = trips_merged.to_frame()
    land_use_df = land_use.to_frame()

    locals_dict = {"network_los": network_los}
    locals_dict.update(config.get_model_constants(model_settings))

    if preprocessor_settings:
        expressions.assign_columns(
            df=trips_merged_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    parking_locations, save_sample_df = run_parking_destination(
        model_settings,
        trips_merged_df,
        land_use_df,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    assign_in_place(trips_df, parking_locations.to_frame(alt_destination_col_name))

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(
            trips_df,
            label=trace_label,
            slicer="trip_id",
            index_label="trip_id",
            warn_if_empty=True,
        )

    if save_sample_df is not None:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(
            trips_df[trips_df.trip_num < trips_df.trip_count]
        )

        sample_table_name = model_settings.get("PARKING_LOCATION_SAMPLE_TABLE_NAME")
        assert sample_table_name is not None

        logger.info(
            "adding %s samples to %s" % (len(save_sample_df), sample_table_name)
        )

        # lest they try to put tour samples into the same table
        if pipeline.is_table(sample_table_name):
            raise RuntimeError("sample table %s already exists" % sample_table_name)
        pipeline.extend_table(sample_table_name, save_sample_df)

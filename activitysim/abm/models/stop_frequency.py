# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing
from activitysim.core.util import assign_in_place, reindex

from .util import estimation, trip

logger = logging.getLogger(__name__)


@inject.step()
def stop_frequency(
    tours, tours_merged, stop_frequency_alts, network_los, chunk_size, trace_hh_id
):
    """
    stop frequency model

    For each tour, shoose a number of intermediate inbound stops and outbound stops.
    Create a trip table with inbound and outbound trips.

    Thus, a tour with stop_frequency '2out_0in' will have two outbound and zero inbound stops,
    and four corresponding trips: three outbound, and one inbound.

    Adds stop_frequency str column to trips, with fields

    creates trips table with columns:

    ::

        - person_id
        - household_id
        - tour_id
        - primary_purpose
        - atwork
        - trip_num
        - outbound
        - trip_count

    """

    trace_label = "stop_frequency"
    model_settings_file_name = "stop_frequency.yaml"

    model_settings = config.read_model_settings(model_settings_file_name)

    tours = tours.to_frame()
    tours_merged = tours_merged.to_frame()
    assert not tours_merged.household_id.isnull().any()
    assert not (tours_merged.origin == -1).any()
    assert not (tours_merged.destination == -1).any()

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    # - run preprocessor to annotate tours_merged
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        # hack: preprocessor adds origin column in place if it does not exist already
        assert "origin" in tours_merged
        assert "destination" in tours_merged
        od_skim_stack_wrapper = network_los.get_default_skim_dict().wrap(
            "origin", "destination"
        )
        skims = [od_skim_stack_wrapper]

        locals_dict = {"od_skims": od_skim_stack_wrapper, "network_los": network_los}
        locals_dict.update(constants)

        simulate.set_skim_wrapper_targets(tours_merged, skims)

        # this should be pre-slice as some expressions may count tours by type
        annotations = expressions.compute_columns(
            df=tours_merged,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

        assign_in_place(tours_merged, annotations)

    tracing.print_summary(
        "stop_frequency segments", tours_merged.primary_purpose, value_counts=True
    )

    spec_segments = model_settings.get("SPEC_SEGMENTS")
    assert (
        spec_segments is not None
    ), f"SPEC_SEGMENTS setting not found in model settings: {model_settings_file_name}"
    segment_col = model_settings.get("SEGMENT_COL")
    assert (
        segment_col is not None
    ), f"SEGMENT_COL setting not found in model settings: {model_settings_file_name}"

    nest_spec = config.get_logit_model_settings(model_settings)

    choices_list = []
    for segment_settings in spec_segments:

        segment_name = segment_settings[segment_col]
        segment_value = segment_settings[segment_col]

        chooser_segment = tours_merged[tours_merged[segment_col] == segment_value]

        if len(chooser_segment) == 0:
            logging.info(f"{trace_label} skipping empty segment {segment_name}")
            continue

        logging.info(
            f"{trace_label} running segment {segment_name} with {chooser_segment.shape[0]} chooser rows"
        )

        estimator = estimation.manager.begin_estimation(
            model_name=segment_name, bundle_name="stop_frequency"
        )

        segment_spec = simulate.read_model_spec(file_name=segment_settings["SPEC"])
        assert segment_spec is not None, (
            "spec for segment_type %s not found" % segment_name
        )

        coefficients_file_name = segment_settings["COEFFICIENTS"]
        coefficients_df = simulate.read_model_coefficients(
            file_name=coefficients_file_name
        )
        segment_spec = simulate.eval_coefficients(
            segment_spec, coefficients_df, estimator
        )

        if estimator:
            estimator.write_spec(segment_settings, bundle_directory=False)
            estimator.write_model_settings(
                model_settings, model_settings_file_name, bundle_directory=True
            )
            estimator.write_coefficients(coefficients_df, segment_settings)
            estimator.write_choosers(chooser_segment)

            estimator.set_chooser_id(chooser_segment.index.name)

        choices = simulate.simple_simulate(
            choosers=chooser_segment,
            spec=segment_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, segment_name),
            trace_choice_name="stops",
            estimator=estimator,
        )

        # convert indexes to alternative names
        choices = pd.Series(segment_spec.columns[choices.values], index=choices.index)

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "tours", "stop_frequency"
            )  # override choices
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    tracing.print_summary("stop_frequency", choices, value_counts=True)

    # add stop_frequency choices to tours table
    assign_in_place(tours, choices.to_frame("stop_frequency"))

    # FIXME should have added this when tours created?
    assert "primary_purpose" not in tours
    if "primary_purpose" not in tours.columns:
        # if not already there, then it will have been added by stop_freq_annotate_tours_preprocessor
        assign_in_place(tours, tours_merged[["primary_purpose"]])

    pipeline.replace_table("tours", tours)

    # create trips table
    trips = trip.initialize_from_tours(tours, stop_frequency_alts)
    pipeline.replace_table("trips", trips)
    tracing.register_traceable_table("trips", trips)
    pipeline.get_rn_generator().add_channel("trips", trips)

    if estimator:
        # make sure they created trips with the expected tour_ids
        columns = ["person_id", "household_id", "tour_id", "outbound"]

        survey_trips = estimation.manager.get_survey_table(table_name="trips")
        different = False
        survey_trips_not_in_trips = survey_trips[~survey_trips.index.isin(trips.index)]
        if len(survey_trips_not_in_trips) > 0:
            print(f"survey_trips_not_in_trips\n{survey_trips_not_in_trips}")
            different = True
        trips_not_in_survey_trips = trips[~trips.index.isin(survey_trips.index)]
        if len(survey_trips_not_in_trips) > 0:
            print(f"trips_not_in_survey_trips\n{trips_not_in_survey_trips}")
            different = True
        assert not different

        survey_trips = estimation.manager.get_survey_values(
            trips, table_name="trips", column_names=columns
        )

        trips_differ = (trips[columns] != survey_trips[columns]).any(axis=1)

        if trips_differ.any():
            print("trips_differ\n%s" % trips_differ)
            print("%s of %s tours differ" % (trips_differ.sum(), len(trips_differ)))
            print("differing survey_trips\n%s" % survey_trips[trips_differ])
            print("differing modeled_trips\n%s" % trips[columns][trips_differ])

        assert not trips_differ.any()

    if trace_hh_id:
        tracing.trace_df(
            tours, label="stop_frequency.tours", slicer="person_id", columns=None
        )

        tracing.trace_df(
            trips, label="stop_frequency.trips", slicer="person_id", columns=None
        )

        tracing.trace_df(annotations, label="stop_frequency.annotations", columns=None)

        tracing.trace_df(
            tours_merged,
            label="stop_frequency.tours_merged",
            slicer="person_id",
            columns=None,
        )

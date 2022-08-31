# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    assign,
    chunk,
    config,
    expressions,
    inject,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.pathbuilder import TransitVirtualPathBuilder
from activitysim.core.util import assign_in_place

from .util import estimation
from .util.mode import mode_choice_simulate

logger = logging.getLogger(__name__)


@inject.step()
def trip_mode_choice(trips, network_los, chunk_size, trace_hh_id):
    """
    Trip mode choice - compute trip_mode (same values as for tour_mode) for each trip.

    Modes for each primary tour putpose are calculated separately because they have different
    coefficient values (stored in trip_mode_choice_coefficients.csv coefficient file.)

    Adds trip_mode column to trip table
    """

    trace_label = "trip_mode_choice"
    model_settings_file_name = "trip_mode_choice.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME")
    mode_column_name = "trip_mode"

    trips_df = trips.to_frame()
    logger.info("Running %s with %d trips", trace_label, trips_df.shape[0])

    # give trip mode choice the option to run without calling tours_merged. Useful for xborder
    # model where tour_od_choice needs trip mode choice logsums before some of the join keys
    # needed by tour_merged (e.g. home_zone_id) exist
    tours_cols = [
        col
        for col in model_settings["TOURS_MERGED_CHOOSER_COLUMNS"]
        if col not in trips_df.columns
    ]
    if len(tours_cols) > 0:
        tours_merged = inject.get_table("tours_merged").to_frame(columns=tours_cols)
    else:
        tours_merged = pd.DataFrame()

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips_df, tours_merged, left_on="tour_id", right_index=True, how="left"
    )
    assert trips_merged.index.equals(trips.index)

    tracing.print_summary(
        "primary_purpose", trips_df.primary_purpose, value_counts=True
    )

    # setup skim keys
    assert "trip_period" not in trips_merged
    trips_merged["trip_period"] = network_los.skim_time_period_label(
        trips_merged.depart
    )

    orig_col = "origin"
    dest_col = "destination"
    min_per_period = network_los.skim_time_periods["period_minutes"]
    periods_per_hour = 60 / min_per_period

    constants = {}
    constants.update(config.get_model_constants(model_settings))
    constants.update(
        {
            "ORIGIN": orig_col,
            "DESTINATION": dest_col,
            "MIN_PER_PERIOD": min_per_period,
            "PERIODS_PER_HOUR": periods_per_hour,
        }
    )

    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col, dest_key=dest_col, dim3_key="trip_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col, dest_key=orig_col, dim3_key="trip_period"
    )
    od_skim_wrapper = skim_dict.wrap("origin", "destination")

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?
        tvpb = network_los.tvpb
        tvpb_recipe = model_settings.get("TVPB_recipe", "tour_mode_choice")
        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col,
            dest_key=dest_col,
            tod_key="trip_period",
            segment_key="demographic_segment",
            recipe=tvpb_recipe,
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        skims.update(
            {
                "tvpb_logsum_odt": tvpb_logsum_odt,
            }
        )

        # This if-clause gives the user the option of NOT inheriting constants
        # from the tvpb settings. previously, these constants were inherited
        # automatically, which had the undesirable effect of overwriting any
        # trip mode choice model constants/coefficients that shared the same
        # name. The default behavior is still the same (True), but the user
        # can now avoid any chance of squashing these local variables by
        # adding `use_TVPB_constants: False` to the trip_mode_choice.yaml file.
        # the tvpb will still use the constants as defined in the recipe
        # specified above in `tvpb.wrap_logsum()` but they will not be used
        # in the trip mode choice expressions.
        if model_settings.get("use_TVPB_constants", True):
            constants.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

    # don't create estimation data bundle if trip mode choice is being called
    # from another model step (e.g. tour mode choice logsum creation)
    if pipeline._PIPELINE.rng().step_name != "trip_mode_choice":
        estimator = None
    else:
        estimator = estimation.manager.begin_estimation("trip_mode_choice")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    nest_spec = config.get_logit_model_settings(model_settings)

    choices_list = []
    for primary_purpose, trips_segment in trips_merged.groupby("primary_purpose"):

        segment_trace_label = tracing.extend_trace_label(trace_label, primary_purpose)

        logger.info(
            "trip_mode_choice tour_type '%s' (%s trips)"
            % (
                primary_purpose,
                len(trips_segment.index),
            )
        )

        # name index so tracing knows how to slice
        assert trips_segment.index.name == "trip_id"

        if network_los.zone_system == los.THREE_ZONE:
            tvpb_logsum_odt.extend_trace_label(primary_purpose)
            # tvpb_logsum_dot.extend_trace_label(primary_purpose)

        coefficients = simulate.get_segment_coefficients(
            model_settings, primary_purpose
        )

        locals_dict = {}
        locals_dict.update(constants)

        constants_keys = constants.keys()
        if any([coeff in constants_keys for coeff in coefficients.keys()]):
            logger.warning("coefficients are obscuring constants in locals_dict")
        locals_dict.update(coefficients)

        # have to initialize chunker for preprocessing in order to access
        # tvpb logsum terms in preprocessor expressions.
        with chunk.chunk_log(
            tracing.extend_trace_label(trace_label, "preprocessing"), base=True
        ):
            expressions.annotate_preprocessors(
                trips_segment, locals_dict, skims, model_settings, segment_trace_label
            )

        if estimator:
            # write choosers after annotation
            estimator.write_choosers(trips_segment)

        locals_dict.update(skims)

        choices = mode_choice_simulate(
            choosers=trips_segment,
            spec=simulate.eval_coefficients(model_spec, coefficients, estimator),
            nest_spec=simulate.eval_nest_coefficients(
                nest_spec, coefficients, segment_trace_label
            ),
            skims=skims,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            mode_column_name=mode_column_name,
            logsum_column_name=logsum_column_name,
            trace_label=segment_trace_label,
            trace_choice_name="trip_mode_choice",
            estimator=estimator,
        )

        if trace_hh_id:
            # trace the coefficients
            tracing.trace_df(
                pd.Series(locals_dict),
                label=tracing.extend_trace_label(segment_trace_label, "constants"),
                transpose=False,
                slicer="NONE",
            )

            # so we can trace with annotations
            assign_in_place(trips_segment, choices)

            tracing.trace_df(
                trips_segment,
                label=tracing.extend_trace_label(segment_trace_label, "trip_mode"),
                slicer="tour_id",
                index_label="tour_id",
                warn_if_empty=True,
            )

        choices_list.append(choices)

    choices_df = pd.concat(choices_list)

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:

        tvpb_mode_path_types = model_settings.get("tvpb_mode_path_types")
        for mode, path_type in tvpb_mode_path_types.items():

            skim_cache = tvpb_logsum_odt.cache[path_type]

            for c in skim_cache:
                dest_col = c
                if dest_col not in choices_df:
                    choices_df[dest_col] = (
                        np.nan if pd.api.types.is_numeric_dtype(skim_cache[c]) else ""
                    )
                choices_df[dest_col].where(
                    choices_df[mode_column_name] != mode, skim_cache[c], inplace=True
                )

    if estimator:
        estimator.write_choices(choices_df.trip_mode)
        choices_df.trip_mode = estimator.get_survey_values(
            choices_df.trip_mode, "trips", "trip_mode"
        )
        estimator.write_override_choices(choices_df.trip_mode)
        estimator.end_estimation()
    trips_df = trips.to_frame()
    assign_in_place(trips_df, choices_df)

    tracing.print_summary("trip_modes", trips_merged.tour_mode, value_counts=True)

    tracing.print_summary(
        "trip_mode_choice choices", trips_df[mode_column_name], value_counts=True
    )

    assert not trips_df[mode_column_name].isnull().any()

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(
            trips_df,
            label=tracing.extend_trace_label(trace_label, "trip_mode"),
            slicer="trip_id",
            index_label="trip_id",
            warn_if_empty=True,
        )

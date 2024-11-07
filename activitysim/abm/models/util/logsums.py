# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import config, expressions, los, simulate, tracing, workflow
from activitysim.core.configuration import PydanticBase
from activitysim.core.configuration.logit import (
    TourLocationComponentSettings,
    TourModeComponentSettings,
)

logger = logging.getLogger(__name__)


def filter_chooser_columns(
    choosers, logsum_settings: dict | PydanticBase, model_settings: dict | PydanticBase
):
    try:
        chooser_columns = logsum_settings.LOGSUM_CHOOSER_COLUMNS
    except AttributeError:
        chooser_columns = logsum_settings.get("LOGSUM_CHOOSER_COLUMNS", [])

    if (
        isinstance(model_settings, dict)
        and "CHOOSER_ORIG_COL_NAME" in model_settings
        and model_settings["CHOOSER_ORIG_COL_NAME"] not in chooser_columns
    ):
        chooser_columns.append(model_settings["CHOOSER_ORIG_COL_NAME"])
    if (
        isinstance(model_settings, PydanticBase)
        and hasattr(model_settings, "CHOOSER_ORIG_COL_NAME")
        and model_settings.CHOOSER_ORIG_COL_NAME
        and model_settings.CHOOSER_ORIG_COL_NAME not in chooser_columns
    ):
        chooser_columns.append(model_settings.CHOOSER_ORIG_COL_NAME)

    missing_columns = [c for c in chooser_columns if c not in choosers]
    if missing_columns:
        logger.debug(
            "logsum.filter_chooser_columns missing_columns %s" % missing_columns
        )

    # ignore any columns not appearing in choosers df
    chooser_columns = [c for c in chooser_columns if c in choosers]

    choosers = choosers[chooser_columns]
    return choosers


def compute_location_choice_logsums(
    state: workflow.State,
    choosers: pd.DataFrame,
    tour_purpose,
    logsum_settings: TourModeComponentSettings,
    model_settings: TourLocationComponentSettings,
    network_los: los.Network_LOS,
    chunk_size: int,
    chunk_tag: str,
    trace_label: str,
    in_period_col: str | None = None,
    out_period_col: str | None = None,
    duration_col: str | None = None,
):
    """

    Parameters
    ----------
    choosers
    tour_purpose
    logsum_settings : TourModeComponentSettings
    model_settings : TourLocationComponentSettings
    network_los
    chunk_size
    trace_hh_id
    trace_label

    Returns
    -------
    logsums: pandas series
        computed logsums with same index as choosers
    """
    if isinstance(model_settings, dict):
        model_settings = TourLocationComponentSettings.model_validate(model_settings)
    if isinstance(logsum_settings, dict):
        logsum_settings = TourModeComponentSettings.model_validate(logsum_settings)

    trace_label = tracing.extend_trace_label(trace_label, "compute_logsums")
    logger.debug(f"Running compute_logsums with {choosers.shape[0]:d} choosers")

    # compute_logsums needs to know name of dest column in interaction_sample
    orig_col_name = model_settings.CHOOSER_ORIG_COL_NAME
    dest_col_name = model_settings.ALT_DEST_COL_NAME

    assert (in_period_col is not None) or (model_settings.IN_PERIOD is not None)
    assert (out_period_col is not None) or (model_settings.OUT_PERIOD is not None)

    # FIXME - are we ok with altering choosers (so caller doesn't have to set these)?
    if (in_period_col is not None) and (out_period_col is not None):
        choosers["in_period"] = network_los.skim_time_period_label(
            choosers[in_period_col], as_cat=True
        )
        choosers["out_period"] = network_los.skim_time_period_label(
            choosers[out_period_col], as_cat=True
        )
    elif ("in_period" not in choosers.columns) and (
        "out_period" not in choosers.columns
    ):
        if (
            type(model_settings.IN_PERIOD) is dict
            and type(model_settings.OUT_PERIOD) is dict
        ):
            if (
                tour_purpose in model_settings.IN_PERIOD
                and tour_purpose in model_settings.OUT_PERIOD
            ):
                choosers["in_period"] = network_los.skim_time_period_label(
                    model_settings.IN_PERIOD[tour_purpose],
                    as_cat=True,
                    broadcast_to=choosers.index,
                )
                choosers["out_period"] = network_los.skim_time_period_label(
                    model_settings.OUT_PERIOD[tour_purpose],
                    as_cat=True,
                    broadcast_to=choosers.index,
                )
        else:
            choosers["in_period"] = network_los.skim_time_period_label(
                model_settings.IN_PERIOD, as_cat=True, broadcast_to=choosers.index
            )
            choosers["out_period"] = network_los.skim_time_period_label(
                model_settings.OUT_PERIOD, as_cat=True, broadcast_to=choosers.index
            )
    else:
        logger.error("Choosers table already has columns 'in_period' and 'out_period'.")

    if duration_col is not None:
        choosers["duration"] = choosers[duration_col]
    elif "duration" not in choosers.columns:
        if (
            type(model_settings.IN_PERIOD) is dict
            and type(model_settings.OUT_PERIOD) is dict
        ):
            if (
                tour_purpose in model_settings.IN_PERIOD
                and tour_purpose in model_settings.OUT_PERIOD
            ):
                choosers["duration"] = (
                    model_settings.IN_PERIOD[tour_purpose]
                    - model_settings.OUT_PERIOD[tour_purpose]
                )
        else:
            choosers["duration"] = model_settings.IN_PERIOD - model_settings.OUT_PERIOD
    else:
        logger.error("Choosers table already has column 'duration'.")

    logsum_spec = state.filesystem.read_model_spec(file_name=logsum_settings.SPEC)
    coefficients = state.filesystem.get_segment_coefficients(
        logsum_settings, tour_purpose
    )

    logsum_spec = simulate.eval_coefficients(
        state, logsum_spec, coefficients, estimator=None
    )

    nest_spec = config.get_logit_model_settings(logsum_settings)
    if nest_spec is not None:  # nest_spec is None for MNL
        nest_spec = simulate.eval_nest_coefficients(
            nest_spec, coefficients, trace_label
        )

    locals_dict = {}
    # model_constants can appear in expressions
    locals_dict.update(config.get_model_constants(logsum_settings))
    # constrained coefficients can appear in expressions
    locals_dict.update(coefficients)

    # setup skim keys
    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="out_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="in_period"
    )
    odr_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="in_period"
    )
    dor_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="out_period"
    )
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,
        "dor_skims": dor_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        "orig_col_name": orig_col_name,
        "dest_col_name": dest_col_name,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?
        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col_name,
            dest_key=dest_col_name,
            tod_key="out_period",
            segment_key="demographic_segment",
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col_name,
            dest_key=orig_col_name,
            tod_key="in_period",
            segment_key="demographic_segment",
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

        # TVPB constants can appear in expressions
        if logsum_settings.use_TVPB_constants:
            locals_dict.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

    locals_dict.update(skims)

    # - run preprocessor to annotate choosers
    # allow specification of alternate preprocessor for nontour choosers
    preprocessor = model_settings.LOGSUM_PREPROCESSOR
    preprocessor_settings = getattr(logsum_settings, preprocessor, None)

    if preprocessor_settings:
        simulate.set_skim_wrapper_targets(choosers, skims)

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    logsums = simulate.simple_simulate_logsums(
        state,
        choosers,
        logsum_spec,
        nest_spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        explicit_chunk_size=model_settings.explicit_chunk,
        compute_settings=logsum_settings.compute_settings,
    )

    return logsums

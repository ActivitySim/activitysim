# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, los, simulate, tracing
from activitysim.core.pathbuilder import TransitVirtualPathBuilder

logger = logging.getLogger(__name__)


def filter_chooser_columns(choosers, logsum_settings, model_settings):

    chooser_columns = logsum_settings.get("LOGSUM_CHOOSER_COLUMNS", [])

    if (
        "CHOOSER_ORIG_COL_NAME" in model_settings
        and model_settings["CHOOSER_ORIG_COL_NAME"] not in chooser_columns
    ):
        chooser_columns.append(model_settings["CHOOSER_ORIG_COL_NAME"])

    missing_columns = [c for c in chooser_columns if c not in choosers]
    if missing_columns:
        logger.debug(
            "logsum.filter_chooser_columns missing_columns %s" % missing_columns
        )

    # ignore any columns not appearing in choosers df
    chooser_columns = [c for c in chooser_columns if c in choosers]

    choosers = choosers[chooser_columns]
    return choosers


def compute_logsums(
    choosers,
    tour_purpose,
    logsum_settings,
    model_settings,
    network_los,
    chunk_size,
    chunk_tag,
    trace_label,
    in_period_col=None,
    out_period_col=None,
    duration_col=None,
):
    """

    Parameters
    ----------
    choosers
    tour_purpose
    logsum_settings
    model_settings
    network_los
    chunk_size
    trace_hh_id
    trace_label

    Returns
    -------
    logsums: pandas series
        computed logsums with same index as choosers
    """

    trace_label = tracing.extend_trace_label(trace_label, "compute_logsums")
    logger.debug("Running compute_logsums with %d choosers" % choosers.shape[0])

    # compute_logsums needs to know name of dest column in interaction_sample
    orig_col_name = model_settings["CHOOSER_ORIG_COL_NAME"]
    dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    # FIXME - are we ok with altering choosers (so caller doesn't have to set these)?
    if (in_period_col is not None) and (out_period_col is not None):
        choosers["in_period"] = network_los.skim_time_period_label(
            choosers[in_period_col]
        )
        choosers["out_period"] = network_los.skim_time_period_label(
            choosers[out_period_col]
        )
    elif ("in_period" not in choosers.columns) and (
        "out_period" not in choosers.columns
    ):
        if (
            type(model_settings["IN_PERIOD"]) is dict
            and type(model_settings["OUT_PERIOD"]) is dict
        ):
            if (
                tour_purpose in model_settings["IN_PERIOD"]
                and tour_purpose in model_settings["OUT_PERIOD"]
            ):
                choosers["in_period"] = network_los.skim_time_period_label(
                    model_settings["IN_PERIOD"][tour_purpose]
                )
                choosers["out_period"] = network_los.skim_time_period_label(
                    model_settings["OUT_PERIOD"][tour_purpose]
                )
        else:
            choosers["in_period"] = network_los.skim_time_period_label(
                model_settings["IN_PERIOD"]
            )
            choosers["out_period"] = network_los.skim_time_period_label(
                model_settings["OUT_PERIOD"]
            )
    else:
        logger.error("Choosers table already has columns 'in_period' and 'out_period'.")

    if duration_col is not None:
        choosers["duration"] = choosers[duration_col]
    elif "duration" not in choosers.columns:
        if (
            type(model_settings["IN_PERIOD"]) is dict
            and type(model_settings["OUT_PERIOD"]) is dict
        ):
            if (
                tour_purpose in model_settings["IN_PERIOD"]
                and tour_purpose in model_settings["OUT_PERIOD"]
            ):
                choosers["duration"] = (
                    model_settings["IN_PERIOD"][tour_purpose]
                    - model_settings["OUT_PERIOD"][tour_purpose]
                )
        else:
            choosers["duration"] = (
                model_settings["IN_PERIOD"] - model_settings["OUT_PERIOD"]
            )
    else:
        logger.error("Choosers table already has column 'duration'.")

    logsum_spec = simulate.read_model_spec(file_name=logsum_settings["SPEC"])
    coefficients = simulate.get_segment_coefficients(logsum_settings, tour_purpose)

    logsum_spec = simulate.eval_coefficients(logsum_spec, coefficients, estimator=None)

    nest_spec = config.get_logit_model_settings(logsum_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)

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
        if logsum_settings.get("use_TVPB_constants", True):
            locals_dict.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

    locals_dict.update(skims)

    # - run preprocessor to annotate choosers
    # allow specification of alternate preprocessor for nontour choosers
    preprocessor = model_settings.get("LOGSUM_PREPROCESSOR", "preprocessor")
    preprocessor_settings = logsum_settings[preprocessor]

    if preprocessor_settings:

        simulate.set_skim_wrapper_targets(choosers, skims)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
    )

    return logsums

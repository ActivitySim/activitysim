# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, expressions, simulate, tracing

"""
At this time, these utilities are mostly for transforming the mode choice
spec, which is more complicated than the other specs, into something that
looks like the other specs.
"""

logger = logging.getLogger(__name__)


def mode_choice_simulate(
    choosers,
    spec,
    nest_spec,
    skims,
    locals_d,
    chunk_size,
    mode_column_name,
    logsum_column_name,
    trace_label,
    trace_choice_name,
    trace_column_names=None,
    estimator=None,
):
    """
    common method for  both tour_mode_choice and trip_mode_choice

    Parameters
    ----------
    choosers
    spec
    nest_spec
    skims
    locals_d
    chunk_size
    mode_column_name
    logsum_column_name
    trace_label
    trace_choice_name
    estimator

    Returns
    -------

    """
    want_logsums = logsum_column_name is not None

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=spec,
        nest_spec=nest_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        want_logsums=want_logsums,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name,
        estimator=estimator,
        trace_column_names=trace_column_names,
    )

    # for consistency, always return dataframe, whether or not logsums were requested
    if isinstance(choices, pd.Series):
        choices = choices.to_frame("choice")

    choices.rename(
        columns={"logsum": logsum_column_name, "choice": mode_column_name}, inplace=True
    )

    alts = spec.columns
    choices[mode_column_name] = choices[mode_column_name].map(
        dict(list(zip(list(range(len(alts))), alts)))
    )

    return choices


def run_tour_mode_choice_simulate(
    choosers,
    tour_purpose,
    model_settings,
    mode_column_name,
    logsum_column_name,
    network_los,
    skims,
    constants,
    estimator,
    chunk_size,
    trace_label=None,
    trace_choice_name=None,
):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are tour/trip purposes).  Pass in the tours/trip that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients = simulate.get_segment_coefficients(model_settings, tour_purpose)

    spec = simulate.eval_coefficients(spec, coefficients, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(skims)

    # coefficients can appear in expressions
    locals_keys = locals_dict.keys()
    if any([coeff in locals_keys for coeff in coefficients.keys()]):
        logger.warning("coefficients are obscuring locals_dict values")
    locals_dict.update(coefficients)

    assert ("in_period" not in choosers) and ("out_period" not in choosers)
    in_time = skims["in_time_col_name"]
    out_time = skims["out_time_col_name"]
    choosers["in_period"] = network_los.skim_time_period_label(choosers[in_time])
    choosers["out_period"] = network_los.skim_time_period_label(choosers[out_time])

    expressions.annotate_preprocessors(
        choosers, locals_dict, skims, model_settings, trace_label
    )

    trace_column_names = choosers.index.name
    assert trace_column_names == "tour_id"
    if trace_column_names not in choosers:
        choosers[trace_column_names] = choosers.index

    if estimator:
        # write choosers after annotation
        estimator.write_choosers(choosers)

    choices = mode_choice_simulate(
        choosers=choosers,
        spec=spec,
        nest_spec=nest_spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        mode_column_name=mode_column_name,
        logsum_column_name=logsum_column_name,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name,
        trace_column_names=trace_column_names,
        estimator=estimator,
    )

    return choices

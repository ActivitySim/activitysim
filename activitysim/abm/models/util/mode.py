# ActivitySim
# See full license in LICENSE.txt.
import pandas as pd

from activitysim.core import simulate
from activitysim.core import config

from . import expressions
from . import estimation


"""
At this time, these utilities are mostly for transforming the mode choice
spec, which is more complicated than the other specs, into something that
looks like the other specs.
"""


def mode_choice_simulate(
        choosers, spec, nest_spec, skims, locals_d,
        chunk_size,
        mode_column_name,
        logsum_column_name,
        trace_label,
        trace_choice_name,
        estimator=None):

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
        estimator=estimator)

    # for consistency, always return dataframe, whether or not logsums were requested
    if isinstance(choices, pd.Series):
        choices = choices.to_frame('choice')

    choices.rename(columns={'logsum': logsum_column_name,
                            'choice': mode_column_name},
                   inplace=True)

    alts = spec.columns
    choices[mode_column_name] = \
        choices[mode_column_name].map(dict(list(zip(list(range(len(alts))), alts))))

    return choices


def run_tour_mode_choice_simulate(
        choosers,
        tour_purpose, model_settings,
        mode_column_name,
        logsum_column_name,
        skims,
        constants,
        estimator,
        chunk_size,
        trace_label=None, trace_choice_name=None):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are tour/trip purposes).  Pass in the tours/trip that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients = simulate.get_segment_coefficients(model_settings, tour_purpose)
    spec = simulate.eval_coefficients(spec, coefficients, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(skims)

    # constrained coefficients can appear in expressions
    locals_dict.update(coefficients)

    assert ('in_period' not in choosers) and ('out_period' not in choosers)
    in_time = skims['in_time_col_name']
    out_time = skims['out_time_col_name']
    choosers['in_period'] = expressions.skim_time_period_label(choosers[in_time])
    choosers['out_period'] = expressions.skim_time_period_label(choosers[out_time])

    expressions.annotate_preprocessors(
        choosers, locals_dict, skims,
        model_settings, trace_label)

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
        estimator=estimator)

    return choices

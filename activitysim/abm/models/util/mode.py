# ActivitySim
# See full license in LICENSE.txt.
from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import pandas as pd

from activitysim.core import simulate
from activitysim.core import config
from activitysim.core.assign import evaluate_constants

from . import expressions


"""
At this time, these utilities are mostly for transforming the mode choice
spec, which is more complicated than the other specs, into something that
looks like the other specs.
"""


def tour_mode_choice_spec(model_settings):

    assert 'SPEC' in model_settings

    return simulate.read_model_spec(file_name=model_settings['SPEC'])


def tour_mode_choice_coeffecients_spec(model_settings):

    assert 'COEFFS' in model_settings
    coeffs_file_name = model_settings['COEFFS']

    file_path = config.config_file_path(coeffs_file_name)
    return pd.read_csv(file_path, comment='#', index_col='Expression')


def run_tour_mode_choice_simulate(
        choosers,
        spec, tour_purpose, model_settings,
        skims,
        constants,
        nest_spec,
        chunk_size,
        trace_label=None, trace_choice_name=None):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are tour/trip purposes).  Pass in the tours/trip that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    omnibus_coefficient_spec = tour_mode_choice_coeffecients_spec(model_settings)
    locals_dict = evaluate_constants(omnibus_coefficient_spec[tour_purpose], constants=constants)
    locals_dict.update(constants)
    locals_dict.update(skims)

    assert ('in_period' not in choosers) and ('out_period' not in choosers)
    in_time = skims['in_time_col_name']
    out_time = skims['out_time_col_name']
    choosers['in_period'] = expressions.skim_time_period_label(choosers[in_time])
    choosers['out_period'] = expressions.skim_time_period_label(choosers[out_time])

    expressions.annotate_preprocessors(
        choosers, locals_dict, skims,
        model_settings, trace_label)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=spec,
        nest_spec=nest_spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name)

    alts = spec.columns
    choices = choices.map(dict(list(zip(list(range(len(alts))), alts))))

    return choices

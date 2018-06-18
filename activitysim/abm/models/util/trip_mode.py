# ActivitySim
# See full license in LICENSE.txt.

import os
import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.util import assign_in_place

import expressions


def evaluate_constants(expressions, constants):
    """
    Evaluate a list of constant expressions - each one can depend on the one before
    it.  These are usually used for the coefficients which have relationships
    to each other.  So ivt=.7 and then ivt_lr=ivt*.9.

    Parameters
    ----------
    expressions : Series
        the index are the names of the expressions which are
        used in subsequent evals - thus naming the expressions is required.
    constants : dict
        will be passed as the scope of eval - usually a separate set of
        constants are passed in here

    Returns
    -------
    d : dict

    """

    # FIXME why copy?
    d = {}
    for k, v in expressions.iteritems():
        d[k] = eval(str(v), d.copy(), constants)

    return d


def trip_mode_choice_spec(model_settings):

    configs_dir = inject.get_injectable('configs_dir')

    assert 'SPEC' in model_settings
    return simulate.read_model_spec(configs_dir, model_settings['SPEC'])


def trip_mode_choice_coeffs(model_settings):

    configs_dir = inject.get_injectable('configs_dir')

    assert 'COEFFS' in model_settings
    with open(os.path.join(configs_dir, model_settings['COEFFS'])) as f:
        return pd.read_csv(f, comment='#', index_col='Expression')

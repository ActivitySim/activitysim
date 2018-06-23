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


def trip_mode_choice_spec(model_settings):

    configs_dir = inject.get_injectable('configs_dir')

    assert 'SPEC' in model_settings
    return simulate.read_model_spec(configs_dir, model_settings['SPEC'])


def trip_mode_choice_coeffecients_spec(model_settings):

    configs_dir = inject.get_injectable('configs_dir')

    assert 'COEFFS' in model_settings
    with open(os.path.join(configs_dir, model_settings['COEFFS'])) as f:
        return pd.read_csv(f, comment='#', index_col='Expression')

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt
from activitysim.core import assign
from activitysim.core import config

from .util import expressions
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)
DUMP = True


@inject.step()
def annotate_table(configs_dir):

    # model_settings name should have been provided as a step argument
    model_name = inject.get_step_arg('model_name')

    model_settings = config.read_model_settings(configs_dir, '%s.yaml' % model_name)

    df_name = model_settings['DF']
    df = inject.get_table(df_name).to_frame()

    results = expressions.compute_columns(
        df,
        model_settings=model_settings,
        configs_dir=configs_dir,
        trace_label=None)

    assign_in_place(df, results)

    pipeline.replace_table(df_name, df)

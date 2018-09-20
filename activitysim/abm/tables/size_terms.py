# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import config


logger = logging.getLogger(__name__)


@inject.table()
def size_terms():
    f = config.config_file_path('destination_choice_size_terms.csv')
    return pd.read_csv(f, comment='#', index_col='segment')

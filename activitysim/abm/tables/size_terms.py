# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import inject


logger = logging.getLogger(__name__)


@inject.table()
def size_terms(configs_dir):
    f = os.path.join(configs_dir, 'destination_choice_size_terms.csv')
    return pd.read_csv(f, comment='#', index_col='segment')

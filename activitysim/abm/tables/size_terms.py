# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging
import pandas as pd

from activitysim.core import inject
from activitysim.core import config


logger = logging.getLogger(__name__)


@inject.table()
def size_terms():
    f = config.config_file_path('destination_choice_size_terms.csv')
    return pd.read_csv(f, comment='#', index_col='segment')

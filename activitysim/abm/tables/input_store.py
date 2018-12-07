# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import os
import logging

import pandas as pd

from activitysim.core import config
from activitysim.core.config import setting

# FIXME
# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


def read_input_table(table_name):

    filename = setting('input_store', None)

    if not filename:
        logger.error("input store file name not specified in settings")
        raise RuntimeError("store file name not specified in settings")

    input_store_path = config.data_file_path(filename)

    if not os.path.exists(input_store_path):
        logger.error("store file not found: %s" % input_store_path)
        raise RuntimeError("store file not found: %s" % input_store_path)

    df = pd.read_hdf(input_store_path, table_name)

    return df

# ActivitySim
# See full license in LICENSE.txt.

import os
import warnings
import logging

import pandas as pd

from activitysim.core import inject
from activitysim.core.config import setting

# FIXME
# warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


def read_input_table(table_name):

    input_store_path = inject.get_injectable("input_store_path", None)

    if not input_store_path:

        filename = setting('input_store', None)

        if not filename:
            logger.error("input store file name not specified in settings")
            raise RuntimeError("store file name not specified in settings")

        data_dir = inject.get_injectable("data_dir")
        input_store_path = os.path.join(data_dir, filename)

    if not os.path.exists(input_store_path):
        logger.error("store file not found: %s" % input_store_path)
        raise RuntimeError("store file not found: %s" % input_store_path)

    df = pd.read_hdf(input_store_path, table_name)

    return df

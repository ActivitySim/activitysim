# ActivitySim
# See full license in LICENSE.txt.
import logging
import sys
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

from activitysim.core.config import setting

logger = logging.getLogger(__name__)


@inject.step()
def summarize(households_merged, tours_merged):
    """
        summarize is a standard model which uses expression files
        to reduce tables
        """
    trace_label = 'summarize'
    model_settings_file_name = 'summarize.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)
    pass

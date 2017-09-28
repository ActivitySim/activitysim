# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import numpy as np

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import pipeline

logger = logging.getLogger(__name__)


@inject.step()
def annotate_tours():

    pipeline.add_dependent_columns("tours", "tours_extras")

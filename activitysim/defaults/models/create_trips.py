# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim import tracing
from activitysim import pipeline

logger = logging.getLogger(__name__)


@orca.step()
def create_simple_trips():
    """
    Create a simple trip table
    """
    orca.get_injectable("trips_table")
    pipeline.get_rn_generator().add_channel(orca.get_table("trips"), 'trips')

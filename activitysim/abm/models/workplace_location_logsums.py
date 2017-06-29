# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config

from activitysim.core import pipeline

logger = logging.getLogger(__name__)


@orca.step()
def workplace_location_logsums(persons_merged,
                               skim_dict,
                               destination_size_terms,
                               chunk_size,
                               trace_hh_id):

    pass

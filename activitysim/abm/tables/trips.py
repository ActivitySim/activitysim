# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

from activitysim.abm.tables.util import simple_table_join
from activitysim.core import workflow

logger = logging.getLogger(__name__)


@workflow.temp_table
def trips_merged(state: workflow.State, trips, tours):
    return simple_table_join(trips, tours, "tour_id")

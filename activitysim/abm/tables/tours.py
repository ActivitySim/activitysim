# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.tables.util import simple_table_join
from activitysim.core import workflow

logger = logging.getLogger(__name__)


@workflow.temp_table
def tours_merged(
    state: workflow.State, tours: pd.DataFrame, persons_merged: pd.DataFrame
):
    return simple_table_join(
        tours,
        persons_merged,
        left_on="person_id",
    )

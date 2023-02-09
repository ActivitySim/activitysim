# ActivitySim
# See full license in LICENSE.txt.
import logging
import os

import numpy as np
import pandas as pd

from ...core import config, inject
from ...core import timetable as tt
from ...core.pipeline import Whale
from ...core.workflow import workflow_cached_object, workflow_table

logger = logging.getLogger(__name__)


@workflow_cached_object
def tdd_alts(whale) -> pd.DataFrame:
    # right now this file just contains the start and end hour
    file_path = whale.filesystem.get_config_file_path(
        "tour_departure_and_duration_alternatives.csv"
    )
    df = pd.read_csv(file_path)

    df["duration"] = df.end - df.start

    # - NARROW
    df = df.astype(np.int8)

    return df


@workflow_cached_object
def tdd_alt_segments(whale: Whale) -> pd.DataFrame:
    # tour_purpose,time_period,start,end
    # work,EA,3,5
    # work,AM,6,8
    # ...
    # school,PM,15,17
    # school,EV,18,22

    file_path = whale.filesystem.get_config_file_path(
        "tour_departure_and_duration_segments.csv", mandatory=False
    )

    if file_path:
        df = pd.read_csv(file_path, comment="#")

        # - NARROW
        df["start"] = df["start"].astype(np.int8)
        df["end"] = df["end"].astype(np.int8)

    else:
        df = None

    return df


@workflow_table
def person_windows(
    whale: Whale,
    persons: pd.DataFrame,
    tdd_alts: pd.DataFrame,
) -> pd.DataFrame:
    df = tt.create_timetable_windows(persons, tdd_alts)

    return df


@inject.injectable()
def timetable(person_windows, tdd_alts):
    logging.debug("@inject timetable")
    return tt.TimeTable(person_windows.to_frame(), tdd_alts, person_windows.name)

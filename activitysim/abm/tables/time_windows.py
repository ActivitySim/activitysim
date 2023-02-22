# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import timetable as tt
from activitysim.core import workflow

logger = logging.getLogger(__name__)


@workflow.cached_object
def tdd_alts(state: workflow.State) -> pd.DataFrame:
    # right now this file just contains the start and end hour
    file_path = state.filesystem.get_config_file_path(
        "tour_departure_and_duration_alternatives.csv"
    )
    df = pd.read_csv(file_path)

    df["duration"] = df.end - df.start

    # - NARROW
    df = df.astype(np.int8)

    return df


@workflow.cached_object
def tdd_alt_segments(state: workflow.State) -> pd.DataFrame:
    # tour_purpose,time_period,start,end
    # work,EA,3,5
    # work,AM,6,8
    # ...
    # school,PM,15,17
    # school,EV,18,22

    file_path = state.filesystem.get_config_file_path(
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


@workflow.table
def person_windows(
    state: workflow.State,
    persons: pd.DataFrame,
    tdd_alts: pd.DataFrame,
) -> pd.DataFrame:
    df = tt.create_timetable_windows(persons, tdd_alts)

    return df


@workflow.cached_object
def timetable(
    state: workflow.State, person_windows: pd.DataFrame, tdd_alts: pd.DataFrame
) -> tt.TimeTable:
    logging.debug("@workflow.cached_object timetable")
    return tt.TimeTable(person_windows, tdd_alts, "person_windows")

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.tables.util import simple_table_join
from activitysim.core import workflow

logger = logging.getLogger(__name__)


@workflow.table
def vehicles(state: workflow.State, households: pd.DataFrame):
    """Creates the vehicles table and load it as an injectable

    This method initializes the `vehicles` table, where the number of rows
    is equal to the sum of `households["auto_ownership"]`.

    Parameters
    ----------
    households : DataFrame

    Returns
    -------
    vehicles : pandas.DataFrame
    """

    # initialize vehicles table
    vehicles = households.loc[households.index.repeat(households["auto_ownership"])]
    vehicles = vehicles.reset_index()[["household_id"]]

    vehicles["vehicle_num"] = vehicles.groupby("household_id").cumcount() + 1
    # tying the vehicle id to the household id in order to ensure reproducability
    vehicles["vehicle_id"] = vehicles.household_id * 10 + vehicles.vehicle_num
    vehicles.set_index("vehicle_id", inplace=True)

    # replace table function with dataframe
    state.add_table("vehicles", vehicles)

    state.get_rn_generator().add_channel("vehicles", vehicles)
    state.tracing.register_traceable_table("vehicles", vehicles)

    return vehicles


@workflow.temp_table
def vehicles_merged(
    state: workflow.State, vehicles: pd.DataFrame, households_merged: pd.DataFrame
):
    """Augments the vehicles table with household attributes

    Parameters
    ----------
    vehicles :  DataFrame
    households_merged :  DataFrame

    Returns
    -------
    vehicles_merged : pandas.DataFrame
    """
    return simple_table_join(vehicles, households_merged, "household_id")

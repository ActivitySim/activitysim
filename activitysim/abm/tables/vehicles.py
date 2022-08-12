# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import inject, pipeline, tracing

logger = logging.getLogger(__name__)


@inject.table()
def vehicles(households):
    """Creates the vehicles table and load it as an injectable

    This method initializes the `vehicles` table, where the number of rows
    is equal to the sum of `households["auto_ownership"]`.

    Parameters
    ----------
    households :  orca.DataFrameWrapper

    Returns
    -------
    vehicles : pandas.DataFrame
    """

    # initialize vehicles table
    vehicles = households.to_frame().loc[
        households.index.repeat(households["auto_ownership"])
    ]
    vehicles = vehicles.reset_index()[["household_id"]]

    vehicles["vehicle_num"] = vehicles.groupby("household_id").cumcount() + 1
    # tying the vehicle id to the household id in order to ensure reproducability
    vehicles["vehicle_id"] = vehicles.household_id * 10 + vehicles.vehicle_num
    vehicles.set_index("vehicle_id", inplace=True)

    # replace table function with dataframe
    inject.add_table("vehicles", vehicles)

    pipeline.get_rn_generator().add_channel("vehicles", vehicles)
    tracing.register_traceable_table("households", vehicles)

    return vehicles


@inject.table()
def vehicles_merged(vehicles, households_merged):
    """Augments the vehicles table with household attributes

    Parameters
    ----------
    vehicles :  orca.DataFrameWrapper
    households_merged :  orca.DataFrameWrapper

    Returns
    -------
    vehicles_merged : pandas.DataFrame
    """

    vehicles_merged = inject.merge_tables(
        vehicles.name, tables=[vehicles, households_merged]
    )
    return vehicles_merged


inject.broadcast(
    "households_merged", "vehicles", cast_index=True, onto_on="household_id"
)

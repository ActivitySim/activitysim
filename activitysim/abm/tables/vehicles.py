# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import inject, pipeline, tracing

logger = logging.getLogger(__name__)


@inject.table()
def vehicles(households):

    # initialize vehicles table
    vehicles = households.to_frame().loc[
        households.index.repeat(households['auto_ownership'])]
    vehicles = vehicles.reset_index()[['household_id']]
    vehicles.index.name = 'vehicle_id'
    vehicle_cols = ['vehicle_type', 'primary_person_id']
    
    for col in vehicle_cols:
        vehicles.loc[:, col] = None

    # I do not understand why this line is necessary, it seems circular
    # to inject the vehicles table in the inside the table definition
    # that injects it, but without it I found that it failed the assert
    # statement at random.py L144. This appears to be how its done for 
    # the persons, households tables as well.
    inject.add_table('vehicles', vehicles)
    
    pipeline.get_rn_generator().add_channel('vehicles', vehicles)
    tracing.register_traceable_table('households', vehicles)

    return vehicles


@inject.table()
def vehicles_merged(vehicles, households_merged):

    vehicles_merged = inject.merge_tables(
        vehicles.name, tables=[vehicles, households_merged])
    return vehicles_merged


inject.broadcast('households_merged', 'vehicles', cast_index=True, onto_on='household_id')

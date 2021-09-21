# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import inject, pipeline, tracing

logger = logging.getLogger(__name__)


@inject.table()
def vehicles(households):

	# initialize vehicles tables
    vehicles = households.to_frame().loc[
        households.index.repeat(households['auto_ownership'])]
    vehicles = vehicles.reset_index()[['household_id']]
    vehicles.index.name = 'vehicle_id'
    vehicle_cols = ['vehicle_type', 'primary_person_id']
    
    for col in vehicle_cols:
        vehicles.loc[:, col] = None

    return vehicles


inject.broadcast('households_merged', 'vehicles', cast_index=True, onto_on='household_id')

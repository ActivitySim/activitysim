# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import inject

logger = logging.getLogger(__name__)


@inject.table()
def trips_merged(trips, tours):
    return inject.merge_tables(trips.name, tables=[trips, tours])


inject.broadcast("tours", "trips", cast_index=True, onto_on="tour_id")

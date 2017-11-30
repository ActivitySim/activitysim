# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core.util import reindex
from activitysim.core import inject

logger = logging.getLogger(__name__)


@inject.table()
def tours_merged(tours, persons_merged):
    return inject.merge_tables(tours.name, tables=[
        tours, persons_merged])


inject.broadcast('persons_merged', 'tours', cast_index=True, onto_on='person_id')


# # this is the placeholder for all the columns to update after tour destinations are assigned
# @inject.table()
# def tours_with_dest(tours):
#     return pd.DataFrame(index=tours.index)
#
#
# @inject.column("tours_with_dest")
# def destination_in_cbd(tours, land_use, settings):
#     # protection until filled in by destination choice model
#     assert "destination" in tours.columns
#
#     s = reindex(land_use.area_type, tours.destination)
#     return s < settings['cbd_threshold']


# # this is the placeholder for all the columns to update after destination choice and scheduling
# @inject.table()
# def tours_extras(tours):
#     return pd.DataFrame(index=tours.index)
#
#
# @inject.column("tours_extras")
# def sov_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def hov2_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def hov2toll_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def hov3_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def sovtoll_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def drive_local_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def drive_lrf_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def drive_express_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def drive_heavyrail_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def drive_commuter_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def walk_local_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def walk_lrf_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def walk_commuter_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def walk_express_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def walk_heavyrail_available(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def is_joint(tours):
#     # FIXME
#     return pd.Series(False, index=tours.index)
#
#
# @inject.column("tours_extras")
# def is_not_joint(tours):
#     # FIXME
#     return pd.Series(True, index=tours.index)
#
#
# @inject.column("tours_extras")
# def number_of_participants(tours):
#     # FIXME
#     return pd.Series(1, index=tours.index)
#
#
# @inject.column("tours_extras")
# def work_tour_is_drive(tours):
#     # FIXME
#     # FIXME note that there's something about whether this is a subtour?
#     # FIXME though I'm not sure how it can be a subtour in the tour mode choice
#     return pd.Series(0, index=tours.index)
#
#
# @inject.column("tours_extras")
# def terminal_time(tours):
#     # FIXME
#     return pd.Series(0, index=tours.index)
#
#
# @inject.column("tours_extras")
# def origin_walk_time(tours):
#     # FIXME
#     return pd.Series(0, index=tours.index)
#
#
# @inject.column("tours_extras")
# def destination_walk_time(tours):
#     # FIXME
#     return pd.Series(0, index=tours.index)
#
#
# @inject.column("tours_extras")
# def daily_parking_cost(tours):
#     # FIXME - this is a computation based on the tour destination
#     return pd.Series(0, index=tours.index)
#
#
# @inject.column("tours_extras")
# def dest_density_index(tours, land_use):
#     return reindex(land_use.density_index,
#                    tours.destination)
#
#
# @inject.column("tours_extras")
# def dest_topology(tours, land_use):
#     return reindex(land_use.TOPOLOGY,
#                    tours.destination)
#
#
# @inject.column("tours_extras")
# def out_period(tours, settings):
#     cats = pd.cut(tours.end,
#                   settings['skim_time_periods']['hours'],
#                   labels=settings['skim_time_periods']['labels'])
#     # cut returns labelled categories but we convert to str
#     return cats.astype(str)
#
#
# @inject.column("tours_extras")
# def in_period(tours, settings):
#     cats = pd.cut(tours.start,
#                   settings['skim_time_periods']['hours'],
#                   labels=settings['skim_time_periods']['labels'])
#     # cut returns labelled categories but we convert to str
#     return cats.astype(str)

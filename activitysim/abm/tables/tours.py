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


# @inject.table()
# def joint_tours_merged(tours, households):
#     return inject.merge_tables(joint_tours_merged.name, tables=[
#         tours, households])
#
#
# inject.broadcast('households', 'joint_tours', cast_index=True, onto_on='household_id')


@inject.table()
def participants_merged(joint_tour_participants, joint_tours):
    return inject.merge_tables(joint_tour_participants.name, tables=[
        joint_tour_participants, joint_tours])


inject.broadcast('joint_tours', 'joint_tour_participants', cast_index=True, onto_on='tour_id')

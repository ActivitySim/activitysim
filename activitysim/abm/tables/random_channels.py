# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import inject


logger = logging.getLogger(__name__)


CHANNEL_INFO = {
    'households': {
        'max_steps': 3,
        'index': 'HHID'
    },
    'persons': {
        'max_steps': 8,
        'index': 'PERID'
    },
    'tours': {
        'max_steps': 15,
        'index': 'tour_id'
    },
    'joint_tour_participants': {
        'max_steps': 1,
        'index': 'participant_id'
    },
    'trips': {
        'max_steps': 1,
        'index': 'trip_id'
    },
}


@inject.injectable()
def channel_info():

    return CHANNEL_INFO

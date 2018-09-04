# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import inject


logger = logging.getLogger(__name__)

"""
When the pipeline is restarted and tables are loaded, we need to know which ones
should be registered as random number generator channels.
"""

RANDOM_CHANNELS = ['households', 'persons', 'tours', 'joint_tour_participants', 'trips']
TRACEABLE_TABLES = ['households', 'persons', 'tours', 'joint_tour_participants', 'trips']


@inject.injectable()
def rng_channels():

    # bug
    return RANDOM_CHANNELS


@inject.injectable()
def traceable_tables():

    # names of all traceable tables ordered by dependency on household_id
    # e.g. 'persons' has to be registered AFTER 'households'

    return TRACEABLE_TABLES

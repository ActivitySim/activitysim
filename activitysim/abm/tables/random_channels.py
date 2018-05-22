# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import inject


logger = logging.getLogger(__name__)

"""
We expect that the random number channel can be determined by the name of the index of the
dataframe accompanying the request.

channel_info is a dict with keys and value of the form:

::

    <channel_name>: <table_index_name>


channel_name: str
    The channel name is just the table name used by the pipeline and inject.
table_index_name: str
    name of the table index (so we can deduce the channel for a dataframe by index name)

"""

CHANNEL_INFO = {
    'households': 'HHID',
    'persons': 'PERID',
    'tours': 'tour_id',
    'joint_tour_participants': 'participant_id',
    'trips': 'trip_id',
}


@inject.injectable()
def channel_info():

    return CHANNEL_INFO

# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging
from collections import OrderedDict

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


@inject.injectable()
def traceable_table_indexes():
    # traceable_table_indexes is OrderedDict {<index_name>: <table_name>}
    # so we can find first registered table to slice by ref_col
    return OrderedDict()


@inject.injectable()
def traceable_table_ids():
    # traceable_table_ids is dict {<table_name>: [<id>, <id>]}
    return dict()

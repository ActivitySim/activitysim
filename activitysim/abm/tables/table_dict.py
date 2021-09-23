# ActivitySim
# See full license in LICENSE.txt.
import logging
from collections import OrderedDict

from activitysim.abm.models.util import canonical_ids as cid
from activitysim.core import inject

logger = logging.getLogger(__name__)

"""
When the pipeline is restarted and tables are loaded, we need to know which ones
should be registered as random number generator channels.
"""


@inject.injectable()
def rng_channels():

    return cid.RANDOM_CHANNELS


@inject.injectable()
def traceable_tables():

    # names of all traceable tables ordered by dependency on household_id
    # e.g. 'persons' has to be registered AFTER 'households'

    return cid.TRACEABLE_TABLES


@inject.injectable()
def traceable_table_indexes():
    # traceable_table_indexes is OrderedDict {<index_name>: <table_name>}
    # so we can find first registered table to slice by ref_col
    return OrderedDict()


@inject.injectable()
def traceable_table_ids():
    # traceable_table_ids is dict {<table_name>: [<id>, <id>]}
    return dict()


@inject.injectable()
def canonical_table_index_names():
    # traceable_table_ids is dict {<table_name>: [<id>, <id>]}
    return cid.CANONICAL_TABLE_INDEX_NAMES

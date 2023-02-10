# ActivitySim
# See full license in LICENSE.txt.
import logging
from collections import OrderedDict

from activitysim.abm.models.util import canonical_ids as cid
from activitysim.core import inject, workflow

logger = logging.getLogger(__name__)

"""
When the pipeline is restarted and tables are loaded, we need to know which ones
should be registered as random number generator channels.
"""


@workflow.cached_object
def rng_channels(whale: workflow.Whale):

    return cid.RANDOM_CHANNELS


@workflow.cached_object
def traceable_tables(whale: workflow.Whale):

    # names of all traceable tables ordered by dependency on household_id
    # e.g. 'persons' has to be registered AFTER 'households'

    return cid.TRACEABLE_TABLES


@workflow.cached_object
def traceable_table_indexes(whale: workflow.Whale):
    # traceable_table_indexes is OrderedDict {<index_name>: <table_name>}
    # so we can find first registered table to slice by ref_col
    return OrderedDict()


@workflow.cached_object
def traceable_table_ids(whale: workflow.Whale):
    # traceable_table_ids is dict {<table_name>: [<id>, <id>]}
    return dict()


@workflow.cached_object
def canonical_table_index_names(whale: workflow.Whale):
    # traceable_table_ids is dict {<table_name>: [<id>, <id>]}
    return cid.CANONICAL_TABLE_INDEX_NAMES

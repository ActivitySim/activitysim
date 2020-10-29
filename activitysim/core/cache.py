# ActivitySim
# See full license in LICENSE.txt.
from builtins import range

import logging
import time
import math
import os



import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import config
from activitysim.core import chunk
from activitysim.core import logit
from activitysim.core import simulate
from activitysim.core import los

from activitysim.core.los import memo

logger = logging.getLogger(__name__)


class TableCache(object):
    def __init__(self, network_los):

        #self.network_los = network_los
        self.cache_dir = network_los.get_cache_dir()
        self.read_cache =network_los.setting('read_tvpb_cache', False)
        self.write_cache = network_los.setting('write_tvpb_cache', False)
        self.caches = {}

    def get_cache_path(self, cache_tag):
        cache_path = os.path.join(self.cache_dir, f'{cache_tag}.arrow')
        return cache_path

    def close(self):
        for tag in list(self.caches.keys()):  # iterate list because close_cache pops key
            self.close_cache(tag)
        assert not self.caches  # should be empty

    def open_cache(self, cache_tag):

        assert self.caches.get(cache_tag) == None

        cache = {'changed': False}

        if self.read_cache:
            cache_path = self.get_cache_path(cache_tag)

            if os.path.isfile(cache_path):

                with memo(f'open_cache {cache_tag}'):
                    # table = pa.ipc.RecordBatchFileReader(pa.memory_map(cache_path, 'r')).read_all()
                    # df = table.to_pandas(split_blocks=True, self_destruct=True)
                    # del table
                    #
                    df = pa.feather.read_feather(cache_path, columns=None, use_threads=True, memory_map=True)

                cache['df'] = df
                logger.debug(f"open_cache read {cache['df'].shape} table {cache_tag} from {cache_path}")

                #print(df)
                #bug
            else:
                logger.warning(f"open_cache file for {cache_tag} not found: {cache_path}")


        self.caches[cache_tag] = cache

        return cache

    def close_cache(self, cache_tag):

        assert cache_tag in self.caches

        cache = self.caches[cache_tag]

        if self.write_cache:

            if cache['changed']:
                cache_path = self.get_cache_path(cache_tag)
                df = cache['df']

                with memo(f'close_cache {cache_tag}'):

                    pa.feather.write_feather(df, cache_path, compression=None, compression_level=None, chunksize=None, version=2)

                    # table = pa.Table.from_pandas(df)
                    # with pa.OSFile(cache_path, 'wb') as sink:
                    #     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    #         writer.write_table(table)
                    # del table


                logger.debug(f"wrote cache table {cache_tag} ({cache['df'].shape}) to {cache_path}")

            else:
                logger.debug(f"not writing {cache_tag} table to cache since unchanged.")

            #utility_cols = [c for c in self.utility_cache.columns if c not in ['btap', 'atap', 'tod', 'demographic_segment']]
            #print(utility_cols)
            #num_unavailable = (self.utility_cache[utility_cols] < -998).all(axis=1).sum()
            #percent_unavailable = round(100*num_unavailable/len(self.utility_cache) if num_unavailable>0 else 0, 2)
            #logger.debug(f"close_utility_cache {percent_unavailable}% ({num_unavailable}) tap pairs unavailable")

        self.caches.pop(cache_tag)

    def get_cached_table(self, cache_tag):

        if not cache_tag in self.caches:
            logger.debug(f"TableCache.get_cached_table opening {cache_tag}")
            self.open_cache(cache_tag)

        assert cache_tag in self.caches
        return self.caches[cache_tag].get('df')

    def extend_cached_table(self, cache_tag, new_rows):

        assert cache_tag in self.caches
        assert len(new_rows) > 0

        # local reference for legibility, but any changes write through
        cache = self.caches.get(cache_tag, {})

        cache['changed'] = True

        if cache.get('df') is None:
            cache['df'] = new_rows
        else:
            cache['df'] = pd.concat([cache['df'], new_rows], axis=0)

        assert not cache['df'].duplicated().any()

        elements = np.prod(cache['df'].shape, dtype=np.int64)
        logger.debug(f"#UCACHE extended {cache_tag} cache by {len(new_rows)} rows "
                     f"to {cache['df'].shape} ({elements} elements)")

        self.caches[cache_tag] = cache  # redundant, since changes to local reference write through

# ActivitySim
# See full license in LICENSE.txt.
# from builtins import int

import sys
import os
import logging
import multiprocessing

# from collections import OrderedDict

import yaml
import numpy as np
import pandas as pd
import openmatrix as omx
import pyarrow as pa

from activitysim.core import util
from activitysim.core import config

from activitysim.core import inject
from activitysim.core import skim

from activitysim.core import skim_omx
from activitysim.core.tracing import memo

logger = logging.getLogger(__name__)


JIT_MMAP = True

class JitMemMapSkimData(skim.SkimData):

    def __init__(self, skim_cache_path, skim_info):
        self.skim_cache_path = skim_cache_path
        self.dtype = np.dtype(skim_info['dtype_name'])
        self._shape = skim_info['skim_data_shape']

    def __getitem__(self, indexes):
        assert len(indexes) == 3, f'number of indexes ({len(indexes)}) should be 3'
        # open memmap
        data = np.memmap(self.skim_cache_path, shape=self._shape, dtype=self.dtype, mode='r')
        # dereference skim values
        result = data[indexes]
        # closing memmap's underlying mmap frees data read into (not really needed as we are exiting scope)
        data._mmap.close()
        return result

    @property
    def shape(self):
        return self._shape

    def close(self):
        pass


class MemMapSkimData(skim.SkimData):

    def __init__(self, skim_cache_path, skim_info):
        self.skim_cache_path = skim_cache_path
        self.dtype = np.dtype(skim_info['dtype_name'])
        self._shape = skim_info['skim_data_shape']

    def close(self):
        self._skim_data._mmap.close()
        del self._skim_data

class MemMapSkimFactory(skim_omx.AbstractSkimFactory):

    def __init__(self, network_los):
        super().__init__(network_los)

    def get_skim_data(self, skim_tag, skim_info):

        logger.debug(f"create_skim_dict {skim_tag} creating OMX_SKIM")

        # don't expect legacy shared memory buffers
        assert inject.get_injectable('data_buffers', None) is None

        skim_cache_path = self.memmap_skim_data_path(skim_tag)
        if not os.path.isfile(skim_cache_path):
            self.copy_omx_to_mmap_file(skim_info)

        if JIT_MMAP:
            skim_data = JitMemMapSkimData(skim_cache_path, skim_info)
        else:
            skim_data = self.open_existing_readonly_memmap_skim_cache(skim_info)
            skim_data = skim.SkimData(skim_data)

        logger.info(f"get_skim_data {skim_tag} SkimData shape {skim_data.shape}")

        return skim_data


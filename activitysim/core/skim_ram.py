# ActivitySim
# See full license in LICENSE.txt.
# from builtins import int

import sys
import os
import logging
import multiprocessing

# from collections import OrderedDict

import numpy as np
import pandas as pd
import openmatrix as omx
import pyarrow as pa

from activitysim.core import util
from activitysim.core import config

from activitysim.core import inject
from activitysim.core import skim

from activitysim.core import skim_omx

logger = logging.getLogger(__name__)


class NumpyArraySkimFactory(skim_omx.AbstractSkimFactory):

    def __init__(self, network_los):
        super().__init__(network_los)

    @property
    def share_data_for_multiprocessing(self):
        return True

    def allocate_skim_buffer(self, skim_info, shared=False):

        dtype_name = skim_info['dtype_name']
        dtype = np.dtype(dtype_name)
        skim_data_shape = skim_info['skim_data_shape']
        skim_tag = skim_info['skim_tag']

        # buffer_size must be int, not np.int64
        buffer_size = int(np.prod(skim_data_shape))

        csz = buffer_size * dtype.itemsize
        logger.info(f"allocating shared buffer {skim_tag} shape {skim_data_shape} total size: {csz} ({util.GB(csz)})")

        if shared:
            if dtype_name == 'float64':
                typecode = 'd'
            elif dtype_name == 'float32':
                typecode = 'f'
            else:
                raise RuntimeError("allocate_skim_buffer unrecognized dtype %s" % dtype_name)

            buffer = multiprocessing.RawArray(typecode, buffer_size)
        else:
            buffer = np.zeros(buffer_size, dtype=dtype)

        return buffer

    def skim_data_from_buffer(self, skim_info, skim_buffer):

        skim_data_shape = skim_info['skim_data_shape']
        dtype = np.dtype(skim_info['dtype_name'])
        assert len(skim_buffer) == np.prod(skim_data_shape)
        skim_data = np.frombuffer(skim_buffer, dtype=dtype).reshape(skim_data_shape)
        return skim_data

    def load_skims_to_buffer(self, skim_info, skim_buffer):

        read_cache = self.network_los.setting('read_skim_cache', False)
        write_cache = self.network_los.setting('write_skim_cache', False)

        skim_data = self.skim_data_from_buffer(skim_info, skim_buffer)
        assert skim_data.shape == skim_info['skim_data_shape']

        if read_cache:
            # returns None if cache file not found
            cache_data = self.open_existing_readonly_memmap_skim_cache(skim_info)

            # copy memmapped cache to RAM numpy ndarray
            if cache_data is not None:
                assert cache_data.shape == skim_data.shape
                skim_data[::] = cache_data[::]
                cache_data._mmap.close()
                del cache_data
                return

        # read omx skims into skim_buffer (np array)
        self.read_skims_from_omx(skim_info, skim_data)

        if write_cache:
            cache_data = self.create_empty_writable_memmap_skim_cache(skim_info)
            cache_data[::] = skim_data[::]
            cache_data._mmap.close()
            del cache_data

            # bug - do we need to close it?

        logger.info(f"load_skims_to_buffer {skim_info['skim_tag']} shape {skim_data.shape}")

    def get_skim_data(self, skim_tag, skim_info):

        logger.debug(f"create_skim_dict {skim_tag} creating OMX_SKIM")

        data_buffers = inject.get_injectable('data_buffers', None)
        if data_buffers:
            # we assume any existing skim buffers will already have skim data loaded into them
            logger.info(f"create_skim_dict {skim_tag} using existing skim_buffers for skims")
            skim_buffer = data_buffers[skim_tag]
        else:
            skim_buffer = self.allocate_skim_buffer(skim_info, shared=False)
            self.load_skims_to_buffer(skim_info, skim_buffer)

        skim_data = skim.SkimData(self.skim_data_from_buffer(skim_info, skim_buffer))
        return skim_data


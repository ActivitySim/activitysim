# ActivitySim
# See full license in LICENSE.txt.
import gc as _gc
import itertools
import logging
import multiprocessing
import os
import time
from builtins import range
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil

from activitysim.core import config, inject, simulate, tracing, util

logger = logging.getLogger(__name__)

RAWARRAY = False
DTYPE_NAME = "float32"
RESCALE = 1000

DYNAMIC = "dynamic"
STATIC = "static"
TRACE = "trace"

MEMO_STACK = []


@contextmanager
def memo(tag, console=False, disable_gc=True):
    t0 = time.time()

    MEMO_STACK.append(tag)

    gc_was_enabled = _gc.isenabled()
    if gc_was_enabled:
        _gc.collect()
        if disable_gc:
            _gc.disable()

    previous_mem = psutil.Process().memory_info().rss
    try:
        yield
    finally:
        elapsed_time = time.time() - t0

        current_mem = psutil.Process().memory_info().rss
        marginal_mem = current_mem - previous_mem
        mem_str = f"net {util.GB(marginal_mem)} ({util.INT(marginal_mem)}) total {util.GB(current_mem)}"

        if gc_was_enabled and disable_gc:
            _gc.enable()
        if _gc.isenabled():
            _gc.collect()

        if console:
            print(f"MEMO {tag} Time: {util.SEC(elapsed_time)} Memory: {mem_str} ")
        else:
            logger.debug(f"MEM  {tag} {mem_str} in {util.SEC(elapsed_time)}")

        MEMO_STACK.pop()


class TVPBCache(object):
    """
    Transit virtual path builder cache for three zone systems
    """

    def __init__(self, network_los, uid_calculator, cache_tag):

        # lightweight until opened

        self.cache_tag = cache_tag

        self.network_los = network_los
        self.uid_calculator = uid_calculator

        self.is_open = False
        self.is_changed = False
        self._data = None

    @property
    def cache_path(self):
        file_type = "mmap"
        return os.path.join(config.get_cache_dir(), f"{self.cache_tag}.{file_type}")

    @property
    def csv_trace_path(self):
        file_type = "csv"
        return os.path.join(config.get_cache_dir(), f"{self.cache_tag}.{file_type}")

    def cleanup(self):
        """
        Called prior to
        """
        if os.path.isfile(self.cache_path):
            logger.debug(f"deleting cache {self.cache_path}")
            os.unlink(self.cache_path)

    def write_static_cache(self, data):

        assert not self.is_open
        assert self._data is None
        assert not self.is_changed

        data = data.reshape(self.uid_calculator.fully_populated_shape)

        # np.savetxt(self.csv_trace_path, data, fmt='%.18e', delimiter=',')

        logger.debug(f"#TVPB CACHE write_static_cache df {data.shape}")

        mm_data = np.memmap(
            self.cache_path, shape=data.shape, dtype=DTYPE_NAME, mode="w+"
        )
        np.copyto(mm_data, data)
        mm_data._mmap.close()
        del mm_data

        logger.debug(
            f"#TVPB CACHE write_static_cache wrote static cache table "
            f"({data.shape}) to {self.cache_path}"
        )

    def open(self):
        """
        open STATIC cache and populate with cached data

        if multiprocessing
            always STATIC cache with data fully_populated preloaded shared data buffer
        """
        # MMAP only supported for fully_populated_uids (STATIC)
        # otherwise we would have to store uid index as float, which has roundoff issues for float32

        assert not self.is_open, f"TVPBCache open called but already open"
        self.is_open = True

        if self.network_los.multiprocess():
            # multiprocessing usex preloaded fully_populated shared data buffer
            with memo("TVPBCache.open get_data_and_lock_from_buffers"):
                data, _ = self.get_data_and_lock_from_buffers()
            logger.info(
                f"TVPBCache.open {self.cache_tag} STATIC cache using existing data_buffers"
            )
        elif os.path.isfile(self.cache_path):
            # single process ought have created a precomputed fully_populated STATIC file
            data = np.memmap(self.cache_path, dtype=DTYPE_NAME, mode="r")

            # FIXME - why leave memmap open - maybe should copy since it will be read into memory when accessed anyway
            # mm_data = np.memmap(self.cache_path, dtype=DTYPE_NAME, mode='r')
            # data = np.empty_like(mm_data)
            # np.copyto(data, mm_data)
            # mm_data._mmap.close()
            # del mm_data

            logger.info(
                f"TVPBCache.open {self.cache_tag} read fully_populated data array from mmap file"
            )
        else:
            raise RuntimeError(
                f"Pathbuilder cache not found. Did you forget to run initialize tvpb?"
                f"Expected cache file: {self.cache_path}"
            )

        # create no-copy pandas DataFrame from numpy wrapped RawArray or Memmap buffer
        column_names = self.uid_calculator.set_names
        with memo("TVPBCache.open data.reshape"):
            data = data.reshape(
                (-1, len(column_names))
            )  # reshape so there is one column per set

        # data should be fully_populated and in canonical order - so we can assign canonical uid index
        with memo("TVPBCache.open uid_calculator.fully_populated_uids"):
            fully_populated_uids = self.uid_calculator.fully_populated_uids

        # check fully_populated, but we have to take order on faith (internal error if it is not)
        assert data.shape[0] == len(fully_populated_uids)

        self._data = data
        logger.debug(f"TVPBCache.open initialized STATIC cache table")

    def close(self, trace=False):
        """
        write any changes, free data, and mark as closed
        """

        assert self.is_open, f"TVPBCache close called but not open"

        self.is_open = False
        self._data = None
        self.cache_type = None

    @property
    def data(self):
        assert self._data is not None
        return self._data

    def allocate_data_buffer(self, shared=False):
        """
        allocate fully_populated_shape data buffer for cached data

        if shared, return a multiprocessing.Array that can be shared across subprocesses
        if not shared, return a numpy ndarrray

        Parameters
        ----------
        shared: boolean

        Returns
        -------
            multiprocessing.Array or numpy ndarray sized to hole fully_populated utility array
        """

        assert not self.is_open
        assert shared == self.network_los.multiprocess()

        dtype_name = DTYPE_NAME
        dtype = np.dtype(DTYPE_NAME)

        # multiprocessing.Array argument buffer_size must be int, not np.int64
        shape = self.uid_calculator.fully_populated_shape
        buffer_size = util.iprod(self.uid_calculator.fully_populated_shape)

        csz = buffer_size * dtype.itemsize
        logger.info(
            f"TVPBCache.allocate_data_buffer allocating data buffer "
            f"shape {shape} buffer_size {util.INT(buffer_size)} total size: {util.INT(csz)} ({util.GB(csz)})"
        )

        if shared:
            if dtype_name == "float64":
                typecode = "d"
            elif dtype_name == "float32":
                typecode = "f"
            else:
                raise RuntimeError(
                    "allocate_data_buffer unrecognized dtype %s" % dtype_name
                )

            if RAWARRAY:
                with memo("TVPBCache.allocate_data_buffer allocate RawArray"):
                    buffer = multiprocessing.RawArray(typecode, buffer_size)
                logger.info(
                    f"TVPBCache.allocate_data_buffer allocated shared multiprocessing.RawArray as buffer"
                )
            else:
                with memo("TVPBCache.allocate_data_buffer allocate Array"):
                    buffer = multiprocessing.Array(typecode, buffer_size)
                logger.info(
                    f"TVPBCache.allocate_data_buffer allocated shared multiprocessing.Array as buffer"
                )

        else:
            buffer = np.empty(buffer_size, dtype=dtype)
            np.copyto(buffer, np.nan)  # fill with np.nan

            logger.info(
                f"TVPBCache.allocate_data_buffer allocating non-shared numpy array as buffer"
            )

        return buffer

    def load_data_to_buffer(self, data_buffer):
        # 1) we are called before initialize_los, there is a saved cache, and it will be honored
        # 2) we are called before initialize_los and there is no saved cache yet
        # 3) we are resuming after initialize_los and so there must be a saved cache

        assert not self.is_open

        # wrap multiprocessing.Array (or RawArray) as a numpy array
        with memo("TVPBCache.load_data_to_buffer frombuffer"):
            if RAWARRAY:
                np_wrapped_data_buffer = np.ctypeslib.as_array(data_buffer)
            else:
                np_wrapped_data_buffer = np.ctypeslib.as_array(data_buffer.get_obj())

        if os.path.isfile(self.cache_path):
            with memo("TVPBCache.load_data_to_buffer copy memmap"):
                data = np.memmap(self.cache_path, dtype=DTYPE_NAME, mode="r")
                np.copyto(np_wrapped_data_buffer, data)
                data._mmap.close()
                del data
            logger.debug(
                f"TVPBCache.load_data_to_buffer loaded data from {self.cache_path}"
            )
        else:
            np.copyto(np_wrapped_data_buffer, np.nan)
            logger.debug(f"TVPBCache.load_data_to_buffer - saved cache file not found.")

    def get_data_and_lock_from_buffers(self):
        """
        return shared data buffer previously allocated by allocate_data_buffer and injected mp_tasks.run_simulation
        Returns
        -------
        either multiprocessing.Array and lock or multiprocessing.RawArray and None according to RAWARRAY
        """
        data_buffers = inject.get_injectable("data_buffers", None)
        assert self.cache_tag in data_buffers  # internal error
        logger.debug(f"TVPBCache.get_data_and_lock_from_buffers")
        data_buffer = data_buffers[self.cache_tag]
        if RAWARRAY:
            data = np.ctypeslib.as_array(data_buffer)
            lock = None
        else:
            data = np.ctypeslib.as_array(data_buffer.get_obj())
            lock = data_buffer.get_lock()

        return data, lock


class TapTapUidCalculator(object):
    """
    Transit virtual path builder TAP to TAP unique ID calculator for three zone systems
    """

    def __init__(self, network_los):

        self.network_los = network_los

        # ensure that tap_df has been loaded
        # (during multiprocessing we are initialized before network_los.load_data is called)
        assert network_los.tap_df is not None
        self.tap_ids = network_los.tap_df["TAP"].values

        self.segmentation = network_los.setting(
            "TVPB_SETTINGS.tour_mode_choice.tap_tap_settings.attribute_segments"
        )

        # e.g. [(0, 'AM', 'walk'), (0, 'AM', 'walk')...]) for attributes demographic_segment, tod, and access_mode
        self.attribute_combination_tuples = list(
            itertools.product(*list(self.segmentation.values()))
        )

        # ordinalizers - for mapping attribute values to canonical ordinal values for uid computation
        # (pandas series of ordinal position with attribute value index (e.g. map tod value 'AM' to 0, 'MD' to 1,...)
        # FIXME dict might be faster than Series.map() and Series.at[]?
        self.ordinalizers = {}
        for k, v in self.segmentation.items():
            self.ordinalizers[k] = pd.Series(range(len(v)), index=v)
        # orig/dest go last so all rows in same 'skim' end up with adjacent uids
        self.ordinalizers["btap"] = pd.Series(
            range(len(self.tap_ids)), index=self.tap_ids
        )
        self.ordinalizers["atap"] = self.ordinalizers["btap"]

        # for k,v in self.ordinalizers.items():
        #     print(f"\ordinalizer {k}\n{v}")

        spec_name = self.network_los.setting(
            f"TVPB_SETTINGS.tour_mode_choice.tap_tap_settings.SPEC"
        )
        self.set_names = list(simulate.read_model_spec(file_name=spec_name).columns)

    @property
    def fully_populated_shape(self):
        # (num_combinations * num_orig_zones * num_dest_zones, num_sets)
        num_combinations = len(self.attribute_combination_tuples)
        num_orig_zones = num_dest_zones = len(self.tap_ids)
        num_rows = num_combinations * num_orig_zones * num_dest_zones
        num_sets = len(self.set_names)
        return (num_rows, num_sets)

    @property
    def skim_shape(self):
        # (num_combinations, num_od_rows, num_sets)
        num_combinations = len(self.attribute_combination_tuples)
        num_orig_zones = num_dest_zones = len(self.tap_ids)
        num_od_rows = num_orig_zones * num_dest_zones
        num_sets = len(self.set_names)
        return (num_combinations, num_od_rows, num_sets)

    @property
    def fully_populated_uids(self):
        num_combinations = len(self.attribute_combination_tuples)
        num_orig_zones = num_dest_zones = len(self.tap_ids)
        return np.arange(num_combinations * num_orig_zones * num_dest_zones)

    def get_unique_ids(self, df, scalar_attributes):
        """
        compute canonical unique_id for each row in df
        btap and atap will be in dataframe, but the other attributes may be either df columns or scalar_attributes

        Parameters
        ----------
        df: pandas DataFrame
            with btap, atap, and optionally additional attribute columns
        scalar_attributes: dict
            dict of scalar attributes e.g. {'tod': 'AM', 'demographic_segment': 0}
        Returns
        -------
        ndarray of integer uids
        """
        uid = np.zeros(len(df), dtype=int)

        # need to know cardinality and integer representation of each tap/attribute
        for name, ordinalizer in self.ordinalizers.items():

            cardinality = ordinalizer.max() + 1

            if name in df:
                # if there is a column, use it
                uid = uid * cardinality + np.asanyarray(df[name].map(ordinalizer))
            else:
                # otherwise it should be in scalar_attributes
                assert (
                    name in scalar_attributes
                ), f"attribute '{name}' not found in df.columns or scalar_attributes."
                uid = uid * cardinality + ordinalizer.at[scalar_attributes[name]]

        return uid

    def get_od_dataframe(self, scalar_attributes):
        """
        return tap-tap od dataframe with unique_id index for 'skim_offset' for scalar_attributes

        i.e. a dataframe which may be used to compute utilities, together with scalar or column attributes

        Parameters
        ----------
        scalar_attributes: dict of scalar attribute name:value pairs

        Returns
        -------
        pandas.Dataframe
        """

        # create OD dataframe in ROW_MAJOR_LAYOUT
        num_taps = len(self.tap_ids)
        od_choosers_df = pd.DataFrame(
            data={
                "btap": np.repeat(self.tap_ids, num_taps),
                "atap": np.tile(self.tap_ids, num_taps),
            }
        )
        od_choosers_df.index = self.get_unique_ids(od_choosers_df, scalar_attributes)
        assert not od_choosers_df.index.duplicated().any()

        return od_choosers_df

    def get_skim_offset(self, scalar_attributes):
        # return ordinal position of this set of attributes in the list of attribute_combination_tuples
        offset = 0
        for name, ordinalizer in self.ordinalizers.items():
            cardinality = ordinalizer.max() + 1
            if name in scalar_attributes:
                offset = offset * cardinality + ordinalizer.at[scalar_attributes[name]]
        return offset

    def each_scalar_attribute_combination(self):
        # iterate through attribute_combination_tuples, yielding dict of scalar attribute name:value pairs

        # attribute names as list of strings
        attribute_names = list(self.segmentation.keys())
        for attribute_value_tuple in self.attribute_combination_tuples:

            # attribute_value_tuple is an tuple of attribute values - e.g. (0, 'AM', 'walk')
            # build dict of attribute name:value pairs - e.g. {'demographic_segment': 0, 'tod': 'AM', })
            scalar_attributes = {
                name: value
                for name, value in zip(attribute_names, attribute_value_tuple)
            }

            yield scalar_attributes

    def scalar_attribute_combinations(self):
        attribute_names = list(self.segmentation.keys())
        attribute_tuples = self.attribute_combination_tuples
        x = [list(t) for t in attribute_tuples]
        df = pd.DataFrame(data=x, columns=attribute_names)
        df.index.name = "offset"
        return df

# ActivitySim
# See full license in LICENSE.txt.
from builtins import range
from builtins import int

import sys
import os
import logging
import multiprocessing
import warnings

from collections import OrderedDict
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
import openmatrix as omx

from activitysim.core import skim
from activitysim.core import skim_maz

from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config

logger = logging.getLogger(__name__)

LOS_SETTINGS_FILE_NAME = 'network_los.yaml'

ONE_ZONE = 1
TWO_ZONE = 2
THREE_ZONE = 3


def multiply_large_numbers(list_of_numbers):
    return reduce(mul, list_of_numbers)


def allocate_skim_buffer(skim_info, shared=False):
    skim_dtype = skim_info['dtype']
    omx_shape = skim_info['omx_shape']
    num_skims = skim_info['num_skims']
    skim_tag = skim_info['skim_tag']

    # buffer_size must be int, not np.int64
    buffer_size = int(multiply_large_numbers(omx_shape) * num_skims)

    itemsize = np.dtype(skim_dtype).itemsize
    csz = buffer_size * itemsize
    logger.info("allocating shared buffer %s for %s skims (skim size: %s * %s bytes = %s) total size: %s (%s)" %
                (skim_tag, num_skims, omx_shape, itemsize, buffer_size, csz, util.GB(csz)))

    if shared:
        if np.issubdtype(skim_dtype, np.float64):
            typecode = 'd'
        elif np.issubdtype(skim_dtype, np.float32):
            typecode = 'f'
        else:
            raise RuntimeError("allocate_skim_buffer unrecognized dtype %s" % skim_dtype)

        buffer = multiprocessing.RawArray(typecode, buffer_size)
    else:
        buffer = np.zeros(buffer_size, dtype=skim_dtype)

    return buffer


def skim_data_from_buffer(skim_info, skim_buffer):

    omx_shape = skim_info['omx_shape']
    skim_dtype = skim_info['dtype']
    num_skims = skim_info['num_skims']

    skims_shape = omx_shape + (num_skims,)

    assert len(skim_buffer) == int(multiply_large_numbers(skims_shape))
    skim_data = np.frombuffer(skim_buffer, dtype=skim_dtype).reshape(skims_shape)

    return skim_data


def default_skim_cache_dir():
    return inject.get_injectable('output_dir')


def build_skim_cache_file_name(skim_tag):
    return f"cached_{skim_tag}.mmap"


def read_skim_cache(skim_info, skim_data, skim_cache_dir):
    """
        read cached memmapped skim data from canonically named cache file(s) in output directory into skim_data
    """

    logger.info(f"reading skims data from cache directory {skim_cache_dir}")

    skim_tag = skim_info['skim_tag']
    dtype = np.dtype(skim_info['dtype'])

    skim_cache_file_name = build_skim_cache_file_name(skim_tag)
    skim_cache_path = os.path.join(skim_cache_dir, skim_cache_file_name)

    assert os.path.isfile(skim_cache_path), \
        "read_skim_cache could not find skim_cache_path: %s" % (skim_cache_path,)

    logger.info(f"reading skim cache {skim_tag} {skim_data.shape} from {skim_cache_file_name}")

    data = np.memmap(skim_cache_path, shape=skim_data.shape, dtype=dtype, mode='r')
    assert data.shape == skim_data.shape

    skim_data[::] = data[::]


def write_skim_cache(skim_info, skim_data, skim_cache_dir):
    """
        write skim data from skim_data to canonically named cache file(s) in output directory
    """

    logger.info(f"writing skims data to cache directory {skim_cache_dir}")

    skim_tag = skim_info['skim_tag']
    dtype = np.dtype(skim_info['dtype'])

    skim_cache_file_name = build_skim_cache_file_name(skim_tag)
    skim_cache_path = os.path.join(skim_cache_dir, skim_cache_file_name)

    logger.info(f"writing skim cache {skim_tag} {skim_data.shape} to {skim_cache_file_name}")

    data = np.memmap(skim_cache_path, shape=skim_data.shape, dtype=dtype, mode='w+')
    data[::] = skim_data


def read_skims_from_omx(skim_info, skim_data):
    """
    read skims from omx file into skim_data
    """

    block_offsets = skim_info['block_offsets']
    omx_keys = skim_info['omx_keys']
    omx_file_names = skim_info['omx_file_names']
    omx_manifest = skim_info['omx_manifest']   # dict mapping { omx_key: skim_name }

    for omx_file_name in omx_file_names:

        omx_file_path = config.data_file_path(omx_file_name)
        num_skims_loaded = 0

        logger.info(f"read_skims_from_omx {omx_file_path}")

        # read skims into skim_data
        with omx.open_file(omx_file_path) as omx_file:
            for skim_key, omx_key in omx_keys.items():

                if omx_manifest[omx_key] == omx_file_name:
                    omx_data = omx_file[omx_key]
                    assert np.issubdtype(omx_data.dtype, np.floating)

                    offset = block_offsets[skim_key]

                    logger.debug(f"read_skims_from_omx file {omx_file_name} omx_key {omx_key} "
                                 f"skim_key {skim_key} to offset {offset}")

                    # this will trigger omx readslice to read and copy data to skim_data's buffer
                    a = skim_data[:, :, offset]
                    a[:] = omx_data[:]

                    num_skims_loaded += 1

        logger.info(f"read_skims_from_omx loaded {num_skims_loaded} skims from {omx_file_name}")


def load_skims(skim_info, skim_buffer, network_los):

    read_cache = network_los.setting('read_skim_cache', False)
    write_cache = network_los.setting('write_skim_cache', False)

    assert not (read_cache and write_cache), \
        "read_skim_cache and write_skim_cache are both True in settings file. I am assuming this is a mistake"

    skim_data = skim_data_from_buffer(skim_info, skim_buffer)

    if read_cache:
        read_skim_cache(skim_info, skim_data, network_los.setting('skim_cache_dir', default_skim_cache_dir()))
    else:
        read_skims_from_omx(skim_info, skim_data)

    if write_cache:
        write_skim_cache(skim_info, skim_data, network_los.setting('skim_cache_dir', default_skim_cache_dir()))


def load_skim_info(skim_tag, omx_file_names, skim_time_periods):
    """
    Read omx files for skim <skim_tag> (e.g. 'TAZ') and build skim_info dict

    Parameters
    ----------
    skim_tag

    Returns
    -------

    """

    # accept a single file_name str as well as list of file names
    omx_file_names = [omx_file_names] if isinstance(omx_file_names, str) else omx_file_names

    tags_to_load = skim_time_periods and skim_time_periods['labels']

    # Note: we load all skims except those with key2 not in tags_to_load
    # Note: we require all skims to be of same dtype so they can share buffer - is that ok?
    # fixme is it ok to require skims be all the same type? if so, is this the right choice?
    skim_dtype = np.float32

    omx_shape = offset_map = offset_map_name = None
    omx_manifest = {}  # dict mapping { omx_key: skim_name }

    for omx_file_name in omx_file_names:

        omx_file_path = config.data_file_path(omx_file_name)

        logger.debug("get_skim_info reading %s" % (omx_file_path,))

        with omx.open_file(omx_file_path) as omx_file:
            # omx_shape = tuple(map(int, tuple(omx_file.shape())))  # sometimes omx shape are floats!

            # fixme call to omx_file.shape() failing in windows p3.5
            if omx_shape is None:
                omx_shape = tuple(int(i) for i in omx_file.shape())  # sometimes omx shape are floats!
            else:
                assert (omx_shape == tuple(int(i) for i in omx_file.shape()))

            for skim_name in omx_file.listMatrices():
                assert skim_name not in omx_manifest, \
                    f"duplicate skim '{skim_name}' found in {omx_manifest[skim_name]} and {omx_file}"
                omx_manifest[skim_name] = omx_file_name

            for m in omx_file.listMappings():
                if offset_map is None:
                    offset_map_name = m
                    offset_map = omx_file.mapentries(offset_map_name)
                    assert len(offset_map) == omx_shape[0]

                    logger.debug(f"get_skim_info skim_tag {skim_tag} using offset_map {m}")
                else:
                    # don't really expect more than one, but ok if they are all the same
                    if not (offset_map == omx_file.mapentries(m)):
                        raise RuntimeError(f"Multiple different mappings in omx file: {offset_map_name} != {m}")

    # - omx_keys dict maps skim key to omx_key
    # DISTWALK: DISTWALK
    # ('DRV_COM_WLK_BOARDS', 'AM'): DRV_COM_WLK_BOARDS__AM, ...
    omx_keys = OrderedDict()
    for skim_name in omx_manifest.keys():
        key1, sep, key2 = skim_name.partition('__')

        # - ignore composite tags not in tags_to_load
        if tags_to_load and sep and key2 not in tags_to_load:
            continue

        skim_key = (key1, key2) if sep else key1

        omx_keys[skim_key] = skim_name

    num_skims = len(omx_keys)

    # - key1_subkeys dict maps key1 to dict of subkeys with that key1
    # DIST: {'DIST': 0}
    # DRV_COM_WLK_BOARDS: {'MD': 1, 'AM': 0, 'PM': 2}, ...
    key1_subkeys = OrderedDict()
    for skim_key, omx_key in omx_keys.items():
        if isinstance(skim_key, tuple):
            key1, key2 = skim_key
        else:
            key1 = key2 = skim_key
        key2_dict = key1_subkeys.setdefault(key1, {})
        key2_dict[key2] = len(key2_dict)

    key1_block_offsets = OrderedDict()
    offset = 0
    for key1, v in key1_subkeys.items():
        num_subkeys = len(v)
        key1_block_offsets[key1] = offset
        offset += num_subkeys

    # - block_offsets dict maps skim_key to offset of omx matrix
    # DIST: 0,
    # ('DRV_COM_WLK_BOARDS', 'AM'): 3,
    # ('DRV_COM_WLK_BOARDS', 'MD') 4, ...
    block_offsets = OrderedDict()
    for skim_key in omx_keys:

        if isinstance(skim_key, tuple):
            key1, key2 = skim_key
        else:
            key1 = key2 = skim_key

        key1_offset = key1_block_offsets[key1]
        key2_relative_offset = key1_subkeys.get(key1).get(key2)
        block_offsets[skim_key] = key1_offset + key2_relative_offset

    logger.debug("get_skim_info skim_dtype %s omx_shape %s num_skims %s" %
                 (skim_dtype, omx_shape, num_skims,))

    skim_info = {
        'skim_tag': skim_tag,
        'omx_file_names': omx_file_names,  # list of omx_file_names
        'omx_manifest': omx_manifest,  # dict mapping { omx_key: omx_file_name }
        'omx_shape': omx_shape,
        'num_skims': num_skims,
        'dtype': skim_dtype,
        'offset_map_name': offset_map_name,
        'offset_map': offset_map,
        'omx_keys': omx_keys,
        'base_keys': list(key1_block_offsets.keys()),  # list of base (key1) keys
        'block_offsets': block_offsets,
    }

    return skim_info


def create_skim_dict(skim_tag, skim_info, network_los):

    logger.info(f"create_skim_dict loading skim dict skim_tag: {skim_tag}")

    # select the skims to load

    logger.debug(f"create_skim_dict {skim_tag} omx_shape {skim_info['omx_shape']} skim_dtype {skim_info['dtype']}")

    data_buffers = inject.get_injectable('data_buffers', None)
    if data_buffers:
        # we assume any existing skim buffers will already have skim data loaded into them
        logger.info('create_skim_dict {skim_tag} using existing skim_buffers for skims')
        skim_buffer = data_buffers[skim_tag]
    else:
        skim_buffer = allocate_skim_buffer(skim_info, shared=False)
        load_skims(skim_info, skim_buffer, network_los)

    skim_data = skim_data_from_buffer(skim_info, skim_buffer)

    logger.info(f"create_skim_dict {skim_tag} bytes {skim_data.nbytes} ({util.GB(skim_data.nbytes)})")

    # create skim dict
    skim_dict = skim.SkimDict(skim_data, skim_info)

    # set offset
    offset_map = skim_info['offset_map']
    if offset_map is not None:
        skim_dict.offset_mapper.set_offset_list(offset_map)
        logger.debug(f"create_skim_dict {skim_tag} "
                     f"using offset map {skim_info['offset_map_name']} "
                     f"from omx file: {offset_map}")
    else:
        # assume this is a one-based skim map
        skim_dict.offset_mapper.set_offset_int(-1)

    return skim_dict


class Network_LOS(object):

    def __init__(self, los_settings_file_name=LOS_SETTINGS_FILE_NAME):

        self.zone_system = None
        self.skim_time_periods = None

        self.skims_info = {}
        self.skim_buffers = {}
        self.skim_dicts = {}
        self.skim_stacks = {}

        self.tables = {}

        # TWO_ZONE and THREE_ZONE
        self.maz_df = None
        self.maz_to_maz_df = None
        self.maz_ceiling = None
        self.max_blend_distance = {}

        # THREE_ZONE only
        self.tap_df = None
        self.maz_to_tap_dfs = {}
        # map maz_to_tap_attributes to mode (for now, attribute names must be unique across modes)
        self.maz_to_tap_attributes = {}

        self.los_settings_file_name = los_settings_file_name
        self.load_settings()

    def setting(self, keys, default='<REQUIRED>'):
        # get setting value for single key or dot-delimited key path (e.g. 'maz_to_tap.modes')
        key_list = keys.split('.')
        s = self.los_settings
        for key in key_list[:-1]:
            s = s.get(key)
            assert isinstance(s, dict), f"expected key '{key}' not found in '{keys}' in {self.los_settings_file_name}"
        key = key_list[-1]  # last key
        if default == '<REQUIRED>':
            assert key in s, f"Expected setting {keys} not found in in {LOS_SETTINGS_FILE_NAME}"
        return s.get(key, default)

    def load_settings(self):

        try:
            self.los_settings = config.read_settings_file(self.los_settings_file_name, mandatory=True)
        except config.SettingsFileNotFound as e:

            print(f"los_settings_file_name {self.los_settings_file_name} not found - trying global settings")
            print(f"skims_file: {config.setting('skims_file')}")
            print(f"skim_time_periods: {config.setting('skim_time_periods')}")
            print(f"source_file_paths: {config.setting('source_file_paths')}")
            print(f"inject.get_injectable('configs_dir') {inject.get_injectable('configs_dir')}")

            # look for legacy 'skims_file' setting in global settings file
            if config.setting('skims_file'):

                warnings.warn("Support for 'skims_file' setting in global settings file will be removed."
                              "Use 'taz_skims' in network_los.yaml config file instead.", FutureWarning)

                # in which case, we also expect to find skim_time_periods in settings file
                skim_time_periods = config.setting('skim_time_periods')
                assert skim_time_periods is not None, "'skim_time_periods' setting not found."
                warnings.warn("Support for 'skim_time_periods' setting in global settings file will be removed."
                              "Put 'skim_time_periods' in network_los.yaml config file instead.", FutureWarning)

                self.los_settings = {
                    'taz_skims': config.setting('skims_file'),
                    'zone_system': ONE_ZONE,
                    'skim_time_periods': skim_time_periods
                }

            else:
                raise e

        # validate skim_time_periods
        self.skim_time_periods = self.setting('skim_time_periods')
        if 'hours' in self.skim_time_periods:
            self.skim_time_periods['periods'] = self.skim_time_periods.pop('hours')
            warnings.warn('support for `skim_time_periods` key `hours` will be removed in '
                          'future verions. Use `periods` instead',
                          FutureWarning)
        assert 'periods' in self.skim_time_periods, "'periods' key not found in network_los.skim_time_periods"
        assert 'labels' in self.skim_time_periods, "'labels' key not found in network_los.skim_time_periods"

        self.zone_system = self.setting('zone_system', default=ONE_ZONE)
        assert self.zone_system in [ONE_ZONE, TWO_ZONE, THREE_ZONE], \
            f"Network_LOS: unrecognized zone_system: {self.zone_system}"

        # load taz skim_info
        skim_file_names = self.setting('taz_skims')
        self.skims_info['taz'] = self.load_skim_info('taz', skim_file_names)

        if self.zone_system in [TWO_ZONE, THREE_ZONE]:

            # maz_to_maz_settings
            self.max_blend_distance = self.setting('maz_to_maz.max_blend_distance', {})
            if isinstance(self.max_blend_distance, int):
                self.max_blend_distance = {'DEFAULT': self.max_blend_distance}
            self.blend_distance_skim_name = self.setting('maz_to_maz.blend_distance_skim_name', None)

        if self.zone_system == THREE_ZONE:

            # load tap skim_info
            skim_file_names = self.setting('tap_to_tap.skims')
            self.skims_info['tap'] = self.load_skim_info('tap', skim_file_names)

        # validate skim_time_periods
        self.skim_time_periods = self.setting('skim_time_periods')
        assert {'periods', 'labels'}.issubset(set(self.skim_time_periods.keys()))

    def load_data(self):

        def as_list(file_name):
            return [file_name] if isinstance(file_name, str) else file_name

        # load maz tables
        if self.zone_system in [TWO_ZONE, THREE_ZONE]:

            # maz
            file_name = self.setting('maz')
            self.maz_df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

            self.maz_ceiling = self.maz_df.MAZ.max() + 1

            # maz_to_maz_df
            for file_name in as_list(self.setting('maz_to_maz.tables')):

                df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

                df['i'] = df.OMAZ * self.maz_ceiling + df.DMAZ
                df.set_index('i', drop=True, inplace=True, verify_integrity=True)
                logger.debug(f"loading maz_to_maz table {file_name} with {len(df)} rows")

                # FIXME - don't really need these columns, but if we do want them,
                #  we would need to merge them in since files may have different numbers of rows
                df.drop(columns=['OMAZ', 'DMAZ'], inplace=True)

                if self.maz_to_maz_df is None:
                    self.maz_to_maz_df = df
                else:
                    self.maz_to_maz_df = pd.concat([self.maz_to_maz_df, df], axis=1)

        # load tap tables
        if self.zone_system == THREE_ZONE:

            # tap
            file_name = self.setting('tap')
            self.tap_df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

            # self.tap_ceiling = self.tap_df.TAP.max() + 1

            # maz_to_tap_dfs
            maz_to_tap_modes = self.setting('maz_to_tap.modes')

            tables = self.setting('maz_to_tap.tables')
            assert set(maz_to_tap_modes) == set(tables.keys()), \
                f"maz_to_tap.tables keys do not match maz_to_tap.modes"
            for mode in maz_to_tap_modes:
                # FIXME - should we ensure that there is no attribute name collision?
                # different sized sparse arrays, so we keep them seperate
                for file_name in as_list(tables[mode]):

                    df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

                    # df['i'] = df.MAZ * self.maz_ceiling + df.TAP
                    df.set_index(['MAZ', 'TAP'], drop=True, inplace=True, verify_integrity=True)
                    logger.debug(f"loading maz_to_tap table {file_name} with {len(df)} rows")

                    # ensure that there is no attribute name collision
                    colliding_attributes = set(self.maz_to_tap_attributes.keys()).intersection(set(df.columns))
                    assert not colliding_attributes, \
                        f"duplicate maz_to_tap attributes {colliding_attributes}." \
                        f"Attribute names should be unique across modes."
                    self.maz_to_tap_attributes.update({c: mode for c in df.columns})

                    if mode not in self.maz_to_tap_dfs:
                        self.maz_to_tap_dfs[mode] = df
                    else:
                        self.maz_to_tap_dfs[mode] = pd.concat([self.maz_to_tap_dfs[mode], df], axis=1)

        # create taz skim dict
        assert 'taz' in self.skims_info
        self.skim_dicts['taz'] = self.create_skim_dict('taz')

        # create MazSkimDict facade
        if self.zone_system in [TWO_ZONE, THREE_ZONE]:
            # create MazSkimDict facade skim_dict
            # (need to have already loaded both taz skim and maz tables)
            self.skim_dicts['maz'] = skim_maz.MazSkimDict(self)

        # create tap skim dict
        if self.zone_system == THREE_ZONE:
            assert 'tap' in self.skims_info
            self.skim_dicts['tap'] = self.create_skim_dict('tap')

    def load_skim_info(self, skim_tag, omx_file_names):
        """
        Read omx files for skim <skim_tag> (e.g. 'TAZ') and build skims dict
        """

        # we could just do nothing and return if already loaded, but for now tidier to track this
        assert skim_tag not in self.skims_info
        return load_skim_info(skim_tag, omx_file_names, self.skim_time_periods)

    def create_skim_dict(self, skim_tag):
        return create_skim_dict(skim_tag, self.skims_info[skim_tag], self)

    def load_shared_data(self, shared_data_buffers):
        for skim_tag in self.skims_info.keys():
            load_skims(self.skims_info[skim_tag], shared_data_buffers[skim_tag], self)

        # FIXME - should also load tables - if we knew how to share data
        # for table_name in self.table_info:
        #     do something

    def allocate_shared_skim_buffers(self):

        assert not self.skim_buffers

        for skim_tag in self.skims_info.keys():
            self.skim_buffers[skim_tag] = allocate_skim_buffer(self.skims_info[skim_tag], shared=True)

        return self.skim_buffers

    def get_skim_dict(self, skim_tag):
        return self.skim_dicts[skim_tag]

    def get_default_skim_dict(self):
        if self.zone_system == ONE_ZONE:
            return self.get_skim_dict('taz')
        else:
            return self.get_skim_dict('maz')

    def get_skim_stack(self, skim_tag):
        assert skim_tag in self.skim_dicts
        if skim_tag not in self.skim_stacks:
            logger.debug(f"network_los get_skim_stack initializing skim_stack for {skim_tag}")
            self.skim_stacks[skim_tag] = skim.SkimStack(self.skim_dicts[skim_tag])
        return self.skim_stacks[skim_tag]

    def get_table(self, table_name):
        assert table_name in self.tables, f"get_table: table '{table_name}' not loaded"
        return self.tables.get(table_name)

    def get_mazpairs(self, omaz, dmaz, attribute):

        # # this is slower
        # s = pd.merge(pd.DataFrame({'OMAZ': omaz, 'DMAZ': dmaz}),
        #              self.maz_to_maz_df,
        #              how="left")[attribute]

        # synthetic index method i : omaz_dmaz
        i = np.asanyarray(omaz) * self.maz_ceiling + np.asanyarray(dmaz)
        s = util.quick_loc_df(i, self.maz_to_maz_df, attribute)

        # FIXME - no point in returning series? unless maz and tap have same index?
        return np.asanyarray(s)

    def get_maztappairs(self, maz, tap, attribute):

        mode = self.maz_to_tap_attributes.get(attribute)

        assert mode is not None, \
            f"get_maztappairs attribute {attribute} not in any maz_to_tap table." \
            f"Existing attributes are: {self.maz_to_tap_attributes.keys()}"

        maz_to_tap_df = self.maz_to_tap_dfs[mode]

        return maz_to_tap_df.loc[zip(maz, tap), attribute].values

    def get_tappairs3d(self, otap, dtap, dim3, key):

        s = self.get_skim_stack('tap').lookup(otap, dtap, dim3, key)
        return s

    def skim_time_period_label(self, time_period):
        """
        convert time period times to skim time period labels (e.g. 9 -> 'AM')

        Parameters
        ----------
        time_period : pandas Series

        Returns
        -------
        pandas Series
            string time period labels
        """

        assert self.skim_time_periods is not None, "'skim_time_periods' setting not found."

        # Default to 60 minute time periods
        period_minutes = self.skim_time_periods.get('period_minutes', 60)

        # Default to a day
        model_time_window_min = self.skim_time_periods.get('time_window', 1440)

        # Check to make sure the intervals result in no remainder time through 24 hour day
        assert 0 == model_time_window_min % period_minutes
        total_periods = model_time_window_min / period_minutes

        # FIXME - eventually test and use np version always?
        if np.isscalar(time_period):
            bin = np.digitize([time_period % total_periods],
                              self.skim_time_periods['periods'], right=True)[0] - 1
            return self.skim_time_periods['labels'][bin]

        return pd.cut(time_period, self.skim_time_periods['periods'],
                      labels=self.skim_time_periods['labels'], right=True).astype(str)


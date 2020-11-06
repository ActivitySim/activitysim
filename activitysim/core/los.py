# ActivitySim
# See full license in LICENSE.txt.
#from builtins import range
#from builtins import int

import os
import logging
import warnings

import numpy as np
import pandas as pd

from activitysim.core import skim
from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config
from activitysim.core import tracing

#
from activitysim.core.skim_ram import NumpyArraySkimFactory
from activitysim.core.skim_mmap import MemMapSkimFactory

skim_factories = {
    'NumpyArraySkimFactory': NumpyArraySkimFactory,
    'MemMapSkimFactory': MemMapSkimFactory,
}

logger = logging.getLogger(__name__)

LOS_SETTINGS_FILE_NAME = 'network_los.yaml'

ONE_ZONE = 1
TWO_ZONE = 2
THREE_ZONE = 3


class Network_LOS(object):

    def __init__(self, los_settings_file_name=LOS_SETTINGS_FILE_NAME):

        self.skim_dtype_name = 'float32'

        self.zone_system = None
        self.skim_time_periods = None

        self.skims_info = {}
        self.skim_buffers = {}
        self.skim_dicts = {}
        self.skim_stacks = {}

        self.tables = {}

        # TWO_ZONE and THREE_ZONE
        self.maz_taz_df = None
        self.maz_to_maz_df = None
        self.maz_ceiling = None
        self.max_blend_distance = {}

        # THREE_ZONE only
        self.tap_df = None
        self.tap_lines_df = None
        self.maz_to_tap_dfs = {}

        self.los_settings_file_name = los_settings_file_name
        self.load_settings()

        # skim factory
        skim_factory_name = self.setting('skim_factory', default='NumpyArraySkimFactory')
        assert skim_factory_name in skim_factories, f"Unrecognized skim_factory setting '{skim_factory_name}"
        self.skim_factory = skim_factories[skim_factory_name](network_los=self)
        logger.info(f"Network_LOS using {self.skim_factory.name}")
        self.load_skim_info()

    def setting(self, keys, default='<REQUIRED>'):
        # get setting value for single key or dot-delimited key path (e.g. 'maz_to_maz.tables')
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

        if self.zone_system in [TWO_ZONE, THREE_ZONE]:
            # maz_to_maz_settings
            self.max_blend_distance = self.setting('maz_to_maz.max_blend_distance', {})
            if isinstance(self.max_blend_distance, int):
                self.max_blend_distance = {'DEFAULT': self.max_blend_distance}
            self.blend_distance_skim_name = self.setting('maz_to_maz.blend_distance_skim_name', None)

        # validate skim_time_periods
        self.skim_time_periods = self.setting('skim_time_periods')
        assert {'periods', 'labels'}.issubset(set(self.skim_time_periods.keys()))

    def load_skim_info(self):
        assert self.skim_factory is not None
        # load taz skim_info
        self.skims_info['taz'] = self.skim_factory.load_skim_info('taz')
        if self.zone_system == THREE_ZONE:
            # load tap skim_info
            self.skims_info['tap'] = self.skim_factory.load_skim_info('tap')

    def load_data(self):

        def as_list(file_name):
            return [file_name] if isinstance(file_name, str) else file_name

        # load maz tables
        if self.zone_system in [TWO_ZONE, THREE_ZONE]:

            # maz
            file_name = self.setting('maz')
            self.maz_taz_df = pd.read_csv(config.data_file_path(file_name, mandatory=True))
            self.maz_taz_df = self.maz_taz_df[['MAZ', 'TAZ']]  # only fields we need

            self.maz_ceiling = self.maz_taz_df.MAZ.max() + 1

            # maz_to_maz_df
            for file_name in as_list(self.setting('maz_to_maz.tables')):

                df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

                df['i'] = df.OMAZ * self.maz_ceiling + df.DMAZ
                df.set_index('i', drop=True, inplace=True, verify_integrity=True)
                logger.debug(f"loading maz_to_maz table {file_name} with {len(df)} rows")

                # FIXME - don't really need these columns, but if we do want them,
                #  we would need to merge them in since files may have different numbers of rows
                df.drop(columns=['OMAZ', 'DMAZ'], inplace=True)

                # besides, we only want data columns so we can coerce to same type as skims
                df = df.astype(np.dtype(self.skim_dtype_name))

                if self.maz_to_maz_df is None:
                    self.maz_to_maz_df = df
                else:
                    self.maz_to_maz_df = pd.concat([self.maz_to_maz_df, df], axis=1)

        # load tap tables
        if self.zone_system == THREE_ZONE:

            # tap
            file_name = self.setting('tap')
            self.tap_df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

            # maz_to_tap_dfs - different sized sparse arrays with different columns, so we keep them seperate
            for mode, maz_to_tap_settings in self.setting('maz_to_tap').items():

                assert 'table' in maz_to_tap_settings, \
                    f"Expected setting maz_to_tap.{mode}.table not found in in {LOS_SETTINGS_FILE_NAME}"

                df = pd.read_csv(config.data_file_path(maz_to_tap_settings['table'], mandatory=True))

                # trim tap set
                # if provided, use tap_line_distance_col together with tap_lines table to trim the near tap set
                # to only include the nearest tap to origin when more than one tap serves the same line
                distance_col = maz_to_tap_settings.get('tap_line_distance_col')
                if distance_col:

                    if self.tap_lines_df is None:
                        # load tap_lines on demand (required if they specify tap_line_distance_col)
                        file_name = self.setting('tap_lines',)
                        self.tap_lines_df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

                        # csv file has one row per TAP with space-delimited list of lines served by that TAP
                        #  TAP                                      LINES
                        # 6020  GG_024b_SB GG_068_RT GG_228_WB GG_023X_RT
                        # stack to create dataframe with one column 'line' indexed by TAP with one row per line served
                        #  TAP        line
                        # 6020  GG_024b_SB
                        # 6020   GG_068_RT
                        # 6020   GG_228_WB
                        self.tap_lines_df = \
                            self.tap_lines_df.set_index('TAP').LINES.str.split(expand=True)\
                                .stack().droplevel(1).to_frame('line')

                    # NOTE - merge will remove unused taps (not appearing in tap_lines)
                    df = pd.merge(df, self.tap_lines_df, left_on='TAP', right_index=True)

                    # find nearest TAP to MAz that serves line
                    df = df.sort_values(by=distance_col).drop_duplicates(subset=['MAZ', 'line'])

                    # we don't need to remember which lines are served by which TAPs
                    df = df.drop(columns='line').drop_duplicates(subset=['MAZ', 'TAP'])

                    #df = df.sort_values(by=['MAZ', 'TAP']) #FIXME - not actually necessary

                df.set_index(['MAZ', 'TAP'], drop=True, inplace=True, verify_integrity=True)
                logger.debug(f"loading maz_to_tap table {file_name} with {len(df)} rows")

                assert mode not in self.maz_to_tap_dfs
                self.maz_to_tap_dfs[mode] = df

        # create taz skim dict
        assert 'taz' in self.skims_info
        self.skim_dicts['taz'] = self.create_skim_dict('taz')

        # create MazSkimDict facade
        if self.zone_system in [TWO_ZONE, THREE_ZONE]:
            # create MazSkimDict facade skim_dict
            # (need to have already loaded both taz skim and maz tables)
            self.skim_dicts['maz'] = skim.MazSkimDict('maz', self)

        # create tap skim dict
        if self.zone_system == THREE_ZONE:
            assert 'tap' in self.skims_info
            self.skim_dicts['tap'] = self.create_skim_dict('tap')


    def create_skim_dict(self, skim_tag):

        skim_info = self.skims_info[skim_tag]
        logger.debug(f"create_skim_dict {skim_tag} omx_shape {skim_info['omx_shape']} type {skim_info['dtype_name']}")

        skim_data = self.skim_factory.get_skim_data(skim_tag, skim_info)

        skim_dict = skim.SkimDict(skim_tag, skim_info, skim_data)

        return skim_dict

    def get_cache_dir(self):

        cache_dir = self.setting('cache_dir', os.path.join(inject.get_injectable('output_dir'), 'cache'))

        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        assert os.path.isdir(cache_dir)

        return cache_dir

    def omx_file_names(self, skim_tag):

        omx_file_names = self.setting(f'{skim_tag}_skims')

        # accept a single file_name str as well as list of file names
        omx_file_names = [omx_file_names] if isinstance(omx_file_names, str) else omx_file_names

        return omx_file_names

    def load_shared_data(self, shared_data_buffers):

        if self.skim_factory.share_data_for_multiprocessing:
            for skim_tag in self.skims_info.keys():
                assert skim_tag in shared_data_buffers, f"load_shared_data expected allocated shared_data_buffers"
                self.skim_factory.load_skims_to_buffer(self.skims_info[skim_tag], shared_data_buffers[skim_tag])

    def allocate_shared_skim_buffers(self):

        assert not self.skim_buffers

        if self.skim_factory.share_data_for_multiprocessing:
            for skim_tag in self.skims_info.keys():
                self.skim_buffers[skim_tag] = self.skim_factory.allocate_skim_buffer(self.skims_info[skim_tag], shared=True)

        return self.skim_buffers

    def get_skim_dict(self, skim_tag):
        return self.skim_dicts[skim_tag]

    def get_default_skim_dict(self):
        if self.zone_system == ONE_ZONE:
            return self.get_skim_dict('taz')
        else:
            return self.get_skim_dict('maz')

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

    def get_tappairs3d(self, otap, dtap, dim3, key):

        s = self.get_skim_dict('tap').lookup_3d(otap, dtap, dim3, key)
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

# ActivitySim
# See full license in LICENSE.txt.

import logging
import os
import warnings

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    inject,
    mem,
    pathbuilder,
    skim_dictionary,
    tracing,
    util,
)
from activitysim.core.skim_dict_factory import MemMapSkimFactory, NumpyArraySkimFactory
from activitysim.core.skim_dictionary import NOT_IN_SKIM_ZONE_ID

skim_factories = {
    "NumpyArraySkimFactory": NumpyArraySkimFactory,
    "MemMapSkimFactory": MemMapSkimFactory,
}

logger = logging.getLogger(__name__)

LOS_SETTINGS_FILE_NAME = "network_los.yaml"

ONE_ZONE = 1
TWO_ZONE = 2
THREE_ZONE = 3

DEFAULT_SETTINGS = {
    "rebuild_tvpb_cache": True,
    "zone_system": ONE_ZONE,
    "skim_dict_factory": "NumpyArraySkimFactory",
}

TRACE_TRIMMED_MAZ_TO_TAP_TABLES = True


class Network_LOS(object):
    """
    ::

      singleton object to manage skims and skim-related tables

      los_settings_file_name: str         # e.g. 'network_los.yaml'
      skim_dtype_name:str                 # e.g. 'float32'

      dict_factory_name: str              # e.g. 'NumpyArraySkimFactory'
      zone_system: str                    # str (ONE_ZONE, TWO_ZONE, or THREE_ZONE)
      skim_time_periods = None            # list of str e.g. ['AM', 'MD', 'PM''

      skims_info: dict                    # dict of SkimInfo keyed by skim_tag
      skim_buffers: dict                  # if multiprocessing, dict of multiprocessing.Array buffers keyed by skim_tag
      skim_dicts: dice                    # dict of SkimDict keyed by skim_tag

      # TWO_ZONE and THREE_ZONE
      maz_taz_df: pandas.DataFrame        # DataFrame with two columns, MAZ and TAZ, mapping MAZ to containing TAZ
      maz_to_maz_df: pandas.DataFrame     # maz_to_maz attributes for MazSkimDict sparse skims
                                          # indexed by synthetic omaz/dmaz index for faster get_mazpairs lookup)
      maz_ceiling: int                    # max maz_id + 1 (to compute synthetic omaz/dmaz index by get_mazpairs)
      max_blend_distance: dict            # dict of int maz_to_maz max_blend_distance values keyed by skim_tag

      # THREE_ZONE only
      tap_df: pandas.DataFrame
      tap_lines_df: pandas.DataFrame      # if specified in settings, list of transit lines served, indexed by TAP
                                          # use to prune maz_to_tap_dfs to drop more distant TAPS with redundant service
                                          # since a TAP can serve multiple lines, tap_lines_df TAP index is not unique
      maz_to_tap_dfs: dict                # dict of maz_to_tap DataFrames indexed by access mode (e.g. 'walk', 'drive')
                                          # maz_to_tap dfs have OMAZ and DMAZ columns plus additional attribute columns
      tap_tap_uid: TapTapUidCalculator
    """

    def __init__(self, los_settings_file_name=LOS_SETTINGS_FILE_NAME):

        # Note: we require all skims to be of same dtype so they can share buffer - is that ok?
        # fixme is it ok to require skims be all the same type? if so, is this the right choice?
        self.skim_dtype_name = "float32"
        self.zone_system = None
        self.skim_time_periods = None
        self.skims_info = {}
        self.skim_dicts = {}

        # TWO_ZONE and THREE_ZONE
        self.maz_taz_df = None
        self.maz_to_maz_df = None
        self.maz_ceiling = None
        self.max_blend_distance = {}

        # THREE_ZONE only
        self.tap_lines_df = None
        self.maz_to_tap_dfs = {}
        self.tvpb = None

        self.los_settings_file_name = los_settings_file_name
        self.load_settings()

        # dependency injection of skim factory (of type specified in skim_dict_factory setting)
        skim_dict_factory_name = self.setting("skim_dict_factory")
        assert (
            skim_dict_factory_name in skim_factories
        ), f"Unrecognized skim_dict_factory setting '{skim_dict_factory_name}"
        self.skim_dict_factory = skim_factories[skim_dict_factory_name](
            network_los=self
        )
        logger.info(
            f"Network_LOS using skim_dict_factory: {type(self.skim_dict_factory).__name__}"
        )

        # load SkimInfo for all skims for this zone_system (TAZ for ONE_ZONE and TWO_ZONE, TAZ and MAZ for THREE_ZONE)
        self.load_skim_info()

    @property
    def rebuild_tvpb_cache(self):
        # setting as property here so others don't need to know default
        assert (
            self.zone_system == THREE_ZONE
        ), f"Should not even be asking about rebuild_tvpb_cache if not THREE_ZONE"
        return self.setting("rebuild_tvpb_cache")

    def setting(self, keys, default="<REQUIRED>"):

        # if they dont specify a default, check the default defaults
        default = (
            DEFAULT_SETTINGS.get(keys, "<REQUIRED>")
            if default == "<REQUIRED>"
            else default
        )

        # get setting value for single key or dot-delimited key path (e.g. 'maz_to_maz.tables')
        key_list = keys.split(".")
        s = self.los_settings
        for key in key_list[:-1]:
            s = s.get(key)
            assert isinstance(
                s, dict
            ), f"expected key '{key}' not found in '{keys}' in {self.los_settings_file_name}"
        key = key_list[-1]  # last key
        if default == "<REQUIRED>":
            assert (
                key in s
            ), f"Expected setting {keys} not found in in {LOS_SETTINGS_FILE_NAME}"
        return s.get(key, default)

    def load_settings(self):
        """
        Read setting file and initialize object variables (see class docstring for list of object variables)
        """

        try:
            self.los_settings = config.read_settings_file(
                self.los_settings_file_name, mandatory=True
            )
        except config.SettingsFileNotFound as e:

            print(
                f"los_settings_file_name {self.los_settings_file_name} not found - trying global settings"
            )
            print(f"skims_file: {config.setting('skims_file')}")
            print(f"skim_time_periods: {config.setting('skim_time_periods')}")
            print(f"source_file_paths: {config.setting('source_file_paths')}")
            print(
                f"inject.get_injectable('configs_dir') {inject.get_injectable('configs_dir')}"
            )

            # look for legacy 'skims_file' setting in global settings file
            if config.setting("skims_file"):

                warnings.warn(
                    "Support for 'skims_file' setting in global settings file will be removed."
                    "Use 'taz_skims' in network_los.yaml config file instead.",
                    FutureWarning,
                )

                # in which case, we also expect to find skim_time_periods in settings file
                skim_time_periods = config.setting("skim_time_periods")
                assert (
                    skim_time_periods is not None
                ), "'skim_time_periods' setting not found."
                warnings.warn(
                    "Support for 'skim_time_periods' setting in global settings file will be removed."
                    "Put 'skim_time_periods' in network_los.yaml config file instead.",
                    FutureWarning,
                )

                self.los_settings = {
                    "taz_skims": config.setting("skims_file"),
                    "zone_system": ONE_ZONE,
                    "skim_time_periods": skim_time_periods,
                }

            else:
                raise e

        # validate skim_time_periods
        self.skim_time_periods = self.setting("skim_time_periods")
        if "hours" in self.skim_time_periods:
            self.skim_time_periods["periods"] = self.skim_time_periods.pop("hours")
            warnings.warn(
                "support for `skim_time_periods` key `hours` will be removed in "
                "future verions. Use `periods` instead",
                FutureWarning,
            )
        assert (
            "periods" in self.skim_time_periods
        ), "'periods' key not found in network_los.skim_time_periods"
        assert (
            "labels" in self.skim_time_periods
        ), "'labels' key not found in network_los.skim_time_periods"

        self.zone_system = self.setting("zone_system")
        assert self.zone_system in [
            ONE_ZONE,
            TWO_ZONE,
            THREE_ZONE,
        ], f"Network_LOS: unrecognized zone_system: {self.zone_system}"

        if self.zone_system in [TWO_ZONE, THREE_ZONE]:
            # maz_to_maz_settings
            self.max_blend_distance = self.setting(
                "maz_to_maz.max_blend_distance", default={}
            )
            if isinstance(self.max_blend_distance, int):
                self.max_blend_distance = {"DEFAULT": self.max_blend_distance}
            self.blend_distance_skim_name = self.setting(
                "maz_to_maz.blend_distance_skim_name", default=None
            )

        # validate skim_time_periods
        self.skim_time_periods = self.setting("skim_time_periods")
        assert {"periods", "labels"}.issubset(set(self.skim_time_periods.keys()))

    def load_skim_info(self):
        """
        read skim info from omx files into SkimInfo, and store in self.skims_info dict keyed by skim_tag

        ONE_ZONE and TWO_ZONE systems have only TAZ skims
        THREE_ZONE systems have both TAZ and TAP skims
        """
        assert self.skim_dict_factory is not None
        # load taz skim_info
        self.skims_info["taz"] = self.skim_dict_factory.load_skim_info("taz")

        if self.zone_system == THREE_ZONE:
            # load tap skim_info
            self.skims_info["tap"] = self.skim_dict_factory.load_skim_info("tap")

        if self.zone_system == THREE_ZONE:
            # load this here rather than in load_data as it is required during multiprocessing to size TVPBCache
            self.tap_df = pd.read_csv(
                config.data_file_path(self.setting("tap"), mandatory=True)
            ).sort_values("TAP")
            self.tvpb = pathbuilder.TransitVirtualPathBuilder(
                self
            )  # dependent on self.tap_df

    def load_data(self):
        """
        Load tables and skims from files specified in network_los settigns
        """

        # load maz tables
        if self.zone_system in [TWO_ZONE, THREE_ZONE]:

            # maz
            file_name = self.setting("maz")
            self.maz_taz_df = pd.read_csv(
                config.data_file_path(file_name, mandatory=True)
            )
            self.maz_taz_df = self.maz_taz_df[["MAZ", "TAZ"]].sort_values(
                by="MAZ"
            )  # only fields we need

            self.maz_ceiling = self.maz_taz_df.MAZ.max() + 1

            # maz_to_maz_df
            maz_to_maz_tables = self.setting("maz_to_maz.tables")
            maz_to_maz_tables = (
                [maz_to_maz_tables]
                if isinstance(maz_to_maz_tables, str)
                else maz_to_maz_tables
            )
            for file_name in maz_to_maz_tables:

                df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

                df["i"] = df.OMAZ * self.maz_ceiling + df.DMAZ
                df.set_index("i", drop=True, inplace=True, verify_integrity=True)
                logger.debug(
                    f"loading maz_to_maz table {file_name} with {len(df)} rows"
                )

                # FIXME - don't really need these columns, but if we do want them,
                #  we would need to merge them in since files may have different numbers of rows
                df.drop(columns=["OMAZ", "DMAZ"], inplace=True)

                # besides, we only want data columns so we can coerce to same type as skims
                df = df.astype(np.dtype(self.skim_dtype_name))

                if self.maz_to_maz_df is None:
                    self.maz_to_maz_df = df
                else:
                    self.maz_to_maz_df = pd.concat([self.maz_to_maz_df, df], axis=1)

        # load tap tables
        if self.zone_system == THREE_ZONE:

            # tap_df should already have been loaded by load_skim_info because,
            # during multiprocessing, it is required by TapTapUidCalculator to size TVPBCache
            # self.tap_df = pd.read_csv(config.data_file_path(self.setting('tap'), mandatory=True))
            assert self.tap_df is not None

            # maz_to_tap_dfs - different sized sparse arrays with different columns, so we keep them seperate
            for mode, maz_to_tap_settings in self.setting("maz_to_tap").items():

                assert (
                    "table" in maz_to_tap_settings
                ), f"Expected setting maz_to_tap.{mode}.table not found in in {LOS_SETTINGS_FILE_NAME}"

                file_name = maz_to_tap_settings["table"]
                df = pd.read_csv(config.data_file_path(file_name, mandatory=True))

                # trim tap set
                # if provided, use tap_line_distance_col together with tap_lines table to trim the near tap set
                # to only include the nearest tap to origin when more than one tap serves the same line
                distance_col = maz_to_tap_settings.get("tap_line_distance_col")
                if distance_col:

                    if self.tap_lines_df is None:
                        # load tap_lines on demand (required if they specify tap_line_distance_col)
                        tap_lines_file_name = self.setting(
                            "tap_lines",
                        )
                        self.tap_lines_df = pd.read_csv(
                            config.data_file_path(tap_lines_file_name, mandatory=True)
                        )

                        # csv file has one row per TAP with space-delimited list of lines served by that TAP
                        #  TAP                                      LINES
                        # 6020  GG_024b_SB GG_068_RT GG_228_WB GG_023X_RT
                        # stack to create dataframe with one column 'line' indexed by TAP with one row per line served
                        #  TAP        line
                        # 6020  GG_024b_SB
                        # 6020   GG_068_RT
                        # 6020   GG_228_WB
                        self.tap_lines_df = (
                            self.tap_lines_df.set_index("TAP")
                            .LINES.str.split(expand=True)
                            .stack()
                            .droplevel(1)
                            .to_frame("line")
                        )

                    old_len = len(df)

                    # NOTE - merge will remove unused taps (not appearing in tap_lines)
                    df = pd.merge(
                        df, self.tap_lines_df, left_on="TAP", right_index=True
                    )

                    # find nearest TAP to MAz that serves line
                    df = df.sort_values(by=distance_col).drop_duplicates(
                        subset=["MAZ", "line"]
                    )

                    # we don't need to remember which lines are served by which TAPs
                    df = (
                        df.drop(columns="line")
                        .drop_duplicates(subset=["MAZ", "TAP"])
                        .sort_values(["MAZ", "TAP"])
                    )

                    logger.debug(
                        f"trimmed maz_to_tap table {file_name} from {old_len} to {len(df)} rows "
                        f"based on tap_lines"
                    )
                    logger.debug(
                        f"maz_to_tap table {file_name} max {distance_col} {df[distance_col].max()}"
                    )

                    max_dist = maz_to_tap_settings.get("max_dist", None)
                    if max_dist:
                        old_len = len(df)
                        df = df[df[distance_col] <= max_dist]
                        logger.debug(
                            f"trimmed maz_to_tap table {file_name} from {old_len} to {len(df)} rows "
                            f"based on max_dist {max_dist}"
                        )

                    if TRACE_TRIMMED_MAZ_TO_TAP_TABLES:
                        tracing.write_csv(
                            df,
                            file_name=f"trimmed_{maz_to_tap_settings['table']}",
                            transpose=False,
                        )

                else:
                    logger.warning(
                        f"tap_line_distance_col not provided in {LOS_SETTINGS_FILE_NAME} so maz_to_tap "
                        f"pairs will not be trimmed which may result in high memory use and long runtimes"
                    )

                df.set_index(
                    ["MAZ", "TAP"], drop=True, inplace=True, verify_integrity=True
                )
                logger.debug(f"loaded maz_to_tap table {file_name} with {len(df)} rows")

                assert mode not in self.maz_to_tap_dfs
                self.maz_to_tap_dfs[mode] = df

        # create taz skim dict
        assert "taz" not in self.skim_dicts
        self.skim_dicts["taz"] = self.create_skim_dict("taz")

        # make sure skim has all taz_ids
        # FIXME - weird that there is no list of tazs?

        # create MazSkimDict facade
        if self.zone_system in [TWO_ZONE, THREE_ZONE]:
            # create MazSkimDict facade skim_dict
            # (must have already loaded dependencies: taz skim_dict, maz_to_maz_df, and maz_taz_df)
            assert "maz" not in self.skim_dicts
            maz_skim_dict = self.create_skim_dict("maz")
            self.skim_dicts["maz"] = maz_skim_dict

            # make sure skim has all maz_ids
            assert not (
                maz_skim_dict.offset_mapper.map(self.maz_taz_df["MAZ"].values)
                == NOT_IN_SKIM_ZONE_ID
            ).any()

        # create tap skim dict
        if self.zone_system == THREE_ZONE:
            assert "tap" not in self.skim_dicts
            tap_skim_dict = self.create_skim_dict("tap")
            self.skim_dicts["tap"] = tap_skim_dict
            # make sure skim has all tap_ids
            assert not (
                tap_skim_dict.offset_mapper.map(self.tap_df["TAP"].values)
                == NOT_IN_SKIM_ZONE_ID
            ).any()

    def create_skim_dict(self, skim_tag):
        """
        Create a new SkimDict of type specified by skim_tag (e.g. 'taz', 'maz' or 'tap')

        Parameters
        ----------
        skim_tag: str

        Returns
        -------
        SkimDict or subclass (e.g. MazSkimDict)
        """
        assert (
            skim_tag not in self.skim_dicts
        )  # avoid inadvertently creating multiple copies

        if skim_tag == "maz":
            # MazSkimDict gets a reference to self here, because it has dependencies on self.load_data
            # (e.g. maz_to_maz_df, maz_taz_df...) We pass in taz_skim_dict as a parameter
            # to hilight the fact that we do not want two copies of its (very large) data array in memory
            assert (
                "taz" in self.skim_dicts
            ), f"create_skim_dict 'maz': backing taz skim_dict not in skim_dicts"
            taz_skim_dict = self.skim_dicts["taz"]
            skim_dict = skim_dictionary.MazSkimDict("maz", self, taz_skim_dict)
        else:
            skim_info = self.skims_info[skim_tag]
            skim_data = self.skim_dict_factory.get_skim_data(skim_tag, skim_info)
            skim_dict = skim_dictionary.SkimDict(skim_tag, skim_info, skim_data)

        logger.debug(f"create_skim_dict {skim_tag} omx_shape {skim_dict.omx_shape}")

        return skim_dict

    def omx_file_names(self, skim_tag):
        """
        Return list of omx file names from network_los settings file for the specified skim_tag (e.g. 'taz')

        Parameters
        ----------
        skim_tag: str (e.g. 'taz')

        Returns
        -------
        list of str
        """
        file_names = self.setting(f"{skim_tag}_skims")
        file_names = [file_names] if isinstance(file_names, str) else file_names
        return file_names

    def multiprocess(self):
        """
        return True if this is a multiprocessing run (even if it is a main or single-process subprocess)

        Returns
        -------
            bool
        """
        is_multiprocess = config.setting("multiprocess", False)
        return is_multiprocess

    def load_shared_data(self, shared_data_buffers):
        """
        Load omx skim data into shared_data buffers
        Only called when multiprocessing - BEFORE any models are run or any call to load_data()

        Parameters
        ----------
        shared_data_buffers: dict of multiprocessing.RawArray keyed by skim_tag
        """

        assert self.multiprocess()
        # assert self.skim_dict_factory.supports_shared_data_for_multiprocessing

        if self.skim_dict_factory.supports_shared_data_for_multiprocessing:
            for skim_tag in self.skims_info.keys():
                assert (
                    skim_tag in shared_data_buffers
                ), f"load_shared_data expected allocated shared_data_buffers"
                self.skim_dict_factory.load_skims_to_buffer(
                    self.skims_info[skim_tag], shared_data_buffers[skim_tag]
                )

        if self.zone_system == THREE_ZONE:
            assert self.tvpb is not None

            if self.rebuild_tvpb_cache and not config.setting("resume_after", None):
                # delete old cache at start of new run so that stale cache is not loaded by load_data_to_buffer
                # when singleprocess, this call is made (later in program flow) in the initialize_los step
                self.tvpb.tap_cache.cleanup()

            self.tvpb.tap_cache.load_data_to_buffer(
                shared_data_buffers[self.tvpb.tap_cache.cache_tag]
            )

    def allocate_shared_skim_buffers(self):
        """
        Allocate multiprocessing.RawArray shared data buffers sized to hold data for the omx skims.
        Only called when multiprocessing - BEFORE load_data()

        Returns dict of allocated buffers so they can be added to mp_tasks can add them to dict of data
        to be shared with subprocesses.

        Note: we are only allocating storage, but not loading any skim data into it

        Returns
        -------
        dict of multiprocessing.RawArray keyed by skim_tag
        """

        assert self.multiprocess()
        assert (
            not self.skim_dicts
        ), f"allocate_shared_skim_buffers must be called BEFORE, not after, load_data"

        skim_buffers = {}

        if self.skim_dict_factory.supports_shared_data_for_multiprocessing:
            for skim_tag in self.skims_info.keys():
                skim_buffers[skim_tag] = self.skim_dict_factory.allocate_skim_buffer(
                    self.skims_info[skim_tag], shared=True
                )

        if self.zone_system == THREE_ZONE:
            assert self.tvpb is not None
            skim_buffers[
                self.tvpb.tap_cache.cache_tag
            ] = self.tvpb.tap_cache.allocate_data_buffer(shared=True)

        return skim_buffers

    def get_skim_dict(self, skim_tag):
        """
        Get SkimDict for the specified skim_tag (e.g. 'taz', 'maz', or 'tap')

        Returns
        -------
        SkimDict or subclass (e.g. MazSkimDict)
        """

        assert (
            skim_tag in self.skim_dicts
        ), f"network_los.get_skim_dict: skim tag '{skim_tag}' not in skim_dicts"
        return self.skim_dicts[skim_tag]

    def get_default_skim_dict(self):
        """
        Get the default (non-transit) skim dict for the (1, 2, or 3) zone_system

        Returns
        -------
        TAZ SkimDict for ONE_ZONE, MazSkimDict for TWO_ZONE and THREE_ZONE
        """
        if self.zone_system == ONE_ZONE:
            return self.get_skim_dict("taz")
        else:
            return self.get_skim_dict("maz")

    def get_mazpairs(self, omaz, dmaz, attribute):
        """
        look up attribute values of maz od pairs in sparse maz_to_maz df

        Parameters
        ----------
        omaz: array-like list of omaz zone_ids
        dmaz: array-like list of omaz zone_ids
        attribute: str name of attribute column in maz_to_maz_df

        Returns
        -------
        Numpy.ndarray: list of attribute values for od pairs
        """

        # # this is slower
        # s = pd.merge(pd.DataFrame({'OMAZ': omaz, 'DMAZ': dmaz}),
        #              self.maz_to_maz_df,
        #              how="left")[attribute]

        # synthetic index method i : omaz_dmaz
        i = np.asanyarray(omaz) * self.maz_ceiling + np.asanyarray(dmaz)
        s = util.quick_loc_df(i, self.maz_to_maz_df, attribute)

        # FIXME - no point in returning series?
        return np.asanyarray(s)

    def get_tappairs3d(self, otap, dtap, dim3, key):
        """
        TAP skim lookup

        FIXME - why do we provide this for taps, but use skim wrappers for TAZ?

        Parameters
        ----------
        otap: pandas.Series
            origin (boarding tap) zone_ids
        dtap: pandas.Series
            dest (aligting tap) zone_ids
        dim3: pandas.Series or str
            dim3 (e.g. tod) str
        key
            skim key (e.g. 'IWAIT_SET1')

        Returns
        -------
            Numpy.ndarray: list of tap skim values for odt tuples
        """

        s = self.get_skim_dict("tap").lookup_3d(otap, dtap, dim3, key)
        return s

    def skim_time_period_label(self, time_period):
        """
        convert time period times to skim time period labels (e.g. 9 -> 'AM')

        Parameters
        ----------
        time_period : pandas Series

        Returns
        -------
        numpy.array
            string time period labels
        """

        assert (
            self.skim_time_periods is not None
        ), "'skim_time_periods' setting not found."

        # Default to 60 minute time periods
        period_minutes = self.skim_time_periods.get("period_minutes", 60)

        # Default to a day
        model_time_window_min = self.skim_time_periods.get("time_window", 1440)

        # Check to make sure the intervals result in no remainder time through 24 hour day
        assert 0 == model_time_window_min % period_minutes
        total_periods = model_time_window_min / period_minutes

        bins = (
            np.digitize(
                [np.array(time_period) % total_periods],
                self.skim_time_periods["periods"],
                right=True,
            )[0]
            - 1
        )
        return np.array(self.skim_time_periods["labels"])[bins]

    def get_tazs(self):
        # FIXME - should compute on init?
        if self.zone_system == ONE_ZONE:
            tazs = inject.get_table("land_use").index.values
        else:
            tazs = self.maz_taz_df.TAZ.unique()
        assert isinstance(tazs, np.ndarray)
        return tazs

    def get_mazs(self):
        # FIXME - should compute on init?
        assert self.zone_system in [TWO_ZONE, THREE_ZONE]
        mazs = self.maz_taz_df.MAZ.values
        assert isinstance(mazs, np.ndarray)
        return mazs

    def get_taps(self):
        # FIXME - should compute on init?
        assert self.zone_system == THREE_ZONE
        taps = self.tap_df.TAP.values
        assert isinstance(taps, np.ndarray)
        return taps

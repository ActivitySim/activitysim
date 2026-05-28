# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from activitysim.core import input, skim_dictionary, util
from activitysim.core.cleaning import recode_based_on_table
from activitysim.core.configuration.network import NetworkSettings, TAZ_Settings
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

DEFAULT_SETTINGS = {
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
      zone_system: str                    # str (ONE_ZONE or TWO_ZONE)
      skim_time_periods = None            # list of str e.g. ['AM', 'MD', 'PM''

      skims_info: dict                    # dict of SkimInfo keyed by skim_tag
      skim_buffers: dict                  # if multiprocessing, dict of multiprocessing.Array buffers keyed by skim_tag
      skim_dicts: dice                    # dict of SkimDict keyed by skim_tag

      # TWO_ZONE
      maz_taz_df: pandas.DataFrame        # DataFrame with two columns, MAZ and TAZ, mapping MAZ to containing TAZ
      maz_to_maz_df: pandas.DataFrame     # maz_to_maz attributes for MazSkimDict sparse skims
                                          # indexed by synthetic omaz/dmaz index for faster get_mazpairs lookup)
      maz_ceiling: int                    # max maz_id + 1 (to compute synthetic omaz/dmaz index by get_mazpairs)
      max_blend_distance: dict            # dict of int maz_to_maz max_blend_distance values keyed by skim_tag

    """

    def __init__(self, state, los_settings_file_name=LOS_SETTINGS_FILE_NAME):
        self.state = state
        # Note: we require all skims to be of same dtype so they can share buffer - is that ok?
        # fixme is it ok to require skims be all the same type? if so, is this the right choice?
        self.skim_dtype_name = "float32"
        self.zone_system = None
        self.skim_time_periods = None
        self.skims_info = {}
        self.skim_dicts = {}

        # TWO_ZONE
        self.maz_taz_df = None
        self.maz_to_maz_df = None
        self.maz_ceiling = None
        self.max_blend_distance = {}

        self.los_settings_file_name = los_settings_file_name
        self.load_settings()
        self.sharrow_enabled = state.settings.sharrow

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

        # load SkimInfo for all skims for this zone_system (TAZ for ONE_ZONE + MAZ for TWO_ZONE)
        self.load_skim_info()

    def get_network_cache_dir(self) -> Path:
        if self.los_settings.network_cache_dir:
            result = self.state.filesystem.get_working_subdir(
                self.los_settings.network_cache_dir
            )
            result.mkdir(parents=True, exist_ok=True)
            return result
        return self.state.filesystem.get_cache_dir()

    def setting(self, keys, default: Any = "<REQUIRED>"):
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
            if isinstance(s, dict):
                s = s.get(key, None)
            else:
                s = getattr(s, key, None)
            if default == "<REQUIRED>" and s is None:
                raise ValueError(
                    f"expected key '{key}' not found in '{keys}' in {self.los_settings_file_name}"
                )
                # assert isinstance(
                #     s, dict
                # ), f"expected key '{key}' not found in '{keys}' in {self.los_settings_file_name}"
        key = key_list[-1]  # last key
        if default == "<REQUIRED>":
            if isinstance(s, dict):
                assert (
                    key in s
                ), f"Expected setting {keys} not found in in {LOS_SETTINGS_FILE_NAME}"
            else:
                assert hasattr(s, key)
        if isinstance(s, dict):
            return s.get(key, default)
        else:
            return getattr(s, key, default)

    def load_settings(self):
        """
        Read setting file and initialize object variables (see class docstring for list of object variables)
        """

        try:
            self.los_settings = NetworkSettings.read_settings_file(
                self.state.filesystem,
                file_name=self.los_settings_file_name,
                mandatory=True,
            )
        except ValidationError as err:
            err_msg = str(err)
            print(err_msg)
            raise
        self.state.network_settings = self.los_settings

        # validate skim_time_periods
        self.skim_time_periods = self.state.network_settings.skim_time_periods

        self.zone_system = self.setting("zone_system")
        assert self.zone_system in [
            ONE_ZONE,
            TWO_ZONE,
        ], f"Network_LOS: unrecognized zone_system: {self.zone_system}"

        if self.zone_system == TWO_ZONE:
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

    def load_skim_info(self):
        """
        read skim info from omx files into SkimInfo, and store in self.skims_info dict keyed by skim_tag

        ONE_ZONE and TWO_ZONE systems have only TAZ skims
        """
        assert self.skim_dict_factory is not None
        # load taz skim_info
        self.skims_info["taz"] = self.skim_dict_factory.load_skim_info(
            self.state, "taz"
        )

    def load_data(self):
        """
        Load tables and skims from files specified in network_los settigns
        """

        # load maz tables
        if self.zone_system == TWO_ZONE:
            # maz
            file_name = self.setting("maz")
            self.maz_taz_df = input.read_input_file(
                self.state.filesystem.get_data_file_path(
                    file_name,
                    mandatory=True,
                    alternative_suffixes=(".csv.gz", ".parquet"),
                )
            )
            self.maz_taz_df = self.maz_taz_df[["MAZ", "TAZ"]].sort_values(
                by="MAZ"
            )  # only fields we need

            # recode MAZs if needed
            self.maz_taz_df["MAZ"] = recode_based_on_table(
                self.state, self.maz_taz_df["MAZ"], "land_use"
            )
            self.maz_taz_df["TAZ"] = recode_based_on_table(
                self.state, self.maz_taz_df["TAZ"], "land_use_taz"
            )

            self.maz_ceiling = self.maz_taz_df.MAZ.max() + 1

            # maz_to_maz_df
            maz_to_maz_tables = self.setting("maz_to_maz.tables")
            maz_to_maz_tables = (
                [maz_to_maz_tables]
                if isinstance(maz_to_maz_tables, str)
                else maz_to_maz_tables
            )
            for file_name in maz_to_maz_tables:
                df = input.read_input_file(
                    self.state.filesystem.get_data_file_path(
                        file_name,
                        mandatory=True,
                        alternative_suffixes=(".csv.gz", ".parquet"),
                    )
                )

                # recode MAZs if needed
                df["OMAZ"] = recode_based_on_table(self.state, df["OMAZ"], "land_use")
                df["DMAZ"] = recode_based_on_table(self.state, df["DMAZ"], "land_use")

                if self.maz_ceiling > (1 << 31):
                    raise ValueError("maz ceiling too high, will overflow int64")
                elif self.maz_ceiling > 32767:
                    # too many MAZs, or un-recoded MAZ ID's that are too large
                    # will overflow a 32-bit index, so upgrade to 64bit.
                    df["i"] = df.OMAZ.astype(np.int64) * np.int64(
                        self.maz_ceiling
                    ) + df.DMAZ.astype(np.int64)
                else:
                    df["i"] = df.OMAZ.astype(np.int32) * np.int32(
                        self.maz_ceiling
                    ) + df.DMAZ.astype(np.int32)
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

        # create taz skim dict
        if not self.sharrow_enabled:
            assert "taz" not in self.skim_dicts
            # If offset_preprocessing was completed, then TAZ values
            # will be pre-offset and there's no need to re-offset them.
            if self.state.settings.offset_preprocessing:
                _override_offset_int = 0
            else:
                _override_offset_int = None
            self.skim_dicts["taz"] = self.create_skim_dict(
                "taz", _override_offset_int=_override_offset_int
            )
            # make sure skim has all taz_ids
            # FIXME - weird that there is no list of tazs?
        else:
            self.skim_dicts["taz"] = self.get_skim_dict("taz")

        # create MazSkimDict facade
        if self.zone_system == TWO_ZONE:
            if not self.sharrow_enabled:
                # create MazSkimDict facade skim_dict
                # (must have already loaded dependencies: taz skim_dict, maz_to_maz_df, and maz_taz_df)
                assert "maz" not in self.skim_dicts
                maz_skim_dict = self.create_skim_dict("maz")
                self.skim_dicts["maz"] = maz_skim_dict

                # make sure skim has all maz_ids
                assert not (
                    maz_skim_dict.offset_mapper.map(self.maz_taz_df["MAZ"].values)
                    == NOT_IN_SKIM_ZONE_ID
                ).any(), (
                    "every MAZ in the MAZ-to-TAZ mapping must map to a TAZ that exists"
                )
            else:
                self.skim_dicts["maz"] = self.get_skim_dict("maz")
                # TODO:SHARROW: make sure skim has all maz_ids

        # check that the number of rows in land_use_taz matches the number of zones in the skims
        if "land_use_taz" in self.state:
            skims = self.get_skim_dict("taz")
            if hasattr(skims, "zone_ids"):  # SkimDict
                assert len(skims.zone_ids) == len(
                    self.state.get_dataframe("land_use_taz")
                )
            else:  # SkimDataset
                assert len(skims.dataset.indexes["otaz"]) == len(
                    self.state.get_dataframe("land_use_taz")
                ), (
                    f"land_use_taz table length {len(self.state.get_dataframe('land_use_taz'))} does not match "
                    f"taz skim length {len(skims.dataset.indexes['otaz'])}"
                )

    def create_skim_dict(self, skim_tag, _override_offset_int=None):
        """
        Create a new SkimDict of type specified by skim_tag (e.g. 'taz', or 'maz')

        Parameters
        ----------
        skim_tag : str
        _override_offset_int : int, optional
            Override the offset int for this dictionary.  Use this to set that
            offset to zero when zone id's have been pre-processed to be zero-based
            contiguous integers.

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
            skim_dict = skim_dictionary.MazSkimDict(
                self.state, "maz", self, taz_skim_dict
            )
        else:
            skim_info = self.skims_info[skim_tag]
            skim_data = self.skim_dict_factory.get_skim_data(skim_tag, skim_info)
            skim_dict = skim_dictionary.SkimDict(
                self.state, skim_tag, skim_info, skim_data
            )

        logger.debug(f"create_skim_dict {skim_tag} omx_shape {skim_dict.omx_shape}")

        if _override_offset_int is not None:
            skim_dict.offset_mapper.set_offset_int(
                _override_offset_int
            )  # default is -1

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
        if isinstance(file_names, TAZ_Settings):
            file_names = file_names.omx
        if isinstance(file_names, dict):
            for i in ("file", "files", "omx"):
                if i in file_names:
                    file_names = file_names[i]
                    break
        if isinstance(file_names, dict):
            raise ValueError(
                f"must specify `{skim_tag}_skims.file` in network_los settings file"
            )
        file_names = [file_names] if isinstance(file_names, str) else file_names
        return file_names

    def zarr_file_name(self, skim_tag):
        """
        Return zarr directory name from network_los settings file for the specified skim_tag (e.g. 'taz')

        Parameters
        ----------
        skim_tag: str (e.g. 'taz')

        Returns
        -------
        str
        """
        skim_setting = self.setting(f"{skim_tag}_skims")
        if isinstance(skim_setting, dict):
            return skim_setting.get("zarr", None)
        elif isinstance(skim_setting, TAZ_Settings):
            return skim_setting.zarr
        else:
            return None

    def zarr_pre_encoding(self, skim_tag):
        """
        Return digital encoding pre-processing before writing to zarr for the specified skim_tag (e.g. 'taz')

        Parameters
        ----------
        skim_tag: str (e.g. 'taz')

        Returns
        -------
        list or None
        """
        skim_setting = self.setting(f"{skim_tag}_skims")
        if isinstance(skim_setting, dict):
            return skim_setting.get("zarr-digital-encoding", None)
        else:
            return None

    def skim_backing_store(self, skim_tag):
        name = self.setting("name", "unnamed")
        return self.setting(
            f"{skim_tag}_skims.backend", f"shared_memory_{skim_tag}_{name}"
        )

    def skim_max_float_precision(self, skim_tag):
        return self.setting(f"{skim_tag}_skims.max_float_precision", 32)

    def skim_digital_encoding(self, skim_tag):
        return self.setting(f"{skim_tag}_skims.digital-encoding", [])

    def multiprocess(self):
        """
        return True if this is a multiprocessing run (even if it is a main or single-process subprocess)

        Returns
        -------
            bool
        """
        is_multiprocess = self.state.settings.multiprocess
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

        return skim_buffers

    def get_skim_dict(self, skim_tag):
        """
        Get SkimDict for the specified skim_tag (e.g. 'taz', or 'maz')

        Returns
        -------
        SkimDict or subclass (e.g. MazSkimDict) or SkimDataset
        """
        sharrow_enabled = self.sharrow_enabled
        if sharrow_enabled and skim_tag in ("taz", "maz"):
            # non-global import avoids circular references
            from .skim_dataset import SkimDataset

            skim_dataset = self.state.get_injectable("skim_dataset")
            if skim_tag == "maz":
                return SkimDataset(skim_dataset)
            else:
                dropdims = ["omaz", "dmaz"]
                skim_dataset = skim_dataset.drop_dims(dropdims, errors="ignore")
                for dd in dropdims:
                    if f"dim_redirection_{dd}" in skim_dataset.attrs:
                        del skim_dataset.attrs[f"dim_redirection_{dd}"]
                return SkimDataset(skim_dataset)
        else:
            assert (
                skim_tag in self.skim_dicts
            ), f"network_los.get_skim_dict: skim tag '{skim_tag}' not in skim_dicts"
            return self.skim_dicts[skim_tag]

    def get_default_skim_dict(self):
        """
        Get the default (non-transit) skim dict for the (1, 2, or 3) zone_system

        Returns
        -------
        TAZ SkimDict for ONE_ZONE, MazSkimDict for TWO_ZONE
        """
        if self.zone_system == ONE_ZONE:
            return self.get_skim_dict("taz")
        else:
            # TODO:SHARROW: taz and maz are the same
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
        if self.maz_ceiling > 32767:
            # too many MAZs, or un-recoded MAZ ID's that are too large
            # will overflow a 32-bit index, so upgrade to 64bit.
            i = np.asanyarray(omaz, dtype=np.int64) * np.int64(
                self.maz_ceiling
            ) + np.asanyarray(dmaz, dtype=np.int64)
        else:
            # if we have less than a 32-bit index, it will
            # overflow so we need to upgrade to at least 32 bit
            omaz_as_array = np.asanyarray(omaz)
            if omaz_as_array.dtype not in (np.int32, np.int64):
                omaz_as_array = omaz_as_array.astype(np.int32)
            dmaz_as_array = np.asanyarray(dmaz)
            if dmaz_as_array.dtype not in (np.int32, np.int64):
                dmaz_as_array = dmaz_as_array.astype(np.int32)
            i = omaz_as_array * self.maz_ceiling + dmaz_as_array
        s = util.quick_loc_df(i, self.maz_to_maz_df, attribute)

        # FIXME - no point in returning series?
        return np.asanyarray(s)

    def skim_time_period_label(
        self, time_period, fillna=None, as_cat=False, broadcast_to=None
    ):
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

        assert (
            self.skim_time_periods is not None
        ), "'skim_time_periods' setting not found."

        # Default to 60 minute time periods
        period_minutes = self.skim_time_periods.period_minutes

        # Default to a day
        model_time_window_min = self.skim_time_periods.time_window

        # Check to make sure the intervals result in no remainder time through 24 hour day
        assert 0 == model_time_window_min % period_minutes
        total_periods = model_time_window_min / period_minutes

        try:
            time_label_dtype = self.skim_dicts["taz"].time_label_dtype
        except (KeyError, AttributeError):
            # if the "taz" skim_dict is missing, or if using old SkimDict
            # instead of SkimDataset, this labeling shortcut is unavailable.
            time_label_dtype = str
            as_cat = False

        # FIXME - eventually test and use np version always?
        if np.isscalar(time_period):
            bin = (
                np.digitize(
                    [time_period % total_periods],
                    self.skim_time_periods.periods,
                    right=True,
                )[0]
                - 1
            )
            if fillna is not None:
                default = self.skim_time_periods.labels[fillna]
                result = self.skim_time_periods.labels.get(bin, default=default)
            else:
                result = self.skim_time_periods.labels[bin]
            if broadcast_to is not None:
                result = pd.Series(
                    data=result,
                    index=broadcast_to,
                    dtype=time_label_dtype if as_cat else str,
                )
        else:
            result = pd.cut(
                time_period,
                self.skim_time_periods.periods,
                labels=self.skim_time_periods.labels,
                ordered=False,
            )
            if fillna is not None:
                default = self.skim_time_periods.labels[fillna]
                result = result.fillna(default)
            if as_cat:
                result = result.astype(time_label_dtype)
            else:
                result = result.astype(str)
        return result

    def get_tazs(self, state):
        # FIXME - should compute on init?
        if self.zone_system == ONE_ZONE:
            tazs = state.get_dataframe("land_use").index.values
        else:
            try:
                land_use_taz = state.get_dataframe("land_use_taz")
            except (RuntimeError, KeyError):
                # land_use_taz is missing, use fallback
                tazs = self.maz_taz_df.TAZ.unique()
            else:
                if "_original_TAZ" in land_use_taz:
                    tazs = land_use_taz["_original_TAZ"].values
                else:
                    tazs = self.maz_taz_df.TAZ.unique()
        assert isinstance(tazs, np.ndarray)
        return tazs

    def get_mazs(self):
        # FIXME - should compute on init?
        assert self.zone_system == TWO_ZONE
        mazs = self.maz_taz_df.MAZ.values
        assert isinstance(mazs, np.ndarray)
        return mazs

    def get_maz_to_taz_series(self, state):
        """
        pd.Series: Index is the MAZ, value is the corresponding TAZ
        """
        if self.sharrow_enabled:
            # FIXME:SHARROW - this assumes that both MAZ and TAZ have been recoded to
            #                 zero-based indexes, but what if that was not done?
            #                 Should we check it and error out here or bravely march forward?
            skim_dataset = state.get_injectable("skim_dataset")
            maz_to_taz = skim_dataset["_digitized_otaz_of_omaz"].to_series()
        else:
            maz_to_taz = self.maz_taz_df[["MAZ", "TAZ"]].set_index("MAZ").TAZ
        return maz_to_taz

    def map_maz_to_taz(self, s):
        """
        Convert MAZ's to TAZ's

        Parameters
        ----------
        s : Array-like
            Integer MAZ values

        Returns
        -------
        pd.Series
            Integer TAZ values
        """
        if not isinstance(s, (pd.Series, pd.Index)):
            s = pd.Series(s)
            input_was_series = False
        else:
            input_was_series = True
        out = s.map(self.get_maz_to_taz_series(self.state))
        if np.issubdtype(out, np.floating):
            if out.isna().any():
                raise KeyError("failed in mapping MAZ to TAZ")
            else:
                out = out.astype(np.int32)
        if input_was_series:
            return out
        else:
            return out.to_numpy()

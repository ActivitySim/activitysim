# ActivitySim
# See full license in LICENSE.txt.

from builtins import range
from builtins import object

import logging

from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core import skim
from activitysim.core import los

logger = logging.getLogger(__name__)


class MazSkimDictFacade(object):
    """
    Facade for SkimDict to override offset_mapper to map maz ids to taz ids
    """

    def __init__(self, network_los):

        taz_skim_dict = network_los.get_skim_dict('taz')
        maz_to_taz_df = network_los.maz_to_taz_df

        print(f"MazSkimDictFacade shape {taz_skim_dict.skim_data.shape}")

        offset_series = taz_skim_dict.offset_mapper.offset_series
        if offset_series is None:
            offset_int = taz_skim_dict.offset_mapper.offset_int or 0
            print(f"fake offset_series {offset_int}")
            offsets = np.arange(taz_skim_dict.skim_data.shape[0])
            offset_series = pd.Series(data=offsets, index=offsets + offset_int)

        # taz_skim_dict offset_series has taz_id index and 0-based offset values for each taz_id
        # we need offset_series with maz_id index and 0-based offset values for each maz_id
        maz_offset_df = pd.merge(offset_series.to_frame('offset'), maz_to_taz_df, left_index=True, right_on='TAZ')

        offset_series = pd.Series(data=maz_offset_df['offset'].values, index=maz_offset_df['MAZ'].values)
        # print(f"offset_series\n{offset_series.head(10)}")

        self.taz_skim_dict = taz_skim_dict
        self.offset_mapper = skim.OffsetMapper(offset_series)

    def get_skim_info(self, key):
        return self.taz_skim_dict.get_skim_info(key)

    def get_skim_data(self):
        return self.taz_skim_dict.get_skim_data()

    def get_skim_usage(self):
        return self.taz_skim_dict.get_skim_usage()

    def get(self, key):
        taz_skim_wrapper = self.taz_skim_dict.get(key)
        taz_skim_wrapper.offset_mapper = self.offset_mapper
        return taz_skim_wrapper

    def wrap(self, left_key, right_key):
        return skim.SkimDictWrapper(self, left_key, right_key)


class MazSparseSkimWrapper(object):
    """
    Container for skim arrays.

    Parameters
    ----------
    data : 2D array
    offset : int, optional
        An optional offset that will be added to origin/destination
        values to turn them into array indices.
        For example, if zone IDs are 1-based, an offset of -1
        would turn them into 0-based array indices.

    """
    def __init__(self, network_los, key, backstop_skim_dict):

        self.network_los = network_los
        self.key = key
        self.backstop = backstop_skim_dict

    def get(self, orig, dest):
        """
        Get impedence values for a set of origin, destination pairs.

        Parameters
        ----------
        orig : 1D array
        dest : 1D array

        Returns
        -------
        values : numpy 1D array
        """

        # fixme - remove?
        assert not (np.isnan(orig) | np.isnan(dest)).any()

        result = self.network_los.get_mazpairs(orig, dest, self.key)

        is_nan = np.isnan(result)

        if is_nan.any():
            # print(f"{is_nan.sum()} nans out of {len(is_nan)} for key '{self.key}")

            backstop_values = self.backstop.get(self.key).get(orig, dest)
            result = np.where(is_nan, backstop_values, result)

        return result


class MazSkimDict(object):
    """

    """

    def __init__(self, network_los):

        self.network_los = network_los
        assert network_los.maz_to_maz_df is not None
        self.sparse_keys = list(set(network_los.maz_to_maz_df.columns) - {'OMAZ', 'DMAZ'})
        self.sparse_key_usage = set()

        self.skim_dict_facade = MazSkimDictFacade(network_los)

    def get_taz_skim_dict(self):
        return self.skim_dict_facade

    def get_skim_info(self, key):
        return self.skim_dict_facade.get_skim_info(key)

    def get_skim_usage(self):

        # print(f"sparse_key_usage: {self.sparse_key_usage}")
        # print(f"backstop_usage: {self.skim_dict_facade.get_skim_usage()}")

        return self.sparse_key_usage.union(self.skim_dict_facade.get_skim_usage())

    def get(self, key):

        if key in self.sparse_keys:
            # logger.debug(f"MazSkimDict using SparseSkimDict for key '{key}'")
            self.sparse_key_usage.add(key)
            skim_wrapper = MazSparseSkimWrapper(self.network_los, key, backstop_skim_dict=self.skim_dict_facade)
        else:
            # logger.debug(f"MazSkimDict using MazSkimDictFacade for key '{key}'")
            skim_wrapper = self.skim_dict_facade.get(key)

        return skim_wrapper

    def wrap(self, left_key, right_key):
        return skim.SkimDictWrapper(self, left_key, right_key)

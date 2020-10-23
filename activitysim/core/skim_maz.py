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


def maz_taz_mapper(network_los):

    # return series with MAZ zone_id index and TAZ zone_id values to map MAZ to TAZ zone ids
    return network_los.maz_taz_df[['MAZ', 'TAZ']].set_index('MAZ').sort_values(by='TAZ').TAZ


def maz_taz_offset_mapper(network_los):

    # we want to build a series with MAZ zone_id index and TAZ skim array offset values

    # start with a series with MAZ zone_id index and TAZ zone id values
    maz_to_taz = maz_taz_mapper(network_los)

    # use taz_skim_dict offset_mapper to create series mapping directly from MAZ to TAZ skim index
    taz_skim_dict = network_los.get_skim_dict('taz')
    maz_zone_id_to_taz_skim_index = taz_skim_dict.offset_mapper.map(maz_to_taz)

    offset_mapper = skim.OffsetMapper(offset_series=maz_zone_id_to_taz_skim_index)

    return offset_mapper


class MazSkimDictFacade(object):
    """
    Duck subclass of taz SkimDict but takes maz orig,dest instead of taz orig,dest
    Maps maz to taz on the fly using an override offset_mapper to map maz ids to taz ids
    Returns taz skim results
    """

    def __init__(self, network_los):

        # offset mapper to map maz orig, dest series into offsets into taz_skim_dict
        self.offset_mapper = maz_taz_offset_mapper(network_los)

        self.taz_skim_dict = network_los.get_skim_dict('taz')

    def has_key(self, key):
        return self.taz_skim_dict.has_key(key)

    def get_skim_info(self, key):
        return self.taz_skim_dict.get_skim_info(key)

    def get_skim_data(self):
        return self.taz_skim_dict.get_skim_data()

    def get_skim_usage(self):
        return self.taz_skim_dict.get_skim_usage()

    def get(self, key):
        taz_skim_wrapper = self.taz_skim_dict.get(key)
        # patch taz_skim_wrapper with offset mapper to map maz orig,dest series directly to taz skim_dict offsets
        taz_skim_wrapper.offset_mapper = self.offset_mapper
        return taz_skim_wrapper

    def wrap(self, orig_key, dest_key):
        return skim.SkimDictWrapper(self, orig_key, dest_key)


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
        self.max_blend_distance = network_los.max_blend_distance.get(key, 0)

        #bug
        # self.blend_distance_skim_name = network_los.blend_distance_skim_name
        # if (self.max_blend_distance == 0) or (self.blend_distance_skim_name == key):
        #     self.blend_distance_skim_name = None

        if self.max_blend_distance == 0:
            self.blend_distance_skim_name = None
        else:
            self.blend_distance_skim_name = network_los.blend_distance_skim_name

        # FIXME should we deduce this, config it?
        self.skim_data_type = backstop_skim_dict.get_skim_info('dtype')

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

        # want to return same type as backstop skim
        values = self.network_los.get_mazpairs(orig, dest, self.key).astype(self.skim_data_type)

        is_nan = np.isnan(values)

        if self.max_blend_distance > 0:

            # print(f"{is_nan.sum()} nans out of {len(is_nan)} for key '{self.key}")
            # print(f"blend_distance_skim_name {self.blend_distance_skim_name}")

            backstop_values = self.backstop.get(self.key).get(orig, dest)

            # get distance skim if a different key was specified by blend_distance_skim_name
            if (self.blend_distance_skim_name != self.key):
                distance = self.network_los.get_mazpairs(orig, dest, self.blend_distance_skim_name)
            else:
                distance = values

            # for distances less than max_blend_distance, we blend maz-maz and skim backstop values
            # shorter distances have less fractional backstop, and more maz-maz
            # beyond max_blend_distance, just use the skim values
            backstop_fractions = np.minimum(distance / self.max_blend_distance, 1)

            values = np.where(is_nan,
                              backstop_values,
                              backstop_fractions * backstop_values + (1 - backstop_fractions) * values)

        elif is_nan.any():

            # print(f"{is_nan.sum()} nans out of {len(is_nan)} for key '{self.key}")

            if self.backstop.has_key(self.key):
                # replace nan values using simple backstop without blending
                backstop_values = self.backstop.get(self.key).get(orig, dest)
                values = np.where(is_nan, backstop_values, values)
            else:
                #bug
                # FIXME - if no backstop skim, then return 0 (which conventionally means "not available")
                values = np.where(is_nan, 0, values)

        return values


class MazSkimDict(object):
    """
    Duck subclass of taz SkimDict that
    returns MazSparseSkimWrapper for skim keys when sparse maz skim data is available
    returns MazSkimDictFacade for skim keys when only backstop taz skims are available
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

    def wrap(self, orig_key, dest_key):
        return skim.SkimDictWrapper(self, orig_key, dest_key)


class MazSkimStackFacade(object):

    def __init__(self, network_los):

        self.taz_skim_stack = network_los.get_skim_stack('taz')

        # offset mapper to map MAZ zone_ids to TAZ zone_ids
        self.xmaz_taz_mapper = skim.OffsetMapper(offset_series=maz_taz_mapper(network_los))

        #bug

    def touch(self, key):
        self.taz_skim_stack.usage.add(key)

    def lookup(self, orig, dest, dim3, key):

        #bug
        #print(f"maz orig: {orig}")
        #print(f"maz dest: {dest}")

        orig = self.xmaz_taz_mapper.map(orig)
        dest = self.xmaz_taz_mapper.map(dest)

        #print(f"taz orig: {orig}")
        #print(f"taz dest: {dest}")
        #bug

        result = self.taz_skim_stack.lookup(orig, dest, dim3, key)

        return result

    def wrap(self, orig_key, dest_key, dim3_key):
        """
        return a SkimStackWrapper for self
        """
        return skim.SkimStackWrapper(stack=self, orig_key=orig_key, dest_key=dest_key, dim3_key=dim3_key)

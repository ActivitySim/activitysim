# ActivitySim
# See full license in LICENSE.txt.

from builtins import range
from builtins import object

import logging

from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core.util import quick_loc_series

logger = logging.getLogger(__name__)

NOT_IN_SKIM_ZONE_ID = -1
NOT_IN_SKIM_NAN = np.nan

ROW_MAJOR_LAYOUT = True


class SkimData(object):
    def __init__(self, skim_data):
        self._skim_data = skim_data

    def __getitem__(self, indexes):
        if len(indexes) != 3:
            raise ValueError(f'number of indexes ({len(indexes)}) should be 3')
        return self._skim_data[indexes]

    @property
    def shape(self):
        return self._skim_data.shape

    # def close(self):
        # self._skim_data._mmap.close()
        # del self._skim_data



class OffsetMapper(object):
    """
    Utility to map skim zone ids to ordinal offsets (e.g. numpy array indices)

    Can map either by a fixed offset (e.g. -1 to map 1-based to 0-based)
    or by an explicit mapping of zone id to offset (slower but more flexible)

    offset_int: int which when added to zone_id yields skim array index
    (e.g. offset_int of 1-based zone ids to 0-based array indices)

    offset_int:
        int offset which when added to zone_id yields skim array index (e.g. -1 to map 1-based to 0-based)
    offset_list:
        list the same size as target skim dimension with zone_id values corresponding to as the skim array index
    offset_series:
        pandas series with zone_id index and skim array offset values (can map many zone_ids to skim array index)
    """

    def __init__(self, offset_int=None, offset_list=None, offset_series=None):

        self.offset_int = self.offset_series = None

        assert (offset_int is not None) + (offset_list is not None) + (offset_series is not None) <= 1

        if offset_int is not None:
            self.set_offset_int(offset_int)
        elif offset_list is not None:
            self.set_offset_list(offset_list)
        elif offset_series is not None:
            self.set_offset_series(offset_series)

    def print_offset(self, message=''):
        assert (self.offset_int is not None) or (self.offset_series is not None)

        if self.offset_int is not None:
            print(f"{message} offset_int: {self.offset_int}")
        elif self.offset_series is not None:
            print(f"{message} offset_series:\n {self.offset_series}")
        else:
            print(f"{message} offset: None")

    def set_offset_series(self, offset_series):
        """
        offset_series: pandas.Series
            series with zone_id index and skim array offset values (can map many zone_ids to skim array index)
        """
        assert isinstance(offset_series, pd.Series)
        self.offset_series = offset_series
        self.offset_int = None

    def set_offset_list(self, offset_list):
        """
        offset_list
            list the same size as target skim dimension with zone_id values corresponding to as the skim array index

        set_offset_list([10, 20, 30, 40])
        map([30, 10, 40])
        returns offsets [2, 0, 3]

        Parameters
        ----------
        offset_list : list of int
        """
        assert isinstance(offset_list, list)

        # - for performance, check if this is a simple range that can ber represented by an int offset
        first_offset = offset_list[0]
        if (offset_list == list(range(first_offset, len(offset_list)+first_offset))):
            offset_int = -1 * first_offset
            self.set_offset_int(offset_int)
            return

        if self.offset_series is None:
            offset_series = pd.Series(data=list(range(len(offset_list))), index=offset_list)
            self.set_offset_series(offset_series)
        else:
            # make sure offsets are the same
            assert (offset_list == self.offset_series.index).all()
            #FIXME - does this ever happen?
            bug

    def set_offset_int(self, offset_int):
        """
        specify int offset which when added to zone_id yields skim array index (e.g. -1 to map 1-based to 0-based)

        Parameters
        ----------
        offset_int : int

        """
        # should be some duck subtype of integer (but might be, say, numpy.int64)
        assert int(offset_int) == offset_int

        self.offset_int = int(offset_int)
        self.offset_series = None

    def map(self, zone_ids):
        """
        map zone_ids to offsets

        Parameters
        ----------
        zone_ids

        Returns
        -------
        offsets : numpy array of int
        """

        if self.offset_series is not None:
            assert(self.offset_int is None)
            assert isinstance(self.offset_series, pd.Series)
            #FIXME - faster to use series.map if zone_ids is a series?
            offsets = quick_loc_series(zone_ids, self.offset_series).fillna(NOT_IN_SKIM_ZONE_ID).astype(int)

        elif self.offset_int:
            assert (self.offset_series is None)
            offsets = zone_ids + self.offset_int
        else:
            offsets = zone_ids

        return offsets



class SkimDict(object):
    """
    A SkimDict object is a wrapper around a dict of multiple skim objects,
    where each object is identified by a key.

    Note that keys are either strings or tuples of two strings (to support stacking of skims.)
    """

    def __init__(self, skim_tag, skim_info, skim_data):

        logger.info(f"SkimDict init {skim_tag}")

        self.skim_tag = skim_tag
        self.skim_info = skim_info
        self.usage = set()

        self.offset_mapper = self._offset_mapper()  # (in function so subclass can override)

        self.omx_shape = skim_info['omx_shape']
        self.skim_data = skim_data
        self.dtype = np.dtype(skim_info['dtype_name'])  # so we can coerce if we have missing values

        # - skim_dim3 dict maps key1 to dict of key2 absolute offsets into block
        # DRV_COM_WLK_BOARDS: {'MD': 4, 'AM': 3, 'PM': 5}, ...
        block_offsets = skim_info['block_offsets']
        self.skim_dim3 = {}
        for skim_key, offset in block_offsets.items():
            if isinstance(skim_key, tuple):
                key1, key2 = skim_key
                self.skim_dim3.setdefault(key1, {})[key2] = offset
        logger.info(f"SkimDict.build_3d_skim_block_offset_table registered {len(self.skim_dim3)} 3d keys")

    def _offset_mapper(self):
        offset_mapper = OffsetMapper()
        offset_map = self.skim_info.get('offset_map', None)
        if offset_map is not None:
            #logger.debug(f"SkimDict _offset_mapper {self.skim_info['skim_tag']} offset_map: {offset_map}")
            offset_mapper.set_offset_list(offset_list=offset_map)
        else:
            # assume this is a one-based skim map
            offset_mapper.set_offset_int(-1)

        return offset_mapper

    def get_skim_usage(self):
        return self.usage

    def touch(self, key):
        self.usage.add(key)

    def _lookup(self, orig, dest, block_offsets):

        # fixme - remove?
        assert not (np.isnan(orig) | np.isnan(dest)).any()

        # only working with numpy in here
        orig = np.asanyarray(orig).astype(int)
        dest = np.asanyarray(dest).astype(int)

        mapped_orig = self.offset_mapper.map(orig)
        mapped_dest = self.offset_mapper.map(dest)
        if ROW_MAJOR_LAYOUT:
            result = self.skim_data[block_offsets, mapped_orig, mapped_dest]
        else:
            result = self.skim_data[mapped_orig, mapped_dest, block_offsets]

        # FIXME - should return nan if not in skim (negative indices wrap around)
        in_skim = (mapped_orig >= 0) & (mapped_orig < self.omx_shape[0]) & \
                  (mapped_dest >= 0) & (mapped_dest < self.omx_shape[1])

        # check for bad indexes (other than NOT_IN_SKIM_ZONE_ID)
        assert (in_skim | (orig == NOT_IN_SKIM_ZONE_ID) | (dest == NOT_IN_SKIM_ZONE_ID)).all(), \
            f"{(~in_skim).sum()} od pairs not in skim"

        if not in_skim.all():
            result = np.where(in_skim, result, NOT_IN_SKIM_NAN).astype(self.dtype)

        return result

    def lookup(self, orig, dest, key):

        block_offset = self.skim_info['block_offsets'].get(key)
        assert block_offset is not None, f"SkimDict lookup key '{key}' not in skims"

        self.touch(key)

        try:
            result = self._lookup(orig, dest, block_offset)
        except Exception as err:
            logger.error("SkimDict lookup error: %s: %s", type(err).__name__, str(err))
            logger.error(f"key {key}")
            logger.error(f"orig max {orig.max()} min {orig.min()}")
            logger.error(f"dest max {dest.max()} min {dest.min()}")
            raise err

        return result

    def lookup_3d(self, orig, dest, dim3, key):

        assert key in self.skim_dim3, f"3d skim key {key} not in skims."

        # map dim3 to block_offsets
        skim_keys_to_indexes = self.skim_dim3[key]

        # skim_indexes = dim3.map(skim_keys_to_indexes).astype('int')
        try:
            block_offsets = np.vectorize(skim_keys_to_indexes.get)(dim3)  # this should be faster than map
            result = self._lookup(orig, dest, block_offsets)
        except Exception as err:
            logger.error("SkimDict lookup_3d error: %s: %s", type(err).__name__, str(err))
            logger.error(f"key {key}")
            logger.error(f"orig max {orig.max()} min {orig.min()}")
            logger.error(f"dest max {dest.max()} min {dest.min()}")
            logger.error(f"skim_keys_to_indexes: {skim_keys_to_indexes}")
            logger.error(f"dim3 {np.unique(dim3)}")
            logger.error(f"dim3 block_offsets {np.unique(block_offsets)}")
            raise err

        return result

    def wrap(self, orig_key, dest_key):
        """
        return a SkimWrapper for self
        """
        return SkimWrapper(self, orig_key, dest_key)

    def wrap_3d(self, orig_key, dest_key, dim3_key):
        """
        return a SkimWrapper for self
        """
        return Skim3dWrapper(self, orig_key, dest_key, dim3_key)


class SkimWrapper(object):
    """
    A SkimWrapper object is an access wrapper around a SkimDict of multiple skim objects,
    where each object is identified by a key.  It operates like a
    dictionary - i.e. use brackets to add and get skim objects - but also
    has information on how to lookup against the skim objects.
    Specifically, this object has a dataframe, an orig_key and dest_key.
    It is assumed that orig_key and dest_key identify columns in df.  The
    parameter df is usually set by the simulation itself as it's a result of
    interacting choosers and alternatives.

    When the user calls skims[key], key is an identifier for which skim
    to use, and the object automatically looks up impedances of that skim
    using the specified orig_key column in df as the origin and
    the dest_key column in df as the destination.  In this way, the user
    does not do the O-D lookup by hand and only specifies which skim to use
    for this lookup.  This is the only purpose of this object: to
    abstract away the O-D lookup and use skims by specifying which skim
    to use in the expressions.

    Note that keys are either strings or tuples of two strings (to support stacking of skims.)
    """

    def __init__(self, skim_dict, orig_key, dest_key):
        self.skim_dict = skim_dict
        self.orig_key = orig_key
        self.dest_key = dest_key
        self.df = None

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitiate chaining)
        """
        self.df = df
        return self

    def lookup(self, key, reverse=False):
        """
        Generally not called by the user - use __getitem__ instead

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        od : bool (optional)
            od=True means lookup standard origin-destination skim value
            od=False means lookup destination-origin skim value

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """

        assert self.df is not None, "Call set_df first"

        if reverse:
            s = self.skim_dict.lookup(self.df[self.dest_key], self.df[self.orig_key], key)
        else:
            s = self.skim_dict.lookup(self.df[self.orig_key], self.df[self.dest_key], key)

        return pd.Series(s, index=self.df.index)

    def reverse(self, key):
        """
        return skim value in reverse (d-o) direction
        """
        return self.lookup(key, reverse=True)

    def max(self, key):
        """
        return max skim value in either o-d or d-o direction
        """

        #skim_wrapper = self.skim_dict.get(key)

        assert self.df is not None, "Call set_df first"

        s = np.maximum(
            self.skim_dict.lookup(self.df[self.dest_key], self.df[self.orig_key], key),
            self.skim_dict.lookup(self.df[self.orig_key], self.df[self.dest_key], key)
        )

        return pd.Series(s, index=self.df.index)

    def __getitem__(self, key):
        """
        Get the (df implicit) lookup for an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for the skim object

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """

        return self.lookup(key)


class Skim3dWrapper(object):
    """

    you can say something like out_skim['SOV'] and it will automatically dereference the 3D matrix
    using origin, destination, and time of day.

    Parameters
    ----------
    skims: Skims
        This is the Skims object to wrap
    dim3_key : str
        This identifies the column in the dataframe which is used to
        select among Skim object using the SECOND item in each tuple (see
        above for a more complete description)
    """

    def __init__(self, skim_dict, orig_key, dest_key, dim3_key):

        self.skim_dict = skim_dict

        self.orig_key = orig_key
        self.dest_key = dest_key
        self.dim3_key = dim3_key
        self.df = None

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitiate chaining)
        """
        self.df = df
        return self

    def __getitem__(self, key):
        """
        Get an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """

        assert self.df is not None, "Call set_df first"
        orig = self.df[self.orig_key].astype('int')
        dest = self.df[self.dest_key].astype('int')
        dim3 = self.df[self.dim3_key]

        skim_values = self.skim_dict.lookup_3d(orig, dest, dim3, key)

        return pd.Series(skim_values, self.df.index)


class MazSkimDict(SkimDict):

    def __init__(self, skim_tag, network_los):

        self.network_los = network_los

        taz_skim_dict = network_los.get_skim_dict('taz')
        super().__init__(skim_tag, taz_skim_dict.skim_info, taz_skim_dict.skim_data)
        assert self.offset_mapper is not None  # _offset_mapper

        self.dtype = np.dtype(self.skim_info['dtype_name'])
        self.base_keys = taz_skim_dict.skim_info['base_keys']
        self.sparse_keys = list(set(network_los.maz_to_maz_df.columns) - {'OMAZ', 'DMAZ'})
        self.sparse_key_usage = set()

    def _offset_mapper(self):
        # called by super().__init__

        # we want to build a series with MAZ zone_id index and TAZ skim array offset values

        # start with a series with MAZ zone_id index and TAZ zone id values
        maz_to_taz = self.network_los.maz_taz_df[['MAZ', 'TAZ']].set_index('MAZ').sort_values(by='TAZ').TAZ

        # use taz offset_mapper to create series mapping directly from MAZ to TAZ skim index
        taz_offset_mapper = super()._offset_mapper()
        maz_to_skim_offset = taz_offset_mapper.map(maz_to_taz)

        offset_mapper = OffsetMapper(offset_series=maz_to_skim_offset)

        return offset_mapper

    def get_skim_usage(self):
        return self.sparse_key_usage.union(self.usage)

    def sparse_lookup(self, orig, dest, key):
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

        self.sparse_key_usage.add(key)

        max_blend_distance = self.network_los.max_blend_distance.get(key, 0)

        if max_blend_distance == 0:
            blend_distance_skim_name = None
        else:
            blend_distance_skim_name = self.network_los.blend_distance_skim_name

        # fixme - remove?
        assert not (np.isnan(orig) | np.isnan(dest)).any()

        # we want values from mazpairs, where we have them
        values = self.network_los.get_mazpairs(orig, dest, key)

        is_nan = np.isnan(values)

        if max_blend_distance > 0:

            # print(f"{is_nan.sum()} nans out of {len(is_nan)} for key '{self.key}")
            # print(f"blend_distance_skim_name {self.blend_distance_skim_name}")

            backstop_values = super().lookup(orig, dest, key)

            # get distance skim if a different key was specified by blend_distance_skim_name
            if (blend_distance_skim_name != key):
                distance = self.network_los.get_mazpairs(orig, dest, blend_distance_skim_name)
            else:
                distance = values

            # for distances less than max_blend_distance, we blend maz-maz and skim backstop values
            # shorter distances have less fractional backstop, and more maz-maz
            # beyond max_blend_distance, just use the skim values
            backstop_fractions = np.minimum(distance / max_blend_distance, 1)

            values = np.where(is_nan,
                              backstop_values,
                              backstop_fractions * backstop_values + (1 - backstop_fractions) * values)

        elif is_nan.any():

            # print(f"{is_nan.sum()} nans out of {len(is_nan)} for key '{self.key}")

            if key in self.base_keys:
                # replace nan values using simple backstop without blending
                backstop_values = super().lookup(orig, dest, key)
                values = np.where(is_nan, backstop_values, values)
            else:
                #bug
                # FIXME - if no backstop skim, then return 0 (which conventionally means "not available")
                values = np.where(is_nan, 0, values)

        # want to return same type as backstop skim
        values = values.astype(self.dtype)

        return values

    def lookup(self, orig, dest, key):

        if key in self.sparse_keys:
            # logger.debug(f"MazSkimDict using SparseSkimDict for key '{key}'")
            values = self.sparse_lookup(orig, dest, key)
        else:
            values = super().lookup(orig, dest, key)

        return values


class DataFrameMatrix(object):
    """
    Utility class to allow a pandas dataframe to be treated like a 2-D array,
    indexed by rowid, colname

    For use in vectorized expressions where the desired values depend on both a row column selector
    e.g. size_terms.get(df.dest_taz, df.purpose)

    ::

      df = pd.DataFrame({'a': [1,2,3,4,5], 'b': [10,20,30,40,50]}, index=[100,101,102,103,104])

      dfm = DataFrameMatrix(df)

      dfm.get(row_ids=[100,100,103], col_ids=['a', 'b', 'a'])

      returns [1, 10,  4]

    """

    def __init__(self, df):
        """

        Parameters
        ----------
        df - pandas dataframe of uniform type
        """

        self.df = df
        self.data = df.values

        self.offset_mapper = OffsetMapper()
        self.offset_mapper.set_offset_list(list(df.index))

        self.cols_to_indexes = {k: v for v, k in enumerate(df.columns)}

    def get(self, row_ids, col_ids):
        """

        Parameters
        ----------
        row_ids - list of row_ids (df index values)
        col_ids - list of column names, one per row_id,
                  specifying column from which the value for that row should be retrieved

        Returns
        -------

        series with one row per row_id, with the value from the column specified in col_ids

        """
        # col_indexes = segments.map(self.cols_to_indexes).astype('int')
        # this should be faster than map
        col_indexes = np.vectorize(self.cols_to_indexes.get)(col_ids)

        row_indexes = self.offset_mapper.map(np.asanyarray(row_ids))

        assert (row_indexes >= 0).all(), f"{row_indexes}"

        result = self.data[row_indexes, col_indexes]

        # FIXME - if ids (or col_ids?) is a series, return series with same index?
        if isinstance(row_ids, pd.Series):
            result = pd.Series(result, index=row_ids.index)

        return result

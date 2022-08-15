# ActivitySim
# See full license in LICENSE.txt.

import logging
from builtins import object, range

import numpy as np
import pandas as pd

from activitysim.core.util import quick_loc_series

logger = logging.getLogger(__name__)

NOT_IN_SKIM_ZONE_ID = -1
NOT_IN_SKIM_NAN = np.nan

ROW_MAJOR_LAYOUT = True


class OffsetMapper(object):
    """
    Utility to map skim zone ids to ordinal offsets (e.g. numpy array indices)

    Can map either by a fixed offset (e.g. -1 to map 1-based to 0-based)
    or by an explicit mapping of zone id to offset (slower but more flexible)

    Internally, there are two representations:

    offset_int:
        int offset which when added to zone_id yields skim array index (e.g. -1 to map 1-based zones to 0-based index)
    offset_series:
        pandas series with zone_id index and skim array offset values. Ordinarily, index is just range(0, omx_size)
        if series has duplicate offset values, this can map multiple zone_ids to a single skim array index
        (e.g. can map maz zone_ids to corresponding taz skim offset)
    """

    def __init__(self, offset_int=None, offset_list=None, offset_series=None):

        self.offset_int = self.offset_series = None

        assert (offset_int is not None) + (offset_list is not None) + (
            offset_series is not None
        ) <= 1

        if offset_int is not None:
            self.set_offset_int(offset_int)
        elif offset_list is not None:
            self.set_offset_list(offset_list)
        elif offset_series is not None:
            self.set_offset_series(offset_series)

    def print_offset(self, message=""):
        assert (self.offset_int is not None) or (self.offset_series is not None)

        if self.offset_int is not None:
            print(f"{message} offset_int: {self.offset_int}")
        elif self.offset_series is not None:
            print(f"{message} offset_series:\n {self.offset_series}")
        else:
            print(f"{message} offset: None")

    def set_offset_series(self, offset_series):
        """
        Parameters
        ----------
        offset_series: pandas.Series
            series with zone_id index and skim array offset values (can map many zone_ids to skim array index)
        """
        assert isinstance(offset_series, pd.Series)
        self.offset_series = offset_series
        self.offset_int = None

    def set_offset_list(self, offset_list):
        """
        Convenience method to set offset_series using an integer list the same size as target skim dimension
        with implicit skim index mapping (e.g. an omx mapping as returned by omx_file.mapentries)

        Parameters
        ----------
        offset_list : list of int
        """
        assert isinstance(offset_list, list) or isinstance(offset_list, np.ndarray)

        if isinstance(offset_list, np.ndarray):
            offset_list = list(offset_list)

        # - for performance, check if this is a simple range that can ber represented by an int offset
        first_offset = offset_list[0]
        if offset_list == list(range(first_offset, len(offset_list) + first_offset)):
            offset_int = -1 * first_offset
            self.set_offset_int(offset_int)
        else:
            offset_series = pd.Series(
                data=list(range(len(offset_list))), index=offset_list
            )
            self.set_offset_series(offset_series)

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
        map zone_ids to skim indexes

        Parameters
        ----------
        zone_ids : list-like (numpy.ndarray, pandas.Int64Index, or pandas.Series)

        Returns
        -------
        offsets : numpy array of int
        """

        if self.offset_series is not None:
            assert self.offset_int is None
            assert isinstance(self.offset_series, pd.Series)

            # FIXME - turns out it is faster to use series.map if zone_ids is a series
            # offsets = quick_loc_series(zone_ids, self.offset_series).fillna(NOT_IN_SKIM_ZONE_ID).astype(int)

            if isinstance(zone_ids, np.ndarray):
                zone_ids = pd.Series(zone_ids)
            offsets = (
                zone_ids.map(self.offset_series, na_action="ignore")
                .fillna(NOT_IN_SKIM_ZONE_ID)
                .astype(int)
            )

        elif self.offset_int:
            assert self.offset_series is None
            # apply integer offset, but map NOT_IN_SKIM_ZONE_ID to self
            offsets = np.where(
                zone_ids == NOT_IN_SKIM_ZONE_ID,
                NOT_IN_SKIM_ZONE_ID,
                zone_ids + self.offset_int,
            )
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
        self.usage = set()  # track keys of skims looked up

        self.offset_mapper = (
            self._offset_mapper()
        )  # (in function so subclass can override)

        self.omx_shape = skim_info.omx_shape
        self.skim_data = skim_data
        self.dtype = np.dtype(
            skim_info.dtype_name
        )  # so we can coerce if we have missing values

        # - skim_dim3 dict maps key1 to dict of key2 absolute offsets into block
        # DRV_COM_WLK_BOARDS: {'MD': 4, 'AM': 3, 'PM': 5}, ...
        self.skim_dim3 = {}

        for skim_key, offset in skim_info.block_offsets.items():
            if isinstance(skim_key, tuple):
                key1, key2 = skim_key
                self.skim_dim3.setdefault(key1, {})[key2] = offset
        logger.info(
            f"SkimDict.build_3d_skim_block_offset_table registered {len(self.skim_dim3)} 3d keys"
        )

    def _offset_mapper(self):
        """
        Return an OffsetMapper to set self.offset_mapper for use with skims
        This allows subclasses (e.g. MazSkimDict) to 'tweak' the parent offset mapper.

        Returns
        -------
        OffsetMapper
        """
        offset_mapper = OffsetMapper()
        if self.skim_info.offset_map is not None:
            offset_mapper.set_offset_list(offset_list=self.skim_info.offset_map)
        else:
            # assume this is a one-based skim map
            offset_mapper.set_offset_int(-1)

        return offset_mapper

    @property
    def zone_ids(self):
        """
        Return list of zone_ids we grok in skim index order

        Returns
        -------
        ndarray of int domain zone_ids
        """

        if self.offset_mapper.offset_series is not None:
            ids = self.offset_mapper.offset_series.index.values
        else:
            ids = np.array(range(self.omx_shape[0])) - self.offset_mapper.offset_int
        return ids

    def get_skim_usage(self):
        """
        return set of keys of skims looked up. e.g. {'DIST', 'SOV'}

        Returns
        -------
        set:
        """
        return self.usage

    def _lookup(self, orig, dest, block_offsets):
        """
        Return list of skim values of skims(s) at orig/dest for the skim(s) at block_offset in skim_data

        Supplying a single int block_offset makes the lookup 2-D
        Supplying a list of block_offsets (same length as orig and dest lists) allows 3D lookup

        Parameters
        ----------
        orig: list of orig zone_ids
        dest: list of dest zone_ids
        block_offsets: int or list of dim3 blockoffsets for the od pairs

        Returns
        -------
        Numpy.ndarray: list of skim values for od pairs
        """

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
        in_skim = (
            (mapped_orig >= 0)
            & (mapped_orig < self.omx_shape[0])
            & (mapped_dest >= 0)
            & (mapped_dest < self.omx_shape[1])
        )

        # if not ((in_skim | (orig == NOT_IN_SKIM_ZONE_ID) | (dest == NOT_IN_SKIM_ZONE_ID)).all()):
        #     print(f"orig\n{orig}")
        #     print(f"dest\n{dest}")
        #     print(f"in_skim\n{in_skim}")

        # check for bad indexes (other than NOT_IN_SKIM_ZONE_ID)
        assert (
            in_skim | (orig == NOT_IN_SKIM_ZONE_ID) | (dest == NOT_IN_SKIM_ZONE_ID)
        ).all(), f"{(~in_skim).sum()} od pairs not in skim"

        if not in_skim.all():
            result = np.where(in_skim, result, NOT_IN_SKIM_NAN).astype(self.dtype)

        return result

    def lookup(self, orig, dest, key):
        """
        Return list of skim values of skims(s) at orig/dest in skim with the specified key (e.g. 'DIST')

        Parameters
        ----------
        orig: list of orig zone_ids
        dest: list of dest zone_ids
        key: str

        Returns
        -------
        Numpy.ndarray: list of skim values for od pairs
        """

        self.usage.add(key)

        block_offset = self.skim_info.block_offsets.get(key)
        assert block_offset is not None, f"SkimDict lookup key '{key}' not in skims"

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
        """
        3D lookup of skim values of skims(s) at orig/dest for stacked skims indexed by dim3 selector

        The idea is that skims may be stacked in groups with a base key and a dim3 key (usually a time of day key)

        On import (from omx) skims stacks are represented by base and dim3 keys seperated by a double_underscore

        e.g. DRV_COM_WLK_BOARDS__AM indicates base skim key DRV_COM_WLK_BOARDS with a time of day (dim3) of 'AM'

        Since all the skimsa re stored in a single contiguous 3D array, we can use the dim3 key as a third index
        and thus rapidly get skim values for a list of (orig, dest, tod) tuples using index arrays ('fancy indexing')

        Parameters
        ----------
        orig: list of orig zone_ids
        dest: list of dest zone_ids
        block_offsets: list with one dim3 key for each orig/dest pair

        Returns
        -------
        Numpy.ndarray: list of skim values
        """

        self.usage.add(key)  # should we keep usage stats by (key, dim3)?

        assert key in self.skim_dim3, f"3d skim key {key} not in skims."

        # map dim3 to block_offsets
        skim_keys_to_indexes = self.skim_dim3[key]

        # skim_indexes = dim3.map(skim_keys_to_indexes).astype('int')
        try:
            block_offsets = np.vectorize(skim_keys_to_indexes.get)(
                dim3
            )  # this should be faster than map
            result = self._lookup(orig, dest, block_offsets)
        except Exception as err:
            logger.error(
                "SkimDict lookup_3d error: %s: %s", type(err).__name__, str(err)
            )
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
    where each object is identified by a key.

    This is just a way to simplify expression files by hiding the and orig, dest arguments
    when the orig and dest vectors are in a dataframe with known column names (specified at init time)
    The dataframe is identified by set_df because it may not be available (e.g. due to chunking)
    at the time the SkimWrapper is instantiated.

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
        """

        Parameters
        ----------
        skim_dict: SkimDict

        orig_key: str
            name of column in dataframe to use as implicit orig for lookups
        dest_key: str
            name of column in dataframe to use as implicit dest for lookups
        """
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
        assert (
            self.orig_key in df
        ), f"orig_key '{self.orig_key}' not in df columns: {list(df.columns)}"
        assert (
            self.dest_key in df
        ), f"dest_key '{self.dest_key}' not in df columns: {list(df.columns)}"
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
            s = self.skim_dict.lookup(
                self.df[self.dest_key], self.df[self.orig_key], key
            )
        else:
            s = self.skim_dict.lookup(
                self.df[self.orig_key], self.df[self.dest_key], key
            )

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
        assert self.df is not None, "Call set_df first"

        s = np.maximum(
            self.skim_dict.lookup(self.df[self.dest_key], self.df[self.orig_key], key),
            self.skim_dict.lookup(self.df[self.orig_key], self.df[self.dest_key], key),
        )

        return pd.Series(s, index=self.df.index)

    def __getitem__(self, key):
        """
        Get the lookup for an available skim object (df and orig/dest and column names implicit)

        Parameters
        ----------
        key : hashable
             The key (identifier) for the skim object

        Returns
        -------
        impedances: pd.Series with the same index as df
            A Series of impedances values from the single Skim with specified key, indexed byt orig/dest pair
        """

        return self.lookup(key)


class Skim3dWrapper(object):
    """

    This works the same as a SkimWrapper above, except the third dim3 is also supplied,
    and a 3D lookup is performed using orig, dest, and dim3.

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
        """

        Parameters
        ----------
        skim_dict: SkimDict

        orig_key: str
            name of column of zone_ids in dataframe to use as implicit orig for lookups
        dest_key: str
            name of column of zone_ids  in dataframe to use as implicit dest for lookups
        dim3_key: str
            name of column of dim3 keys in dataframe to use as implicit third dim3 key for 3D lookups
            e.g. string column with time_of_day keys (such as 'AM', 'MD', 'PM', etc.)
        """
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
            The dataframe which contains the orig, dest, and dim3 values

        Returns
        -------
        self (to facilitiate chaining)
        """
        assert (
            self.orig_key in df
        ), f"orig_key '{self.orig_key}' not in df columns: {list(df.columns)}"
        assert (
            self.dest_key in df
        ), f"dest_key '{self.dest_key}' not in df columns: {list(df.columns)}"
        assert (
            self.dim3_key in df
        ), f"dim3_key '{self.dim3_key}' not in df columns: {list(df.columns)}"
        self.df = df
        return self

    def __getitem__(self, key):
        """
        Get the lookup for an available skim object (df and orig/dest/dim3 and column names implicit)

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        impedances: pd.Series with the same index as df
            A Series of impedances values from the set of skims with specified base key, indexed by orig/dest/dim3
        """
        assert self.df is not None, "Call set_df first"
        orig = self.df[self.orig_key].astype("int")
        dest = self.df[self.dest_key].astype("int")
        dim3 = self.df[self.dim3_key]

        skim_values = self.skim_dict.lookup_3d(orig, dest, dim3, key)

        return pd.Series(skim_values, self.df.index)


class MazSkimDict(SkimDict):
    """
    MazSkimDict provides a facade that allows skim-like lookup by maz orig,dest zone_id
    when there are often too many maz zones to create maz skims.

    Dependencies: network_los.load_data must have already loaded: taz skim_dict, maz_to_maz_df, and maz_taz_df

    It performs lookups from a sparse list of maz-maz od pairs on selected attributes (e.g. WALKDIST)
    where accuracy for nearby od pairs is critical. And is backed by a fallback taz skim dict
    to return values of for more distant pairs (or for skims that are not attributes in the maz-maz table.)
    """

    def __init__(self, skim_tag, network_los, taz_skim_dict):
        """
        we need network_los because we have dependencies on network_los.load_data (e.g. maz_to_maz_df, maz_taz_df,
        and the fallback taz skim_dict)

        We require taz_skim_dict as an explicit parameter to emphasize that we are piggybacking on taz_skim_dict's
        preexisting skim_data and skim_info, rather than instantiating duplicate copies thereof.

        Note, however, that we override _offset_mapper (called by super.__init__) to create our own
        custom self.offset_mapper that maps directly from MAZ zone_ids to TAZ skim array indexes

        Parameters
        ----------
        skim_tag: str
        network_los: Network_LOS
        taz_skim_dict: SkimDict
        """

        self.network_los = network_los

        super().__init__(skim_tag, taz_skim_dict.skim_info, taz_skim_dict.skim_data)
        assert (
            self.offset_mapper is not None
        )  # should have been set with _init_offset_mapper

        self.dtype = np.dtype(self.skim_info.dtype_name)
        self.base_keys = taz_skim_dict.skim_info.base_keys
        self.sparse_keys = list(
            set(network_los.maz_to_maz_df.columns) - {"OMAZ", "DMAZ"}
        )
        self.sparse_key_usage = set()

    def _offset_mapper(self):
        """
        return an OffsetMapper to map maz zone_ids to taz skim indexes
        Specifically, an offset_series with MAZ zone_id index and TAZ skim array offset values

        This is called by super().__init__ AFTER

        Returns
        -------
        OffsetMapper
        """

        # start with a series with MAZ zone_id index and TAZ zone id values
        maz_to_taz = (
            self.network_los.maz_taz_df[["MAZ", "TAZ"]]
            .set_index("MAZ")
            .sort_values(by="TAZ")
            .TAZ
        )

        # use taz offset_mapper to create series mapping directly from MAZ to TAZ skim index
        taz_offset_mapper = super()._offset_mapper()
        maz_to_skim_offset = taz_offset_mapper.map(maz_to_taz)

        if isinstance(maz_to_skim_offset, np.ndarray):
            maz_to_skim_offset = pd.Series(maz_to_skim_offset, maz_to_taz.index)  # bug

        # MAZ
        # 19062    330 <- The TAZ would be, say, 331, and the offset is 330
        # 8429     330
        # 9859     331

        assert isinstance(maz_to_skim_offset, np.ndarray) or isinstance(
            maz_to_skim_offset, pd.Series
        )
        if isinstance(maz_to_skim_offset, pd.Series):
            offset_mapper = OffsetMapper(offset_series=maz_to_skim_offset)
        elif isinstance(maz_to_skim_offset, np.ndarray):
            offset_mapper = OffsetMapper(offset_list=maz_to_skim_offset)

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
        key : str
            skim key

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
            if blend_distance_skim_name != key:
                distance = self.network_los.get_mazpairs(
                    orig, dest, blend_distance_skim_name
                )
            else:
                distance = values

            # for distances less than max_blend_distance, we blend maz-maz and skim backstop values
            # shorter distances have less fractional backstop, and more maz-maz
            # beyond max_blend_distance, just use the skim values
            backstop_fractions = np.minimum(distance / max_blend_distance, 1)

            values = np.where(
                is_nan,
                backstop_values,
                backstop_fractions * backstop_values
                + (1 - backstop_fractions) * values,
            )

        elif is_nan.any():
            # print(f"{is_nan.sum()} nans out of {len(is_nan)} for key '{self.key}")

            if key in self.base_keys:

                # replace nan values using simple backstop without blending
                backstop_values = super().lookup(orig, dest, key)
                values = np.where(is_nan, backstop_values, values)
            else:
                # FIXME - if no backstop skim, then return 0 (which conventionally means "not available")
                logger.warning(
                    "No backstop skims found for {0}, so setting Nulls to 0. Make sure "
                    "mode availability flags are set to > 0"
                )
                values = np.where(is_nan, 0, values)

        # want to return same type as backstop skim
        values = values.astype(self.dtype)

        return values

    def lookup(self, orig, dest, key):
        """
        Return list of skim values of skims(s) at orig/dest in skim with the specified key (e.g. 'DIST')

        Look up in sparse table (backed by taz skims) if key is a sparse_key, otherwise look up in taz skims
        For taz skim lookups, the offset_mapper will convert maz zone_ids directly to taz skim indexes.

        Parameters
        ----------
        orig: list of orig zone_ids
        dest: list of dest zone_ids
        key: str

        Returns
        -------
        Numpy.ndarray: list of skim values for od pairs
        """

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

        not_in_skim = row_indexes == NOT_IN_SKIM_ZONE_ID
        if not_in_skim.any():
            logger.warning(
                f"DataFrameMatrix: {not_in_skim.sum()} row_ids of {len(row_ids)} not in skim."
            )
            not_in_skim = not_in_skim.values
            logger.warning(f"row_ids: {row_ids[not_in_skim]}")
            logger.warning(f"col_ids: {col_ids[not_in_skim]}")
            raise RuntimeError(
                f"DataFrameMatrix: {not_in_skim.sum()} row_ids of {len(row_ids)} not in skim."
            )

        assert (row_indexes >= 0).all(), f"{row_indexes}"

        result = self.data[row_indexes, col_indexes]

        # FIXME - if ids (or col_ids?) is a series, return series with same index?
        if isinstance(row_ids, pd.Series):
            result = pd.Series(result, index=row_ids.index)

        return result

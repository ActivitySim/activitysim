# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import range
from builtins import object

from future.utils import iteritems
from future.utils import listvalues

import logging

from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core.util import quick_loc_series


logger = logging.getLogger(__name__)


class OffsetMapper(object):
    """
    Utility to map skim zone ids to ordinal offsets (e.g. numpy array indices)

    Can map either by a fixed offset (e.g. -1 to map 1-based to 0-based)
    or by an explicit mapping of zone id to offset (slower but more flexible)
    """

    def __init__(self, offset_int=None):
        self.offset_series = None
        self.offset_int = offset_int

    def set_offset_list(self, offset_list):
        """
        Specify the zone ids corresponding to the offsets (ordinal positions)

        set_offset_list([10, 20, 30, 40])
        map([30, 20, 40])
        returns offsets [2, 1, 3]

        Parameters
        ----------
        offset_list : list of int
        """
        assert isinstance(offset_list, list)
        assert self.offset_int is None

        # - for performance, check if this is a simple int-based series
        first_offset = offset_list[0]
        if (offset_list == list(range(first_offset, len(offset_list)+first_offset))):
            offset_int = -1 * first_offset
            # print "set_offset_list substituting offset_int of %s" % offset_int
            self.set_offset_int(offset_int)
            return

        if self.offset_series is None:
            self.offset_series = pd.Series(data=list(range(len(offset_list))), index=offset_list)
        else:
            # make sure it offsets are the same
            assert (offset_list == self.offset_series.index).all()

    def set_offset_int(self, offset_int):
        """
        specify fixed offset (e.g. -1 to map 1-based to 0-based)

        Parameters
        ----------
        offset_int : int
        """

        # should be some kind of integer
        assert int(offset_int) == offset_int
        assert self.offset_series is None

        if self.offset_int is None:
            self.offset_int = int(offset_int)
        else:
            # make sure it is the same
            assert offset_int == self.offset_int

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

        # print "\nmap_offsets zone_ids", zone_ids

        if self.offset_series is not None:
            assert(self.offset_int is None)
            assert isinstance(self.offset_series, pd.Series)

            offsets = np.asanyarray(quick_loc_series(zone_ids, self.offset_series))

        elif self.offset_int:
            assert (self.offset_series is None)
            offsets = zone_ids + self.offset_int
        else:
            offsets = zone_ids

        # print "map_offsets offsets", offsets

        return offsets


class SkimWrapper(object):
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
    def __init__(self, data, offset_mapper=None):

        self.data = data
        self.offset_mapper = offset_mapper if offset_mapper is not None else OffsetMapper()

    def get(self, orig, dest):
        """
        Get impedence values for a set of origin, destination pairs.

        Parameters
        ----------
        orig : 1D array
        dest : 1D array

        Returns
        -------
        values : 1D array

        """

        # fixme - remove?
        assert not (np.isnan(orig) | np.isnan(dest)).any()

        # only working with numpy in here
        orig = np.asanyarray(orig).astype(int)
        dest = np.asanyarray(dest).astype(int)

        orig = self.offset_mapper.map(orig)
        dest = self.offset_mapper.map(dest)

        result = self.data[orig, dest]

        return result


class SkimDict(object):
    """
    A SkimDict object is a wrapper around a dict of multiple skim objects,
    where each object is identified by a key.  It operates like a
    dictionary - i.e. use brackets to add and get skim objects.

    Note that keys are either strings or tuples of two strings (to support stacking of skims.)
    """

    def __init__(self, skim_data, skim_info):

        self.skim_info = skim_info
        self.skim_data = skim_data

        self.offset_mapper = OffsetMapper()
        self.usage = set()

    def touch(self, key):

        self.usage.add(key)

    def get(self, key):
        """
        Get an available wrapped skim object (not the lookup)

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """

        block, offset = self.skim_info['block_offsets'].get(key)
        block_data = self.skim_data[block]

        self.touch(key)

        data = block_data[:, :, offset]

        return SkimWrapper(data, self.offset_mapper)

    def wrap(self, left_key, right_key):
        """
        return a SkimDictWrapper for self
        """
        return SkimDictWrapper(self, left_key, right_key)


class SkimDictWrapper(object):
    """
    A SkimDictWrapper object is an access wrapper around a SkimDict of multiple skim objects,
    where each object is identified by a key.  It operates like a
    dictionary - i.e. use brackets to add and get skim objects - but also
    has information on how to lookup against the skim objects.
    Specifically, this object has a dataframe, a left_key and right_key.
    It is assumed that left_key and right_key identify columns in df.  The
    parameter df is usually set by the simulation itself as it's a result of
    interacting choosers and alternatives.

    When the user calls skims[key], key is an identifier for which skim
    to use, and the object automatically looks up impedances of that skim
    using the specified left_key column in df as the origin and
    the right_key column in df as the destination.  In this way, the user
    does not do the O-D lookup by hand and only specifies which skim to use
    for this lookup.  This is the only purpose of this object: to
    abstract away the O-D lookup and use skims by specifying which skim
    to use in the expressions.

    Note that keys are either strings or tuples of two strings (to support stacking of skims.)
    """

    def __init__(self, skim_dict, left_key, right_key):
        self.skim_dict = skim_dict
        self.left_key = left_key
        self.right_key = right_key
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
        Nothing
        """
        self.df = df

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

        # The skim object to perform the lookup
        # using df[left_key] as the origin and df[right_key] as the destination
        skim = self.skim_dict.get(key)

        # assert self.df is not None, "Call set_df first"
        # origins = self.df[self.left_key].astype('int')
        # destinations = self.df[self.right_key].astype('int')
        # if self.offset:
        #     origins = origins + self.offset
        #     destinations = destinations + self.offset

        assert self.df is not None, "Call set_df first"

        if reverse:
            s = skim.get(self.df[self.right_key], self.df[self.left_key])
        else:
            s = skim.get(self.df[self.left_key], self.df[self.right_key])

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

        skim = self.skim_dict.get(key)

        assert self.df is not None, "Call set_df first"

        s = np.maximum(
            skim.get(self.df[self.right_key], self.df[self.left_key]),
            skim.get(self.df[self.left_key], self.df[self.right_key])
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


class SkimStack(object):

    def __init__(self, skim_dict):

        self.offset_mapper = skim_dict.offset_mapper
        self.skim_dict = skim_dict

        # - key1_blocks dict maps key1 to block number
        # DISTWALK: 0,
        # DRV_COM_WLK_BOARDS: 0, ...
        key1_block_offsets = skim_dict.skim_info['key1_block_offsets']
        self.key1_blocks = {k: v[0] for k, v in iteritems(key1_block_offsets)}

        # - skim_dim3 dict maps key1 to dict of key2 absolute offsets into block
        # DRV_COM_WLK_BOARDS: {'MD': 4, 'AM': 3, 'PM': 5}, ...
        block_offsets = skim_dict.skim_info['block_offsets']
        skim_dim3 = OrderedDict()
        for skim_key in block_offsets:

            if not isinstance(skim_key, tuple):
                continue

            key1, key2 = skim_key
            block, offset = block_offsets[skim_key]

            assert block == self.key1_blocks[key1]

            skim_dim3.setdefault(key1, OrderedDict())[key2] = offset

        self.skim_dim3 = skim_dim3

        logger.info("SkimStack.__init__ loaded %s keys with %s total skims"
                    % (len(self.skim_dim3),
                       sum([len(d) for d in listvalues(self.skim_dim3)])))

        self.usage = set()

    def touch(self, key):
        self.usage.add(key)

    def lookup(self, orig, dest, dim3, key):

        orig = self.offset_mapper.map(orig)
        dest = self.offset_mapper.map(dest)

        assert key in self.key1_blocks, "SkimStack key %s missing" % key
        assert key in self.skim_dim3, "SkimStack key %s missing" % key

        block = self.key1_blocks[key]
        stacked_skim_data = self.skim_dict.skim_data[block]
        skim_keys_to_indexes = self.skim_dim3[key]

        self.touch(key)

        # skim_indexes = dim3.map(skim_keys_to_indexes).astype('int')
        # this should be faster than map
        skim_indexes = np.vectorize(skim_keys_to_indexes.get)(dim3)

        return stacked_skim_data[orig, dest, skim_indexes]

    def wrap(self, left_key, right_key, skim_key):
        """
        return a SkimStackWrapper for self
        """
        return SkimStackWrapper(stack=self,
                                left_key=left_key, right_key=right_key, skim_key=skim_key)


class SkimStackWrapper(object):
    """
    A SkimStackWrapper object wraps a SkimStack object to add an additional wrinkle of
    lookup functionality.  Upon init the separate skims objects are
    processed into a 3D matrix so that lookup of the different skims can
    be performed quickly for each row in the dataframe.  In this very
    particular formulation, the keys are assumed to be tuples with two
    elements - the second element of which will be taken from the
    different rows in the dataframe.  The first element can then be
    dereferenced like an array.  This is useful, for instance, to have a
    certain skim vary by time of day - the skims are set with keys of
    ('SOV', 'AM"), ('SOV', 'PM') etc.  The time of day is then taken to
    be different for every row in the tours table, and the 'SOV' portion
    of the key can be used in __getitem__.

    To be more explicit, the input is a dictionary of Skims objects, each of
    which contains a 2D matrix.  These are stacked into a 3D matrix with a
    mapping of keys to indexes which is applied using pandas .map to a third
    column in the object dataframe.  The three columns - left_key and
    right_key from the Skims object and skim_key from this one, are then used to
    dereference the 3D matrix.  The tricky part comes in defining the key which
    matches the 3rd dimension of the matrix, and the key which is passed into
    __getitem__ below (i.e. the one used in the specs).  By convention,
    every key in the Skims object that is passed in MUST be a tuple with 2
    items.  The second item in the tuple maps to the items in the dataframe
    referred to by the skim_key column and the first item in the tuple is
    then available to pass directly to __getitem__.

    The sum conclusion of this is that in the specs, you can say something
    like out_skim['SOV'] and it will automatically dereference the 3D matrix
    using origin, destination, and time of day.

    Parameters
    ----------
    skims: Skims
        This is the Skims object to wrap
    skim_key : str
        This identifies the column in the dataframe which is used to
        select among Skim object using the SECOND item in each tuple (see
        above for a more complete description)
    """

    def __init__(self, stack, left_key, right_key, skim_key):

        self.stack = stack

        self.left_key = left_key
        self.right_key = right_key
        self.skim_key = skim_key
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
        Nothing
        """
        self.df = df

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
        orig = self.df[self.left_key].astype('int')
        dest = self.df[self.right_key].astype('int')
        dim3 = self.df[self.skim_key]

        skim_values = self.stack.lookup(orig, dest, dim3, key)

        return pd.Series(skim_values, self.df.index)


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

        result = self.data[row_indexes, col_indexes]

        # FIXME - if ids (or col_ids?) is a series, return series with same index?
        if isinstance(row_ids, pd.Series):
            result = pd.Series(result, index=row_ids.index)

        return result

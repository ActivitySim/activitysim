# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class OffsetMapper(object):

    def __init__(self, offset_int=None):
        self.offset_series = None
        self.offset_int = offset_int

    def set_offset_list(self, offset_list):

        assert isinstance(offset_list, list)
        assert self.offset_int is None

        # for performance, check if this is a simple int-based series
        first_offset = offset_list[0]
        if (offset_list == range(first_offset, len(offset_list)+first_offset)):
            offset_int = -1 * first_offset
            # print "set_offset_list substituting offset_int of %s" % offset_int
            self.set_offset_int(offset_int)
            return

        if self.offset_series is None:
            self.offset_series = pd.Series(data=range(len(offset_list)), index=offset_list)
        else:
            # make sure it offsets are the same
            assert (offset_list == self.offset_series.index).all()

    def set_offset_int(self, offset_int):

        # should be some kind of integer
        assert long(offset_int) == offset_int
        assert self.offset_series is None

        if self.offset_int is None:
            self.offset_int = offset_int
        else:
            # make sure it is the same
            assert offset_int == self.offset_int

    def map(self, zone_ids):

        # print "\nmap_offsets zone_ids", zone_ids

        if self.offset_series is not None:
            assert(self.offset_int is None)
            assert isinstance(self.offset_series, pd.Series)
            offsets = np.asanyarray(self.offset_series.loc[zone_ids])
        elif self.offset_int:
            # should be some kind of integer
            assert long(self.offset_int) == self.offset_int
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
        # only working with numpy in here
        orig = np.asanyarray(orig)
        dest = np.asanyarray(dest)
        out_shape = orig.shape

        # filter orig and dest to only the real-number pairs
        notnan = ~(np.isnan(orig) | np.isnan(dest))
        orig = orig[notnan].astype('int')
        dest = dest[notnan].astype('int')

        orig = self.offset_mapper.map(orig)
        dest = self.offset_mapper.map(dest)

        result = self.data[orig, dest]

        # add the nans back to the result
        out = np.empty(out_shape)
        out[notnan] = result
        out[~notnan] = np.nan

        return out


class SkimDict(object):
    """
    A SkimDict object is a wrapper around a dict of multiple skim objects,
    where each object is identified by a key.  It operates like a
    dictionary - i.e. use brackets to add and get skim objects.

    Note that keys are either strings or tuples of two strings (to support stacking of skims.)
    """

    def __init__(self):
        self.skims = {}
        self.offset_mapper = OffsetMapper()

    def set(self, key, skim_data):
        """
        Set skim data for key

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object
        skim_data : Skim
             The skim object

        Returns
        -------
        Nothing
        """

        if not isinstance(key, str):
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(key[0], str) and isinstance(key[1], str)

        self.skims[key] = np.asanyarray(skim_data)

        # print "\n### %s" % (key,)
        # print "type(skim_data)", type(skim_data)
        # print "skim_data.shape", skim_data.shape

    def get(self, key):
        """
        Get an available skim object (not the lookup)

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """
        return SkimWrapper(self.skims[key], self.offset_mapper)

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

    def lookup(self, key):
        """
        Generally not called by the user - use __getitem__ instead

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

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
        s = skim.get(self.df[self.left_key],
                     self.df[self.right_key])
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

        self.skims_data = {}
        self.skim_keys_to_indexes = {}
        self.offset_mapper = skim_dict.offset_mapper

        # pass to make dictionary of dictionaries where highest level is unique
        # first items of the tuples and the 2nd level is the second items of
        # the tuples
        for key, skim_data in skim_dict.skims.iteritems():
            if not isinstance(key, tuple) or not len(key) == 2:
                logger.debug("SkimStack __init__ skipping key: %s" % key)
                continue
            logger.debug("SkimStack __init__ loading key: %s" % (key,))
            skim_key1, skim_key2 = key
            # logger.debug("SkimStack init key: key1='%s' key2='%s'" % (skim_key1, skim_key2))
            # FIXME - this copys object reference
            self.skims_data.setdefault(skim_key1, {})[skim_key2] = skim_data

            # print "\n### %s" % (key,)
            # print "type(skim_data)", type(skim_data)
            # print "skim_data.shape", skim_data.shape

        # second pass to turn the each highest level value into a 3D array
        # with a dictionary to make second level keys to indexes
        for skim_key1, value in self.skims_data.iteritems():
            # FIXME - this actually copies/creates new stacked data
            self.skims_data[skim_key1] = np.dstack(value.values())
            self.skim_keys_to_indexes[skim_key1] = dict(zip(value.keys(), range(len(value))))

        logger.info("SkimStack.__init__ loaded %s keys with %s total skims"
                    % (len(self.skim_keys_to_indexes),
                       sum([len(d) for d in self.skim_keys_to_indexes.values()])))

    def __str__(self):

        return "\n".join(
            "%s %s" % (key1, sub_dict)
            for key1, sub_dict in self.skim_keys_to_indexes.iteritems())

    # def key_count(self):
    #     return len(self.skim_keys_to_indexes.keys())
    #
    # def contains(self, key):
    #     return key in self.skims_data

    def get(self, key):
        return self.skims_data[key], self.skim_keys_to_indexes[key]

    def lookup(self, orig, dest, dim3, key):

        orig = self.offset_mapper.map(orig)
        dest = self.offset_mapper.map(dest)

        assert key in self.skims_data, "SkimStack key %s missing" % key

        stacked_skim_data = self.skims_data[key]
        skim_keys_to_indexes = self.skim_keys_to_indexes[key]

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
    A SkimStackWrapper object wraps a skims object to add an additional wrinkle of
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

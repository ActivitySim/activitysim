# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd


class Skim(object):
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
    def __init__(self, data, offset=None):

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data = data
        self.offset = offset

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
        orig = np.asanyarray(orig)
        dest = np.asanyarray(dest)

        if self.offset:
            orig = orig + self.offset
            dest = dest + self.offset

        return self.data[orig, dest]


class Skims(object):

    def __init__(self):
        """
        A skims object is a wrapper around multiple skim objects,
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
        abstract away the O-D lookup and use skims by specifiying which skim
        to use in the expressions.

        Note that keys are any hashable object, not just strings.  So calling
        skim[('AM', 'SOV')] is valid and useful.
        """
        self.skims = {}
        self.left_key = "TAZ"
        self.right_key = "TAZ_r"
        self.df = None

    def set_keys(self, left_key, right_key):
        """
        Set the left and right keys.

        Parameters
        ----------
        left_key : String
            The left key (origin) column in the dataframe
        right_key : String
            The right key (destination) column in the dataframe

        Returns
        --------
        Nothing
        """
        self.left_key = left_key
        self.right_key = right_key

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

    def lookup(self, skim):
        """
        Generally not called by the user - use __getitem__ instead

        Parameters
        ----------
        skim: Skim
            The skim object to perform the lookup using df[left_key] as the
            origin and df[right_key] as the destination

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """
        assert self.df is not None, "Call set_df first"
        s = skim.get(self.df[self.left_key],
                     self.df[self.right_key])
        return pd.Series(s, index=self.df.index)

    def __setitem__(self, key, value):
        """
        Set an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object
        value : Skim
             The skim object

        Returns
        -------
        Nothing
        """
        self.skims[key] = value

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
        return self.lookup(self.skims[key])
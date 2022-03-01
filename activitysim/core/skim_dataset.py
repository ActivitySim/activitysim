import pandas as pd
import xarray as xr
import numpy as np
import logging
from sharrow import array_decode

from . import flow as __flow

logger = logging.getLogger(__name__)

POSITIONS_AS_DICT = True

def _iat(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        loaders[k] = xr.DataArray(v, dims=[_index_name])
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds.load()
    return ds.isel(**loaders)


def _at(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        loaders[k] = xr.DataArray(v, dims=[_index_name])
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds.load()
    return ds.sel(**loaders)


def gather(source, indexes):
    """
    Extract values by label on the coordinates indicated by columns of a DataFrame.

    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
        The source of the values to extract.
    indexes : Mapping[str, array-like]
        The keys of `indexes` (if given as a dataframe, the column names)
        should match the named dimensions of `source`.  The resulting extracted
        data will have a shape one row per row of `df`, and columns matching
        the data variables in `source`, and each value is looked up by the labels.

    Returns
    -------
    pd.DataFrame
    """
    result = _at(source, **indexes).reset_coords(drop=True)
    return result


def igather(source, positions):
    """
    Extract values by position on the coordinates indicated by columns of a DataFrame.

    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
    positions : pd.DataFrame or Mapping[str, array-like]
        The columns (or keys) of `df` should match the named dimensions of
        this Dataset.  The resulting extracted DataFrame will have one row
        per row of `df`, columns matching the data variables in this dataset,
        and each value is looked up by the positions.

    Returns
    -------
    pd.DataFrame
    """
    result = _iat(source, **positions).reset_coords(drop=True)
    return result


class SkimDataset:

    def __init__(self, dataset):
        self.dataset = dataset
        self.time_map = {j: i for i, j in enumerate(self.dataset.indexes['time_period'])}
        self.usage = set()  # track keys of skims looked up

    def get_skim_usage(self):
        """
        return set of keys of skims looked up. e.g. {'DIST', 'SOV'}

        Returns
        -------
        set:
        """
        return self.usage

    def wrap(self, orig_key, dest_key):
        """
        return a SkimWrapper for self
        """
        return DatasetWrapper(self.dataset, orig_key, dest_key, time_map=self.time_map)

    def wrap_3d(self, orig_key, dest_key, dim3_key):
        """
        return a SkimWrapper for self
        """
        return DatasetWrapper(self.dataset, orig_key, dest_key, dim3_key, time_map=self.time_map)

    def lookup(self, orig, dest, key):
        self.usage.add(key)
        use_index = None

        if use_index is None and hasattr(orig, 'index'):
            use_index = orig.index
        if use_index is None and hasattr(dest, 'index'):
            use_index = dest.index

        orig = np.asanyarray(orig).astype(int)
        dest = np.asanyarray(dest).astype(int)

        # TODO offset mapper if required
        positions = {'otaz':orig, 'dtaz':dest}

        # When asking for a particular time period
        if isinstance(key, tuple) and len(key) == 2:
            main_key, time_key = key
            if time_key in self.time_map:
                positions['time_period'] = np.full_like(orig, self.time_map[time_key])
                key = main_key
            else:
                raise KeyError(key)

        result = igather(self.dataset[key], positions)

        if 'digital_encoding' in self.dataset[key].attrs:
            result = array_decode(result, self.dataset[key].attrs['digital_encoding'])

        result = result.to_series()

        if use_index is not None:
            result.index = use_index
        return result

    def map_time_periods_from_series(self, time_period_labels):
        logger.info(f"vectorize lookup for time_period={time_period_labels.name}")
        time_period_idxs = pd.Series(
            np.vectorize(self.time_map.get)(time_period_labels),
            index=time_period_labels.index,
        )
        return time_period_idxs


class DatasetWrapper:

    def __init__(self, dataset, orig_key, dest_key, time_key=None, *, time_map=None):
        """

        Parameters
        ----------
        skim_dict: SkimDict

        orig_key: str
            name of column in dataframe to use as implicit orig for lookups
        dest_key: str
            name of column in dataframe to use as implicit dest for lookups
        """
        self.dataset = dataset
        self.orig_key = orig_key
        self.dest_key = dest_key
        self.time_key = time_key
        self.df = None
        if time_map is None:
            self.time_map = {j: i for i, j in enumerate(self.dataset.indexes['time_period'])}
        else:
            self.time_map = time_map

    def map_time_periods(self, df):
        if self.time_key:
            logger.info(f"vectorize lookup for time_period={self.time_key}")
            time_period_idxs = pd.Series(
                np.vectorize(self.time_map.get)(df[self.time_key]),
                index=df.index,
            )
            return time_period_idxs

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitate chaining)
        """
        assert self.orig_key in df, f"orig_key '{self.orig_key}' not in df columns: {list(df.columns)}"
        assert self.dest_key in df, f"dest_key '{self.dest_key}' not in df columns: {list(df.columns)}"
        if self.time_key:
            assert self.time_key in df, f"time_key '{self.time_key}' not in df columns: {list(df.columns)}"
        self.df = df

        # TODO allow offsets if needed
        positions = {
            'otaz': df[self.orig_key],
            'dtaz': df[self.dest_key],
        }
        if self.time_key:
            if np.issubdtype(df[self.time_key].dtype, np.integer) and df[self.time_key].max() < self.dataset.dims['time_period']:
                logger.info(f"natural use for time_period={self.time_key}")
                positions['time_period'] = df[self.time_key]
            else:
                logger.info(f"vectorize lookup for time_period={self.time_key}")
                positions['time_period'] = pd.Series(
                    np.vectorize(self.time_map.get)(df[self.time_key]),
                    index=df.index,
                )

        if POSITIONS_AS_DICT:
            self.positions = {}
            for k, v in positions.items():
                self.positions[k] = v.astype(int)
        else:
            self.positions = pd.DataFrame(positions).astype(int)

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
            if isinstance(self.positions, dict):
                x = self.positions.copy()
                x.update({
                    'otaz': self.positions['dtaz'],
                    'dtaz': self.positions['otaz'],
                })
            else:
                x = self.positions.rename(columns={'otaz': 'dtaz', 'dtaz': 'otaz'})
        else:
            if isinstance(self.positions, dict):
                x = self.positions.copy()
            else:
                x = self.positions

        # When asking for a particular time period
        if isinstance(key, tuple) and len(key) == 2:
            main_key, time_key = key
            if time_key in self.time_map:
                if isinstance(x, dict):
                    x['time_period'] = np.full_like(x['otaz'], fill_value=self.time_map[time_key])
                else:
                    x = x.assign(time_period=self.time_map[time_key])
                key = main_key
            else:
                raise KeyError(key)

        result = igather(self.dataset[key], x)
        if 'digital_encoding' in self.dataset[key].attrs:
            result = array_decode(result, self.dataset[key].attrs['digital_encoding'])

        # Return a series, consistent with ActivitySim SkimWrapper
        return result.to_series()

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
            self.lookup(key),
            self.lookup(key, True),
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



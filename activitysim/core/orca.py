# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

try:
    from inspect import getfullargspec as getargspec
except ImportError:
    from inspect import getargspec
import logging
import warnings
from collections import Callable, namedtuple
from contextlib import contextmanager
from functools import wraps
import inspect

import pandas as pd
import tables
import tlz as tz

from collections import namedtuple

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('orca')

_TABLES = {}
_COLUMNS = {}
_STEPS = {}
_BROADCASTS = {}
_INJECTABLES = {}

_CACHING = True
_TABLE_CACHE = {}
_COLUMN_CACHE = {}
_INJECTABLE_CACHE = {}
_MEMOIZED = {}

_CS_FOREVER = 'forever'
_CS_ITER = 'iteration'
_CS_STEP = 'step'

CacheItem = namedtuple('CacheItem', ['name', 'value', 'scope'])


@contextmanager
def log_start_finish(msg, logger, level=logging.DEBUG):
    """
    A context manager to log messages with "start: " and "finish: "
    prefixes before and after a block.

    Parameters
    ----------
    msg : str
        Will be prefixed with "start: " and "finish: ".
    logger : logging.Logger
    level : int, optional
        Level at which to log, passed to ``logger.log``.

    """
    # logger.log(level, 'start: ' + msg)
    yield
    # logger.log(level, 'finish: ' + msg)


def _func_source_data(func):
    """
    Return data about a function source, including file name,
    line number, and source code.

    Parameters
    ----------
    func : object
        May be anything support by the inspect module, such as a function,
        method, or class.

    Returns
    -------
    filename : str
    lineno : int
        The line number on which the function starts.
    source : str

    """
    filename = inspect.getsourcefile(func)
    lineno = inspect.getsourcelines(func)[1]
    source = inspect.getsource(func)

    return filename, lineno, source


def clear_all():
    """
    Clear any and all stored state from Orca.

    """
    _TABLES.clear()
    _COLUMNS.clear()
    _STEPS.clear()
    _BROADCASTS.clear()
    _INJECTABLES.clear()
    _TABLE_CACHE.clear()
    _COLUMN_CACHE.clear()
    _INJECTABLE_CACHE.clear()
    for m in _MEMOIZED.values():
        m.value.clear_cached()
    _MEMOIZED.clear()
    logger.debug('pipeline state cleared')


def clear_cache(scope=None):
    """
    Clear all cached data.

    Parameters
    ----------
    scope : {None, 'step', 'iteration', 'forever'}, optional
        Clear cached values with a given scope.
        By default all cached values are removed.

    """
    if not scope:
        _TABLE_CACHE.clear()
        _COLUMN_CACHE.clear()
        _INJECTABLE_CACHE.clear()
        for m in _MEMOIZED.values():
            m.value.clear_cached()
        logger.debug('pipeline cache cleared')
    else:
        for d in (_TABLE_CACHE, _COLUMN_CACHE, _INJECTABLE_CACHE):
            items = tz.valfilter(lambda x: x.scope == scope, d)
            for k in items:
                del d[k]
        for m in tz.filter(lambda x: x.scope == scope, _MEMOIZED.values()):
            m.value.clear_cached()
        logger.debug('cleared cached values with scope {!r}'.format(scope))


def enable_cache():
    """
    Allow caching of registered variables that explicitly have
    caching enabled.

    """
    global _CACHING
    _CACHING = True


def disable_cache():
    """
    Turn off caching across Orca, even for registered variables
    that have caching enabled.

    """
    global _CACHING
    _CACHING = False


def cache_on():
    """
    Whether caching is currently enabled or disabled.

    Returns
    -------
    on : bool
        True if caching is enabled.

    """
    return _CACHING


@contextmanager
def cache_disabled():
    turn_back_on = True if cache_on() else False
    disable_cache()

    yield

    if turn_back_on:
        enable_cache()


# for errors that occur during Orca runs
class OrcaError(Exception):
    pass


class DataFrameWrapper(object):
    """
    Wraps a DataFrame so it can provide certain columns and handle
    computed columns.

    Parameters
    ----------
    name : str
        Name for the table.
    frame : pandas.DataFrame
    copy_col : bool, optional
        Whether to return copies when evaluating columns.

    Attributes
    ----------
    name : str
        Table name.
    copy_col : bool
        Whether to return copies when evaluating columns.
    local : pandas.DataFrame
        The wrapped DataFrame.

    """
    def __init__(self, name, frame, copy_col=True):
        self.name = name
        self.local = frame
        self.copy_col = copy_col

    @property
    def columns(self):
        """
        Columns in this table.

        """
        return self.local_columns + list_columns_for_table(self.name)

    @property
    def local_columns(self):
        """
        Columns that are part of the wrapped DataFrame.

        """
        return list(self.local.columns)

    @property
    def index(self):
        """
        Table index.

        """
        return self.local.index

    def to_frame(self, columns=None):
        """
        Make a DataFrame with the given columns.

        Will always return a copy of the underlying table.

        Parameters
        ----------
        columns : sequence or string, optional
            Sequence of the column names desired in the DataFrame. A string
            can also be passed if only one column is desired.
            If None all columns are returned, including registered columns.

        Returns
        -------
        frame : pandas.DataFrame

        """
        extra_cols = _columns_for_table(self.name)

        if columns is not None:
            columns = [columns] if isinstance(columns, str) else columns
            columns = set(columns)
            set_extra_cols = set(extra_cols)
            local_cols = set(self.local.columns) & columns - set_extra_cols
            df = self.local[list(local_cols)].copy()
            extra_cols = {k: extra_cols[k] for k in (columns & set_extra_cols)}
        else:
            df = self.local.copy()

        with log_start_finish(
                'computing {!r} columns for table {!r}'.format(
                    len(extra_cols), self.name),
                logger):
            for name, col in extra_cols.items():
                with log_start_finish(
                        'computing column {!r} for table {!r}'.format(
                            name, self.name),
                        logger):
                    df[name] = col()

        return df

    def update_col(self, column_name, series):
        """
        Add or replace a column in the underlying DataFrame.

        Parameters
        ----------
        column_name : str
            Column to add or replace.
        series : pandas.Series or sequence
            Column data.

        """
        logger.debug('updating column {!r} in table {!r}'.format(
            column_name, self.name))
        self.local[column_name] = series

    def __setitem__(self, key, value):
        return self.update_col(key, value)

    def get_column(self, column_name):
        """
        Returns a column as a Series.

        Parameters
        ----------
        column_name : str

        Returns
        -------
        column : pandas.Series

        """
        with log_start_finish(
                'getting single column {!r} from table {!r}'.format(
                    column_name, self.name),
                logger):
            extra_cols = _columns_for_table(self.name)
            if column_name in extra_cols:
                with log_start_finish(
                        'computing column {!r} for table {!r}'.format(
                            column_name, self.name),
                        logger):
                    column = extra_cols[column_name]()
            else:
                column = self.local[column_name]
            if self.copy_col:
                return column.copy()
            else:
                return column

    def __getitem__(self, key):
        return self.get_column(key)

    def __getattr__(self, key):
        return self.get_column(key)

    def column_type(self, column_name):
        """
        Report column type as one of 'local', 'series', or 'function'.

        Parameters
        ----------
        column_name : str

        Returns
        -------
        col_type : {'local', 'series', 'function'}
            'local' means that the column is part of the registered table,
            'series' means the column is a registered Pandas Series,
            and 'function' means the column is a registered function providing
            a Pandas Series.

        """
        extra_cols = list_columns_for_table(self.name)

        if column_name in extra_cols:
            col = _COLUMNS[(self.name, column_name)]

            if isinstance(col, _SeriesWrapper):
                return 'series'
            elif isinstance(col, _ColumnFuncWrapper):
                return 'function'

        elif column_name in self.local_columns:
            return 'local'

        raise KeyError('column {!r} not found'.format(column_name))

    def update_col_from_series(self, column_name, series, cast=False):
        """
        Update existing values in a column from another series.
        Index values must match in both column and series. Optionally
        casts data type to match the existing column.

        Parameters
        ---------------
        column_name : str
        series : panas.Series
        cast: bool, optional, default False
        """
        logger.debug('updating column {!r} in table {!r}'.format(
            column_name, self.name))

        col_dtype = self.local[column_name].dtype
        if series.dtype != col_dtype:
            if cast:
                series = series.astype(col_dtype)
            else:
                err_msg = "Data type mismatch, existing:{}, update:{}"
                err_msg = err_msg.format(col_dtype, series.dtype)
                raise ValueError(err_msg)

        self.local.loc[series.index, column_name] = series

    def __len__(self):
        return len(self.local)

    def clear_cached(self):
        """
        Remove cached results from this table's computed columns.

        """
        _TABLE_CACHE.pop(self.name, None)
        for col in _columns_for_table(self.name).values():
            col.clear_cached()
        logger.debug('cleared cached columns for table {!r}'.format(self.name))


class TableFuncWrapper(object):
    """
    Wrap a function that provides a DataFrame.

    Parameters
    ----------
    name : str
        Name for the table.
    func : callable
        Callable that returns a DataFrame.
    cache : bool, optional
        Whether to cache the results of calling the wrapped function.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.
    copy_col : bool, optional
        Whether to return copies when evaluating columns.

    Attributes
    ----------
    name : str
        Table name.
    cache : bool
        Whether caching is enabled for this table.
    copy_col : bool
        Whether to return copies when evaluating columns.

    """
    def __init__(
            self, name, func, cache=False, cache_scope=_CS_FOREVER,
            copy_col=True):
        self.name = name
        self._func = func
        self._argspec = getargspec(func)
        self.cache = cache
        self.cache_scope = cache_scope
        self.copy_col = copy_col
        self._columns = []
        self._index = None
        self._len = 0

    @property
    def columns(self):
        """
        Columns in this table. (May contain only computed columns
        if the wrapped function has not been called yet.)

        """
        return self._columns + list_columns_for_table(self.name)

    @property
    def local_columns(self):
        """
        Only the columns contained in the DataFrame returned by the
        wrapped function. (No registered columns included.)

        """
        if self._columns:
            return self._columns
        else:
            self._call_func()
            return self._columns

    @property
    def index(self):
        """
        Index of the underlying table. Will be None if that index is
        unknown.

        """
        return self._index

    def _call_func(self):
        """
        Call the wrapped function and return the result wrapped by
        DataFrameWrapper.
        Also updates attributes like columns, index, and length.

        """
        if _CACHING and self.cache and self.name in _TABLE_CACHE:
            logger.debug('returning table {!r} from cache'.format(self.name))
            return _TABLE_CACHE[self.name].value

        with log_start_finish(
                'call function to get frame for table {!r}'.format(
                    self.name),
                logger):
            kwargs = _collect_variables(names=self._argspec.args,
                                        expressions=self._argspec.defaults)
            frame = self._func(**kwargs)

        self._columns = list(frame.columns)
        self._index = frame.index
        self._len = len(frame)

        wrapped = DataFrameWrapper(self.name, frame, copy_col=self.copy_col)

        if self.cache:
            _TABLE_CACHE[self.name] = CacheItem(
                self.name, wrapped, self.cache_scope)

        return wrapped

    def __call__(self):
        return self._call_func()

    def to_frame(self, columns=None):
        """
        Make a DataFrame with the given columns.

        Will always return a copy of the underlying table.

        Parameters
        ----------
        columns : sequence, optional
            Sequence of the column names desired in the DataFrame.
            If None all columns are returned.

        Returns
        -------
        frame : pandas.DataFrame

        """
        return self._call_func().to_frame(columns)

    def get_column(self, column_name):
        """
        Returns a column as a Series.

        Parameters
        ----------
        column_name : str

        Returns
        -------
        column : pandas.Series

        """
        frame = self._call_func()
        return DataFrameWrapper(self.name, frame,
                                copy_col=self.copy_col).get_column(column_name)

    def __getitem__(self, key):
        return self.get_column(key)

    def __getattr__(self, key):
        return self.get_column(key)

    def __len__(self):
        return self._len

    def column_type(self, column_name):
        """
        Report column type as one of 'local', 'series', or 'function'.

        Parameters
        ----------
        column_name : str

        Returns
        -------
        col_type : {'local', 'series', 'function'}
            'local' means that the column is part of the registered table,
            'series' means the column is a registered Pandas Series,
            and 'function' means the column is a registered function providing
            a Pandas Series.

        """
        extra_cols = list_columns_for_table(self.name)

        if column_name in extra_cols:
            col = _COLUMNS[(self.name, column_name)]

            if isinstance(col, _SeriesWrapper):
                return 'series'
            elif isinstance(col, _ColumnFuncWrapper):
                return 'function'

        elif column_name in self.local_columns:
            return 'local'

        raise KeyError('column {!r} not found'.format(column_name))

    def clear_cached(self):
        """
        Remove this table's cached result and that of associated columns.

        """
        _TABLE_CACHE.pop(self.name, None)
        for col in _columns_for_table(self.name).values():
            col.clear_cached()
        logger.debug(
            'cleared cached result and cached columns for table {!r}'.format(
                self.name))

    def func_source_data(self):
        """
        Return data about the wrapped function source, including file name,
        line number, and source code.

        Returns
        -------
        filename : str
        lineno : int
            The line number on which the function starts.
        source : str

        """
        return _func_source_data(self._func)


class _ColumnFuncWrapper(object):
    """
    Wrap a function that returns a Series.

    Parameters
    ----------
    table_name : str
        Table with which the column will be associated.
    column_name : str
        Name for the column.
    func : callable
        Should return a Series that has an
        index matching the table to which it is being added.
    cache : bool, optional
        Whether to cache the result of calling the wrapped function.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.

    Attributes
    ----------
    name : str
        Column name.
    table_name : str
        Name of table this column is associated with.
    cache : bool
        Whether caching is enabled for this column.

    """
    def __init__(
            self, table_name, column_name, func, cache=False,
            cache_scope=_CS_FOREVER):
        self.table_name = table_name
        self.name = column_name
        self._func = func
        self._argspec = getargspec(func)
        self.cache = cache
        self.cache_scope = cache_scope

    def __call__(self):
        """
        Evaluate the wrapped function and return the result.

        """
        if (_CACHING and
                self.cache and
                (self.table_name, self.name) in _COLUMN_CACHE):
            logger.debug(
                'returning column {!r} for table {!r} from cache'.format(
                    self.name, self.table_name))
            return _COLUMN_CACHE[(self.table_name, self.name)].value

        with log_start_finish(
                ('call function to provide column {!r} for table {!r}'
                 ).format(self.name, self.table_name), logger):
            kwargs = _collect_variables(names=self._argspec.args,
                                        expressions=self._argspec.defaults)
            col = self._func(**kwargs)

        if self.cache:
            _COLUMN_CACHE[(self.table_name, self.name)] = CacheItem(
                (self.table_name, self.name), col, self.cache_scope)

        return col

    def clear_cached(self):
        """
        Remove any cached result of this column.

        """
        x = _COLUMN_CACHE.pop((self.table_name, self.name), None)
        if x is not None:
            logger.debug(
                'cleared cached value for column {!r} in table {!r}'.format(
                    self.name, self.table_name))

    def func_source_data(self):
        """
        Return data about the wrapped function source, including file name,
        line number, and source code.

        Returns
        -------
        filename : str
        lineno : int
            The line number on which the function starts.
        source : str

        """
        return _func_source_data(self._func)


class _SeriesWrapper(object):
    """
    Wrap a Series for the purpose of giving it the same interface as a
    `_ColumnFuncWrapper`.

    Parameters
    ----------
    table_name : str
        Table with which the column will be associated.
    column_name : str
        Name for the column.
    series : pandas.Series
        Series with index matching the table to which it is being added.

    Attributes
    ----------
    name : str
        Column name.
    table_name : str
        Name of table this column is associated with.

    """
    def __init__(self, table_name, column_name, series):
        self.table_name = table_name
        self.name = column_name
        self._column = series

    def __call__(self):
        return self._column

    def clear_cached(self):
        """
        Here for compatibility with `_ColumnFuncWrapper`.

        """
        pass


class _InjectableFuncWrapper(object):
    """
    Wraps a function that will provide an injectable value elsewhere.

    Parameters
    ----------
    name : str
    func : callable
    cache : bool, optional
        Whether to cache the result of calling the wrapped function.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.

    Attributes
    ----------
    name : str
        Name of this injectable.
    cache : bool
        Whether caching is enabled for this injectable function.

    """
    def __init__(self, name, func, cache=False, cache_scope=_CS_FOREVER):
        self.name = name
        self._func = func
        self._argspec = getargspec(func)
        self.cache = cache
        self.cache_scope = cache_scope

    def __call__(self):
        if _CACHING and self.cache and self.name in _INJECTABLE_CACHE:
            logger.debug(
                'returning injectable {!r} from cache'.format(self.name))
            return _INJECTABLE_CACHE[self.name].value

        with log_start_finish(
                'call function to provide injectable {!r}'.format(self.name),
                logger):
            kwargs = _collect_variables(names=self._argspec.args,
                                        expressions=self._argspec.defaults)
            result = self._func(**kwargs)

        if self.cache:
            _INJECTABLE_CACHE[self.name] = CacheItem(
                self.name, result, self.cache_scope)

        return result

    def clear_cached(self):
        """
        Clear a cached result for this injectable.

        """
        x = _INJECTABLE_CACHE.pop(self.name, None)
        if x:
            logger.debug(
                'injectable {!r} removed from cache'.format(self.name))


class _StepFuncWrapper(object):
    """
    Wrap a step function for argument matching.

    Parameters
    ----------
    step_name : str
    func : callable

    Attributes
    ----------
    name : str
        Name of step.

    """
    def __init__(self, step_name, func):
        self.name = step_name
        self._func = func
        self._argspec = getargspec(func)

    def __call__(self):
        with log_start_finish('calling step {!r}'.format(self.name), logger):
            kwargs = _collect_variables(names=self._argspec.args,
                                        expressions=self._argspec.defaults)
            return self._func(**kwargs)

    def _tables_used(self):
        """
        Tables injected into the step.

        Returns
        -------
        tables : set of str

        """
        args = list(self._argspec.args)
        if self._argspec.defaults:
            default_args = list(self._argspec.defaults)
        else:
            default_args = []
        # Combine names from argument names and argument default values.
        names = args[:len(args) - len(default_args)] + default_args
        tables = set()
        for name in names:
            parent_name = name.split('.')[0]
            if is_table(parent_name):
                tables.add(parent_name)
        return tables

    def func_source_data(self):
        """
        Return data about a step function's source, including file name,
        line number, and source code.

        Returns
        -------
        filename : str
        lineno : int
            The line number on which the function starts.
        source : str

        """
        return _func_source_data(self._func)


def is_table(name):
    """
    Returns whether a given name refers to a registered table.

    """
    return name in _TABLES


def list_tables():
    """
    List of table names.

    """
    return list(_TABLES.keys())


def list_columns():
    """
    List of (table name, registered column name) pairs.

    """
    return list(_COLUMNS.keys())


def list_steps():
    """
    List of registered step names.

    """
    return list(_STEPS.keys())


def list_injectables():
    """
    List of registered injectables.

    """
    return list(_INJECTABLES.keys())


def list_broadcasts():
    """
    List of registered broadcasts as (cast table name, onto table name).

    """
    return list(_BROADCASTS.keys())


def is_expression(name):
    """
    Checks whether a given name is a simple variable name or a compound
    variable expression.

    Parameters
    ----------
    name : str

    Returns
    -------
    is_expr : bool

    """
    return '.' in name


def _collect_variables(names, expressions=None):
    """
    Map labels and expressions to registered variables.

    Handles argument matching.

    Example:

        _collect_variables(names=['zones', 'zone_id'],
                           expressions=['parcels.zone_id'])

    Would return a dict representing:

        {'parcels': <DataFrameWrapper for zones>,
         'zone_id': <pandas.Series for parcels.zone_id>}

    Parameters
    ----------
    names : list of str
        List of registered variable names and/or labels.
        If mixing names and labels, labels must come at the end.
    expressions : list of str, optional
        List of registered variable expressions for labels defined
        at end of `names`. Length must match the number of labels.

    Returns
    -------
    variables : dict
        Keys match `names`. Values correspond to registered variables,
        which may be wrappers or evaluated functions if appropriate.

    """
    # Map registered variable labels to expressions.
    if not expressions:
        expressions = []
    offset = len(names) - len(expressions)
    labels_map = dict(tz.concatv(
        tz.compatibility.zip(names[:offset], names[:offset]),
        tz.compatibility.zip(names[offset:], expressions)))

    all_variables = tz.merge(_INJECTABLES, _TABLES)
    variables = {}
    for label, expression in labels_map.items():
        # In the future, more registered variable expressions could be
        # supported. Currently supports names of registered variables
        # and references to table columns.
        if '.' in expression:
            # Registered variable expression refers to column.
            table_name, column_name = expression.split('.')
            table = get_table(table_name)
            variables[label] = table.get_column(column_name)
        else:
            thing = all_variables[expression]
            if isinstance(thing, (_InjectableFuncWrapper, TableFuncWrapper)):
                # Registered variable object is function.
                variables[label] = thing()
            else:
                variables[label] = thing

    return variables


def add_table(
        table_name, table, cache=False, cache_scope=_CS_FOREVER,
        copy_col=True):
    """
    Register a table with Orca.

    Parameters
    ----------
    table_name : str
        Should be globally unique to this table.
    table : pandas.DataFrame or function
        If a function, the function should return a DataFrame.
        The function's argument names and keyword argument values
        will be matched to registered variables when the function
        needs to be evaluated by Orca.
    cache : bool, optional
        Whether to cache the results of a provided callable. Does not
        apply if `table` is a DataFrame.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.
    copy_col : bool, optional
        Whether to return copies when evaluating columns.

    Returns
    -------
    wrapped : `DataFrameWrapper` or `TableFuncWrapper`

    """
    if isinstance(table, Callable):
        table = TableFuncWrapper(table_name, table, cache=cache,
                                 cache_scope=cache_scope, copy_col=copy_col)
    else:
        table = DataFrameWrapper(table_name, table, copy_col=copy_col)

    # clear any cached data from a previously registered table
    table.clear_cached()

    logger.debug('registering table {!r}'.format(table_name))
    _TABLES[table_name] = table

    return table


def table(
        table_name=None, cache=False, cache_scope=_CS_FOREVER, copy_col=True):
    """
    Decorates functions that return DataFrames.

    Decorator version of `add_table`. Table name defaults to
    name of function.

    The function's argument names and keyword argument values
    will be matched to registered variables when the function
    needs to be evaluated by Orca.
    The argument name "iter_var" may be used to have the current
    iteration variable injected.

    """
    def decorator(func):
        if table_name:
            name = table_name
        else:
            name = func.__name__
        add_table(
            name, func, cache=cache, cache_scope=cache_scope,
            copy_col=copy_col)
        return func
    return decorator


def get_raw_table(table_name):
    """
    Get a wrapped table by name and don't do anything to it.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    table : DataFrameWrapper or TableFuncWrapper

    """
    if is_table(table_name):
        return _TABLES[table_name]
    else:
        raise KeyError('table not found: {}'.format(table_name))


def get_table(table_name):
    """
    Get a registered table.

    Decorated functions will be converted to `DataFrameWrapper`.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    table : `DataFrameWrapper`

    """
    table = get_raw_table(table_name)
    if isinstance(table, TableFuncWrapper):
        table = table()
    return table


def table_type(table_name):
    """
    Returns the type of a registered table.

    The type can be either "dataframe" or "function".

    Parameters
    ----------
    table_name : str

    Returns
    -------
    table_type : {'dataframe', 'function'}

    """
    table = get_raw_table(table_name)

    if isinstance(table, DataFrameWrapper):
        return 'dataframe'
    elif isinstance(table, TableFuncWrapper):
        return 'function'


def add_column(
        table_name, column_name, column, cache=False, cache_scope=_CS_FOREVER):
    """
    Add a new column to a table from a Series or callable.

    Parameters
    ----------
    table_name : str
        Table with which the column will be associated.
    column_name : str
        Name for the column.
    column : pandas.Series or callable
        Series should have an index matching the table to which it
        is being added. If a callable, the function's argument
        names and keyword argument values will be matched to
        registered variables when the function needs to be
        evaluated by Orca. The function should return a Series.
    cache : bool, optional
        Whether to cache the results of a provided callable. Does not
        apply if `column` is a Series.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.

    """
    if isinstance(column, Callable):
        column = \
            _ColumnFuncWrapper(
                table_name, column_name, column,
                cache=cache, cache_scope=cache_scope)
    else:
        column = _SeriesWrapper(table_name, column_name, column)

    # clear any cached data from a previously registered column
    column.clear_cached()

    logger.debug('registering column {!r} on table {!r}'.format(
        column_name, table_name))
    _COLUMNS[(table_name, column_name)] = column

    return column


def column(table_name, column_name=None, cache=False, cache_scope=_CS_FOREVER):
    """
    Decorates functions that return a Series.

    Decorator version of `add_column`. Series index must match
    the named table. Column name defaults to name of function.

    The function's argument names and keyword argument values
    will be matched to registered variables when the function
    needs to be evaluated by Orca.
    The argument name "iter_var" may be used to have the current
    iteration variable injected.
    The index of the returned Series must match the named table.

    """
    def decorator(func):
        if column_name:
            name = column_name
        else:
            name = func.__name__
        add_column(
            table_name, name, func, cache=cache, cache_scope=cache_scope)
        return func
    return decorator


def list_columns_for_table(table_name):
    """
    Return a list of all the extra columns registered for a given table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    columns : list of str

    """
    return [cname for tname, cname in _COLUMNS.keys() if tname == table_name]


def _columns_for_table(table_name):
    """
    Return all of the columns registered for a given table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    columns : dict of column wrappers
        Keys will be column names.

    """
    return {cname: col
            for (tname, cname), col in _COLUMNS.items()
            if tname == table_name}


def column_map(tables, columns):
    """
    Take a list of tables and a list of column names and resolve which
    columns come from which table.

    Parameters
    ----------
    tables : sequence of _DataFrameWrapper or _TableFuncWrapper
        Could also be sequence of modified pandas.DataFrames, the important
        thing is that they have ``.name`` and ``.columns`` attributes.
    columns : sequence of str
        The column names of interest.

    Returns
    -------
    col_map : dict
        Maps table names to lists of column names.
    """
    if not columns:
        return {t.name: None for t in tables}

    columns = set(columns)
    colmap = {
        t.name: list(set(t.columns).intersection(columns)) for t in tables}
    foundcols = tz.reduce(
        lambda x, y: x.union(y), (set(v) for v in colmap.values()))
    if foundcols != columns:
        raise RuntimeError('Not all required columns were found. '
                           'Missing: {}'.format(list(columns - foundcols)))
    return colmap


def get_raw_column(table_name, column_name):
    """
    Get a wrapped, registered column.

    This function cannot return columns that are part of wrapped
    DataFrames, it's only for columns registered directly through Orca.

    Parameters
    ----------
    table_name : str
    column_name : str

    Returns
    -------
    wrapped : _SeriesWrapper or _ColumnFuncWrapper

    """
    try:
        return _COLUMNS[(table_name, column_name)]
    except KeyError:
        raise KeyError('column {!r} not found for table {!r}'.format(
            column_name, table_name))


def _memoize_function(f, name, cache_scope=_CS_FOREVER):
    """
    Wraps a function for memoization and ties it's cache into the
    Orca cacheing system.

    Parameters
    ----------
    f : function
    name : str
        Name of injectable.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.

    """
    cache = {}

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            cache_key = (
                args or None, frozenset(kwargs.items()) if kwargs else None)
            in_cache = cache_key in cache
        except TypeError:
            raise TypeError(
                'function arguments must be hashable for memoization')

        if _CACHING and in_cache:
            return cache[cache_key]
        else:
            result = f(*args, **kwargs)
            cache[cache_key] = result
            return result

    wrapper.__wrapped__ = f
    wrapper.cache = cache
    wrapper.clear_cached = lambda: cache.clear()
    _MEMOIZED[name] = CacheItem(name, wrapper, cache_scope)

    return wrapper


def add_injectable(
        name, value, autocall=True, cache=False, cache_scope=_CS_FOREVER,
        memoize=False):
    """
    Add a value that will be injected into other functions.

    Parameters
    ----------
    name : str
    value
        If a callable and `autocall` is True then the function's
        argument names and keyword argument values will be matched
        to registered variables when the function needs to be
        evaluated by Orca. The return value will
        be passed to any functions using this injectable. In all other
        cases, `value` will be passed through untouched.
    autocall : bool, optional
        Set to True to have injectable functions automatically called
        (with argument matching) and the result injected instead of
        the function itself.
    cache : bool, optional
        Whether to cache the return value of an injectable function.
        Only applies when `value` is a callable and `autocall` is True.
    cache_scope : {'step', 'iteration', 'forever'}, optional
        Scope for which to cache data. Default is to cache forever
        (or until manually cleared). 'iteration' caches data for each
        complete iteration of the pipeline, 'step' caches data for
        a single step of the pipeline.
    memoize : bool, optional
        If autocall is False it is still possible to cache function results
        by setting this flag to True. Cached values are stored in a dictionary
        keyed by argument values, so the argument values must be hashable.
        Memoized functions have their caches cleared according to the same
        rules as universal caching.

    """
    if isinstance(value, Callable):
        if autocall:
            value = _InjectableFuncWrapper(
                name, value, cache=cache, cache_scope=cache_scope)
            # clear any cached data from a previously registered value
            value.clear_cached()
        elif not autocall and memoize:
            value = _memoize_function(value, name, cache_scope=cache_scope)

    logger.debug('registering injectable {!r}'.format(name))
    _INJECTABLES[name] = value


def injectable(
        name=None, autocall=True, cache=False, cache_scope=_CS_FOREVER,
        memoize=False):
    """
    Decorates functions that will be injected into other functions.

    Decorator version of `add_injectable`. Name defaults to
    name of function.

    The function's argument names and keyword argument values
    will be matched to registered variables when the function
    needs to be evaluated by Orca.
    The argument name "iter_var" may be used to have the current
    iteration variable injected.

    """
    def decorator(func):
        if name:
            n = name
        else:
            n = func.__name__
        add_injectable(
            n, func, autocall=autocall, cache=cache, cache_scope=cache_scope,
            memoize=memoize)
        return func
    return decorator


def is_injectable(name):
    """
    Checks whether a given name can be mapped to an injectable.

    """
    return name in _INJECTABLES


def get_raw_injectable(name):
    """
    Return a raw, possibly wrapped injectable.

    Parameters
    ----------
    name : str

    Returns
    -------
    inj : _InjectableFuncWrapper or object

    """
    if is_injectable(name):
        return _INJECTABLES[name]
    else:
        raise KeyError('injectable not found: {!r}'.format(name))


def injectable_type(name):
    """
    Classify an injectable as either 'variable' or 'function'.

    Parameters
    ----------
    name : str

    Returns
    -------
    inj_type : {'variable', 'function'}
        If the injectable is an automatically called function or any other
        type of callable the type will be 'function', all other injectables
        will be have type 'variable'.

    """
    inj = get_raw_injectable(name)
    if isinstance(inj, (_InjectableFuncWrapper, Callable)):
        return 'function'
    else:
        return 'variable'


def get_injectable(name):
    """
    Get an injectable by name. *Does not* evaluate wrapped functions.

    Parameters
    ----------
    name : str

    Returns
    -------
    injectable
        Original value or evaluated value of an _InjectableFuncWrapper.

    """
    i = get_raw_injectable(name)
    return i() if isinstance(i, _InjectableFuncWrapper) else i


def get_injectable_func_source_data(name):
    """
    Return data about an injectable function's source, including file name,
    line number, and source code.

    Parameters
    ----------
    name : str

    Returns
    -------
    filename : str
    lineno : int
        The line number on which the function starts.
    source : str

    """
    if injectable_type(name) != 'function':
        raise ValueError('injectable {!r} is not a function'.format(name))

    inj = get_raw_injectable(name)

    if isinstance(inj, _InjectableFuncWrapper):
        return _func_source_data(inj._func)
    elif hasattr(inj, '__wrapped__'):
        return _func_source_data(inj.__wrapped__)
    else:
        return _func_source_data(inj)


def add_step(step_name, func):
    """
    Add a step function to Orca.

    The function's argument names and keyword argument values
    will be matched to registered variables when the function
    needs to be evaluated by Orca.
    The argument name "iter_var" may be used to have the current
    iteration variable injected.

    Parameters
    ----------
    step_name : str
    func : callable

    """
    if isinstance(func, Callable):
        logger.debug('registering step {!r}'.format(step_name))
        _STEPS[step_name] = _StepFuncWrapper(step_name, func)
    else:
        raise TypeError('func must be a callable')


def step(step_name=None):
    """
    Decorates functions that will be called by the `run` function.

    Decorator version of `add_step`. step name defaults to
    name of function.

    The function's argument names and keyword argument values
    will be matched to registered variables when the function
    needs to be evaluated by Orca.
    The argument name "iter_var" may be used to have the current
    iteration variable injected.

    """
    def decorator(func):
        if step_name:
            name = step_name
        else:
            name = func.__name__
        add_step(name, func)
        return func
    return decorator


def is_step(step_name):
    """
    Check whether a given name refers to a registered step.

    """
    return step_name in _STEPS


def get_step(step_name):
    """
    Get a wrapped step by name.

    Parameters
    ----------

    """
    if is_step(step_name):
        return _STEPS[step_name]
    else:
        raise KeyError('no step named {}'.format(step_name))


Broadcast = namedtuple(
    'Broadcast',
    ['cast', 'onto', 'cast_on', 'onto_on', 'cast_index', 'onto_index'])


def broadcast(cast, onto, cast_on=None, onto_on=None,
              cast_index=False, onto_index=False):
    """
    Register a rule for merging two tables by broadcasting one onto
    the other.

    Parameters
    ----------
    cast, onto : str
        Names of registered tables.
    cast_on, onto_on : str, optional
        Column names used for merge, equivalent of ``left_on``/``right_on``
        parameters of pandas.merge.
    cast_index, onto_index : bool, optional
        Whether to use table indexes for merge. Equivalent of
        ``left_index``/``right_index`` parameters of pandas.merge.

    """
    logger.debug(
        'registering broadcast of table {!r} onto {!r}'.format(cast, onto))
    _BROADCASTS[(cast, onto)] = \
        Broadcast(cast, onto, cast_on, onto_on, cast_index, onto_index)


def _get_broadcasts(tables):
    """
    Get the broadcasts associated with a set of tables.

    Parameters
    ----------
    tables : sequence of str
        Table names for which broadcasts have been registered.

    Returns
    -------
    casts : dict of `Broadcast`
        Keys are tuples of strings like (cast_name, onto_name).

    """
    tables = set(tables)
    casts = tz.keyfilter(
        lambda x: x[0] in tables and x[1] in tables, _BROADCASTS)
    if tables - set(tz.concat(casts.keys())):
        raise ValueError('Not enough links to merge all tables.')
    return casts


def is_broadcast(cast_name, onto_name):
    """
    Checks whether a relationship exists for broadcast `cast_name`
    onto `onto_name`.

    """
    return (cast_name, onto_name) in _BROADCASTS


def get_broadcast(cast_name, onto_name):
    """
    Get a single broadcast.

    Broadcasts are stored data about how to do a Pandas join.
    A Broadcast object is a namedtuple with these attributes:

        - cast: the name of the table being broadcast
        - onto: the name of the table onto which "cast" is broadcast
        - cast_on: The optional name of a column on which to join.
          None if the table index will be used instead.
        - onto_on: The optional name of a column on which to join.
          None if the table index will be used instead.
        - cast_index: True if the table index should be used for the join.
        - onto_index: True if the table index should be used for the join.

    Parameters
    ----------
    cast_name : str
        The name of the table being braodcast.
    onto_name : str
        The name of the table onto which `cast_name` is broadcast.

    Returns
    -------
    broadcast : Broadcast

    """
    if is_broadcast(cast_name, onto_name):
        return _BROADCASTS[(cast_name, onto_name)]
    else:
        raise KeyError(
            'no rule found for broadcasting {!r} onto {!r}'.format(
                cast_name, onto_name))


# utilities for merge_tables
def _all_reachable_tables(t):
    """
    A generator that provides all the names of tables that can be
    reached via merges starting at the given target table.

    """
    for k, v in t.items():
        for tname in _all_reachable_tables(v):
            yield tname
        yield k


def _recursive_getitem(d, key):
    """
    Descend into a dict of dicts to return the one that contains
    a given key. Every value in the dict must be another dict.

    """
    if key in d:
        return d
    else:
        for v in d.values():
            return _recursive_getitem(v, key)
        else:
            raise KeyError('Key not found: {}'.format(key))


def _dict_value_to_pairs(d):
    """
    Takes the first value of a dictionary (which it self should be
    a dictionary) and turns it into a series of {key: value} dicts.

    For example, _dict_value_to_pairs({'c': {'a': 1, 'b': 2}}) will yield
    {'a': 1} and {'b': 2}.

    """
    d = d[tz.first(d)]

    for k, v in d.items():
        yield {k: v}


def _is_leaf_node(merge_node):
    """
    Returns True for dicts like {'a': {}}.

    """
    return len(merge_node) == 1 and not next(iter(merge_node.values()))


def _next_merge(merge_node):
    """
    Gets a node that has only leaf nodes below it. This table and
    the ones below are ready to be merged to make a new leaf node.

    """
    if all(_is_leaf_node(d) for d in _dict_value_to_pairs(merge_node)):
        return merge_node
    else:
        for d in tz.remove(_is_leaf_node, _dict_value_to_pairs(merge_node)):
            return _next_merge(d)
        else:
            raise OrcaError('No node found for next merge.')


def merge_tables(target, tables, columns=None, drop_intersection=True):
    """
    Merge a number of tables onto a target table. Tables must have
    registered merge rules via the `broadcast` function.

    Parameters
    ----------
    target : str, DataFrameWrapper, or TableFuncWrapper
        Name of the table (or wrapped table) onto which tables will be merged.
    tables : list of `DataFrameWrapper`, `TableFuncWrapper`, or str
        All of the tables to merge. Should include the target table.
    columns : list of str, optional
        If given, columns will be mapped to `tables` and only those columns
        will be requested from each table. The final merged table will have
        only these columns. By default all columns are used from every
        table.
    drop_intersection : bool
        If True, keep the left most occurence of any column name if it occurs
        on more than one table.  This prevents getting back the same column
        with suffixes applied by pd.merge.  If false, columns names will be
        suffixed with the table names - e.g. zone_id_buildings and
        zone_id_parcels.

    Returns
    -------
    merged : pandas.DataFrame

    """
    # allow target to be string or table wrapper
    if isinstance(target, (DataFrameWrapper, TableFuncWrapper)):
        target = target.name

    # allow tables to be strings or table wrappers
    tables = [get_table(t)
              if not isinstance(t, (DataFrameWrapper, TableFuncWrapper)) else t
              for t in tables]

    merges = {t.name: {} for t in tables}
    tables = {t.name: t for t in tables}
    casts = _get_broadcasts(tables.keys())
    logger.debug(
        'attempting to merge tables {} to target table {}'.format(
            tables.keys(), target))

    # relate all the tables by registered broadcasts
    for table, onto in casts:
        merges[onto][table] = merges[table]
    merges = {target: merges[target]}

    # verify that all the tables can be merged to the target
    all_tables = set(_all_reachable_tables(merges))

    if all_tables != set(tables.keys()):
        raise RuntimeError(
            ('Not all tables can be merged to target "{}". Unlinked tables: {}'
             ).format(target, list(set(tables.keys()) - all_tables)))

    # add any columns necessary for indexing into other tables
    # during merges
    if columns:
        columns = list(columns)
        for c in casts.values():
            if c.onto_on:
                columns.append(c.onto_on)
            if c.cast_on:
                columns.append(c.cast_on)

    # get column map for which columns go with which table
    colmap = column_map(tables.values(), columns)

    # get frames
    frames = {name: t.to_frame(columns=colmap[name])
              for name, t in tables.items()}

    past_intersections = set()

    # perform merges until there's only one table left
    while merges[target]:
        nm = _next_merge(merges)
        onto = tz.first(nm)
        onto_table = frames[onto]

        # loop over all the tables that can be broadcast onto
        # the onto_table and merge them all in.
        for cast in nm[onto]:
            cast_table = frames[cast]
            bc = casts[(cast, onto)]

            with log_start_finish(
                    'merge tables {} and {}'.format(onto, cast), logger):

                intersection = set(onto_table.columns).\
                    intersection(cast_table.columns)
                # intersection is ok if it's the join key
                intersection.discard(bc.onto_on)
                intersection.discard(bc.cast_on)
                # otherwise drop so as not to create conflicts
                if drop_intersection:
                    cast_table = cast_table.drop(intersection, axis=1)
                else:
                    # add suffix to past intersections which wouldn't get
                    # picked up by the merge - these we have to rename by hand
                    renames = dict(zip(
                        past_intersections,
                        [c+'_'+onto for c in past_intersections]
                    ))
                    onto_table = onto_table.rename(columns=renames)

                # keep track of past intersections in case there's an odd
                # number of intersections
                past_intersections = past_intersections.union(intersection)

                onto_table = pd.merge(
                    onto_table, cast_table,
                    suffixes=['_'+onto, '_'+cast],
                    left_on=bc.onto_on, right_on=bc.cast_on,
                    left_index=bc.onto_index, right_index=bc.cast_index)

        # replace the existing table with the merged one
        frames[onto] = onto_table

        # free up space by dropping the cast table
        del frames[cast]

        # mark the onto table as having no more things to broadcast
        # onto it.
        _recursive_getitem(merges, onto)[onto] = {}

    logger.debug('finished merge')
    return frames[target]


def get_step_table_names(steps):
    """
    Returns a list of table names injected into the provided steps.

    Parameters
    ----------
    steps: list of str
        Steps to gather table inputs from.

    Returns
    -------
    list of str

    """
    table_names = set()
    for s in steps:
        table_names |= get_step(s)._tables_used()
    return list(table_names)


def write_tables(fname, table_names=None, prefix=None, compress=False, local=False):
    """
    Writes tables to a pandas.HDFStore file.

    Parameters
    ----------
    fname : str
        File name for HDFStore. Will be opened in append mode and closed
        at the end of this function.
    table_names: list of str, optional, default None
        List of tables to write. If None, all registered tables will
        be written.
    prefix: str
        If not None, used to prefix the output table names so that
        multiple iterations can go in the same file.
    compress: boolean
        Whether to compress output file using standard HDF5-readable
        zlib compression, default False.

    """
    if table_names is None:
        table_names = list_tables()

    tables = (get_table(t) for t in table_names)
    key_template = '{}/{{}}'.format(prefix) if prefix is not None else '{}'

    # set compression options to zlib level-1 if compress arg is True
    complib = compress and 'zlib' or None
    complevel = compress and 1 or 0

    with pd.HDFStore(fname, mode='a', complib=complib, complevel=complevel) as store:
        for t in tables:
            # if local arg is True, store only local columns
            columns = None
            if local is True:
                columns = t.local_columns
            store[key_template.format(t.name)] = t.to_frame(columns=columns)


iter_step = namedtuple('iter_step', 'step_num,step_name')


def run(steps, iter_vars=None, data_out=None, out_interval=1,
        out_base_tables=None, out_run_tables=None, compress=False,
        out_base_local=True, out_run_local=True):
    """
    Run steps in series, optionally repeatedly over some sequence.
    The current iteration variable is set as a global injectable
    called ``iter_var``.

    Parameters
    ----------
    steps : list of str
        List of steps to run identified by their name.
    iter_vars : iterable, optional
        The values of `iter_vars` will be made available as an injectable
        called ``iter_var`` when repeatedly running `steps`.
    data_out : str, optional
        An optional filename to which all tables injected into any step
        in `steps` will be saved every `out_interval` iterations.
        File will be a pandas HDF data store.
    out_interval : int, optional
        Iteration interval on which to save data to `data_out`. For example,
        2 will save out every 2 iterations, 5 every 5 iterations.
        Default is every iteration.
        The results of the first and last iterations are always included.
        The input (base) tables are also included and prefixed with `base/`,
        these represent the state of the system before any steps have been
        executed.
        The interval is defined relative to the first iteration. For example,
        a run begining in 2015 with an out_interval of 2, will write out
        results for 2015, 2017, etc.
    out_base_tables: list of str, optional, default None
        List of base tables to write. If not provided, tables injected
        into 'steps' will be written.
    out_run_tables: list of str, optional, default None
        List of run tables to write. If not provided, tables injected
        into 'steps' will be written.
    compress: boolean, optional, default False
        Whether to compress output file using standard HDF5 zlib compression.
        Compression yields much smaller files using slightly more CPU.
    out_base_local: boolean, optional, default True
        For tables in out_base_tables, whether to store only local columns (True)
        or both, local and computed columns (False).
    out_run_local: boolean, optional, default True
        For tables in out_run_tables, whether to store only local columns (True)
        or both, local and computed columns (False).
    """
    iter_vars = iter_vars or [None]
    max_i = len(iter_vars)

    # get the tables to write out
    if out_base_tables is None or out_run_tables is None:
        step_tables = get_step_table_names(steps)

        if out_base_tables is None:
            out_base_tables = step_tables

        if out_run_tables is None:
            out_run_tables = step_tables

    # write out the base (inputs)
    if data_out:
        add_injectable('iter_var', iter_vars[0])
        write_tables(data_out, out_base_tables,
                     prefix='base', compress=compress, local=out_base_local)

    # run the steps
    for i, var in enumerate(iter_vars, start=1):
        add_injectable('iter_var', var)

        if var is not None:
            logger.debug(
                'running iteration {} with iteration value {!r}'.format(
                    i, var))

        for j, step_name in enumerate(steps):
            add_injectable('iter_step', iter_step(j, step_name))
            with log_start_finish(
                    'run step {!r}'.format(step_name), logger,
                    logging.INFO):
                step = get_step(step_name)
                step()
            clear_cache(scope=_CS_STEP)

        # write out the results for the current iteration
        if data_out:
            if (i - 1) % out_interval == 0 or i == max_i:
                write_tables(data_out, out_run_tables,
                             prefix=var, compress=compress, local=out_run_local)

        clear_cache(scope=_CS_ITER)


@contextmanager
def injectables(**kwargs):
    """
    Temporarily add injectables to the pipeline environment.
    Takes only keyword arguments.

    Injectables will be returned to their original state when the context
    manager exits.

    """
    global _INJECTABLES

    original = _INJECTABLES.copy()
    _INJECTABLES.update(kwargs)
    yield
    _INJECTABLES = original


@contextmanager
def temporary_tables(**kwargs):
    """
    Temporarily set DataFrames as registered tables.

    Tables will be returned to their original state when the context
    manager exits. Caching is not enabled for tables registered via
    this function.

    """
    global _TABLES

    original = _TABLES.copy()

    for k, v in kwargs.items():
        if not isinstance(v, pd.DataFrame):
            raise ValueError('tables only accepts DataFrames')
        add_table(k, v)

    yield

    _TABLES = original


def eval_variable(name, **kwargs):
    """
    Execute a single variable function registered with Orca
    and return the result. Any keyword arguments are temporarily set
    as injectables. This gives the value as would be injected into a function.

    Parameters
    ----------
    name : str
        Name of variable to evaluate.
        Use variable expressions to specify columns.

    Returns
    -------
    object
        For injectables and columns this directly returns whatever
        object is returned by the registered function.
        For tables this returns a DataFrameWrapper as if the table
        had been injected into a function.

    """
    with injectables(**kwargs):
        vars = _collect_variables([name], [name])
        return vars[name]


def eval_step(name, **kwargs):
    """
    Evaluate a step as would be done within the pipeline environment
    and return the result. Any keyword arguments are temporarily set
    as injectables.

    Parameters
    ----------
    name : str
        Name of step to run.

    Returns
    -------
    object
        Anything returned by a step. (Though note that in Orca runs
        return values from steps are ignored.)

    """
    with injectables(**kwargs):
        return get_step(name)()

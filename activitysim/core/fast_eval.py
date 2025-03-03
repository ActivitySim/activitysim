from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.core.computation.eval import eval as _eval

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping, Sequence

    from pandas._typing import ArrayLike


def _get_cleaned_column_resolvers(
    df: pd.DataFrame, raw: bool = True
) -> dict[Hashable, ArrayLike | pd.Series]:
    """
    Return the special character free column resolvers of a dataframe.

    Column names with special characters are 'cleaned up' so that they can
    be referred to by backtick quoting.
    Used in :meth:`DataFrame.eval`.
    """
    from pandas.core.computation.parsing import clean_column_name
    from pandas.core.series import Series

    if isinstance(df, pd.Series):
        return {clean_column_name(df.name): df}

    # CHANGED FROM PANDAS: do not even convert the arrays to pd.Series, just
    # give the raw arrays to the compute engine. This is potentially a breaking
    # change if any of the operations in the eval string require a pd.Series.
    if raw:
        return {
            clean_column_name(k): v
            for k, v in zip(df.columns, df._iter_column_arrays())
            if not isinstance(k, int)
        }

    # CHANGED FROM PANDAS: do not call df.dtype inside the dict comprehension loop
    # This update has been made in https://github.com/pandas-dev/pandas/pull/59573,
    # but appears not to have been released yet as of pandas 2.2.3
    dtypes = df.dtypes

    return {
        clean_column_name(k): Series(
            v, copy=False, index=df.index, name=k, dtype=dtypes[k]
        ).__finalize__(df)
        for k, v in zip(df.columns, df._iter_column_arrays())
        if not isinstance(k, int)
    }


def fast_eval(df: pd.DataFrame, expr: str, **kwargs) -> Any | None:
    """
    Evaluate a string describing operations on DataFrame columns.

    Operates on columns only, not specific rows or elements.  This allows
    `eval` to run arbitrary code, which can make you vulnerable to code
    injection if you pass user input to this function.

    Parameters
    ----------
    expr : str
        The expression string to evaluate.
    **kwargs
        See the documentation for  :meth:`~pandas.DataFrame.eval` for complete
        details on the keyword arguments accepted.

    Returns
    -------
    ndarray, scalar, or pandas object
        The result of the evaluation.
    """

    inplace = False
    kwargs["level"] = kwargs.pop("level", 0) + 1
    index_resolvers = df._get_index_resolvers()
    column_resolvers = _get_cleaned_column_resolvers(df)
    resolvers = column_resolvers, index_resolvers
    if "target" not in kwargs:
        kwargs["target"] = df
    kwargs["resolvers"] = tuple(kwargs.get("resolvers", ())) + resolvers

    try:
        return _eval(expr, inplace=inplace, **kwargs)
    except Exception as e:
        # Initially assume that the exception is caused by the potentially
        # breaking change in _get_cleaned_column_resolvers, and try again
        # TODO: what kind of exception should be caught here so it is less broad
        column_resolvers = _get_cleaned_column_resolvers(df, raw=False)
        resolvers = column_resolvers, index_resolvers
        kwargs["resolvers"] = kwargs["resolvers"][:-2] + resolvers
        return _eval(expr, inplace=inplace, **kwargs)

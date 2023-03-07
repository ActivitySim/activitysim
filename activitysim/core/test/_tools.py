# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd


def run_if_exists(filename):
    import pytest

    stack = traceback.extract_stack()
    base_dir = Path(stack[-2].filename).parent
    target_file = base_dir.joinpath(filename)

    return pytest.mark.skipif(
        not target_file.exists(), reason=f"required file {filename} is missing"
    )


def assert_frame_substantively_equal(
    left,
    right,
    *args,
    ignore_column_order=True,
    ignore_extra_columns_left=False,
    **kwargs,
):
    """
    Check that left and right DataFrame are substantively equal.

    This method generalizes the usual pandas DataFrame test, by allowing
    the ordering of columns to be different, and allowing the left dataframe to
    have extra columns (e.g. as might happen if more reporting or debugging
    data is output into a dataframe, but we want to make sure that the "core"
    expected things are all there and correct.

    Parameters
    ----------
    left, right : pd.DataFrame
    *args
        Forwarded to pandas.testing.assert_frame_equal
    ignore_column_order : bool, default True
        Keyword only argument.
    ignore_extra_columns_left : bool, default False
        This cannot be True unless `ignore_column_order` is also True
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    **kwargs
        Forwarded to pandas.testing.assert_frame_equal
    """
    if ignore_extra_columns_left:
        assert ignore_column_order
        assert set(right.columns).issubset(left.columns)
        left = left[right.columns]

    elif ignore_column_order:
        # column order may not match, so fix it before checking
        assert sorted(left.columns) == sorted(right.columns)
        # if there are duplicate column names, we disavow this option
        if not left.columns.has_duplicates:
            left = left[right.columns]

    try:
        pd.testing.assert_frame_equal(left, right, *args, **kwargs)
    except Exception as err:
        print(err)
        raise


def assert_equal(x, y):
    assert x == y

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
import pytest


def run_if_exists(filename):
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
        left = left[right.columns]

    pd.testing.assert_frame_equal(left, right, *args, **kwargs)

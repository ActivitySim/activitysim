# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core.steps._decode import _apply_decode_filter, _decode_output_column


@pytest.fixture
def zone_labels():
    """Sample zone ID lookup array (index 0..4 -> zone labels)."""
    return np.array([100, 200, 300, 400, 500])


def test_decode_output_column_no_nulls(zone_labels):
    """Basic decode without nulls: each integer index maps to the zone label."""
    map_func = zone_labels.__getitem__
    column = pd.array([0, 1, 2, 3, 4])
    result = _decode_output_column(column, map_func)
    expected = pd.Series([100, 200, 300, 400, 500])
    pdt.assert_series_equal(result, expected)


def test_decode_output_column_preserve_nulls(zone_labels):
    """With preserve_nulls=True, NaN entries pass through unchanged."""
    map_func = zone_labels.__getitem__
    column = pd.array([0, pd.NA, 2, pd.NA, 4], dtype="Int64")
    result = _decode_output_column(column, map_func, preserve_nulls=True)
    # Non-null positions are decoded; null positions remain null.
    assert result[0] == 100
    assert result[2] == 300
    assert result[4] == 500
    assert pd.isna(result[1])
    assert pd.isna(result[3])


def test_apply_decode_filter_none(zone_labels):
    """No filter: map_func is direct indexing and preserve_nulls is False."""
    map_func, preserve_nulls = _apply_decode_filter(zone_labels, None)
    assert not preserve_nulls
    assert map_func(2) == zone_labels[2]


def test_apply_decode_filter_nonnegative(zone_labels):
    """nonnegative filter: negative values pass through; others are decoded."""
    map_func, preserve_nulls = _apply_decode_filter(zone_labels, "nonnegative")
    assert not preserve_nulls
    assert map_func(1) == 200
    assert map_func(-1) == -1


def test_apply_decode_filter_nullable_nonnegative(zone_labels):
    """nullable_nonnegative filter: negative values pass through; preserve_nulls is True."""
    map_func, preserve_nulls = _apply_decode_filter(zone_labels, "nullable_nonnegative")
    assert preserve_nulls
    assert map_func(1) == 200
    assert map_func(-1) == -1


def test_apply_decode_filter_unknown():
    """Unknown filter name raises ValueError."""
    with pytest.raises(ValueError, match="unknown decode_filter"):
        _apply_decode_filter([], "bad_filter")


def test_nullable_nonnegative_with_nulls_and_sentinels(zone_labels):
    """
    End-to-end: a column with nulls and -1 sentinel values decoded via
    nullable_nonnegative should pass nulls and negative values through unchanged
    while mapping non-negative indexes to zone labels.
    """
    map_func, preserve_nulls = _apply_decode_filter(zone_labels, "nullable_nonnegative")
    column = pd.array([0, pd.NA, -1, 3, pd.NA], dtype="Int64")
    result = _decode_output_column(column, map_func, preserve_nulls=preserve_nulls)
    assert result[0] == 100
    assert pd.isna(result[1])
    assert result[2] == -1
    assert result[3] == 400
    assert pd.isna(result[4])

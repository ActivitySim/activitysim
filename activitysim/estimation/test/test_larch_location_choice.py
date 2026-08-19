from __future__ import annotations

import pandas as pd

from activitysim.estimation.larch.location_choice import (
    _suffix_overlapping_chooser_columns,
)


def test_suffix_overlapping_chooser_columns():
    chooser_data = pd.DataFrame(
        {
            "DISTRICT": [1, 2],
            "income": [50_000, 75_000],
        }
    )
    alternative_data = pd.DataFrame(
        {
            "DISTRICT": [3, 4],
            "size_term": [10.0, 20.0],
        }
    )

    result = _suffix_overlapping_chooser_columns(chooser_data, alternative_data)

    assert list(result.columns) == ["DISTRICT_chooser", "income"]
    assert list(chooser_data.columns) == ["DISTRICT", "income"]


def test_suffix_overlapping_chooser_columns_without_overlap():
    chooser_data = pd.DataFrame({"income": [50_000, 75_000]})
    alternative_data = pd.DataFrame({"size_term": [10.0, 20.0]})

    result = _suffix_overlapping_chooser_columns(chooser_data, alternative_data)

    assert result is chooser_data

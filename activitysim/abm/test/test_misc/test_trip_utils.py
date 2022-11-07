import numpy as np
import pandas as pd
import pytest

from activitysim.abm.models.util.trip import get_time_windows


@pytest.mark.parametrize(
    "duration, levels, expected",
    [
        (24, 3, 2925),
        (24, 2, 325),
        (24, 1, 25),
        (48, 3, 20825),
        (48, 2, 1225),
        (48, 1, 49),
    ],
)
def test_get_time_windows(duration, levels, expected):
    time_windows = get_time_windows(duration, levels)

    if levels == 1:
        assert time_windows.ndim == 1
        assert len(time_windows) == expected
        assert np.sum(time_windows <= duration) == expected
    else:
        assert len(time_windows) == levels
        assert len(time_windows[0]) == expected
        total_duration = np.sum(time_windows, axis=0)
        assert np.sum(total_duration <= duration) == expected

    df = pd.DataFrame(np.transpose(time_windows))
    assert len(df) == len(df.drop_duplicates())

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import numpy as np
import pandas as pd


def _decode_output_column(column, map_func, preserve_nulls=False):
    series = pd.Series(column)
    if preserve_nulls:
        revised_col = series.copy()
        notna = series.notna()
        revised_col.loc[notna] = series.loc[notna].astype(int).map(map_func)
        return revised_col
    return series.astype(int).map(map_func)


def _apply_decode_filter(map_col, decode_filter):
    preserve_nulls = False

    if decode_filter is None:
        return map_col.__getitem__, preserve_nulls

    if decode_filter == "nonnegative":

        def map_func(x):
            return x if x < 0 else map_col[x]

        return map_func, preserve_nulls

    if decode_filter == "nullable_nonnegative":
        preserve_nulls = True

        def map_func(x):
            return x if x < 0 else map_col[x]

        return map_func, preserve_nulls

    raise ValueError(f"unknown decode_filter {decode_filter}")

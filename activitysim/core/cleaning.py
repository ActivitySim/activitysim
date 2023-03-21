from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import workflow

logger = logging.getLogger(__name__)


def recode_to_zero_based(values, mapping):
    values = np.asarray(values)
    zone_ids = pd.Index(mapping, dtype=np.int32)
    if (
        zone_ids.is_monotonic_increasing
        and zone_ids[-1] == len(zone_ids) + zone_ids[0] - 1
    ):
        offset = zone_ids[0]
        result = values - offset
    else:
        n = len(zone_ids)
        remapper = dict(zip(zone_ids, pd.RangeIndex(n)))
        if n < 128:
            out_dtype = np.int8
        elif n < (1 << 15):
            out_dtype = np.int16
        elif n < (1 << 31):
            out_dtype = np.int32
        else:
            out_dtype = np.int64
        result = np.fromiter((remapper.get(xi) for xi in values), out_dtype)
    return result


def should_recode_based_on_table(state: workflow.State, tablename):
    try:
        base_df = state.get_dataframe(tablename)
    except (KeyError, RuntimeError):
        # the basis table is missing, do not
        return False
    except AssertionError:
        if state.settings.input_table_list is None:
            # some tests don't include table definitions.
            return False
        raise
    if base_df.index.name and f"_original_{base_df.index.name}" in base_df:
        return True
    return False


def recode_based_on_table(state: workflow.State, values, tablename):
    try:
        base_df = state.get_dataframe(tablename)
    except (KeyError, RuntimeError):
        # the basis table is missing, do nothing
        logger.warning(f"unable to recode based on missing {tablename} table")
        return values
    except AssertionError:
        if state.settings.input_table_list is None:
            # some tests don't include table definitions.
            logger.warning(f"unable to recode based on missing {tablename} table")
            return values
        raise
    if base_df.index.name and f"_original_{base_df.index.name}" in base_df:
        source_ids = base_df[f"_original_{base_df.index.name}"]
        if (
            isinstance(base_df.index, pd.RangeIndex)
            and base_df.index.start == 0
            and base_df.index.step == 1
        ):
            logger.info(f"recoding to zero-based values based on {tablename} table")
            return recode_to_zero_based(values, source_ids)
        elif (
            base_df.index.is_monotonic_increasing
            and base_df.index[0] == 0
            and base_df.index[-1] == len(base_df) - 1
        ):
            logger.info(f"recoding to zero-based values based on {tablename} table")
            return recode_to_zero_based(values, source_ids)
        else:
            logger.info(f"recoding to mapped values based on {tablename} table")
            remapper = dict(zip(source_ids, base_df.index))
            return np.fromiter((remapper.get(xi) for xi in values), base_df.index.dtype)
    else:
        return values

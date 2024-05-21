# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from activitysim.core import workflow
from activitysim.core.exceptions import MissingInputTableDefinition
from activitysim.core.input import read_input_table
from activitysim.core.los import Network_LOS
from activitysim.core.skim_dictionary import SkimDict

logger = logging.getLogger(__name__)


@workflow.table
def land_use(state: workflow.State):
    df = read_input_table(state, "land_use")

    # try to make life easy for everybody by keeping everything in canonical order
    # but as long as coalesce_pipeline doesn't sort tables it coalesces, it might not stay in order
    # so even though we do this, anyone downstream who depends on it, should look out for themselves...
    if not df.index.is_monotonic_increasing:
        logger.info(f"sorting land_use index")
        df = df.sort_index()

    sharrow_enabled = state.settings.sharrow
    if sharrow_enabled:
        err_msg = (
            "a zero-based land_use index is required for sharrow,\n"
            "try adding `recode_pipeline_columns: true` to your settings file."
        )
        # when using sharrow, the land use file must be organized (either in raw
        # form or via recoding) so that the index is zero-based and contiguous
        assert df.index.is_monotonic_increasing, err_msg
        assert df.index[0] == 0, err_msg
        assert df.index[-1] == len(df.index) - 1, err_msg
        assert df.index.dtype.kind == "i", err_msg

    logger.info("loaded land_use %s" % (df.shape,))
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug("land_use.info:\n" + buffer.getvalue())
    return df


@workflow.table
def land_use_taz(state: workflow.State):
    try:
        df = read_input_table(state, "land_use_taz")
    except MissingInputTableDefinition:
        # if the land_use_taz table is not given explicitly in the settings,
        # we will construct our best approximation of the table by collecting
        # a sorted list of unique TAZ ids found in the land_use table of MAZs.
        # In nearly all cases this should be good enough, unless the model
        # includes TAZs without MAZs (e.g. external stations) or for some
        # reason wants TAZs in some not-sorted ordering.
        land_use = state.get_dataframe("land_use")
        if "TAZ" not in land_use:
            raise
        logger.warning(
            "no land_use_taz defined in input_table_list, constructing "
            "from discovered TAZ values in land_use"
        )
        # use original TAZ values if available, otherwise use current TAZ values
        if state.settings.recode_pipeline_columns and "_original_TAZ" in land_use:
            unique_tazs = np.unique(land_use["_original_TAZ"])
        else:
            unique_tazs = np.unique(land_use["TAZ"])
        if state.settings.recode_pipeline_columns:
            df = pd.Series(
                unique_tazs,
                name="_original_TAZ",
                index=pd.RangeIndex(unique_tazs.size, name="TAZ"),
            ).to_frame()
        else:
            df = pd.DataFrame(
                index=pd.Index(unique_tazs, name="TAZ"),
            )

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded land_use_taz %s" % (df.shape,))
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug("land_use_taz.info:\n" + buffer.getvalue())

    # replace table function with dataframe
    state.add_table("land_use_taz", df)

    return df

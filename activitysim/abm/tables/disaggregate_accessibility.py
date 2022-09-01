# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import inject
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


@inject.table()
def accessibility(land_use):
    """
    If 'accessibility' is in input_tables list, then read it in,
    otherwise create skeleton table with same index as landuse.

    This allows loading of pre-computed accessibility table, which is particularly useful
    for single-process small household sample runs when there are many zones in landuse

    skeleton table only required if multiprocessing wants to slice accessibility,
    otherwise it will simply be replaced when accessibility model is run
    """

    accessibility_df = read_input_table("accessibility", required=False)

    if accessibility_df is None:
        accessibility_df = pd.DataFrame(index=land_use.index)
        logger.debug(
            "created placeholder accessibility table %s" % (accessibility_df.shape,)
        )
    else:
        assert accessibility_df.sort_index().index.equals(
            land_use.to_frame().sort_index().index
        ), f"loaded accessibility table index does not match index of land_use table"
        logger.info("loaded land_use %s" % (accessibility_df.shape,))

    # replace table function with dataframe
    inject.add_table("accessibility", accessibility_df)

    return accessibility_df

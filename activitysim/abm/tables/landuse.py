# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import inject
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


@inject.table()
def land_use():

    df = read_input_table("land_use")

    # try to make life easy for everybody by keeping everything in canonical order
    # but as long as coalesce_pipeline doesn't sort tables it coalesces, it might not stay in order
    # so even though we do this, anyone downstream who depends on it, should look out for themselves...
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded land_use %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table("land_use", df)

    return df


inject.broadcast("land_use", "households", cast_index=True, onto_on="home_zone_id")

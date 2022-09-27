# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import inject
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

@inject.table()
def maz_centroids():
    df = read_input_table("maz_centroids")

    # try to make life easy for everybody by keeping everything in canonical order
    # but as long as coalesce_pipeline doesn't sort tables it coalesces, it might not stay in order
    # so even though we do this, anyone downstream who depends on it, should look out for themselves...
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded maz_centroids %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table("maz_centroids", df)

    return df

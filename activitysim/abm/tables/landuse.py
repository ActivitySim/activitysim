# ActivitySim
# See full license in LICENSE.txt.
import io
import logging

from activitysim.core import config, inject
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


@inject.table()
def land_use():

    df = read_input_table("land_use")

    sharrow_enabled = config.setting("sharrow", False)
    if sharrow_enabled:
        # when using sharrow, the land use file must be organized (either in raw
        # form or via recoding) so that the index is zero-based and contiguous
        assert df.index.is_monotonic_increasing
        assert df.index[0] == 0
        assert df.index[-1] == len(df.index) - 1
        assert df.index.dtype.kind == "i"

    # try to make life easy for everybody by keeping everything in canonical order
    # but as long as coalesce_pipeline doesn't sort tables it coalesces, it might not stay in order
    # so even though we do this, anyone downstream who depends on it, should look out for themselves...
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded land_use %s" % (df.shape,))
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug("land_use.info:\n" + buffer.getvalue())

    # replace table function with dataframe
    inject.add_table("land_use", df)

    return df


inject.broadcast("land_use", "households", cast_index=True, onto_on="home_zone_id")


@inject.table()
def land_use_taz():

    df = read_input_table("land_use_taz")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded land_use_taz %s" % (df.shape,))
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug("land_use_taz.info:\n" + buffer.getvalue())

    # replace table function with dataframe
    inject.add_table("land_use_taz", df)

    return df

# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import inject
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


@inject.table()
def land_use():

    df = read_input_table("land_use")

    logger.info("loaded land_use %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table('land_use', df)

    return df


inject.broadcast('land_use', 'households', cast_index=True, onto_on='home_zone_id')

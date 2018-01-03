# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import inject

logger = logging.getLogger(__name__)


@inject.table()
def land_use(store):

    df = store["land_use/taz_data"]

    logger.info("loaded land_use %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table('land_use', df)

    return df


inject.broadcast('land_use', 'households', cast_index=True, onto_on='TAZ')

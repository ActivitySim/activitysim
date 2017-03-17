# ActivitySim
# See full license in LICENSE.txt.

import logging
import orca


logger = logging.getLogger(__name__)


@orca.table()
def land_use(store):

    df = store["land_use/taz_data"]

    logger.info("loaded land_use %s" % (df.shape,))

    # replace table function with dataframe
    orca.add_table('land_use', df)

    return df


orca.broadcast('land_use', 'households', cast_index=True, onto_on='TAZ')

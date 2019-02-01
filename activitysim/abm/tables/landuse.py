# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import inject
from .input_store import read_input_table

logger = logging.getLogger(__name__)


@inject.table()
def land_use():

    df = read_input_table("land_use_taz")

    logger.info("loaded land_use %s" % (df.shape,))

    df.index.name = 'TAZ'

    # replace table function with dataframe
    inject.add_table('land_use', df)

    return df


inject.broadcast('land_use', 'households', cast_index=True, onto_on='TAZ')

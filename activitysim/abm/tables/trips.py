# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import inject


logger = logging.getLogger(__name__)


@inject.table()
def trips_merged(trips, tours):
    return inject.merge_tables(trips.name, tables=[trips, tours])


inject.broadcast('tours', 'trips', cast_index=True, onto_on='tour_id')

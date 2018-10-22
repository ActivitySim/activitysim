# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import inject

logger = logging.getLogger(__name__)


@inject.table()
def tours_merged(tours, persons_merged):
    return inject.merge_tables(tours.name, tables=[
        tours, persons_merged])


inject.broadcast('persons_merged', 'tours', cast_index=True, onto_on='person_id')

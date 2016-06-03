# ActivitySim
# See full license in LICENSE.txt.

import logging
import orca


logger = logging.getLogger(__name__)


def add_dependent_columns(base_dfname, new_dfname):
    tbl = orca.get_table(new_dfname)
    for col in tbl.columns:
        logger.info("Adding dependent column %s" % col)
        orca.add_column(base_dfname, col, tbl[col])

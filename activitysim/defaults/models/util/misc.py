# ActivitySim
# See full license in LICENSE.txt.

import orca


def add_dependent_columns(base_dfname, new_dfname):
    tbl = orca.get_table(new_dfname)
    for col in tbl.columns:
        print "Adding", col
        orca.add_column(base_dfname, col, tbl[col])

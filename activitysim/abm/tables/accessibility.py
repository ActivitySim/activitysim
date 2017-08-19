# ActivitySim
# See full license in LICENSE.txt.

import orca
import pandas as pd


@orca.table()
def accessibility(store):
    df = store["skims/accessibility"]
    # FIXME - should eventually replace when activity model is stable
    # FIXME - but will break regression tests
    # df.columns = ["%s_regress" % c.upper() for c in df.columns]
    df.columns = [c.upper() for c in df.columns]

    # replace table function with dataframe
    orca.add_table('accessibility', df)
    return df


# this would be accessibility around the household location - be careful with
# this one as accessibility at some other location can also matter
orca.broadcast('accessibility', 'households', cast_index=True, onto_on='TAZ')

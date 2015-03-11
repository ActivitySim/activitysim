import pandas as pd
import urbansim.sim.simulation as sim


@sim.table()
def tours(non_mandatory_tours, mandatory_tours):
    return pd.concat([non_mandatory_tours.to_frame(),
                      mandatory_tours.to_frame()],
                     ignore_index=True)


@sim.column("tours")
def sov_available(tours):
    # FIXME this means cars can be appear magically during the day
    return pd.Series(1, index=tours.index)

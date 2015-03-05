import pandas as pd
import urbansim.sim.simulation as sim


@sim.table(cache=True)
def accessibility(store):
    df = store["skims/accessibility"]
    df.columns = [c.upper() for c in df.columns]
    return df


@sim.column("accessibility")
def mode_choice_logsums(accessibility):
    # TODO a big todo here is to compute actual mode choice logsums from our
    # TODO upcoming mode choice model
    return pd.Series(0, accessibility.index)


# this would be accessibility around the household location - be careful with
# this one as accessibility at some other location can also matter
sim.broadcast('accessibility', 'households', cast_index=True, onto_on='TAZ')

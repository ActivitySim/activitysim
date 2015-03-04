import urbansim.sim.simulation as sim


@sim.table(cache=True)
def accessibility(store):
    df = store["skims/accessibility"]
    df.columns = [c.upper() for c in df.columns]
    return df


sim.broadcast('accessibility', 'households', cast_index=True, onto_on='TAZ')

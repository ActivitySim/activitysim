import urbansim.sim.simulation as sim


def add_dependent_columns(base_dfname, new_dfname):
    tbl = sim.get_table(new_dfname)
    for col in tbl.columns:
        print "Adding", col
        sim.add_column(base_dfname, col, tbl[col])
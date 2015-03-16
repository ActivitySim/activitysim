import pandas as pd
import urbansim.sim.simulation as sim


@sim.table()
def tours(non_mandatory_tours, mandatory_tours):
    return pd.concat([non_mandatory_tours.to_frame(),
                      mandatory_tours.to_frame()],
                     ignore_index=True)


@sim.table()
def mandatory_tours_merged(mandatory_tours, persons_merged):
    return sim.merge_tables(mandatory_tours.name,
                            [mandatory_tours, persons_merged])


@sim.table()
def non_mandatory_tours_merged(non_mandatory_tours, persons_merged):
    tours = non_mandatory_tours
    return sim.merge_tables(tours.name, tables=[tours,
                                                persons_merged])


@sim.table()
def tours_merged(tours, persons_merged):
    return sim.merge_tables(tours.name, tables=[tours,
                                                persons_merged])


# broadcast trips onto persons using the person_id
sim.broadcast('persons', 'non_mandatory_tours',
              cast_index=True, onto_on='person_id')
sim.broadcast('persons_merged', 'non_mandatory_tours',
              cast_index=True, onto_on='person_id')
sim.broadcast('persons_merged', 'tours', cast_index=True, onto_on='person_id')


@sim.column("tours")
def sov_available(tours):
    # FIXME this means cars can be appear magically during the day
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def is_joint(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@sim.column("tours")
def work_tour_is_da(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@sim.column("tours")
def terminal_time(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@sim.column("tours")
def out_period(tours):
    # FIXME these are time periods that should come from the scheduling process
    return pd.Series("am", index=tours.index)


@sim.column("tours")
def in_period(tours):
    # FIXME these are time periods that should come from the scheduling process
    return pd.Series("am", index=tours.index)

import pandas as pd
import urbansim.sim.simulation as sim


@sim.table()
def tours(non_mandatory_tours, mandatory_tours, tdd_alts):
    tours = pd.concat([non_mandatory_tours.to_frame(),
                      mandatory_tours.to_frame()],
                      ignore_index=True)
    # go ahead here and add the start, end, and duration here for future use
    chosen_tours = tdd_alts.to_frame().loc[tours.tour_departure_and_duration]
    chosen_tours.index = tours.index
    return pd.concat([tours, chosen_tours], axis=1)


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
    # FIXME
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def hov2_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def hov2toll_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def hov3_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def sovtoll_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def is_joint(tours):
    # FIXME
    return pd.Series(False, index=tours.index)


@sim.column("tours")
def number_of_participants(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@sim.column("tours")
def work_tour_is_drive(tours):
    # FIXME
    # FIXME note that there's something about whether this is a subtour?
    # FIXME though I'm not sure how it can be a subtour in the tour mode choice
    return pd.Series(0, index=tours.index)


@sim.column("tours")
def terminal_time(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@sim.column("tours")
def daily_parking_cost(tours):
    # FIXME - this is a computation based on the tour destination
    return pd.Series(0, index=tours.index)


@sim.column("tours")
def out_period(tours, settings):
    return pd.cut(tours.end,
                  settings['time_periods']['hours'],
                  labels=settings['time_periods']['labels'])


@sim.column("tours")
def in_period(tours, settings):
    return pd.cut(tours.start,
                  settings['time_periods']['hours'],
                  labels=settings['time_periods']['labels'])

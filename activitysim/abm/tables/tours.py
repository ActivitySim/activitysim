# ActivitySim
# See full license in LICENSE.txt.

import logging

import orca
import numpy as np
import pandas as pd

from activitysim.core.util import reindex


logger = logging.getLogger(__name__)


@orca.table()
def tours(non_mandatory_tours, mandatory_tours, tdd_alts):

    non_mandatory_df = non_mandatory_tours.local
    mandatory_df = mandatory_tours.local

    # don't expect indexes to overlap
    assert len(non_mandatory_df.index.intersection(mandatory_df.index)) == 0

    # expect non-overlapping indexes (so the tripids dont change)
    assert len(np.intersect1d(non_mandatory_df.index, mandatory_df.index, assume_unique=True)) == 0

    tours = pd.concat([non_mandatory_tours.to_frame(),
                      mandatory_tours.to_frame()],
                      ignore_index=False)

    # go ahead here and add the start, end, and duration here for future use
    chosen_tours = tdd_alts.to_frame().loc[tours.tour_departure_and_duration]
    chosen_tours.index = tours.index

    df = pd.concat([tours, chosen_tours], axis=1)
    assert df.index.name == 'tour_id'

    # replace table function with dataframe
    orca.add_table('tours', df)

    return df


@orca.table()
def mandatory_tours_merged(mandatory_tours, persons_merged):
    return orca.merge_tables(mandatory_tours.name,
                             [mandatory_tours, persons_merged])


@orca.table()
def non_mandatory_tours_merged(non_mandatory_tours, persons_merged):
    tours = non_mandatory_tours
    return orca.merge_tables(tours.name, tables=[
        tours, persons_merged])


@orca.table()
def tours_merged(tours, persons_merged):
    return orca.merge_tables(tours.name, tables=[
        tours, persons_merged])


# broadcast trips onto persons using the person_id
orca.broadcast('persons', 'non_mandatory_tours',
               cast_index=True, onto_on='person_id')
orca.broadcast('persons_merged', 'non_mandatory_tours',
               cast_index=True, onto_on='person_id')
orca.broadcast('persons_merged', 'tours', cast_index=True, onto_on='person_id')


@orca.column("tours")
def sov_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def hov2_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def hov2toll_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def hov3_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def sovtoll_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def drive_local_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def drive_lrf_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def drive_express_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def drive_heavyrail_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def drive_commuter_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def walk_local_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def walk_lrf_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def walk_commuter_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def walk_express_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def walk_heavyrail_available(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def is_joint(tours):
    # FIXME
    return pd.Series(False, index=tours.index)


@orca.column("tours")
def is_not_joint(tours):
    # FIXME
    return pd.Series(True, index=tours.index)


@orca.column("tours")
def number_of_participants(tours):
    # FIXME
    return pd.Series(1, index=tours.index)


@orca.column("tours")
def work_tour_is_drive(tours):
    # FIXME
    # FIXME note that there's something about whether this is a subtour?
    # FIXME though I'm not sure how it can be a subtour in the tour mode choice
    return pd.Series(0, index=tours.index)


@orca.column("tours")
def terminal_time(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@orca.column("tours")
def origin_walk_time(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@orca.column("tours")
def destination_walk_time(tours):
    # FIXME
    return pd.Series(0, index=tours.index)


@orca.column("tours")
def daily_parking_cost(tours):
    # FIXME - this is a computation based on the tour destination
    return pd.Series(0, index=tours.index)


@orca.column("tours")
def dest_density_index(tours, land_use):
    return reindex(land_use.density_index,
                   tours.destination)


@orca.column("tours")
def dest_topology(tours, land_use):
    return reindex(land_use.TOPOLOGY,
                   tours.destination)


@orca.column("tours")
def out_period(tours, settings):
    cats = pd.cut(tours.end,
                  settings['time_periods']['hours'],
                  labels=settings['time_periods']['labels'])
    # cut returns labelled categories but we convert to str
    return cats.astype(str)


@orca.column("tours")
def in_period(tours, settings):
    cats = pd.cut(tours.start,
                  settings['time_periods']['hours'],
                  labels=settings['time_periods']['labels'])
    # cut returns labelled categories but we convert to str
    return cats.astype(str)

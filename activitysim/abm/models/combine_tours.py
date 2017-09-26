# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import numpy as np

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject

logger = logging.getLogger(__name__)

@inject.step()
def combine_tours(non_mandatory_tours,
                  mandatory_tours,
                  tdd_alts,
                  chunk_size,
                  trace_hh_id):

    print "\nmandatory_tours\n", mandatory_tours.to_frame()

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
    inject.add_table('tours', df)

# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import numpy as np

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import pipeline

logger = logging.getLogger(__name__)


@inject.step()
def annotate_tours(tours,
                   tdd_alts,
                   chunk_size,
                   trace_hh_id):

    print "\ntours\n", tours.to_frame()

    tours = tours.to_frame()

    # go ahead here and add the start, end, and duration here for future use
    chosen_tours = tdd_alts.to_frame().loc[tours.tour_departure_and_duration]
    chosen_tours.index = tours.index

    df = pd.concat([tours, chosen_tours], axis=1)
    assert df.index.name == 'tour_id'

    pipeline.replace_table('tours', df)

    pipeline.add_dependent_columns("tours", "tours_extras")

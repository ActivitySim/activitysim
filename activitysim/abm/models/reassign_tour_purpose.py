# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
import numpy as np

from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline

logger = logging.getLogger(__name__)


@inject.step()
def reassign_tour_purpose_by_poe(
        tours,
        chunk_size,
        trace_hh_id):

    trace_label = 'reassign_tour_purpose_by_poe'
    probs_df = pd.read_csv(config.config_file_path('tour_purpose_probs_by_poe.csv'))
    probs_df.columns = [col if col in ['Purpose', 'Description'] else int(col) for col in probs_df.columns]

    tours_df = tours.to_frame(columns=['tour_type','origin'])
    tour_types = probs_df[['Purpose','Description']].set_index('Purpose')['Description']

    tours_df['purpose_id'] = None
    for poe, group in tours_df.groupby('origin'):
        num_tours = len(group)
        purpose_probs = probs_df[poe]
        purpose_cum_probs = purpose_probs.values.cumsum()
        purpose_scaled_probs = np.subtract(purpose_cum_probs, np.random.rand(num_tours, 1))
        purpose = np.argmax((purpose_scaled_probs + 1.0).astype('i4'), axis=1)
        tours_df.loc[group.index, 'purpose_id'] = purpose
    tours_df['new_tour_type'] = tours_df['purpose_id'].map(tour_types)    
        
    # # for debugging
    
    # purp_pcts = tours_df.groupby(['origin', 'new_tour_type']).count().reset_index(level='new_tour_type').merge(
    #     tours_df.groupby('origin').count().rename(
    #         columns={'purpose_id':'origin_total'})[['origin_total']], left_index=True, right_index=True)
    # purp_pcts['pct'] = purp_pcts['purpose_id'] / purp_pcts['origin_total']
    
    tours = tours.to_frame()
    tours['tour_type'] = tours_df['new_tour_type'].reindex(tours.index)
    tours['purpose_id'] = tours_df['purpose_id'].reindex(tours.index)
    tours['tour_category'] = 'non_mandatory'
    tours.loc[tours['tour_type'].isin(['home','work']), 'tour_category'] = 'mandatory'

    pipeline.replace_table("tours", tours)

    return

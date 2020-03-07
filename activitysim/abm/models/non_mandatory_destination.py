# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core.interaction_simulate import interaction_simulate

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import simulate

from activitysim.core.util import assign_in_place
from activitysim.abm.tables.size_terms import tour_destination_size_terms

from .util import tour_destination

logger = logging.getLogger(__name__)


@inject.step()
def non_mandatory_tour_destination(
        tours,
        persons_merged,
        skim_dict, skim_stack,
        chunk_size,
        trace_hh_id):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = 'non_mandatory_tour_destination'
    model_settings = config.read_model_settings('non_mandatory_tour_destination.yaml')

    logsum_column_name = model_settings.get('DEST_CHOICE_LOGSUM_COLUMN_NAME')
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get('DEST_CHOICE_SAMPLE_TABLE_NAME')
    want_sample_table = sample_table_name is not None

    tours = tours.to_frame()

    persons_merged = persons_merged.to_frame()

    # choosers are tours - in a sense tours are choosing their destination
    non_mandatory_tours = tours[tours.tour_category == 'non_mandatory']

    # - if no mandatory_tours
    if non_mandatory_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        non_mandatory_tours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        skim_dict,
        skim_stack,
        chunk_size, trace_hh_id, trace_label)

    non_mandatory_tours['destination'] = choices_df['choice']
    assign_in_place(tours, non_mandatory_tours[['destination']])

    if want_logsums:
        non_mandatory_tours[logsum_column_name] = choices_df['logsum']
        assign_in_place(tours, non_mandatory_tours[[logsum_column_name]])

    pipeline.replace_table("tours", tours)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        print(save_sample_df)
        pipeline.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(tours[tours.tour_category == 'non_mandatory'],
                         label="non_mandatory_tour_destination",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)

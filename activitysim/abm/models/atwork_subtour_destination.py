# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample
from activitysim.core.util import assign_in_place

from .util import tour_destination
from .util import estimation

logger = logging.getLogger(__name__)
DUMP = False


@inject.step()
def atwork_subtour_destination(
        tours,
        persons_merged,
        network_los,
        chunk_size, trace_hh_id):

    trace_label = 'atwork_subtour_destination'
    model_settings_file_name = 'atwork_subtour_destination.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    future_settings = {
        'SIZE_TERM_SELECTOR': 'atwork',
        'SEGMENTS': ['atwork'],
        'ORIG_ZONE_ID': 'workplace_zone_id'
    }
    model_settings = config.future_model_settings(model_settings_file_name, model_settings, future_settings)

    destination_column_name = 'destination'
    logsum_column_name = model_settings.get('DEST_CHOICE_LOGSUM_COLUMN_NAME')
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get('DEST_CHOICE_SAMPLE_TABLE_NAME')
    want_sample_table = config.setting('want_dest_choice_sample_tables') and sample_table_name is not None

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == 'atwork']

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results('atwork_subtour_destination')
        return

    estimator = estimation.manager.begin_estimation('atwork_subtour_destination')
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag='SPEC')
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(inject.get_injectable('size_terms'), 'size_terms', append=False)
        estimator.write_table(inject.get_table('land_use').to_frame(), 'landuse', append=False)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        subtours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        chunk_size, trace_hh_id, trace_label)

    if estimator:
        estimator.write_choices(choices_df['choice'])
        choices_df['choice'] = estimator.get_survey_values(choices_df['choice'], 'tours', 'destination')
        estimator.write_override_choices(choices_df['choice'])
        estimator.end_estimation()

    subtours[destination_column_name] = choices_df['choice']
    assign_in_place(tours, subtours[[destination_column_name]])

    if want_logsums:
        subtours[logsum_column_name] = choices_df['logsum']
        assign_in_place(tours, subtours[[logsum_column_name]])

    pipeline.replace_table("tours", tours)

    tracing.print_summary(destination_column_name,
                          subtours[destination_column_name],
                          describe=True)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        pipeline.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label='atwork_subtour_destination',
                         columns=['destination'])

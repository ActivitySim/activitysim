# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import simulate

from activitysim.core.util import assign_in_place

from .util import tour_od
from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def tour_od_choice(
        tours,
        persons,
        households,
        land_use,
        network_los,
        chunk_size,
        trace_hh_id):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = 'tour_od_choice'
    model_settings_file_name = 'tour_od_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)
    origin_col_name = model_settings['ORIG_COL_NAME']
    dest_col_name = model_settings['DEST_COL_NAME']

    sample_table_name = model_settings.get('OD_CHOICE_SAMPLE_TABLE_NAME')
    want_sample_table = config.setting('want_dest_choice_sample_tables') and sample_table_name is not None

    tours = tours.to_frame()

    # interaction_sample_simulate insists choosers appear in same order as alts
    tours = tours.sort_index()

    estimator = estimation.manager.begin_estimation('tour_od_choice')
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag='SPEC')
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(inject.get_injectable('size_terms'), 'size_terms', append=False)
        estimator.write_table(inject.get_table('land_use').to_frame(), 'landuse', append=False)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_od.run_tour_od(
        tours,
        persons,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        chunk_size, trace_hh_id, trace_label)
    breakpoint()

    if estimator:
        estimator.write_choices(choices_df.choice)
        choices_df.choice = estimator.get_survey_values(
            choices_df.choice, 'tours', ['origin','destination'])
        estimator.write_override_choices(choices_df.choice)
        estimator.end_estimation()

    tours[origin_col_name] = choices_df[origin_col_name]
    tours[dest_col_name] = choices_df[dest_col_name]
    tours['poe_id'] = tours[origin_col_name].map(land_use.to_frame(columns='poe_id').poe_id)

    households = households.to_frame()
    persons = persons.to_frame()
    households[origin_col_name] = tours.set_index('household_id')[origin_col_name].reindex(households.index)
    persons[origin_col_name] = households[origin_col_name].reindex(persons.household_id).values

    # downstream steps require 'home_zone_id' column, but for the xborder
    # model this field is inherited from the tour origin which is only set now
    households['home_zone_id'] = households[origin_col_name]
    persons['home_zone_id'] = persons[origin_col_name]

    pipeline.replace_table("tours", tours)  # replace runs on pandas dfs
    pipeline.replace_table("persons", persons)  # extend runs on orca df wrappers
    pipeline.replace_table("households", households)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        pipeline.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label="tours_od_choice",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)
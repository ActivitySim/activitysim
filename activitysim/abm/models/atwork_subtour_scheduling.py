# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import timetable as tt
from .util.vectorize_tour_scheduling import vectorize_subtour_scheduling
from .util.expressions import annotate_preprocessors

from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)

DUMP = False


@inject.step()
def atwork_subtour_scheduling(
        tours,
        persons_merged,
        tdd_alts,
        skim_dict,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for at work subtours tours
    """

    trace_label = 'atwork_subtour_scheduling'
    model_settings = config.read_model_settings('tour_scheduling_atwork.yaml')
    model_spec = simulate.read_model_spec(file_name='tour_scheduling_atwork.csv')

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == 'atwork']

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    logger.info("Running %s with %d tours", trace_label, len(subtours))

    # preprocessor
    constants = config.get_model_constants(model_settings)
    od_skim_wrapper = skim_dict.wrap('origin', 'destination')
    do_skim_wrapper = skim_dict.wrap('destination', 'origin')
    skims = {
        "od_skims": od_skim_wrapper,
        "do_skims": do_skim_wrapper,
    }
    annotate_preprocessors(
        subtours, constants, skims,
        model_settings, trace_label)

    # parent_tours table with columns ['tour_id', 'tdd'] index = tour_id
    parent_tour_ids = subtours.parent_tour_id.astype(int).unique()
    parent_tours = pd.DataFrame({'tour_id': parent_tour_ids}, index=parent_tour_ids)
    parent_tours = parent_tours.merge(tours[['tdd']], left_index=True, right_index=True)

    tdd_choices = vectorize_subtour_scheduling(
        parent_tours,
        subtours,
        persons_merged,
        tdd_alts, model_spec,
        model_settings,
        chunk_size=chunk_size,
        trace_label=trace_label)

    assign_in_place(tours, tdd_choices)
    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(tours[tours.tour_category == 'atwork'],
                         label="atwork_subtour_scheduling",
                         slicer='person_id',
                         index_label='tour_id',
                         columns=None)

    if DUMP:
        subtours = tours[tours.tour_category == 'atwork']
        parent_tours = tours[tours.index.isin(subtours.parent_tour_id)]

        tracing.dump_df(DUMP, subtours, trace_label, 'sub_tours')
        tracing.dump_df(DUMP, parent_tours, trace_label, 'parent_tours')

        parent_tours['parent_tour_id'] = parent_tours.index
        subtours = pd.concat([parent_tours, subtours])
        tracing.dump_df(DUMP,
                        tt.tour_map(parent_tours, subtours, tdd_alts,
                                    persons_id_col='parent_tour_id'),
                        trace_label, 'tour_map')

# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt
from activitysim.core import simulate
from activitysim.core import expressions

from .util import estimation

from .util.vectorize_tour_scheduling import vectorize_tour_scheduling
from activitysim.core.util import assign_in_place


logger = logging.getLogger(__name__)
DUMP = False


@inject.step()
def non_mandatory_tour_scheduling(tours,
                                  persons_merged,
                                  tdd_alts,
                                  chunk_size,
                                  trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for non-mandatory tours
    """

    trace_label = 'non_mandatory_tour_scheduling'
    model_settings_file_name = 'non_mandatory_tour_scheduling.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    tours = tours.to_frame()
    non_mandatory_tours = tours[tours.tour_category == 'non_mandatory']

    logger.info("Running non_mandatory_tour_scheduling with %d tours", len(tours))

    persons_merged = persons_merged.to_frame()

    if 'SIMULATE_CHOOSER_COLUMNS' in model_settings:
        persons_merged =\
            expressions.filter_chooser_columns(persons_merged,
                                               model_settings['SIMULATE_CHOOSER_COLUMNS'])

    constants = config.get_model_constants(model_settings)

    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=non_mandatory_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    timetable = inject.get_injectable("timetable")

    estimator = estimation.manager.begin_estimation('non_mandatory_tour_scheduling')

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df)
        timetable.begin_transaction(estimator)

    # - non_mandatory tour scheduling is not segmented by tour type
    spec_info = {
        'spec': model_spec,
        'estimator': estimator
    }

    choices = vectorize_tour_scheduling(
        non_mandatory_tours, persons_merged,
        tdd_alts, timetable,
        tour_segments=spec_info,
        tour_segment_col=None,
        model_settings=model_settings,
        chunk_size=chunk_size,
        trace_label=trace_label)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'tours', 'tdd')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

        # update timetable to reflect the override choices (assign tours in tour_num order)
        timetable.rollback()
        for tour_num, nth_tours in non_mandatory_tours.groupby('tour_num', sort=True):
            timetable.assign(window_row_ids=nth_tours['person_id'], tdds=choices.reindex(nth_tours.index))

    timetable.replace_table()

    # choices are tdd alternative ids
    # we want to add start, end, and duration columns to tours, which we have in tdd_alts table
    choices = pd.merge(choices.to_frame('tdd'), tdd_alts, left_on=['tdd'], right_index=True, how='left')

    assign_in_place(tours, choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    non_mandatory_tours = tours[tours.tour_category == 'non_mandatory']

    tracing.dump_df(DUMP,
                    tt.tour_map(persons_merged, non_mandatory_tours, tdd_alts),
                    trace_label, 'tour_map')

    if trace_hh_id:
        tracing.trace_df(non_mandatory_tours,
                         label="non_mandatory_tour_scheduling",
                         slicer='person_id',
                         index_label='tour_id',
                         columns=None,
                         warn_if_empty=True)

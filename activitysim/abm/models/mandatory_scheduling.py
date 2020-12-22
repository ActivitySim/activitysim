# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt
from activitysim.core import expressions

from activitysim.core.util import reindex

from .util import estimation
from .util import vectorize_tour_scheduling as vts

from activitysim.core.util import assign_in_place


logger = logging.getLogger(__name__)

DUMP = False


@inject.step()
def mandatory_tour_scheduling(tours,
                              persons_merged,
                              tdd_alts,
                              chunk_size,
                              trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for mandatory tours
    """
    trace_label = 'mandatory_tour_scheduling'
    model_settings_file_name = 'mandatory_tour_scheduling.yaml'
    estimators = {}

    model_settings = config.read_model_settings(model_settings_file_name)
    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    tours = tours.to_frame()
    mandatory_tours = tours[tours.tour_category == 'mandatory']

    # - if no mandatory_tours
    if mandatory_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    persons_merged = persons_merged.to_frame()

    # - filter chooser columns for both logsums and simulate
    logsum_columns = logsum_settings.get('LOGSUM_CHOOSER_COLUMNS', [])
    model_columns = model_settings.get('SIMULATE_CHOOSER_COLUMNS', [])
    chooser_columns = logsum_columns + [c for c in model_columns if c not in logsum_columns]
    persons_merged = expressions.filter_chooser_columns(persons_merged, chooser_columns)

    # - add tour segmentation column
    # mtctm1 segments mandatory_scheduling spec by tour_type
    # (i.e. there are different specs for work and school tour_types)
    # mtctm1 logsum coefficients are segmented by primary_purpose
    # (i.e. there are different locsum coefficents for work, school, univ primary_purposes
    # for simplicity managing these different segmentation schemes,
    # we conflate them by segmenting the skims to align with primary_purpose
    tour_segment_col = 'mandatory_tour_seg'
    assert tour_segment_col not in mandatory_tours
    is_university_tour = \
        (mandatory_tours.tour_type == 'school') & \
        reindex(persons_merged.is_university, mandatory_tours.person_id)
    mandatory_tours[tour_segment_col] = \
        mandatory_tours.tour_type.where(~is_university_tour, 'univ')

    # load specs
    spec_segment_settings = model_settings.get('SPEC_SEGMENTS', {})
    specs = {}
    estimators = {}
    for spec_segment_name, spec_settings in spec_segment_settings.items():

        # estimator for this tour_segment
        estimator = estimation.manager.begin_estimation(model_name='mandatory_tour_scheduling_%s' % spec_segment_name,
                                                        bundle_name='mandatory_tour_scheduling')

        spec_file_name = spec_settings['SPEC']
        model_spec = simulate.read_model_spec(file_name=spec_file_name)
        coefficients_df = simulate.read_model_coefficients(spec_segment_settings[spec_segment_name])
        specs[spec_segment_name] = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

        if estimator:
            estimators[spec_segment_name] = estimator  # add to local list
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(spec_settings)
            estimator.write_coefficients(coefficients_df)

    # - spec dict segmented by primary_purpose
    tour_segment_settings = model_settings.get('TOUR_SPEC_SEGMENTS', {})
    tour_segments = {}
    for tour_segment_name, spec_segment_name in tour_segment_settings.items():
        tour_segments[tour_segment_name] = {}
        tour_segments[tour_segment_name]['spec_segment_name'] = spec_segment_name
        tour_segments[tour_segment_name]['spec'] = specs[spec_segment_name]
        tour_segments[tour_segment_name]['estimator'] = estimators.get(spec_segment_name)

    timetable = inject.get_injectable("timetable")

    if estimators:
        timetable.begin_transaction(list(estimators.values()))

    logger.info("Running mandatory_tour_scheduling with %d tours", len(tours))
    choices = vts.vectorize_tour_scheduling(
        mandatory_tours, persons_merged,
        tdd_alts, timetable,
        tour_segments=tour_segments, tour_segment_col=tour_segment_col,
        model_settings=model_settings,
        chunk_size=chunk_size,
        trace_label=trace_label)

    if estimators:
        # overrride choices for all estimators
        choices_list = []
        for spec_segment_name, estimator in estimators.items():
            model_choices = choices[(mandatory_tours.tour_type == spec_segment_name)]

            # FIXME vectorize_tour_scheduling calls used to write_choices but perhaps shouldn't
            estimator.write_choices(model_choices)
            override_choices = estimator.get_survey_values(model_choices, 'tours', 'tdd')
            estimator.write_override_choices(override_choices)

            choices_list.append(override_choices)
            estimator.end_estimation()
        choices = pd.concat(choices_list)

        # update timetable to reflect the override choices (assign tours in tour_num order)
        timetable.rollback()
        for tour_num, nth_tours in tours.groupby('tour_num', sort=True):
            timetable.assign(window_row_ids=nth_tours['person_id'], tdds=choices.reindex(nth_tours.index))

    # choices are tdd alternative ids
    # we want to add start, end, and duration columns to tours, which we have in tdd_alts table
    choices = pd.merge(choices.to_frame('tdd'), tdd_alts, left_on=['tdd'], right_index=True, how='left')

    assign_in_place(tours, choices)
    pipeline.replace_table("tours", tours)

    timetable.replace_table()

    # updated df for tracing
    mandatory_tours = tours[tours.tour_category == 'mandatory']

    tracing.dump_df(DUMP,
                    tt.tour_map(persons_merged, mandatory_tours, tdd_alts),
                    trace_label, 'tour_map')

    if trace_hh_id:
        tracing.trace_df(mandatory_tours,
                         label="mandatory_tour_scheduling",
                         slicer='person_id',
                         index_label='tour',
                         columns=None,
                         warn_if_empty=True)

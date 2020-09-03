# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core.mem import force_garbage_collect
from activitysim.core.util import assign_in_place

from .util.mode import run_tour_mode_choice_simulate
from .util import estimation

logger = logging.getLogger(__name__)

"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


def write_coefficient_template(model_settings):
    coefficients = simulate.read_model_coefficients(model_settings)

    coefficients = coefficients.transpose()
    coefficients.columns.name = None

    template = coefficients.copy()

    coef_names = []
    coef_values = []

    for c in coefficients.columns:

        values = coefficients[c]
        unique_values = values.unique()

        for uv in unique_values:

            if len(unique_values) == 1:
                uv_coef_name = c + '_all'
            else:
                uv_coef_name = c + '_' + '_'.join(values[values == uv].index.values)

            coef_names.append(uv_coef_name)
            coef_values.append(uv)

            template[c] = template[c].where(values != uv, uv_coef_name)

    refactored_coefficients = pd.DataFrame({'coefficient_name': coef_names,  'value': coef_values})
    refactored_coefficients.value = refactored_coefficients.value.astype(np.float32)
    print(refactored_coefficients)

    template = template.transpose()
    template.to_csv(
        config.output_file_path('tour_mode_choice_coefficients_template.csv'),
        mode='w', index=True, header=True)

    refactored_coefficients.to_csv(
        config.output_file_path('tour_mode_choice_refactored_coefficients.csv'),
        mode='w', index=False, header=True)


@inject.step()
def tour_mode_choice_simulate(tours, persons_merged,
                              skim_dict, skim_stack,
                              chunk_size,
                              trace_hh_id):
    """
    Tour mode choice simulate
    """
    trace_label = 'tour_mode_choice'
    model_settings_file_name = 'tour_mode_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    mode_column_name = 'tour_mode'  # FIXME - should be passed in?

    primary_tours = tours.to_frame()
    assert not (primary_tours.tour_category == 'atwork').any()

    persons_merged = persons_merged.to_frame()

    constants = config.get_model_constants(model_settings)

    logger.info("Running %s with %d tours" % (trace_label, primary_tours.shape[0]))

    tracing.print_summary('tour_types',
                          primary_tours.tour_type, value_counts=True)

    primary_tours_merged = pd.merge(primary_tours, persons_merged, left_on='person_id',
                                    right_index=True, how='left', suffixes=('', '_r'))

    # setup skim keys
    orig_col_name = 'TAZ'
    dest_col_name = 'destination'
    out_time_col_name = 'start'
    in_time_col_name = 'end'
    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col_name, right_key=dest_col_name,
                                             skim_key='out_period')
    dot_skim_stack_wrapper = skim_stack.wrap(left_key=dest_col_name, right_key=orig_col_name,
                                             skim_key='in_period')
    odr_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col_name, right_key=dest_col_name,
                                             skim_key='in_period')
    dor_skim_stack_wrapper = skim_stack.wrap(left_key=dest_col_name, right_key=orig_col_name,
                                             skim_key='out_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,
        "dor_skims": dor_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name,
        'out_time_col_name': out_time_col_name,
        'in_time_col_name': in_time_col_name
    }

    estimator = estimation.manager.begin_estimation('tour_mode_choice')
    if estimator:
        estimator.write_coefficients(simulate.read_model_coefficients(model_settings))
        estimator.write_coefficients_template(simulate.read_model_coefficient_template(model_settings))
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # FIXME run_tour_mode_choice_simulate writes choosers post-annotation

    choices_list = []
    primary_tours_merged['primary_purpose'] = \
        primary_tours_merged.tour_type.where((primary_tours_merged.tour_type != 'school') |
                                             ~primary_tours_merged.is_university, 'univ')

    for primary_purpose, tours_segment in primary_tours_merged.groupby('primary_purpose'):

        logger.info("tour_mode_choice_simulate primary_purpose '%s' (%s tours)" %
                    (primary_purpose, len(tours_segment.index), ))

        # name index so tracing knows how to slice
        assert tours_segment.index.name == 'tour_id'

        choices_df = run_tour_mode_choice_simulate(
            tours_segment,
            primary_purpose, model_settings,
            mode_column_name=mode_column_name,
            logsum_column_name=logsum_column_name,
            skims=skims,
            constants=constants,
            estimator=estimator,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, primary_purpose),
            trace_choice_name='tour_mode_choice')

        tracing.print_summary('tour_mode_choice_simulate %s choices_df' % primary_purpose,
                              choices_df.tour_mode, value_counts=True)

        choices_list.append(choices_df)

        # FIXME - force garbage collection
        force_garbage_collect()

    choices_df = pd.concat(choices_list)

    if estimator:
        estimator.write_choices(choices_df.tour_mode)
        choices_df.tour_mode = estimator.get_survey_values(choices_df.tour_mode, 'tours', 'tour_mode')
        estimator.write_override_choices(choices_df.tour_mode)
        estimator.end_estimation()

    tracing.print_summary('tour_mode_choice_simulate all tour type choices',
                          choices_df.tour_mode, value_counts=True)

    # so we can trace with annotations
    assign_in_place(primary_tours, choices_df)

    # but only keep mode choice col
    all_tours = tours.to_frame()
    assign_in_place(all_tours, choices_df)

    pipeline.replace_table("tours", all_tours)

    if trace_hh_id:
        tracing.trace_df(primary_tours,
                         label=tracing.extend_trace_label(trace_label, mode_column_name),
                         slicer='tour_id',
                         index_label='tour_id',
                         warn_if_empty=True)

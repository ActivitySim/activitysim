# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt

from activitysim.core.util import reindex

from .util import expressions
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
    model_settings = config.read_model_settings('mandatory_tour_scheduling.yaml')
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

    # - add primary_purpose column
    # mtctm1 segments mandatory_scheduling spec by tour_type
    # (i.e. there are different specs for work and school tour_types)
    # mtctm1 logsum coefficients are segmented by primary_purpose
    # (i.e. there are different locsum coefficents for work, school, univ primary_purposes
    # for simplicity managing these different segmentation schemes,
    # we conflate them by segmenting the skims to align with primary_purpose
    segment_col = 'primary_purpose'
    if segment_col not in mandatory_tours:

        is_university_tour = \
            (mandatory_tours.tour_type == 'school') & \
            reindex(persons_merged.is_university, mandatory_tours.person_id)

        mandatory_tours['primary_purpose'] = \
            mandatory_tours.tour_type.where(~is_university_tour, 'univ')

    # - spec dict segmented by primary_purpose
    work_spec = simulate.read_model_spec(file_name='tour_scheduling_work.csv')
    school_spec = simulate.read_model_spec(file_name='tour_scheduling_school.csv')
    segment_specs = {
        'work': work_spec,
        'school': school_spec,
        'univ': school_spec
    }

    logger.info("Running mandatory_tour_scheduling with %d tours", len(tours))
    tdd_choices, timetable = vts.vectorize_tour_scheduling(
        mandatory_tours, persons_merged,
        tdd_alts,
        spec=segment_specs, segment_col=segment_col,
        model_settings=model_settings,
        chunk_size=chunk_size,
        trace_label=trace_label)

    timetable.replace_table()

    assign_in_place(tours, tdd_choices)
    pipeline.replace_table("tours", tours)

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

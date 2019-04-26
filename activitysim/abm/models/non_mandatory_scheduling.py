# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt
from activitysim.core import simulate

from .util import expressions
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
    model_settings = config.read_model_settings('non_mandatory_tour_scheduling.yaml')

    model_spec = simulate.read_model_spec(file_name='tour_scheduling_nonmandatory.csv')
    segment_col = None  # no segmentation of model_spec

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

    tdd_choices, timetable = vectorize_tour_scheduling(
        non_mandatory_tours, persons_merged,
        tdd_alts, model_spec, segment_col,
        model_settings=model_settings,
        chunk_size=chunk_size,
        trace_label=trace_label)

    timetable.replace_table()

    assign_in_place(tours, tdd_choices)
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

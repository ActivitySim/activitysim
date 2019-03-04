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

from .util import expressions
from .util.vectorize_tour_scheduling import vectorize_joint_tour_scheduling
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)


@inject.step()
def joint_tour_scheduling(
        tours,
        persons_merged,
        tdd_alts,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the departure time and duration of each joint tour
    """
    trace_label = 'joint_tour_scheduling'
    model_settings = config.read_model_settings('joint_tour_scheduling.yaml')
    model_spec = simulate.read_model_spec(file_name='tour_scheduling_joint.csv')

    tours = tours.to_frame()
    joint_tours = tours[tours.tour_category == 'joint']

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    # use inject.get_table as this won't exist if there are no joint_tours
    joint_tour_participants = inject.get_table('joint_tour_participants').to_frame()

    persons_merged = persons_merged.to_frame()

    logger.info("Running %s with %d joint tours", trace_label, joint_tours.shape[0])

    # it may seem peculiar that we are concerned with persons rather than households
    # but every joint tour is (somewhat arbitrarily) assigned a "primary person"
    # some of whose characteristics are used in the spec
    # and we get household attributes along with person attributes in persons_merged
    persons_merged = persons_merged[persons_merged.num_hh_joint_tours > 0]

    # since a households joint tours each potentially different participants
    # they may also have different joint tour masks (free time of all participants)
    # so we have to either chunk processing by joint_tour_num and build timetable by household
    # or build timetables by unique joint_tour

    constants = config.get_model_constants(model_settings)

    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=joint_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    tdd_choices, timetable = vectorize_joint_tour_scheduling(
        joint_tours, joint_tour_participants,
        persons_merged,
        tdd_alts,
        spec=model_spec,
        model_settings=model_settings,
        chunk_size=chunk_size,
        trace_label=trace_label)

    timetable.replace_table()

    assign_in_place(tours, tdd_choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    joint_tours = tours[tours.tour_category == 'joint']

    if trace_hh_id:
        tracing.trace_df(joint_tours,
                         label="joint_tour_scheduling",
                         slicer='household_id')

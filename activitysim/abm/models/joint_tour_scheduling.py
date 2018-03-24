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

from .util import expressions
from .util.vectorize_tour_scheduling import vectorize_joint_tour_scheduling

logger = logging.getLogger(__name__)


@inject.injectable()
def tdd_joint_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'tour_departure_and_duration_joint.csv')


@inject.injectable()
def joint_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'joint_tour_scheduling.yaml')


@inject.step()
def joint_tour_scheduling(
        joint_tours, joint_tour_participants,
        persons_merged,
        tdd_alts,
        tdd_joint_spec,
        joint_tour_scheduling_settings,
        configs_dir,
        chunk_size,
        trace_hh_id):
    """
     This model predicts the frequency of making mandatory trips (see the
     alternatives above) - these trips include work and school in some combination.
     """
    trace_label = 'joint_tour_scheduling'

    joint_tours_df = joint_tours.to_frame()
    joint_tour_participants = joint_tour_participants.to_frame()
    persons_merged = persons_merged.to_frame()

    logger.info("Running joint_tour_scheduling with %d joint tours" % joint_tours_df.shape[0])

    # it may seem peculiar that we are concerned with persons rather than households
    # but every joint tour is (somewhat arbitrarily) assigned a "primary person"
    # some of whose characteristics are used in the spec
    # and we get household attributes along with person attributes in persons_merged
    persons_merged = persons_merged[persons_merged.num_hh_joint_tours > 0]

    # drop the household_id column from persons as it also occurs in joint_tours_df
    # and we don't want it to get re-named when joint_tours_df and persons_merged are joined
    del persons_merged['household_id']

    # since a households joint tours each potentially different participants
    # they may also have different joint tour masks (free time of all participants)
    # so we have to either chunk processing by joint_tour_num and build timetable by household
    # or build timetables by unique joint_tour_id

    constants = config.get_model_constants(joint_tour_scheduling_settings)

    # - run preprocessor to annotate choosers
    preprocessor_settings = joint_tour_scheduling_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=joint_tours_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    tdd_choices = vectorize_joint_tour_scheduling(
        joint_tours_df, joint_tour_participants,
        persons_merged,
        tdd_alts,
        spec=tdd_joint_spec,
        constants=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    if preprocessor_settings:
        # if we annotated joint_tours, then we want a fresh copy before replace_table
        joint_tours_df = joint_tours.to_frame()

    # add tdd_choices columns to tours
    for c in tdd_choices.columns:
        joint_tours_df.loc[tdd_choices.index, c] = tdd_choices[c]

    pipeline.replace_table("joint_tours", joint_tours_df)

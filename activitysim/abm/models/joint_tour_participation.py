# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util import expressions
from activitysim.core.util import reindex
from .util.overlap import person_time_window_overlap

from ..tables import constants as ccc

logger = logging.getLogger(__name__)


@inject.injectable()
def joint_tour_participation_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'joint_tour_participation.csv')


@inject.injectable()
def joint_tour_participation_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'joint_tour_participation.yaml')


def joint_tour_participation_candidates(joint_tours, persons_merged):

    # - only interested in persons from households with joint_tours
    persons_merged = persons_merged[persons_merged.num_hh_joint_tours > 0]

    # person_cols = ['household_id', 'PNUM', 'ptype', 'adult', 'travel_active']
    # household_cols = ['num_hh_joint_tours', 'home_is_urban', 'home_is_rural',
    #                   'car_sufficiency', 'income_in_thousands',
    #                   'num_adults', 'num_children', 'num_travel_active',
    #                   'num_travel_active_adults', 'num_travel_active_children']
    # persons_merged = persons_merged[person_cols + household_cols]

    # - create candidates table
    candidates = pd.merge(
        joint_tours.reset_index(),
        persons_merged.reset_index().rename(columns={persons_merged.index.name: 'person_id'}),
        left_on=['household_id'], right_on=['household_id'])

    # should have all joint_tours
    assert len(candidates['joint_tour_id'].unique()) == joint_tours.shape[0]

    # - filter out ineligible candidates (adults for children-only tours, and vice-versa)
    eligible = ~(
        ((candidates.composition == 'adults') & ~candidates.adult) |
        ((candidates.composition == 'children') & candidates.adult)
    )
    candidates = candidates[eligible]

    # - stable (predictable) index
    MAX_PNUM = 100
    if candidates.PNUM.max() > MAX_PNUM:
        # if this happens, channel random seeds will overlap at MAX_PNUM (not probably a big deal)
        logger.warn("max persons.PNUM (%s) > MAX_PNUM (%s)" % (candidates.PNUM.max(), MAX_PNUM))

    candidates['participant_id'] = (candidates[joint_tours.index.name] * MAX_PNUM) + candidates.PNUM
    candidates.set_index('participant_id', drop=True, inplace=True, verify_integrity=True)

    return candidates


def get_tour_satisfaction(candidates):

    joint_tour_ids = candidates.joint_tour_id.unique()

    if candidates.participate.any():

        candidates = candidates[candidates.participate]

        # if this happens, we would need to filter them out!
        assert not ((candidates.composition == 'adults') & ~candidates.adult).any()
        assert not ((candidates.composition == 'children') & candidates.adult).any()

        cols = ['joint_tour_id', 'composition', 'adult']

        # tour satisfaction
        x = candidates[cols].groupby(['joint_tour_id', 'composition']).adult.agg(['size', 'sum']).\
            reset_index('composition').rename(columns={'size': 'participants', 'sum': 'adults'})

        satisfaction = (x.composition != 'mixed') & (x.participants > 1) | \
                       (x.composition == 'mixed') & (x.adults > 0) & (x.participants > x.adults)

        satisfaction = satisfaction.reindex(joint_tour_ids).fillna(False).astype(bool)

    else:
        satisfaction = pd.Series([])

    # ensure we return a result for every joint tour, even if no participants
    satisfaction = satisfaction.reindex(joint_tour_ids).fillna(False).astype(bool)

    return satisfaction


@inject.step()
def joint_tour_participation(
        joint_tours, persons, persons_merged,
        joint_tour_participation_spec,
        joint_tour_participation_settings,
        configs_dir,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """
    trace_label = 'joint_tour_participation'

    joint_tours = joint_tours.to_frame()
    persons_merged = persons_merged.to_frame()

    # - create joint_tour_participation_candidates table
    candidates = joint_tour_participation_candidates(joint_tours, persons_merged)
    tracing.register_traceable_table('participants', candidates)
    pipeline.get_rn_generator().add_channel(candidates, 'joint_tour_participants')

    logger.info("Running joint_tours_participation with %d potential participants (candidates)" %
                candidates.shape[0])

    # - preprocessor
    preprocessor_settings = joint_tour_participation_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_dict = {
            'person_time_window_overlap': person_time_window_overlap,
            'persons': persons_merged
        }

        expressions.assign_columns(
            df=candidates,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    # - simple_simulate

    nest_spec = config.get_logit_model_settings(joint_tour_participation_settings)
    constants = config.get_model_constants(joint_tour_participation_settings)

    participants_list = []
    iter = 0
    MAX_ITERATIONS = 20
    while candidates.shape[0] > 0:

        iter += 1

        logger.info('iteration %s : %s tours (%s candidates).' %
                    (iter, len(candidates.joint_tour_id.unique()), candidates.shape[0]))

        if iter > MAX_ITERATIONS:
            logger.warn('giving up on remaining tours after %s iterations' % iter)
            tracing.trace_df(candidates,
                             label="joint_tour_participation.unsatisfied",
                             slicer='NONE', transpose=False,
                             warn_if_empty=True)
            assert False
            break

        choices = simulate.simple_simulate(
            candidates,
            spec=joint_tour_participation_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, 'iter_%s' % iter),
            trace_choice_name='participation')

        # choice is boolean (participate or not)
        choice_col = joint_tour_participation_settings.get('participation_choice', 'participate')
        assert choice_col in joint_tour_participation_spec.columns, \
            "couldn't find participation choice column '%s' in spec"
        PARTICIPATE_CHOICE = joint_tour_participation_spec.columns.get_loc(choice_col)
        choices = (choices.values == PARTICIPATE_CHOICE)

        candidates['participate'] = choices

        # satisfaction indexed by joint_tour_id
        tour_satisfaction = get_tour_satisfaction(candidates)
        num_tours_satisfied = tour_satisfaction.sum()

        if num_tours_satisfied > 0:

            candidates['satisfied'] = reindex(tour_satisfaction, candidates.joint_tour_id)

            PARTICIPANT_COLS = ['joint_tour_id', 'household_id', 'person_id']
            participant = candidates.satisfied & candidates.participate
            participants = candidates[participant][PARTICIPANT_COLS].copy()
            participants_list.append(participants)

            # remove candidates of satisfied tours
            candidates = candidates[~candidates.satisfied]

        num_tours_satisfied = tour_satisfaction.sum()
        logger.info('iteration %s : %s joint tours satisfied.' % (iter, num_tours_satisfied,))

    participants = pd.concat(participants_list)
    pipeline.replace_table("joint_tour_participants", participants)

    # FIXME drop channel if we aren't using any more?
    # pipeline.get_rn_generator().drop_channel('joint_tours_participants')

    # - annotate persons table
    persons = persons.to_frame()
    expressions.assign_columns(
        df=persons,
        model_settings=joint_tour_participation_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))
    pipeline.replace_table("persons", persons)

    # FIXME - shold annotate joint_tours?
    joint_tours['person_id'] = participants.groupby('joint_tour_id').person_id.min()
    joint_tours['number_of_participants'] = participants.groupby('joint_tour_id').size()
    joint_tours['is_joint'] = True
    pipeline.replace_table("joint_tours", joint_tours)

    if trace_hh_id:
        tracing.trace_df(inject.get_table('participants_merged').to_frame(),
                         label="joint_tour_participation.participants_merged",
                         warn_if_empty=True)

        tracing.trace_df(persons,
                         label="joint_tour_participation.persons",
                         slicer='household_id',
                         warn_if_empty=True)

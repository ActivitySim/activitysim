# ActivitySim
# See full license in LICENSE.txt.

import os
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import simulate

from activitysim.core.util import reindex
from activitysim.core.util import assign_in_place

from .util import logsums as logsum
from .util.tour_destination import tour_destination_size_terms


logger = logging.getLogger(__name__)


# list of tour_types in order that doesn't change across the 3 joint_tour_destination models
# so that choices remain aligned with chooser rows.
# save space by using int8 not str to identify school type in sample df
TOUR_TYPE_ID = OrderedDict([
    # ('escort', 1),
    ('shopping', 2),
    ('othmaint', 3),
    ('othdiscr', 4),
    ('eatout', 5),
    ('social', 6)
])


@inject.step()
def joint_tour_destination_sample(
        tours,
        households_merged,
        skim_dict,
        land_use, size_terms,
        chunk_size, trace_hh_id):
    """
    Chooses a sample of destinations from all possible tour destinations by choosing
    <sample_size> times from among destination alternatives.
    Since choice is with replacement, the number of sampled alternative may be smaller
    than <sample_size>, and the pick_count column indicates hom many times the sampled
    alternative was chosen.

    Household_id column is added for convenience of merging with households when the
    joint_tour_destination_simulate choice model is run subsequently.

    adds 'joint_tour_destination_sample' table to pipeline

    +------------+-------------+-----------+-------------+-------------+-------------+
    | tour_id    |  alt_dest   |   prob    |  pick_count | tour_type_id| household_id|
    +============+=============+===========+=============+=============+=============+
    | 1605124    +          14 + 0.043873  +         1   +          3  +     160512  |
    +------------+-------------+-----------+-------------+-------------+-------------+
    | 1605124    +          18 + 0.034979  +         2   +          3  +     160512  |
    +------------+-------------+-----------+-------------+-------------+-------------+
    | 1605124    +          16 + 0.105658  +         9   +          3  +     160512  |
    +------------+-------------+-----------+-------------+-------------+-------------+
    | 1605124    +          17 + 0.057670  +         1   +          3  +     160512  |
    +------------+-------------+-----------+-------------+-------------+-------------+


    Parameters
    ----------
    tours: pipeline table
    households_merged : pipeline table
        injected merge table created on the fly
    skim_dict
    joint_tour_destination_sample_spec
    land_use
    size_terms
    chunk_size
    trace_hh_id

    Returns
    -------

    none

    """

    trace_label = 'joint_tour_destination_sample'
    model_settings = config.read_model_settings('joint_tour_destination.yaml')
    model_spec = simulate.read_model_spec(file_name='non_mandatory_tour_destination_sample.csv')

    joint_tours = tours.to_frame()
    joint_tours = joint_tours[joint_tours.tour_category == 'joint']

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    households_merged = households_merged.to_frame()

    # same size terms as non_mandatory
    size_terms = tour_destination_size_terms(land_use, size_terms, 'non_mandatory')

    # choosers are tours - in a sense tours are choosing their destination
    choosers = pd.merge(joint_tours, households_merged,
                        left_on='household_id', right_index=True, how='left')
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    sample_size = model_settings["SAMPLE_SIZE"]

    # specify name interaction_sample should give the alternative column (logsums needs to know it)
    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # interaction_dataset adds '_r' suffix to duplicate columns,
    # so TAZ column from households is TAZ and TAZ column from alternatives becomes TAZ_r
    skims = skim_dict.wrap("TAZ", "TAZ_r")

    locals_d = {
        'skims': skims
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    logger.info("Running joint_tour_destination_sample with %d joint_tours" % len(choosers))

    choices_list = []
    # segment by trip type and pick the right spec for each person type
    # for tour_type, choosers_segment in choosers.groupby('tour_type'):
    for tour_type, tour_type_id in TOUR_TYPE_ID.iteritems():

        locals_d['segment'] = tour_type

        choosers_segment = choosers[choosers.tour_type == tour_type]

        if choosers_segment.shape[0] == 0:
            logger.info("%s skipping tour_type %s: no tours" % (trace_label, tour_type))
            continue

        # alts indexed by taz with one column containing size_term for  this tour_type
        alternatives_segment = size_terms[[tour_type]]

        # FIXME - no point in considering impossible alternatives (where dest size term is zero)
        alternatives_segment = alternatives_segment[alternatives_segment[tour_type] > 0]

        logger.info("Running segment '%s' of %d joint_tours %d alternatives" %
                    (tour_type, len(choosers_segment), len(alternatives_segment)))

        if len(choosers_segment.index) > 0:
            # want named index so tracing knows how to slice
            assert choosers_segment.index.name == 'tour_id'

            choices = interaction_sample(
                choosers_segment,
                alternatives_segment,
                sample_size=sample_size,
                alt_col_name=alt_dest_col_name,
                spec=model_spec[[tour_type]],
                skims=skims,
                locals_d=locals_d,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(trace_label, tour_type))

            choices['tour_type_id'] = tour_type_id

            choices_list.append(choices)

    choices = pd.concat(choices_list)

    # - # NARROW
    choices['tour_type_id'] = choices['tour_type_id'].astype(np.uint8)

    inject.add_table('joint_tour_destination_sample', choices)

    if trace_hh_id:
        tracing.trace_df(choices,
                         label="joint_tour_destination_sample",
                         transpose=True)


@inject.step()
def joint_tour_destination_logsums(
        tours,
        persons_merged,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id):

    """
    add logsum column to existing joint_tour_destination_sample table

    logsum is calculated by computing the mode_choice model utilities for each
    sampled (joint_tour, dest_taz) destination alternative in joint_tour_destination_sample,
    and computing the logsum of all the utilities for each destination.
    """

    trace_label = 'joint_tour_destination_logsums'

    # use inject.get_table as this won't exist if there are no joint_tours
    destination_sample = inject.get_table('joint_tour_destination_sample', default=None)
    if destination_sample is None:
        tracing.no_results(trace_label)
        return

    destination_sample = destination_sample.to_frame()

    model_settings = config.read_model_settings('joint_tour_destination.yaml')
    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    joint_tours = tours.to_frame()
    joint_tours = joint_tours[joint_tours.tour_category == 'joint']

    persons_merged = persons_merged.to_frame()
    joint_tours_merged = pd.merge(joint_tours, persons_merged,
                                  left_on='person_id', right_index=True, how='left')

    # - only include columns actually used in spec
    joint_tours_merged = \
        logsum.filter_chooser_columns(joint_tours_merged, logsum_settings, model_settings)

    logsums_list = []
    for tour_type, tour_type_id in TOUR_TYPE_ID.iteritems():

        choosers = destination_sample[destination_sample['tour_type_id'] == tour_type_id]

        if choosers.shape[0] == 0:
            logger.info("%s skipping tour_type %s: no tours" % (trace_label, tour_type))
            continue

        # sample is sorted by TOUR_TYPE_ID, tour_id
        # merge order is stable only because left join on ordered index
        assert choosers.index.is_monotonic_increasing
        choosers = pd.merge(
            choosers,
            joint_tours_merged,
            left_index=True,
            right_index=True,
            how="left",
            sort=False)

        logger.info("%s running %s with %s rows" % (trace_label, tour_type, len(choosers)))

        tour_purpose = tour_type
        logsums = logsum.compute_logsums(
            choosers,
            tour_purpose,
            logsum_settings, model_settings,
            skim_dict, skim_stack,
            chunk_size, trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, tour_type))

        logsums_list.append(logsums)

    logsums = pd.concat(logsums_list)

    destination_sample['mode_choice_logsum'] = logsums
    pipeline.replace_table("joint_tour_destination_sample", destination_sample)

    if trace_hh_id:
        tracing.trace_df(destination_sample,
                         label="joint_tour_destination_logsums")


@inject.step()
def joint_tour_destination_simulate(
        tours,
        households_merged,
        skim_dict,
        land_use, size_terms,
        chunk_size, trace_hh_id):
    """
    choose a joint tour destination from amont the destination sample alternatives
    (annotated with logsums) and add destination TAZ column to joint_tours table
    """

    trace_label = 'joint_tour_destination_simulate'

    # use inject.get_table as this won't exist if there are no joint_tours
    destination_sample = inject.get_table('joint_tour_destination_sample', default=None)
    if destination_sample is None:
        tracing.no_results(trace_label)
        return

    model_settings = config.read_model_settings('joint_tour_destination.yaml')

    # - tour types are subset of non_mandatory tour types and use same expressions
    model_spec = simulate.read_model_spec(file_name='non_mandatory_tour_destination.csv')

    destination_sample = destination_sample.to_frame()
    tours = tours.to_frame()
    joint_tours = tours[tours.tour_category == 'joint']

    # interaction_sample_simulate insists choosers appear in same order as alts
    joint_tours = joint_tours.sort_index()

    households_merged = households_merged.to_frame()

    destination_size_terms = tour_destination_size_terms(land_use, size_terms, 'non_mandatory')

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running joint_tour_destination_simulate with %d joint_tours" %
                joint_tours.shape[0])

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", alt_dest_col_name)

    locals_d = {
        'skims': skims,
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    # choosers are tours - in a sense tours are choosing their destination
    choosers = pd.merge(joint_tours, households_merged,
                        left_on='household_id', right_index=True, how='left')
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    choices_list = []
    # segment by trip type and pick the right spec for each person type
    # for tour_type, choosers_segment in choosers.groupby('tour_type'):
    for tour_type, tour_type_id in TOUR_TYPE_ID.iteritems():

        locals_d['segment'] = tour_type

        choosers_segment = choosers[choosers.tour_type == tour_type]

        # - skip empty segments
        if choosers_segment.shape[0] == 0:
            logger.info("%s skipping tour_type %s: no tours" % (trace_label, tour_type))
            continue

        alts_segment = destination_sample[destination_sample.tour_type_id == tour_type_id]

        assert tour_type not in alts_segment

        # alternatives are pre-sampled and annotated with logsums and pick_count
        # but we have to merge size_terms column into alt sample list
        alts_segment[tour_type] = \
            reindex(destination_size_terms[tour_type], alts_segment[alt_dest_col_name])

        logger.info("Running segment '%s' of %d joint_tours %d alternatives" %
                    (tour_type, len(choosers_segment), len(alts_segment)))

        assert choosers_segment.index.is_monotonic_increasing
        assert alts_segment.index.is_monotonic_increasing

        choices = interaction_sample_simulate(
            choosers_segment,
            alts_segment,
            spec=model_spec[[tour_type]],
            choice_column=alt_dest_col_name,
            skims=skims,
            locals_d=locals_d,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name='joint_tour_destination')

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    # add column as we want joint_tours table for tracing.
    joint_tours['destination'] = choices

    assign_in_place(tours, joint_tours[['destination']])
    pipeline.replace_table("tours", tours)

    # drop bulky joint_tour_destination_sample table as we don't use it further
    pipeline.drop_table('joint_tour_destination_sample')

    tracing.print_summary('destination', joint_tours.destination, describe=True)

    if trace_hh_id:
        tracing.trace_df(joint_tours,
                         label="joint_tour_destination.joint_tours")

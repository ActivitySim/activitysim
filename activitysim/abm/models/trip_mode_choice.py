# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import yaml

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core.util import force_garbage_collect
from activitysim.core.util import assign_in_place

from .util.mode import _mode_choice_spec
from .util.mode import get_segment_and_unstack
from .util.mode import mode_choice_simulate
from .util.mode import annotate_preprocessors

logger = logging.getLogger(__name__)

"""
Trip mode choice is run for all trips to determine the transportation mode that
will be used for the trip
"""


@inject.injectable()
def trip_mode_choice_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'trip_mode_choice.yaml')


@inject.injectable()
def trip_mode_choice_spec_df(configs_dir):
    return simulate.read_model_spec(configs_dir, 'trip_mode_choice.csv')


@inject.injectable()
def trip_mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir, 'trip_mode_choice_coeffs.csv')) as f:
        return pd.read_csv(f, index_col='Expression')


@inject.injectable()
def trip_mode_choice_spec(trip_mode_choice_spec_df,
                          trip_mode_choice_coeffs,
                          trip_mode_choice_settings,
                          trace_hh_id):
    return _mode_choice_spec(trip_mode_choice_spec_df,
                             trip_mode_choice_coeffs,
                             trip_mode_choice_settings,
                             trace_spec=trace_hh_id,
                             trace_label='trip_mode_choice')


@inject.step()
def trip_mode_choice_simulate(trips_merged,
                              trip_mode_choice_spec,
                              trip_mode_choice_settings,
                              skim_dict,
                              skim_stack,
                              chunk_size,
                              trace_hh_id):
    """
    Trip mode choice simulate
    """
    trace_label = 'tour_mode_choice'

    trips = trips_merged.to_frame()

    nest_spec = config.get_logit_model_settings(trip_mode_choice_settings)
    constants = config.get_model_constants(trip_mode_choice_settings)

    logger.info("Running %s with %d trips" % (trace_label, trips.shape[0]))

    # setup skim keys
    orig_col_name = 'OTAZ'
    dest_col_name = 'DTAZ'
    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col_name, right_key=dest_col_name,
                                             skim_key='out_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
    }

    locals_dict = {
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name
    }
    locals_dict.update(constants)

    annotations = annotate_preprocessors(
        trips, locals_dict, skims,
        trip_mode_choice_settings, trace_label)

    choices_list = []

    # loop by tour_type in order to easily query the expression coefficient file
    for tour_type, segment in trips.groupby('tour_type'):

        logger.info("running %s tour_type '%s'" % (len(segment.index), tour_type, ))

        # name index so tracing knows how to slice
        assert segment.index.name == 'trip_id'

        # FIXME - check that destination is not null

        choices = mode_choice_simulate(
            segment,
            skims=skims,
            spec=get_segment_and_unstack(trip_mode_choice_spec, tour_type),
            constants=constants,
            nest_spec=nest_spec,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_type),
            trace_choice_name='trip_mode_choice')

        #   FIXME - no point in printing verbose value_counts now that we have tracing?
        tracing.print_summary('trip_mode_choice_simulate %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        force_garbage_collect()

    choices = pd.concat(choices_list)

    tracing.print_summary('trip_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    inject.add_column("trips", "trip_mode", choices)

    if trace_hh_id:

        tracing.trace_df(inject.get_table('trips').to_frame(),
                         label="trip_mode",
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)

    force_garbage_collect()

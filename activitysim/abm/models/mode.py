# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import yaml

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config

from .util.mode import _mode_choice_spec

logger = logging.getLogger(__name__)

"""
Generic functions for both tour and trip mode choice
"""


def _mode_choice_simulate(records,
                          skim_dict,
                          skim_stack,
                          odt_skim_stack_wrapper,
                          dot_skim_stack_wrapper,
                          od_skim_stack_wrapper,
                          spec,
                          constants,
                          nest_spec,
                          trace_label=None, trace_choice_name=None
                          ):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are tour/trip purposes).  Pass in the tours/trip that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    locals_d = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper
    }
    if constants is not None:
        locals_d.update(constants)

    skims = []
    if odt_skim_stack_wrapper is not None:
        skims.append(odt_skim_stack_wrapper)
    if dot_skim_stack_wrapper is not None:
        skims.append(dot_skim_stack_wrapper)
    if od_skim_stack_wrapper is not None:
        skims.append(od_skim_stack_wrapper)

    choices = asim.simple_simulate(records,
                                   spec,
                                   nest_spec,
                                   skims=skims,
                                   locals_d=locals_d,
                                   trace_label=trace_label,
                                   trace_choice_name=trace_choice_name)

    alts = spec.columns
    choices = choices.map(dict(zip(range(len(alts)), alts)))

    return choices


def get_segment_and_unstack(omnibus_spec, segment):
    """
    This does what it says.  Take the spec, get the column from the spec for
    the given segment, and unstack.  It is assumed that the last column of
    the multiindex is alternatives so when you do this unstacking,
    each alternative is in a column (which is the format this as used for the
    simple_simulate call.  The weird nuance here is the "Rowid" column -
    since many expressions are repeated (e.g. many are just "1") a Rowid
    column is necessary to identify which alternatives are actually part of
    which original row - otherwise the unstack is incorrect (i.e. the index
    is not unique)
    """
    spec = omnibus_spec[segment].unstack().reset_index(level="Rowid", drop=True).fillna(0)

    spec = spec.groupby(spec.index).sum()

    return spec


"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@orca.injectable()
def tour_mode_choice_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'tour_mode_choice.yaml')


@orca.injectable()
def tour_mode_choice_spec_df(configs_dir):
    with open(os.path.join(configs_dir, 'tour_mode_choice.csv')) as f:
        return asim.read_model_spec(f)


@orca.injectable()
def tour_mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir, 'tour_mode_choice_coeffs.csv')) as f:
        return pd.read_csv(f, index_col='Expression')


@orca.injectable()
def tour_mode_choice_spec(tour_mode_choice_spec_df,
                          tour_mode_choice_coeffs,
                          tour_mode_choice_settings):
    return _mode_choice_spec(tour_mode_choice_spec_df,
                             tour_mode_choice_coeffs,
                             tour_mode_choice_settings,
                             trace_label='tour_mode_choice')


@orca.step()
def tour_mode_choice_simulate(tours_merged,
                              tour_mode_choice_spec,
                              tour_mode_choice_settings,
                              skim_dict, skim_stack,
                              omx_file,
                              trace_hh_id):
    """
    Tour mode choice simulate
    """

    trace_label = trace_hh_id and 'tour_mode_choice'

    tours = tours_merged.to_frame()

    nest_spec = config.get_logit_model_settings(tour_mode_choice_settings)
    constants = config.get_model_constants(tour_mode_choice_settings)

    logger.info("Running tour_mode_choice_simulate with %d tours" % len(tours.index))

    tracing.print_summary('tour_mode_choice_simulate tour_type',
                          tours.tour_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(tour_mode_choice_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    # setup skim keys
    odt_skim_stack_wrapper = skim_stack.wrap(left_key='TAZ', right_key='destination',
                                             skim_key="out_period")
    dot_skim_stack_wrapper = skim_stack.wrap(left_key='destination', right_key='TAZ',
                                             skim_key="in_period")
    od_skims = skim_dict.wrap('TAZ', 'destination')

    choices_list = []

    for tour_type, segment in tours.groupby('tour_type'):

        # if tour_type != 'work':
        #     continue

        logger.info("tour_mode_choice_simulate tour_type '%s' (%s tours)" %
                    (tour_type, len(segment.index), ))

        # name index so tracing knows how to slice
        segment.index.name = 'tour_id'

        spec = get_segment_and_unstack(tour_mode_choice_spec, tour_type)

        if trace_hh_id:
            tracing.trace_df(spec, tracing.extend_trace_label(trace_label, 'spec.%s' % tour_type),
                             slicer='NONE', transpose=False)

        choices = _mode_choice_simulate(
            segment,
            skim_dict=skim_dict,
            skim_stack=skim_stack,
            odt_skim_stack_wrapper=odt_skim_stack_wrapper,
            dot_skim_stack_wrapper=dot_skim_stack_wrapper,
            od_skim_stack_wrapper=od_skims,
            spec=spec,
            constants=constants,
            nest_spec=nest_spec,
            trace_label=tracing.extend_trace_label(trace_label, tour_type),
            trace_choice_name='tour_mode_choice')

        tracing.print_summary('tour_mode_choice_simulate %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        mem = asim.memory_info()
        logger.debug('memory_info tour_type %s, %s' % (tour_type, mem))

    choices = pd.concat(choices_list)

    tracing.print_summary('tour_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    orca.add_column("tours", "mode", choices)

    if trace_hh_id:
        trace_columns = ['mode', 'person_id', 'tour_type', 'tour_num']
        tracing.trace_df(orca.get_table('tours').to_frame(),
                         label=tracing.extend_trace_label(trace_label, 'mode'),
                         slicer='tour_id',
                         index_label='tour_id',
                         columns=trace_columns,
                         warn_if_empty=True)

    # FIXME - this forces garbage collection
    asim.memory_info()


"""
Trip mode choice is run for all trips to determine the transportation mode that
will be used for the trip
"""


@orca.injectable()
def trip_mode_choice_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'trip_mode_choice.yaml')


@orca.injectable()
def trip_mode_choice_spec_df(configs_dir):
    with open(os.path.join(configs_dir, 'trip_mode_choice.csv')) as f:
        return asim.read_model_spec(f)


@orca.injectable()
def trip_mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir, 'trip_mode_choice_coeffs.csv')) as f:
        return pd.read_csv(f, index_col='Expression')


@orca.injectable()
def trip_mode_choice_spec(trip_mode_choice_spec_df,
                          trip_mode_choice_coeffs,
                          trip_mode_choice_settings):
    return _mode_choice_spec(trip_mode_choice_spec_df,
                             trip_mode_choice_coeffs,
                             trip_mode_choice_settings)


@orca.step()
def trip_mode_choice_simulate(trips_merged,
                              trip_mode_choice_spec,
                              trip_mode_choice_settings,
                              skim_dict,
                              skim_stack,
                              omx_file,
                              trace_hh_id):
    """
    Trip mode choice simulate
    """

    trips = trips_merged.to_frame()

    nest_spec = config.get_logit_model_settings(trip_mode_choice_settings)
    constants = config.get_model_constants(trip_mode_choice_settings)

    logger.info("Running trip_mode_choice_simulate with %d trips" % len(trips))

    odt_skim_stack_wrapper = skim_stack.wrap(left_key='OTAZ', right_key='DTAZ',
                                             skim_key="start_period")

    od_skims = skim_dict.wrap('OTAZ', 'DTAZ')

    choices_list = []

    # loop by tour_type in order to easily query the expression coefficient file
    for tour_type, segment in trips.groupby('tour_type'):

        logger.info("running %s tour_type '%s'" % (len(segment.index), tour_type, ))

        # name index so tracing knows how to slice
        segment.index.name = 'trip_id'

        # FIXME - check that destination is not null

        trace_label = trace_hh_id and ('trip_mode_choice_%s' % tour_type)

        choices = _mode_choice_simulate(
            segment,
            skim_dict=skim_dict,
            skim_stack=skim_stack,
            odt_skim_stack_wrapper=odt_skim_stack_wrapper,
            dot_skim_stack_wrapper=None,
            od_skim_stack_wrapper=od_skims,
            spec=get_segment_and_unstack(trip_mode_choice_spec, tour_type),
            constants=constants,
            nest_spec=nest_spec,
            trace_label=trace_label,
            trace_choice_name='trip_mode_choice')

        # FIXME - no point in printing verbose value_counts now that we have tracing?
        tracing.print_summary('trip_mode_choice_simulate %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        mem = asim.memory_info()
        logger.debug('memory_info tour_type %s, %s' % (tour_type, mem))

    choices = pd.concat(choices_list)

    tracing.print_summary('trip_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    # FIXME - is this a NOP if trips table doesn't exist
    orca.add_column("trips", "trip_mode", choices)

    if trace_hh_id:

        tracing.trace_df(orca.get_table('trips').to_frame(),
                         label="trip_mode",
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)

    # FIXME - this forces garbage collection
    asim.memory_info()

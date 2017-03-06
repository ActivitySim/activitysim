# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import yaml

from activitysim import activitysim as asim
from activitysim import tracing

from .util.mode import _mode_choice_spec

from .util.misc import read_model_settings, get_logit_model_settings, get_model_constants

logger = logging.getLogger(__name__)


"""
Mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@orca.injectable()
def tour_mode_choice_settings(configs_dir):
    return read_model_settings(configs_dir, 'tour_mode_choice.yaml')


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


@orca.injectable()
def trip_mode_choice_settings(configs_dir):
    return read_model_settings(configs_dir, 'trip_mode_choice.yaml')


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


def _mode_choice_simulate(tours,
                          skim_dict,
                          skim_stack,
                          orig_key,
                          dest_key,
                          spec,
                          constants,
                          nest_spec,
                          omx=None,
                          trace_label=None, trace_choice_name=None
                          ):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are trip purposes).  Pass in the tours that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    # FIXME - check that periods are in time_periods?

    in_skims = skim_stack.wrap(left_key=orig_key, right_key=dest_key, skim_key="in_period",
                               offset=-1, omx=omx)

    out_skims = skim_stack.wrap(left_key=dest_key, right_key=orig_key, skim_key="out_period",
                                offset=-1, omx=omx)

    # create wrapper with keys for this lookup
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap(orig_key, dest_key)

    locals_d = {
        "in_skims": in_skims,
        "out_skims": out_skims,
        "skims": skims
    }
    if constants is not None:
        locals_d.update(constants)

    choices = asim.simple_simulate(tours,
                                   spec,
                                   nest_spec,
                                   skims=[in_skims, out_skims, skims],
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


@orca.step()
def tour_mode_choice_simulate(tours_merged,
                              tour_mode_choice_spec,
                              tour_mode_choice_settings,
                              skim_dict, skim_stack,
                              omx_file,
                              trace_hh_id):

    trace_label = trace_hh_id and 'tour_mode_choice'

    logger.info("calling tours_merged")
    tours = tours_merged.to_frame()
    logger.info("back from tours_merged")

    nest_spec = get_logit_model_settings(tour_mode_choice_settings)
    constants = get_model_constants(tour_mode_choice_settings)

    if trace_hh_id:
        tracing.register_tours(tours, trace_hh_id)

    logger.info("Running tour_mode_choice_simulate with %d tours" % len(tours.index))

    tracing.print_summary('tour_mode_choice_simulate tour_type',
                          tours.tour_type, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(tour_mode_choice_spec,
                         tracing.extend_trace_label(trace_label, 'spec'),
                         slicer='NONE', transpose=False)

    choices_list = []
    for tour_type, segment in tours.groupby('tour_type'):

        # if tour_type != 'work':
        #     continue

        logger.info("running tour_type '%s'" % tour_type)

        orig_key = 'TAZ'
        dest_key = 'destination'

        # name index so tracing knows how to slice
        segment.index.name = 'tour_id'

        logger.info("tour_mode_choice_simulate running %s tour_type '%s'" %
                    (len(segment.index), tour_type, ))

        spec = get_segment_and_unstack(tour_mode_choice_spec, tour_type)

        if trace_hh_id:
            tracing.trace_df(spec, tracing.extend_trace_label(trace_label, 'spec.%s' % tour_type),
                             slicer='NONE', transpose=False)

        choices = _mode_choice_simulate(
            segment,
            skim_dict=skim_dict,
            skim_stack=skim_stack,
            orig_key=orig_key,
            dest_key=dest_key,
            spec=spec,
            constants=constants,
            nest_spec=nest_spec,
            omx=omx_file,
            trace_label=tracing.extend_trace_label(trace_label, tour_type),
            trace_choice_name='tour_mode_choice')

        tracing.print_summary('tour_mode_choice_simulate %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        mem = asim.memory_info()
        logger.info('memory_info tour_type %s, %s' % (tour_type, mem))

    choices = pd.concat(choices_list)

    tracing.print_summary('tour_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    orca.add_column("tours", "mode", choices)

    if trace_hh_id:
        trace_columns = ['mode']
        tracing.trace_df(orca.get_table('tours').to_frame(),
                         label=tracing.extend_trace_label(trace_label, 'mode'),
                         slicer='tour_id',
                         index_label='tour',
                         columns=trace_columns,
                         warn_if_empty=True)

    # FIXME - this forces garbage collection
    asim.memory_info()


@orca.step()
def trip_mode_choice_simulate(tours_merged,
                              trip_mode_choice_spec,
                              trip_mode_choice_settings,
                              skim_dict,
                              skim_stack,
                              omx_file,
                              trace_hh_id):

    # FIXME - running the trips model on tours
    logging.error('trips not implemented running the trips model on tours')

    trips = tours_merged.to_frame()

    nest_spec = get_logit_model_settings(trip_mode_choice_settings)
    constants = get_model_constants(trip_mode_choice_settings)

    logger.info("Running trip_mode_choice_simulate with %d trips" % len(trips))

    choices_list = []

    for tour_type, segment in trips.groupby('tour_type'):

        logger.info("running %s tour_type '%s'" % (len(segment.index), tour_type, ))

        orig_key = 'TAZ'
        dest_key = 'destination'

        # name index so tracing knows how to slice
        segment.index.name = 'tour_id'

        # FIXME - check that destination is not null (patch_mandatory_tour_destination not run?)

        # FIXME - no point in printing verbose dest_taz value_counts now that we have tracing?
        # tracing.print_summary('trip_mode_choice_simulate %s dest_taz' % tour_type,
        #                       segment[dest_key], value_counts=True)

        trace_label = trace_hh_id and ('trip_mode_choice_%s' % tour_type)

        choices = _mode_choice_simulate(
            segment,
            skim_dict=skim_dict,
            skim_stack=skim_stack,
            orig_key=orig_key,
            dest_key=dest_key,
            spec=get_segment_and_unstack(trip_mode_choice_spec, tour_type),
            constants=constants,
            nest_spec=nest_spec,
            omx=omx_file,
            trace_label=trace_label,
            trace_choice_name='trip_mode_choice')

        # FIXME - no point in printing verbose value_counts now that we have tracing?
        tracing.print_summary('trip_mode_choice_simulate %s choices' % tour_type,
                              choices, value_counts=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        mem = asim.memory_info()
        logger.info('memory_info tour_type %s, %s' % (tour_type, mem))

    choices = pd.concat(choices_list)

    tracing.print_summary('trip_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    # FIXME - is this a NOP if trips table doesn't exist
    orca.add_column("trips", "mode", choices)

    if trace_hh_id:

        logger.warn("can't dump trips table because it doesn't exist"
                    " - trip_mode_choice_simulate is not really implemented")
        # FIXME - commented out because trips table doesn't really exist
        # trace_columns = ['mode']
        # tracing.trace_df(orca.get_table('trips').to_frame(),
        #                  label = "mode",
        #                  slicer='tour_id',
        #                  index_label='tour_id',
        #                  columns = trace_columns,
        #                  warn_if_empty=True)

    # FIXME - this forces garbage collection
    asim.memory_info()

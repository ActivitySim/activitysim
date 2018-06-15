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

from .util.mode import expand_alternatives
from .util.mode import get_segment_and_unstack

from .util.mode import mode_choice_simulate
from .util.mode import annotate_preprocessors

from .util.expressions import skim_time_period_label

logger = logging.getLogger(__name__)


def evaluate_constants(expressions, constants):
    """
    Evaluate a list of constant expressions - each one can depend on the one before
    it.  These are usually used for the coefficients which have relationships
    to each other.  So ivt=.7 and then ivt_lr=ivt*.9.

    Parameters
    ----------
    expressions : Series
        the index are the names of the expressions which are
        used in subsequent evals - thus naming the expressions is required.
    constants : dict
        will be passed as the scope of eval - usually a separate set of
        constants are passed in here

    Returns
    -------
    d : dict

    """

    # FIXME why copy?
    d = {}
    for k, v in expressions.iteritems():
        d[k] = eval(str(v), d.copy(), constants)

    return d


def trip_mode_choice_spec(model_settings, configs_dir):

    spec = simulate.read_model_spec(configs_dir, 'trip_mode_choice.csv')
    return spec

    # set index to ['Expression', 'Alternative']
    spec = spec.set_index('Alternative', append=True)

    spec = expand_alternatives(spec)

    return spec


def trip_mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir, 'trip_mode_choice_coeffs.csv')) as f:
        return pd.read_csv(f, comment='#', index_col='Expression')


@inject.step()
def trip_mode_choice(
        trips,
        tours_merged,
        skim_dict, skim_stack,
        configs_dir, chunk_size, trace_hh_id):

    trace_label = 'trip_mode_choice'
    model_settings = config.read_model_settings(configs_dir, 'trip_mode_choice.yaml')

    spec = trip_mode_choice_spec(model_settings, configs_dir)
    omnibus_coefficients = trip_mode_choice_coeffs(configs_dir)

    trips_df = trips.to_frame()
    logger.info("Running %s with %d trips" % (trace_label, trips_df.shape[0]))

    tours_merged = tours_merged.to_frame()
    tours_merged = tours_merged[model_settings['TOURS_MERGED_CHOOSER_COLUMNS']]

    nest_spec = config.get_logit_model_settings(model_settings)

    tracing.print_summary('primary_purpose',
                          trips_df.primary_purpose, value_counts=True)

    tracing.trace_df(spec,
                     tracing.extend_trace_label(trace_label, 'spec'),
                     slicer='NONE', transpose=False)

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips_df,
        tours_merged,
        left_on='tour_id',
        right_index=True,
        how="left")
    assert trips_merged.index.equals(trips.index)

    # setup skim keys
    orig_col = 'origin'
    dest_col = 'destination'

    assert ('trip_period' not in trips_merged)
    trips_merged['trip_period'] = skim_time_period_label(trips_merged.depart)

    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col, right_key=dest_col,
                                             skim_key='trip_period')
    od_skim_stack_wrapper = skim_dict.wrap('origin', 'destination')

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
    }

    constants = config.get_model_constants(model_settings)
    constants.update({
        'ORIGIN': orig_col,
        'DESTINATION': dest_col
    })

    choices_list = []
    for primary_purpose, trips_segment in trips_merged.groupby('primary_purpose'):

        segment_trace_label = tracing.extend_trace_label(trace_label, primary_purpose)

        logger.info("trip_mode_choice tour_type '%s' (%s trips)" %
                    (primary_purpose, len(trips_segment.index), ))

        # name index so tracing knows how to slice
        assert trips_segment.index.name == 'trip_id'

        locals_dict = evaluate_constants(omnibus_coefficients[primary_purpose], constants=constants)
        locals_dict.update(constants)

        annotate_preprocessors(
            trips_segment, locals_dict, skims,
            model_settings, trace_label)

        choices = mode_choice_simulate(
            trips_segment,
            skims=skims,
            spec=spec,
            constants=locals_dict,
            nest_spec=nest_spec,
            chunk_size=chunk_size,
            trace_label=segment_trace_label,
            trace_choice_name='trip_mode_choice')

        tracing.print_summary('%s tour_modes' % primary_purpose,
                              trips_segment.tour_mode, value_counts=True)

        tracing.print_summary('trip_mode_choice %s choices' % primary_purpose,
                              choices, value_counts=True)

        if trace_hh_id:
            # trace the coefficients
            tracing.trace_df(pd.Series(locals_dict),
                             label=tracing.extend_trace_label(segment_trace_label, 'constants'),
                             transpose=False,
                             slicer='NONE')

            # so we can trace with annotations
            trips_segment['trip_mode'] = choices
            tracing.trace_df(trips_segment,
                             label=tracing.extend_trace_label(segment_trace_label, 'trip_mode'),
                             slicer='tour_id',
                             index_label='tour_id',
                             warn_if_empty=True)

        choices_list.append(choices)

        # FIXME - force garbage collection
        force_garbage_collect()

    choices = pd.concat(choices_list)

    tracing.print_summary('tour_mode_choice_simulate all tour type choices',
                          choices, value_counts=True)

    tracing.print_summary('trip_mode_choice_simulate all tour type choices',
                          tours_merged.tour_mode, value_counts=True)

    trips_df = trips.to_frame()
    trips_df['trip_mode'] = choices

    assert not trips_df.trip_mode.isnull().any()

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(trips_df,
                         label=tracing.extend_trace_label(trace_label, 'trip_mode'),
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)

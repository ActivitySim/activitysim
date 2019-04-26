# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import zip
from builtins import range

import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core.mem import force_garbage_collect

from .util.expressions import annotate_preprocessors

from activitysim.core import assign

from .util.expressions import skim_time_period_label

logger = logging.getLogger(__name__)


@inject.step()
def trip_mode_choice(
        trips,
        tours_merged,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id):
    """
    Trip mode choice - compute trip_mode (same values as for tour_mode) for each trip.

    Modes for each primary tour putpose are calculated separately because they have different
    coefficient values (stored in trip_mode_choice_coeffs.csv coefficient file.)

    Adds trip_mode column to trip table
    """
    trace_label = 'trip_mode_choice'
    model_settings = config.read_model_settings('trip_mode_choice.yaml')

    model_spec = \
        simulate.read_model_spec(file_name=model_settings['SPEC'])
    omnibus_coefficients = \
        assign.read_constant_spec(config.config_file_path(model_settings['COEFFS']))

    trips_df = trips.to_frame()
    logger.info("Running %s with %d trips", trace_label, trips_df.shape[0])

    tours_merged = tours_merged.to_frame()
    tours_merged = tours_merged[model_settings['TOURS_MERGED_CHOOSER_COLUMNS']]

    nest_spec = config.get_logit_model_settings(model_settings)

    tracing.print_summary('primary_purpose',
                          trips_df.primary_purpose, value_counts=True)

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips_df,
        tours_merged,
        left_on='tour_id',
        right_index=True,
        how="left")
    assert trips_merged.index.equals(trips.index)

    # setup skim keys
    assert ('trip_period' not in trips_merged)
    trips_merged['trip_period'] = skim_time_period_label(trips_merged.depart)

    orig_col = 'origin'
    dest_col = 'destination'

    odt_skim_stack_wrapper = skim_stack.wrap(left_key=orig_col, right_key=dest_col,
                                             skim_key='trip_period')
    od_skim_wrapper = skim_dict.wrap('origin', 'destination')

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
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

        locals_dict = assign.evaluate_constants(omnibus_coefficients[primary_purpose],
                                                constants=constants)
        locals_dict.update(constants)

        annotate_preprocessors(
            trips_segment, locals_dict, skims,
            model_settings, segment_trace_label)

        locals_dict.update(skims)
        choices = simulate.simple_simulate(
            choosers=trips_segment,
            spec=model_spec,
            nest_spec=nest_spec,
            skims=skims,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            trace_label=segment_trace_label,
            trace_choice_name='trip_mode_choice')

        alts = model_spec.columns
        choices = choices.map(dict(list(zip(list(range(len(alts))), alts))))

        # tracing.print_summary('trip_mode_choice %s choices' % primary_purpose,
        #                       choices, value_counts=True)

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

    trips_df = trips.to_frame()
    trips_df['trip_mode'] = choices

    tracing.print_summary('tour_modes',
                          trips_merged.tour_mode, value_counts=True)

    tracing.print_summary('trip_mode_choice choices',
                          choices, value_counts=True)

    assert not trips_df.trip_mode.isnull().any()

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(trips_df,
                         label=tracing.extend_trace_label(trace_label, 'trip_mode'),
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)

# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import expressions

from activitysim.core.util import assign_in_place
from activitysim.core.util import reindex

from .util import trip

logger = logging.getLogger(__name__)


@inject.step()
def stop_frequency(
        tours, tours_merged,
        stop_frequency_alts,
        network_los,
        chunk_size,
        trace_hh_id):
    """
    stop frequency model

    For each tour, shoose a number of intermediate inbound stops and outbound stops.
    Create a trip table with inbound and outbound trips.

    Thus, a tour with stop_frequency '2out_0in' will have two outbound and zero inbound stops,
    and four corresponding trips: three outbound, and one inbound.

    Adds stop_frequency str column to trips, with fields

    creates trips table with columns:

    ::

        - person_id
        - household_id
        - tour_id
        - primary_purpose
        - atwork
        - trip_num
        - outbound
        - trip_count

    """

    trace_label = 'stop_frequency'
    model_settings = config.read_model_settings('stop_frequency.yaml')

    tours = tours.to_frame()
    tours_merged = tours_merged.to_frame()

    assert not tours_merged.household_id.isnull().any()

    assert not (tours_merged.origin == -1).any()
    assert not (tours_merged.destination == -1).any()

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    # - run preprocessor to annotate tours_merged
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        # hack: preprocessor adds origin column in place if it does not exist already
        assert 'origin' in tours_merged
        assert 'destination' in tours_merged
        od_skim_stack_wrapper = network_los.get_default_skim_dict().wrap('origin', 'destination')
        skims = [od_skim_stack_wrapper]

        locals_dict = {
            "od_skims": od_skim_stack_wrapper,
            'network_los': network_los
        }
        locals_dict.update(constants)

        simulate.set_skim_wrapper_targets(tours_merged, skims)

        # this should be pre-slice as some expressions may count tours by type
        annotations = expressions.compute_columns(
            df=tours_merged,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

        assign_in_place(tours_merged, annotations)

    tracing.print_summary('stop_frequency segments',
                          tours_merged.primary_purpose, value_counts=True)

    choices_list = []
    for segment_type, choosers in tours_merged.groupby('primary_purpose'):

        logging.info("%s running segment %s with %s chooser rows" %
                     (trace_label, segment_type, choosers.shape[0]))

        spec = simulate.read_model_spec(file_name='stop_frequency_%s.csv' % segment_type)

        assert spec is not None, "spec for segment_type %s not found" % segment_type

        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, segment_type),
            trace_choice_name='stops')

        # convert indexes to alternative names
        choices = pd.Series(spec.columns[choices.values], index=choices.index)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    tracing.print_summary('stop_frequency', choices, value_counts=True)

    # add stop_frequency choices to tours table
    assign_in_place(tours, choices.to_frame('stop_frequency'))

    if 'primary_purpose' not in tours.columns:
        assign_in_place(tours, tours_merged[['primary_purpose']])

    pipeline.replace_table("tours", tours)

    # create trips table
    trips = trip.initialize_from_tours(tours, stop_frequency_alts)
    trips = pipeline.replace_table("trips", trips)
    tracing.register_traceable_table('trips', trips)
    pipeline.get_rn_generator().add_channel('trips', trips)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label="stop_frequency.tours",
                         slicer='person_id',
                         columns=None)

        tracing.trace_df(trips,
                         label="stop_frequency.trips",
                         slicer='person_id',
                         columns=None)

        tracing.trace_df(annotations,
                         label="stop_frequency.annotations",
                         columns=None)

        tracing.trace_df(tours_merged,
                         label="stop_frequency.tours_merged",
                         slicer='person_id',
                         columns=None)

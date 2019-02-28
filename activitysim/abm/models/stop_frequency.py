# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import numpy as np
import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from activitysim.core.util import assign_in_place
from .util import expressions
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


@inject.injectable()
def stop_frequency_alts():
    # alt file for building trips even though simulation is simple_simulate not interaction_simulate
    file_path = config.config_file_path('stop_frequency_alternatives.csv')
    df = pd.read_csv(file_path, comment='#')
    df.set_index('alt', inplace=True)
    return df


def process_trips(tours, stop_frequency_alts):

    MAX_TRIPS_PER_LEG = 4  # max number of trips per leg (inbound or outbound) of tour
    OUTBOUND_ALT = 'out'
    assert OUTBOUND_ALT in stop_frequency_alts.columns

    # get the actual alternatives for each person - have to go back to the
    # stop_frequency_alts dataframe to get this - the stop_frequency choice
    # column has the index values for the chosen alternative

    trips = stop_frequency_alts.loc[tours.stop_frequency]

    # assign tour ids to the index
    trips.index = tours.index

    """

    ::

      tours.stop_frequency    =>    proto trips table
      ________________________________________________________
                stop_frequency      |                out  in
      tour_id                       |     tour_id
      954910          1out_1in      |     954910       1   1
      985824          0out_1in      |     985824       0   1
    """

    # reformat with the columns given below
    trips = trips.stack().reset_index()
    trips.columns = ['tour_id', 'direction', 'trip_count']

    # tours legs have one more leg than stop
    trips.trip_count += 1

    # prefer direction as boolean
    trips['outbound'] = trips.direction == OUTBOUND_ALT

    """
           tour_id direction  trip_count  outbound
    0       954910       out           2      True
    1       954910        in           2     False
    2       985824       out           1      True
    3       985824        in           2     False
    """

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    trips = trips.take(np.repeat(trips.index.values, trips.trip_count.values))
    trips = trips.reset_index(drop=True)

    grouped = trips.groupby(['tour_id', 'outbound'])
    trips['trip_num'] = grouped.cumcount() + 1

    trips['person_id'] = reindex(tours.person_id, trips.tour_id)
    trips['household_id'] = reindex(tours.household_id, trips.tour_id)

    trips['primary_purpose'] = reindex(tours.primary_purpose, trips.tour_id)

    # reorder columns and drop 'direction'
    trips = trips[['person_id', 'household_id', 'tour_id', 'primary_purpose',
                   'trip_num', 'outbound', 'trip_count']]

    """
      person_id  household_id  tour_id  primary_purpose trip_num  outbound  trip_count
    0     32927         32927   954910             work        1      True           2
    1     32927         32927   954910             work        2      True           2
    2     32927         32927   954910             work        1     False           2
    3     32927         32927   954910             work        2     False           2
    4     33993         33993   985824             univ        1      True           1
    5     33993         33993   985824             univ        1     False           2
    6     33993         33993   985824             univ        2     False           2

    """

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * MAX_TRIPS_PER_LEG) + trips.trip_num
    trips['trip_id'] = trips.tour_id * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num

    trips.set_index('trip_id', inplace=True, verify_integrity=True)

    return trips


@inject.step()
def stop_frequency(
        tours, tours_merged,
        stop_frequency_alts,
        skim_dict,
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
        od_skim_stack_wrapper = skim_dict.wrap('origin', 'destination')
        skims = [od_skim_stack_wrapper]

        locals_dict = {
            "od_skims": od_skim_stack_wrapper
        }
        if constants is not None:
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
    trips = process_trips(tours, stop_frequency_alts)
    trips = pipeline.extend_table("trips", trips)
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

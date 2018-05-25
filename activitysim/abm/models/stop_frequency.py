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

from activitysim.core.util import assign_in_place
from .util import expressions
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


def get_stop_frequency_spec(tour_type):

    configs_dir = inject.get_injectable('configs_dir')
    file_name = 'stop_frequency_%s.csv' % tour_type

    if not os.path.exists(os.path.join(configs_dir, file_name)):
        return None

    return simulate.read_model_spec(configs_dir, file_name)


@inject.injectable()
def stop_frequency_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'stop_frequency.yaml')


@inject.injectable()
def stop_frequency_alts(configs_dir):
    # alt file for building trips even though simulation is simple_simulate not interaction_simulate
    f = os.path.join(configs_dir, 'stop_frequency_alternatives.csv')
    df = pd.read_csv(f, comment='#')
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
    trips['atwork'] = reindex(tours.tour_category, trips.tour_id) == 'atwork'

    # reorder columns and drop 'direction'
    trips = trips[['person_id', 'household_id', 'tour_id',
                   'primary_purpose', 'atwork',
                   'trip_num', 'outbound', 'trip_count']]

    trips['first'] = (trips.trip_num == 1)
    trips['last'] = (trips.trip_num == trips.trip_count)
    # omit because redundant?
    # trips['intermediate'] = (trips.trip_num>1) & (trips.trip_num<trips.trip_count)

    """
      person_id  household_id  primary_purpose tour_id  trip_num  outbound  trip_count  first  last
    0     32927         32927             work  954910         1      True           2   True False
    1     32927         32927             work  954910         2      True           2  False  True
    2     32927         32927             work  954910         1     False           2   True False
    3     32927         32927             work  954910         2     False           2  False  True
    4     33993         33993             univ  985824         1      True           1   True True
    5     33993         33993             univ  985824         1     False           2   True False
    6     33993         33993             univ  985824         2     False           2  False  True

    """

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * MAX_TRIPS_PER_LEG) + trips.trip_num
    trips['trip_id'] = trips.tour_id * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num

    # id of next trip in inbound or outbound leg
    trips['next_trip_id'] = np.where(trips['last'], 0, trips.trip_id + 1)

    trips.set_index('trip_id', inplace=True, verify_integrity=True)

    return trips


@inject.step()
def stop_frequency(
        tours, tours_merged,
        stop_frequency_alts,
        stop_frequency_settings,
        skim_dict, skim_stack,
        chunk_size,
        trace_hh_id):
    """
    stop frequency
    """

    trace_label = 'stop_frequency'

    tours = tours.to_frame()
    tours_merged = tours_merged.to_frame()
    assert not tours_merged.household_id.isnull().any()

    nest_spec = config.get_logit_model_settings(stop_frequency_settings)
    constants = config.get_model_constants(stop_frequency_settings)

    # - run preprocessor to annotate tours_merged
    preprocessor_settings = stop_frequency_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        # hack: preprocessor adds origin column in place if it does not exist already
        od_skim_stack_wrapper = skim_dict.wrap('origin', 'destination')
        skims = [od_skim_stack_wrapper]

        locals_dict = {
            "od_skims": od_skim_stack_wrapper
        }
        if constants is not None:
            locals_dict.update(constants)

        simulate.add_skims(tours_merged, skims)

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

        spec = get_stop_frequency_spec(segment_type)

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
    pipeline.get_rn_generator().add_channel(trips, 'trips')

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

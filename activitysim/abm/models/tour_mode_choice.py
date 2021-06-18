# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import logit
from activitysim.core import orca
from activitysim.core.mem import force_garbage_collect
from activitysim.core.util import assign_in_place, reindex

from activitysim.core import los
from activitysim.core.pathbuilder import TransitVirtualPathBuilder

from activitysim.core import los

from .util.mode import run_tour_mode_choice_simulate
from .util import trip
from .util import estimation

logger = logging.getLogger(__name__)

"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@inject.step()
def tour_mode_choice_simulate(tours, persons_merged,
                              network_los,
                              chunk_size,
                              trace_hh_id):
    """
    Tour mode choice simulate
    """
    trace_label = 'tour_mode_choice'
    model_settings_file_name = 'tour_mode_choice.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')
    mode_column_name = 'tour_mode'

    primary_tours = tours.to_frame()
    assert not (primary_tours.tour_category == 'atwork').any()

    logger.info("Running %s with %d tours" % (trace_label, primary_tours.shape[0]))

    tracing.print_summary('tour_types',
                          primary_tours.tour_type, value_counts=True)

    persons_merged = persons_merged.to_frame()
    primary_tours_merged = pd.merge(primary_tours, persons_merged, left_on='person_id',
                                    right_index=True, how='left', suffixes=('', '_r'))

    constants = {}
    # model_constants can appear in expressions
    constants.update(config.get_model_constants(model_settings))

    skim_dict = network_los.get_default_skim_dict()

    # setup skim keys
    orig_col_name = 'home_zone_id'
    dest_col_name = 'destination'

    out_time_col_name = 'start'
    in_time_col_name = 'end'
    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col_name, dest_key=dest_col_name,
                                               dim3_key='out_period')
    dot_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col_name, dest_key=orig_col_name,
                                               dim3_key='in_period')
    odr_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col_name, dest_key=dest_col_name,
                                               dim3_key='in_period')
    dor_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col_name, dest_key=orig_col_name,
                                               dim3_key='out_period')
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,  # dot return skims for e.g. TNC bridge return fare
        "dor_skims": dor_skim_stack_wrapper,  # odt return skims for e.g. TNC bridge return fare
        "od_skims": od_skim_stack_wrapper,
        'orig_col_name': orig_col_name,
        'dest_col_name': dest_col_name,
        'out_time_col_name': out_time_col_name,
        'in_time_col_name': in_time_col_name
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?

        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(orig_key=orig_col_name, dest_key=dest_col_name,
                                           tod_key='out_period', segment_key='demographic_segment',
                                           cache_choices=True,
                                           trace_label=trace_label, tag='tvpb_logsum_odt')
        tvpb_logsum_dot = tvpb.wrap_logsum(orig_key=dest_col_name, dest_key=orig_col_name,
                                           tod_key='in_period', segment_key='demographic_segment',
                                           cache_choices=True,
                                           trace_label=trace_label, tag='tvpb_logsum_dot')

        skims.update({
            'tvpb_logsum_odt': tvpb_logsum_odt,
            'tvpb_logsum_dot': tvpb_logsum_dot
        })

        # TVPB constants can appear in expressions
        if model_settings.get('use_TVPB_constants', True):
            constants.update(network_los.setting('TVPB_SETTINGS.tour_mode_choice.CONSTANTS'))

    estimator = estimation.manager.begin_estimation('tour_mode_choice')
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # (run_tour_mode_choice_simulate writes choosers post-annotation)

    # FIXME should normalize handling of tour_type and tour_purpose
    # mtctm1 school tour_type includes univ, which has different coefficients from elementary and HS
    # we should either add this column when tours created or add univ to tour_types
    not_university = (primary_tours_merged.tour_type != 'school') | ~primary_tours_merged.is_university
    primary_tours_merged['tour_purpose'] = \
        primary_tours_merged.tour_type.where(not_university, 'univ')

    # if trip logsums are used, run trip mode choice 
    if model_settings.get('COMPUTE_TRIP_MODE_CHOICE_LOGSUMS', False):

        # Construct table of hypothetical trips from tours for each potential
        # tour mode. Two trips (1 inbound, 1 outbound) per [tour, mode] bundle.
        # O/D, purpose, and departure times are inherited from tour.
        primary_tours_merged['stop_frequency'] = '0out_0in'  # no intermediate stops
        primary_tours_merged['primary_purpose'] = primary_tours_merged['tour_purpose']
        trips = trip.initialize_from_tours(primary_tours_merged)
        trips['stop_frequency'] = '0out_0in'
        outbound = trips['outbound']
        trips['depart'] = reindex(primary_tours_merged.start, trips.tour_id)
        trips.loc[~outbound, 'depart'] = reindex(primary_tours_merged.end, trips.loc[~outbound,'tour_id'])

        logsum_trips = pd.DataFrame()
        nest_spec = config.get_logit_model_settings(model_settings)

        # actual coeffs dont matter here, just need them to load the nest structure
        coefficients = simulate.get_segment_coefficients(
            model_settings, primary_tours_merged.iloc[0]['tour_purpose'])
        nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)
        tour_mode_alts = []
        for nest in logit.each_nest(nest_spec):
            if nest.is_leaf:
                tour_mode_alts.append(nest.name)

        # repeat rows from the trips table iterating over tour mode
        for tour_mode in tour_mode_alts:
            trips['tour_mode'] = tour_mode
            logsum_trips = pd.concat((logsum_trips, trips), ignore_index=True)
        assert len(logsum_trips) == len(trips) * len(tour_mode_alts)
        logsum_trips.index.name = 'trip_id'

        pipeline.replace_table('trips', logsum_trips)
        tracing.register_traceable_table('trips', logsum_trips)
        pipeline.get_rn_generator().add_channel('trips', logsum_trips)

        # run trip mode choice on pseudo-trips. use orca instead of pipeline to
        # execute the step because pipeline can only handle one open step at a time
        orca.run(['trip_mode_choice'])

        # grab trip mode choice logsums and pivot by tour mode and direction, index
        # on tour_id to enable merge back to choosers table
        trips = inject.get_table('trips').to_frame()
        trip_dir_mode_logsums = trips.pivot(
            index='tour_id', columns=['tour_mode', 'outbound'], values='trip_mode_choice_logsum')
        new_cols = [
            '_'.join(['logsum', mode, 'outbound' if outbound else 'inbound'])
            for mode, outbound in trip_dir_mode_logsums.columns]
        trip_dir_mode_logsums.columns = new_cols
        trip_dir_mode_logsums.reindex(primary_tours_merged.index)
        primary_tours_merged = pd.merge(primary_tours_merged, trip_dir_mode_logsums, left_index=True, right_index=True)
        pipeline.get_rn_generator().drop_channel('trips')
        tracing.deregister_traceable_table('trips')

    choices_list = []
    for tour_purpose, tours_segment in primary_tours_merged.groupby('tour_purpose'):

        logger.info("tour_mode_choice_simulate tour_type '%s' (%s tours)" %
                    (tour_purpose, len(tours_segment.index), ))

        if network_los.zone_system == los.THREE_ZONE:
            tvpb_logsum_odt.extend_trace_label(tour_purpose)
            tvpb_logsum_dot.extend_trace_label(tour_purpose)

        # name index so tracing knows how to slice
        assert tours_segment.index.name == 'tour_id'

        choices_df = run_tour_mode_choice_simulate(
            tours_segment,
            tour_purpose, model_settings,
            mode_column_name=mode_column_name,
            logsum_column_name=logsum_column_name,
            network_los=network_los,
            skims=skims,
            constants=constants,
            estimator=estimator,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
            trace_choice_name='tour_mode_choice')

        tracing.print_summary('tour_mode_choice_simulate %s choices_df' % tour_purpose,
                              choices_df.tour_mode, value_counts=True)

        choices_list.append(choices_df)

        # FIXME - force garbage collection
        force_garbage_collect()

    choices_df = pd.concat(choices_list)

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:

        tvpb_mode_path_types = model_settings.get('tvpb_mode_path_types', False)
        if tvpb_mode_path_types:
            for mode, path_types in tvpb_mode_path_types.items():

                for direction, skim in zip(['od', 'do'], [tvpb_logsum_odt, tvpb_logsum_dot]):

                    path_type = path_types[direction]
                    skim_cache = skim.cache[path_type]

                    print(f"mode {mode} direction {direction} path_type {path_type}")

                    for c in skim_cache:

                        dest_col = f'{direction}_{c}'

                        if dest_col not in choices_df:
                            choices_df[dest_col] = 0 if pd.api.types.is_numeric_dtype(skim_cache[c]) else ''
                        choices_df[dest_col].where(choices_df.tour_mode != mode, skim_cache[c], inplace=True)

    if estimator:
        estimator.write_choices(choices_df.tour_mode)
        choices_df.tour_mode = estimator.get_survey_values(choices_df.tour_mode, 'tours', 'tour_mode')
        estimator.write_override_choices(choices_df.tour_mode)
        estimator.end_estimation()

    tracing.print_summary('tour_mode_choice_simulate all tour type choices',
                          choices_df.tour_mode, value_counts=True)

    # so we can trace with annotations
    assign_in_place(primary_tours, choices_df)

    # update tours table with mode choice (and optionally logsums)
    all_tours = tours.to_frame()
    assign_in_place(all_tours, choices_df)

    pipeline.replace_table("tours", all_tours)

    if trace_hh_id:
        tracing.trace_df(primary_tours,
                         label=tracing.extend_trace_label(trace_label, mode_column_name),
                         slicer='tour_id',
                         index_label='tour_id',
                         warn_if_empty=True)

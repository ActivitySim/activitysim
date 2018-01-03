# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.core.util import reindex
from activitysim.core.util import left_merge_on_index_and_col

from .util import expressions
from activitysim.core.util import assign_in_place

from .util.logsums import compute_logsums
from .util.logsums import mode_choice_logsums_spec

from .util.expressions import skim_time_period_label

logger = logging.getLogger(__name__)
DUMP = False


@inject.injectable()
def atwork_subtour_destination_sample_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'atwork_subtour_destination_sample.csv')


@inject.injectable()
def atwork_subtour_destination_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'atwork_subtour_destination.yaml')


@inject.step()
def atwork_subtour_destination_sample(tours,
                                      persons_merged,
                                      atwork_subtour_destination_sample_spec,
                                      skim_dict,
                                      destination_size_terms,
                                      chunk_size,
                                      trace_hh_id):

    trace_label = 'atwork_subtour_location_sample'
    model_settings = inject.get_injectable('atwork_subtour_destination_settings')

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    tours = tours[tours.tour_category == 'subtour']

    # merge persons into tours
    choosers = pd.merge(tours, persons_merged, left_on='person_id', right_index=True)

    alternatives = destination_size_terms.to_frame()

    constants = config.get_model_constants(model_settings)

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_col_name = model_settings["ALT_COL_NAME"]
    chooser_col_name = 'workplace_taz'

    logger.info("Running atwork_subtour_location_sample with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a workplace_taz
    # in the choosers and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap(chooser_col_name, 'TAZ')

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    choices = interaction_sample(
        choosers,
        alternatives,
        sample_size=sample_size,
        alt_col_name=alt_col_name,
        spec=atwork_subtour_destination_sample_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    choices['person_id'] = choosers.person_id
    choices['workplace_taz'] = choosers.workplace_taz

    inject.add_table('atwork_subtour_destination_sample', choices)


@inject.step()
def atwork_subtour_destination_logsums(persons_merged,
                                       land_use,
                                       skim_dict, skim_stack,
                                       atwork_subtour_destination_sample,
                                       configs_dir,
                                       chunk_size,
                                       trace_hh_id):
    """
    add logsum column to existing workplace_location_sample able

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in workplace_location_sample, and computing the logsum of all the utilities

    +-------+--------------+----------------+------------+----------------+
    | PERID | dest_TAZ     | rand           | pick_count | logsum (added) |
    +=======+==============+================+============+================+
    | 23750 |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-------+--------------+----------------+------------+----------------+
    + 23750 | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-------+--------------+----------------+------------+----------------+
    + ...   |              |                |            |                |
    +-------+--------------+----------------+------------+----------------+
    | 23751 | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-------+--------------+----------------+------------+----------------+
    | 23751 | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-------+--------------+----------------+------------+----------------+

    """

    trace_label = 'atwork_subtour_destination_logsums'
    model_settings = inject.get_injectable('atwork_subtour_destination_settings')

    logsums_spec = mode_choice_logsums_spec(configs_dir, 'work')

    alt_col_name = model_settings["ALT_COL_NAME"]
    chooser_col_name = 'workplace_taz'

    # FIXME - just using settings from tour_mode_choice
    logsum_settings = config.read_model_settings(configs_dir, 'tour_mode_choice.yaml')

    persons_merged = persons_merged.to_frame()
    atwork_subtour_destination_sample = atwork_subtour_destination_sample.to_frame()

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['LOGSUM_CHOOSER_COLUMNS']
    persons_merged = persons_merged[chooser_columns]

    # merge persons into tours
    choosers = pd.merge(atwork_subtour_destination_sample,
                        persons_merged,
                        left_on='person_id',
                        right_index=True,
                        how="left")

    choosers['in_period'] = skim_time_period_label(model_settings['IN_PERIOD'])
    choosers['out_period'] = skim_time_period_label(model_settings['OUT_PERIOD'])

    # FIXME - should do this in expression file?
    choosers['dest_topology'] = reindex(land_use.TOPOLOGY, choosers[alt_col_name])
    choosers['dest_density_index'] = reindex(land_use.density_index, choosers[alt_col_name])

    logger.info("Running atwork_subtour_destination_logsums with %s rows" % len(choosers))

    tracing.dump_df(DUMP, persons_merged, trace_label, 'persons_merged')
    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    logsums = compute_logsums(
        choosers, logsums_spec, logsum_settings,
        skim_dict, skim_stack, chooser_col_name, alt_col_name, chunk_size, trace_hh_id, trace_label)

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved. logsums does have a
    # matching index, since atwork_subtour_destination_sample was on left side of merge de-dup merge
    inject.add_column("atwork_subtour_destination_sample", "mode_choice_logsum", logsums)


@inject.injectable()
def atwork_subtour_destination_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'atwork_subtour_destination.csv')


@inject.step()
def atwork_subtour_destination_simulate(tours,
                                        persons_merged,
                                        atwork_subtour_destination_sample,
                                        atwork_subtour_destination_spec,
                                        skim_dict,
                                        destination_size_terms,
                                        configs_dir,
                                        chunk_size,
                                        trace_hh_id):
    """
    atwork_subtour_destination model on atwork_subtour_destination_sample
    annotated with mode_choice logsum to select a destination from sample alternatives
    """

    trace_label = 'atwork_subtour_destination_simulate'
    model_settings = inject.get_injectable('atwork_subtour_destination_settings')

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == 'subtour']
    # merge persons into tours
    choosers = pd.merge(subtours,
                        persons_merged.to_frame(),
                        left_on='person_id', right_index=True)

    alt_col_name = model_settings["ALT_COL_NAME"]
    chooser_col_name = 'workplace_taz'

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge additional alt columns into alt sample list
    atwork_subtour_destination_sample = atwork_subtour_destination_sample.to_frame()
    destination_size_terms = destination_size_terms.to_frame()
    alternatives = \
        pd.merge(atwork_subtour_destination_sample, destination_size_terms,
                 left_on=alt_col_name, right_index=True, how="left")

    tracing.dump_df(DUMP, alternatives, trace_label, 'alternatives')

    constants = config.get_model_constants(model_settings)

    sample_pool_size = len(destination_size_terms.index)

    logger.info("Running atwork_subtour_destination_simulate with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap(chooser_col_name, alt_col_name)

    locals_d = {
        'skims': skims,
        'sample_pool_size': float(sample_pool_size)
    }
    if constants is not None:
        locals_d.update(constants)

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    choices = interaction_sample_simulate(
        choosers,
        alternatives,
        spec=atwork_subtour_destination_spec,
        choice_column=alt_col_name,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='workplace_location')

    tracing.print_summary('subtour destination', choices, describe=True)

    subtours['destination'] = choices

    results = expressions.compute_columns(
        df=subtours,
        model_settings='annotate_tours_with_dest',
        configs_dir=configs_dir,
        trace_label=trace_label)

    assign_in_place(tours, subtours[['destination']])
    assign_in_place(tours, results)

    pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label=trace_label,
                         columns=['destination'],
                         warn_if_empty=True)

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

from .util import logsums as logsum

from .util.expressions import skim_time_period_label
from .util.tour_destination import tour_destination_size_terms

logger = logging.getLogger(__name__)
DUMP = False


@inject.step()
def atwork_subtour_destination_sample(tours,
                                      persons_merged,
                                      skim_dict,
                                      land_use, size_terms,
                                      chunk_size, trace_hh_id):

    trace_label = 'atwork_subtour_location_sample'
    model_settings = config.read_model_settings('atwork_subtour_destination.yaml')
    model_spec = simulate.read_model_spec(file_name='atwork_subtour_destination_sample.csv')

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    tours = tours[tours.tour_category == 'atwork']

    # - if no atwork subtours
    if tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    # merge persons into tours
    choosers = pd.merge(tours, persons_merged, left_on='person_id', right_index=True)
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    alternatives = tour_destination_size_terms(land_use, size_terms, 'atwork')

    constants = config.get_model_constants(model_settings)

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running atwork_subtour_location_sample with %d tours" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a workplace_taz
    # in the choosers and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap('workplace_taz', 'TAZ')

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    choices = interaction_sample(
        choosers,
        alternatives,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        spec=model_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    choices['person_id'] = choosers.person_id

    inject.add_table('atwork_subtour_destination_sample', choices)


@inject.step()
def atwork_subtour_destination_logsums(persons_merged,
                                       skim_dict, skim_stack,
                                       chunk_size, trace_hh_id):
    """
    add logsum column to existing workplace_location_sample able

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in workplace_location_sample, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | person_id | dest_TAZ     | rand           | pick_count | logsum (added) |
    +===========+==============+================+============+================+
    | 23750     |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-----------+--------------+----------------+------------+----------------+
    + 23750     | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-----------+--------------+----------------+------------+----------------+
    + ...       |              |                |            |                |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-----------+--------------+----------------+------------+----------------+

    """

    trace_label = 'atwork_subtour_destination_logsums'

    # use inject.get_table as this won't exist if there are no atwork subtours
    destination_sample = inject.get_table('atwork_subtour_destination_sample', default=None)
    if destination_sample is None:
        tracing.no_results(trace_label)
        return

    model_settings = config.read_model_settings('atwork_subtour_destination.yaml')
    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    destination_sample = destination_sample.to_frame()
    persons_merged = persons_merged.to_frame()

    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(persons_merged, logsum_settings, model_settings)

    # merge persons into tours
    choosers = pd.merge(destination_sample,
                        persons_merged,
                        left_on='person_id',
                        right_index=True,
                        how="left")

    logger.info("Running %s with %s rows" % (trace_label, len(choosers)))

    tracing.dump_df(DUMP, persons_merged, trace_label, 'persons_merged')
    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    tour_purpose = 'atwork'
    logsums = logsum.compute_logsums(
        choosers,
        tour_purpose,
        logsum_settings, model_settings,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id,
        trace_label)

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved. logsums does have a
    # matching index, since atwork_subtour_destination_sample was on left side of merge
    inject.add_column("atwork_subtour_destination_sample", "mode_choice_logsum", logsums)


@inject.step()
def atwork_subtour_destination_simulate(tours,
                                        persons_merged,
                                        skim_dict,
                                        land_use, size_terms,
                                        chunk_size, trace_hh_id):
    """
    atwork_subtour_destination model on atwork_subtour_destination_sample
    annotated with mode_choice logsum to select a destination from sample alternatives
    """

    trace_label = 'atwork_subtour_destination_simulate'

    # use inject.get_table as this won't exist if there are no atwork subtours
    destination_sample = inject.get_table('atwork_subtour_destination_sample', default=None)
    if destination_sample is None:
        tracing.no_results(trace_label)
        return

    destination_sample = destination_sample.to_frame()

    model_settings = config.read_model_settings('atwork_subtour_destination.yaml')
    model_spec = simulate.read_model_spec(file_name='atwork_subtour_destination.csv')

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == 'atwork']

    # interaction_sample_simulate insists choosers appear in same order as alts
    subtours = subtours.sort_index()

    # merge persons into tours
    choosers = pd.merge(subtours,
                        persons_merged.to_frame(),
                        left_on='person_id', right_index=True)
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]
    chooser_col_name = 'workplace_taz'

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge additional alt columns into alt sample list
    destination_size_terms = tour_destination_size_terms(land_use, size_terms, 'atwork')

    alternatives = \
        pd.merge(destination_sample, destination_size_terms,
                 left_on=alt_dest_col_name, right_index=True, how="left")

    tracing.dump_df(DUMP, alternatives, trace_label, 'alternatives')

    constants = config.get_model_constants(model_settings)

    logger.info("Running atwork_subtour_destination_simulate with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap(chooser_col_name, alt_dest_col_name)

    locals_d = {
        'skims': skims,
    }
    if constants is not None:
        locals_d.update(constants)

    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    choices = interaction_sample_simulate(
        choosers,
        alternatives,
        spec=model_spec,
        choice_column=alt_dest_col_name,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='workplace_location')

    subtours['destination'] = choices

    assign_in_place(tours, subtours[['destination']])

    pipeline.replace_table("tours", tours)

    pipeline.drop_table('atwork_subtour_destination_sample')

    tracing.print_summary('subtour destination', subtours.destination, describe=True)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label=trace_label,
                         columns=['destination'])

# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample
from activitysim.core.util import assign_in_place

from .util import estimation

from .util import logsums as logsum
from activitysim.abm.tables.size_terms import tour_destination_size_terms

logger = logging.getLogger(__name__)
DUMP = False


def atwork_subtour_destination_sample(
        tours,
        persons_merged,
        model_settings,
        network_los,
        destination_size_terms,
        estimator,
        chunk_size, trace_label):

    model_spec = simulate.read_model_spec(file_name=model_settings['SAMPLE_SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    # merge persons into tours
    choosers = pd.merge(tours, persons_merged, left_on='person_id', right_index=True)
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    constants = config.get_model_constants(model_settings)

    sample_size = model_settings['SAMPLE_SIZE']
    if estimator:
        # FIXME interaction_sample will return unsampled complete alternatives with probs and pick_count
        logger.info("Estimation mode for %s using unsampled alternatives short_circuit_choices" % (trace_label,))
        sample_size = 0

    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']

    logger.info("Running atwork_subtour_location_sample with %d tours", len(choosers))

    # create wrapper with keys for this lookup - in this case there is a workplace_zone_id
    # in the choosers and a zone_id in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    dest_column_name = destination_size_terms.index.name
    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap('workplace_zone_id', dest_column_name)

    locals_d = {
        'skims': skims
    }
    if constants is not None:
        locals_d.update(constants)

    choices = interaction_sample(
        choosers,
        alternatives=destination_size_terms,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        spec=model_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    # remember person_id in chosen alts so we can merge with persons in subsequent steps
    choices['person_id'] = choosers.person_id

    return choices


def atwork_subtour_destination_logsums(
        persons_merged,
        destination_sample,
        model_settings,
        network_los,
        chunk_size, trace_label):
    """
    add logsum column to existing atwork_subtour_destination_sample table

    logsum is calculated by running the mode_choice model for each sample (person, dest_zone_id) pair
    in atwork_subtour_destination_sample, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | person_id | dest_zone_id | rand           | pick_count | logsum (added) |
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

    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(persons_merged, logsum_settings, model_settings)

    # merge persons into tours
    choosers = pd.merge(destination_sample,
                        persons_merged,
                        left_on='person_id',
                        right_index=True,
                        how="left")

    logger.info("Running %s with %s rows", trace_label, len(choosers))

    tracing.dump_df(DUMP, persons_merged, trace_label, 'persons_merged')
    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    tour_purpose = 'atwork'
    logsums = logsum.compute_logsums(
        choosers,
        tour_purpose,
        logsum_settings, model_settings,
        network_los,
        chunk_size,
        trace_label)

    destination_sample['mode_choice_logsum'] = logsums

    return destination_sample


def atwork_subtour_destination_simulate(
        subtours,
        persons_merged,
        destination_sample,
        want_logsums,
        model_settings,
        network_los,
        destination_size_terms,
        estimator,
        chunk_size, trace_label):
    """
    atwork_subtour_destination model on atwork_subtour_destination_sample
    annotated with mode_choice logsum to select a destination from sample alternatives
    """

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    # interaction_sample_simulate insists choosers appear in same order as alts
    subtours = subtours.sort_index()

    # merge persons into tours
    choosers = pd.merge(subtours,
                        persons_merged,
                        left_on='person_id', right_index=True)
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    if estimator:
        estimator.write_choosers(choosers)

    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']
    chooser_col_name = 'workplace_zone_id'

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge destination_size_terms columns into alt sample list
    alternatives = \
        pd.merge(destination_sample, destination_size_terms,
                 left_on=alt_dest_col_name, right_index=True, how="left")

    tracing.dump_df(DUMP, alternatives, trace_label, 'alternatives')

    constants = config.get_model_constants(model_settings)

    logger.info("Running atwork_subtour_destination_simulate with %d persons", len(choosers))

    # create wrapper with keys for this lookup - in this case there is a home_zone_id in the choosers
    # and a zone_id in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_default_skim_dict()
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
        want_logsums=want_logsums,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='atwork_subtour',
        estimator=estimator)

    if not want_logsums:
        # for consistency, always return a dataframe with canonical column name
        assert isinstance(choices, pd.Series)
        choices = choices.to_frame('choice')

    return choices


@inject.step()
def atwork_subtour_destination(
        tours,
        persons_merged,
        network_los,
        land_use, size_terms,
        chunk_size, trace_hh_id):

    trace_label = 'atwork_subtour_destination'
    model_settings_file_name = 'atwork_subtour_destination.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    destination_column_name = 'destination'
    logsum_column_name = model_settings.get('DEST_CHOICE_LOGSUM_COLUMN_NAME')
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get('DEST_CHOICE_SAMPLE_TABLE_NAME')
    want_sample_table = config.setting('want_dest_choice_sample_tables') and sample_table_name is not None

    persons_merged = persons_merged.to_frame()

    tours = tours.to_frame()
    subtours = tours[tours.tour_category == 'atwork']
    # interaction_sample_simulate insists choosers appear in same order as alts
    subtours = subtours.sort_index()

    # - if no atwork subtours
    if subtours.shape[0] == 0:
        tracing.no_results('atwork_subtour_destination')
        return

    estimator = estimation.manager.begin_estimation('atwork_subtour_destination')
    if estimator:
        estimator.write_coefficients(simulate.read_model_coefficients(model_settings))
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag='SPEC')
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(inject.get_injectable('size_terms'), 'size_terms', append=False)
        estimator.write_table(inject.get_table('land_use').to_frame(), 'landuse', append=False)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    destination_size_terms = tour_destination_size_terms(land_use, size_terms, 'atwork')

    destination_sample_df = atwork_subtour_destination_sample(
        subtours,
        persons_merged,
        model_settings,
        network_los,
        destination_size_terms,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'sample'))

    destination_sample_df = atwork_subtour_destination_logsums(
        persons_merged,
        destination_sample_df,
        model_settings,
        network_los,
        chunk_size=chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'logsums'))

    choices_df = atwork_subtour_destination_simulate(
        subtours,
        persons_merged,
        destination_sample_df,
        want_logsums,
        model_settings,
        network_los,
        destination_size_terms,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'simulate'))

    if estimator:
        estimator.write_choices(choices_df['choice'])
        choices_df['choice'] = estimator.get_survey_values(choices_df['choice'], 'tours', 'destination')
        estimator.write_override_choices(choices_df['choice'])
        estimator.end_estimation()

    subtours[destination_column_name] = choices_df['choice']
    assign_in_place(tours, subtours[[destination_column_name]])

    if want_logsums:
        subtours[logsum_column_name] = choices_df['logsum']
        assign_in_place(tours, subtours[[logsum_column_name]])

    pipeline.replace_table("tours", tours)

    if want_sample_table:
        # FIXME - sample_table
        assert len(destination_sample_df.index.unique()) == len(choices_df)
        destination_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'],
                                        append=True, inplace=True)
        pipeline.extend_table(sample_table_name, destination_sample_df)

    tracing.print_summary(destination_column_name,
                          subtours[destination_column_name],
                          describe=True)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label='atwork_subtour_destination',
                         columns=['destination'])

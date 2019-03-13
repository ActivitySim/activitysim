# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.util import reindex

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.core.mem import force_garbage_collect


from . import logsums as logsum
from activitysim.abm.tables.size_terms import tour_destination_size_terms

logger = logging.getLogger(__name__)
DUMP = False


class SizeTermCalculator(object):
    """
    convenience object to provide size_terms for a selector (e.g. non_mandatory)
    for various segments (e.g. tour_type or purpose)
    returns size terms for specified segment in df or series form
    """

    def __init__(self, size_term_selector):

        # do this once so they can request siae_terms for various segments (tour_type or purpose)
        land_use = inject.get_table('land_use')
        size_terms = inject.get_injectable('size_terms')
        self.destination_size_terms = \
            tour_destination_size_terms(land_use, size_terms, size_term_selector)

    def dest_size_terms_df(self, segment_name):
        # return size terms as df with one column named 'size_term'
        # convenient if creating or merging with alts

        size_terms = self.destination_size_terms[[segment_name]].copy()
        size_terms.columns = ['size_term']
        return size_terms

    def dest_size_terms_series(self, segment_name):
        # return size terms as as series
        # convenient (and no copy overhead) if reindexing and assigning into alts column
        return self.destination_size_terms[segment_name]


def run_destination_sample(
        spec_segment_name,
        tours,
        persons_merged,
        model_settings,
        skim_dict,
        destination_size_terms,
        chunk_size, trace_label):

    model_spec_file_name = model_settings['SAMPLE_SPEC']
    model_spec = simulate.read_model_spec(file_name=model_spec_file_name)
    model_spec = model_spec[[spec_segment_name]]

    # merge persons into tours
    choosers = pd.merge(tours, persons_merged, left_on='person_id', right_index=True, how='left')
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    constants = config.get_model_constants(model_settings)

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("running %s with %d tours", trace_label, len(choosers))

    # create wrapper with keys for this lookup - in this case there is a workplace_taz
    # in the choosers and a TAZ in the alternatives which get merged during interaction
    # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
    # the skims will be available under the name "skims" for any @ expressions
    origin_col_name = model_settings['CHOOSER_ORIG_COL_NAME']
    if origin_col_name == 'TAZ':
        origin_col_name = 'TAZ_chooser'
    skims = skim_dict.wrap(origin_col_name, 'TAZ')

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
    # (broadcasts person_id onto all alternatives sharing the same tour_id index value)
    choices['person_id'] = choosers.person_id

    return choices


def run_destination_logsums(
        tour_purpose,
        persons_merged,
        destination_sample,
        model_settings,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, trace_label):
    """
    add logsum column to existing tour_destination_sample table

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in destination_sample, and computing the logsum of all the utilities
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

    logsums = logsum.compute_logsums(
        choosers,
        tour_purpose,
        logsum_settings, model_settings,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id,
        trace_label)

    destination_sample['mode_choice_logsum'] = logsums

    return destination_sample


def run_destination_simulate(
        spec_segment_name,
        tours,
        persons_merged,
        destination_sample,
        model_settings,
        skim_dict,
        destination_size_terms,
        chunk_size, trace_label):
    """
    run destination_simulate on tour_destination_sample
    annotated with mode_choice logsum to select a destination from sample alternatives
    """

    model_spec_file_name = model_settings['SPEC']
    model_spec = simulate.read_model_spec(file_name=model_spec_file_name)
    model_spec = model_spec[[spec_segment_name]]

    # merge persons into tours
    choosers = pd.merge(tours,
                        persons_merged,
                        left_on='person_id', right_index=True, how='left')
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]
    origin_col_name = model_settings['CHOOSER_ORIG_COL_NAME']

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge size_terms column into alt sample list
    destination_sample['size_term'] = \
        reindex(destination_size_terms.size_term, destination_sample[alt_dest_col_name])

    tracing.dump_df(DUMP, destination_sample, trace_label, 'alternatives')

    constants = config.get_model_constants(model_settings)

    logger.info("Running tour_destination_simulate with %d persons", len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap(origin_col_name, alt_dest_col_name)

    locals_d = {
        'skims': skims,
    }
    if constants is not None:
        locals_d.update(constants)

    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    choices = interaction_sample_simulate(
        choosers,
        destination_sample,
        spec=model_spec,
        choice_column=alt_dest_col_name,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='destination')

    return choices


def run_tour_destination(
        tours,
        persons_merged,
        model_settings,
        skim_dict,
        skim_stack,
        chunk_size, trace_hh_id, trace_label):

    size_term_calculator = SizeTermCalculator(model_settings['SIZE_TERM_SELECTOR'])

    chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN_NAME']

    # maps segment names to compact (integer) ids
    segments = model_settings['SEGMENTS']

    # interaction_sample_simulate insists choosers appear in same order as alts
    tours = tours.sort_index()

    choices_list = []
    for segment_name in segments:

        choosers = tours[tours[chooser_segment_column] == segment_name]

        # size_term segment is segment_name
        segment_destination_size_terms = size_term_calculator.dest_size_terms_df(segment_name)

        if choosers.shape[0] == 0:
            logger.info("%s skipping segment %s: no choosers", trace_label, segment_name)
            continue

        # - destination_sample
        spec_segment_name = segment_name  # spec_segment_name is segment_name
        location_sample_df = \
            run_destination_sample(
                spec_segment_name,
                choosers,
                persons_merged,
                model_settings,
                skim_dict,
                segment_destination_size_terms,
                chunk_size,
                tracing.extend_trace_label(trace_label, 'sample.%s' % segment_name))

        # - destination_logsums
        tour_purpose = segment_name  # tour_purpose is segment_name
        location_sample_df = \
            run_destination_logsums(
                tour_purpose,
                persons_merged,
                location_sample_df,
                model_settings,
                skim_dict, skim_stack,
                chunk_size, trace_hh_id,
                tracing.extend_trace_label(trace_label, 'logsums.%s' % segment_name))

        # - destination_simulate
        spec_segment_name = segment_name  # spec_segment_name is segment_name
        choices = \
            run_destination_simulate(
                spec_segment_name,
                choosers,
                persons_merged,
                location_sample_df,
                model_settings,
                skim_dict,
                segment_destination_size_terms,
                chunk_size,
                tracing.extend_trace_label(trace_label, 'simulate.%s' % segment_name))

        choices_list.append(choices)

        # FIXME - want to do this here?
        del location_sample_df
        force_garbage_collect()

    return pd.concat(choices_list) if len(choices_list) > 0 else pd.Series()

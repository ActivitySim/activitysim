# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import los
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core import pipeline

from activitysim.core.util import reindex

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

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

        # do this once so they can request size_terms for various segments (tour_type or purpose)
        land_use = inject.get_table('land_use')
        size_terms = inject.get_injectable('size_terms')
        self.destination_size_terms = \
            tour_destination_size_terms(land_use, size_terms, size_term_selector)

        assert not self.destination_size_terms.isna().any(axis=None)

    # def omnibus_size_terms_df(self):
    #     return self.destination_size_terms

    def dest_size_terms_df(self, segment_name, trace_label):
        # return size terms as df with one column named 'size_term'
        # convenient if creating or merging with alts

        size_terms = self.destination_size_terms[[segment_name]].copy()
        size_terms.columns = ['size_term']

        # FIXME - no point in considering impossible alternatives (where dest size term is zero)
        logger.debug(f"SizeTermCalculator dropping {(~(size_terms.size_term > 0)).sum()} "
                     f"of {len(size_terms)} rows where size_term is zero for {segment_name}")
        size_terms = size_terms[size_terms.size_term > 0]

        if len(size_terms) == 0:
            logger.warning(f"SizeTermCalculator: no zones with non-zero size terms for {segment_name} in {trace_label}")

        return size_terms

    # def dest_size_terms_series(self, segment_name):
    #     # return size terms as as series
    #     # convenient (and no copy overhead) if reindexing and assigning into alts column
    #     return self.destination_size_terms[segment_name]


def _destination_sample(
        spec_segment_name,
        choosers,
        destination_size_terms,
        skims,
        estimator,
        model_settings,
        alt_dest_col_name,
        chunk_size,
        chunk_tag,
        trace_label):

    model_spec = simulate.spec_for_segment(model_settings, spec_id='SAMPLE_SPEC',
                                           segment_name=spec_segment_name, estimator=estimator)

    logger.info("running %s with %d tours", trace_label, len(choosers))

    sample_size = model_settings['SAMPLE_SIZE']
    if config.setting('disable_destination_sampling', False) or (estimator and estimator.want_unsampled_alternatives):
        # FIXME interaction_sample will return unsampled complete alternatives with probs and pick_count
        logger.info("Estimation mode for %s using unsampled alternatives short_circuit_choices" % (trace_label,))
        sample_size = 0

    locals_d = {
        'skims': skims
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    log_alt_losers = config.setting('log_alt_losers', False)

    choices = interaction_sample(
        choosers,
        alternatives=destination_size_terms,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        spec=model_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label)

    # remember person_id in chosen alts so we can merge with persons in subsequent steps
    # (broadcasts person_id onto all alternatives sharing the same tour_id index value)
    choices['person_id'] = choosers.person_id

    return choices


def destination_sample(
        spec_segment_name,
        choosers,
        model_settings,
        network_los,
        destination_size_terms,
        estimator,
        chunk_size, trace_label):

    chunk_tag = 'tour_destination.sample'

    # create wrapper with keys for this lookup
    # the skims will be available under the name "skims" for any @ expressions
    skim_origin_col_name = model_settings['CHOOSER_ORIG_COL_NAME']
    skim_dest_col_name = destination_size_terms.index.name
    # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
    if (skim_origin_col_name == skim_dest_col_name):
        skim_origin_col_name = f'{skim_origin_col_name}_chooser'

    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap(skim_origin_col_name, skim_dest_col_name)

    # the name of the dest column to be returned in choices
    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']

    choices = _destination_sample(
        spec_segment_name,
        choosers,
        destination_size_terms,
        skims,
        estimator,
        model_settings,
        alt_dest_col_name,
        chunk_size, chunk_tag=chunk_tag,
        trace_label=trace_label)

    return choices


# temp column names for presampling
DEST_MAZ = 'dest_MAZ'
DEST_TAZ = 'dest_TAZ'
ORIG_TAZ = 'TAZ'  # likewise a temp, but if already in choosers, we assume we can use it opportunistically


def map_maz_to_taz(s, network_los):
    maz_to_taz = network_los.maz_taz_df[['MAZ', 'TAZ']].set_index('MAZ').TAZ
    return s.map(maz_to_taz)


def aggregate_size_terms(dest_size_terms, network_los):
    #
    # aggregate MAZ_size_terms to TAZ_size_terms
    #

    MAZ_size_terms = dest_size_terms.copy()

    # add crosswalk DEST_TAZ column to MAZ_size_terms
    MAZ_size_terms[DEST_TAZ] = map_maz_to_taz(MAZ_size_terms.index, network_los)

    # aggregate to TAZ
    TAZ_size_terms = MAZ_size_terms.groupby(DEST_TAZ).agg({'size_term': 'sum'})
    TAZ_size_terms[DEST_TAZ] = TAZ_size_terms.index
    assert not TAZ_size_terms['size_term'].isna().any()

    #           size_term
    # dest_TAZ
    # 2              45.0
    # 3              44.0
    # 4              59.0

    # add crosswalk DEST_TAZ column to MAZ_size_terms
    # MAZ_size_terms = MAZ_size_terms.sort_values([DEST_TAZ, 'size_term'])  # maybe helpful for debugging
    MAZ_size_terms = MAZ_size_terms[[DEST_TAZ, 'size_term']].reset_index(drop=False)
    MAZ_size_terms = MAZ_size_terms.sort_values([DEST_TAZ, 'zone_id']).reset_index(drop=True)

    #       zone_id  dest_TAZ  size_term
    # 0        6097         2       10.0
    # 1       16421         2       13.0
    # 2       24251         3       14.0

    # print(f"TAZ_size_terms ({TAZ_size_terms.shape})\n{TAZ_size_terms}")
    # print(f"MAZ_size_terms ({MAZ_size_terms.shape})\n{MAZ_size_terms}")

    return MAZ_size_terms, TAZ_size_terms


def choose_MAZ_for_TAZ(taz_sample, MAZ_size_terms, trace_label):
    """
    Convert taz_sample table with TAZ zone sample choices to a table with a MAZ zone chosen for each TAZ
    choose MAZ probabilistically (proportionally by size_term) from set of MAZ zones in parent TAZ

    Parameters
    ----------
    taz_sample: dataframe with duplicated index <chooser_id_col> and columns: <DEST_TAZ>, prob, pick_count
    MAZ_size_terms: dataframe with duplicated index <chooser_id_col> and columns: zone_id, dest_TAZ, size_term

    Returns
    -------
    dataframe with with duplicated index <chooser_id_col> and columns: <DEST_MAZ>, prob, pick_count
    """

    # print(f"taz_sample\n{taz_sample}")
    #           dest_TAZ      prob  pick_count  person_id
    # tour_id
    # 542963          18  0.004778           1      13243
    # 542963          53  0.004224           2      13243
    # 542963          59  0.008628           1      13243

    trace_hh_id = inject.get_injectable("trace_hh_id", None)
    have_trace_targets = trace_hh_id and tracing.has_trace_targets(taz_sample)
    if have_trace_targets:
        trace_label = tracing.extend_trace_label(trace_label, 'choose_MAZ_for_TAZ')

        CHOOSER_ID = taz_sample.index.name  # zone_id for tours, but person_id for location choice
        assert CHOOSER_ID is not None

        # write taz choices, pick_counts, probs
        trace_targets = tracing.trace_targets(taz_sample)
        tracing.trace_df(taz_sample[trace_targets],
                         label=tracing.extend_trace_label(trace_label, 'taz_sample'),
                         transpose=False)

    # redupe taz_sample[[DEST_TAZ, 'prob']] using pick_count to repeat rows
    taz_choices = taz_sample[[DEST_TAZ, 'prob']].reset_index(drop=False)
    taz_choices = taz_choices.reindex(taz_choices.index.repeat(taz_sample.pick_count)).reset_index(drop=True)
    taz_choices = taz_choices.rename(columns={'prob': 'TAZ_prob'})

    # print(f"taz_choices\n{taz_choices}")
    #        tour_id  dest_TAZ  TAZ_prob
    # 0       542963        18  0.004778
    # 1       542963        53  0.004224
    # 2       542963        53  0.004224
    # 3       542963        59  0.008628

    # print(f"MAZ_size_terms\n{MAZ_size_terms}")
    #       zone_id  dest_TAZ  size_term
    # 0        6097         2      7.420
    # 1       16421         2      9.646
    # 2       24251         2     10.904

    # just to make it clear we are siloing choices by chooser_id
    chooser_id_col = taz_sample.index.name  # should be canonical chooser index name (e.g. 'person_id')

    # for random_for_df, we need df with de-duplicated chooser canonical index
    chooser_df = pd.DataFrame(index=taz_sample.index[~taz_sample.index.duplicated()])
    num_choosers = len(chooser_df)
    assert chooser_df.index.name == chooser_id_col

    # to make choices, <taz_sample_size> rands for each chooser (one rand for each sampled TAZ)
    # taz_sample_size will be model_settings['SAMPLE_SIZE'] samples, except if we are estimating
    taz_sample_size = taz_choices.groupby(chooser_id_col)[DEST_TAZ].count().max()

    # taz_choices index values should be contiguous
    assert (taz_choices[chooser_id_col] == np.repeat(chooser_df.index, taz_sample_size)).all()

    # we need to choose a MAZ for each DEST_TAZ choice
    # probability of choosing MAZ based on MAZ size_term fraction of TAZ total
    # there will be a different set (and number) of candidate MAZs for each TAZ
    # (preserve index, which will have duplicates as result of join)
    # maz_sizes.index is the integer offset into taz_choices of the taz for which the maz_size row is a candidate)
    maz_sizes = pd.merge(taz_choices[[chooser_id_col, DEST_TAZ]].reset_index(),
                         MAZ_size_terms, how='left', on=DEST_TAZ).set_index('index')

    #         tour_id  dest_TAZ  zone_id  size_term
    # index
    # 0        542963        18      498     12.130
    # 0        542963        18     7696     18.550
    # 0        542963        18    15431      8.678
    # 0        542963        18    21429     29.938
    # 1        542963        53    17563     34.252

    if have_trace_targets:
        # write maz_sizes: maz_sizes[index,tour_id,dest_TAZ,zone_id,size_term]

        maz_sizes_trace_targets = tracing.trace_targets(maz_sizes, slicer=CHOOSER_ID)
        trace_maz_sizes = maz_sizes[maz_sizes_trace_targets]
        tracing.trace_df(trace_maz_sizes,
                         label=tracing.extend_trace_label(trace_label, 'maz_sizes'),
                         transpose=False)

    # number of DEST_TAZ candidates per chooser
    maz_counts = maz_sizes.groupby(maz_sizes.index).size().values

    # max number of MAZs for any TAZ
    max_maz_count = maz_counts.max()

    # offsets of the first and last rows of each chooser in sparse interaction_utilities
    last_row_offsets = maz_counts.cumsum()
    first_row_offsets = np.insert(last_row_offsets[:-1], 0, 0)

    # repeat the row offsets once for each dummy utility to insert
    # (we want to insert dummy utilities at the END of the list of alternative utilities)
    # inserts is a list of the indices at which we want to do the insertions
    inserts = np.repeat(last_row_offsets, max_maz_count - maz_counts)

    # insert zero filler to pad each alternative set to same size
    padded_maz_sizes = np.insert(maz_sizes.size_term.values, inserts, 0.0).reshape(-1, max_maz_count)

    # prob array with one row TAZ_choice, one column per alternative
    row_sums = padded_maz_sizes.sum(axis=1)
    maz_probs = np.divide(padded_maz_sizes, row_sums.reshape(-1, 1))
    assert maz_probs.shape == (num_choosers * taz_sample_size, max_maz_count)

    rands = pipeline.get_rn_generator().random_for_df(chooser_df, n=taz_sample_size)
    rands = rands.reshape(-1, 1)
    assert len(rands) == num_choosers * taz_sample_size
    assert len(rands) == maz_probs.shape[0]

    # make choices
    # positions is array with the chosen alternative represented as a column index in probs
    # which is an integer between zero and max_maz_count
    positions = np.argmax((maz_probs.cumsum(axis=1) - rands) > 0.0, axis=1)

    # shouldn't have chosen any of the dummy pad positions
    assert (positions < maz_counts).all()

    taz_choices[DEST_MAZ] = maz_sizes['zone_id'].take(positions + first_row_offsets)
    taz_choices['MAZ_prob'] = maz_probs[np.arange(maz_probs.shape[0]), positions]
    taz_choices['prob'] = taz_choices['TAZ_prob'] * taz_choices['MAZ_prob']

    if have_trace_targets:

        taz_choices_trace_targets = tracing.trace_targets(taz_choices, slicer=CHOOSER_ID)
        trace_taz_choices_df = taz_choices[taz_choices_trace_targets]
        tracing.trace_df(trace_taz_choices_df,
                         label=tracing.extend_trace_label(trace_label, 'taz_choices'),
                         transpose=False)

        lhs_df = trace_taz_choices_df[[CHOOSER_ID, DEST_TAZ]]
        alt_dest_columns = [f'dest_maz_{c}' for c in range(max_maz_count)]

        # following the same logic as the full code, but for trace cutout
        trace_maz_counts = maz_counts[taz_choices_trace_targets]
        trace_last_row_offsets = maz_counts[taz_choices_trace_targets].cumsum()
        trace_inserts = np.repeat(trace_last_row_offsets, max_maz_count - trace_maz_counts)

        # trace dest_maz_alts
        padded_maz_sizes = np.insert(trace_maz_sizes[CHOOSER_ID].values, trace_inserts, 0.0).reshape(-1, max_maz_count)
        df = pd.DataFrame(data=padded_maz_sizes,
                          columns=alt_dest_columns, index=trace_taz_choices_df.index)
        df = pd.concat([lhs_df, df], axis=1)
        tracing.trace_df(df, label=tracing.extend_trace_label(trace_label, 'dest_maz_alts'), transpose=False)

        # trace dest_maz_size_terms
        padded_maz_sizes = np.insert(trace_maz_sizes['size_term'].values, trace_inserts, 0.0).reshape(-1, max_maz_count)
        df = pd.DataFrame(data=padded_maz_sizes,
                          columns=alt_dest_columns, index=trace_taz_choices_df.index)
        df = pd.concat([lhs_df, df], axis=1)
        tracing.trace_df(df, label=tracing.extend_trace_label(trace_label, 'dest_maz_size_terms'), transpose=False)

        # trace dest_maz_probs
        df = pd.DataFrame(data=maz_probs[taz_choices_trace_targets],
                          columns=alt_dest_columns, index=trace_taz_choices_df.index)
        df = pd.concat([lhs_df, df], axis=1)
        df['rand'] = rands[taz_choices_trace_targets]
        tracing.trace_df(df, label=tracing.extend_trace_label(trace_label, 'dest_maz_probs'), transpose=False)

    taz_choices = taz_choices.drop(columns=['TAZ_prob', 'MAZ_prob'])
    taz_choices = taz_choices.groupby([chooser_id_col, DEST_MAZ]).agg(prob=('prob', 'max'),
                                                                      pick_count=('prob', 'count'))

    taz_choices.reset_index(level=DEST_MAZ, inplace=True)

    return taz_choices


def destination_presample(
        spec_segment_name,
        choosers,
        model_settings,
        network_los,
        destination_size_terms,
        estimator,
        chunk_size, trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'presample')
    chunk_tag = 'tour_destination.presample'

    logger.info(f"{trace_label} location_presample")

    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']
    assert DEST_TAZ != alt_dest_col_name

    MAZ_size_terms, TAZ_size_terms = aggregate_size_terms(destination_size_terms, network_los)

    orig_maz = model_settings['CHOOSER_ORIG_COL_NAME']
    assert orig_maz in choosers
    if ORIG_TAZ not in choosers:
        choosers[ORIG_TAZ] = map_maz_to_taz(choosers[orig_maz], network_los)

    # create wrapper with keys for this lookup - in this case there is a HOME_TAZ in the choosers
    # and a DEST_TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_skim_dict('taz')
    skims = skim_dict.wrap(ORIG_TAZ, DEST_TAZ)

    taz_sample = _destination_sample(
        spec_segment_name,
        choosers,
        TAZ_size_terms,
        skims,
        estimator,
        model_settings,
        DEST_TAZ,
        chunk_size, chunk_tag=chunk_tag,
        trace_label=trace_label)

    # choose a MAZ for each DEST_TAZ choice, choice probability based on MAZ size_term fraction of TAZ total
    maz_choices = choose_MAZ_for_TAZ(taz_sample, MAZ_size_terms, trace_label)

    assert DEST_MAZ in maz_choices
    maz_choices = maz_choices.rename(columns={DEST_MAZ: alt_dest_col_name})

    return maz_choices


def run_destination_sample(
        spec_segment_name,
        tours,
        persons_merged,
        model_settings,
        network_los,
        destination_size_terms,
        estimator,
        chunk_size, trace_label):

    # FIXME - MEMORY HACK - only include columns actually used in spec (omit them pre-merge)
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    persons_merged = persons_merged[[c for c in persons_merged.columns if c in chooser_columns]]
    tours = tours[[c for c in tours.columns if c in chooser_columns or c == 'person_id']]
    choosers = pd.merge(tours, persons_merged, left_on='person_id', right_index=True, how='left')

    # interaction_sample requires that choosers.index.is_monotonic_increasing
    if not choosers.index.is_monotonic_increasing:
        logger.debug(f"run_destination_sample {trace_label} sorting choosers because not monotonic_increasing")
        choosers = choosers.sort_index()

    # by default, enable presampling for multizone systems, unless they disable it in settings file
    pre_sample_taz = not (network_los.zone_system == los.ONE_ZONE)
    if pre_sample_taz and not config.setting('want_dest_choice_presampling', True):
        pre_sample_taz = False
        logger.info(f"Disabled destination zone presampling for {trace_label} "
                    f"because 'want_dest_choice_presampling' setting is False")

    if pre_sample_taz:

        logger.info("Running %s destination_presample with %d tours" % (trace_label, len(tours)))

        choices = destination_presample(
            spec_segment_name,
            choosers,
            model_settings,
            network_los,
            destination_size_terms,
            estimator,
            chunk_size, trace_label)

    else:
        choices = destination_sample(
            spec_segment_name,
            choosers,
            model_settings,
            network_los,
            destination_size_terms,
            estimator,
            chunk_size, trace_label)

    # remember person_id in chosen alts so we can merge with persons in subsequent steps
    # (broadcasts person_id onto all alternatives sharing the same tour_id index value)
    choices['person_id'] = tours.person_id

    return choices


def run_destination_logsums(
        tour_purpose,
        persons_merged,
        destination_sample,
        model_settings,
        network_los,
        chunk_size,
        trace_label):
    """
    add logsum column to existing tour_destination_sample table

    logsum is calculated by running the mode_choice model for each sample (person, dest_zone_id) pair
    in destination_sample, and computing the logsum of all the utilities

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

    chunk_tag = 'tour_destination.logsums'

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
        network_los,
        chunk_size,
        chunk_tag,
        trace_label)

    destination_sample['mode_choice_logsum'] = logsums

    return destination_sample


def run_destination_simulate(
        spec_segment_name,
        tours,
        persons_merged,
        destination_sample,
        want_logsums,
        model_settings,
        network_los,
        destination_size_terms,
        estimator,
        chunk_size, trace_label):
    """
    run destination_simulate on tour_destination_sample
    annotated with mode_choice logsum to select a destination from sample alternatives
    """
    chunk_tag = 'tour_destination.simulate'

    model_spec = simulate.spec_for_segment(model_settings, spec_id='SPEC',
                                           segment_name=spec_segment_name, estimator=estimator)

    # FIXME - MEMORY HACK - only include columns actually used in spec (omit them pre-merge)
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    persons_merged = persons_merged[[c for c in persons_merged.columns if c in chooser_columns]]
    tours = tours[[c for c in tours.columns if c in chooser_columns or c == 'person_id']]
    choosers = pd.merge(tours, persons_merged, left_on='person_id', right_index=True, how='left')

    # interaction_sample requires that choosers.index.is_monotonic_increasing
    if not choosers.index.is_monotonic_increasing:
        logger.debug(f"run_destination_simulate {trace_label} sorting choosers because not monotonic_increasing")
        choosers = choosers.sort_index()

    if estimator:
        estimator.write_choosers(choosers)

    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']
    origin_col_name = model_settings['CHOOSER_ORIG_COL_NAME']

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge size_terms column into alt sample list
    destination_sample['size_term'] = \
        reindex(destination_size_terms.size_term, destination_sample[alt_dest_col_name])

    tracing.dump_df(DUMP, destination_sample, trace_label, 'alternatives')

    constants = config.get_model_constants(model_settings)

    logger.info("Running tour_destination_simulate with %d persons", len(choosers))

    # create wrapper with keys for this lookup - in this case there is a home_zone_id in the choosers
    # and a zone_id in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap(origin_col_name, alt_dest_col_name)

    locals_d = {
        'skims': skims,
    }
    if constants is not None:
        locals_d.update(constants)

    tracing.dump_df(DUMP, choosers, trace_label, 'choosers')

    log_alt_losers = config.setting('log_alt_losers', False)

    choices = interaction_sample_simulate(
        choosers,
        destination_sample,
        spec=model_spec,
        choice_column=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        want_logsums=want_logsums,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size, chunk_tag=chunk_tag,
        trace_label=trace_label,
        trace_choice_name='destination',
        estimator=estimator)

    if not want_logsums:
        # for consistency, always return a dataframe with canonical column name
        assert isinstance(choices, pd.Series)
        choices = choices.to_frame('choice')

    return choices


def run_tour_destination(
        tours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        chunk_size, trace_hh_id, trace_label):

    size_term_calculator = SizeTermCalculator(model_settings['SIZE_TERM_SELECTOR'])

    # maps segment names to compact (integer) ids
    segments = model_settings['SEGMENTS']

    chooser_segment_column = model_settings.get('CHOOSER_SEGMENT_COLUMN_NAME', None)
    if chooser_segment_column is None:
        assert len(segments) == 1, \
            f"CHOOSER_SEGMENT_COLUMN_NAME not specified in model_settings to slice SEGMENTS: {segments}"

    choices_list = []
    sample_list = []
    for segment_name in segments:

        segment_trace_label = tracing.extend_trace_label(trace_label, segment_name)

        if chooser_segment_column is not None:
            choosers = tours[tours[chooser_segment_column] == segment_name]
        else:
            choosers = tours.copy()

        # Note: size_term_calculator omits zones with impossible alternatives (where dest size term is zero)
        segment_destination_size_terms = size_term_calculator.dest_size_terms_df(segment_name, segment_trace_label)

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
                network_los,
                segment_destination_size_terms,
                estimator,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(segment_trace_label, 'sample'))

        # - destination_logsums
        tour_purpose = segment_name  # tour_purpose is segment_name
        location_sample_df = \
            run_destination_logsums(
                tour_purpose,
                persons_merged,
                location_sample_df,
                model_settings,
                network_los,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(segment_trace_label, 'logsums'))

        # - destination_simulate
        spec_segment_name = segment_name  # spec_segment_name is segment_name
        choices = \
            run_destination_simulate(
                spec_segment_name,
                choosers,
                persons_merged,
                destination_sample=location_sample_df,
                want_logsums=want_logsums,
                model_settings=model_settings,
                network_los=network_los,
                destination_size_terms=segment_destination_size_terms,
                estimator=estimator,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(segment_trace_label, 'simulate'))

        choices_list.append(choices)

        if want_sample_table:
            # FIXME - sample_table
            location_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
            sample_list.append(location_sample_df)
        else:
            # del this so we dont hold active reference to it while run_location_sample is creating its replacement
            del location_sample_df

    if len(choices_list) > 0:
        choices_df = pd.concat(choices_list)
    else:
        # this will only happen with small samples (e.g. singleton) with no (e.g.) school segs
        logger.warning("%s no choices", trace_label)
        choices_df = pd.DataFrame(columns=['choice', 'logsum'])

    if len(sample_list) > 0:
        save_sample_df = pd.concat(sample_list)
    else:
        # this could happen either with small samples as above, or if no saved sample desired
        save_sample_df = None

    return choices_df, save_sample_df

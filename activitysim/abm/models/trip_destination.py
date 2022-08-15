# ActivitySim
# See full license in LICENSE.txt.
import logging
from builtins import range

import numpy as np
import pandas as pd

from activitysim.abm.models.util.trip import (
    cleanup_failed_trips,
    flag_failed_trip_leg_mates,
)
from activitysim.abm.tables.size_terms import tour_destination_size_terms
from activitysim.core import (
    assign,
    chunk,
    config,
    expressions,
    inject,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.interaction_sample import interaction_sample
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.pathbuilder import TransitVirtualPathBuilder
from activitysim.core.skim_dictionary import DataFrameMatrix
from activitysim.core.tracing import print_elapsed_time
from activitysim.core.util import assign_in_place, reindex

from .util import estimation

logger = logging.getLogger(__name__)

NO_DESTINATION = -1

# TRIP_ORIG_TAZ = 'TAZ'
ALT_DEST_TAZ = "ALT_DEST_TAZ"
# PRIMARY_DEST_TAZ = 'PRIMARY_DEST_TAZ'
# DEST_MAZ = 'dest_maz'


def _destination_sample(
    primary_purpose,
    trips,
    alternatives,
    model_settings,
    size_term_matrix,
    skims,
    alt_dest_col_name,
    estimator,
    chunk_size,
    chunk_tag,
    trace_label,
):
    """

    Note: trips with no viable destination receive no sample rows
    (because we call interaction_sample with allow_zero_probs=True)
    All other trips will have one or more rows with pick_count summing to sample_size

    returns
        choices: pandas.DataFrame

               alt_dest      prob  pick_count
    trip_id
    102829169      2898  0.002333           1
    102829169      2901  0.004976           1
    102829169      3193  0.002628           1
    """

    spec = simulate.spec_for_segment(
        model_settings,
        spec_id="DESTINATION_SAMPLE_SPEC",
        segment_name=primary_purpose,
        estimator=estimator,
    )

    sample_size = model_settings["SAMPLE_SIZE"]
    if config.setting("disable_destination_sampling", False) or (
        estimator and estimator.want_unsampled_alternatives
    ):
        # FIXME interaction_sample will return unsampled complete alternatives with probs and pick_count
        logger.info(
            "Estimation mode for %s using unsampled alternatives short_circuit_choices"
            % (trace_label,)
        )
        sample_size = 0

    locals_dict = config.get_model_constants(model_settings).copy()

    # size_terms of destination zones are purpose-specific, and trips have various purposes
    # so the relevant size_term for each interaction_sample row
    # cannot be determined until after choosers are joined with alternatives
    # (unless we iterate over trip.purpose - which we could, though we are already iterating over trip_num)
    # so, instead, expressions determine row-specific size_term by a call to: size_terms.get(df.alt_dest, df.purpose)
    locals_dict.update({"size_terms": size_term_matrix})
    locals_dict.update(skims)

    log_alt_losers = config.setting("log_alt_losers", False)

    choices = interaction_sample(
        choosers=trips,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        allow_zero_probs=True,
        spec=spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
    )

    return choices


def destination_sample(
    primary_purpose,
    trips,
    alternatives,
    model_settings,
    size_term_matrix,
    skim_hotel,
    estimator,
    chunk_size,
    trace_label,
):

    chunk_tag = "trip_destination.sample"

    skims = skim_hotel.sample_skims(presample=False)
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    choices = _destination_sample(
        primary_purpose,
        trips,
        alternatives,
        model_settings,
        size_term_matrix,
        skims,
        alt_dest_col_name,
        estimator,
        chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
    )

    return choices


def aggregate_size_term_matrix(maz_size_term_matrix, maz_taz):

    df = maz_size_term_matrix.df
    assert ALT_DEST_TAZ not in df

    dest_taz = df.index.map(maz_taz)
    taz_size_term_matrix = df.groupby(dest_taz).sum()

    taz_size_term_matrix = DataFrameMatrix(taz_size_term_matrix)

    return taz_size_term_matrix


def choose_MAZ_for_TAZ(
    taz_sample, MAZ_size_terms, trips, network_los, alt_dest_col_name, trace_label
):
    """
    Convert taz_sample table with TAZ zone sample choices to a table with a MAZ zone chosen for each TAZ
    choose MAZ probabilistically (proportionally by size_term) from set of MAZ zones in parent TAZ

    Parameters
    ----------
    taz_sample: dataframe with duplicated index <chooser_id_col> and columns: <alt_dest_col_name>, prob, pick_count
    MAZ_size_terms: dataframe with duplicated index <chooser_id_col> and columns: zone_id, dest_TAZ, size_term

    Returns
    -------
    dataframe with with duplicated index <chooser_id_col> and columns: <alt_dest_col_name>, prob, pick_count
    """

    if len(taz_sample) == 0:
        # it can happen that all trips have no viable destinations (and so are dropped from the sample)
        # in which case we can just return the empty taz_sample, since it has the same columns
        return taz_sample.copy()

    # we had to use alt_dest_col_name as specified in model_settings for interaction_sample
    # because expressions reference it to look up size_terms by trip purpose
    DEST_MAZ = alt_dest_col_name
    DEST_TAZ = f"{alt_dest_col_name}_TAZ"

    taz_sample.rename(columns={alt_dest_col_name: DEST_TAZ}, inplace=True)

    trace_hh_id = inject.get_injectable("trace_hh_id", None)
    have_trace_targets = trace_hh_id and tracing.has_trace_targets(taz_sample)
    if have_trace_targets:
        trace_label = tracing.extend_trace_label(trace_label, "choose_MAZ_for_TAZ")

        # write taz choices, pick_counts, probs
        trace_targets = tracing.trace_targets(taz_sample)
        tracing.trace_df(
            taz_sample[trace_targets],
            label=tracing.extend_trace_label(trace_label, "taz_sample"),
            transpose=False,
        )

    # print(f"taz_sample\n{taz_sample}")
    #            alt_dest_TAZ      prob  pick_count
    # trip_id
    # 4343721              12  0.000054           1
    # 4343721              20  0.001864           2

    taz_choices = taz_sample[[DEST_TAZ, "prob"]].reset_index(drop=False)
    taz_choices = taz_choices.reindex(
        taz_choices.index.repeat(taz_sample.pick_count)
    ).reset_index(drop=True)
    taz_choices = taz_choices.rename(columns={"prob": "TAZ_prob"})

    # print(f"taz_choices\n{taz_choices}")
    #         trip_id  alt_dest_TAZ      prob
    # 0       4343721            12  0.000054
    # 1       4343721            20  0.001864
    # 2       4343721            20  0.001864

    # print(f"MAZ_size_terms\n{MAZ_size_terms.df}")
    #           work  escort  shopping  eatout  othmaint  social  othdiscr   univ
    # alt_dest
    # 2         31.0   9.930     0.042   0.258     0.560   0.520    10.856  0.042
    # 3          0.0   3.277     0.029   0.000     0.029   0.029     7.308  0.029
    # 4          0.0   1.879     0.023   0.000     0.023   0.023     5.796  0.023

    # just to make it clear we are siloing choices by chooser_id
    chooser_id_col = (
        taz_sample.index.name
    )  # should be canonical chooser index name (e.g. 'trip_id')

    # for random_for_df, we need df with de-duplicated chooser canonical index
    chooser_df = pd.DataFrame(index=taz_sample.index[~taz_sample.index.duplicated()])
    num_choosers = len(chooser_df)
    assert chooser_df.index.name == chooser_id_col

    # to make choices, <taz_sample_size> rands for each chooser (one rand for each sampled TAZ)
    # taz_sample_size will be model_settings['SAMPLE_SIZE'] samples, except if we are estimating
    taz_sample_size = taz_choices.groupby(chooser_id_col)[DEST_TAZ].count().max()

    # taz_choices index values should be contiguous
    assert (
        taz_choices[chooser_id_col] == np.repeat(chooser_df.index, taz_sample_size)
    ).all()

    # we need to choose a MAZ for each DEST_TAZ choice
    # probability of choosing MAZ based on MAZ size_term fraction of TAZ total
    # there will be a different set (and number) of candidate MAZs for each TAZ
    # (preserve index, which will have duplicates as result of join)

    maz_taz = network_los.maz_taz_df[["MAZ", "TAZ"]].rename(
        columns={"TAZ": DEST_TAZ, "MAZ": DEST_MAZ}
    )
    maz_sizes = pd.merge(
        taz_choices[[chooser_id_col, DEST_TAZ]].reset_index(),
        maz_taz,
        how="left",
        on=DEST_TAZ,
    ).set_index("index")

    purpose = maz_sizes["trip_id"].map(trips.purpose)  # size term varies by purpose
    maz_sizes["size_term"] = MAZ_size_terms.get(maz_sizes[DEST_MAZ], purpose)

    # print(f"maz_sizes\n{maz_sizes}")
    #          trip_id  alt_dest_TAZ  alt_dest  size_term
    # index
    # 0        4343721            12      3445      0.019
    # 0        4343721            12     11583      0.017
    # 0        4343721            12     21142      0.020

    if have_trace_targets:
        # write maz_sizes: maz_sizes[index,trip_id,dest_TAZ,zone_id,size_term]
        maz_sizes_trace_targets = tracing.trace_targets(maz_sizes, slicer="trip_id")
        trace_maz_sizes = maz_sizes[maz_sizes_trace_targets]
        tracing.trace_df(
            trace_maz_sizes,
            label=tracing.extend_trace_label(trace_label, "maz_sizes"),
            transpose=False,
        )

    # number of DEST_TAZ candidates per chooser
    maz_counts = maz_sizes.groupby(maz_sizes.index).size().values
    # print(maz_counts)

    # max number of MAZs for any TAZ
    max_maz_count = maz_counts.max()
    # print(f"max_maz_count {max_maz_count}")

    # offsets of the first and last rows of each chooser in sparse interaction_utilities
    last_row_offsets = maz_counts.cumsum()
    first_row_offsets = np.insert(last_row_offsets[:-1], 0, 0)

    # repeat the row offsets once for each dummy utility to insert
    # (we want to insert dummy utilities at the END of the list of alternative utilities)
    # inserts is a list of the indices at which we want to do the insertions
    inserts = np.repeat(last_row_offsets, max_maz_count - maz_counts)

    # insert zero filler to pad each alternative set to same size
    padded_maz_sizes = np.insert(maz_sizes.size_term.values, inserts, 0.0)
    padded_maz_sizes = padded_maz_sizes.reshape(-1, max_maz_count)

    # prob array with one row TAZ_choice, one column per alternative
    row_sums = padded_maz_sizes.sum(axis=1)
    maz_probs = np.divide(padded_maz_sizes, row_sums.reshape(-1, 1))
    assert maz_probs.shape == (num_choosers * taz_sample_size, max_maz_count)

    rands = (
        pipeline.get_rn_generator()
        .random_for_df(chooser_df, n=taz_sample_size)
        .reshape(-1, 1)
    )
    assert len(rands) == num_choosers * taz_sample_size
    assert len(rands) == maz_probs.shape[0]

    # make choices
    # positions is array with the chosen alternative represented as a column index in probs
    # which is an integer between zero and max_maz_count
    positions = np.argmax((maz_probs.cumsum(axis=1) - rands) > 0.0, axis=1)

    # shouldn't have chosen any of the dummy pad positions
    assert (positions < maz_counts).all()

    taz_choices[DEST_MAZ] = maz_sizes[DEST_MAZ].take(positions + first_row_offsets)
    taz_choices["MAZ_prob"] = maz_probs[np.arange(maz_probs.shape[0]), positions]
    taz_choices["prob"] = taz_choices["TAZ_prob"] * taz_choices["MAZ_prob"]

    if have_trace_targets:

        taz_choices_trace_targets = tracing.trace_targets(taz_choices, slicer="trip_id")
        trace_taz_choices_df = taz_choices[taz_choices_trace_targets]
        tracing.trace_df(
            trace_taz_choices_df,
            label=tracing.extend_trace_label(trace_label, "taz_choices"),
            transpose=False,
        )

        lhs_df = trace_taz_choices_df[["trip_id", DEST_TAZ]]
        alt_dest_columns = [f"dest_maz_{c}" for c in range(max_maz_count)]

        # following the same logic as the full code, but for trace cutout
        trace_maz_counts = maz_counts[taz_choices_trace_targets]
        trace_last_row_offsets = maz_counts[taz_choices_trace_targets].cumsum()
        trace_inserts = np.repeat(
            trace_last_row_offsets, max_maz_count - trace_maz_counts
        )

        # trace dest_maz_alts
        padded_maz_sizes = np.insert(
            trace_maz_sizes[DEST_MAZ].values, trace_inserts, 0.0
        ).reshape(-1, max_maz_count)
        df = pd.DataFrame(
            data=padded_maz_sizes,
            columns=alt_dest_columns,
            index=trace_taz_choices_df.index,
        )
        df = pd.concat([lhs_df, df], axis=1)
        tracing.trace_df(
            df,
            label=tracing.extend_trace_label(trace_label, "dest_maz_alts"),
            transpose=False,
        )

        # trace dest_maz_size_terms
        padded_maz_sizes = np.insert(
            trace_maz_sizes["size_term"].values, trace_inserts, 0.0
        ).reshape(-1, max_maz_count)
        df = pd.DataFrame(
            data=padded_maz_sizes,
            columns=alt_dest_columns,
            index=trace_taz_choices_df.index,
        )
        df = pd.concat([lhs_df, df], axis=1)
        tracing.trace_df(
            df,
            label=tracing.extend_trace_label(trace_label, "dest_maz_size_terms"),
            transpose=False,
        )

        # trace dest_maz_probs
        df = pd.DataFrame(
            data=maz_probs[taz_choices_trace_targets],
            columns=alt_dest_columns,
            index=trace_taz_choices_df.index,
        )
        df = pd.concat([lhs_df, df], axis=1)
        df["rand"] = rands[taz_choices_trace_targets]
        tracing.trace_df(
            df,
            label=tracing.extend_trace_label(trace_label, "dest_maz_probs"),
            transpose=False,
        )

    taz_choices = taz_choices.drop(columns=["TAZ_prob", "MAZ_prob"])
    taz_choices = taz_choices.groupby([chooser_id_col, DEST_MAZ]).agg(
        prob=("prob", "max"), pick_count=("prob", "count")
    )

    taz_choices.reset_index(level=DEST_MAZ, inplace=True)

    return taz_choices


def destination_presample(
    primary_purpose,
    trips,
    alternatives,
    model_settings,
    size_term_matrix,
    skim_hotel,
    network_los,
    estimator,
    chunk_size,
    trace_hh_id,
    trace_label,
):

    trace_label = tracing.extend_trace_label(trace_label, "presample")
    chunk_tag = "trip_destination.presample"  # distinguish from trip_destination.sample

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]
    maz_taz = network_los.maz_taz_df[["MAZ", "TAZ"]].set_index("MAZ").TAZ

    TAZ_size_term_matrix = aggregate_size_term_matrix(size_term_matrix, maz_taz)

    TRIP_ORIGIN = model_settings["TRIP_ORIGIN"]
    PRIMARY_DEST = model_settings["PRIMARY_DEST"]
    trips_taz = trips.copy()

    trips_taz[TRIP_ORIGIN] = trips_taz[TRIP_ORIGIN].map(maz_taz)
    trips_taz[PRIMARY_DEST] = trips_taz[PRIMARY_DEST].map(maz_taz)

    # alternatives is just an empty dataframe indexed by maz with index name <alt_dest_col_name>
    # but logically, we are aggregating so lets do it, as there is no particular gain in being clever
    alternatives = alternatives.groupby(alternatives.index.map(maz_taz)).sum()

    # # i did this but after changing alt_dest_col_name to 'trip_dest' it
    # # shouldn't be needed anymore
    # alternatives.index.name = ALT_DEST_TAZ

    skims = skim_hotel.sample_skims(presample=True)

    taz_sample = _destination_sample(
        primary_purpose,
        trips_taz,
        alternatives,
        model_settings,
        TAZ_size_term_matrix,
        skims,
        alt_dest_col_name,
        estimator,
        chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
    )

    # choose a MAZ for each DEST_TAZ choice, choice probability based on MAZ size_term fraction of TAZ total
    maz_sample = choose_MAZ_for_TAZ(
        taz_sample, size_term_matrix, trips, network_los, alt_dest_col_name, trace_label
    )

    assert alt_dest_col_name in maz_sample

    return maz_sample


def trip_destination_sample(
    primary_purpose,
    trips,
    alternatives,
    model_settings,
    size_term_matrix,
    skim_hotel,
    estimator,
    chunk_size,
    trace_hh_id,
    trace_label,
):
    """

    Returns
    -------
    destination_sample: pandas.dataframe
        choices_df from interaction_sample with (up to) sample_size alts for each chooser row
        index (non unique) is trip_id from trips (duplicated for each alt)
        and columns dest_zone_id, prob, and pick_count

        dest_zone_id: int
            alt identifier from alternatives[<alt_col_name>]
        prob: float
            the probability of the chosen alternative
        pick_count : int
            number of duplicate picks for chooser, alt
    """
    trace_label = tracing.extend_trace_label(trace_label, "sample")

    assert len(trips) > 0
    assert len(alternatives) > 0

    # by default, enable presampling for multizone systems, unless they disable it in settings file
    network_los = inject.get_injectable("network_los")
    pre_sample_taz = network_los.zone_system != los.ONE_ZONE
    if pre_sample_taz and not config.setting("want_dest_choice_presampling", True):
        pre_sample_taz = False
        logger.info(
            f"Disabled destination zone presampling for {trace_label} "
            f"because 'want_dest_choice_presampling' setting is False"
        )

    if pre_sample_taz:

        logger.info(
            "Running %s trip_destination_presample with %d trips"
            % (trace_label, len(trips))
        )

        choices = destination_presample(
            primary_purpose,
            trips,
            alternatives,
            model_settings,
            size_term_matrix,
            skim_hotel,
            network_los,
            estimator,
            chunk_size,
            trace_hh_id,
            trace_label,
        )

    else:
        choices = destination_sample(
            primary_purpose,
            trips,
            alternatives,
            model_settings,
            size_term_matrix,
            skim_hotel,
            estimator,
            chunk_size,
            trace_label,
        )

    return choices


def compute_ood_logsums(
    choosers,
    logsum_settings,
    nest_spec,
    logsum_spec,
    od_skims,
    locals_dict,
    chunk_size,
    trace_label,
    chunk_tag,
):
    """
    Compute one (of two) out-of-direction logsums for destination alternatives

    Will either be trip_origin -> alt_dest or alt_dest -> primary_dest
    """

    locals_dict.update(od_skims)

    # if preprocessor contains tvpb logsums term, `pathbuilder.get_tvpb_logsum()`
    # will get called before a ChunkSizers class object has been instantiated,
    # causing pathbuilder to throw an error at L815 due to the assert statement
    # in `chunk.chunk_log()` at chunk.py L927. To avoid failing this assertion,
    # the preprocessor must be called from within a "null chunker" as follows:
    with chunk.chunk_log(
        tracing.extend_trace_label(trace_label, "annotate_preprocessor"), base=True
    ):
        expressions.annotate_preprocessors(
            choosers, locals_dict, od_skims, logsum_settings, trace_label
        )

    logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=od_skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        chunk_tag=chunk_tag,
    )

    assert logsums.index.equals(choosers.index)

    # FIXME not strictly necessary, but would make trace files more legible?
    # logsums = logsums.replace(-np.inf, -999)

    return logsums


def compute_logsums(
    primary_purpose,
    trips,
    destination_sample,
    tours_merged,
    model_settings,
    skim_hotel,
    chunk_size,
    trace_label,
):
    """
    Calculate mode choice logsums using the same recipe as for trip_mode_choice, but do it twice
    for each alternative since we need out-of-direction logsum
    (i.e . origin to alt_dest, and alt_dest to half-tour destination)

    Returns
    -------
        adds od_logsum and dp_logsum columns to trips (in place)
    """
    trace_label = tracing.extend_trace_label(trace_label, "compute_logsums")
    logger.info("Running %s with %d samples", trace_label, destination_sample.shape[0])

    # chunk usage is uniform so better to combine
    chunk_tag = "trip_destination.compute_logsums"

    # FIXME should pass this in?
    network_los = inject.get_injectable("network_los")

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips, tours_merged, left_on="tour_id", right_index=True, how="left"
    )
    assert trips_merged.index.equals(trips.index)

    # - choosers - merge destination_sample and trips_merged
    # re/set index because pandas merge does not preserve left index if it has duplicate values!
    choosers = pd.merge(
        destination_sample,
        trips_merged.reset_index(),
        left_index=True,
        right_on="trip_id",
        how="left",
        suffixes=("", "_r"),
    ).set_index("trip_id")
    assert choosers.index.equals(destination_sample.index)

    logsum_settings = config.read_model_settings(model_settings["LOGSUM_SETTINGS"])
    coefficients = simulate.get_segment_coefficients(logsum_settings, primary_purpose)

    nest_spec = config.get_logit_model_settings(logsum_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)

    logsum_spec = simulate.read_model_spec(file_name=logsum_settings["SPEC"])
    logsum_spec = simulate.eval_coefficients(logsum_spec, coefficients, estimator=None)

    locals_dict = {}
    locals_dict.update(config.get_model_constants(logsum_settings))

    # coefficients can appear in expressions
    locals_dict.update(coefficients)

    skims = skim_hotel.logsum_skims()
    if network_los.zone_system == los.THREE_ZONE:
        # TVPB constants can appear in expressions
        if logsum_settings.get("use_TVPB_constants", True):
            locals_dict.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

    # - od_logsums
    od_skims = {
        "ORIGIN": model_settings["TRIP_ORIGIN"],
        "DESTINATION": model_settings["ALT_DEST_COL_NAME"],
        "odt_skims": skims["odt_skims"],
        "dot_skims": skims["dot_skims"],
        "od_skims": skims["od_skims"],
    }
    if network_los.zone_system == los.THREE_ZONE:
        od_skims.update(
            {
                "tvpb_logsum_odt": skims["tvpb_logsum_odt"],
                "tvpb_logsum_dot": skims["tvpb_logsum_dot"],
            }
        )
    destination_sample["od_logsum"] = compute_ood_logsums(
        choosers,
        logsum_settings,
        nest_spec,
        logsum_spec,
        od_skims,
        locals_dict,
        chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, "od"),
        chunk_tag=chunk_tag,
    )

    # - dp_logsums
    dp_skims = {
        "ORIGIN": model_settings["ALT_DEST_COL_NAME"],
        "DESTINATION": model_settings["PRIMARY_DEST"],
        "odt_skims": skims["dpt_skims"],
        "dot_skims": skims["pdt_skims"],
        "od_skims": skims["dp_skims"],
    }
    if network_los.zone_system == los.THREE_ZONE:
        dp_skims.update(
            {
                "tvpb_logsum_odt": skims["tvpb_logsum_dpt"],
                "tvpb_logsum_dot": skims["tvpb_logsum_pdt"],
            }
        )

    destination_sample["dp_logsum"] = compute_ood_logsums(
        choosers,
        logsum_settings,
        nest_spec,
        logsum_spec,
        dp_skims,
        locals_dict,
        chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, "dp"),
        chunk_tag=chunk_tag,
    )

    return destination_sample


def trip_destination_simulate(
    primary_purpose,
    trips,
    destination_sample,
    model_settings,
    want_logsums,
    size_term_matrix,
    skim_hotel,
    estimator,
    chunk_size,
    trace_hh_id,
    trace_label,
):
    """
    Chose destination from destination_sample (with od_logsum and dp_logsum columns added)


    Returns
    -------
    choices - pandas.Series
        destination alt chosen
    """
    trace_label = tracing.extend_trace_label(trace_label, "trip_dest_simulate")
    chunk_tag = "trip_destination.simulate"

    spec = simulate.spec_for_segment(
        model_settings,
        spec_id="DESTINATION_SPEC",
        segment_name=primary_purpose,
        estimator=estimator,
    )

    if estimator:
        estimator.write_choosers(trips)

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running trip_destination_simulate with %d trips", len(trips))

    skims = skim_hotel.sample_skims(presample=False)

    locals_dict = config.get_model_constants(model_settings).copy()
    locals_dict.update({"size_terms": size_term_matrix})
    locals_dict.update(skims)

    log_alt_losers = config.setting("log_alt_losers", False)
    destinations = interaction_sample_simulate(
        choosers=trips,
        alternatives=destination_sample,
        spec=spec,
        choice_column=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        want_logsums=want_logsums,
        allow_zero_probs=True,
        zero_prob_choice_val=NO_DESTINATION,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        trace_choice_name="trip_dest",
        estimator=estimator,
    )

    if not want_logsums:
        # for consistency, always return a dataframe with canonical column name
        assert isinstance(destinations, pd.Series)
        destinations = destinations.to_frame("choice")

    if estimator:
        # need to overwrite choices here before any failed choices are suppressed
        estimator.write_choices(destinations.choice)

        destinations.choice = estimator.get_survey_values(
            destinations.choice, "trips", "destination"
        )
        estimator.write_override_choices(destinations.choice)

    # drop any failed zero_prob destinations
    if (destinations.choice == NO_DESTINATION).any():
        # logger.debug("dropping %s failed destinations", (destinations == NO_DESTINATION).sum())
        destinations = destinations[destinations.choice != NO_DESTINATION]

    return destinations


def choose_trip_destination(
    primary_purpose,
    trips,
    alternatives,
    tours_merged,
    model_settings,
    want_logsums,
    want_sample_table,
    size_term_matrix,
    skim_hotel,
    estimator,
    chunk_size,
    trace_hh_id,
    trace_label,
):

    logger.info("choose_trip_destination %s with %d trips", trace_label, trips.shape[0])

    t0 = print_elapsed_time()

    # - trip_destination_sample
    destination_sample = trip_destination_sample(
        primary_purpose=primary_purpose,
        trips=trips,
        alternatives=alternatives,
        model_settings=model_settings,
        size_term_matrix=size_term_matrix,
        skim_hotel=skim_hotel,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    dropped_trips = ~trips.index.isin(destination_sample.index.unique())
    if dropped_trips.any():
        logger.warning(
            "%s trip_destination_sample %s trips "
            "without viable destination alternatives"
            % (trace_label, dropped_trips.sum())
        )
        trips = trips[~dropped_trips]

    t0 = print_elapsed_time("%s.trip_destination_sample" % trace_label, t0)

    if trips.empty:
        return pd.Series(index=trips.index).to_frame("choice"), None

    # - compute logsums
    destination_sample = compute_logsums(
        primary_purpose=primary_purpose,
        trips=trips,
        destination_sample=destination_sample,
        tours_merged=tours_merged,
        model_settings=model_settings,
        skim_hotel=skim_hotel,
        chunk_size=chunk_size,
        trace_label=trace_label,
    )

    t0 = print_elapsed_time("%s.compute_logsums" % trace_label, t0)

    # - trip_destination_simulate
    destinations = trip_destination_simulate(
        primary_purpose=primary_purpose,
        trips=trips,
        destination_sample=destination_sample,
        model_settings=model_settings,
        want_logsums=want_logsums,
        size_term_matrix=size_term_matrix,
        skim_hotel=skim_hotel,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    dropped_trips = ~trips.index.isin(destinations.index)
    if dropped_trips.any():
        logger.warning(
            "%s trip_destination_simulate %s trips "
            "without viable destination alternatives"
            % (trace_label, dropped_trips.sum())
        )

    if want_sample_table:
        # FIXME - sample_table
        destination_sample.set_index(
            model_settings["ALT_DEST_COL_NAME"], append=True, inplace=True
        )
    else:
        destination_sample = None

    t0 = print_elapsed_time("%s.trip_destination_simulate" % trace_label, t0)

    return destinations, destination_sample


class SkimHotel(object):
    def __init__(self, model_settings, network_los, trace_label):

        self.model_settings = model_settings
        self.trace_label = tracing.extend_trace_label(trace_label, "skim_hotel")
        self.network_los = network_los
        self.zone_system = network_los.zone_system

    def sample_skims(self, presample):

        o = self.model_settings["TRIP_ORIGIN"]
        d = self.model_settings["ALT_DEST_COL_NAME"]
        p = self.model_settings["PRIMARY_DEST"]

        if presample:
            assert not (self.zone_system == los.ONE_ZONE)
            skim_dict = self.network_los.get_skim_dict("taz")
        else:
            skim_dict = self.network_los.get_default_skim_dict()

        skims = {
            "od_skims": skim_dict.wrap(o, d),
            "dp_skims": skim_dict.wrap(d, p),
            "odt_skims": skim_dict.wrap_3d(
                orig_key=o, dest_key=d, dim3_key="trip_period"
            ),
            "dot_skims": skim_dict.wrap_3d(
                orig_key=d, dest_key=o, dim3_key="trip_period"
            ),
            "dpt_skims": skim_dict.wrap_3d(
                orig_key=d, dest_key=p, dim3_key="trip_period"
            ),
            "pdt_skims": skim_dict.wrap_3d(
                orig_key=p, dest_key=d, dim3_key="trip_period"
            ),
        }

        return skims

    def logsum_skims(self):

        o = self.model_settings["TRIP_ORIGIN"]
        d = self.model_settings["ALT_DEST_COL_NAME"]
        p = self.model_settings["PRIMARY_DEST"]
        skim_dict = self.network_los.get_default_skim_dict()

        skims = {
            "odt_skims": skim_dict.wrap_3d(
                orig_key=o, dest_key=d, dim3_key="trip_period"
            ),
            "dot_skims": skim_dict.wrap_3d(
                orig_key=d, dest_key=o, dim3_key="trip_period"
            ),
            "dpt_skims": skim_dict.wrap_3d(
                orig_key=d, dest_key=p, dim3_key="trip_period"
            ),
            "pdt_skims": skim_dict.wrap_3d(
                orig_key=p, dest_key=d, dim3_key="trip_period"
            ),
            "od_skims": skim_dict.wrap(o, d),
            "dp_skims": skim_dict.wrap(d, p),
        }

        if self.zone_system == los.THREE_ZONE:
            # fixme - is this a lightweight object?
            tvpb = self.network_los.tvpb

            tvpb_logsum_odt = tvpb.wrap_logsum(
                orig_key=o,
                dest_key=d,
                tod_key="trip_period",
                segment_key="demographic_segment",
                trace_label=self.trace_label,
                tag="tvpb_logsum_odt",
            )
            tvpb_logsum_dot = tvpb.wrap_logsum(
                orig_key=d,
                dest_key=o,
                tod_key="trip_period",
                segment_key="demographic_segment",
                trace_label=self.trace_label,
                tag="tvpb_logsum_dot",
            )
            tvpb_logsum_dpt = tvpb.wrap_logsum(
                orig_key=d,
                dest_key=p,
                tod_key="trip_period",
                segment_key="demographic_segment",
                trace_label=self.trace_label,
                tag="tvpb_logsum_dpt",
            )
            tvpb_logsum_pdt = tvpb.wrap_logsum(
                orig_key=p,
                dest_key=d,
                tod_key="trip_period",
                segment_key="demographic_segment",
                trace_label=self.trace_label,
                tag="tvpb_logsum_pdt",
            )

            skims.update(
                {
                    "tvpb_logsum_odt": tvpb_logsum_odt,
                    "tvpb_logsum_dot": tvpb_logsum_dot,
                    "tvpb_logsum_dpt": tvpb_logsum_dpt,
                    "tvpb_logsum_pdt": tvpb_logsum_pdt,
                }
            )

        return skims


def run_trip_destination(
    trips,
    tours_merged,
    estimator,
    chunk_size,
    trace_hh_id,
    trace_label,
    fail_some_trips_for_testing=False,
):
    """
    trip destination - main functionality separated from model step so it can be called iteratively

    Run the trip_destination model, assigning destinations for each (intermediate) trip
    (last trips already have a destination - either the tour primary destination or Home)

    Set trip destination and origin columns, and a boolean failed flag for any failed trips
    (destination for flagged failed trips will be set to -1)

    Parameters
    ----------
    trips
    tours_merged
    want_sample_table
    chunk_size
    trace_hh_id
    trace_label

    Returns
    -------

    """

    model_settings_file_name = "trip_destination.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)
    preprocessor_settings = model_settings.get("preprocessor", None)
    logsum_settings = config.read_model_settings(model_settings["LOGSUM_SETTINGS"])

    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    land_use = inject.get_table("land_use")
    size_terms = inject.get_injectable("size_terms")
    network_los = inject.get_injectable("network_los")
    trips = trips.sort_index()
    trips["next_trip_id"] = np.roll(trips.index, -1)
    trips.next_trip_id = trips.next_trip_id.where(trips.trip_num < trips.trip_count, 0)

    # - initialize trip origin and destination to those of half-tour
    # (we will sequentially adjust intermediate trips origin and destination as we choose them)
    # this is now probably redundant with stop_frequency.py L174
    tour_destination = reindex(tours_merged.destination, trips.tour_id).astype(np.int64)
    tour_origin = reindex(tours_merged.origin, trips.tour_id).astype(np.int64)

    # these values are now automatically created when trips are instantiated when
    # stop_frequency step calls trip.initialize_from_tours. But if this module is being
    # called from trip_destination_and_purpose, these columns will have been deleted
    # so they must be re-created
    if pipeline.get_rn_generator().step_name == "trip_purpose_and_destination":
        trips["destination"] = np.where(trips.outbound, tour_destination, tour_origin)
        trips["origin"] = np.where(trips.outbound, tour_origin, tour_destination)
        trips["failed"] = False

    if estimator:
        # need to check or override non-intermediate trip destination
        # should check consistency of survey trips origin, destination with parent tour and subsequent/prior trip?
        # FIXME if not consistent, do we fail or override? (seems weird to override them to bad values?)

        # expect all the same trips
        survey_trips = estimator.get_survey_table("trips").sort_index()
        assert survey_trips.index.equals(trips.index)

        first = survey_trips.trip_num == 1
        last = survey_trips.trip_num == trips.trip_count

        # expect survey's outbound first trip origin to be same as half tour origin
        assert (
            survey_trips.origin[survey_trips.outbound & first]
            == tour_origin[survey_trips.outbound & first]
        ).all()
        # expect outbound last trip destination to be same as half tour destination
        assert (
            survey_trips.destination[survey_trips.outbound & last]
            == tour_destination[survey_trips.outbound & last]
        ).all()

        # expect inbound first trip origin to be same as half tour destination
        assert (
            survey_trips.origin[~survey_trips.outbound & first]
            == tour_destination[~survey_trips.outbound & first]
        ).all()
        # expect inbound last trip destination to be same as half tour origin
        assert (
            survey_trips.destination[~survey_trips.outbound & last]
            == tour_origin[~survey_trips.outbound & last]
        ).all()

    # - filter tours_merged (AFTER copying destination and origin columns to trips)
    # tours_merged is used for logsums, we filter it here upfront to save space and time
    tours_merged_cols = logsum_settings["TOURS_MERGED_CHOOSER_COLUMNS"]
    redundant_cols = model_settings.get("REDUNDANT_TOURS_MERGED_CHOOSER_COLUMNS", [])
    if redundant_cols:
        tours_merged_cols = [c for c in tours_merged_cols if c not in redundant_cols]

    assert model_settings["PRIMARY_DEST"] not in tours_merged_cols
    tours_merged = tours_merged[tours_merged_cols]

    # - skims
    skim_hotel = SkimHotel(model_settings, network_los, trace_label)

    # - size_terms and alternatives
    alternatives = tour_destination_size_terms(land_use, size_terms, "trip")

    # DataFrameMatrix alows us to treat dataframe as virtual a 2-D array, indexed by zone_id, purpose
    # e.g. size_terms.get(df.dest_zone_id, df.purpose)
    # returns a series of size_terms for each chooser's dest_zone_id and purpose with chooser index
    size_term_matrix = DataFrameMatrix(alternatives)

    # don't need size terms in alternatives, just zone_id index
    alternatives = alternatives.drop(alternatives.columns, axis=1)
    alternatives.index.name = model_settings["ALT_DEST_COL_NAME"]

    sample_list = []

    # - process intermediate trips in ascending trip_num order
    intermediate = trips.trip_num < trips.trip_count
    if intermediate.any():

        first_trip_num = trips[intermediate].trip_num.min()
        last_trip_num = trips[intermediate].trip_num.max()

        # iterate over trips in ascending trip_num order
        for trip_num in range(first_trip_num, last_trip_num + 1):

            nth_trips = trips[intermediate & (trips.trip_num == trip_num)]
            nth_trace_label = tracing.extend_trace_label(
                trace_label, "trip_num_%s" % trip_num
            )

            locals_dict = {"network_los": network_los}
            locals_dict.update(config.get_model_constants(model_settings))

            # - annotate nth_trips
            if preprocessor_settings:
                expressions.assign_columns(
                    df=nth_trips,
                    model_settings=preprocessor_settings,
                    locals_dict=locals_dict,
                    trace_label=nth_trace_label,
                )

            logger.info("Running %s with %d trips", nth_trace_label, nth_trips.shape[0])

            # - choose destination for nth_trips, segmented by primary_purpose
            choices_list = []
            for primary_purpose, trips_segment in nth_trips.groupby("primary_purpose"):
                choices, destination_sample = choose_trip_destination(
                    primary_purpose,
                    trips_segment,
                    alternatives,
                    tours_merged,
                    model_settings,
                    want_logsums,
                    want_sample_table,
                    size_term_matrix,
                    skim_hotel,
                    estimator,
                    chunk_size,
                    trace_hh_id,
                    trace_label=tracing.extend_trace_label(
                        nth_trace_label, primary_purpose
                    ),
                )

                choices_list.append(choices)
                if want_sample_table:
                    assert destination_sample is not None
                    sample_list.append(destination_sample)

            destinations_df = pd.concat(choices_list)

            if fail_some_trips_for_testing:
                if len(destinations_df) > 0:
                    destinations_df = destinations_df.drop(destinations_df.index[0])

            failed_trip_ids = nth_trips.index.difference(destinations_df.index)
            if failed_trip_ids.any():
                logger.warning(
                    "%s sidelining %s trips without viable destination alternatives"
                    % (nth_trace_label, failed_trip_ids.shape[0])
                )
                next_trip_ids = nth_trips.next_trip_id.reindex(failed_trip_ids)
                trips.loc[failed_trip_ids, "failed"] = True
                trips.loc[failed_trip_ids, "destination"] = -1
                trips.loc[next_trip_ids, "origin"] = trips.loc[
                    failed_trip_ids
                ].origin.values

            if len(destinations_df) == 0:
                assert failed_trip_ids.all()
                logger.warning(
                    f"all {len(nth_trips)} {primary_purpose} trip_num {trip_num} trips failed"
                )

            if len(destinations_df) > 0:
                # - assign choices to this trip's destinations
                # if estimator, then the choices will already have been overridden by trip_destination_simulate
                # because we need to overwrite choices before any failed choices are suppressed
                assign_in_place(trips, destinations_df.choice.to_frame("destination"))
                if want_logsums:
                    assert "logsum" in destinations_df.columns
                    assign_in_place(
                        trips, destinations_df.logsum.to_frame(logsum_column_name)
                    )

                # - assign choice to next trip's origin
                destinations_df.index = nth_trips.next_trip_id.reindex(
                    destinations_df.index
                )
                assign_in_place(trips, destinations_df.choice.to_frame("origin"))

    del trips["next_trip_id"]

    if len(sample_list) > 0:
        save_sample_df = pd.concat(sample_list)
    else:
        # this could happen if no intermediate trips, or if no saved sample desired
        save_sample_df = None

    return trips, save_sample_df


@inject.step()
def trip_destination(trips, tours_merged, chunk_size, trace_hh_id):
    """
    Choose a destination for all 'intermediate' trips based on trip purpose.

    Final trips already have a destination (the primary tour destination for outbound trips,
    and home for inbound trips.)


    """
    trace_label = "trip_destination"

    model_settings_file_name = "trip_destination.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    CLEANUP = model_settings.get("CLEANUP", True)
    fail_some_trips_for_testing = model_settings.get(
        "fail_some_trips_for_testing", False
    )

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    estimator = estimation.manager.begin_estimation("trip_destination")

    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag="SPEC")
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(
            inject.get_injectable("size_terms"), "size_terms", append=False
        )
        estimator.write_table(
            inject.get_table("land_use").to_frame(), "landuse", append=False
        )
        estimator.write_model_settings(model_settings, model_settings_file_name)

    logger.info("Running %s with %d trips", trace_label, trips_df.shape[0])

    trips_df, save_sample_df = run_trip_destination(
        trips_df,
        tours_merged_df,
        estimator=estimator,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
        fail_some_trips_for_testing=fail_some_trips_for_testing,
    )

    # testing feature t0 make sure at least one trip fails so trip_purpose_and_destination model is run
    if (
        config.setting("testing_fail_trip_destination", False)
        and not trips_df.failed.any()
    ):
        if (trips_df.trip_num < trips_df.trip_count).sum() == 0:
            raise RuntimeError(
                f"can't honor 'testing_fail_trip_destination' setting because no intermediate trips"
            )

        fail_o = trips_df[trips_df.trip_num < trips_df.trip_count].origin.max()
        trips_df.failed = (trips_df.origin == fail_o) & (
            trips_df.trip_num < trips_df.trip_count
        )

    if trips_df.failed.any():
        logger.warning("%s %s failed trips", trace_label, trips_df.failed.sum())
        if inject.get_injectable("pipeline_file_prefix", None):
            file_name = f"{trace_label}_failed_trips_{inject.get_injectable('pipeline_file_prefix')}"
        else:
            file_name = f"{trace_label}_failed_trips"
        logger.info("writing failed trips to %s", file_name)
        tracing.write_csv(
            trips_df[trips_df.failed], file_name=file_name, transpose=False
        )

    if estimator:
        estimator.end_estimation()
        # no trips should have failed since we overwrite choices and sample should have not failed trips
        assert not trips_df.failed.any()

    if CLEANUP:

        if trips_df.failed.any():
            flag_failed_trip_leg_mates(trips_df, "failed")

            if save_sample_df is not None:
                save_sample_df.drop(
                    trips_df.index[trips_df.failed], level="trip_id", inplace=True
                )

            trips_df = cleanup_failed_trips(trips_df)

        trips_df.drop(columns="failed", inplace=True, errors="ignore")

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(
            trips_df,
            label=trace_label,
            slicer="trip_id",
            index_label="trip_id",
            warn_if_empty=True,
        )

    if save_sample_df is not None:
        # might be none if want_sample_table but there are no intermediate trips
        # expect samples only for intermediate trip destinations

        assert len(save_sample_df.index.get_level_values(0).unique()) == len(
            trips_df[trips_df.trip_num < trips_df.trip_count]
        )

        sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
        assert sample_table_name is not None

        logger.info(
            "adding %s samples to %s" % (len(save_sample_df), sample_table_name)
        )

        # lest they try to put tour samples into the same table
        if pipeline.is_table(sample_table_name):
            raise RuntimeError("sample table %s already exists" % sample_table_name)
        pipeline.extend_table(sample_table_name, save_sample_df)

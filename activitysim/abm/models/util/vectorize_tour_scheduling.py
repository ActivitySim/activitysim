# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    chunk,
    config,
    expressions,
    inject,
    logit,
    los,
    mem,
    simulate,
)
from activitysim.core import timetable as tt
from activitysim.core import tracing
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.pathbuilder import TransitVirtualPathBuilder
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)

TDD_CHOICE_COLUMN = "tdd"
USE_BRUTE_FORCE_TO_COMPUTE_LOGSUMS = False

RUN_ALTS_PREPROCESSOR_BEFORE_MERGE = True  # see FIXME below before changing this


def skims_for_logsums(tour_purpose, model_settings, trace_label):

    assert "LOGSUM_SETTINGS" in model_settings

    network_los = inject.get_injectable("network_los")

    skim_dict = network_los.get_default_skim_dict()

    orig_col_name = "home_zone_id"

    destination_for_tour_purpose = model_settings.get("DESTINATION_FOR_TOUR_PURPOSE")
    if isinstance(destination_for_tour_purpose, str):
        dest_col_name = destination_for_tour_purpose
    elif isinstance(destination_for_tour_purpose, dict):
        dest_col_name = destination_for_tour_purpose.get(tour_purpose)
    else:
        raise RuntimeError(
            f"expected string or dict DESTINATION_FOR_TOUR_PURPOSE model_setting for {tour_purpose}"
        )

    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="out_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="in_period"
    )
    odr_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="in_period"
    )
    dor_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="out_period"
    )
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,
        "dor_skims": dor_skim_stack_wrapper,
        "od_skims": od_skim_stack_wrapper,
        "orig_col_name": orig_col_name,
        "dest_col_name": dest_col_name,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?
        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col_name,
            dest_key=dest_col_name,
            tod_key="out_period",
            segment_key="demographic_segment",
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col_name,
            dest_key=orig_col_name,
            tod_key="in_period",
            segment_key="demographic_segment",
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

    return skims


def _compute_logsums(
    alt_tdd, tours_merged, tour_purpose, model_settings, network_los, skims, trace_label
):
    """
    compute logsums for tours using skims for alt_tdd out_period and in_period
    """

    trace_label = tracing.extend_trace_label(trace_label, "logsums")

    with chunk.chunk_log(trace_label):
        logsum_settings = config.read_model_settings(model_settings["LOGSUM_SETTINGS"])
        choosers = alt_tdd.join(tours_merged, how="left", rsuffix="_chooser")
        logger.info(
            f"{trace_label} compute_logsums for {choosers.shape[0]} choosers {alt_tdd.shape[0]} alts"
        )

        # - locals_dict
        constants = config.get_model_constants(logsum_settings)
        locals_dict = {}
        locals_dict.update(constants)

        if network_los.zone_system == los.THREE_ZONE:
            # TVPB constants can appear in expressions
            locals_dict.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

        locals_dict.update(skims)

        # constrained coefficients can appear in expressions
        coefficients = simulate.get_segment_coefficients(logsum_settings, tour_purpose)
        locals_dict.update(coefficients)

        # - run preprocessor to annotate choosers
        # allow specification of alternate preprocessor for nontour choosers
        preprocessor = model_settings.get("LOGSUM_PREPROCESSOR", "preprocessor")
        preprocessor_settings = logsum_settings[preprocessor]

        if preprocessor_settings:

            simulate.set_skim_wrapper_targets(choosers, skims)

            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=trace_label,
            )

        # - compute logsums
        logsum_spec = simulate.read_model_spec(file_name=logsum_settings["SPEC"])
        logsum_spec = simulate.eval_coefficients(
            logsum_spec, coefficients, estimator=None
        )

        nest_spec = config.get_logit_model_settings(logsum_settings)
        nest_spec = simulate.eval_nest_coefficients(
            nest_spec, coefficients, trace_label
        )

        logsums = simulate.simple_simulate_logsums(
            choosers,
            logsum_spec,
            nest_spec,
            skims=skims,
            locals_d=locals_dict,
            chunk_size=0,
            trace_label=trace_label,
        )

    return logsums


def dedupe_alt_tdd(alt_tdd, tour_purpose, trace_label):

    tdd_segments = inject.get_injectable("tdd_alt_segments", None)
    alt_tdd_periods = None

    logger.info(f"tdd_alt_segments specified for representative logsums")

    with chunk.chunk_log(tracing.extend_trace_label(trace_label, "dedupe_alt_tdd")):

        if tdd_segments is not None:

            dedupe_columns = ["out_period", "in_period"]

            # tdd_alt_segments is optionally segmented by tour purpose
            if "tour_purpose" in tdd_segments:

                is_tdd_for_tour_purpose = tdd_segments.tour_purpose == tour_purpose
                if not is_tdd_for_tour_purpose.any():
                    is_tdd_for_tour_purpose = tdd_segments.tour_purpose.isnull()
                assert (
                    is_tdd_for_tour_purpose.any()
                ), f"no segments found for tour purpose {tour_purpose} in tour_departure_and_duration_segments"

                tdd_segments = tdd_segments[is_tdd_for_tour_purpose].drop(
                    columns=["tour_purpose"]
                )
                assert (
                    len(tdd_segments) > 0
                ), f"tour_purpose '{tour_purpose}' not in tdd_alt_segments"

            # left join representative start on out_period
            alt_tdd_periods = pd.merge(
                alt_tdd[["out_period", "in_period"]].reset_index(),
                tdd_segments[["time_period", "start"]].rename(
                    columns={"time_period": "out_period"}
                ),
                how="left",
                on="out_period",
            )
            chunk.log_df(trace_label, "alt_tdd_periods", alt_tdd_periods)

            # left join representative end on in_period
            alt_tdd_periods = pd.merge(
                alt_tdd_periods,
                tdd_segments[["time_period", "end"]].rename(
                    columns={"time_period": "in_period"}
                ),
                how="left",
                on=["in_period"],
            )
            chunk.log_df(trace_label, "alt_tdd_periods", alt_tdd_periods)

            if tdd_segments.start.isnull().any():
                missing_periods = tdd_segments.out_period[
                    tdd_segments.start.isnull()
                ].unique()
                logger.warning(
                    f"missing out_periods in tdd_alt_segments: {missing_periods}"
                )

            if tdd_segments.end.isnull().any():
                missing_periods = tdd_segments.in_period[
                    tdd_segments.end.isnull()
                ].unique()
                logger.warning(
                    f"missing in_periods in tdd_alt_segments: {missing_periods}"
                )

            assert not tdd_segments.start.isnull().any()
            assert not tdd_segments.end.isnull().any()

            # drop duplicates
            alt_tdd_periods = alt_tdd_periods.drop_duplicates().set_index(
                alt_tdd.index.name
            )
            chunk.log_df(trace_label, "alt_tdd_periods", alt_tdd_periods)

            # representative duration
            alt_tdd_periods["duration"] = (
                alt_tdd_periods["end"] - alt_tdd_periods["start"]
            )
            chunk.log_df(trace_label, "alt_tdd_periods", alt_tdd_periods)

            logger.debug(
                f"{trace_label} "
                f"dedupe_alt_tdd.tdd_alt_segments reduced number of rows by "
                f"{round(100 * (len(alt_tdd) - len(alt_tdd_periods)) / len(alt_tdd), 2)}% "
                f"from {len(alt_tdd)} to {len(alt_tdd_periods)}"
            )

        # if there is no tdd_alt_segments file, we can at least dedupe on 'out_period', 'in_period', 'duration'
        if alt_tdd_periods is None:

            # FIXME This won't work if they reference start or end in logsum calculations
            # for MTC only duration is used (to calculate all_day parking cost)
            dedupe_columns = ["out_period", "in_period", "duration"]

            logger.warning(
                f"No tdd_alt_segments for representative logsums so fallback to "
                f"deduping tdd_alts by time_period and duration"
            )

            # - get list of unique (tour_id, out_period, in_period, duration) in alt_tdd_periods
            # we can cut the number of alts roughly in half (for mtctm1) by conflating duplicates
            alt_tdd_periods = (
                alt_tdd[dedupe_columns]
                .reset_index()
                .drop_duplicates()
                .set_index(alt_tdd.index.name)
            )
            chunk.log_df(trace_label, "alt_tdd_periods", alt_tdd_periods)

            logger.debug(
                f"{trace_label} "
                f"dedupe_alt_tdd.drop_duplicates reduced number of rows by "
                f"{round(100 * (len(alt_tdd) - len(alt_tdd_periods)) / len(alt_tdd), 2)}% "
                f"from {len(alt_tdd)} to {len(alt_tdd_periods)}"
            )

    return alt_tdd_periods, dedupe_columns


def compute_logsums(
    alt_tdd, tours_merged, tour_purpose, model_settings, skims, trace_label
):
    """
    Compute logsums for the tour alt_tdds, which will differ based on their different start, stop
    times of day, which translate to different odt_skim out_period and in_periods.

    In mtctm1, tdds are hourly, but there are only 5 skim time periods, so some of the tdd_alts
    will be the same, once converted to skim time periods. With 5 skim time periods there are
    15 unique out-out period pairs but 190 tdd alternatives.

    For efficiency, rather compute a lot of redundant logsums, we compute logsums for the unique
    (out-period, in-period) pairs and then join them back to the alt_tdds.
    """

    trace_label = tracing.extend_trace_label(trace_label, "compute_logsums")
    network_los = inject.get_injectable("network_los")

    # - in_period and out_period
    assert "out_period" not in alt_tdd
    assert "in_period" not in alt_tdd
    alt_tdd["out_period"] = network_los.skim_time_period_label(alt_tdd["start"])
    alt_tdd["in_period"] = network_los.skim_time_period_label(alt_tdd["end"])
    alt_tdd["duration"] = alt_tdd["end"] - alt_tdd["start"]

    # outside chunk_log context because we extend log_df call for alt_tdd made by our only caller _schedule_tours
    chunk.log_df(trace_label, "alt_tdd", alt_tdd)

    with chunk.chunk_log(trace_label):

        if USE_BRUTE_FORCE_TO_COMPUTE_LOGSUMS:
            # compute logsums for all the tour alt_tdds (inefficient)
            logsums = _compute_logsums(
                alt_tdd,
                tours_merged,
                tour_purpose,
                model_settings,
                network_los,
                skims,
                trace_label,
            )
            return logsums

        index_name = alt_tdd.index.name
        deduped_alt_tdds, redupe_columns = dedupe_alt_tdd(
            alt_tdd, tour_purpose, trace_label
        )
        chunk.log_df(trace_label, "deduped_alt_tdds", deduped_alt_tdds)

        logger.info(
            f"{trace_label} compute_logsums "
            f"deduped_alt_tdds reduced number of rows by "
            f"{round(100 * (len(alt_tdd) - len(deduped_alt_tdds)) / len(alt_tdd), 2)}% "
            f"from {len(alt_tdd)} to {len(deduped_alt_tdds)} compared to USE_BRUTE_FORCE_TO_COMPUTE_LOGSUMS"
        )

        t0 = tracing.print_elapsed_time()

        # - compute logsums for the alt_tdd_periods
        deduped_alt_tdds["logsums"] = _compute_logsums(
            deduped_alt_tdds,
            tours_merged,
            tour_purpose,
            model_settings,
            network_los,
            skims,
            trace_label,
        )

        # tracing.log_runtime(model_name=trace_label, start_time=t0)

        # redupe - join the alt_tdd_period logsums to alt_tdd to get logsums for alt_tdd
        logsums = (
            pd.merge(
                alt_tdd.reset_index(),
                deduped_alt_tdds.reset_index(),
                on=[index_name] + redupe_columns,
                how="left",
            )
            .set_index(index_name)
            .logsums
        )
        chunk.log_df(trace_label, "logsums", logsums)

        del deduped_alt_tdds
        chunk.log_df(trace_label, "deduped_alt_tdds", None)

        # this is really expensive
        TRACE = False
        if TRACE:
            trace_logsums_df = logsums.to_frame("representative_logsum")
            trace_logsums_df["brute_force_logsum"] = _compute_logsums(
                alt_tdd,
                tours_merged,
                tour_purpose,
                model_settings,
                network_los,
                skims,
                trace_label,
            )
            tracing.trace_df(
                trace_logsums_df,
                label=tracing.extend_trace_label(trace_label, "representative_logsums"),
                slicer="NONE",
                transpose=False,
            )

    # leave it to our caller to pick up logsums with call to chunk.log_df
    return logsums


def get_previous_tour_by_tourid(
    current_tour_window_ids, previous_tour_by_window_id, alts
):
    """
    Matches current tours with attributes of previous tours for the same
    person.  See the return value below for more information.

    Parameters
    ----------
    current_tour_window_ids : Series
        A Series of parent ids for the tours we're about make the choice for
        - index should match the tours DataFrame.
    previous_tour_by_window_id : Series
        A Series where the index is the parent (window) id and the value is the index
        of the alternatives of the scheduling.
    alts : DataFrame
        The alternatives of the scheduling.

    Returns
    -------
    prev_alts : DataFrame
        A DataFrame with an index matching the CURRENT tours we're making a
        decision for, but with columns from the PREVIOUS tour of the person
        associated with each of the CURRENT tours.  Columns listed in PREV_TOUR_COLUMNS
        from the alternatives will have "_previous" added as a suffix to keep
        differentiated from the current alternatives that will be part of the
        interaction.
    """

    PREV_TOUR_COLUMNS = ["start", "end"]

    previous_tour_by_tourid = previous_tour_by_window_id.loc[current_tour_window_ids]

    previous_tour_by_tourid = alts.loc[previous_tour_by_tourid, PREV_TOUR_COLUMNS]

    previous_tour_by_tourid.index = current_tour_window_ids.index
    previous_tour_by_tourid.columns = [x + "_previous" for x in PREV_TOUR_COLUMNS]

    return previous_tour_by_tourid


def tdd_interaction_dataset(
    tours, alts, timetable, choice_column, window_id_col, trace_label
):
    """
    interaction_sample_simulate expects
    alts index same as choosers (e.g. tour_id)
    name of choice column in alts

    Parameters
    ----------
    tours : pandas DataFrame
        must have person_id column and index on tour_id
    alts : pandas DataFrame
        alts index must be timetable tdd id
    timetable : TimeTable object
    choice_column : str
        name of column to store alt index in alt_tdd DataFrame
        (since alt_tdd is duplicate index on person_id but unique on person_id,alt_id)

    Returns
    -------
    alt_tdd : pandas DataFrame
        columns: start, end , duration, <choice_column>
        index: tour_id


    """

    trace_label = tracing.extend_trace_label(trace_label, "tdd_interaction_dataset")

    with chunk.chunk_log(trace_label):
        alts_ids = np.tile(alts.index, len(tours.index))
        chunk.log_df(trace_label, "alts_ids", alts_ids)

        tour_ids = np.repeat(tours.index, len(alts.index))
        window_row_ids = np.repeat(tours[window_id_col], len(alts.index))

        alt_tdd = alts.take(alts_ids)

        alt_tdd.index = tour_ids
        alt_tdd[window_id_col] = window_row_ids

        # add tdd alternative id
        # by convention, the choice column is the first column in the interaction dataset
        alt_tdd.insert(loc=0, column=choice_column, value=alts_ids)

        # slice out all non-available tours
        available = timetable.tour_available(
            alt_tdd[window_id_col], alt_tdd[choice_column]
        )
        logger.debug(
            f"tdd_interaction_dataset keeping {available.sum()} of ({len(available)}) available alt_tdds"
        )
        assert available.any()

        chunk.log_df(
            trace_label, "alt_tdd", alt_tdd
        )  # catch this before we slice on available

        alt_tdd = alt_tdd[available]

        chunk.log_df(trace_label, "alt_tdd", alt_tdd)

        # FIXME - don't need this any more after slicing
        del alt_tdd[window_id_col]

    return alt_tdd


def run_alts_preprocessor(model_settings, alts, segment, locals_dict, trace_label):
    """
    run preprocessor on alts, as specified by ALTS_PREPROCESSOR in model_settings

    we are agnostic on whether alts are merged or not

    Parameters
    ----------
    model_settings: dict
        yaml model settings file as dict
    alts: pandas.DataFrame
        tdd_alts or tdd_alts merged wiht choosers (we are agnostic)
    segment: string
        segment selector as understood by caller (e.g. logsum_tour_purpose)
    locals_dict: dict
        we let caller worry about what needs to be in it. though actually depends on modelers needs
    trace_label: string

    Returns
    -------
    alts: pandas.DataFrame
        annotated copy of alts
    """

    preprocessor_settings = model_settings.get("ALTS_PREPROCESSOR", {})

    if segment in preprocessor_settings:
        # segmented by logsum_tour_purpose
        preprocessor_settings = preprocessor_settings.get(segment)
        logger.debug(
            f"running ALTS_PREPROCESSOR with spec for {segment}: {preprocessor_settings.get('SPEC')}"
        )
    elif "SPEC" in preprocessor_settings:
        # unsegmented (either because no segmentation, or fallback if settings has generic preprocessor)
        logger.debug(
            f"running ALTS_PREPROCESSOR with unsegmented spec {preprocessor_settings.get('SPEC')}"
        )
    else:
        logger.debug(
            f"skipping alts preprocesser because no ALTS_PREPROCESSOR segment for {segment}"
        )
        preprocessor_settings = None

    if preprocessor_settings:

        logger.debug(
            f"run_alts_preprocessor calling assign_columns for {segment} preprocessor_settings"
        )
        alts = alts.copy()

        expressions.assign_columns(
            df=alts,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    return alts


def _schedule_tours(
    tours,
    persons_merged,
    alts,
    spec,
    logsum_tour_purpose,
    model_settings,
    skims,
    timetable,
    window_id_col,
    previous_tour,
    tour_owner_id_col,
    estimator,
    tour_trace_label,
):
    """
    previous_tour stores values used to add columns that can be used in the spec
    which have to do with the previous tours per person.  Every column in the
    alternatives table is appended with the suffix "_previous" and made
    available.  So if your alternatives table has columns for start and end,
    then start_previous and end_previous will be set to the start and end of
    the most recent tour for a person.  The first time through,
    start_previous and end_previous are undefined, so make sure to protect
    with a tour_num >= 2 in the variable computation.

    Parameters
    ----------
    tours : DataFrame
        chunk of tours to schedule with unique timetable window_id_col
    persons_merged : DataFrame
        DataFrame of persons to be merged with tours containing attributes referenced
        by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent all possible time slots.
        tdd_interaction_dataset function will use timetable to filter them to omit
        unavailable alternatives
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
    model_settings : dict
    timetable : TimeTable
        timetable of timewidows for person (or subtour) with rows for tours[window_id_col]
    window_id_col : str
        column name from tours that identifies timetable owner (or None if tours index)
        - person_id for non/mandatory tours
        - parent_tour_id for subtours,
        - None (tours index) for joint_tours since every tour may have different participants)
    previous_tour: Series
        series with value of tdd_alt choice for last previous tour scheduled for
    tour_owner_id_col : str
        column name from tours that identifies 'owner' of this tour
        (person_id for non/mandatory tours, parent_tour_id for subtours,
        household_id for joint_tours)
    tour_trace_label

    Returns
    -------

    """

    logger.info(
        "%s schedule_tours running %d tour choices" % (tour_trace_label, len(tours))
    )

    # merge persons into tours
    # avoid dual suffix for redundant columns names (e.g. household_id) that appear in both
    tours = pd.merge(
        tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        suffixes=("", "_y"),
    )
    chunk.log_df(tour_trace_label, "tours", tours)

    # - add explicit window_id_col for timetable owner if it is index
    # if no timetable window_id_col specified, then add index as an explicit column
    # (this is not strictly necessary but its presence makes code simpler in several places)
    if window_id_col is None:
        window_id_col = tours.index.name
        tours[window_id_col] = tours.index

    # timetable can't handle multiple tours per window_id
    assert not tours[window_id_col].duplicated().any()

    # - build interaction dataset filtered to include only available tdd alts
    # dataframe columns start, end , duration, person_id, tdd
    # indexed (not unique) on tour_id
    choice_column = TDD_CHOICE_COLUMN
    alt_tdd = tdd_interaction_dataset(
        tours, alts, timetable, choice_column, window_id_col, tour_trace_label
    )
    # print(f"tours {tours.shape} alts {alts.shape}")

    chunk.log_df(tour_trace_label, "alt_tdd", alt_tdd)

    # - add logsums
    if logsum_tour_purpose:
        logsums = compute_logsums(
            alt_tdd, tours, logsum_tour_purpose, model_settings, skims, tour_trace_label
        )
    else:
        logsums = 0
    alt_tdd["mode_choice_logsum"] = logsums

    del logsums
    chunk.log_df(tour_trace_label, "alt_tdd", alt_tdd)

    # - merge in previous tour columns
    # adds start_previous and end_previous, joins on index
    tours = tours.join(
        get_previous_tour_by_tourid(tours[tour_owner_id_col], previous_tour, alts)
    )
    chunk.log_df(tour_trace_label, "tours", tours)

    # - make choices
    locals_d = {"tt": timetable}
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    if not RUN_ALTS_PREPROCESSOR_BEFORE_MERGE:
        # Note: Clint was running alts_preprocessor here on tdd_interaction_dataset instead of on raw (unmerged) alts
        # and he was using logsum_tour_purpose as selector, although logically it should be the spec_segment
        # It just happened to work for prototype_arc.mandatory_tour_scheduling because, in that model, (unlike semcog)
        # logsum_tour_purpose and spec_segments are aligned (both logsums and spec are segmented on work, school, univ)
        # In any case, I don't see any benefit to doing this here - at least not for any existing implementations
        # but if we do, it will require passing spec_segment to schedule_tours  and _schedule_tours
        # or redundently segmenting alts (yuck!) to conform to more granular tour_segmentation (e.g. univ do school)
        spec_segment = (
            logsum_tour_purpose  # FIXME this is not always right - see note above
        )
        alt_tdd = run_alts_preprocessor(
            model_settings, alt_tdd, spec_segment, locals_d, tour_trace_label
        )
        chunk.log_df(tour_trace_label, "alt_tdd", alt_tdd)

    if estimator:
        # write choosers after annotation
        estimator.write_choosers(tours)
        estimator.set_alt_id(choice_column)
        estimator.write_interaction_sample_alternatives(alt_tdd)

    log_alt_losers = config.setting("log_alt_losers", False)

    choices = interaction_sample_simulate(
        tours,
        alt_tdd,
        spec,
        choice_column=choice_column,
        log_alt_losers=log_alt_losers,
        locals_d=locals_d,
        chunk_size=0,
        trace_label=tour_trace_label,
        estimator=estimator,
    )
    chunk.log_df(tour_trace_label, "choices", choices)

    # - update previous_tour and timetable parameters

    # update previous_tour (series with most recent previous tdd choices) with latest values
    previous_tour.loc[tours[tour_owner_id_col]] = choices.values

    # update timetable with chosen tdd footprints
    timetable.assign(tours[window_id_col], choices)

    return choices


def schedule_tours(
    tours,
    persons_merged,
    alts,
    spec,
    logsum_tour_purpose,
    model_settings,
    timetable,
    timetable_window_id_col,
    previous_tour,
    tour_owner_id_col,
    estimator,
    chunk_size,
    tour_trace_label,
    tour_chunk_tag,
):
    """
    chunking wrapper for _schedule_tours

    While interaction_sample_simulate provides chunking support, the merged tours, persons
    dataframe and the tdd_interaction_dataset are very big, so we want to create them inside
    the chunking loop to minimize memory footprint. So we implement the chunking loop here,
    and pass a chunk_size of 0 to interaction_sample_simulate to disable its chunking support.

    """

    if not tours.index.is_monotonic_increasing:
        logger.info("schedule_tours %s tours not monotonic_increasing - sorting df")
        tours = tours.sort_index()

    logger.info(
        "%s schedule_tours running %d tour choices" % (tour_trace_label, len(tours))
    )

    # no more than one tour per timetable_window per call
    if timetable_window_id_col is None:
        assert not tours.index.duplicated().any()
    else:
        assert not tours[timetable_window_id_col].duplicated().any()

    if "LOGSUM_SETTINGS" in model_settings:
        # we need skims to calculate tvpb skim overhead in 3_ZONE systems for use by calc_rows_per_chunk
        skims = skims_for_logsums(logsum_tour_purpose, model_settings, tour_trace_label)
    else:
        skims = None

    result_list = []
    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        tours, chunk_size, tour_trace_label, tour_chunk_tag
    ):

        choices = _schedule_tours(
            chooser_chunk,
            persons_merged,
            alts,
            spec,
            logsum_tour_purpose,
            model_settings,
            skims,
            timetable,
            timetable_window_id_col,
            previous_tour,
            tour_owner_id_col,
            estimator,
            tour_trace_label=chunk_trace_label,
        )

        result_list.append(choices)

        chunk.log_df(tour_trace_label, f"result_list", result_list)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(tours.index))

    return choices


def vectorize_tour_scheduling(
    tours,
    persons_merged,
    alts,
    timetable,
    tour_segments,
    tour_segment_col,
    model_settings,
    chunk_size=0,
    trace_label=None,
):
    """
    The purpose of this method is fairly straightforward - it takes tours
    and schedules them into time slots.  Alternatives should be specified so
    as to define those time slots (usually with start and end times).

    schedule_tours adds variables that can be used in the spec which have
    to do with the previous tours per person.  Every column in the
    alternatives table is appended with the suffix "_previous" and made
    available.  So if your alternatives table has columns for start and end,
    then start_previous and end_previous will be set to the start and end of
    the most recent tour for a person.  The first time through,
    start_previous and end_previous are undefined, so make sure to protect
    with a tour_num >= 2 in the variable computation.

    FIXME - fix docstring: tour_segments, tour_segment_col

    Parameters
    ----------
    tours : DataFrame
        DataFrame of tours containing tour attributes, as well as a person_id
        column to define the nth tour for each person.
    persons_merged : DataFrame
        DataFrame of persons containing attributes referenced by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent time slots.  Will be passed to
        interaction_simulate in batches for each nth tour.
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
        (or dict of specs keyed on tour_type if tour_types is not None)
    model_settings : dict

    Returns
    -------
    choices : Series
        A Series of choices where the index is the index of the tours
        DataFrame and the values are the index of the alts DataFrame.
    timetable : TimeTable
        persons timetable updated with tours (caller should replace_table for it to persist)
    """

    trace_label = tracing.extend_trace_label(trace_label, "vectorize_tour_scheduling")

    assert len(tours.index) > 0
    assert "tour_num" in tours.columns
    assert "tour_type" in tours.columns

    # tours must be scheduled in increasing trip_num order
    # second trip of type must be in group immediately following first
    # this ought to have been ensured when tours are created (tour_frequency.process_tours)
    choice_list = []

    # keep a series of the the most recent tours for each person
    # initialize with first trip from alts
    previous_tour_by_personid = pd.Series(alts.index[0], index=tours.person_id.unique())

    timetable_window_id_col = "person_id"
    tour_owner_id_col = "person_id"
    compute_logsums = "LOGSUM_SETTINGS" in model_settings

    assert isinstance(tour_segments, dict)

    # no more than one tour per person per call to schedule_tours
    # tours must be scheduled in increasing trip_num order
    # second trip of type must be in group immediately following first
    # segregate scheduling by tour_type if multiple specs passed in dict keyed by tour_type

    for tour_num, nth_tours in tours.groupby("tour_num", sort=True):

        tour_trace_label = tracing.extend_trace_label(trace_label, f"tour_{tour_num}")
        tour_chunk_tag = tracing.extend_trace_label(
            trace_label, f"tour_{1 if tour_num == 1 else 'n'}"
        )

        if tour_segment_col is not None:

            for tour_segment_name, tour_segment_info in tour_segments.items():

                segment_trace_label = tracing.extend_trace_label(
                    tour_trace_label, tour_segment_name
                )
                segment_chunk_tag = tracing.extend_trace_label(
                    tour_chunk_tag, tour_segment_name
                )

                # assume segmentation of spec and coefficients are aligned
                spec_segment_name = tour_segment_info.get("spec_segment_name")
                # assume logsum segmentation is same as tours
                logsum_tour_purpose = tour_segment_name if compute_logsums else None

                nth_tours_in_segment = nth_tours[
                    nth_tours[tour_segment_col] == tour_segment_name
                ]
                if nth_tours_in_segment.empty:
                    logger.info("skipping empty segment %s" % tour_segment_name)
                    continue

                if RUN_ALTS_PREPROCESSOR_BEFORE_MERGE:
                    locals_dict = {}
                    alts = run_alts_preprocessor(
                        model_settings,
                        alts,
                        spec_segment_name,
                        locals_dict,
                        tour_trace_label,
                    )

                choices = schedule_tours(
                    nth_tours_in_segment,
                    persons_merged,
                    alts,
                    spec=tour_segment_info["spec"],
                    logsum_tour_purpose=logsum_tour_purpose,
                    model_settings=model_settings,
                    timetable=timetable,
                    timetable_window_id_col=timetable_window_id_col,
                    previous_tour=previous_tour_by_personid,
                    tour_owner_id_col=tour_owner_id_col,
                    estimator=tour_segment_info.get("estimator"),
                    chunk_size=chunk_size,
                    tour_trace_label=segment_trace_label,
                    tour_chunk_tag=segment_chunk_tag,
                )

                choice_list.append(choices)

        else:

            # MTC non_mandatory_tours are not segmented by tour_purpose and do not require logsums
            # FIXME should support logsums?

            assert (
                not compute_logsums
            ), "logsums for unsegmented spec not implemented because not currently needed"
            assert tour_segments.get("spec_segment_name") is None

            choices = schedule_tours(
                nth_tours,
                persons_merged,
                alts,
                spec=tour_segments["spec"],
                logsum_tour_purpose=None,
                model_settings=model_settings,
                timetable=timetable,
                timetable_window_id_col=timetable_window_id_col,
                previous_tour=previous_tour_by_personid,
                tour_owner_id_col=tour_owner_id_col,
                estimator=tour_segments.get("estimator"),
                chunk_size=chunk_size,
                tour_trace_label=tour_trace_label,
                tour_chunk_tag=tour_chunk_tag,
            )

            choice_list.append(choices)

    choices = pd.concat(choice_list)
    return choices


def vectorize_subtour_scheduling(
    parent_tours,
    subtours,
    persons_merged,
    alts,
    spec,
    model_settings,
    estimator,
    chunk_size=0,
    trace_label=None,
):
    """
    Like vectorize_tour_scheduling but specifically for atwork subtours

    subtours have a few peculiarities necessitating separate treatment:

    Timetable has to be initialized to set all timeperiods outside parent tour footprint as
    unavailable. So atwork subtour timewindows are limited to the footprint of the parent work
    tour. And parent_tour_id' column of tours is used instead of parent_id as timetable row_id.

    Parameters
    ----------
    parent_tours : DataFrame
        parent tours of the subtours (because we need to know the tdd of the parent tour to
        assign_subtour_mask of timetable indexed by parent_tour id
    subtours : DataFrame
        atwork subtours to schedule
    persons_merged : DataFrame
        DataFrame of persons containing attributes referenced by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent time slots.  Will be passed to
        interaction_simulate in batches for each nth tour.
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
        (all subtours share same spec regardless of subtour type)
    model_settings : dict
    chunk_size
    trace_label

    Returns
    -------
    choices : Series
        A Series of choices where the index is the index of the subtours
        DataFrame and the values are the index of the alts DataFrame.
    """
    if not trace_label:
        trace_label = "vectorize_non_mandatory_tour_scheduling"

    assert len(subtours.index) > 0
    assert "tour_num" in subtours.columns
    assert "tour_type" in subtours.columns

    timetable_window_id_col = "parent_tour_id"
    tour_owner_id_col = "parent_tour_id"
    logsum_tour_purpose = None  # FIXME logsums not currently supported

    # timetable with a window for each parent tour
    parent_tour_windows = tt.create_timetable_windows(parent_tours, alts)
    timetable = tt.TimeTable(parent_tour_windows, alts)

    # mask the periods outside parent tour footprint
    timetable.assign_subtour_mask(parent_tours.tour_id, parent_tours.tdd)

    # print timetable.windows
    """
    [[7 7 7 0 0 0 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7]
     [7 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
     [7 7 7 7 7 0 0 0 0 0 0 0 0 0 0 7 7 7 7 7 7]
     [7 7 0 0 0 0 0 0 0 7 7 7 7 7 7 7 7 7 7 7 7]]
    """

    choice_list = []

    # keep a series of the the most recent tours for each person
    # initialize with first trip from alts
    previous_tour_by_parent_tour_id = pd.Series(
        alts.index[0], index=subtours["parent_tour_id"].unique()
    )

    # tours must be scheduled in increasing trip_num order
    # second trip of type must be in group immediately following first
    # this ought to have been ensured when tours are created (tour_frequency.process_tours)

    for tour_num, nth_tours in subtours.groupby("tour_num", sort=True):

        tour_trace_label = tracing.extend_trace_label(trace_label, f"tour_{tour_num}")
        tour_chunk_tag = tracing.extend_trace_label(
            trace_label, f"tour_{1 if tour_num == 1 else 'n'}"
        )

        # no more than one tour per timetable window per call to schedule_tours
        assert not nth_tours.parent_tour_id.duplicated().any()

        choices = schedule_tours(
            nth_tours,
            persons_merged,
            alts,
            spec,
            logsum_tour_purpose,
            model_settings,
            timetable,
            timetable_window_id_col,
            previous_tour_by_parent_tour_id,
            tour_owner_id_col,
            estimator,
            chunk_size,
            tour_trace_label,
            tour_chunk_tag,
        )

        choice_list.append(choices)

    choices = pd.concat(choice_list)

    # print "\nfinal timetable.windows\n%s" % timetable.windows
    """
    [[7 7 7 0 0 0 0 2 7 7 4 7 7 7 7 7 7 7 7 7 7]
     [7 0 2 7 4 0 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]
     [7 7 7 7 7 2 4 0 0 0 0 0 0 0 0 7 7 7 7 7 7]
     [7 7 0 2 7 7 4 0 0 7 7 7 7 7 7 7 7 7 7 7 7]]
    """
    # we dont need to call replace_table() for this nonce timetable
    # because subtours are occuring during persons timetable scheduled time

    return choices


def build_joint_tour_timetables(
    joint_tours, joint_tour_participants, persons_timetable, alts
):

    # timetable with a window for each joint tour
    joint_tour_windows_df = tt.create_timetable_windows(joint_tours, alts)
    joint_tour_timetable = tt.TimeTable(joint_tour_windows_df, alts)

    for participant_num, nth_participants in joint_tour_participants.groupby(
        "participant_num", sort=True
    ):

        # nth_participant windows from persons_timetable
        participant_windows = persons_timetable.slice_windows_by_row_id(
            nth_participants.person_id
        )

        # assign them joint_tour_timetable
        joint_tour_timetable.assign_footprints(
            nth_participants.tour_id, participant_windows
        )

    return joint_tour_timetable


def vectorize_joint_tour_scheduling(
    joint_tours,
    joint_tour_participants,
    persons_merged,
    alts,
    persons_timetable,
    spec,
    model_settings,
    estimator,
    chunk_size=0,
    trace_label=None,
):
    """
    Like vectorize_tour_scheduling but specifically for joint tours

    joint tours have a few peculiarities necessitating separate treatment:

    Timetable has to be initialized to set all timeperiods...

    Parameters
    ----------
    tours : DataFrame
        DataFrame of tours containing tour attributes, as well as a person_id
        column to define the nth tour for each person.
    persons_merged : DataFrame
        DataFrame of persons containing attributes referenced by expressions in spec
    alts : DataFrame
        DataFrame of alternatives which represent time slots.  Will be passed to
        interaction_simulate in batches for each nth tour.
    spec : DataFrame
        The spec which will be passed to interaction_simulate.
        (or dict of specs keyed on tour_type if tour_types is not None)
    model_settings : dict

    Returns
    -------
    choices : Series
        A Series of choices where the index is the index of the tours
        DataFrame and the values are the index of the alts DataFrame.
    persons_timetable : TimeTable
        timetable updated with joint tours (caller should replace_table for it to persist)
    """

    trace_label = tracing.extend_trace_label(
        trace_label, "vectorize_joint_tour_scheduling"
    )

    assert len(joint_tours.index) > 0
    assert "tour_num" in joint_tours.columns
    assert "tour_type" in joint_tours.columns

    timetable_window_id_col = None
    tour_owner_id_col = "household_id"
    logsum_tour_purpose = None  # FIXME logsums not currently supported

    choice_list = []

    # keep a series of the the most recent tours for each person
    # initialize with first trip from alts
    previous_tour_by_householdid = pd.Series(
        alts.index[0], index=joint_tours.household_id.unique()
    )

    # tours must be scheduled in increasing trip_num order
    # second trip of type must be in group immediately following first
    # this ought to have been ensured when tours are created (tour_frequency.process_tours)

    # print "participant windows before scheduling\n%s" % \
    #     persons_timetable.slice_windows_by_row_id(joint_tour_participants.person_id)

    for tour_num, nth_tours in joint_tours.groupby("tour_num", sort=True):

        tour_trace_label = tracing.extend_trace_label(trace_label, f"tour_{tour_num}")
        tour_chunk_tag = tracing.extend_trace_label(
            trace_label, f"tour_{1 if tour_num == 1 else 'n'}"
        )

        # no more than one tour per household per call to schedule_tours
        assert not nth_tours.household_id.duplicated().any()

        nth_participants = joint_tour_participants[
            joint_tour_participants.tour_id.isin(nth_tours.index)
        ]

        timetable = build_joint_tour_timetables(
            nth_tours, nth_participants, persons_timetable, alts
        )

        choices = schedule_tours(
            nth_tours,
            persons_merged,
            alts,
            spec,
            logsum_tour_purpose,
            model_settings,
            timetable,
            timetable_window_id_col,
            previous_tour_by_householdid,
            tour_owner_id_col,
            estimator,
            chunk_size,
            tour_trace_label,
            tour_chunk_tag,
        )

        # - update timetables of all joint tour participants
        persons_timetable.assign(
            nth_participants.person_id, reindex(choices, nth_participants.tour_id)
        )

        choice_list.append(choices)

    choices = pd.concat(choice_list)

    return choices

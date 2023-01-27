# ActivitySim
# See full license in LICENSE.txt.
import logging
import time
from builtins import zip
from collections import OrderedDict
from datetime import timedelta

import numpy as np
import pandas as pd

from . import chunk, config, logit, simulate, tracing

logger = logging.getLogger(__name__)

DUMP = False

ALT_CHOOSER_ID = "_chooser_id"


def eval_interaction_utilities(
    spec,
    df,
    locals_d,
    trace_label,
    trace_rows,
    estimator=None,
    log_alt_losers=False,
    extra_data=None,
    zone_layer=None,
):
    """
    Compute the utilities for a single-alternative spec evaluated in the context of df

    We could compute the utilities for interaction datasets just as we do for simple_simulate
    specs with multiple alternative columns by calling eval_variables and then computing the
    utilities by matrix-multiplication of eval results with the utility coefficients in the
    spec alternative columns.

    But interaction simulate computes the utilities of each alternative in the context of a
    separate row in interaction dataset df, and so there is only one alternative in spec.
    This turns out to be quite a bit faster (in this special case) than the pandas dot function.

    For efficiency, we combine eval_variables and multiplication of coefficients into a single step,
    so we don't have to create a separate column for each partial utility. Instead, we simply
    multiply the eval result by a single alternative coefficient and sum the partial utilities.


    spec : dataframe
        one row per spec expression and one col with utility coefficient

    df : dataframe
        cross join (cartesian product) of choosers with alternatives
        combines columns of choosers and alternatives
        len(df) == len(choosers) * len(alternatives)
        index values (non-unique) are index values from alternatives df

    interaction_utilities : dataframe
        the utility of each alternative is sum of the partial utilities determined by the
        various spec expressions and their corresponding coefficients
        yielding a dataframe  with len(interaction_df) rows and one utility column
        having the same index as interaction_df (non-unique values from alternatives df)

    Returns
    -------
    utilities : pandas.DataFrame
        Will have the index of `df` and a single column of utilities

    """
    start_time = time.time()

    trace_label = tracing.extend_trace_label(trace_label, "eval_interaction_utils")
    logger.info("Running eval_interaction_utilities on %s rows" % df.shape[0])

    sharrow_enabled = config.setting("sharrow", False)

    if locals_d is not None and locals_d.get("_sharrow_skip", False):
        sharrow_enabled = False

    # if trace_label.startswith("trip_destination"):
    #     sharrow_enabled = False

    logger.info(f"{trace_label} sharrow_enabled is {sharrow_enabled}")

    trace_eval_results = None

    with chunk.chunk_log(trace_label):

        assert len(spec.columns) == 1

        # avoid altering caller's passed-in locals_d parameter (they may be looping)
        locals_d = locals_d.copy() if locals_d is not None else {}

        utilities = None

        from .flow import TimeLogger

        timelogger = TimeLogger("interaction_simulate")

        # add df for startswith('@') eval expressions
        locals_d["df"] = df

        if sharrow_enabled:

            from .flow import apply_flow

            spec_sh = spec.copy()

            def replace_in_index_level(mi, level, *repls):
                if isinstance(mi, pd.MultiIndex):
                    level = mi._get_level_number(level)
                    content = list(mi.levels[level])
                    new_content = []
                    for i in content:
                        for repl in repls:
                            i = i.replace(*repl)
                        new_content.append(i)
                    return mi.set_levels(new_content, level=level)
                else:
                    new_content = []
                    for i in mi:
                        for repl in repls:
                            i = i.replace(*repl)
                        new_content.append(i)
                    return new_content

            spec_sh.index = replace_in_index_level(
                spec_sh.index,
                simulate.SPEC_EXPRESSION_NAME,
                (
                    "tt.adjacent_window_before(",
                    "sharrow_tt_adjacent_window_before(tt_windows, tt_row_mapper, tt_col_mapper, ",
                ),
                (
                    "tt.adjacent_window_after(",
                    "sharrow_tt_adjacent_window_after(tt_windows, tt_row_mapper, tt_col_mapper, ",
                ),
                (
                    "tt.previous_tour_ends(",
                    "sharrow_tt_previous_tour_ends(tt_windows, tt_row_mapper, tt_col_mapper, ",
                ),
                (
                    "tt.previous_tour_begins(",
                    "sharrow_tt_previous_tour_begins(tt_windows, tt_row_mapper, tt_col_mapper, ",
                ),
                (
                    "tt.remaining_periods_available(",
                    "sharrow_tt_remaining_periods_available(tt_windows, tt_row_mapper, ",
                ),
                (
                    "tt.max_time_block_available(",
                    "sharrow_tt_max_time_block_available(tt_windows, tt_row_mapper, ",
                ),
            )

            # need to zero out any coefficients on temp vars
            if isinstance(spec_sh.index, pd.MultiIndex):
                exprs = spec_sh.index.get_level_values(simulate.SPEC_EXPRESSION_NAME)
                labels = spec_sh.index.get_level_values(simulate.SPEC_LABEL_NAME)
            else:
                exprs = spec_sh.index
                labels = spec_sh.index
            for n, (expr, label) in enumerate(zip(exprs, labels)):
                if expr.startswith("_") and "@" in expr:
                    spec_sh.iloc[n, 0] = 0.0

            for i1, i2 in zip(exprs, labels):
                logger.debug(f"        - expr: {i1}: {i2}")

            timelogger.mark("sharrow preamble", True, logger, trace_label)

            sh_util, sh_flow = apply_flow(
                spec_sh,
                df,
                locals_d,
                trace_label,
                interacts=extra_data,
                zone_layer=zone_layer,
            )
            if sh_util is not None:
                chunk.log_df(trace_label, "sh_util", sh_util)
                utilities = pd.DataFrame(
                    {"utility": sh_util.reshape(-1)},
                    index=df.index if extra_data is None else None,
                )
                chunk.log_df(trace_label, "sh_util", None)  # hand off to caller

            timelogger.mark("sharrow flow", True, logger, trace_label)
        else:
            sh_util, sh_flow = None, None
            timelogger.mark("sharrow flow", False)

        if (
            utilities is None
            or estimator
            or (sharrow_enabled == "test" and extra_data is None)
        ):

            def to_series(x):
                if np.isscalar(x):
                    return pd.Series([x] * len(df), index=df.index)
                if isinstance(x, np.ndarray):
                    return pd.Series(x, index=df.index)
                return x

            if trace_rows is not None and trace_rows.any():
                # # convert to numpy array so we can slice ndarrays as well as series
                # trace_rows = np.asanyarray(trace_rows)
                assert type(trace_rows) == np.ndarray
                trace_eval_results = OrderedDict()
            else:
                trace_eval_results = None

            check_for_variability = config.setting("check_for_variability")

            # need to be able to identify which variables causes an error, which keeps
            # this from being expressed more parsimoniously

            utilities = pd.DataFrame({"utility": 0.0}, index=df.index)

            chunk.log_df(trace_label, "eval.utilities", utilities)

            no_variability = has_missing_vals = 0

            if estimator:
                # ensure alt_id from interaction_dataset is available in expression_values_df for
                # estimator.write_interaction_expression_values and eventual omnibus table assembly
                alt_id = estimator.get_alt_id()
                assert alt_id in df.columns
                expression_values_df = df[[alt_id]]

                # FIXME estimation_requires_chooser_id_in_df_column
                # estimation requires that chooser_id is either in index or a column of interaction_dataset
                # so it can be reformatted (melted) and indexed by chooser_id and alt_id
                # we assume caller has this under control if index is named
                # bug - location choice has df index_name zone_id but should be person_id????
                if df.index.name is None:
                    chooser_id = estimator.get_chooser_id()
                    assert chooser_id in df.columns, (
                        "Expected to find choose_id column '%s' in interaction dataset"
                        % (chooser_id,)
                    )
                    assert df.index.name is None
                    expression_values_df[chooser_id] = df[chooser_id]

            if isinstance(spec.index, pd.MultiIndex):
                exprs = spec.index.get_level_values(simulate.SPEC_EXPRESSION_NAME)
                labels = spec.index.get_level_values(simulate.SPEC_LABEL_NAME)
            else:
                exprs = spec.index
                labels = spec.index

            for expr, label, coefficient in zip(exprs, labels, spec.iloc[:, 0]):
                try:

                    # - allow temps of form _od_DIST@od_skim['DIST']
                    if expr.startswith("_"):

                        target = expr[: expr.index("@")]
                        rhs = expr[expr.index("@") + 1 :]
                        v = to_series(eval(rhs, globals(), locals_d))

                        # update locals to allows us to ref previously assigned targets
                        locals_d[target] = v
                        chunk.log_df(
                            trace_label, target, v
                        )  # track temps stored in locals

                        if trace_eval_results is not None:
                            trace_eval_results[expr] = v[trace_rows]

                        # don't add temps to utility sums
                        # they have a non-zero dummy coefficient to avoid being removed from spec as NOPs
                        continue

                    if expr.startswith("@"):
                        v = to_series(eval(expr[1:], globals(), locals_d))
                    else:
                        v = df.eval(expr, resolvers=[locals_d])

                    if check_for_variability and v.std() == 0:
                        logger.info(
                            "%s: no variability (%s) in: %s"
                            % (trace_label, v.iloc[0], expr)
                        )
                        no_variability += 1

                    # FIXME - how likely is this to happen? Not sure it is really a problem?
                    if (
                        check_for_variability
                        and np.count_nonzero(v.isnull().values) > 0
                    ):
                        logger.info("%s: missing values in: %s" % (trace_label, expr))
                        has_missing_vals += 1

                    if estimator:
                        # in case we modified expression_values_df index
                        expression_values_df.insert(
                            loc=len(expression_values_df.columns),
                            column=label,
                            value=v.values if isinstance(v, pd.Series) else v,
                        )

                    utility = (v * coefficient).astype("float")

                    if log_alt_losers:

                        assert ALT_CHOOSER_ID in df
                        max_utils_by_chooser = utility.groupby(df[ALT_CHOOSER_ID]).max()

                        if (max_utils_by_chooser < simulate.ALT_LOSER_UTIL).any():

                            losers = max_utils_by_chooser[
                                max_utils_by_chooser < simulate.ALT_LOSER_UTIL
                            ]
                            logger.warning(
                                f"{trace_label} - {len(losers)} choosers of {len(max_utils_by_chooser)} "
                                f"with prohibitive utilities for all alternatives for expression: {expr}"
                            )

                            # loser_df = df[df[ALT_CHOOSER_ID].isin(losers.index)]
                            # print(f"\nloser_df\n{loser_df}\n")
                            # print(f"\nloser_max_utils_by_chooser\n{losers}\n")
                            # bug

                        del max_utils_by_chooser

                    utilities.utility.values[:] += utility

                    if trace_eval_results is not None:

                        # expressions should have been uniquified when spec was read
                        # (though we could do it here if need be...)
                        # expr = assign.uniquify_key(trace_eval_results, expr, template="{} # ({})")
                        assert expr not in trace_eval_results

                        trace_eval_results[expr] = v[trace_rows]
                        k = "partial utility (coefficient = %s) for %s" % (
                            coefficient,
                            expr,
                        )
                        trace_eval_results[k] = v[trace_rows] * coefficient

                    del v
                    # chunk.log_df(trace_label, 'v', None)

                except Exception as err:
                    logger.exception(
                        f"{trace_label} - {type(err).__name__} ({str(err)}) evaluating: {str(expr)}"
                    )
                    raise err

            if estimator:
                estimator.log(
                    "eval_interaction_utilities write_interaction_expression_values %s"
                    % trace_label
                )
                estimator.write_interaction_expression_values(expression_values_df)
                del expression_values_df

            if no_variability > 0:
                logger.warning(
                    "%s: %s columns have no variability" % (trace_label, no_variability)
                )

            if has_missing_vals > 0:
                logger.warning(
                    "%s: %s columns have missing values"
                    % (trace_label, has_missing_vals)
                )

            if trace_eval_results is not None:
                trace_eval_results["total utility"] = utilities.utility[trace_rows]

                trace_eval_results = pd.DataFrame.from_dict(trace_eval_results)
                trace_eval_results.index = df[trace_rows].index

                # add df columns to trace_results
                trace_eval_results = pd.concat(
                    [df[trace_rows], trace_eval_results], axis=1
                )
                chunk.log_df(trace_label, "eval.trace_eval_results", trace_eval_results)

            chunk.log_df(trace_label, "v", None)
            chunk.log_df(trace_label, "eval.utilities", None)  # out of out hands...
            chunk.log_df(trace_label, "eval.trace_eval_results", None)

            timelogger.mark("regular interact flow", True, logger, trace_label)
        else:
            timelogger.mark("regular interact flow", False)

        #
        #   Sharrow tracing
        #
        if sh_flow is not None and trace_rows is not None and trace_rows.any():
            assert type(trace_rows) == np.ndarray
            sh_utility_fat = sh_flow.load_dataarray(
                # sh_flow.tree.replace_datasets(
                #     df=df.iloc[trace_rows],
                # ),
                dtype=np.float32,
            )
            sh_utility_fat = sh_utility_fat[trace_rows, :]
            sh_utility_fat = sh_utility_fat.to_dataframe("vals")
            try:
                sh_utility_fat = sh_utility_fat.unstack("expressions")
            except ValueError:
                exprs = sh_utility_fat.index.levels[-1]
                sh_utility_fat = pd.DataFrame(
                    sh_utility_fat.values.reshape(-1, len(exprs)),
                    index=sh_utility_fat.index[:: len(exprs)].droplevel(-1),
                    columns=exprs,
                )
            else:
                sh_utility_fat = sh_utility_fat.droplevel(0, axis=1)
            sh_utility_fat.add_prefix("SH:")
            sh_utility_fat_coef = sh_utility_fat * spec.iloc[:, 0].values.reshape(1, -1)
            sh_utility_fat_coef.columns = [
                f"{i} * ({j})"
                for i, j in zip(sh_utility_fat_coef.columns, spec.iloc[:, 0].values)
            ]
            if utilities.shape[0] > trace_rows.shape[0]:
                trace_rows_ = np.repeat(
                    trace_rows, utilities.shape[0] // trace_rows.shape[0]
                )
            else:
                trace_rows_ = trace_rows
            if trace_eval_results is None:
                trace_eval_results = pd.concat(
                    [
                        sh_utility_fat,
                        sh_utility_fat_coef,
                        utilities.utility[trace_rows_]
                        .rename("total utility")
                        .to_frame()
                        .set_index(sh_utility_fat.index),
                    ],
                    axis=1,
                )
                try:
                    trace_eval_results.index = df[trace_rows].index
                except ValueError:
                    pass
                chunk.log_df(trace_label, "eval.trace_eval_results", trace_eval_results)
            else:
                # in test mode, trace from non-sharrow exists
                trace_eval_results = pd.concat(
                    [
                        trace_eval_results.reset_index(drop=True),
                        sh_utility_fat.reset_index(drop=True),
                        sh_utility_fat_coef.reset_index(drop=True),
                        utilities.utility[trace_rows_]
                        .rename("total utility")
                        .reset_index(drop=True),
                    ],
                    axis=1,
                )
                trace_eval_results.index = df[trace_rows].index
                chunk.log_df(trace_label, "eval.trace_eval_results", trace_eval_results)

            # sh_utility_fat1 = np.dot(sh_utility_fat, spec.values)
            # sh_utility_fat2 = sh_flow.dot(
            #     source=sh_flow.tree.replace_datasets(
            #         df=df.iloc[trace_rows],
            #     ),
            #     coefficients=spec.values.astype(np.float32),
            #     dtype=np.float32,
            # )
            timelogger.mark("sharrow interact trace", True, logger, trace_label)

        if sharrow_enabled == "test":

            try:
                if sh_util is not None:
                    np.testing.assert_allclose(
                        sh_util.reshape(utilities.values.shape),
                        utilities.values,
                        rtol=1e-2,
                        atol=0,
                        err_msg="utility not aligned",
                        verbose=True,
                    )
            except AssertionError as err:
                print(err)
                misses = np.where(
                    ~np.isclose(sh_util, utilities.values, rtol=1e-2, atol=0)
                )
                _sh_util_miss1 = sh_util[tuple(m[0] for m in misses)]
                _u_miss1 = utilities.values[tuple(m[0] for m in misses)]
                diff = _sh_util_miss1 - _u_miss1
                if len(misses[0]) > sh_util.size * 0.01:
                    print("big problem")
                    if "nan location mismatch" in str(err):
                        print("nan location mismatch sh_util")
                        print(np.where(np.isnan(sh_util)))
                        print("nan location mismatch legacy util")
                        print(np.where(np.isnan(utilities.values)))
                    print("misses =>", misses)
                    j = 0
                    while j < len(misses[0]):
                        print(
                            f"miss {j} {tuple(m[j] for m in misses)}:",
                            sh_util[tuple(m[j] for m in misses)],
                            "!=",
                            utilities.values[tuple(m[j] for m in misses)],
                        )
                        j += 1
                        if j > 10:
                            break

                    re_trace = misses[0]
                    retrace_eval_data = {}
                    retrace_eval_parts = {}
                    re_trace_df = df.iloc[re_trace]

                    for expr, label, coefficient in zip(exprs, labels, spec.iloc[:, 0]):
                        if expr.startswith("_"):
                            target = expr[: expr.index("@")]
                            rhs = expr[expr.index("@") + 1 :]
                            v = to_series(eval(rhs, globals(), locals_d))
                            locals_d[target] = v
                            if trace_eval_results is not None:
                                trace_eval_results[expr] = v.iloc[re_trace]
                            continue
                        if expr.startswith("@"):
                            v = to_series(eval(expr[1:], globals(), locals_d))
                        else:
                            v = df.eval(expr)
                        if check_for_variability and v.std() == 0:
                            logger.info(
                                "%s: no variability (%s) in: %s"
                                % (trace_label, v.iloc[0], expr)
                            )
                            no_variability += 1
                        retrace_eval_data[expr] = v.iloc[re_trace]
                        k = "partial utility (coefficient = %s) for %s" % (
                            coefficient,
                            expr,
                        )
                        retrace_eval_parts[k] = (v.iloc[re_trace] * coefficient).astype(
                            "float"
                        )
                    retrace_eval_data_ = pd.concat(retrace_eval_data, axis=1)
                    retrace_eval_parts_ = pd.concat(retrace_eval_parts, axis=1)

                    re_sh_flow_load = sh_flow.load(
                        dtype=np.float32,
                    )
                    re_sh_flow_load_ = re_sh_flow_load[re_trace]

                    look_for_problems_here = np.where(
                        ~np.isclose(
                            re_sh_flow_load_[
                                :, ~spec.index.get_level_values(0).str.startswith("_")
                            ],
                            retrace_eval_data_.values.astype(np.float32),
                        )
                    )

                    raise  # enter debugger now to see what's up
            timelogger.mark("sharrow interact test", True, logger, trace_label)

    logger.info(f"utilities.dtypes {trace_label}\n{utilities.dtypes}")
    end_time = time.time()

    timelogger.summary(logger, "TIMING interact_simulate.eval_utils")
    logger.info(
        f"interact_simulate.eval_utils runtime: {timedelta(seconds=end_time - start_time)} {trace_label}"
    )

    return utilities, trace_eval_results


def _interaction_simulate(
    choosers,
    alternatives,
    spec,
    skims=None,
    locals_d=None,
    sample_size=None,
    trace_label=None,
    trace_choice_name=None,
    log_alt_losers=False,
    estimator=None,
):
    """
    Run a MNL simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    Parameters are same as for public function interaction_simulate

    spec : dataframe
        one row per spec expression and one col with utility coefficient

    interaction_df : dataframe
        cross join (cartesian product) of choosers with alternatives
        combines columns of choosers and alternatives
        len(df) == len(choosers) * len(alternatives)
        index values (non-unique) are index values from alternatives df

    interaction_utilities : dataframe
        the utility of each alternative is sum of the partial utilities determined by the
        various spec expressions and their corresponding coefficients
        yielding a dataframe  with len(interaction_df) rows and one utility column
        having the same index as interaction_df (non-unique values from alternatives df)

    utilities : dataframe
        dot product of model_design.dot(spec)
        yields utility value for element in the cross product of choosers and alternatives
        this is then reshaped as a dataframe with one row per chooser and one column per alternative

    probs : dataframe
        utilities exponentiated and converted to probabilities
        same shape as utilities, one row per chooser and one column for alternative

    positions : series
        choices among alternatives with the chosen alternative represented
        as the integer index of the selected alternative column in probs

    choices : series
        series with the alternative chosen for each chooser
        the index is same as choosers
        and the series value is the alternative df index of chosen alternative

    Returns
    -------
    ret : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    trace_label = tracing.extend_trace_label(trace_label, "interaction_simulate")
    have_trace_targets = tracing.has_trace_targets(choosers)

    if have_trace_targets:
        tracing.trace_df(choosers, tracing.extend_trace_label(trace_label, "choosers"))
        tracing.trace_df(
            alternatives,
            tracing.extend_trace_label(trace_label, "alternatives"),
            slicer="NONE",
            transpose=False,
        )

    if len(spec.columns) > 1:
        raise RuntimeError("spec must have only one column")

    sample_size = sample_size or len(alternatives)

    if sample_size > len(alternatives):
        logger.debug(
            "clipping sample size %s to len(alternatives) %s"
            % (sample_size, len(alternatives))
        )
        sample_size = min(sample_size, len(alternatives))

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for the skims dereference below
    if skims is not None and alternatives.index.name not in alternatives:
        alternatives = alternatives.copy()
        alternatives[alternatives.index.name] = alternatives.index

    # cross join choosers and alternatives (cartesian product)
    # for every chooser, there will be a row for each alternative
    # index values (non-unique) are from alternatives df
    alt_index_id = estimator.get_alt_id() if estimator else None
    chooser_index_id = ALT_CHOOSER_ID if log_alt_losers else None

    sharrow_enabled = config.setting("sharrow", False)
    interaction_utilities = None

    if locals_d is not None and locals_d.get("_sharrow_skip", False):
        sharrow_enabled = False

    if (
        sharrow_enabled
        and skims is None
        and not have_trace_targets
        and sample_size == len(alternatives)
    ):
        # no need to create the merged interaction dataset
        # TODO: can we still do this if skims is not None?

        # TODO: re-enable tracing for sharrow so have_trace_targets can be True
        trace_rows = trace_ids = None

        interaction_utilities, trace_eval_results = eval_interaction_utilities(
            spec,
            choosers,
            locals_d,
            trace_label,
            trace_rows,
            estimator=estimator,
            log_alt_losers=log_alt_losers,
            extra_data=alternatives,
        )

        # set this index here as this is how later code extracts the chosen alt id's
        interaction_utilities.index = np.tile(alternatives.index, len(choosers))

        chunk.log_df(trace_label, "interaction_utilities", interaction_utilities)
        # mem.trace_memory_info(f"{trace_label}.init interaction_utilities sh", force_garbage_collect=True)
        if sharrow_enabled == "test" or True:
            interaction_utilities_sh, trace_eval_results_sh = (
                interaction_utilities,
                trace_eval_results,
            )
        else:
            interaction_utilities_sh = trace_eval_results_sh = None

    else:
        interaction_utilities_sh = trace_eval_results_sh = None

    if (
        not sharrow_enabled
        or (sharrow_enabled == "test")
        or interaction_utilities is None
    ):

        interaction_df = logit.interaction_dataset(
            choosers,
            alternatives,
            sample_size,
            alt_index_id=alt_index_id,
            chooser_index_id=chooser_index_id,
        )
        chunk.log_df(trace_label, "interaction_df", interaction_df)

        if skims is not None:
            simulate.set_skim_wrapper_targets(interaction_df, skims)

        # evaluate expressions from the spec multiply by coefficients and sum
        # spec is df with one row per spec expression and one col with utility coefficient
        # column names of model_design match spec index values
        # utilities has utility value for element in the cross product of choosers and alternatives
        # interaction_utilities is a df with one utility column and one row per row in model_design
        if have_trace_targets:
            trace_rows, trace_ids = tracing.interaction_trace_rows(
                interaction_df, choosers, sample_size
            )

            tracing.trace_df(
                interaction_df[trace_rows],
                tracing.extend_trace_label(trace_label, "interaction_df"),
                slicer="NONE",
                transpose=False,
            )
        else:
            trace_rows = trace_ids = None

        interaction_utilities, trace_eval_results = eval_interaction_utilities(
            spec,
            interaction_df,
            locals_d,
            trace_label,
            trace_rows,
            estimator=estimator,
            log_alt_losers=log_alt_losers,
        )
        chunk.log_df(trace_label, "interaction_utilities", interaction_utilities)
        # mem.trace_memory_info(f"{trace_label}.init interaction_utilities", force_garbage_collect=True)

        # print(f"interaction_df {interaction_df.shape}")
        # print(f"interaction_utilities {interaction_utilities.shape}")

        del interaction_df
        chunk.log_df(trace_label, "interaction_df", None)

        if have_trace_targets:
            tracing.trace_interaction_eval_results(
                trace_eval_results,
                trace_ids,
                tracing.extend_trace_label(trace_label, "eval"),
            )

            tracing.trace_df(
                interaction_utilities[trace_rows],
                tracing.extend_trace_label(trace_label, "interaction_utils"),
                slicer="NONE",
                transpose=False,
            )

    # reshape utilities (one utility column and one row per row in model_design)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.values.reshape(len(choosers), sample_size),
        index=choosers.index,
    )
    chunk.log_df(trace_label, "utilities", utilities)

    if have_trace_targets:
        tracing.trace_df(
            utilities,
            tracing.extend_trace_label(trace_label, "utils"),
            column_labels=["alternative", "utility"],
        )

    tracing.dump_df(DUMP, utilities, trace_label, "utilities")

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(
        utilities, trace_label=trace_label, trace_choosers=choosers
    )
    chunk.log_df(trace_label, "probs", probs)

    del utilities
    chunk.log_df(trace_label, "utilities", None)

    if have_trace_targets:
        tracing.trace_df(
            probs,
            tracing.extend_trace_label(trace_label, "probs"),
            column_labels=["alternative", "probability"],
        )

    # make choices
    # positions is series with the chosen alternative represented as a column index in probs
    # which is an integer between zero and num alternatives in the alternative sample
    positions, rands = logit.make_choices(
        probs, trace_label=trace_label, trace_choosers=choosers
    )
    chunk.log_df(trace_label, "positions", positions)
    chunk.log_df(trace_label, "rands", rands)

    # need to get from an integer offset into the alternative sample to the alternative index
    # that is, we want the index value of the row that is offset by <position> rows into the
    # tranche of this choosers alternatives created by cross join of alternatives and choosers
    # offsets is the offset into model_design df of first row of chooser alternatives
    offsets = np.arange(len(positions)) * sample_size
    # resulting Int64Index has one element per chooser row and is in same order as choosers
    choices = interaction_utilities.index.take(positions + offsets)

    # create a series with index from choosers and the index of the chosen alternative
    choices = pd.Series(choices, index=choosers.index)
    chunk.log_df(trace_label, "choices", choices)

    if have_trace_targets:
        tracing.trace_df(
            choices,
            tracing.extend_trace_label(trace_label, "choices"),
            columns=[None, trace_choice_name],
        )
        tracing.trace_df(
            rands,
            tracing.extend_trace_label(trace_label, "rands"),
            columns=[None, "rand"],
        )

    return choices


def interaction_simulate(
    choosers,
    alternatives,
    spec,
    log_alt_losers=False,
    skims=None,
    locals_d=None,
    sample_size=None,
    chunk_size=0,
    trace_label=None,
    trace_choice_name=None,
    estimator=None,
):

    """
    Run a simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    optionally (if chunk_size > 0) iterates over choosers in chunk_size chunks

    Parameters
    ----------
    choosers : pandas.DataFrame
        DataFrame of choosers
    alternatives : pandas.DataFrame
        DataFrame of alternatives - will be merged with choosers, currently
        without sampling
    spec : pandas.DataFrame
        A Pandas DataFrame that gives the specification of the variables to
        compute and the coefficients for each variable.
        Variable specifications must be in the table index and the
        table should have only one column of coefficients.
    skims : Skims object
        The skims object is used to contain multiple matrices of
        origin-destination impedances.  Make sure to also add it to the
        locals_d below in order to access it in expressions.  The *only* job
        of this method in regards to skims is to call set_df with the
        dataframe that comes back from interacting choosers with
        alternatives.  See the skims module for more documentation on how
        the skims object is intended to be used.
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    sample_size : int, optional
        Sample alternatives with sample of given size.  By default is None,
        which does not sample alternatives.
    chunk_size : int
        if chunk_size > 0 iterates over choosers in chunk_size chunks
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices

    Returns
    -------
    choices : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """

    trace_label = tracing.extend_trace_label(trace_label, "interaction_simulate")

    assert len(choosers) > 0

    result_list = []
    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        choosers, chunk_size, trace_label
    ):

        choices = _interaction_simulate(
            chooser_chunk,
            alternatives,
            spec,
            skims=skims,
            locals_d=locals_d,
            sample_size=sample_size,
            trace_label=chunk_trace_label,
            trace_choice_name=trace_choice_name,
            log_alt_losers=log_alt_losers,
            estimator=estimator,
        )

        result_list.append(choices)

        chunk.log_df(trace_label, "result_list", result_list)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

    return choices

# ActivitySim
# See full license in LICENSE.txt.

import logging
import warnings
from builtins import range
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import assign, chunk, config, logit, pathbuilder, pipeline, tracing, util

logger = logging.getLogger(__name__)

SPEC_DESCRIPTION_NAME = "Description"
SPEC_EXPRESSION_NAME = "Expression"
SPEC_LABEL_NAME = "Label"

ALT_LOSER_UTIL = -900


def random_rows(df, n):

    # only sample if df has more than n rows
    if len(df.index) > n:
        prng = pipeline.get_rn_generator().get_global_rng()
        return df.take(prng.choice(len(df), size=n, replace=False))

    else:
        return df


def uniquify_spec_index(spec):

    # uniquify spec index inplace
    # ensure uniqueness of spec index by appending comment with dupe count
    # this allows us to use pandas dot to compute_utilities
    dict = OrderedDict()
    for expr in spec.index:
        dict[assign.uniquify_key(dict, expr, template="{} # ({})")] = expr

    prev_index_name = spec.index.name
    spec.index = list(dict.keys())
    spec.index.name = prev_index_name

    assert spec.index.is_unique


def read_model_alts(file_name, set_index=None):
    file_path = config.config_file_path(file_name)
    df = pd.read_csv(file_path, comment="#")
    if set_index:
        df.set_index(set_index, inplace=True)
    return df


def read_model_spec(file_name):
    """
    Read a CSV model specification into a Pandas DataFrame or Series.

    file_path : str   absolute or relative path to file

    The CSV is expected to have columns for component descriptions
    and expressions, plus one or more alternatives.

    The CSV is required to have a header with column names. For example:

        Description,Expression,alt0,alt1,alt2

    Parameters
    ----------
    model_settings : dict
        name of spec_file is in model_settings['SPEC'] and file is relative to configs
    file_name : str
        file_name id spec file in configs folder

    description_name : str, optional
        Name of the column in `fname` that contains the component description.
    expression_name : str, optional
        Name of the column in `fname` that contains the component expression.

    Returns
    -------
    spec : pandas.DataFrame
        The description column is dropped from the returned data and the
        expression values are set as the table index.
    """

    assert isinstance(file_name, str)
    if not file_name.lower().endswith(".csv"):
        file_name = "%s.csv" % (file_name,)

    file_path = config.config_file_path(file_name)

    try:
        spec = pd.read_csv(file_path, comment="#")
    except Exception as err:
        logger.error(f"read_model_spec error reading {file_path}")
        logger.error(f"read_model_spec error {type(err).__name__}: {str(err)}")
        raise (err)

    spec = spec.dropna(subset=[SPEC_EXPRESSION_NAME])

    # don't need description and set the expression to the index
    if SPEC_DESCRIPTION_NAME in spec.columns:
        spec = spec.drop(SPEC_DESCRIPTION_NAME, axis=1)

    spec = spec.set_index(SPEC_EXPRESSION_NAME).fillna(0)

    # ensure uniqueness of spec index by appending comment with dupe count
    # this allows us to use pandas dot to compute_utilities
    uniquify_spec_index(spec)

    if SPEC_LABEL_NAME in spec:
        spec = spec.set_index(SPEC_LABEL_NAME, append=True)
        assert isinstance(spec.index, pd.MultiIndex)

    return spec


def read_model_coefficients(model_settings=None, file_name=None):
    """
    Read the coefficient file specified by COEFFICIENTS model setting
    """

    if model_settings is None:
        assert file_name is not None
    else:
        assert file_name is None
        assert (
            "COEFFICIENTS" in model_settings
        ), "'COEFFICIENTS' tag not in model_settings in %s" % model_settings.get(
            "source_file_paths"
        )
        file_name = model_settings["COEFFICIENTS"]
        logger.debug(f"read_model_coefficients file_name {file_name}")

    file_path = config.config_file_path(file_name)
    try:
        coefficients = pd.read_csv(file_path, comment="#", index_col="coefficient_name")
    except ValueError:
        logger.exception("Coefficient File Invalid: %s" % str(file_path))
        raise

    if coefficients.index.duplicated().any():
        logger.warning(
            f"duplicate coefficients in {file_path}\n"
            f"{coefficients[coefficients.index.duplicated(keep=False)]}"
        )
        raise RuntimeError(f"duplicate coefficients in {file_path}")

    if coefficients.value.isnull().any():
        logger.warning(
            f"null coefficients in {file_path}\n{coefficients[coefficients.value.isnull()]}"
        )
        raise RuntimeError(f"null coefficients in {file_path}")

    return coefficients


def spec_for_segment(model_settings, spec_id, segment_name, estimator):
    """
    Select spec for specified segment from omnibus spec containing columns for each segment

    Parameters
    ----------
    model_spec : pandas.DataFrame
        omnibus spec file with expressions in index and one column per segment
    segment_name : str
        segment_name that is also column name in model_spec

    Returns
    -------
    pandas.dataframe
        canonical spec file with expressions in index and single column with utility coefficients
    """

    spec_file_name = model_settings[spec_id]
    spec = read_model_spec(file_name=spec_file_name)

    if len(spec.columns) > 1:
        # if spec is segmented
        spec = spec[[segment_name]]
    else:
        # otherwise we expect a single coefficient column
        # doesn't really matter what it is called, but this may catch errors
        assert spec.columns[0] in ["coefficient", segment_name]

    if "COEFFICIENTS" not in model_settings:
        logger.warning(
            f"no coefficient file specified in model_settings for {spec_file_name}"
        )
        try:
            assert (spec.astype(float) == spec).all(axis=None)
        except (ValueError, AssertionError):
            raise RuntimeError(
                f"No coefficient file specified for {spec_file_name} "
                f"but not all spec column values are numeric"
            )

        return spec

    coefficients = read_model_coefficients(model_settings)

    spec = eval_coefficients(spec, coefficients, estimator)

    return spec


def read_model_coefficient_template(model_settings):
    """
    Read the coefficient template specified by COEFFICIENT_TEMPLATE model setting
    """

    assert (
        "COEFFICIENT_TEMPLATE" in model_settings
    ), "'COEFFICIENT_TEMPLATE' not in model_settings in %s" % model_settings.get(
        "source_file_paths"
    )

    coefficients_file_name = model_settings["COEFFICIENT_TEMPLATE"]

    file_path = config.config_file_path(coefficients_file_name)
    try:
        template = pd.read_csv(file_path, comment="#", index_col="coefficient_name")
    except ValueError:
        logger.exception("Coefficient Template File Invalid: %s" % str(file_path))
        raise

    # by convention, an empty cell in the template indicates that
    # the coefficient name should be propogated to across all segments
    # this makes for a more legible template than repeating the identical coefficient name in each column

    # replace missing cell values with coefficient_name from index
    template = template.where(
        ~template.isnull(),
        np.broadcast_to(template.index.values[:, None], template.shape),
    )

    if template.index.duplicated().any():
        dupes = template[template.index.duplicated(keep=False)].sort_index()
        logger.warning(
            f"duplicate coefficient names in {coefficients_file_name}:\n{dupes}"
        )
        assert not template.index.duplicated().any()

    return template


def dump_mapped_coefficients(model_settings):
    """
    dump template_df with coefficient values
    """

    coefficients_df = read_model_coefficients(model_settings)
    template_df = read_model_coefficient_template(model_settings)

    for c in template_df.columns:
        template_df[c] = template_df[c].map(coefficients_df.value)

    coefficients_template_file_name = model_settings["COEFFICIENT_TEMPLATE"]
    file_path = config.output_file_path(coefficients_template_file_name)
    template_df.to_csv(file_path, index=True)
    logger.info(f"wrote mapped coefficient template to {file_path}")

    coefficients_file_name = model_settings["COEFFICIENTS"]
    file_path = config.output_file_path(coefficients_file_name)
    coefficients_df.to_csv(file_path, index=True)
    logger.info(f"wrote raw coefficients to {file_path}")


def get_segment_coefficients(model_settings, segment_name):
    """
    Return a dict mapping generic coefficient names to segment-specific coefficient values

    some specs mode_choice logsums have the same espression values with different coefficients for various segments
    (e.g. eatout, .. ,atwork) and a template file that maps a flat list of coefficients into segment columns.

    This allows us to provide a coefficient file with just the coefficients for a specific segment,
    that works with generic coefficient names in the spec. For instance coef_ivt can take on the values
    of segment-specific coefficients coef_ivt_school_univ, coef_ivt_work, coef_ivt_atwork,...

    ::

        coefficients_df
                                      value constrain
        coefficient_name
        coef_ivt_eatout_escort_...  -0.0175         F
        coef_ivt_school_univ        -0.0224         F
        coef_ivt_work               -0.0134         F
        coef_ivt_atwork             -0.0188         F

        template_df

        coefficient_name     eatout                       school                 school                 work
        coef_ivt             coef_ivt_eatout_escort_...   coef_ivt_school_univ   coef_ivt_school_univ   coef_ivt_work

        For school segment this will return the generic coefficient name with the segment-specific coefficient value
        e.g. {'coef_ivt': -0.0224, ...}
        ...

    """

    if "COEFFICIENTS" in model_settings and "COEFFICIENT_TEMPLATE" in model_settings:
        legacy = False
    elif "COEFFICIENTS" in model_settings:
        legacy = "COEFFICIENTS"
        warnings.warn(
            "Support for COEFFICIENTS without COEFFICIENT_TEMPLATE in model settings file will be removed."
            "Use COEFFICIENT and COEFFICIENT_TEMPLATE to support estimation.",
            FutureWarning,
        )
    elif "LEGACY_COEFFICIENTS" in model_settings:
        legacy = "LEGACY_COEFFICIENTS"
        warnings.warn(
            "Support for 'LEGACY_COEFFICIENTS' setting in model settings file will be removed."
            "Use COEFFICIENT and COEFFICIENT_TEMPLATE to support estimation.",
            FutureWarning,
        )
    else:
        raise RuntimeError(f"No COEFFICIENTS setting in model_settings")

    if legacy:
        constants = config.get_model_constants(model_settings)
        legacy_coeffs_file_path = config.config_file_path(model_settings[legacy])
        omnibus_coefficients = pd.read_csv(
            legacy_coeffs_file_path, comment="#", index_col="coefficient_name"
        )
        coefficients_dict = assign.evaluate_constants(
            omnibus_coefficients[segment_name], constants=constants
        )
    else:
        coefficients_df = read_model_coefficients(model_settings)
        template_df = read_model_coefficient_template(model_settings)
        coefficients_col = (
            template_df[segment_name].map(coefficients_df.value).astype(float)
        )

        if coefficients_col.isnull().any():
            # show them the offending lines from interaction_coefficients_file
            logger.warning(
                f"bad coefficients in COEFFICIENTS {model_settings['COEFFICIENTS']}\n"
                f"{coefficients_col[coefficients_col.isnull()]}"
            )
            assert not coefficients_col.isnull().any()

        coefficients_dict = coefficients_col.to_dict()

    return coefficients_dict


def eval_nest_coefficients(nest_spec, coefficients, trace_label):
    def replace_coefficients(nest):
        if isinstance(nest, dict):

            assert "coefficient" in nest
            coefficient_name = nest["coefficient"]
            if isinstance(coefficient_name, str):
                assert (
                    coefficient_name in coefficients
                ), "%s not in nest coefficients" % (coefficient_name,)
                nest["coefficient"] = coefficients[coefficient_name]

            assert "alternatives" in nest
            for alternative in nest["alternatives"]:
                if isinstance(alternative, dict):
                    replace_coefficients(alternative)

    if isinstance(coefficients, pd.DataFrame):
        assert "value" in coefficients.columns
        coefficients = coefficients["value"].to_dict()

    replace_coefficients(nest_spec)

    logit.validate_nest_spec(nest_spec, trace_label)

    return nest_spec


def eval_coefficients(spec, coefficients, estimator):

    spec = spec.copy()  # don't clobber input spec

    if isinstance(coefficients, pd.DataFrame):
        assert "value" in coefficients.columns
        coefficients = coefficients["value"].to_dict()

    assert isinstance(
        coefficients, dict
    ), "eval_coefficients doesn't grok type of coefficients: %s" % (type(coefficients))

    for c in spec.columns:
        if c == SPEC_LABEL_NAME:
            continue
        spec[c] = (
            spec[c].apply(lambda x: eval(str(x), {}, coefficients)).astype(np.float32)
        )

    # drop any rows with all zeros since they won't have any effect (0 marginal utility)
    # (do not drop rows in estimation mode as it may confuse the estimation package (e.g. larch)
    zero_rows = (spec == 0).all(axis=1)
    if zero_rows.any():
        if estimator:
            logger.debug("keeping %s all-zero rows in SPEC" % (zero_rows.sum(),))
        else:
            logger.debug("dropping %s all-zero rows from SPEC" % (zero_rows.sum(),))
            spec = spec.loc[~zero_rows]

    return spec


def eval_utilities(
    spec,
    choosers,
    locals_d=None,
    trace_label=None,
    have_trace_targets=False,
    trace_all_rows=False,
    estimator=None,
    trace_column_names=None,
    log_alt_losers=False,
):
    """

    Parameters
    ----------
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    choosers : pandas.DataFrame
    locals_d : Dict or None
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    trace_label: str
    have_trace_targets: boolean - choosers has targets to trace
    trace_all_rows: boolean - trace all chooser rows, bypassing tracing.trace_targets
    estimator :
        called to report intermediate table results (used for estimation)
    trace_column_names: str or list of str
        chooser columns to include when tracing expression_values

    Returns
    -------

    """

    # fixme - restore tracing and _check_for_variability

    trace_label = tracing.extend_trace_label(trace_label, "eval_utils")

    # avoid altering caller's passed-in locals_d parameter (they may be looping)
    locals_dict = assign.local_utilities()

    if locals_d is not None:
        locals_dict.update(locals_d)
    globals_dict = {}

    locals_dict["df"] = choosers

    # - eval spec expressions
    if isinstance(spec.index, pd.MultiIndex):
        # spec MultiIndex with expression and label
        exprs = spec.index.get_level_values(SPEC_EXPRESSION_NAME)
    else:
        exprs = spec.index

    expression_values = np.empty((spec.shape[0], choosers.shape[0]))
    chunk.log_df(trace_label, "expression_values", expression_values)

    i = 0
    for expr, coefficients in zip(exprs, spec.values):

        try:
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                if expr.startswith("@"):
                    expression_value = eval(expr[1:], globals_dict, locals_dict)

                else:
                    expression_value = choosers.eval(expr)

                if len(w) > 0:
                    for wrn in w:
                        logger.warning(
                            f"{trace_label} - {type(wrn).__name__} ({wrn.message}) evaluating: {str(expr)}"
                        )

        except Exception as err:
            logger.exception(
                f"{trace_label} - {type(err).__name__} ({str(err)}) evaluating: {str(expr)}"
            )
            raise err

        if log_alt_losers:
            # utils for each alt for this expression
            # FIXME if we always did tis, we cold uem these and skip np.dot below
            utils = np.outer(expression_value, coefficients)
            losers = np.amax(utils, axis=1) < ALT_LOSER_UTIL

            if losers.any():
                logger.warning(
                    f"{trace_label} - {sum(losers)} choosers of {len(losers)} "
                    f"with prohibitive utilities for all alternatives for expression: {expr}"
                )

        expression_values[i] = expression_value
        i += 1

    chunk.log_df(trace_label, "expression_values", expression_values)

    if estimator:
        df = pd.DataFrame(
            data=expression_values.transpose(),
            index=choosers.index,
            columns=spec.index.get_level_values(SPEC_LABEL_NAME),
        )
        df.index.name = choosers.index.name
        estimator.write_expression_values(df)

    # - compute_utilities
    utilities = np.dot(expression_values.transpose(), spec.astype(np.float64).values)
    utilities = pd.DataFrame(data=utilities, index=choosers.index, columns=spec.columns)

    chunk.log_df(trace_label, "utilities", utilities)

    # sometimes tvpb will drop rows on the fly and we wind up with an empty
    # table of choosers. this will just bypass tracing in that case.
    if (trace_all_rows or have_trace_targets) and (len(choosers) > 0):

        if trace_all_rows:
            trace_targets = pd.Series(True, index=choosers.index)
        else:
            trace_targets = tracing.trace_targets(choosers)
            assert trace_targets.any()  # since they claimed to have targets...

        # get int offsets of the trace_targets (offsets of bool=True values)
        offsets = np.nonzero(list(trace_targets))[0]

        # get array of expression_values
        # expression_values.shape = (len(spec), len(choosers))
        # data.shape = (len(spec), len(offsets))
        data = expression_values[:, offsets]

        # index is utility expressions (and optional label if MultiIndex)
        expression_values_df = pd.DataFrame(data=data, index=spec.index)

        if trace_column_names is not None:
            if isinstance(trace_column_names, str):
                trace_column_names = [trace_column_names]
            expression_values_df.columns = pd.MultiIndex.from_frame(
                choosers.loc[trace_targets, trace_column_names]
            )

        tracing.trace_df(
            expression_values_df,
            tracing.extend_trace_label(trace_label, "expression_values"),
            slicer=None,
            transpose=False,
        )

        if len(spec.columns) > 1:

            for c in spec.columns:
                name = f"expression_value_{c}"

                tracing.trace_df(
                    expression_values_df.multiply(spec[c].values, axis=0),
                    tracing.extend_trace_label(trace_label, name),
                    slicer=None,
                    transpose=False,
                )

    del expression_values
    chunk.log_df(trace_label, "expression_values", None)

    # no longer our problem - but our caller should re-log this...
    chunk.log_df(trace_label, "utilities", None)

    return utilities


def eval_variables(exprs, df, locals_d=None):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    There are two kinds of supported expressions: "simple" expressions are
    evaluated in the context of the DataFrame using DataFrame.eval.
    This is the default type of expression.

    Python expressions are evaluated in the context of this function using
    Python's eval function. Because we use Python's eval this type of
    expression supports more complex operations than a simple expression.
    Python expressions are denoted by beginning with the @ character.
    Users should take care that these expressions must result in
    a Pandas Series.

    # FIXME - for performance, it is essential that spec and expression_values
    # FIXME - not contain booleans when dotted with spec values
    # FIXME - or the arrays will be converted to dtype=object within dot()

    Parameters
    ----------
    exprs : sequence of str
    df : pandas.DataFrame
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @

    Returns
    -------
    variables : pandas.DataFrame
        Will have the index of `df` and columns of eval results of `exprs`.
    """

    # avoid altering caller's passed-in locals_d parameter (they may be looping)
    locals_dict = assign.local_utilities()
    if locals_d is not None:
        locals_dict.update(locals_d)
    globals_dict = {}

    locals_dict["df"] = df

    def to_array(x):

        if x is None or np.isscalar(x):
            a = np.asanyarray([x] * len(df.index))
        elif isinstance(x, pd.Series):
            # fixme
            # assert x.index.equals(df.index)
            # save a little RAM
            a = x.values
        else:
            a = x

        # FIXME - for performance, it is essential that spec and expression_values
        # FIXME - not contain booleans when dotted with spec values
        # FIXME - or the arrays will be converted to dtype=object within dot()
        if not np.issubdtype(a.dtype, np.number):
            a = a.astype(np.int8)

        return a

    values = OrderedDict()
    for expr in exprs:
        try:
            if expr.startswith("@"):
                expr_values = to_array(eval(expr[1:], globals_dict, locals_dict))
            else:
                expr_values = to_array(df.eval(expr))
            # read model spec should ensure uniqueness, otherwise we should uniquify
            assert expr not in values
            values[expr] = expr_values

        except Exception as err:
            logger.exception(
                f"Variable evaluation failed {type(err).__name__} ({str(err)}) evaluating: {str(expr)}"
            )
            raise err

    values = util.df_from_dict(values, index=df.index)

    return values


# no longer used because eval_utilities aggregates expression_values as they are computed to save space
# def compute_utilities(expression_values, spec):
#
#     # matrix product of spec expression_values with utility coefficients of alternatives
#     # sums the partial utilities (represented by each spec row) of the alternatives
#     # resulting in a dataframe with one row per chooser and one column per alternative
#     # pandas.dot depends on column names of expression_values matching spec index values
#
#     # FIXME - for performance, it is essential that spec and expression_values
#     # FIXME - not contain booleans when dotted with spec values
#     # FIXME - or the arrays will be converted to dtype=object within dot()
#
#     spec = spec.astype(np.float64)
#
#     # pandas.dot depends on column names of expression_values matching spec index values
#     # expressions should have been uniquified when spec was read
#     # we could do it here if need be, and then set spec.index and expression_values.columns equal
#     assert spec.index.is_unique
#     assert (spec.index.values == expression_values.columns.values).all()
#
#     utilities = expression_values.dot(spec)
#
#     return utilities


def set_skim_wrapper_targets(df, skims):
    """
    Add the dataframe to the SkimWrapper object so that it can be dereferenced
    using the parameters of the skims object.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to which to add skim data as new columns.
        `df` is modified in-place.
    skims : SkimWrapper or Skim3dWrapper object, or a list or dict of skims
        The skims object is used to contain multiple matrices of
        origin-destination impedances.  Make sure to also add it to the
        locals_d below in order to access it in expressions.  The *only* job
        of this method in regards to skims is to call set_df with the
        dataframe that comes back from interacting choosers with
        alternatives.  See the skims module for more documentation on how
        the skims object is intended to be used.
    """

    skims = (
        skims
        if isinstance(skims, list)
        else skims.values()
        if isinstance(skims, dict)
        else [skims]
    )

    # assume any object in skims can be treated as a skim
    for skim in skims:
        try:
            skim.set_df(df)
        except AttributeError:
            pass


def _check_for_variability(expression_values, trace_label):
    """
    This is an internal method which checks for variability in each
    expression - under the assumption that you probably wouldn't be using a
    variable (in live simulations) if it had no variability.  This is a
    warning to the user that they might have constructed the variable
    incorrectly.  It samples 1000 rows in order to not hurt performance -
    it's likely that if 1000 rows have no variability, the whole dataframe
    will have no variability.
    """

    if trace_label is None:
        trace_label = "_check_for_variability"

    sample = random_rows(expression_values, min(1000, len(expression_values)))

    no_variability = has_missing_vals = 0
    for i in range(len(sample.columns)):
        v = sample.iloc[:, i]
        if v.min() == v.max():
            col_name = sample.columns[i]
            logger.info(
                "%s: no variability (%s) in: %s" % (trace_label, v.iloc[0], col_name)
            )
            no_variability += 1
        # FIXME - how could this happen? Not sure it is really a problem?
        if np.count_nonzero(v.isnull().values) > 0:
            col_name = sample.columns[i]
            logger.info("%s: missing values in: %s" % (trace_label, col_name))
            has_missing_vals += 1

    if no_variability > 0:
        logger.warning(
            "%s: %s columns have no variability" % (trace_label, no_variability)
        )

    if has_missing_vals > 0:
        logger.warning(
            "%s: %s columns have missing values" % (trace_label, has_missing_vals)
        )


def compute_nested_exp_utilities(raw_utilities, nest_spec):
    """
    compute exponentiated nest utilities based on nesting coefficients

    For nest nodes this is the exponentiated logsum of alternatives adjusted by nesting coefficient

    leaf <- exp( raw_utility )
    nest <- exp( ln(sum of exponentiated raw_utility of leaves) * nest_coefficient)

    Parameters
    ----------
    raw_utilities : pandas.DataFrame
        dataframe with the raw alternative utilities of all leaves
        (what in non-nested logit would be the utilities of all the alternatives)
    nest_spec : dict
        Nest tree dict from the model spec yaml file

    Returns
    -------
    nested_utilities : pandas.DataFrame
        Will have the index of `raw_utilities` and columns for exponentiated leaf and node utilities
    """
    nested_utilities = pd.DataFrame(index=raw_utilities.index)

    for nest in logit.each_nest(nest_spec, post_order=True):

        name = nest.name

        if nest.is_leaf:
            # leaf_utility = raw_utility / nest.product_of_coefficients
            nested_utilities[name] = (
                raw_utilities[name].astype(float) / nest.product_of_coefficients
            )

        else:
            # nest node
            # the alternative nested_utilities will already have been computed due to post_order
            # this will RuntimeWarning: divide by zero encountered in log
            # if all nest alternative utilities are zero
            # but the resulting inf will become 0 when exp is applied below
            with np.errstate(divide="ignore"):
                nested_utilities[name] = nest.coefficient * np.log(
                    nested_utilities[nest.alternatives].sum(axis=1)
                )

        # exponentiate the utility
        nested_utilities[name] = np.exp(nested_utilities[name])

    return nested_utilities


def compute_nested_probabilities(nested_exp_utilities, nest_spec, trace_label):
    """
    compute nested probabilities for nest leafs and nodes
    probability for nest alternatives is simply the alternatives's local (to nest) probability
    computed in the same way as the probability of non-nested alternatives in multinomial logit
    i.e. the fractional share of the sum of the exponentiated utility of itself and its siblings
    except in nested logit, its sib group is restricted to the nest

    Parameters
    ----------
    nested_exp_utilities : pandas.DataFrame
        dataframe with the exponentiated nested utilities of all leaves and nodes
    nest_spec : dict
        Nest tree dict from the model spec yaml file
    Returns
    -------
    nested_probabilities : pandas.DataFrame
        Will have the index of `nested_exp_utilities` and columns for leaf and node probabilities
    """

    nested_probabilities = pd.DataFrame(index=nested_exp_utilities.index)

    for nest in logit.each_nest(nest_spec, type="node", post_order=False):

        probs = logit.utils_to_probs(
            nested_exp_utilities[nest.alternatives],
            trace_label=trace_label,
            exponentiated=True,
            allow_zero_probs=True,
        )

        nested_probabilities = pd.concat([nested_probabilities, probs], axis=1)

    return nested_probabilities


def compute_base_probabilities(nested_probabilities, nests, spec):
    """
    compute base probabilities for nest leaves
    Base probabilities will be the nest-adjusted probabilities of all leaves
    This flattens or normalizes all the nested probabilities so that they have the proper global
    relative values (the leaf probabilities sum to 1 for each row.)

    Parameters
    ----------
    nested_probabilities : pandas.DataFrame
        dataframe with the nested probabilities for nest leafs and nodes
    nests : dict
        Nest tree dict from the model spec yaml file
    spec : pandas.Dataframe
        simple simulate spec so we can return columns in appropriate order
    Returns
    -------
    base_probabilities : pandas.DataFrame
        Will have the index of `nested_probabilities` and columns for leaf base probabilities
    """

    base_probabilities = pd.DataFrame(index=nested_probabilities.index)

    for nest in logit.each_nest(nests, type="leaf", post_order=False):

        # skip root: it has a prob of 1 but we didn't compute a nested probability column for it
        ancestors = nest.ancestors[1:]

        base_probabilities[nest.name] = nested_probabilities[ancestors].prod(axis=1)

    # reorder alternative columns to match spec
    # since these are alternatives chosen by column index, order of columns matters
    assert set(base_probabilities.columns) == set(spec.columns)
    base_probabilities = base_probabilities[spec.columns]

    return base_probabilities


def eval_mnl(
    choosers,
    spec,
    locals_d,
    custom_chooser,
    estimator,
    log_alt_losers=False,
    want_logsums=False,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
):
    """
    Run a simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Each row in spec computes a partial utility for each alternative,
    by providing a spec expression (often a boolean 0-1 trigger)
    and a column of utility coefficients for each alternative.

    We compute the utility of each alternative by matrix-multiplication of eval results
    with the utility coefficients in the spec alternative columns
    yielding one row per chooser and one column per alternative

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    locals_d : Dict or None
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    custom_chooser : function(probs, choosers, spec, trace_label) returns choices, rands
        custom alternative to logit.make_choices
    estimator : Estimator object
        called to report intermediate table results (used for estimation)
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices
    trace_column_names: str or list of str
        chooser columns to include when tracing expression_values

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.
    """

    # FIXME - not implemented because not currently needed
    assert not want_logsums

    trace_label = tracing.extend_trace_label(trace_label, "eval_mnl")
    have_trace_targets = tracing.has_trace_targets(choosers)

    if have_trace_targets:
        tracing.trace_df(choosers, "%s.choosers" % trace_label)

    utilities = eval_utilities(
        spec,
        choosers,
        locals_d,
        log_alt_losers=log_alt_losers,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
        estimator=estimator,
        trace_column_names=trace_column_names,
    )
    chunk.log_df(trace_label, "utilities", utilities)

    if have_trace_targets:
        tracing.trace_df(
            utilities,
            "%s.utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    probs = logit.utils_to_probs(
        utilities, trace_label=trace_label, trace_choosers=choosers
    )
    chunk.log_df(trace_label, "probs", probs)

    del utilities
    chunk.log_df(trace_label, "utilities", None)

    if have_trace_targets:
        # report these now in case make_choices throws error on bad_choices
        tracing.trace_df(
            probs,
            "%s.probs" % trace_label,
            column_labels=["alternative", "probability"],
        )

    if custom_chooser:
        choices, rands = custom_chooser(
            probs=probs, choosers=choosers, spec=spec, trace_label=trace_label
        )
    else:
        choices, rands = logit.make_choices(probs, trace_label=trace_label)

    del probs
    chunk.log_df(trace_label, "probs", None)

    if have_trace_targets:
        tracing.trace_df(
            choices, "%s.choices" % trace_label, columns=[None, trace_choice_name]
        )
        tracing.trace_df(rands, "%s.rands" % trace_label, columns=[None, "rand"])

    return choices


def eval_nl(
    choosers,
    spec,
    nest_spec,
    locals_d,
    custom_chooser,
    estimator,
    log_alt_losers=False,
    want_logsums=False,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
):
    """
    Run a nested-logit simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec:
        dictionary specifying nesting structure and nesting coefficients
        (from the model spec yaml file)
    locals_d : Dict or None
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    custom_chooser : function(probs, choosers, spec, trace_label) returns choices, rands
        custom alternative to logit.make_choices
    estimator : Estimator object
        called to report intermediate table results (used for estimation)
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices
    trace_column_names: str or list of str
        chooser columns to include when tracing expression_values

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.
    """

    trace_label = tracing.extend_trace_label(trace_label, "eval_nl")
    assert trace_label
    have_trace_targets = tracing.has_trace_targets(choosers)

    logit.validate_nest_spec(nest_spec, trace_label)

    if have_trace_targets:
        tracing.trace_df(choosers, "%s.choosers" % trace_label)

    raw_utilities = eval_utilities(
        spec,
        choosers,
        locals_d,
        log_alt_losers=log_alt_losers,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
        estimator=estimator,
        trace_column_names=trace_column_names,
    )
    chunk.log_df(trace_label, "raw_utilities", raw_utilities)

    if have_trace_targets:
        tracing.trace_df(
            raw_utilities,
            "%s.raw_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)
    chunk.log_df(trace_label, "nested_exp_utilities", nested_exp_utilities)

    del raw_utilities
    chunk.log_df(trace_label, "raw_utilities", None)

    if have_trace_targets:
        tracing.trace_df(
            nested_exp_utilities,
            "%s.nested_exp_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # probabilities of alternatives relative to siblings sharing the same nest
    nested_probabilities = compute_nested_probabilities(
        nested_exp_utilities, nest_spec, trace_label=trace_label
    )
    chunk.log_df(trace_label, "nested_probabilities", nested_probabilities)

    if want_logsums:
        # logsum of nest root
        logsums = pd.Series(np.log(nested_exp_utilities.root), index=choosers.index)
        chunk.log_df(trace_label, "logsums", logsums)

    del nested_exp_utilities
    chunk.log_df(trace_label, "nested_exp_utilities", None)

    if have_trace_targets:
        tracing.trace_df(
            nested_probabilities,
            "%s.nested_probabilities" % trace_label,
            column_labels=["alternative", "probability"],
        )

    # global (flattened) leaf probabilities based on relative nest coefficients (in spec order)
    base_probabilities = compute_base_probabilities(
        nested_probabilities, nest_spec, spec
    )
    chunk.log_df(trace_label, "base_probabilities", base_probabilities)

    del nested_probabilities
    chunk.log_df(trace_label, "nested_probabilities", None)

    if have_trace_targets:
        tracing.trace_df(
            base_probabilities,
            "%s.base_probabilities" % trace_label,
            column_labels=["alternative", "probability"],
        )

    # note base_probabilities could all be zero since we allowed all probs for nests to be zero
    # check here to print a clear message but make_choices will raise error if probs don't sum to 1
    BAD_PROB_THRESHOLD = 0.001
    no_choices = (base_probabilities.sum(axis=1) - 1).abs() > BAD_PROB_THRESHOLD

    if no_choices.any():

        logit.report_bad_choices(
            no_choices,
            base_probabilities,
            trace_label=tracing.extend_trace_label(trace_label, "bad_probs"),
            trace_choosers=choosers,
            msg="base_probabilities do not sum to one",
        )

    if custom_chooser:
        choices, rands = custom_chooser(
            probs=base_probabilities,
            choosers=choosers,
            spec=spec,
            trace_label=trace_label,
        )
    else:
        choices, rands = logit.make_choices(base_probabilities, trace_label=trace_label)

    del base_probabilities
    chunk.log_df(trace_label, "base_probabilities", None)

    if have_trace_targets:
        tracing.trace_df(
            choices, "%s.choices" % trace_label, columns=[None, trace_choice_name]
        )
        tracing.trace_df(rands, "%s.rands" % trace_label, columns=[None, "rand"])
        if want_logsums:
            tracing.trace_df(
                logsums, "%s.logsums" % trace_label, columns=[None, "logsum"]
            )

    if want_logsums:
        choices = choices.to_frame("choice")
        choices["logsum"] = logsums

    return choices


def _simple_simulate(
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    custom_chooser=None,
    log_alt_losers=False,
    want_logsums=False,
    estimator=None,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
):
    """
    Run an MNL or NL simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    nest_spec:
        for nested logit (nl): dictionary specifying nesting structure and nesting coefficients
        for multinomial logit (mnl): None
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
    custom_chooser : Estimator object
    estimator : function(df, label, table_name)
        called to report intermediate table results (used for estimation)

    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices
    trace_column_names: str or list of str
        chooser columns to include when tracing expression_values

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.
    """

    if skims is not None:
        set_skim_wrapper_targets(choosers, skims)

    if nest_spec is None:
        choices = eval_mnl(
            choosers,
            spec,
            locals_d,
            custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=trace_label,
            trace_choice_name=trace_choice_name,
            trace_column_names=trace_column_names,
        )
    else:
        choices = eval_nl(
            choosers,
            spec,
            nest_spec,
            locals_d,
            custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=trace_label,
            trace_choice_name=trace_choice_name,
            trace_column_names=trace_column_names,
        )

    return choices


def tvpb_skims(skims):
    def list_of_skims(skims):
        return (
            skims
            if isinstance(skims, list)
            else skims.values()
            if isinstance(skims, dict)
            else [skims]
            if skims is not None
            else []
        )

    return [
        skim
        for skim in list_of_skims(skims)
        if isinstance(skim, pathbuilder.TransitVirtualPathLogsumWrapper)
    ]


def simple_simulate(
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    chunk_size=0,
    custom_chooser=None,
    log_alt_losers=False,
    want_logsums=False,
    estimator=None,
    trace_label=None,
    trace_choice_name=None,
    trace_column_names=None,
):
    """
    Run an MNL or NL simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.
    """

    trace_label = tracing.extend_trace_label(trace_label, "simple_simulate")

    assert len(choosers) > 0

    result_list = []
    # segment by person type and pick the right spec for each person type
    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        choosers, chunk_size, trace_label
    ):

        choices = _simple_simulate(
            chooser_chunk,
            spec,
            nest_spec,
            skims=skims,
            locals_d=locals_d,
            custom_chooser=custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=chunk_trace_label,
            trace_choice_name=trace_choice_name,
            trace_column_names=trace_column_names,
        )

        result_list.append(choices)

        chunk.log_df(trace_label, f"result_list", result_list)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

    return choices


def simple_simulate_by_chunk_id(
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    chunk_size=0,
    custom_chooser=None,
    log_alt_losers=False,
    want_logsums=False,
    estimator=None,
    trace_label=None,
    trace_choice_name=None,
):
    """
    chunk_by_chunk_id wrapper for simple_simulate
    """

    result_list = []
    for (
        i,
        chooser_chunk,
        chunk_trace_label,
    ) in chunk.adaptive_chunked_choosers_by_chunk_id(choosers, chunk_size, trace_label):

        choices = _simple_simulate(
            chooser_chunk,
            spec,
            nest_spec,
            skims=skims,
            locals_d=locals_d,
            custom_chooser=custom_chooser,
            log_alt_losers=log_alt_losers,
            want_logsums=want_logsums,
            estimator=estimator,
            trace_label=chunk_trace_label,
            trace_choice_name=trace_choice_name,
        )

        result_list.append(choices)

        chunk.log_df(trace_label, f"result_list", result_list)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


def eval_mnl_logsums(choosers, spec, locals_d, trace_label=None):
    """
    like eval_nl except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be logsum across spec column values
    """

    # FIXME - untested and not currently used by any models...

    trace_label = tracing.extend_trace_label(trace_label, "eval_mnl_logsums")
    have_trace_targets = tracing.has_trace_targets(choosers)

    logger.debug("running eval_mnl_logsums")

    # trace choosers
    if have_trace_targets:
        tracing.trace_df(choosers, "%s.choosers" % trace_label)

    utilities = eval_utilities(
        spec, choosers, locals_d, trace_label, have_trace_targets
    )
    chunk.log_df(trace_label, "utilities", utilities)

    if have_trace_targets:
        tracing.trace_df(
            utilities,
            "%s.raw_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # - logsums
    # logsum is log of exponentiated utilities summed across columns of each chooser row
    logsums = np.log(np.exp(utilities.values).sum(axis=1))
    logsums = pd.Series(logsums, index=choosers.index)
    chunk.log_df(trace_label, "logsums", logsums)

    # trace utilities
    if have_trace_targets:
        tracing.trace_df(
            logsums, "%s.logsums" % trace_label, column_labels=["alternative", "logsum"]
        )

    return logsums


def eval_nl_logsums(choosers, spec, nest_spec, locals_d, trace_label=None):
    """
    like eval_nl except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values
    """

    trace_label = tracing.extend_trace_label(trace_label, "eval_nl_logsums")
    have_trace_targets = tracing.has_trace_targets(choosers)

    logit.validate_nest_spec(nest_spec, trace_label)

    # trace choosers
    if have_trace_targets:
        tracing.trace_df(choosers, "%s.choosers" % trace_label)

    raw_utilities = eval_utilities(
        spec,
        choosers,
        locals_d,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
    )
    chunk.log_df(trace_label, "raw_utilities", raw_utilities)

    if have_trace_targets:
        tracing.trace_df(
            raw_utilities,
            "%s.raw_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )

    # - exponentiated utilities of leaves and nests
    nested_exp_utilities = compute_nested_exp_utilities(raw_utilities, nest_spec)
    chunk.log_df(trace_label, "nested_exp_utilities", nested_exp_utilities)

    del raw_utilities  # done with raw_utilities
    chunk.log_df(trace_label, "raw_utilities", None)

    # - logsums
    logsums = np.log(nested_exp_utilities.root)
    logsums = pd.Series(logsums, index=choosers.index)
    chunk.log_df(trace_label, "logsums", logsums)

    if have_trace_targets:
        # add logsum to nested_exp_utilities for tracing
        nested_exp_utilities["logsum"] = logsums
        tracing.trace_df(
            nested_exp_utilities,
            "%s.nested_exp_utilities" % trace_label,
            column_labels=["alternative", "utility"],
        )
        tracing.trace_df(
            logsums, "%s.logsums" % trace_label, column_labels=["alternative", "logsum"]
        )

    del nested_exp_utilities  # done with nested_exp_utilities
    chunk.log_df(trace_label, "nested_exp_utilities", None)

    return logsums


def _simple_simulate_logsums(
    choosers, spec, nest_spec, skims=None, locals_d=None, trace_label=None
):
    """
    like simple_simulate except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values
    """

    if skims is not None:
        set_skim_wrapper_targets(choosers, skims)

    if nest_spec is None:
        logsums = eval_mnl_logsums(choosers, spec, locals_d, trace_label=trace_label)
    else:
        logsums = eval_nl_logsums(
            choosers, spec, nest_spec, locals_d, trace_label=trace_label
        )

    return logsums


def simple_simulate_logsums(
    choosers,
    spec,
    nest_spec,
    skims=None,
    locals_d=None,
    chunk_size=0,
    trace_label=None,
    chunk_tag=None,
):
    """
    like simple_simulate except return logsums instead of making choices

    Returns
    -------
    logsums : pandas.Series
        Index will be that of `choosers`, values will be nest logsum based on spec column values
    """

    assert len(choosers) > 0
    chunk_tag = chunk_tag or trace_label

    result_list = []
    # segment by person type and pick the right spec for each person type
    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        choosers, chunk_size, trace_label, chunk_tag
    ):

        logsums = _simple_simulate_logsums(
            chooser_chunk, spec, nest_spec, skims, locals_d, chunk_trace_label
        )

        result_list.append(logsums)

        chunk.log_df(trace_label, f"result_list", result_list)

    if len(result_list) > 1:
        logsums = pd.concat(result_list)

    assert len(logsums.index == len(choosers.index))

    return logsums

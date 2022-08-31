# ActivitySim
# See full license in LICENSE.txt.
import logging
from builtins import object, zip
from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core import chunk, config, inject, pipeline, util

logger = logging.getLogger(__name__)


def uniquify_key(dict, key, template="{} ({})"):
    """
    rename key so there are no duplicates with keys in dict

    e.g. if there is already a key named "dog", the second key will be reformatted to "dog (2)"
    """
    n = 1
    new_key = key
    while new_key in dict:
        n += 1
        new_key = template.format(key, n)

    return new_key


def evaluate_constants(expressions, constants):
    """
    Evaluate a list of constant expressions - each one can depend on the one before
    it.  These are usually used for the coefficients which have relationships
    to each other.  So ivt=.7 and then ivt_lr=ivt*.9.

    Parameters
    ----------
    expressions : Series
        the index are the names of the expressions which are
        used in subsequent evals - thus naming the expressions is required.
    constants : dict
        will be passed as the scope of eval - usually a separate set of
        constants are passed in here

    Returns
    -------
    d : dict

    """

    # FIXME why copy?
    d = {}
    for k, v in expressions.items():
        try:
            d[k] = eval(str(v), d.copy(), constants)
        except Exception as err:
            print(f"error evaluating {str(v)}")
            raise err

    return d


def read_assignment_spec(
    file_name,
    description_name="Description",
    target_name="Target",
    expression_name="Expression",
):
    """
    Read a CSV model specification into a Pandas DataFrame or Series.

    The CSV is expected to have columns for component descriptions
    targets, and expressions,

    The CSV is required to have a header with column names. For example:

        Description,Target,Expression

    Parameters
    ----------
    file_name : str
        Name of a CSV spec file.
    description_name : str, optional
        Name of the column in `fname` that contains the component description.
    target_name : str, optional
        Name of the column in `fname` that contains the component target.
    expression_name : str, optional
        Name of the column in `fname` that contains the component expression.

    Returns
    -------
    spec : pandas.DataFrame
        dataframe with three columns: ['description' 'target' 'expression']
    """

    try:
        cfg = pd.read_csv(file_name, comment="#")
    except Exception as e:
        logger.error(f"Error reading spec file: {file_name}")
        logger.error(str(e))
        raise e

    # drop null expressions
    # cfg = cfg.dropna(subset=[expression_name])

    cfg.rename(
        columns={
            target_name: "target",
            expression_name: "expression",
            description_name: "description",
        },
        inplace=True,
    )

    # backfill description
    if "description" not in cfg.columns:
        cfg.description = ""

    cfg.target = cfg.target.str.strip()
    cfg.expression = cfg.expression.str.strip()

    return cfg


class NumpyLogger(object):
    def __init__(self, logger):
        self.logger = logger
        self.target = ""
        self.expression = ""

    def write(self, msg):
        self.logger.warning(
            "numpy: %s expression: %s = %s"
            % (msg.rstrip(), str(self.target), str(self.expression))
        )


def local_utilities():
    """
    Dict of useful modules and functions to provides as locals for use in eval of expressions

    Returns
    -------
    utility_dict : dict
        name, entity pairs of locals
    """

    utility_dict = {
        "pd": pd,
        "np": np,
        "reindex": util.reindex,
        "reindex_i": util.reindex_i,
        "setting": config.setting,
        "other_than": util.other_than,
        "rng": pipeline.get_rn_generator(),
    }

    utility_dict.update(config.get_global_constants())

    return utility_dict


def is_throwaway(target):
    return target == "_"


def is_temp_scalar(target):
    return target.startswith("_") and target.isupper()


def is_temp(target):
    return target.startswith("_")


def assign_variables(
    assignment_expressions,
    df,
    locals_dict,
    df_alias=None,
    trace_rows=None,
    trace_label=None,
    chunk_log=None,
):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    Expressions are evaluated using Python's eval function.
    Python expressions have access to variables in locals_d (and df being
    accessible as variable df.) They also have access to previously assigned
    targets as the assigned target name.

    lowercase variables starting with underscore are temp variables (e.g. _local_var)
    and not returned except in trace_results

    uppercase variables starting with underscore are temp singular variables (e.g. _LOCAL_SCALAR)
    and not returned except in trace_assigned_locals
    This is useful for defining general purpose local variables that don't vary across
    choosers or alternatives and therefore don't need to be stored as series/columns
    in the main choosers dataframe from which utilities are computed.

    Users should take care that expressions (other than temp scalar variables) should result in
    a Pandas Series (scalars will be automatically promoted to series.)

    Parameters
    ----------
    assignment_expressions : pandas.DataFrame of target assignment expressions
        target: target column names
        expression: pandas or python expression to evaluate
    df : pandas.DataFrame
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of "python" expression.
    trace_rows: series or array of bools to use as mask to select target rows to trace

    Returns
    -------
    variables : pandas.DataFrame
        Will have the index of `df` and columns named by target and containing
        the result of evaluating expression
    trace_df : pandas.DataFrame or None
        a dataframe containing the eval result values for each assignment expression
    """

    np_logger = NumpyLogger(logger)

    def is_throwaway(target):
        return target == "_"

    def is_temp_singular(target):
        return target.startswith("_") and target.isupper()

    def is_temp_series_val(target):
        return target.startswith("_")

    def to_series(x):
        if x is None or np.isscalar(x):
            return pd.Series([x] * len(df.index), index=df.index)
        return x

    assert assignment_expressions.shape[0] > 0

    trace_assigned_locals = trace_results = None
    if trace_rows is not None:
        # convert to numpy array so we can slice ndarrays as well as series

        trace_rows = np.asanyarray(trace_rows)
        if trace_rows.any():
            trace_results = OrderedDict()
            trace_assigned_locals = OrderedDict()

    # avoid touching caller's passed-in locals_d parameter (they may be looping)
    _locals_dict = local_utilities()
    if locals_dict is not None:
        _locals_dict.update(locals_dict)
    if df_alias:
        _locals_dict[df_alias] = df
    else:
        _locals_dict["df"] = df
    local_keys = list(_locals_dict.keys())

    # build a dataframe of eval results for non-temp targets
    # since we allow targets to be recycled, we want to only keep the last usage
    variables = OrderedDict()
    temps = OrderedDict()

    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously

    for e in zip(assignment_expressions.target, assignment_expressions.expression):
        target, expression = e

        assert isinstance(
            target, str
        ), "expected target '%s' for expression '%s' to be string not %s" % (
            target,
            expression,
            type(target),
        )

        if target in local_keys:
            logger.warning(
                "assign_variables target obscures local_d name '%s'", str(target)
            )

        if trace_label:
            logger.info(f"{trace_label}.assign_variables {target} = {expression}")

        if is_temp_singular(target) or is_throwaway(target):
            try:
                x = eval(expression, globals(), _locals_dict)
            except Exception as err:
                logger.error(
                    "assign_variables error: %s: %s", type(err).__name__, str(err)
                )
                logger.error(
                    "assign_variables expression: %s = %s", str(target), str(expression)
                )
                raise err

            if not is_throwaway(target):
                _locals_dict[target] = x
                if trace_assigned_locals is not None:
                    trace_assigned_locals[
                        uniquify_key(trace_assigned_locals, target)
                    ] = x

            continue

        try:

            # FIXME - log any numpy warnings/errors but don't raise
            np_logger.target = str(target)
            np_logger.expression = str(expression)
            saved_handler = np.seterrcall(np_logger)
            save_err = np.seterr(all="log")

            # FIXME should whitelist globals for security?
            globals_dict = {}
            expr_values = to_series(eval(expression, globals_dict, _locals_dict))

            np.seterr(**save_err)
            np.seterrcall(saved_handler)

        # except Exception as err:
        #     logger.error("assign_variables error: %s: %s", type(err).__name__, str(err))
        #     logger.error("assign_variables expression: %s = %s", str(target), str(expression))
        #     raise err

        except Exception as err:
            logger.exception(
                f"assign_variables - {type(err).__name__} ({str(err)}) evaluating: {str(expression)}"
            )
            raise err

        if not is_temp_series_val(target):
            variables[target] = expr_values

        if trace_results is not None:
            trace_results[uniquify_key(trace_results, target)] = expr_values[trace_rows]

        # just keeping track of temps so we can chunk.log_df
        if is_temp(target):
            temps[target] = expr_values
        else:
            variables[target] = expr_values

        # update locals to allows us to ref previously assigned targets
        _locals_dict[target] = expr_values

    if trace_results is not None:

        trace_results = pd.DataFrame.from_dict(trace_results)

        trace_results.index = df[trace_rows].index

        # add df columns to trace_results
        trace_results = pd.concat([df[trace_rows], trace_results], axis=1)

    assert variables, "No non-temp variables were assigned."

    if chunk_log:
        chunk.log_df(trace_label, "temps", temps)
        chunk.log_df(trace_label, "variables", variables)
        # these are going away - let caller log result df
        chunk.log_df(trace_label, "temps", None)
        chunk.log_df(trace_label, "variables", None)

    # we stored result in dict - convert to df
    variables = util.df_from_dict(variables, index=df.index)

    return variables, trace_results, trace_assigned_locals

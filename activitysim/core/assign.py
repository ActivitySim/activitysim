# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def undupe_column_names(df, template="{} ({})"):
    """
    rename df column names so there are no duplicates (in place)

    e.g. if there are two columns named "dog", the second column will be reformatted to "dog (2)"

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe whose column names should be de-duplicated
    template : template taking two arguments (old_name, int) to use to rename columns

    Returns
    -------
    df : pandas.DataFrame
        dataframe that was renamed in place, for convenience in chaining
    """

    new_names = []
    seen = set()
    for name in df.columns:
        n = 1
        new_name = name
        while new_name in seen:
            n += 1
            new_name = template.format(name, n)
        new_names.append(new_name)
        seen.add(new_name)
    df.columns = new_names
    return df


def read_assignment_spec(fname,
                         description_name="Description",
                         target_name="Target",
                         expression_name="Expression"):
    """
    Read a CSV model specification into a Pandas DataFrame or Series.

    The CSV is expected to have columns for component descriptions
    targets, and expressions,

    The CSV is required to have a header with column names. For example:

        Description,Target,Expression

    Parameters
    ----------
    fname : str
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

    cfg = pd.read_csv(fname, comment='#')

    # drop null expressions
    # cfg = cfg.dropna(subset=[expression_name])

    cfg.rename(columns={target_name: 'target',
                        expression_name: 'expression',
                        description_name: 'description'},
               inplace=True)

    # backfill description
    if 'description' not in cfg.columns:
        cfg.description = ''

    cfg.target = cfg.target.str.strip()
    cfg.expression = cfg.expression.str.strip()

    return cfg


class NumpyLogger(object):
    def __init__(self, logger):
        self.logger = logger
        self.target = ''
        self.expression = ''

    def write(self, msg):
        self.logger.error("numpy warning: %s" % (msg.rstrip()))
        self.logger.error("expression: %s = %s" % (str(self.target), str(self.expression)))


def assign_variables(assignment_expressions, df, locals_dict, df_alias=None, trace_rows=None):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    Expressions are evaluated using Python's eval function.
    Python expressions have access to variables in locals_d (and df being
    accessible as variable df.) They also have access to previously assigned
    targets as the assigned target name.

    lowercase variables starting with underscore are temp variables (e.g. _local_var)
    and not returned except in trace_restults

    uppercase variables starting with underscore are temp variables (e.g. _LOCAL_SCALAR)
    and not returned except in trace_assigned_locals
    This is useful for defining general purpose local constants in expression file

    Users should take care that expressions should result in
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

    def is_local(target):
        return target.startswith('_') and target.isupper()

    def is_temp(target):
        return target.startswith('_')

    def to_series(x, target=None):
        if x is None or np.isscalar(x):
            if target:
                logger.warn("WARNING: assign_variables promoting scalar %s to series" % target)
            return pd.Series([x] * len(df.index), index=df.index)
        return x

    trace_assigned_locals = trace_results = None
    if trace_rows is not None:
        # convert to numpy array so we can slice ndarrays as well as series
        trace_rows = np.asanyarray(trace_rows)
        if trace_rows.any():
            trace_results = []
            trace_assigned_locals = {}

    # avoid touching caller's passed-in locals_d parameter (they may be looping)
    locals_dict = locals_dict.copy() if locals_dict is not None else {}
    if df_alias:
        locals_dict[df_alias] = df
    else:
        locals_dict['df'] = df
    local_keys = locals_dict.keys()

    l = []
    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously
    for e in zip(assignment_expressions.target, assignment_expressions.expression):
        target, expression = e

        if target in local_keys:
            logger.warn("assign_variables target obscures local_d name '%s'" % str(target))

        if is_local(target):
            x = eval(expression, globals(), locals_dict)
            locals_dict[target] = x
            if trace_assigned_locals is not None:
                trace_assigned_locals[target] = x
            continue

        try:

            # FIXME - log any numpy warnings/errors but don't raise
            np_logger.target = str(target)
            np_logger.expression = str(expression)
            saved_handler = np.seterrcall(np_logger)
            save_err = np.seterr(all='log')

            values = to_series(eval(expression, globals(), locals_dict), target=target)

            np.seterr(**save_err)
            np.seterrcall(saved_handler)

        except Exception as err:
            logger.error("assign_variables error: %s: %s" % (type(err).__name__, str(err)))

            logger.error("assign_variables expression: %s = %s"
                         % (str(target), str(expression)))

            # values = to_series(None, target=target)
            raise err

        l.append((target, values))

        if trace_results is not None:
            trace_results.append((target, values[trace_rows]))

        # update locals to allows us to ref previously assigned targets
        locals_dict[target] = values

    # build a dataframe of eval results for non-temp targets
    # since we allow targets to be recycled, we want to only keep the last usage
    # we scan through targets in reverse order and add them to the front of the list
    # the first time we see them so they end up in execution order
    variables = []
    seen = set()
    for statement in reversed(l):
        # statement is a tuple (<target_name>, <eval results in pandas.Series>)
        target_name = statement[0]
        if not is_temp(target_name) and target_name not in seen:
            variables.insert(0, statement)
            seen.add(target_name)

    # DataFrame from list of tuples [<target_name>, <eval results>), ...]
    variables = pd.DataFrame.from_items(variables)

    if trace_results is not None:

        trace_results = pd.DataFrame.from_items(trace_results)
        trace_results.index = df[trace_rows].index

        trace_results = undupe_column_names(trace_results)

        # add df columns to trace_results
        trace_results = pd.concat([df[trace_rows], trace_results], axis=1)

    return variables, trace_results, trace_assigned_locals

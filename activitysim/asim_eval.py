# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger('activitysim')


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

    # print "read_assignment_spec", fname

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


def assign_variables(assignment_expressions, df, locals_d, trace_rows=None):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    Expressions are evaluated using Python's eval function.
    Python expressions have access to variables in locals_d (and df being
    accessible as variable df.) They also have access to previously assigned
    targets as the assigned target name.

    variables starting with underscore are considered temps variables and returned seperately
    (and only if return_temp_variables is true)

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

    def to_series(x, target=None):
        if x is None or np.isscalar(x):
            if target:
                logger.warning("WARNING: assign_variables promoting scalar %s to series" % target)
            return pd.Series([x] * len(df.index), index=df.index)
        return x

    trace_results = None
    if trace_rows is not None:
        # convert to numpy array so we can slice ndarrays as well as series
        trace_rows = np.asanyarray(trace_rows)
        if trace_rows.any():
            trace_results = []

    # avoid touching caller's passed-in locals_d parameter (they may be looping)
    locals_d = locals_d.copy() if locals_d is not None else {}
    locals_d['df'] = df
    local_keys = locals_d.keys()

    l = []
    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously
    for e in zip(assignment_expressions.target, assignment_expressions.expression):
        target, expression = e

        if target in local_keys:
            logger.warn("assign_variables target obscures local_d name '%s'" % str(target))

        try:
            values = to_series(eval(expression, globals(), locals_d), target=target)
        except Exception as err:
            logger.error("assign_variables failed target: %s expression: %s"
                         % (str(target), str(expression)))
            # raise err
            raise err

        l.append((target, values))

        if trace_results is not None:
            trace_results.append((target, values[trace_rows]))

        # update locals to allows us to ref previously assigned targets
        locals_d[target] = values

    # build a dataframe of eval results for non-temp targets
    # since we allow targets to be recycled, we want to only keep the last usage
    # we scan through targets in reverse order and add them to the front of the list
    # the first time we see them so they end up in execution order
    variables = []
    seen = set()
    for statement in reversed(l):
        # statement is a tuple (<target_name>, <eval results in pandas.Series>)
        target_name = statement[0]
        if not target_name.startswith('_') and target_name not in seen:
            variables.insert(0, statement)
            seen.add(target_name)

    # DataFrame from list of tuples [<target_name>, <eval results>), ...]
    variables = pd.DataFrame.from_items(variables)

    if trace_results is not None:
        trace_results = undupe_column_names(pd.DataFrame.from_items(trace_results))

    return variables, trace_results


def assign_variables_locals(locals=None):
    """
    assign locals whose values will be accessible to the execution context
    when the expressions in spec are applied to choosers

    adds numpy functions so they can be called from expressions
    plus any additional values passed in locals dict

    Parameters
    ----------
    locals : dict
        dict of local variables to assign (probably from settings file)

    Returns
    -------
    locals_d : dict
        dict of locals suitable to pass to assign_variables method

    """

    locals_d = {
        'log': np.log,
        'exp': np.exp
    }
    if locals:
        locals_d.update(locals)
    return locals_d

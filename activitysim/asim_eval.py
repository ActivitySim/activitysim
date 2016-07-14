# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger('activitysim')


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


def assign_variables(assignment_expressions, df, locals_d):
    """
    Evaluate a set of variable expressions from a spec in the context
    of a given data table.

    Expressions are evaluated using Python's eval function.
    Python expressions have access to variables in locals_d (and df being
    accessible as variable df.) They also have access to previously assigned
    targets as the assigned target name.

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

    Returns
    -------
    variables : pandas.DataFrame
        Will have the index of `df` and columns of `exprs`.

    """

    def to_series(x, target=None):
        if x is None or np.isscalar(x):
            if target:
                logger.warning("WARNING: assign_variables promoting scalar %s to series" % target)
            return pd.Series([x] * len(df.index), index=df.index)
        return x

    # avoid trashing parameter when we add target
    locals_d = locals_d.copy() if locals_d is not None else {}
    locals_d['df'] = df

    l = []
    # need to be able to identify which variables causes an error, which keeps
    # this from being expressed more parsimoniously
    for e in zip(assignment_expressions.target, assignment_expressions.expression):
        target = e[0]
        expression = e[1]
        try:
            values = to_series(eval(expression, globals(), locals_d), target=target)
        except Exception as err:
            logger.error("assign_variables failed target: %s expression: %s"
                         % (str(target), str(expression)))
            # raise err
            values = to_series(None, target=target)

        l.append((target, values))

        # update locals to allows us to ref previously assigned targets
        locals_d[target] = values

    # since we allow targets to be recycled, we want to only keep the last usage
    keepers = []
    for statement in reversed(l):
        # don't keep targets that start with underscore
        if statement[0].startswith('_'):
            continue
        # add statement to keepers list unless target is already in list
        if not next((True for keeper in keepers if keeper[0] == statement[0]), False):
            keepers.append(statement)

    return pd.DataFrame.from_items(keepers)


def assign_variables_locals(locals=None):
    # locals whose values will be accessible to the execution context
    # when the expressions in spec are applied to choosers
    locals_d = {
        'log': np.log,
        'exp': np.exp
    }
    if locals:
        locals_d.update(locals)
    return locals_d

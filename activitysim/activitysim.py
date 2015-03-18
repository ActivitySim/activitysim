# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import urbansim.sim.simulation as sim
from urbansim.urbanchoice import interaction, mnl

from .mnl import utils_to_probs, make_choices


def random_rows(df, n):
    return df.take(np.random.choice(len(df), size=n, replace=False))


def read_model_spec(fname,
                    description_name="Description",
                    expression_name="Expression",
                    stack=False):
    """
    Read a CSV model specification into a Pandas DataFrame or Series.

    The CSV is expected to have columns for component descriptions
    and expressions, plus one or more alternatives.

    The CSV is required to have a header with column names. For example:

        Description,Expression,alt0,alt1,alt2

    Parameters
    ----------
    fname : str
        Name of a CSV spec file.
    description_name : str, optional
        Name of the column in `fname` that contains the component description.
    expression_name : str, optional
        Name of the column in `fname` that contains the component expression.
    stack : bool, optional
        If True, the returned data will be stacked so that it has a
        multi-index of (expression, alt-name) and values are the
        alternative specific utilities.

    Returns
    -------
    spec : pandas.DataFrame
        The description column is dropped from the returned data and the
        expression values are set as the table index.
    """
    cfg = pd.read_csv(fname)
    # don't need description and set the expression to the index
    cfg = cfg.drop(description_name, axis=1).set_index(expression_name)
    if stack:
        cfg = cfg.stack()
    return cfg


def identity_matrix(alt_names):
    return pd.DataFrame(np.identity(len(alt_names)),
                        columns=alt_names,
                        index=alt_names)


def eval_variables(exprs, df):
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

    Parameters
    ----------
    exprs : sequence of str
    df : pandas.DataFrame

    Returns
    -------
    variables : pandas.DataFrame
        Will have the index of `df` and columns of `exprs`.

    """
    def to_series(x):
        if np.isscalar(x):
            return pd.Series([x] * len(df), index=df.index)
        return x
    return pd.DataFrame.from_items(
        [(e, to_series(eval(e[1:])) if e.startswith('@') else df.eval(e))
         for e in exprs])


def add_skims(df, skims, skim_join_name):
    """
    Add skim data to a table. The table must contain the relevant skim
    indexers for both origin and destination.

    The table is modified in-place.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to which to add skim data as new columns.
        `df` is modified in-place.
    skims : dict
        Keys will be used as variable names and values are Skim objects - it
        will be assumed that there is a field zone_id in both choosers and
        alternatives which is used to dereference the given Skim object as
        the "origin" (on choosers) and destination (on alternatives).
    skim_join_name : str
        The name of the column that contains the origin in the choosers table
        and the destination in the alternates table - is required to be the
        same in both tables.

    """
    for key, value in skims.iteritems():
        df[key] = value.get(df[skim_join_name], df[skim_join_name + "_r"])


def simple_simulate(choosers, spec, skims=None, skim_join_name='zone_id'):
    """
    Run a simulation for when the model spec does not involve alternative
    specific data, e.g. there are no interactions with alternative
    properties and no need to sample from alternatives.

    Parameters
    ----------
    choosers : pandas.DataFrame
    spec : pandas.DataFrame
        A table of variable specifications and coefficient values.
        Variable expressions should be in the table index and the table
        should have a column for each alternative.
    skims : dict, optional
        Keys will be used as variable names and values are Skim objects - it
        will be assumed that there is a field zone_id in both choosers and
        alternatives which is used to dereference the given Skim object as
        the "origin" (on choosers) and destination (on alternatives)
    skim_join_name : str, optional
        The name of the column that contains the origin in the choosers table
        and the destination in the alternates table - is required to be the
        same in both tables - is 'zone_id' by default

    Returns
    -------
    choices : pandas.Series
        Index will be that of `choosers`, values will match the columns
        of `spec`.

    """
    if skims:
        add_skims(df, skims, skim_join_name)

    variables = eval_variables(spec.index, choosers)
    utilities = variables.dot(spec)
    probs = utils_to_probs(utilities)
    choices = make_choices(probs)

    return choices


def interaction_simulate(
        choosers, alternatives, spec,
        skims=None, skim_join_name='zone_id',
        mult_by_alt_col=False, sample_size=None):
    """
    A simple discrete choice simulation routine

    Parameters
    ----------
    choosers : DataFrame
        DataFrame of choosers
    alternatives : DataFrame
        DataFrame of alternatives - will be merged with choosers, currently
        without sampling
    spec : Series
        A Pandas series that gives the specification of the variables to
        compute and the coefficients - more on this later
    skims : dict, optional
        Keys will be used as variable names and values are Skim objects - it
        will be assumed that there is a field zone_id in both choosers and
        alternatives which is used to dereference the given Skim object as
        the "origin" (on choosers) and destination (on alternatives)
    skim_join_name : str, optional
        The name of the column that contains the origin in the choosers table
        and the destination in the alternates table - is required to be the
        same in both tables - is 'zone_id' by default
    mult_by_alt_col : boolean, optional
        Whether to multiply the expression by the name of the column in the
        specification - this is useful for alternative specific coefficients
    sample_size : int, optional
        Sample alternatives with sample of given size.  By default is None,
        which does not sample alternatives.

    Returns
    -------
    ret : Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """
    exprs = spec.index
    coeffs = spec.values
    sample_size = sample_size or len(alternatives)

    # now the index is also in the dataframe, which means it will be
    # available as the "destination" for the skims dereference below
    alternatives[alternatives.index.name] = alternatives.index

    # merge choosers and alternatives
    _, df, _ = interaction.mnl_interaction_dataset(
        choosers, alternatives, sample_size)

    if skims:
        add_skims(df, skims, skim_join_name)

    # evaluate the expressions to build the final matrix
    vars = []
    for expr in exprs:
        if expr[0][0] == "@":
            if mult_by_alt_col:
                expr = "({}) * df.{}".format(expr[0][1:], expr[1])
            else:
                if isinstance(expr, tuple):
                    expr = expr[0][1:]
                else:
                    # it's already a string, but need to remove the "@"
                    expr = expr[1:]
            try:
                s = eval(expr)
            except Exception as e:
                print "Failed with Python eval:\n%s" % expr
                raise e
        else:
            if mult_by_alt_col:
                expr = "({}) * {}".format(*expr)
            else:
                if isinstance(expr, tuple):
                    expr = expr[0]
                else:
                    # it's already a string, which is fine
                    pass
            try:
                s = df.eval(expr)
            except Exception as e:
                print "Failed with DataFrame eval:\n%s" % expr
                raise e
        vars.append((expr, s.astype('float')))
    model_design = pd.DataFrame.from_items(vars)
    model_design.index = df.index

    df = random_rows(model_design, min(100000, len(model_design)))\
        .describe().transpose()
    df = df[df["std"] == 0]
    if len(df):
        print "WARNING: Some columns have no variability:\n", df.index.values

    positions = mnl.mnl_simulate(
        model_design.as_matrix(),
        coeffs,
        numalts=sample_size,
        returnprobs=False)

    # positions come back between zero and num alternatives in the sample -
    # need to get back to the indexes
    offsets = np.arange(positions.size) * sample_size
    choices = model_design.index.take(positions + offsets)

    return pd.Series(choices, index=choosers.index), model_design

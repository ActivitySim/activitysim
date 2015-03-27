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
                    expression_name="Expression"):
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

    Returns
    -------
    spec : pandas.DataFrame
        The description column is dropped from the returned data and the
        expression values are set as the table index.
    """
    cfg = pd.read_csv(fname, comment='#')
    # don't need description and set the expression to the index
    cfg = cfg.drop(description_name, axis=1).set_index(expression_name)
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
        skims=None, skim_join_name='zone_id', sample_size=None):
    """
    Run a simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

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
    skims : dict, optional
        Keys will be used as variable names and values are Skim objects - it
        will be assumed that there is a field zone_id in both choosers and
        alternatives which is used to dereference the given Skim object as
        the "origin" (on choosers) and destination (on alternatives)
    skim_join_name : str, optional
        The name of the column that contains the origin in the choosers table
        and the destination in the alternates table - is required to be the
        same in both tables - is 'zone_id' by default
    sample_size : int, optional
        Sample alternatives with sample of given size.  By default is None,
        which does not sample alternatives.

    Returns
    -------
    ret : pandas.Series
        A series where index should match the index of the choosers DataFrame
        and values will match the index of the alternatives DataFrame -
        choices are simulated in the standard Monte Carlo fashion
    """
    if len(spec.columns) > 1:
        raise RuntimeError('spec must have only one column')

    sample_size = sample_size or len(alternatives)

    # now the index is also in the dataframe, which means it will be
    # available as the "destination" for the skims dereference below
    alternatives[alternatives.index.name] = alternatives.index

    # merge choosers and alternatives
    _, df, _ = interaction.mnl_interaction_dataset(
        choosers, alternatives, sample_size)

    if skims:
        add_skims(df, skims, skim_join_name)

    # evaluate variables from the spec
    model_design = eval_variables(spec.index, df)

    sample = random_rows(model_design, min(100000, len(model_design)))\
        .describe().transpose()
    sample = sample[sample["std"] == 0]
    if len(sample):
        print "WARNING: Some columns have no variability:\n", sample.index.values

    # multiply by coefficients and reshape into choosers by alts
    utilities = model_design.dot(spec).astype('float')
    utilities = pd.DataFrame(
        utilities.as_matrix().reshape(len(choosers), sample_size),
        index=choosers.index)

    # convert to probabilities and make choices
    probs = utils_to_probs(utilities)
    positions = make_choices(probs)

    # positions come back between zero and num alternatives in the sample -
    # need to get back to the indexes
    offsets = np.arange(len(positions)) * sample_size
    choices = model_design.index.take(positions + offsets)

    return pd.Series(choices, index=choosers.index), model_design

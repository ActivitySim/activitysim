# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

from skim import Skims
from urbansim.urbanchoice import interaction, mnl
import pandas as pd
import numpy as np


def random_rows(df, n):
    return df.take(np.random.choice(len(df), size=n, replace=False))


def read_model_spec(fname,
                    description_name="Description",
                    expression_name="Expression",
                    stack=True):
    """
    Read in the excel file and reformat for machines
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


def simple_simulate(choosers, alternatives, spec,
                    skims=None,
                    locals_d=None,
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
    mult_by_alt_col : boolean
        Whether to multiply the expression by the name of the column in the
        specification - this is useful for alternative specific coefficients
    sample_size : int
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

    if locals_d is None:
        locals_d = {}

    # now the index is also in the dataframe, which means it will be
    # available as the "destination" for the skims dereference below
    alternatives[alternatives.index.name] = alternatives.index

    # merge choosers and alternatives
    _, df, _ = interaction.mnl_interaction_dataset(
        choosers, alternatives, sample_size)

    if skims:
        assert isinstance(skims, Skims)
        skims.set_df(df)

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
                s = eval(expr, locals_d, locals())
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

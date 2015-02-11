# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

import urbansim.sim.simulation as sim
from urbansim.urbanchoice import interaction, mnl
import pandas as pd
import numpy as np
import os


def random_rows(df, n):
    return df.take(np.random.choice(len(df), size=n, replace=False))


def read_model_spec(fname,
                    description_name="Description",
                    expression_name="Expression"):
    """
    Read in the excel file and reformat for machines
    """
    cfg = pd.read_csv(fname)
    # don't need description and set the expression to the index
    cfg = cfg.drop(description_name, axis=1).set_index(expression_name).stack()
    return cfg


def identity_matrix(alt_names):
    return pd.DataFrame(np.identity(len(alt_names)),
                        columns=alt_names,
                        index=alt_names)


def simple_simulate(choosers, alternatives, spec):
    exprs = spec.index
    coeffs = spec.values

    # merge choosers and alternatives
    _, df, _ = interaction.mnl_interaction_dataset(
        choosers, alternatives, len(alternatives))

    # evaluate the expressions to build the final matrix
    vars, names = [], []
    for expr in exprs:
        if expr[0][0] == "@":
            expr = "({}) * df.{}".format(expr[0][1:], expr[1])
            try:
                s = eval(expr)
            except Exception as e:
                print "Failed with Python eval:\n%s" % expr
                raise e
        else:
            expr = "({}) * {}".format(*expr)
            try:
                s = df.eval(expr)
            except Exception as e:
                print "Failed with DataFrame eval:\n%s" % expr
                raise e
        names.append(expr)
        vars.append(s)
    model_design = pd.concat(vars, axis=1)
    model_design.columns = names

    df = random_rows(model_design, 100000).describe().transpose()
    df = df[df["std"] == 0]
    if len(df):
        print "WARNING: Describe of columns with no variability:\n", df

    choices = mnl.mnl_simulate(
        model_design.as_matrix(),
        coeffs,
        numalts=len(alternatives),
        returnprobs=False)

    return pd.Series(choices, index=choosers.index), model_design

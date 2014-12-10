import urbansim.sim.simulation as sim
from urbansim.utils import misc
from urbansim.urbanchoice import interaction, mnl
import pandas as pd
import numpy as np
import os


def random_rows(df, n):
    return df.take(np.random.randint(0, len(df), n))


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
    model_design=pd.concat(vars, axis=1)
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


@sim.table()
def auto_alts():
    return identity_matrix(["cars%d"%i for i in range(5)])


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join(misc.configs_dir(), "auto_ownership_coeffs.csv")
    return read_model_spec(f).head(4*20)


@sim.model()
def auto_ownership_simulate(households,
                            auto_alts,
                            auto_ownership_spec,
                            land_use):

    choosers = sim.merge_tables(households.name, tables=[households, land_use])
    alternatives = auto_alts.to_frame()

    choices, model_design = \
        simple_simulate(choosers, alternatives, auto_ownership_spec)

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)

    return model_design

import urbansim.sim.simulation as sim
from urbansim.utils import misc
from urbansim.urbanchoice import interaction, mnl
import pandas as pd
import numpy as np
import os


def read_model_spec(fname,
                    description_name="Description",
                    expression_name="Expression"):
    """
    Read in the excel file and reformat for machines
    """
    cfg = pd.read_csv(fname)
    # don't need description and set the expression to the index
    cfg = cfg.drop(description_name, axis=1).set_index(expression_name).stack()
    # expressions are index names times column names
    cfg.index = ["({}) * {}".format(*k) for k in cfg.index]
    return cfg


def identity_matrix(alt_names):
    return pd.DataFrame(np.identity(len(alt_names)),
                        columns=alt_names,
                        index=alt_names)


def simple_simulate(choosers, alternatives, spec):
    exprs = spec.index
    coeffs = spec.values

    # merge choosers and alternatives
    _, merged, _ = interaction.mnl_interaction_dataset(
        choosers, alternatives, len(alternatives))

    # evaluate the expressions to build the final matrix
    model_design=pd.concat([merged.eval(s) for s in exprs], axis=1)
    print "Describe of design matrix:\n", model_design.describe()

    probabilities = mnl.mnl_simulate(
        model_design.as_matrix(),
        coeffs,
        numalts=len(alternatives), returnprobs=True)

    def rand(x): return np.random.choice(alternatives.index, p=x)
    choices = np.apply_along_axis(rand, 1, probabilities)

    return pd.Series(choices, index=choosers.index)


@sim.table()
def auto_alts():
    return identity_matrix(["cars%d"%i for i in range(5)])


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join(misc.configs_dir(), "auto_ownership_coeffs.csv")
    return read_model_spec(f).head(8)


@sim.model()
def auto_ownership_simulate(households, auto_alts, auto_ownership_spec):
    print auto_ownership_spec
    choosers = households.to_frame()
    alternatives = auto_alts.to_frame()
    choices = simple_simulate(choosers, alternatives, auto_ownership_spec)

    print "Choices\n", choices.describe()

import os
import copy
import yaml
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim

"""
Mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@sim.injectable()
def mode_choice_settings(configs_dir):
    with open(os.path.join(configs_dir, "configs", "mode_choice.yaml")) as f:
        return yaml.load(f)


# FIXME move into activitysim as a utility function
def get_leaves(d):
    # returns the leaves of a multi-level dictionary

    def walk(node, out):
        # pass the dict and the list of outputs
        for key, item in node.items():
            # ignore keys at this point
            if isinstance(item, dict):
                out = walk(item, out)
            elif isinstance(item, list):
                out += item
            else:
                out += [item]
        return out

    return walk(d, [])


# FIXME move into activitysim as a utility function
def evaluate_expression_list(expressions, locals_d):
    """
    Evaluate a list of expressions - each one can depend on the one before it

    Parameters
    ----------
    expressions : Series
        indexes are names and values are expressions
    locals_d : dict
        will be passed directly to eval

    Returns
    -------
    expressions : Series
        index is the same as above, but values are now confirmed to be floats
    """
    d = {}
    # this could be a simple expression except that the dictionary
    # is accumulating expressions - i.e. they're not all independent
    # and must be evaluated in order
    for k, v in expressions.iteritems():
        # make sure it can be converted to a float
        d[k] = float(eval(str(v), copy.copy(d), locals_d))
    return pd.Series(d)


@sim.injectable()
def mode_choice_coefficients(mode_choice_settings):
    # coefficients comes as a list of dicts and needs to be tuples
    coeffs = [x.items()[0] for x in mode_choice_settings['COEFFICIENTS']]
    expressions = pd.Series(zip(*coeffs)[1], index=zip(*coeffs)[0])
    constants = mode_choice_settings['CONSTANTS']
    return evaluate_expression_list(expressions, locals_d=constants)


@sim.table()
def mode_choice_alts(mode_choice_settings):
    mode_choice_alts = get_leaves(mode_choice_settings['NESTS'])
    return asim.identity_matrix(mode_choice_alts)


@sim.injectable()
def mode_choice_spec(configs_dir, mode_choice_coefficients):
    f = os.path.join(configs_dir, 'configs', "mode_choice_work.csv")
    df = asim.read_model_spec(f, stack=False)
    df['work'] = evaluate_expression_list(df['work'],
                                          mode_choice_coefficients.to_dict())
    return df.set_index('Alternative', append=True)


@sim.model()
def mode_choice_simulate(tours_merged,
                         mode_choice_alts,
                         mode_choice_spec,
                         mode_choice_settings,
                         skims):

    mode_choice_spec = mode_choice_spec.head(6)
    print mode_choice_spec

    # set the keys for this lookup - in this case orig and dest are both
    # already part of the trip
    skims.set_keys("TAZ", "workplace_taz")
    # the skims will be available under the name "skims" for any @ expressions
    locals_d = {
        "skims": skims,
        "out_period": "AM",
        "in_period": "PM"
    }
    locals_d.update(mode_choice_settings['CONSTANTS'])

    # FIXME lots of other segments here - for now running the mode choice for
    # FIXME work on all kinds of tour types
    # FIXME note that in particular the spec above only has work tours in it

    choices, _ = asim.simple_simulate(tours_merged.to_frame(),
                                      mode_choice_alts.to_frame(),
                                      mode_choice_spec,
                                      skims=skims,
                                      locals_d=locals_d,
                                      mult_by_alt_col=False)

    print "Choices:\n", choices.value_counts()
    sim.add_column("tours", "mode", choices)

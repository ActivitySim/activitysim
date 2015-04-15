import os
import copy
import yaml
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim
from activitysim import skim as askim

"""
Mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@sim.injectable()
def mode_choice_settings(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice.yaml")) as f:
        return yaml.load(f)


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


def pre_process_expressions(expressions, variable_templates):
    return [eval(e[1:], variable_templates) if e.startswith('$') else e for
            e in expressions]


@sim.injectable()
def mode_choice_spec(configs_dir, mode_choice_coefficients,
                     mode_choice_settings):
    f = os.path.join(configs_dir, 'configs', "tour_mode_choice.csv")
    df = asim.read_model_spec(f).head(53)
    df['EatOut'] = evaluate_expression_list(df['EatOut'],
                                            mode_choice_coefficients.to_dict())

    df.index = pre_process_expressions(df.index,
                                       mode_choice_settings['VARIABLE_TEMPLATES'])

    return df.set_index('Alternative', append=True)


def get_segment_and_unstack(spec, segment):
    return spec[segment].unstack().fillna(0)


def _mode_choice_simulate(tours, skims, spec, additional_constants):

    in_skims = askim.Skims3D(skims.set_keys("TAZ", "workplace_taz"),
                             "in_period", -1)
    out_skims = askim.Skims3D(skims.set_keys("workplace_taz", "TAZ"),
                              "out_period", -1)
    locals_d = {
        "in_skims": in_skims,
        "out_skims": out_skims
    }
    locals_d.update(additional_constants)

    choices, _ = asim.simple_simulate(tours,
                                      spec,
                                      skims=[in_skims, out_skims],
                                      locals_d=locals_d)

    print "Choices:\n", choices.value_counts()
    sim.add_column("tours", "mode", choices)


@sim.model()
def mode_choice_simulate(tours_merged,
                         mode_choice_spec,
                         mode_choice_settings,
                         skims):

    tours = tours_merged.to_frame()

    print mode_choice_spec.EatOut

    _mode_choice_simulate(tours[tours.tour_type == "work"],
                          skims,
                          get_segment_and_unstack(mode_choice_spec, 'EatOut'),
                          mode_choice_settings['CONSTANTS'])

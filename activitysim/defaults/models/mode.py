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


@sim.injectable()
def mode_choice_spec_df(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice.csv")) as f:
        return asim.read_model_spec(f).head(223)


@sim.injectable()
def mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice_coeffs.csv")) as f:
        return pd.read_csv(f, index_col='Expression')


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


def pre_process_expressions(expressions, variable_templates):
    return [eval(e[1:], variable_templates) if e.startswith('$') else e for
            e in expressions]


def get_segment_and_unstack(spec, segment):
    return spec[segment].unstack().fillna(0)


@sim.injectable()
def mode_choice_spec(mode_choice_spec_df, mode_choice_coeffs,
                     mode_choice_settings):

    # ok we have read in the spec - we need to do several things to reformat it
    # to the same style spec that all the other models have

    constants = mode_choice_settings['CONSTANTS']
    templates = mode_choice_settings['VARIABLE_TEMPLATES']
    df = mode_choice_spec_df

    # the expressions themselves can be prepended with a "$" in order to use
    # model templates that are shared by several different expressions
    df.index = pre_process_expressions(df.index, templates)

    df = df.set_index('Alternative', append=True)

    # for each segment - e.g. eatout vs social vs work vs ...
    for col in df.columns:

        # first the coeffs come as expressions that refer to previous cells
        # as well as constants that come from the settings file
        mode_choice_coeffs[col] = evaluate_expression_list(
            mode_choice_coeffs[col],
            locals_d=constants)

        # then use the coeffs we just evaluated within the spec (they occur
        # multiple times in the spec which is why they get stored uniquely
        # in a different file
        df[col] = evaluate_expression_list(
            df[col],
            mode_choice_coeffs[col].to_dict())

    # FIXME alternatives are a comma separated list then need to be copied
    # FIXME and put into their own rows at this point

    return df


def _mode_choice_simulate(tours, skims, spec, additional_constants):

    # FIXME - this is only really going for the workplace trip
    in_skims = askim.Skims3D(skims.set_keys("TAZ", "workplace_taz"),
                             "in_period", -1)
    out_skims = askim.Skims3D(skims.set_keys("workplace_taz", "TAZ"),
                              "out_period", -1)
    skims.set_keys("TAZ", "workplace_taz")

    locals_d = {
        "in_skims": in_skims,
        "out_skims": out_skims,
        "skims": skims
    }
    locals_d.update(additional_constants)

    choices, _ = asim.simple_simulate(tours,
                                      spec,
                                      skims=[in_skims, out_skims, skims],
                                      locals_d=locals_d)

    alts = spec.columns
    choices = choices.map(dict(zip(range(len(alts)), alts)))

    return choices


@sim.model()
def mode_choice_simulate(tours_merged,
                         mode_choice_spec,
                         mode_choice_settings,
                         skims):

    tours = tours_merged.to_frame()

    mode_choice_spec = mode_choice_spec
    print mode_choice_spec.eatout

    choices = _mode_choice_simulate(
        tours[tours.tour_type == "eatout"],
        skims,
        get_segment_and_unstack(mode_choice_spec, 'eatout'),
        mode_choice_settings['CONSTANTS'])

    print "Choices:\n", choices.value_counts()
    sim.add_column("tours", "mode", choices)

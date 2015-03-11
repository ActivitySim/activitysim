import os
import yaml
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


@sim.table()
def mode_choice_alts(mode_choice_settings):
    mode_choice_alts = get_leaves(mode_choice_settings['NESTS'])
    return asim.identity_matrix(mode_choice_alts)


@sim.injectable()
def mode_choice_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "mode_choice_work.csv")
    return asim.read_model_spec(f, stack=False).\
        set_index('Alternative', append=True)


@sim.model()
def mode_choice_simulate(tours,
                         mode_choice_alts,
                         mode_choice_spec):

    mode_choice_spec = mode_choice_spec.head(1)
    print mode_choice_spec.values

    # FIXME lots of other segments here - for now running the mode choice for
    # FIXME work on all kinds of tour types
    # FIXME note that in particular the spec above only has work tours in it

    choices, d = asim.simple_simulate(tours.to_frame(),
                                      mode_choice_alts.to_frame(),
                                      mode_choice_spec,
                                      mult_by_alt_col=False)

    print d
    print "Choices:\n", choices.value_counts()
    sim.add_column("all_tours_merged", "mode", choices)

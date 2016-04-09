import os
import yaml

import orca
import pandas as pd
import yaml

from activitysim import activitysim as asim
from activitysim import skim as askim
from .util.mode import _mode_choice_spec

"""
Mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


@orca.injectable()
def mode_choice_settings(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice.yaml")) as f:
        return yaml.load(f)


@orca.injectable()
def mode_choice_spec_df(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice.csv")) as f:
        return asim.read_model_spec(f)


@orca.injectable()
def mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice_coeffs.csv")) as f:
        return pd.read_csv(f, index_col='Expression')


@orca.injectable()
def mode_choice_spec(mode_choice_spec_df, mode_choice_coeffs,
                     mode_choice_settings):
    return _mode_choice_spec(mode_choice_spec_df, mode_choice_coeffs,
                             mode_choice_settings)


def _mode_choice_simulate(tours,
                          skims,
                          orig_key,
                          dest_key,
                          spec,
                          additional_constants,
                          omx=None):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are trip purposes).  Pass in the tours that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    # FIXME - jwd - get list of unique second-tuple-item keys (aka skim_key2)
    print "Skims3D.__init__ %s skim_key values = [%s] " % ('in_period', tours['in_period'].unique())
    print "Skims3D.__init__ %s skim_key values = [%s] " % ('out_period', tours['out_period'].unique())

    in_skims = askim.Skims3D(skims.set_keys(orig_key, dest_key),
                             "in_period", -1)
    out_skims = askim.Skims3D(skims.set_keys(dest_key, orig_key),
                              "out_period", -1)
    skims.set_keys(orig_key, dest_key)

    if omx:
        in_skims.set_omx(omx)
        out_skims.set_omx(omx)

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


def get_segment_and_unstack(spec, segment):
    """
    This does what it says.  Take the spec, get the column from the spec for
    the given segment, and unstack.  It is assumed that the last column of
    the multiindex is alternatives so when you do this unstacking,
    each alternative is in a column (which is the format this as used for the
    simple_simulate call.  The weird nuance here is the "Rowid" column -
    since many expressions are repeated (e.g. many are just "1") a Rowid
    column is necessary to identify which alternatives are actually part of
    which original row - otherwise the unstack is incorrect (i.e. the index
    is not unique)
    """
    return spec[segment].unstack().\
        reset_index(level="Rowid", drop=True).fillna(0)


@orca.step()
def mode_choice_simulate(tours_merged,
                         mode_choice_spec,
                         mode_choice_settings,
                         skims, omx_file):

    tours = tours_merged.to_frame()

    choices_list = []

    print "Tour types:\n", tours.tour_type.value_counts()

    # FIXME this only runs eatout
    od_key_map = {
        'default': dict(orig_key='TAZ', dest_key='destination'),
        'work': dict(orig_key='TAZ', dest_key='workplace_taz'),
        'school': dict(orig_key='TAZ', dest_key='school_taz'),
    }

    for tour_type, segment in tours.groupby('tour_type'):

        od_keys = od_key_map.get(tour_type) or od_key_map['default']

        print "running tour_type '%s'" % tour_type
        print "   orig_key='%s' dest_key='%s'" % (od_keys['orig_key'], od_keys['dest_key'])

        # if tour_type not in ['work']:
        #     print "skipping tour_type %s" % tour_type
        #     continue

        tour_type_tours = tours[tours.tour_type == tour_type]

        print "dest_taz counts:\n", tour_type_tours[od_keys['dest_key']].value_counts()

        choices = _mode_choice_simulate(
            tours[tours.tour_type == tour_type],
            skims,
            orig_key=od_keys['orig_key'],
            dest_key=od_keys['dest_key'],
            spec=get_segment_and_unstack(mode_choice_spec, tour_type),
            additional_constants=mode_choice_settings['CONSTANTS'],
            omx=omx_file)

        print "Choices:\n", choices.value_counts()
        choices_list.append(choices)

    choices = pd.concat(choices_list)

    print "Choices for all tour types:\n", choices.value_counts()
    orca.add_column("tours", "mode", choices)

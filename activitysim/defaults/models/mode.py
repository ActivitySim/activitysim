# ActivitySim
# See full license in LICENSE.txt.

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
def tour_mode_choice_settings(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice.yaml")) as f:
        return yaml.load(f)


@orca.injectable()
def tour_mode_choice_spec_df(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice.csv")) as f:
        return asim.read_model_spec(f)


@orca.injectable()
def tour_mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "tour_mode_choice_coeffs.csv")) as f:
        return pd.read_csv(f, index_col='Expression')


@orca.injectable()
def tour_mode_choice_spec(tour_mode_choice_spec_df,
                          tour_mode_choice_coeffs,
                          tour_mode_choice_settings):
    return _mode_choice_spec(tour_mode_choice_spec_df,
                             tour_mode_choice_coeffs,
                             tour_mode_choice_settings)


@orca.injectable()
def trip_mode_choice_settings(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "trip_mode_choice.yaml")) as f:
        return yaml.load(f)


@orca.injectable()
def trip_mode_choice_spec_df(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "trip_mode_choice.csv")) as f:
        return asim.read_model_spec(f)


@orca.injectable()
def trip_mode_choice_coeffs(configs_dir):
    with open(os.path.join(configs_dir,
                           "configs",
                           "trip_mode_choice_coeffs.csv")) as f:
        return pd.read_csv(f, index_col='Expression')


@orca.injectable()
def trip_mode_choice_spec(trip_mode_choice_spec_df,
                          trip_mode_choice_coeffs,
                          trip_mode_choice_settings):
    return _mode_choice_spec(trip_mode_choice_spec_df,
                             trip_mode_choice_coeffs,
                             trip_mode_choice_settings)


def _mode_choice_simulate(tours,
                          skims,
                          orig_key,
                          dest_key,
                          spec,
                          additional_constants,
                          omx=None,
                          cache_skim_key_values=None):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are trip purposes).  Pass in the tours that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    # FIXME - log
    # print "Skims3D %s skim_key2 values = %s" % ('in_period', tours['in_period'].unique())
    # print "Skims3D %s skim_key2 values = %s" % ('out_period', tours['out_period'].unique())

    # FIXME - check that periods are in time_periods?

    in_skims = askim.Skims3D(skims=skims.set_keys(orig_key, dest_key),
                             skim_key="in_period",
                             offset=-1)
    out_skims = askim.Skims3D(skims=skims.set_keys(dest_key, orig_key),
                              skim_key="out_period",
                              offset=-1)
    skims.set_keys(orig_key, dest_key)

    if omx is not None:
        in_skims.set_omx(omx, cache_skim_key_values=cache_skim_key_values)
        out_skims.set_omx(omx, cache_skim_key_values=cache_skim_key_values)

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
def tour_mode_choice_simulate(tours_merged,
                              tour_mode_choice_spec,
                              tour_mode_choice_settings,
                              skims, omx_file, cache_skim_key_values):

    tours = tours_merged.to_frame()

    choices_list = []

    print "Tour types:\n", tours.tour_type.value_counts()

    for tour_type, segment in tours.groupby('tour_type'):

        # FIXME - log
        print "running tour_type '%s'" % tour_type

        # FIXME - hack to run subset of tour types
        # TOUR_TYPE_SUBSET = []
        # # TOUR_TYPE_SUBSET = ['eatout']
        # if TOUR_TYPE_SUBSET and tour_type not in TOUR_TYPE_SUBSET:
        #     print "skipping tour_type %s" % tour_type
        #     continue

        tour_type_tours = tours[tours.tour_type == tour_type]
        orig_key = 'TAZ'
        dest_key = 'destination'

        # FIXME - check that destination is not null (patch_mandatory_tour_destination not run?)

        # FIXME - log
        # print "dest_taz counts:\n", tour_type_tours[dest_key].value_counts()

        choices = _mode_choice_simulate(
            tour_type_tours,
            skims,
            orig_key=orig_key,
            dest_key=dest_key,
            spec=get_segment_and_unstack(tour_mode_choice_spec, tour_type),
            additional_constants=tour_mode_choice_settings['CONSTANTS'],
            omx=omx_file,
            cache_skim_key_values=cache_skim_key_values)

        # FIXME - log
        # print "Choices:\n", choices.value_counts()
        choices_list.append(choices)

        # FIXME - this forces garbage collection
        print asim.usage('tour_type %s' % tour_type)

    choices = pd.concat(choices_list)

    print "Choices for all tour types:\n", choices.value_counts()

    orca.add_column("tours", "mode", choices)

    # FIXME - this forces garbage collection
    print asim.usage('tour_mode_choice_simulate')


@orca.step()
def trip_mode_choice_simulate(tours_merged,
                              trip_mode_choice_spec,
                              trip_mode_choice_settings,
                              skims, omx_file, cache_skim_key_values):

    # FIXME running the trips model on tours
    trips = tours_merged.to_frame()

    choices_list = []

    # FIXME - log
    print "Trip types:\n", trips.tour_type.value_counts()

    for tour_type, segment in trips.groupby('tour_type'):

        # FIXME - log
        print "running tour_type '%s'" % tour_type

        # FIXME - hack to run subset of tour types
        # TOUR_TYPE_SUBSET = []
        # # TOUR_TYPE_SUBSET = ['eatout']
        # if TOUR_TYPE_SUBSET and tour_type not in TOUR_TYPE_SUBSET:
        #     print "skipping tour_type %s" % tour_type
        #     continue

        tour_type_tours = trips[trips.tour_type == tour_type]
        orig_key = 'TAZ'
        dest_key = 'destination'

        # FIXME - check that destination is not null (patch_mandatory_tour_destination not run?)

        # FIXME - log
        # print "dest_taz counts:\n", tour_type_tours[dest_key].value_counts()

        choices = _mode_choice_simulate(
            tour_type_tours,
            skims,
            orig_key=orig_key,
            dest_key=dest_key,
            spec=get_segment_and_unstack(trip_mode_choice_spec, tour_type),
            additional_constants=trip_mode_choice_settings['CONSTANTS'],
            omx=omx_file,
            cache_skim_key_values=cache_skim_key_values)

        # FIXME - log
        # print "Choices:\n", choices.value_counts()

        choices_list.append(choices)

        # FIXME - this forces garbage collection
        print asim.usage('tour_type %s' % tour_type)

    choices = pd.concat(choices_list)

    # FIXME - log
    print "Choices for all trip types:\n", choices.value_counts()

    orca.add_column("trips", "mode", choices)

    # FIXME - this forces garbage collection
    print asim.usage('trip_mode_choice_simulate')

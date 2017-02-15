import orca
from activitysim import defaults
from activitysim import tracing
from activitysim import activitysim as asim
import pandas as pd
import numpy as np
import os
import time


# you will want to configure this with the locations of the canonical datasets
DATA_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'activitysim-data')

@orca.injectable()
def data_dir():
    return os.path.join(DATA_REPO, 'multi_zone')

import extensions


def print_elapsed_time(msg=None, t0=None):
    # FIXME - development debugging code to be removed
    t1 = time.time()
    if msg:
        t = t1 - (t0 or t1)
        msg = "Time to execute %s : %s seconds (%s seconds)" % (msg, t, round(t, 3))
        print msg
    return t1

def get_taz():

    # select some random rows with attributes
    taz_df = network_los.taz_df[~np.isnan(network_los.taz_df.terminal_time)]

    # print "\ntaz_df\n", taz_df.head(20)

    random_taz = asim.random_rows(taz_df, 10)

    print "\nrandom_taz\n", random_taz
    print "\nnetwork_los.get_taz_offsets\n", network_los.get_taz_offsets(random_taz.index)
    print "\nnetwork_los.get_taz(<Int64Index>, 'terminal_time')\n", network_los.get_taz(
        random_taz.index, 'terminal_time')
    print "\nnetwork_los.get_taz(<array>, 'terminal_time')\n", network_los.get_taz(
        random_taz.index.values, 'terminal_time')
    print "\nnetwork_los.get_taz(<Series>, 'terminal_time')\n", network_los.get_taz(
        pd.Series(data=random_taz.index.values), 'terminal_time')


def get_tap():

    tap_df = network_los.tap_df

    # print "\ntap_df\n", tap_df.head(20)

    random_tap = asim.random_rows(tap_df, 10)
    print "\nrandom_tap\n", random_tap
    print "\nnetwork_los.get_tap_offsets\n", network_los.get_tap_offsets(random_tap.index)
    print "\nnetwork_los.get_tap(<Int64Index>, 'TAZ')\n", network_los.get_tap(random_tap.index,
                                                                              'TAZ')
    print "\nnetwork_los.get_tap(<Int64Index>, 'MAZ')\n", network_los.get_tap(random_tap.index,
                                                                              'MAZ')

    # select some random rows with non-null attributes
    random_tap = asim.random_rows(network_los.tap_df[~network_los.taz_df.isnull()], 10)
    print "\nnetwork_los.get_tap(<Int64Index>, 'capacity')\n", network_los.get_tap(random_tap.index,
                                                                                   'capacity')

def get_maz():
    maz_df = network_los.maz_df
    random_maz = asim.random_rows(maz_df, 10)

    print "\nmaz_df\n", maz_df.head(20)
    print "\nnetwork_los.get_maz(<Int64Index>, 'TAZ')\n", network_los.get_maz(random_maz.index,
                                                                              'TAZ')
    print "\nnetwork_los.get_maz(<Int64Index>, 'milestocoast')\n", network_los.get_maz(
        random_maz.index, 'milestocoast')


def taz_skims():

    taz_df = network_los.taz_df

    # otaz = [15]
    # dtaz = [16]
    # tod = ['PM']

    otaz = asim.random_rows(taz_df, 10).index
    dtaz = asim.random_rows(taz_df, 10).index
    tod = np.random.choice(['AM', 'PM'], 10)

    print "\notaz\n", otaz
    print "\notaz\n", dtaz
    print "\ntod\n", tod

    skim = network_los.taz_skim_dict.get(('SOV_TIME', 'PM'))
    sov_time = skim.get(network_los.get_taz_offsets(otaz), network_los.get_taz_offsets(dtaz))
    print "\nraw sov_time\n", sov_time

    sov_time = network_los.get_tazpairs(otaz, dtaz, ('SOV_TIME', 'PM'))
    print "\nget_tazpairs sov_time\n", sov_time

    sov_time = network_los.get_tazpairs3d(otaz, dtaz, tod, 'SOV_TIME')
    print "\nget_tazpairs3d sov_time\n", sov_time
    print "(only expect the PM values to be the same as get_taz_skim)\m"


def tap_skims():

    tap_df = network_los.tap_df

    # otap = [15]
    # dtap = [16]
    # tod = ['PM']

    otap = asim.random_rows(tap_df, 10).index
    dtap = asim.random_rows(tap_df, 10).index
    tod = np.random.choice(['AM', 'PM'], 10)

    print "\notap\n", otap
    print "\notap\n", dtap
    print "\ntod\n", tod

    skim = network_los.tap_skim_dict.get(('LOCAL_BUS_FARE', 'PM'))
    sov_time = skim.get(network_los.get_tap_offsets(otap), network_los.get_tap_offsets(dtap))
    print "\nraw sov_time\n", sov_time

    sov_time = network_los.get_tappairs(otap, dtap, ('LOCAL_BUS_FARE', 'PM'))
    print "\nget_tappairs sov_time\n", sov_time

    sov_time = network_los.get_tappairs3d(otap, dtap, tod, 'LOCAL_BUS_FARE')
    print "\nget_tappairs3d sov_time\n", sov_time
    print "(only expect the PM values to be the same as get_tap_skim)\m"


def get_maz_pairs():


    #    OMAZ   DMAZ  bike_logsum  bike_time  walk_perceived  walk_actual  walk_gain
    # 0  3015  22567       -4.332     12.711          43.520       33.218      303.0
    # 1  3626   3626        9.169      0.107           0.358        0.358        0.0
    # 2  3640   3192       -5.593     16.030          52.334       39.550      376.0

    #print network_los.maz2maz_df.head(10)

    # omaz = [3015, 3626, 3640]
    # dmaz = [22567, 3626, 3192]

    # sparse array so make sure the pairs are there
    maz2maz_df = asim.random_rows(network_los.maz2maz_df, 100000)
    omaz = maz2maz_df.OMAZ
    dmaz = maz2maz_df.DMAZ
    print maz2maz_df.head(5)

    print "\nomaz\n", omaz.head(5)
    print "\ndmaz\n", dmaz.head(5)

    walk_actual = network_los.get_mazpairs(omaz, dmaz, 'walk_actual')
    print "\nget_mazpairs walk_actual\n", walk_actual.head(5)


def get_maz_tap_pairs():

    pass


# uncomment the line below to set random seed so that run results are reproducible
# orca.add_injectable("set_random_seed", set_random_seed)

tracing.config_logger()

t0 = print_elapsed_time()

taz_skim_stack = orca.get_injectable('taz_skim_stack')
t0 = print_elapsed_time("load taz_skim_stack", t0)

tap_skim_stack = orca.get_injectable('tap_skim_stack')
t0 = print_elapsed_time("load tap_skim_stack", t0)

network_los = orca.get_injectable('network_los')
t0 = print_elapsed_time("load network_los", t0)

# print "\n########## get_taz\n"
# get_taz()
#
# print "\n########## get_tap\n"
# get_tap()
#
# print "\n########## get_maz\n"
# get_maz()
#
# print "\n########## TAZ Skims\n"
# taz_skims()
#
# print "\n########## TAP Skims\n"
# tap_skims()
#
# print "\n########## MAZ Skims\n"
# get_maz_pairs()


get_maz_tap_pairs()

import orca
from activitysim import abm
from activitysim.core import tracing
from activitysim.core import simulate as asim
import pandas as pd
import numpy as np
import os
import time


# you will want to configure this with the locations of the canonical datasets
DATA_REPO = "C:/projects/sandag-asim/toRSG/output/"
DATA_REPO = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"

VECTOR_TEST_SIZE = 10000
VECTOR_TEST_SIZE = 20

@orca.injectable()
def output_dir():
    if not os.path.exists('output'):
        os.makedirs('output') #make directory if needed
    return 'output'

@orca.injectable()
def data_dir():
    return os.path.join(DATA_REPO)

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
    random_taz = taz_df.sample(VECTOR_TEST_SIZE, replace=True)

    print "\nrandom_taz\n", random_taz

    print "\nnetwork_los.get_taz(<Int64Index>, 'terminal_time')\n", \
        network_los.get_taz(random_taz.index, 'terminal_time')

    print "\nnetwork_los.get_taz(<array>, 'terminal_time')\n", \
        network_los.get_taz(random_taz.index.values, 'terminal_time')

    print "\nnetwork_los.get_taz(<Series>, 'terminal_time')\n", \
        network_los.get_taz(pd.Series(data=random_taz.index.values), 'terminal_time')


def get_tap():

    tap_df = network_los.tap_df
    random_tap = tap_df.sample(VECTOR_TEST_SIZE, replace=True)

    print "\nrandom_tap\n", random_tap

    print "\nnetwork_los.get_tap(<Int64Index>, 'TAZ')\n", \
        network_los.get_tap(random_tap.index, 'TAZ')

    print "\nnetwork_los.get_tap(<Int64Index>, 'MAZ')\n", \
        network_los.get_tap(random_tap.index, 'MAZ')

    # select some random rows with non-null attributes
    tap_df = network_los.tap_df.dropna(axis=0, how='any')
    random_tap = tap_df.sample(VECTOR_TEST_SIZE, replace=True)

    print "\nnetwork_los.get_tap(<Int64Index>, 'capacity')\n", \
        network_los.get_tap(random_tap.index, 'capacity')


def get_maz():
    maz_df = network_los.maz_df
    random_maz = maz_df.sample(VECTOR_TEST_SIZE, replace=True)

    print "\nrandom_maz\n", random_maz.head(VECTOR_TEST_SIZE)
    
    print "\nnetwork_los.get_maz(<Int64Index>, 'TAZ')\n", \
        network_los.get_maz(random_maz.index, 'TAZ')

    print "\nnetwork_los.get_maz(<Int64Index>, 'milestocoast')\n", \
        network_los.get_maz( random_maz.index, 'milestocoast')


def taz_skims():

    taz_df = network_los.taz_df

    # otaz = [15]
    # dtaz = [16]
    # tod = ['PM']

    otaz = taz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    dtaz = taz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)

    print "\notaz\n", otaz
    print "\ndtaz\n", dtaz
    print "\ntod\n", tod

    skim = network_los.taz_skim_dict.get(('SOV_TIME', 'PM'))
    sov_time = skim.get(otaz, dtaz)
    print "\nraw sov_time\n", sov_time

    sov_time = network_los.get_tazpairs(otaz, dtaz, ('SOV_TIME', 'PM'))
    print "\nget_tazpairs sov_time\n", sov_time

    print(len(otaz))
    sov_time = network_los.get_tazpairs3d(otaz, dtaz, tod, 'SOV_TIME')
    print "\nget_tazpairs3d sov_time\n", sov_time


def tap_skims():

    tap_df = network_los.tap_df

    # otap = [15]
    # dtap = [16]
    # tod = ['PM']

    otap = tap_df.sample(VECTOR_TEST_SIZE, replace=True).index
    dtap = tap_df.sample(VECTOR_TEST_SIZE, replace=True).index
    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)

    print "\notap\n", otap
    print "\ndtap\n", dtap
    print "\ntod\n", tod

    skim = network_los.tap_skim_dict.get(('LOCAL_BUS_FARE', 'PM'))
    local_bus_fare = skim.get(otap, dtap)
    print "\nraw local_bus_fare\n", local_bus_fare

    local_bus_fare = network_los.get_tappairs(otap, dtap, ('LOCAL_BUS_FARE', 'PM'))
    print "\nget_tappairs local_bus_fare\n", local_bus_fare

    local_bus_fare = network_los.get_tappairs3d(otap, dtap, tod, 'LOCAL_BUS_FARE')
    print "\nget_tappairs3d local_bus_fare\n", local_bus_fare


def get_maz_pairs():


    #    OMAZ   DMAZ  bike_logsum  bike_time  walk_perceived  walk_actual  walk_gain
    # 0  3015  22567       -4.332     12.711          43.520       33.218      303.0
    # 1  3626   3626        9.169      0.107           0.358        0.358        0.0
    # 2  3640   3192       -5.593     16.030          52.334       39.550      376.0

    #print network_los.maz2maz_df.head(10)

    # omaz = [3015, 3626, 3640]
    # dmaz = [22567, 3626, 3192]

    # sparse array so make sure the pairs are there
    maz2maz_df = network_los.maz2maz_df.sample(VECTOR_TEST_SIZE, replace=True)
    omaz = maz2maz_df.OMAZ
    dmaz = maz2maz_df.DMAZ
    print maz2maz_df.head(VECTOR_TEST_SIZE)

    print "\nomaz\n", omaz.head(VECTOR_TEST_SIZE)
    print "\ndmaz\n", dmaz.head(VECTOR_TEST_SIZE)

    # FIXME
    walk_actual = network_los.get_mazpairs(omaz, dmaz, 'walk_actual')

    print "\nget_mazpairs walk_actual\n", walk_actual


def get_maz_tap_pairs():

    maz2tap_df = network_los.maz2tap_df.sample(VECTOR_TEST_SIZE, replace=True)
    maz = maz2tap_df.MAZ
    tap = maz2tap_df.TAP

    print maz2tap_df.head(VECTOR_TEST_SIZE)

    # maz = [1, 8]
    # tap = [1764, 1598]

    drive_distance = network_los.get_maztappairs(maz, tap, "drive_distance")
    print "\nget_maz_tap_pairs drive_distance\n", drive_distance


def get_taps_mazs():

    print ""

    maz_df = network_los.maz_df.sample(VECTOR_TEST_SIZE, replace=True)
    omaz = maz_df.index

    print "\nomaz\n", omaz

    maz_tap_distance = network_los.get_taps_mazs(omaz)
    print "\nmaz_tap_distance\n", maz_tap_distance

    # when called with attribute, only returns rows with non-null attributes
    attribute = 'drive_distance'
    maz_tap_distance = network_los.get_taps_mazs(omaz, attribute)
    print "\nmaz_tap_distance w/ drive_distance\n", maz_tap_distance


def set_random_seed():
    np.random.seed(0)

# uncomment the line below to set random seed so that run results are reproducible
set_random_seed()
orca.add_injectable("set_random_seed", set_random_seed)

tracing.config_logger()

t0 = print_elapsed_time()

taz_skim_stack = orca.get_injectable('taz_skim_dict')
t0 = print_elapsed_time("load taz_skim_dict", t0)

tap_skim_stack = orca.get_injectable('tap_skim_dict')
t0 = print_elapsed_time("load tap_skim_dict", t0)

network_los = orca.get_injectable('network_los')
t0 = print_elapsed_time("load network_los", t0)

print "\n########## get_taz\n"
get_taz()
t0 = print_elapsed_time("get_taz", t0)

print "\n########## get_tap\n"
get_tap()
t0 = print_elapsed_time("get_tap", t0)

print "\n########## get_maz\n"
get_maz()
t0 = print_elapsed_time("get_maz", t0)

print "\n########## taz_skims\n"
taz_skims()
t0 = print_elapsed_time("taz_skims", t0)

print "\n########## tap_skims\n"
tap_skims()
t0 = print_elapsed_time("tap_skims", t0)

print "\n########## get_maz_pairs\n"
get_maz_pairs()
t0 = print_elapsed_time("get_maz_pairs", t0)

print "\n########## get_maz_tap_pairs\n"
get_maz_tap_pairs()
t0 = print_elapsed_time("get_maz_tap_pairs", t0)

print "\n########## get_taps_mazs\n"
get_taps_mazs()
t0 = print_elapsed_time("get_taps_mazs", t0)


t0 = print_elapsed_time()
orca.run(["best_transit_path"])
t0 = print_elapsed_time("best_transit_path", t0)

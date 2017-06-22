
import logging

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
DATA_REPO = "E:/activitysim/project/output/"
DATA_REPO = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"


COMPARE_RESULTS = False

tracing.config_logger()
logger = logging.getLogger('activitysim')


@orca.injectable()
def output_dir():
    if not os.path.exists('output'):
        os.makedirs('output')  # make directory if needed
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
        logger.info(msg)
    return time.time()

def print_elapsed_time_per_unit(msg, t0, divisor):
    unit = 1000.
    t1 = time.time()
    if msg:
        t = t1 - (t0 or t1)
        per_unit = unit * t / divisor
        msg = "Time to execute %s : %s seconds (%s per unit)" % (msg, round(t, 3), round(per_unit, 4))
        logger.info(msg)
    return time.time()


def get_taz(VECTOR_TEST_SIZE):
    # select some random rows with attributes
    taz_df = network_los.taz_df[~np.isnan(network_los.taz_df.terminal_time)]
    random_taz = taz_df.sample(VECTOR_TEST_SIZE, replace=True)
    result = network_los.get_taz(random_taz.index.values, 'terminal_time')

    if COMPARE_RESULTS:

        # Int64Index
        result2 = network_los.get_taz(random_taz.index, 'terminal_time')
        assert list(result) == list(result2)

        # Series
        result2 = network_los.get_taz(pd.Series(data=random_taz.index.values), 'terminal_time')
        assert list(result) == list(result2)

    return result


def get_tap(VECTOR_TEST_SIZE):
    tap_df = network_los.tap_df
    random_tap = tap_df.sample(VECTOR_TEST_SIZE, replace=True)
    result = network_los.get_tap(random_tap.index.values, 'TAZ')

    if COMPARE_RESULTS:

        # Int64Index
        result2 = network_los.get_tap(random_tap.index, 'TAZ')
        assert list(result) == list(result2)

        # Series
        result2 = network_los.get_tap(pd.Series(data=random_tap.index.values), 'TAZ')
        assert list(result) == list(result2)

    return result


def get_maz(VECTOR_TEST_SIZE):
    maz_df = network_los.maz_df
    random_maz = maz_df.sample(VECTOR_TEST_SIZE, replace=True)
    result = network_los.get_maz(random_maz.index.values, 'milestocoast')

    if COMPARE_RESULTS:

        # Int64Index
        result2 = network_los.get_maz(random_maz.index, 'milestocoast')
        assert list(result) == list(result2)

        # Series
        result2 = network_los.get_maz(pd.Series(data=random_maz.index.values), 'milestocoast')
        assert list(result) == list(result2)

    return result


def taz_skims(VECTOR_TEST_SIZE):
    taz_df = network_los.taz_df

    otaz = taz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    dtaz = taz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)
    sov_time = network_los.get_tazpairs3d(otaz, dtaz, tod, 'SOV_TIME')


def tap_skims(VECTOR_TEST_SIZE):
    tap_df = network_los.tap_df

    otap = tap_df.sample(VECTOR_TEST_SIZE, replace=True).index
    dtap = tap_df.sample(VECTOR_TEST_SIZE, replace=True).index
    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)
    local_bus_fare = network_los.get_tappairs3d(otap, dtap, tod, 'LOCAL_BUS_FARE')


def get_maz_pairs(VECTOR_TEST_SIZE):
    maz2maz_df = network_los.maz2maz_df.sample(VECTOR_TEST_SIZE, replace=True)
    omaz = maz2maz_df.OMAZ
    dmaz = maz2maz_df.DMAZ
    walk_actual = network_los.get_mazpairs(omaz, dmaz, 'walk_actual')


def get_maz_tap_pairs(VECTOR_TEST_SIZE):
    maz2tap_df = network_los.maz2tap_df.sample(VECTOR_TEST_SIZE, replace=True)
    maz = maz2tap_df.MAZ
    tap = maz2tap_df.TAP
    drive_distance = network_los.get_maztappairs(maz, tap, "drive_distance")


def get_taps_mazs(VECTOR_TEST_SIZE):

    maz_df = network_los.maz_df.sample(VECTOR_TEST_SIZE, replace=True)
    omaz = maz_df.index
    maz_tap_distance = network_los.get_taps_mazs(omaz)
    # when called with attribute, only returns rows with non-null attributes
    attribute = 'drive_distance'
    maz_tap_distance = network_los.get_taps_mazs(omaz, attribute)


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

# test sizes for all implemented methods
VECTOR_TEST_SIZEs = (10000, 100000, 1000000, 5000000, 10000000, 20000000)

#VECTOR_TEST_SIZEs = []

for size in VECTOR_TEST_SIZEs:

    logger.info("VECTOR_TEST_SIZE %s" % size)

    get_taz(size)
    t0 = print_elapsed_time_per_unit("get_taz", t0, size)

    get_tap(size)
    t0 = print_elapsed_time_per_unit("get_tap", t0, size)

    get_maz(size)
    t0 = print_elapsed_time_per_unit("get_maz", t0, size)

    taz_skims(size)
    t0 = print_elapsed_time_per_unit("taz_skims", t0, size)

    tap_skims(size)
    t0 = print_elapsed_time_per_unit("tap_skims", t0, size)

    get_maz_pairs(size)
    t0 = print_elapsed_time_per_unit("get_maz_pairs", t0, size)

    get_maz_tap_pairs(size)
    t0 = print_elapsed_time_per_unit("get_maz_tap_pairs", t0, size)

    get_taps_mazs(size)
    t0 = print_elapsed_time_per_unit("get_taps_mazs", t0, size)


# # taz_skims() test sizes; comment out all other methods
# VECTOR_TEST_SIZEs = (68374080, 568231216)
# for size in VECTOR_TEST_SIZEs:
#     logger.info("VECTOR_TEST_SIZE %s" % size)
#     taz_skims(size)
#     t0 = print_elapsed_time_per_unit("taz_skims", t0, size)
#
# # get_maz_pairs() test sizes; comment out all other methods
# VECTOR_TEST_SIZEs = (5073493, 10146986, 12176383, 15220479, 1522047900)
# for size in VECTOR_TEST_SIZEs:
#     logger.info("VECTOR_TEST_SIZE %s" % size)
#     get_maz_pairs(size)
#     t0 = print_elapsed_time_per_unit("get_maz_pairs", t0, size)


t0 = print_elapsed_time()
orca.run(["best_transit_path"])
t0 = print_elapsed_time("best_transit_path", t0)

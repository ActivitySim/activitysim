
import logging

from activitysim import abm
from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import simulate as asim


import pandas as pd
import numpy as np
import os
import time
import extensions

# you will want to configure this with the locations of the canonical datasets
DATA_REPO = "C:/projects/sandag-asim/toRSG/output/"
DATA_REPO = "E:/activitysim/project/output/"
DATA_REPO = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"

COMPARE_RESULTS = False

tracing.config_logger()
logger = logging.getLogger('activitysim')


@inject.injectable(override=True)
def output_dir():
    if not os.path.exists('output'):
        os.makedirs('output')  # make directory if needed
    return 'output'


@inject.injectable(override=True)
def data_dir():
    return os.path.join(DATA_REPO)


@inject.injectable(override=True)
def preload_injectables():
    # don't want to load standard skims
    return False


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
        msg = "Time to execute %s : %s seconds (%s per unit, divisor %s)" % \
              (msg, round(t, 3), round(per_unit, 4), divisor)
        logger.info(msg)
    return time.time()


def get_taz(VECTOR_TEST_SIZE):
    # select some random rows with non-null attributes

    random_taz = np.random.choice(
        network_los.taz_df.terminal_time.dropna().index.values,
        size=VECTOR_TEST_SIZE, replace=True)

    result = network_los.get_taz(random_taz, 'terminal_time')

    if COMPARE_RESULTS:

        # Int64Index
        result2 = network_los.get_taz(pd.Series(0, index=random_taz).index, 'terminal_time')
        assert list(result) == list(result2)

        # Series
        result2 = network_los.get_taz(pd.Series(data=random_taz), 'terminal_time')
        assert list(result) == list(result2)

    return result


def get_tap(VECTOR_TEST_SIZE):

    random_tap = np.random.choice(
        network_los.tap_df.index.values,
        size=VECTOR_TEST_SIZE, replace=True)

    result = network_los.get_tap(random_tap, 'TAZ')

    if COMPARE_RESULTS:

        # Int64Index
        result2 = network_los.get_tap(pd.Series(index=random_tap).index, 'TAZ')
        assert list(result) == list(result2)

        # Series
        result2 = network_los.get_tap(pd.Series(data=random_tap), 'TAZ')
        assert list(result) == list(result2)

    return result


def get_maz(VECTOR_TEST_SIZE):

    random_maz = np.random.choice(
        network_los.maz_df.index.values,
        size=VECTOR_TEST_SIZE, replace=True)

    result = network_los.get_maz(random_maz, 'milestocoast')

    if COMPARE_RESULTS:

        # Int64Index
        result2 = network_los.get_maz(pd.Series(index=random_maz).index, 'milestocoast')
        assert list(result) == list(result2)

        # Series
        result2 = network_los.get_maz(pd.Series(data=random_maz), 'milestocoast')
        assert list(result) == list(result2)

    return result


def taz_skims(VECTOR_TEST_SIZE):
    taz_values = network_los.taz_df.index.values
    otaz = np.random.choice(taz_values, size=VECTOR_TEST_SIZE, replace=True)
    dtaz = np.random.choice(taz_values, size=VECTOR_TEST_SIZE, replace=True)

    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)
    sov_time = network_los.get_tazpairs3d(otaz, dtaz, tod, 'SOV_TIME')


def tap_skims(VECTOR_TEST_SIZE):
    tap_values = network_los.tap_df.index.values

    otap = np.random.choice(tap_values, size=VECTOR_TEST_SIZE, replace=True)
    dtap = np.random.choice(tap_values, size=VECTOR_TEST_SIZE, replace=True)
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


def get_taps_mazs(VECTOR_TEST_SIZE, attribute=None):

    random_omaz = np.random.choice(network_los.maz_df.index.values, size=VECTOR_TEST_SIZE,
                                   replace=True)

    taps_mazs = network_los.get_taps_mazs(random_omaz, attribute=attribute)

    return len(taps_mazs.index)


def set_random_seed():
    np.random.seed(0)


# uncomment the line below to set random seed so that run results are reproducible
set_random_seed()
inject.add_injectable("set_random_seed", set_random_seed)

tracing.config_logger()

t0 = print_elapsed_time()

taz_skim_stack = inject.get_injectable('taz_skim_dict')
t0 = print_elapsed_time("load taz_skim_dict", t0)

tap_skim_stack = inject.get_injectable('tap_skim_dict')
t0 = print_elapsed_time("load tap_skim_dict", t0)

network_los = inject.get_injectable('network_los')
t0 = print_elapsed_time("load network_los", t0)

# test sizes for all implemented methods
VECTOR_TEST_SIZEs = (10000, 100000, 1000000, 5000000, 10000000, 20000000)

# VECTOR_TEST_SIZEs = [20000000, 40000000]

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

    result_size = get_taps_mazs(size, attribute='drive_distance')
    print_elapsed_time_per_unit("get_taps_mazs drive_distance by input", t0, size)
    t0 = print_elapsed_time_per_unit("get_taps_mazs drive_distance by output", t0, result_size)

    result_size = get_taps_mazs(size)
    print_elapsed_time_per_unit("get_taps_mazs by input", t0, size)
    t0 = print_elapsed_time_per_unit("get_taps_mazs by output", t0, result_size)

    # - not sure why, but runs faster on subsequent calls time...
    result_size = get_taps_mazs(size)
    print_elapsed_time_per_unit("get_taps_mazs2 by input", t0, size)
    t0 = print_elapsed_time_per_unit("get_taps_mazs2 by output", t0, result_size)

    result_size = get_taps_mazs(size)
    print_elapsed_time_per_unit("get_taps_mazs3 by input", t0, size)
    t0 = print_elapsed_time_per_unit("get_taps_mazs3 by output", t0, result_size)


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

# bug
# t0 = print_elapsed_time()
# pipeline.run(models=["best_transit_path"], resume_after=None)
# t0 = print_elapsed_time("best_transit_path", t0)

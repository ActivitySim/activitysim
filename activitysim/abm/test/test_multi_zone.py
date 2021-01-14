# ActivitySim
# See full license in LICENSE.txt.
import os
import logging
import pkg_resources

import openmatrix as omx
import numpy as np
import numpy.testing as npt

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from activitysim.core import random
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

HOUSEHOLDS_SAMPLE_SIZE = 50
EXPECT_2_ZONE_TOUR_COUNT = 120

# 3-zone is currently big and slow - so set this way low
HOUSEHOLDS_SAMPLE_SIZE_3_ZONE = 5
EXPECT_3_ZONE_TOUR_COUNT = 13


# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 257341

# household with WALK_TRANSIT tours and trips
HH_ID_3_ZONE = 2848373

#  [ 257341 1234246 1402915 1511245 1931827 1931908 2307195 2366390 2408855
# 2518594 2549865  982981 1594365 1057690 1234121 2098971]


def example_path(dirname):
    resource = os.path.join('examples', 'example_multiple_zone', dirname)
    return pkg_resources.resource_filename('activitysim', resource)


def mtc_example_path(dirname):
    resource = os.path.join('examples', 'example_mtc', dirname)
    return pkg_resources.resource_filename('activitysim', resource)


def setup_dirs(configs_dir, data_dir):

    print(f"configs_dir: {configs_dir}")
    inject.add_injectable('configs_dir', configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable('output_dir', output_dir)

    print(f"data_dir: {data_dir}")
    inject.add_injectable('data_dir', data_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')
    tracing.delete_output_files('omx')


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def close_handlers():

    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def inject_settings(**kwargs):

    for k in kwargs:
        if k == "two_zone":
            if kwargs[k]:
                settings = config.read_settings_file('settings.yaml', mandatory=True)
            else:
                settings = config.read_settings_file('settings_static.yaml', mandatory=True)
        settings[k] = kwargs[k]

    inject.add_injectable("settings", settings)

    return settings


def full_run(configs_dir, data_dir,
             resume_after=None, chunk_size=0,
             households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
             trace_hh_id=None, trace_od=None, check_for_variability=None, two_zone=True):

    setup_dirs(configs_dir, data_dir)

    settings = inject_settings(
        two_zone=two_zone,
        households_sample_size=households_sample_size,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_od=trace_od,
        check_for_variability=check_for_variability,
        use_shadow_pricing=False)  # shadow pricing breaks replicability when sample_size varies

    MODELS = settings['models']

    pipeline.run(models=MODELS, resume_after=resume_after)

    tours = pipeline.get_table('tours')
    tour_count = len(tours.index)

    return tour_count


def get_trace_csv(file_name):

    file_name = config.output_file_path(file_name)
    df = pd.read_csv(file_name)

    #        label    value_1    value_2    value_3    value_4
    # 0    tour_id        38         201         39         40
    # 1       mode  DRIVE_LOC  DRIVE_COM  DRIVE_LOC  DRIVE_LOC
    # 2  person_id    1888694    1888695    1888695    1888696
    # 3  tour_type       work   othmaint       work     school
    # 4   tour_num          1          1          1          1

    # transpose df and rename columns
    labels = df.label.values
    df = df.transpose()[1:]
    df.columns = labels

    return df


def regress_2_zone():
    pass


def regress_3_zone():

    tours_df = pipeline.get_table('tours')
    assert len(tours_df[tours_df.tour_mode == 'WALK_TRANSIT']) > 0

    # should cache atap and btap for transit modes only
    for c in ['od_atap', 'od_btap', 'do_atap', 'do_btap']:
        # tour_mode_choice sets non-transit taps to 0
        assert not (tours_df[tours_df.tour_mode.isin(['WALK_TRANSIT', 'DRIVE_TRANSIT'])][c] == 0).any()
        baddies = ~tours_df.tour_mode.isin(['WALK_TRANSIT', 'DRIVE_TRANSIT']) & (tours_df[c] != 0)
        if baddies.any():
            print(tours_df[baddies][['tour_type', 'tour_mode', 'od_atap', 'od_btap', 'do_atap', 'do_btap']])
            assert False


def test_full_run_2_zone():

    tour_count = full_run(configs_dir=[example_path('configs_2_zone'), mtc_example_path('configs')],
                          data_dir=example_path('data_2'),
                          trace_hh_id=HH_ID, check_for_variability=True,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE, two_zone=True)

    print("tour_count", tour_count)

    assert(tour_count == EXPECT_2_ZONE_TOUR_COUNT), \
        "EXPECT_2_ZONE_TOUR_COUNT %s but got tour_count %s" % (EXPECT_2_ZONE_TOUR_COUNT, tour_count)

    regress_2_zone()

    pipeline.close_pipeline()


def test_full_run_3_zone():

    tour_count = full_run(configs_dir=[example_path('configs_3_zone'), mtc_example_path('configs')],
                          data_dir=example_path('data_3'),
                          trace_hh_id=HH_ID_3_ZONE, check_for_variability=True,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE_3_ZONE, two_zone=False)

    print("tour_count", tour_count)

    assert(tour_count == EXPECT_3_ZONE_TOUR_COUNT), \
        "EXPECT_3_ZONE_TOUR_COUNT %s but got tour_count %s" % (EXPECT_3_ZONE_TOUR_COUNT, tour_count)

    regress_3_zone()

    pipeline.close_pipeline()


if __name__ == "__main__":

    from activitysim import abm  # register injectables
    print("running test_full_run_2_zone")
    test_full_run_2_zone()

    print("running test_full_run_3_zone")
    test_full_run_3_zone()

    # teardown_function(None)

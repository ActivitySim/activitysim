# ActivitySim
# See full license in LICENSE.txt.

from __future__ import print_function

from builtins import str
import os
import logging

import pandas as pd
import pandas.util.testing as pdt
import pytest
import yaml

from activitysim.core import random
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100

# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 1528374

# test households with all tour types
# [ 897097  921736  934309 1263203 1528374 1577292 1685008 1809710 2123173 2124138]


SKIP_FULL_RUN = True
SKIP_FULL_RUN = False


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


def inject_settings(configs_dir, households_sample_size, chunk_size=None,
                    trace_hh_id=None, trace_od=None, check_for_variability=None):

    with open(os.path.join(configs_dir, 'settings.yaml')) as f:
        settings = yaml.load(f)
        settings['households_sample_size'] = households_sample_size
        if chunk_size is not None:
            settings['chunk_size'] = chunk_size
        if trace_hh_id is not None:
            settings['trace_hh_id'] = trace_hh_id
        if trace_od is not None:
            settings['trace_od'] = trace_od
        if check_for_variability is not None:
            settings['check_for_variability'] = check_for_variability

        inject.add_injectable("settings", settings)

    return settings


def test_rng_access():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inject.add_injectable("data_dir", data_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    inject.clear_cache()

    inject.add_injectable('rng_base_seed', 0)

    pipeline.open_pipeline()

    rng = pipeline.get_rn_generator()

    assert isinstance(rng, random.Random)

    pipeline.close_pipeline()
    inject.clear_cache()


def regress_mini_auto():

    auto_choice = pipeline.get_table("households").auto_ownership

    # regression test: these are among the first 10 households in households table
    hh_ids = [961042, 238146, 23730]
    choices = [1, 1, 0]
    expected_choice = pd.Series(choices, index=pd.Index(hh_ids, name="household_id"),
                                name='auto_ownership')

    print("auto_choice\n", auto_choice.head(10))
    """
    auto_choice
    household_id
    961042     1
    238146     1
    701954     1
    644046     1
    23730      0
    460343     1
    25354      1
    92615      1
    1950284    0
    2122448    0
    Name: auto_ownership, dtype: int64
    """
    pdt.assert_series_equal(auto_choice[hh_ids], expected_choice)


def regress_mini_mtf():

    mtf_choice = pipeline.get_table("persons").mandatory_tour_frequency

    # these choices are for pure regression - their appropriateness has not been checked
    per_ids = [26986, 92615, 93332]
    choices = ['school1', 'work1', 'work1']
    expected_choice = pd.Series(choices, index=pd.Index(per_ids, name='person_id'),
                                name='mandatory_tour_frequency')

    print("mtf_choice\n", mtf_choice.dropna().head(5))
    """
    mtf_choice
    26986    school1
    92615      work1
    93332      work1
    93390      work1
    93898      work1
    Name: mandatory_tour_frequency, dtype: object
    """
    pdt.assert_series_equal(mtf_choice[per_ids], expected_choice)


def test_mini_pipeline_run():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inject.add_injectable("data_dir", data_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    inject.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'initialize_landuse',
        'compute_accessibility',
        'initialize_households',
        'school_location_sample',
        'school_location_logsums',
        'school_location_simulate',
        'workplace_location_sample',
        'workplace_location_logsums',
        'workplace_location_simulate',
        'auto_ownership_simulate'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    regress_mini_auto()

    pipeline.run_model('cdap_simulate')
    pipeline.run_model('mandatory_tour_frequency')

    regress_mini_mtf()

    # try to get a non-existant table
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("bogus")
    assert "never checkpointed" in str(excinfo.value)

    # try to get an existing table from a non-existant checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("households", checkpoint_name="bogus")
    assert "not in checkpoints" in str(excinfo.value)

    pipeline.close_pipeline()
    inject.clear_cache()

    close_handlers()


def test_mini_pipeline_run2():

    # the important thing here is that we should get
    # exactly the same results as for test_mini_pipeline_run
    # when we restart pipeline

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inject.add_injectable("data_dir", data_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    inject.clear_cache()

    # should be able to get this BEFORE pipeline is opened
    checkpoints_df = pipeline.get_checkpoints()
    prev_checkpoint_count = len(checkpoints_df.index)

    # print "checkpoints_df\n", checkpoints_df[['checkpoint_name']]
    assert prev_checkpoint_count == 12

    pipeline.open_pipeline('auto_ownership_simulate')

    regress_mini_auto()

    # try to run a model already in pipeline
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.run_model('auto_ownership_simulate')
    assert "run model 'auto_ownership_simulate' more than once" in str(excinfo.value)

    # and these new ones
    pipeline.run_model('cdap_simulate')
    pipeline.run_model('mandatory_tour_frequency')

    regress_mini_mtf()

    # should be able to get this before pipeline is closed (from existing open store)
    checkpoints_df = pipeline.get_checkpoints()
    assert len(checkpoints_df.index) == prev_checkpoint_count

    pipeline.close_pipeline()
    inject.clear_cache()


def full_run(resume_after=None, chunk_size=0,
             households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
             trace_hh_id=None, trace_od=None, check_for_variability=None):

    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'configs')
    inject.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inject.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    settings = inject_settings(
        configs_dir,
        households_sample_size=households_sample_size,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_od=trace_od,
        check_for_variability=check_for_variability)

    inject.clear_cache()

    tracing.config_logger()

    MODELS = settings['models']

    pipeline.run(models=MODELS, resume_after=resume_after)

    tours = pipeline.get_table('tours')
    tour_count = len(tours.index)

    return tour_count


def get_trace_csv(file_name):

    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    df = pd.read_csv(os.path.join(output_dir, file_name))

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


EXPECT_TOUR_COUNT = 158


def regress_tour_modes(tours_df):

    mode_cols = ['tour_mode', 'person_id', 'tour_type', 'tour_num', 'tour_category']

    tours_df = tours_df[tours_df.household_id == HH_ID]
    tours_df = tours_df.sort_values(by=['person_id', 'tour_category', 'tour_num'])

    print("mode_df\n", tours_df[mode_cols])

    """
    tour_id        tour_mode  person_id tour_type  tour_num  tour_category
    tour_id
    96881402            WALK    3340738  business         1         atwork
    96881411     SHARED2FREE    3340738    eatout         1          joint
    96881429        WALK_LOC    3340738      work         1      mandatory
    96881427        WALK_LOC    3340738  shopping         1  non_mandatory
    96881454        WALK_LOC    3340739    school         1      mandatory
    96881456  DRIVEALONEFREE    3340739  shopping         1  non_mandatory
    96881437     SHARED2FREE    3340739    eatout         2  non_mandatory
    96881457        WALK_LOC    3340739    social         3  non_mandatory
    """

    EXPECT_PERSON_IDS = [
        3340738,
        3340738,
        3340738,
        3340738,
        3340739,
        3340739,
        3340739,
        3340739
        ]

    EXPECT_TOUR_TYPES = [
        'business',
        'eatout',
        'work',
        'shopping',
        'school',
        'shopping',
        'eatout',
        'social'
        ]

    EXPECT_MODES = [
        'WALK',
        'SHARED2FREE',
        'WALK_LOC',
        'WALK_LOC',
        'WALK_LOC',
        'DRIVEALONEFREE',
        'SHARED2FREE',
        'WALK_LOC'
        ]

    assert (tours_df.person_id.values == EXPECT_PERSON_IDS).all()
    assert (tours_df.tour_type.values == EXPECT_TOUR_TYPES).all()
    assert (tours_df.tour_mode.values == EXPECT_MODES).all()


def regress():

    tours_df = pipeline.get_table('tours')

    regress_tour_modes(tours_df)

    assert tours_df.shape[0] > 0
    assert not tours_df.tour_mode.isnull().any()

    trips_df = pipeline.get_table('trips')
    assert trips_df.shape[0] > 0
    assert not trips_df.purpose.isnull().any()
    assert not trips_df.depart.isnull().any()
    assert not trips_df.trip_mode.isnull().any()

    # should be at least two tours per trip
    assert trips_df.shape[0] >= 2*tours_df.shape[0]


def test_full_run1():

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID, check_for_variability=True,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    print("tour_count", tour_count)

    assert(tour_count == EXPECT_TOUR_COUNT)

    regress()

    pipeline.close_pipeline()


def test_full_run2():

    # resume_after should successfully load tours table and replicate results

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(resume_after='non_mandatory_tour_scheduling', trace_hh_id=HH_ID)

    assert(tour_count == EXPECT_TOUR_COUNT)

    regress()

    pipeline.close_pipeline()


def test_full_run_with_chunks():

    # should get the same result with different chunk size

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
                          chunk_size=500000)

    assert(tour_count == EXPECT_TOUR_COUNT)

    regress()

    pipeline.close_pipeline()


def test_full_run_stability():

    # hh should get the same result with different sample size

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE+10)

    regress()

    pipeline.close_pipeline()


if __name__ == "__main__":

    print("running test_full_run1")
    test_full_run1()
    # teardown_function(None)

# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import os

import numpy as np
import openmatrix as omx
import pandas as pd
import pandas.testing as pdt
import pkg_resources
import pytest

from activitysim.core import random, tracing, workflow

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 50
HOUSEHOLDS_SAMPLE_RATE = 0.01  # HOUSEHOLDS_SAMPLE_RATE / 5000 households

# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 257341

#  [ 257341 1234246 1402915 1511245 1931827 1931908 2307195 2366390 2408855
# 2518594 2549865  982981 1594365 1057690 1234121 2098971]

# SKIP_FULL_RUN = True
SKIP_FULL_RUN = False


def example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def setup_dirs(ancillary_configs_dir=None, data_dir=None):
    # ancillary_configs_dir is used by run_mp to test multiprocess

    test_pipeline_configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    example_configs_dir = example_path("configs")
    configs_dir = [test_pipeline_configs_dir, example_configs_dir]

    if ancillary_configs_dir is not None:
        configs_dir = [ancillary_configs_dir] + configs_dir

    output_dir = os.path.join(os.path.dirname(__file__), "output")

    if not data_dir:
        data_dir = example_path("data")

    state = workflow.State.make_default(
        configs_dir=configs_dir,
        output_dir=output_dir,
        data_dir=data_dir,
    )

    state.logging.config_logger()

    state.tracing.delete_output_files("csv")
    state.tracing.delete_output_files("txt")
    state.tracing.delete_output_files("yaml")
    state.tracing.delete_output_files("omx")

    return state


def close_handlers():
    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_rng_access():
    state = setup_dirs()
    state.settings.rng_base_seed = 0

    state.checkpoint.restore()

    rng = state.get_rn_generator()

    assert isinstance(rng, random.Random)

    state.checkpoint.close_store()


def regress_mini_auto(state: workflow.State):
    # regression test: these are among the middle households in households table
    # should be the same results as in run_mp (multiprocessing) test case
    hh_ids = [1099626, 1173905, 1196298, 1286259]
    choices = [1, 1, 0, 0]
    expected_choice = pd.Series(
        choices, index=pd.Index(hh_ids, name="household_id"), name="auto_ownership"
    )

    auto_choice = (
        state.checkpoint.load_dataframe("households").sort_index().auto_ownership
    )

    offset = (
        HOUSEHOLDS_SAMPLE_SIZE // 2
    )  # choose something midway as hh_id ordered by hh size
    print("auto_choice\n%s" % auto_choice.head(offset).tail(4))

    auto_choice = auto_choice.reindex(hh_ids)

    """
    auto_choice
    household_id
    1099626    1
    1173905    1
    1196298    0
    1286259    0
    Name: auto_ownership, dtype: int64
    """
    pdt.assert_series_equal(auto_choice, expected_choice, check_dtype=False)


def regress_mini_mtf(state: workflow.State):
    mtf_choice = (
        state.checkpoint.load_dataframe("persons").sort_index().mandatory_tour_frequency
    )

    # these choices are for pure regression - their appropriateness has not been checked
    per_ids = [2566701, 2566702, 3061895]
    choices = ["school1", "school1", "work1"]
    expected_choice = pd.Series(
        choices,
        index=pd.Index(per_ids, name="person_id"),
        name="mandatory_tour_frequency",
    )

    mtf_choice = mtf_choice[mtf_choice != ""]  # drop null (empty string) choices

    offset = len(mtf_choice) // 2  # choose something midway as hh_id ordered by hh size
    print("mtf_choice\n%s" % mtf_choice.head(offset).tail(3))

    """
    mtf_choice
    person_id
    2566701    school1
    2566702    school1
    3061895      work1
    Name: mandatory_tour_frequency, dtype: object
    """
    pdt.assert_series_equal(
        mtf_choice.astype(str).reindex(per_ids), expected_choice, check_dtype=False
    )


def regress_mini_location_choice_logsums(state: workflow.State):
    persons = state.checkpoint.load_dataframe("persons")

    # DEST_CHOICE_LOGSUM_COLUMN_NAME is specified in school_location.yaml and should be assigned
    assert "school_location_logsum" in persons
    assert not persons.school_location_logsum.isnull().all()

    # DEST_CHOICE_LOGSUM_COLUMN_NAME is NOT specified in workplace_location.yaml
    assert "workplace_location_logsum" not in persons


def test_mini_pipeline_run():
    from activitysim.abm.tables.skims import network_los_preload

    state = setup_dirs()
    state.get(network_los_preload)

    state.settings.households_sample_size = HOUSEHOLDS_SAMPLE_SIZE
    state.network_settings.write_skim_cache = True

    _MODELS = [
        "initialize_landuse",
        "compute_accessibility",
        "initialize_households",
        "school_location",
        "workplace_location",
        "auto_ownership_simulate",
    ]

    state.run(models=_MODELS, resume_after=None)

    regress_mini_auto(state)

    state.run.by_name("cdap_simulate")
    state.run.by_name("mandatory_tour_frequency")

    regress_mini_mtf(state)
    regress_mini_location_choice_logsums(state)

    # try to get a non-existant table
    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("bogus")
    assert "never checkpointed" in str(excinfo.value)

    # try to get an existing table from a non-existant checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("households", checkpoint_name="bogus")
    assert "not in checkpoints" in str(excinfo.value)

    # should create optional workplace_location_sample table
    workplace_location_sample_df = state.checkpoint.load_dataframe(
        "workplace_location_sample"
    )
    assert "mode_choice_logsum" in workplace_location_sample_df

    state.checkpoint.close_store()
    close_handlers()


def test_mini_pipeline_run2():
    # the important thing here is that we should get
    # exactly the same results as for test_mini_pipeline_run
    # when we restart pipeline

    state = setup_dirs()
    from activitysim.abm.tables.skims import network_los_preload

    state.get(network_los_preload)

    state.settings.households_sample_size = HOUSEHOLDS_SAMPLE_SIZE
    state.network_settings.read_skim_cache = True

    # should be able to get this BEFORE pipeline is opened
    checkpoints_df = state.checkpoint.get_inventory()
    prev_checkpoint_count = len(checkpoints_df.index)

    assert "auto_ownership_simulate" in checkpoints_df.checkpoint_name.values
    assert "cdap_simulate" in checkpoints_df.checkpoint_name.values
    assert "mandatory_tour_frequency" in checkpoints_df.checkpoint_name.values

    state.checkpoint.restore("auto_ownership_simulate")

    regress_mini_auto(state)

    # try to run a model already in pipeline
    with pytest.raises(RuntimeError) as excinfo:
        state.run.by_name("auto_ownership_simulate")
    assert "run model 'auto_ownership_simulate' more than once" in str(excinfo.value)

    # and these new ones
    state.run.by_name("cdap_simulate")
    state.run.by_name("mandatory_tour_frequency")

    regress_mini_mtf(state)

    # should be able to get this before pipeline is closed (from existing open store)
    checkpoints_df = state.checkpoint.get_inventory()
    assert len(checkpoints_df.index) == prev_checkpoint_count

    # - write list of override_hh_ids to override_hh_ids.csv in data for use in next test
    num_hh_ids = 10
    hh_ids = state.checkpoint.load_dataframe("households").head(num_hh_ids).index.values
    hh_ids = pd.DataFrame({"household_id": hh_ids})

    hh_ids_path = state.filesystem.get_data_file_path("override_hh_ids.csv")
    hh_ids.to_csv(hh_ids_path, index=False, header=True)

    state.checkpoint.close_store()
    close_handlers()


def test_mini_pipeline_run3():
    # test that hh_ids setting overrides household sampling

    state = setup_dirs()
    state.settings.hh_ids = "override_hh_ids.csv"

    households = state.get_dataframe("households")

    override_hh_ids = pd.read_csv(
        state.filesystem.get_data_file_path("override_hh_ids.csv")
    )

    print("\noverride_hh_ids\n%s" % override_hh_ids)

    print("\nhouseholds\n%s" % households.index)

    assert households.shape[0] == override_hh_ids.shape[0]
    assert households.index.isin(override_hh_ids.household_id).all()

    close_handlers()


def full_run(
    resume_after=None,
    chunk_size=0,
    households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
    trace_hh_id=None,
    trace_od=None,
    check_for_variability=False,
):
    state = setup_dirs()

    state.settings.households_sample_size = households_sample_size
    state.settings.chunk_size = chunk_size
    state.settings.trace_hh_id = trace_hh_id
    state.settings.trace_od = trace_od
    state.settings.testing_fail_trip_destination = False
    state.settings.check_for_variability = check_for_variability
    state.settings.want_dest_choice_sample_tables = False
    state.settings.use_shadow_pricing = False

    # FIXME should enable testing_fail_trip_destination?

    MODELS = state.settings.models

    state.run(models=MODELS, resume_after=resume_after)

    tours = state.checkpoint.load_dataframe("tours")
    tour_count = len(tours.index)

    return state, tour_count


EXPECT_TOUR_COUNT = 121


def regress_tour_modes(tours_df):
    mode_cols = ["tour_mode", "person_id", "tour_type", "tour_num", "tour_category"]

    tours_df = tours_df[tours_df.household_id == HH_ID]
    # convert tour_category from categorical to string for comparison
    tours_df.tour_category = tours_df.tour_category.astype(str)
    tours_df = tours_df.sort_values(by=["person_id", "tour_category", "tour_num"])

    print("mode_df\n%s" % tours_df[mode_cols])

    """
                 tour_mode  person_id tour_type  tour_num  tour_category
    tour_id
    13327106         WALK     325051  othdiscr         1          joint
    13327130         WALK     325051      work         1      mandatory
    13327131  SHARED3FREE     325051      work         2      mandatory
    13327132         WALK     325052  business         1         atwork
    13327171     WALK_LOC     325052      work         1      mandatory
    13327160         WALK     325052  othmaint         1  non_mandatory
    """

    EXPECT_PERSON_IDS = [
        325051,
        325051,
        325051,
        325052,
        325052,
        325052,
    ]

    EXPECT_TOUR_TYPES = ["othdiscr", "work", "work", "business", "work", "othmaint"]

    EXPECT_MODES = [
        "WALK",
        "WALK",
        "SHARED3FREE",
        "WALK",
        "WALK_LOC",
        "WALK",
    ]

    assert len(tours_df) == len(EXPECT_PERSON_IDS)
    assert (tours_df.person_id.values == EXPECT_PERSON_IDS).all()
    assert (tours_df.tour_type.astype(str).values == EXPECT_TOUR_TYPES).all()
    assert (tours_df.tour_mode.astype(str).values == EXPECT_MODES).all()


def regress(state: workflow.State):
    persons_df = state.checkpoint.load_dataframe("persons")
    persons_df = persons_df[persons_df.household_id == HH_ID]
    print("persons_df\n%s" % persons_df[["value_of_time", "distance_to_work"]])

    """
    persons_df
     person_id  value_of_time  distance_to_work
    person_id
    3249922        23.349532              0.62
    3249923        23.349532              0.62
    """

    tours_df = state.checkpoint.load_dataframe("tours")

    regress_tour_modes(tours_df)

    assert tours_df.shape[0] > 0
    assert not tours_df.tour_mode.isnull().any()

    # optional logsum column was added to all tours except mandatory
    assert "destination_logsum" in tours_df
    if (
        tours_df.destination_logsum.isnull() != (tours_df.tour_category == "mandatory")
    ).any():
        print(
            tours_df[
                (
                    tours_df.destination_logsum.isnull()
                    != (tours_df.tour_category == "mandatory")
                )
            ]
        )
    assert (
        tours_df.destination_logsum.isnull() == (tours_df.tour_category == "mandatory")
    ).all()

    # mode choice logsum calculated for all tours
    assert "mode_choice_logsum" in tours_df
    assert not tours_df.mode_choice_logsum.isnull().any()

    trips_df = state.checkpoint.load_dataframe("trips")
    assert trips_df.shape[0] > 0
    assert not trips_df.purpose.isnull().any()
    assert not trips_df.depart.isnull().any()
    assert not trips_df.trip_mode.isnull().any()

    # mode_choice_logsum calculated for all trips
    assert not trips_df.mode_choice_logsum.isnull().any()

    # should be at least two tours per trip
    assert trips_df.shape[0] >= 2 * tours_df.shape[0]

    # write_trip_matrices
    trip_matrices_file = state.get_output_file_path("trips_md.omx")
    assert os.path.exists(trip_matrices_file)
    trip_matrices = omx.open_file(trip_matrices_file)
    assert trip_matrices.shape() == (25, 25)

    assert "WALK_MD" in trip_matrices.list_matrices()
    walk_trips = np.array(trip_matrices["WALK_MD"])
    assert walk_trips.dtype == np.dtype("float64")

    trip_matrices.close()


def test_full_run1():
    if SKIP_FULL_RUN:
        return

    state, tour_count = full_run(
        trace_hh_id=HH_ID,
        check_for_variability=True,
        households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
    )

    print("tour_count", tour_count)

    assert (
        tour_count == EXPECT_TOUR_COUNT
    ), "EXPECT_TOUR_COUNT %s but got tour_count %s" % (EXPECT_TOUR_COUNT, tour_count)

    regress(state)

    state.checkpoint.close_store()


def test_full_run2():
    # resume_after should successfully load tours table and replicate results

    if SKIP_FULL_RUN:
        return

    state, tour_count = full_run(
        resume_after="non_mandatory_tour_scheduling", trace_hh_id=HH_ID
    )

    assert (
        tour_count == EXPECT_TOUR_COUNT
    ), "EXPECT_TOUR_COUNT %s but got tour_count %s" % (EXPECT_TOUR_COUNT, tour_count)

    regress(state)

    state.checkpoint.close_store()


def test_full_run3_with_chunks():
    # should get the same result with different chunk size

    if SKIP_FULL_RUN:
        return

    state, tour_count = full_run(
        trace_hh_id=HH_ID,
        households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
        chunk_size=500000,
    )

    assert (
        tour_count == EXPECT_TOUR_COUNT
    ), "EXPECT_TOUR_COUNT %s but got tour_count %s" % (EXPECT_TOUR_COUNT, tour_count)

    regress(state)

    state.checkpoint.close_store()


def test_full_run4_stability():
    # hh should get the same result with different sample size

    if SKIP_FULL_RUN:
        return

    state, tour_count = full_run(
        trace_hh_id=HH_ID, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE - 10
    )

    regress(state)

    state.checkpoint.close_store()


def test_full_run5_singleton():
    # should work with only one hh
    # run with minimum chunk size to drive potential chunking errors in models
    # where choosers has multiple rows that all have to be included in the same chunk

    if SKIP_FULL_RUN:
        return

    state, tour_count = full_run(
        trace_hh_id=HH_ID, households_sample_size=1, chunk_size=1
    )

    regress(state)

    state.checkpoint.close_store()


if __name__ == "__main__":
    print("running test_full_run1")
    test_full_run1()
    # teardown_function(None)

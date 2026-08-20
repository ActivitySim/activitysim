import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openmatrix as omx
import pandas as pd
import pytest

from activitysim.abm.models import park_and_ride_lot_choice as pnr_lot_choice
from activitysim.abm.models.util import park_and_ride_capacity as pnr_capacity
from activitysim.core.configuration.base import ComputeSettings
from activitysim.core import los, workflow


class _Settings(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as err:
            raise AttributeError(item) from err

    def __setattr__(self, key, value):
        self[key] = value


class _EmptyPreprocessor:
    TABLES = []

    def __bool__(self):
        return False


def _write_model_files(state, spec_name, spec_body, coeff_name, coeff_body):
    config_dir = state.filesystem.get_config_file_path("settings.yaml").parent
    (config_dir / spec_name).write_text(spec_body.strip())
    (config_dir / coeff_name).write_text(coeff_body.strip())


@pytest.fixture
def state(tmp_path):
    root = tmp_path
    config_dir = root / "configs"
    data_dir = root / "data"
    output_dir = root / "output"

    config_dir.mkdir()
    data_dir.mkdir()
    output_dir.mkdir()

    (config_dir / "settings.yaml").write_text(
        """
input_table_list:
  - tablename: households
  - tablename: persons
  - tablename: land_use
""".strip()
    )

    (config_dir / "network_los.yaml").write_text(
        """
zone_system: 1
taz_skims: skims.omx
skim_time_periods:
    time_window: 1440
    period_minutes: 60
    periods: [0, 24]
    labels: ['AM']
""".strip()
    )

    # 2D skim used by destination eligibility tests.
    skim_matrix = np.zeros((5, 5), dtype=np.float32)
    skim_matrix[0, 2] = 1.0  # zone 1 -> 3
    skim_matrix[1, 4] = 2.0  # zone 2 -> 5

    # 3D skims (single AM period) used by utility-calculation tests.
    drive_time_am = np.full((5, 5), 50.0, dtype=np.float32)
    transit_time_am = np.full((5, 5), 50.0, dtype=np.float32)

    # Drive skims: home<->lot terms used by olt_skims and lot_skims.
    # Homes are zones 3 and 4, lots are zones 1 and 2.
    drive_time_am[2, 0] = 2.0  # home 3 -> lot 1
    drive_time_am[2, 1] = 6.0  # home 3 -> lot 2
    drive_time_am[3, 0] = 7.0  # home 4 -> lot 1
    drive_time_am[3, 1] = 1.0  # home 4 -> lot 2
    drive_time_am[0, 2] = 2.0  # lot 1 -> home 3
    drive_time_am[1, 2] = 6.0  # lot 2 -> home 3
    drive_time_am[0, 3] = 7.0  # lot 1 -> home 4
    drive_time_am[1, 3] = 1.0  # lot 2 -> home 4

    # Transit skims: lot<->destination terms used by ldt_skims and dlt_skims.
    # Destinations are zones 4 and 5.
    transit_time_am[0, 4] = 2.0  # lot 1 -> dest 5
    transit_time_am[1, 4] = 6.0  # lot 2 -> dest 5
    transit_time_am[0, 3] = 4.0  # lot 1 -> dest 4
    transit_time_am[1, 3] = 1.0  # lot 2 -> dest 4
    transit_time_am[4, 0] = 2.0  # dest 5 -> lot 1
    transit_time_am[4, 1] = 6.0  # dest 5 -> lot 2
    transit_time_am[3, 0] = 4.0  # dest 4 -> lot 1
    transit_time_am[3, 1] = 1.0  # dest 4 -> lot 2

    with omx.open_file(data_dir / "skims.omx", "w") as skims:
        skims["TRANSIT_TIME"] = skim_matrix
        skims["DRIVE_TIME__AM"] = drive_time_am
        skims["TRANSIT_TIME__AM"] = transit_time_am
        skims.create_mapping("zone_number", [1, 2, 3, 4, 5])

    # Minimal files used by run_park_and_ride_lot_choice reads.
    (config_dir / "park_and_ride_lot_choice.csv").write_text(
        """Description,Expression,coefficient
constant,@1,coef_one
""".strip()
    )
    (config_dir / "park_and_ride_lot_choice_coeffs.csv").write_text(
        """coefficient_name,value,constrain
coef_one,1.0,F
""".strip()
    )

    s = workflow.State.make_default(root)
    s.add_table("persons", pd.DataFrame(index=pd.Index([1, 2, 3, 4], name="person_id")))
    s.add_table(
        "households",
        pd.DataFrame(
            {"sample_rate": [1.0, 1.0]}, index=pd.Index([1, 2], name="household_id")
        ),
    )
    s.add_table(
        "land_use",
        pd.DataFrame({"pnr_spaces": [1, 1]}, index=pd.Index([1, 2], name="zone_id")),
    )

    return s


@pytest.fixture
def network_los(state):
    nl = los.Network_LOS(state)
    nl.skim_dicts["taz"] = nl.create_skim_dict("taz")
    return nl


@pytest.fixture
def model_settings_base():
    return _Settings(
        SPEC="park_and_ride_lot_choice.csv",
        COEFFICIENTS="park_and_ride_lot_choice_coeffs.csv",
        CONSTANTS={},
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        TRANSIT_SKIMS_FOR_ELIGIBILITY=None,
        LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST=None,
        CHOOSER_FILTER_EXPR=None,
        explicit_chunk=0,
        compute_settings=None,
        preprocessor=_EmptyPreprocessor(),
        alts_preprocessor=None,
    )


def test_filter_chooser_with_landuse_eligibility_column(
    state, network_los, model_settings_base
):
    """Filters choosers by a boolean land-use eligibility column."""
    model_settings_base.LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST = "pnr_eligible"

    land_use = pd.DataFrame(
        {
            "pnr_spaces": [10, 5, 0],
            "pnr_eligible": [True, False, True],
        },
        index=pd.Index([1, 2, 3], name="zone_id"),
    )
    choosers = pd.DataFrame(
        {"destination": [1, 2, 3, 2]},
        index=pd.Index([100, 101, 102, 103], name="tour_id"),
    )
    pnr_alts = land_use[land_use["pnr_spaces"] > 0]

    filtered = pnr_lot_choice.filter_chooser_to_transit_accessible_destinations(
        state=state,
        choosers=choosers,
        land_use=land_use,
        pnr_alts=pnr_alts,
        network_los=network_los,
        model_settings=model_settings_base,
        choosers_dest_col_name="destination",
    )

    assert filtered.index.tolist() == [100, 102]


def test_filter_chooser_with_skim_eligibility(state, network_los, model_settings_base):
    """Filters choosers using skim-based transit accessibility checks."""
    model_settings_base.TRANSIT_SKIMS_FOR_ELIGIBILITY = ["TRANSIT_TIME"]

    land_use = pd.DataFrame(
        {"pnr_spaces": [5, 10]},
        index=pd.Index([1, 2], name="zone_id"),
    )
    pnr_alts = land_use.copy()
    choosers = pd.DataFrame(
        {"destination": [3, 4, 5]},
        index=pd.Index([100, 101, 102], name="tour_id"),
    )

    filtered = pnr_lot_choice.filter_chooser_to_transit_accessible_destinations(
        state=state,
        choosers=choosers,
        land_use=land_use,
        pnr_alts=pnr_alts,
        network_los=network_los,
        model_settings=model_settings_base,
        choosers_dest_col_name="destination",
    )

    assert filtered["destination"].tolist() == [3, 5]


def test_filter_chooser_raises_for_invalid_skim(
    state, network_los, model_settings_base
):
    """Raises an error when an eligibility skim name is not available."""
    model_settings_base.TRANSIT_SKIMS_FOR_ELIGIBILITY = ["DOES_NOT_EXIST"]

    land_use = pd.DataFrame(
        {"pnr_spaces": [5]},
        index=pd.Index([1], name="zone_id"),
    )
    choosers = pd.DataFrame({"destination": [3]}, index=pd.Index([100], name="tour_id"))

    with pytest.raises(ValueError):
        pnr_lot_choice.filter_chooser_to_transit_accessible_destinations(
            state=state,
            choosers=choosers,
            land_use=land_use,
            pnr_alts=land_use,
            network_los=network_los,
            model_settings=model_settings_base,
            choosers_dest_col_name="destination",
        )


def test_run_pnr_returns_no_choices_when_no_eligible_destinations(
    state, network_los, model_settings_base
):
    """Returns -1 for all choosers when no destination is park-and-ride eligible."""
    model_settings_base.LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST = "pnr_eligible"

    land_use = pd.DataFrame(
        {
            "pnr_spaces": [10, 20],
            "pnr_eligible": [False, False],
        },
        index=pd.Index([1, 2], name="zone_id"),
    )
    choosers = pd.DataFrame(
        {
            "destination": [1, 2],
            "home_zone_id": [1, 1],
        },
        index=pd.Index([200, 201], name="tour_id"),
    )

    choices = pnr_lot_choice.run_park_and_ride_lot_choice(
        state=state,
        choosers=choosers,
        land_use=land_use,
        network_los=network_los,
        model_settings=model_settings_base,
    )

    assert (choices == -1).all()
    assert choices.index.tolist() == [200, 201]


def test_run_pnr_handles_non_unique_index(state, network_los, model_settings_base):
    """Handles non-unique chooser indexes (aka for logsums) and restores the original index on output."""
    model_settings_base.LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST = "pnr_eligible"

    land_use = pd.DataFrame(
        {
            "pnr_spaces": [10, 20],
            "pnr_eligible": [True, True],
        },
        index=pd.Index([1, 2], name="zone_id"),
    )

    choosers = pd.DataFrame(
        {
            "destination": [1, 2, 2],
            "home_zone_id": [1, 1, 1],
            "in_period": [1, 1, 1],
        },
        index=pd.Index([10, 10, 11], name="tour_id"),
    )

    _write_model_files(
        state,
        "pnr_non_unique_spec.csv",
        """
Description,Expression,coefficient
choose_low_zone,@df.pnr_zone_id,coef_neg
""",
        "pnr_non_unique_coeffs.csv",
        """
coefficient_name,value,constrain
coef_neg,-100.0,F
""",
    )
    model_settings_base.SPEC = "pnr_non_unique_spec.csv"
    model_settings_base.COEFFICIENTS = "pnr_non_unique_coeffs.csv"

    state.get_rn_generator().begin_step("test_run_pnr_handles_non_unique_index")

    choices = pnr_lot_choice.run_park_and_ride_lot_choice(
        state=state,
        choosers=choosers,
        land_use=land_use,
        network_los=network_los,
        model_settings=model_settings_base,
    )
    state.get_rn_generator().end_step("test_run_pnr_handles_non_unique_index")

    assert choices.index.tolist() == [10, 10, 11]
    assert (choices == 1).all()


def test_run_pnr_calculates_drive_transit_and_capacity_utilities(
    state, network_los, model_settings_base
):
    """Evaluates forward and reverse drive/transit skims with a capacity penalty."""
    model_settings_base.compute_settings = ComputeSettings(drop_unused_columns=False)

    land_use = pd.DataFrame(
        {
            "pnr_spaces": [10, 10],
            "pnr_eligible": [True, True],
        },
        index=pd.Index([1, 2], name="zone_id"),
    )

    choosers = pd.DataFrame(
        {
            "destination": [5, 5, 4],
            "home_zone_id": [3, 4, 3],
            "start": [8, 8, 8],
            "end": [17, 17, 17],
        },
        index=pd.Index([100, 101, 102], name="tour_id"),
    )

    _write_model_files(
        state,
        "pnr_utility_spec.csv",
        """
Description,Expression,coefficient
drive_out,@olt_skims['DRIVE_TIME'],coef_drive
transit_out,@ldt_skims['TRANSIT_TIME'],coef_transit
transit_back,@dlt_skims['TRANSIT_TIME'],coef_transit
drive_back,@lot_skims['DRIVE_TIME'],coef_drive
capacity,@df.pnr_lot_full,coef_capacity
""",
        "pnr_utility_coeffs.csv",
        """
coefficient_name,value,constrain
coef_drive,-1.0,F
coef_transit,-1.0,F
coef_capacity,-3.0,F
""",
    )
    model_settings_base.SPEC = "pnr_utility_spec.csv"
    model_settings_base.COEFFICIENTS = "pnr_utility_coeffs.csv"

    class _CapacityStub:
        @staticmethod
        def flag_capacitated_pnr_zones(pnr_alts):
            # zone 2 receives a mild penalty that changes one edge case chooser.
            return np.where(pnr_alts.index == 2, 1, 0)

    choices = pnr_lot_choice.run_park_and_ride_lot_choice(
        state=state,
        choosers=choosers,
        land_use=land_use,
        network_los=network_los,
        model_settings=model_settings_base,
        pnr_capacity_cls=_CapacityStub(),
    )

    assert choices.to_dict() == {100: 1, 101: 2, 102: 1}


def test_capacity_scales_and_aggregates_choices(state):
    """Scales lot capacities by sample rate and aggregates occupancy by chosen lot."""
    land_use = pd.DataFrame(
        {"pnr_spaces": [2, 4, 0]},
        index=pd.Index([1, 2, 3], name="zone_id"),
    )
    households = pd.DataFrame(
        {"sample_rate": [0.5, 0.5]}, index=pd.Index([1, 2], name="household_id")
    )
    state.add_table("land_use", land_use)
    state.add_table("households", households)
    state.add_injectable("num_processes", 1)

    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=0.95,
        RESAMPLE_STRATEGY="latest",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    cap = pnr_capacity.ParkAndRideCapacity(state, settings)

    assert cap.scaled_pnr_capacity_df["pnr_capacity"].to_dict() == {1: 1, 2: 2, 3: 0}

    choices = pd.DataFrame(
        {
            "tour_mode": ["PNR", "DRIVE", "PNR", "PNR"],
            "pnr_zone_id": [1, 1, 2, 2],
            "start": [8, 9, 7, 10],
        },
        index=pd.Index([100, 101, 102, 103], name="tour_id"),
    )

    cap.set_choices(choices)

    assert cap.shared_pnr_occupancy_df["pnr_occupancy"].to_dict() == {1: 1, 2: 2, 3: 0}


def test_capacity_identifies_and_flags_capacitated_zones(state):
    """Identifies capacitated zones at tolerance and flags matching alternatives."""
    land_use = pd.DataFrame(
        {"pnr_spaces": [3, 2, 0]},
        index=pd.Index([1, 2, 3], name="zone_id"),
    )
    households = pd.DataFrame(
        {"sample_rate": [1.0]}, index=pd.Index([1], name="household_id")
    )
    state.add_table("land_use", land_use)
    state.add_table("households", households)
    state.add_injectable("num_processes", 1)

    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=0.95,
        RESAMPLE_STRATEGY="latest",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    cap = pnr_capacity.ParkAndRideCapacity(state, settings)
    cap.shared_pnr_occupancy_df["pnr_occupancy"] = [3, 1, 10]

    cap.determine_capacitated_pnr_zones(state)

    assert list(cap.capacitated_zones) == [1]

    pnr_alts = land_use.copy()
    flagged = cap.flag_capacitated_pnr_zones(pnr_alts)
    assert flagged.tolist() == [1, 0, 0]


def test_capacity_select_new_choosers_latest_strategy(state):
    """Resamples latest-arriving tours in over-capacity lots and updates occupancy."""
    land_use = pd.DataFrame(
        {"pnr_spaces": [2, 1]},
        index=pd.Index([1, 2], name="zone_id"),
    )
    households = pd.DataFrame(
        {"sample_rate": [1.0]}, index=pd.Index([1], name="household_id")
    )
    state.add_table("land_use", land_use)
    state.add_table("households", households)
    state.add_injectable("num_processes", 1)

    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=1.0,
        RESAMPLE_STRATEGY="latest",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    cap = pnr_capacity.ParkAndRideCapacity(state, settings)
    all_choosers = pd.DataFrame(
        {"destination": [10, 11, 12]},
        index=pd.Index([100, 101, 102], name="tour_id"),
    )

    choices = pd.DataFrame(
        {
            "tour_mode": ["PNR", "PNR", "PNR"],
            "pnr_zone_id": [1, 1, 1],
            "start": [8, 10, 9],
        },
        index=all_choosers.index,
    )

    cap.set_choices(choices)
    # zone 1 has capacity 2 and occupancy 3, so exactly one latest arrival should be resampled
    selected = cap.select_new_choosers(state, all_choosers)

    assert selected.index.tolist() == [101]
    assert cap.shared_pnr_occupancy_df["pnr_occupancy"].to_dict() == {1: 2, 2: 0}


def test_capacity_select_new_choosers_random_strategy(state, monkeypatch):
    """Randomly resamples over-capacity tours without sampling full lots."""
    land_use = pd.DataFrame(
        {"pnr_spaces": [2, 2]},
        index=pd.Index([1, 2], name="zone_id"),
    )
    households = pd.DataFrame(
        {"sample_rate": [1.0]}, index=pd.Index([1], name="household_id")
    )
    state.add_table("land_use", land_use)
    state.add_table("households", households)
    state.add_injectable("num_processes", 1)

    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=1.0,
        RESAMPLE_STRATEGY="random",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    cap = pnr_capacity.ParkAndRideCapacity(state, settings)
    all_choosers = pd.DataFrame(
        {"destination": [10, 11, 12, 13, 14]},
        index=pd.Index([100, 101, 102, 103, 104], name="tour_id"),
    )
    choices = pd.DataFrame(
        {
            "tour_mode": ["PNR"] * 5,
            "pnr_zone_id": [1, 1, 1, 2, 2],
            "start": [8, 9, 10, 8, 9],
        },
        index=all_choosers.index,
    )
    cap.set_choices(choices)

    def choose_one_over_capacity_tour(state, probs):
        expected = pd.DataFrame(
            {
                0: [0.5, 0.5, 0.5, 1.0, 1.0],
                1: [0.5, 0.5, 0.5, 0.0, 0.0],
            },
            index=all_choosers.index,
        )
        pd.testing.assert_frame_equal(probs, expected)
        return pd.Series([1, 0, 0, 0, 0], index=probs.index), None

    monkeypatch.setattr(
        pnr_capacity.logit, "make_choices", choose_one_over_capacity_tour
    )

    selected = cap.select_new_choosers(state, all_choosers)

    assert selected.index.tolist() == [100]
    assert cap.shared_pnr_occupancy_df["pnr_occupancy"].to_dict() == {1: 2, 2: 2}


def test_create_capacity_data_buffers(state):
    """Creates and initializes shared multiprocessing buffers for park-and-ride capacity."""
    persons = pd.DataFrame(index=pd.Index([1, 2, 3, 4], name="person_id"))
    state.add_table("persons", persons)

    buffers = pnr_capacity.create_park_and_ride_capacity_data_buffers(state)

    assert set(buffers.keys()) == {
        "shared_pnr_choice",
        "shared_pnr_choice_idx",
        "shared_pnr_choice_start",
        "pnr_mp_tally",
    }
    assert len(buffers["shared_pnr_choice"]) == 4
    assert len(buffers["shared_pnr_choice_idx"]) == 4
    assert len(buffers["shared_pnr_choice_start"]) == 4
    # buffers are initialized to 0
    assert list(buffers["shared_pnr_choice"][:]) == [0, 0, 0, 0]
    assert list(buffers["shared_pnr_choice_idx"][:]) == [0, 0, 0, 0]
    assert list(buffers["shared_pnr_choice_start"][:]) == [0, 0, 0, 0]
    assert list(buffers["pnr_mp_tally"][:]) == [0, 0]


def test_synchronize_choices_combines_and_resets_shared_buffers(state):
    """Synchronizes two worker choice sets and resets shared arrays for the next iteration."""
    state.add_injectable("num_processes", 2)
    state.add_injectable(
        "data_buffers", pnr_capacity.create_park_and_ride_capacity_data_buffers(state)
    )

    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=0.95,
        RESAMPLE_STRATEGY="latest",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    cap_a = pnr_capacity.ParkAndRideCapacity(state, settings)
    cap_b = pnr_capacity.ParkAndRideCapacity(state, settings)

    choices_a = pd.DataFrame(
        {
            "tour_mode": ["PNR", "PNR"],
            "pnr_zone_id": [1, 2],
            "start": [8, 9],
        },
        index=pd.Index([100, 101], name="tour_id"),
    )
    choices_b = pd.DataFrame(
        {
            "tour_mode": ["PNR", "PNR"],
            "pnr_zone_id": [2, 1],
            "start": [10, 11],
        },
        index=pd.Index([102, 103], name="tour_id"),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(cap_a.synchronize_choices, choices_a)
        future_b = pool.submit(cap_b.synchronize_choices, choices_b)
        synced_a = future_a.result()
        synced_b = future_b.result()

    expected = pd.concat(
        [
            choices_a[["pnr_zone_id", "start"]],
            choices_b[["pnr_zone_id", "start"]],
        ]
    ).sort_index()

    expected_content = expected.sort_values(["pnr_zone_id", "start"]).reset_index(
        drop=True
    )
    synced_a_content = (
        synced_a[["pnr_zone_id", "start"]]
        .sort_values(["pnr_zone_id", "start"])
        .reset_index(drop=True)
    )
    synced_b_content = (
        synced_b[["pnr_zone_id", "start"]]
        .sort_values(["pnr_zone_id", "start"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(synced_a_content, expected_content)
    pd.testing.assert_frame_equal(synced_b_content, expected_content)

    data_buffers = state.get_injectable("data_buffers")
    # synchronize_choices barrier 2 clears shared arrays so each new iteration starts empty.
    assert list(data_buffers["shared_pnr_choice"][:]) == [0, 0, 0, 0]
    assert list(data_buffers["shared_pnr_choice_idx"][:]) == [0, 0, 0, 0]
    assert list(data_buffers["shared_pnr_choice_start"][:]) == [0, 0, 0, 0]
    assert list(data_buffers["pnr_mp_tally"][:]) == [0, 2]


def test_synchronize_choices_barrier_with_one_empty_worker(state):
    """Allows an empty worker to satisfy both barriers and receive synced non-empty choices."""
    state.add_injectable("num_processes", 2)
    state.add_injectable(
        "data_buffers", pnr_capacity.create_park_and_ride_capacity_data_buffers(state)
    )

    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=0.95,
        RESAMPLE_STRATEGY="latest",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    cap_empty = pnr_capacity.ParkAndRideCapacity(state, settings)
    cap_full = pnr_capacity.ParkAndRideCapacity(state, settings)

    empty_choices = pd.DataFrame(
        columns=["pnr_zone_id", "start"],
        index=pd.Index([], name="tour_id"),
    )
    full_choices = pd.DataFrame(
        {
            "tour_mode": ["PNR", "PNR"],
            "pnr_zone_id": [2, 1],
            "start": [7, 12],
        },
        index=pd.Index([200, 201], name="tour_id"),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_empty = pool.submit(cap_empty.synchronize_choices, empty_choices)
        future_full = pool.submit(cap_full.synchronize_choices, full_choices)
        synced_empty = future_empty.result()
        synced_full = future_full.result()

    expected_content = (
        full_choices[["pnr_zone_id", "start"]]
        .sort_values(["pnr_zone_id", "start"])
        .reset_index(drop=True)
    )
    synced_empty_content = (
        synced_empty[["pnr_zone_id", "start"]]
        .sort_values(["pnr_zone_id", "start"])
        .reset_index(drop=True)
    )
    synced_full_content = (
        synced_full[["pnr_zone_id", "start"]]
        .sort_values(["pnr_zone_id", "start"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(synced_empty_content, expected_content)
    pd.testing.assert_frame_equal(synced_full_content, expected_content)

    data_buffers = state.get_injectable("data_buffers")
    # Even with one empty worker, barrier 2 resets shared arrays after synchronization.
    assert list(data_buffers["shared_pnr_choice"][:]) == [0, 0, 0, 0]
    assert list(data_buffers["shared_pnr_choice_idx"][:]) == [0, 0, 0, 0]
    assert list(data_buffers["shared_pnr_choice_start"][:]) == [0, 0, 0, 0]
    assert list(data_buffers["pnr_mp_tally"][:]) == [0, 2]


def test_set_choices_single_and_multiprocess_aggregate_match(state):
    """Produces identical occupancy aggregates for single- and two-process paths."""
    settings = types.SimpleNamespace(
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        PARK_AND_RIDE_MODES=["PNR"],
        ACCEPTED_TOLERANCE=0.95,
        RESAMPLE_STRATEGY="latest",
        TRACE_PNR_CAPACITIES_PER_ITERATION=False,
    )

    all_choices = pd.DataFrame(
        {
            "tour_mode": ["PNR", "PNR", "PNR", "DRIVE"],
            "pnr_zone_id": [1, 2, 2, 1],
            "start": [8, 9, 10, 11],
        },
        index=pd.Index([100, 101, 102, 103], name="tour_id"),
    )

    state.add_injectable("num_processes", 1)
    cap_single = pnr_capacity.ParkAndRideCapacity(state, settings)
    cap_single.set_choices(all_choices)
    expected_occ = cap_single.shared_pnr_occupancy_df["pnr_occupancy"].copy()

    state.add_injectable(
        "data_buffers", pnr_capacity.create_park_and_ride_capacity_data_buffers(state)
    )
    state.add_injectable("num_processes", 2)
    cap_a = pnr_capacity.ParkAndRideCapacity(state, settings)
    cap_b = pnr_capacity.ParkAndRideCapacity(state, settings)

    choices_a = all_choices.iloc[[0, 1]].copy()
    choices_b = all_choices.iloc[[2, 3]].copy()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(cap_a.set_choices, choices_a)
        future_b = pool.submit(cap_b.set_choices, choices_b)
        future_a.result()
        future_b.result()

    pd.testing.assert_series_equal(
        cap_a.shared_pnr_occupancy_df["pnr_occupancy"], expected_occ
    )
    pd.testing.assert_series_equal(
        cap_b.shared_pnr_occupancy_df["pnr_occupancy"], expected_occ
    )

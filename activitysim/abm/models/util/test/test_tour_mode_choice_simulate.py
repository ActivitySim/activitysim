import numpy as np
import openmatrix as omx
import pandas as pd
import pytest

from activitysim.abm.models import atwork_subtour_mode_choice as atwork_tmc
from activitysim.abm.models import tour_mode_choice as tmc
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


def _write_text(path, text):
    path.write_text(text.strip())


def _tour_mode_settings(spec_file, run_atwork_pnr_lot_choice=False):
    return _Settings(
        SPEC=spec_file,
        COEFFICIENTS="test_tour_mode_choice_coefficients.csv",
        COEFFICIENT_TEMPLATE="test_tour_mode_choice_coefficients_template.csv",
        LOGIT_TYPE="NL",
        NESTS={
            "name": "root",
            "coefficient": "coef_nest_root",
            "alternatives": [
                {
                    "name": "DRIVE_NEST",
                    "coefficient": "coef_nest_drive",
                    "alternatives": ["DRIVEALONE"],
                },
                {
                    "name": "HOV2_NEST",
                    "coefficient": "coef_nest_hov2",
                    "alternatives": ["HOV2"],
                },
                "WALK",
                "TRANSIT",
            ],
        },
        MODE_CHOICE_LOGSUM_COLUMN_NAME="mode_choice_logsum",
        CONSTANTS={},
        use_TVPB_constants=False,
        COMPUTE_TRIP_MODE_CHOICE_LOGSUMS=False,
        FORCE_ESCORTEE_CHAUFFEUR_MODE_MATCH=False,
        tvpb_mode_path_types=None,
        run_atwork_pnr_lot_choice=run_atwork_pnr_lot_choice,
        explicit_chunk=0,
        compute_settings=None,
    )


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

    _write_text(
        config_dir / "test_tour_mode_choice_coefficients_template.csv",
        """
coefficient_name,work,nonmand,atwork
coef_nest_root,coef_nest_root,coef_nest_root,coef_nest_root
coef_nest_drive,coef_nest_drive,coef_nest_drive,coef_nest_drive
coef_nest_hov2,coef_nest_hov2,coef_nest_hov2,coef_nest_hov2
coef_da_time,coef_da_time_work,coef_da_time_nonmand,coef_da_time_atwork
coef_hov2_time,coef_hov2_time_work,coef_hov2_time_nonmand,coef_hov2_time_atwork
coef_walk_time,coef_walk_time_work,coef_walk_time_nonmand,coef_walk_time_atwork
coef_transit_time,coef_transit_time_work,coef_transit_time_nonmand,coef_transit_time_atwork
""",
    )

    _write_text(
        config_dir / "test_tour_mode_choice_coefficients.csv",
        """
coefficient_name,value,constrain
coef_nest_root,1.0,T
coef_nest_drive,0.8,T
coef_nest_hov2,0.8,T
coef_da_time_work,-1.0,F
coef_da_time_nonmand,-1.0,F
coef_hov2_time_work,-2.0,F
coef_hov2_time_nonmand,-2.0,F
coef_walk_time_work,-0.5,F
coef_walk_time_nonmand,-0.5,F
coef_transit_time_work,-1.0,F
coef_transit_time_nonmand,-1.0,F
coef_da_time_atwork,-1.0,F
coef_hov2_time_atwork,-2.0,F
coef_walk_time_atwork,-0.5,F
coef_transit_time_atwork,-1.0,F
""",
    )

    _write_text(
        config_dir / "test_tour_mode_choice_no_pnr.csv",
        """
Label,Description,Expression,DRIVEALONE,HOV2,WALK,TRANSIT
util_da,Drive time,@od_skims['DRIVE_TIME'],coef_da_time,,,
util_hov2,Drive time HOV2,@od_skims['DRIVE_TIME'],,coef_hov2_time,,
util_walk,Walk time,@od_skims['WALK_TIME'],,,coef_walk_time,
util_transit,Transit time,@od_skims['TRANSIT_TIME'],,,,coef_transit_time
""",
    )

    _write_text(
        config_dir / "test_tour_mode_choice_with_pnr.csv",
        """
Label,Description,Expression,DRIVEALONE,HOV2,WALK,TRANSIT
util_da,Drive time,@od_skims['DRIVE_TIME'],coef_da_time,,,
util_hov2,Drive time HOV2,@od_skims['DRIVE_TIME'],,coef_hov2_time,,
util_walk,Walk time,@od_skims['WALK_TIME'],,,coef_walk_time,
util_transit_pnr,PNR drive and transit time,@olt_skims['DRIVE_TIME'] + ldt_skims['TRANSIT_TIME'],,,,coef_transit_time
""",
    )

    _write_text(
        config_dir / "park_and_ride_lot_choice.yaml",
        """
SPEC: test_pnr_lot_choice_spec.csv
COEFFICIENTS: test_pnr_lot_choice_coefficients.csv
CONSTANTS: {}
LANDUSE_PNR_SPACES_COLUMN: pnr_spaces
LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST: pnr_eligible
TRANSIT_SKIMS_FOR_ELIGIBILITY:
    - TRANSIT_TIME
CHOOSER_FILTER_EXPR: null
explicit_chunk: 0
compute_settings:
    drop_unused_columns: false
ITERATE_WITH_TOUR_MODE_CHOICE: true
MAX_ITERATIONS: 3
ACCEPTED_TOLERANCE: 1.0
RESAMPLE_STRATEGY: latest
PARK_AND_RIDE_MODES:
  - TRANSIT
TRACE_PNR_CAPACITIES_PER_ITERATION: false
""",
    )

    _write_text(
        config_dir / "test_pnr_lot_choice_spec.csv",
        """
Description,Expression,coefficient
drive_out,@olt_skims['DRIVE_TIME'],coef_drive
transit_out,@ldt_skims['TRANSIT_TIME'],coef_transit
transit_back,@dlt_skims['TRANSIT_TIME'],coef_transit
drive_back,@lot_skims['DRIVE_TIME'],coef_drive
capacity,@df.pnr_lot_full,coef_capacity
""",
    )

    _write_text(
        config_dir / "test_pnr_lot_choice_coefficients.csv",
        """
coefficient_name,value,constrain
coef_drive,-1.0,F
coef_transit,-1.0,F
coef_capacity,-999.0,F
""",
    )

    # OD skims for no-PNR mode choice utilities.
    drive_time = np.full((6, 6), 30.0, dtype=np.float32)
    walk_time = np.full((6, 6), 60.0, dtype=np.float32)
    transit_time = np.full((6, 6), 40.0, dtype=np.float32)

    # No-PNR utility setup across two destinations so outcomes include multiple modes.
    # home 3 -> dest 5 : DRIVEALONE is best
    drive_time[2, 4] = 5.0
    walk_time[2, 4] = 30.0
    transit_time[2, 4] = 20.0
    # home 4 -> dest 5 : WALK is best
    drive_time[3, 4] = 20.0
    walk_time[3, 4] = 4.0
    transit_time[3, 4] = 18.0
    # home 3 -> dest 6 : TRANSIT is best
    drive_time[2, 5] = 25.0
    walk_time[2, 5] = 30.0
    transit_time[2, 5] = 3.0
    # home 4 -> dest 6 : DRIVEALONE is best
    drive_time[3, 5] = 8.0
    walk_time[3, 5] = 25.0
    transit_time[3, 5] = 16.0

    # 3D skims for PNR utility pieces (origin->lot and lot->destination).
    drive_time_am = np.full((6, 6), 30.0, dtype=np.float32)
    transit_time_am = np.full((6, 6), 30.0, dtype=np.float32)

    # Origins 3/4 to lots 1/2.
    drive_time_am[2, 0] = 6.0
    drive_time_am[2, 1] = 1.0
    drive_time_am[3, 0] = 30.0
    drive_time_am[3, 1] = 30.0

    # Lots 1/2 to destination 5 and 6.
    transit_time_am[0, 4] = 8.0
    transit_time_am[1, 4] = 1.0
    transit_time_am[0, 5] = 50.0
    transit_time_am[1, 5] = 1.0

    # Reverse directions for PNR lot choice utility rows.
    transit_time_am[4, 0] = 8.0
    transit_time_am[4, 1] = 1.0
    transit_time_am[5, 0] = 50.0
    transit_time_am[5, 1] = 1.0
    drive_time_am[0, 2] = 6.0
    drive_time_am[1, 2] = 1.0
    drive_time_am[0, 3] = 30.0
    drive_time_am[1, 3] = 30.0

    with omx.open_file(data_dir / "skims.omx", "w") as skims:
        skims["DRIVE_TIME"] = drive_time
        skims["WALK_TIME"] = walk_time
        skims["TRANSIT_TIME"] = transit_time
        skims["DRIVE_TIME__AM"] = drive_time_am
        skims["TRANSIT_TIME__AM"] = transit_time_am
        skims.create_mapping("zone_number", [1, 2, 3, 4, 5, 6])

    s = workflow.State.make_default(root)
    s.add_table("persons", pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id")))
    s.add_table(
        "households",
        pd.DataFrame({"sample_rate": [1.0]}, index=pd.Index([1], name="household_id")),
    )
    s.add_table(
        "land_use",
        pd.DataFrame(
            {
                "pnr_spaces": [2, 1, 0, 0],
                "pnr_eligible": [True, True, True, True],
            },
            index=pd.Index([1, 2, 5, 6], name="zone_id"),
        ),
    )
    return s


@pytest.fixture
def network_los(state):
    nl = los.Network_LOS(state)
    nl.skim_dicts["taz"] = nl.create_skim_dict("taz")
    return nl


def _make_tours(include_pnr=False):
    tours = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 1, 2, 3, 1, 2],
            "tour_category": [
                "mandatory",
                "non_mandatory",
                "mandatory",
                "non_mandatory",
                "mandatory",
                "non_mandatory",
                "mandatory",
                "non_mandatory",
            ],
            "tour_type": [
                "work",
                "nonmand",
                "work",
                "nonmand",
                "work",
                "nonmand",
                "work",
                "nonmand",
            ],
            "start": [8, 9, 11, 10, 8, 9, 11, 10],
            "end": [17, 18, 19, 18, 17, 18, 19, 18],
            "destination": [5, 5, 6, 6, 5, 5, 6, 6],
            "home_zone_id": [3, 4, 3, 4, 3, 4, 3, 4],
        },
        index=pd.Index([100, 101, 102, 103, 104, 105, 106, 107], name="tour_id"),
    )
    if include_pnr:
        tours["pnr_zone_id"] = [1, 2, 2, 1, 1, 2, 2, 1]
    return tours


def _make_persons_merged():
    return pd.DataFrame(
        {
            "is_university": [False, False, False],
            "workplace_zone_id": [3, 4, 3],
        },
        index=pd.Index([1, 2, 3], name="person_id"),
    )


def test_tour_mode_choice_simulate_without_pnr_iteration(state, network_los):
    """Runs actual tour_mode_choice_simulate once when no PNR lot iteration is configured."""
    tours = _make_tours(include_pnr=False)
    persons_merged = _make_persons_merged()
    model_settings = _tour_mode_settings("test_tour_mode_choice_no_pnr.csv")

    state.get_rn_generator().begin_step("tour_mode_choice_simulate")

    tmc.tour_mode_choice_simulate(
        state=state,
        tours=tours,
        persons_merged=persons_merged,
        network_los=network_los,
        model_settings=model_settings,
    )
    state.get_rn_generator().end_step("tour_mode_choice_simulate")

    result = state.get_dataframe("tours")

    no_iter_modes = result["tour_mode"].astype(str).to_dict()
    assert no_iter_modes == {
        100: "DRIVEALONE",
        101: "WALK",
        102: "TRANSIT",
        103: "DRIVEALONE",
        104: "DRIVEALONE",
        105: "WALK",
        106: "TRANSIT",
        107: "DRIVEALONE",
    }
    assert sum(m == "TRANSIT" for m in no_iter_modes.values()) == 2
    assert result["mode_choice_logsum"].notna().all()
    assert set(result["tour_type"].tolist()) == {"work", "nonmand"}


def test_tour_mode_choice_simulate_with_pnr_iteration(state, network_los, monkeypatch):
    """Runs actual iterative tour mode choice with real PNR lot choice and capacity updates."""
    tours = _make_tours(include_pnr=True)
    persons_merged = _make_persons_merged()
    model_settings = _tour_mode_settings("test_tour_mode_choice_with_pnr.csv")

    run_calls = {"count": 0}
    pnr_calls = {"count": 0}

    original_run_tmc = tmc.run_tour_mode_choice_simulate
    original_run_pnr = tmc.run_park_and_ride_lot_choice

    def counting_run_tmc(*args, **kwargs):
        run_calls["count"] += 1
        return original_run_tmc(*args, **kwargs)

    def counting_run_pnr(*args, **kwargs):
        pnr_calls["count"] += 1
        model_settings = kwargs.get("model_settings")
        if (
            model_settings is not None
            and getattr(model_settings, "preprocessor", None) is None
        ):
            model_settings.preprocessor = _EmptyPreprocessor()
        return original_run_pnr(*args, **kwargs)

    # Wrap real helpers so we can count invocations while still executing actual logic.
    monkeypatch.setattr(tmc, "run_tour_mode_choice_simulate", counting_run_tmc)
    # Wrap real PNR lot choice similarly; this is instrumentation, not a fake model.
    monkeypatch.setattr(tmc, "run_park_and_ride_lot_choice", counting_run_pnr)

    state.get_rn_generator().begin_step("tour_mode_choice_simulate")

    tmc.tour_mode_choice_simulate(
        state=state,
        tours=tours,
        persons_merged=persons_merged,
        network_los=network_los,
        model_settings=model_settings,
    )
    state.get_rn_generator().end_step("tour_mode_choice_simulate")

    result = state.get_dataframe("tours")

    # make sure iteration is occuring
    assert run_calls["count"] >= 3
    assert pnr_calls["count"] >= 1
    iter_modes = result["tour_mode"].astype(str).to_dict()

    # notice how the TRANSIT counts here are fewer than the non-iterative test
    assert iter_modes == {
        100: "DRIVEALONE",
        101: "WALK",
        102: "WALK",
        103: "DRIVEALONE",
        104: "DRIVEALONE",
        105: "WALK",
        106: "TRANSIT",
        107: "DRIVEALONE",
    }


def test_atwork_subtour_mode_choice_with_pnr(state, network_los):
    """Chooses PNR lots from the workplace for at-work subtours."""
    tours = pd.DataFrame(
        {
            "person_id": [1, 2],
            "tour_category": ["atwork", "atwork"],
            "tour_type": ["eat", "business"],
            "start": [12, 13],
            "end": [13, 14],
            "destination": [5, 5],
        },
        index=pd.Index([200, 201], name="tour_id"),
    )
    persons_merged = _make_persons_merged()
    model_settings = _tour_mode_settings(
        "test_tour_mode_choice_with_pnr.csv", run_atwork_pnr_lot_choice=True
    )

    state.get_rn_generator().begin_step("atwork_subtour_mode_choice")
    atwork_tmc.atwork_subtour_mode_choice(
        state=state,
        tours=tours,
        persons_merged=persons_merged,
        network_los=network_los,
        model_settings=model_settings,
    )
    state.get_rn_generator().end_step("atwork_subtour_mode_choice")

    result = state.get_dataframe("tours")
    assert result["pnr_zone_id"].to_dict() == {200: 2, 201: 2}
    assert result["tour_mode"].astype(str).to_dict() == {
        200: "TRANSIT",
        201: "WALK",
    }
    assert result["mode_choice_logsum"].notna().all()

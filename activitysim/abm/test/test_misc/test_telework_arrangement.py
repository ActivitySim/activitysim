import numpy as np
import pandas as pd
import pytest
import openmatrix as omx

from activitysim.abm.models import telework_arrangement as model
from activitysim.core import workflow, los


class DummyFileSystem:
    def read_model_spec(self, file_name):
        return pd.DataFrame({"alt0": [1.0], "alt1": [0.0]}, index=["1"])

    def read_model_coefficients(self, model_settings):
        return pd.DataFrame({"value": [1.0]}, index=["coef_a"])


class DummyState:
    def __init__(self):
        self.filesystem = DummyFileSystem()
        self.settings = type("Settings", (), {"trace_hh_id": False})()
        self.added_tables = {}

    def add_table(self, name, table):
        self.added_tables[name] = table.copy()


def _settings(filter_col="is_worker", true_alt=0):
    return type(
        "ModelSettings",
        (),
        {
            "CHOOSER_FILTER_COLUMN_NAME": filter_col,
            "HAS_IN_HOME_WORK_ACTIVITY_ALT": true_alt,
            "SPEC": "telework_arrangement.csv",
            "compute_settings": None,
        },
    )()


def test_telework_arrangement_monkeypatch(monkeypatch):
    state = DummyState()

    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    persons_merged = pd.DataFrame(
        {
            "is_worker": [True, False, True],
            "some_var": [10, 20, 30],
        },
        index=persons.index,
    )

    called = {"annotate": False, "annotate_tables": False, "choosers_index": None}

    monkeypatch.setattr(
        model.estimation.manager, "begin_estimation", lambda *a, **k: None
    )
    monkeypatch.setattr(model.config, "get_model_constants", lambda *_: {"CONST": 1})
    monkeypatch.setattr(model.config, "get_logit_model_settings", lambda *_: None)

    def fake_annotate_preprocessors(*args, **kwargs):
        called["annotate"] = True
        assert kwargs["locals_dict"] == {"CONST": 1}

    def fake_annotate_tables(*args, **kwargs):
        called["annotate_tables"] = True
        assert kwargs["locals_dict"] == {"CONST": 1}

    def fake_eval_coefficients(state, spec, coefficients, estimator):
        return spec

    def fake_simple_simulate(*args, **kwargs):
        choosers = kwargs["choosers"]
        called["choosers_index"] = choosers.index.tolist()
        # alt 0 => True, alt 1 => False
        return pd.Series([0, 1], index=choosers.index)

    monkeypatch.setattr(
        model.expressions, "annotate_preprocessors", fake_annotate_preprocessors
    )
    monkeypatch.setattr(model.expressions, "annotate_tables", fake_annotate_tables)
    monkeypatch.setattr(model.simulate, "eval_coefficients", fake_eval_coefficients)
    monkeypatch.setattr(model.simulate, "simple_simulate", fake_simple_simulate)
    monkeypatch.setattr(model.tracing, "print_summary", lambda *a, **k: None)

    model.telework_arrangement(
        state=state,
        persons_merged=persons_merged,
        persons=persons.copy(),
        model_settings=_settings(filter_col="is_worker", true_alt=0),
    )

    assert called["annotate"]
    assert called["annotate_tables"]
    assert called["choosers_index"] == [1, 3]

    out = state.added_tables["persons"]
    assert out["has_in_home_work_activity"].dtype == bool
    assert out["has_in_home_work_activity"].to_dict() == {
        1: True,
        2: False,
        3: False,
    }


@pytest.fixture(scope="session")
def example_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("example")
    config_dir = root / "configs"
    config_dir.mkdir()

    data_dir = root / "data"
    data_dir.mkdir()

    return root


@pytest.fixture(scope="module")
def model_settings(example_root, state):

    model_settings = model.TeleworkArrangementSettings.read_settings_file(
        state.filesystem, "telework_arrangement.yaml"
    )

    return model_settings


@pytest.fixture(scope="module")
def state(example_root, coeffs_configs_csv, configs_csv) -> workflow.State:

    settings = """
        input_table_list:
            - tablename: households
            - tablename: persons
            - tablename: land_use
        """

    network_los_yaml = """
                zone_system: 2
                taz_skims: skims*.omx
                skim_time_periods:
                    time_window: 1440
                    period_minutes: 30
                    periods: [12]
                    labels: &skim_time_period_labels ['AM']
                    """

    skim_matrix = np.array(
        [
            [0.42, 0.89, 4.33, 10.31, 9.98],
            [0.89, 0.39, 3.76, 10.05, 9.72],
            [4.19, 3.61, 0.85, 10.02, 9.69],
            [10.57, 9.99, 9.81, 0.16, 0.37],
            [10.19, 9.61, 9.43, 0.37, 0.16],
        ]
    )

    telework_arrangement_settings = """
        CHOOSER_FILTER_COLUMN_NAME: is_worker
        HAS_IN_HOME_WORK_ACTIVITY_ALT: 0
        SPEC: telework_arrangement.csv
        COEFFICIENTS: telework_arrangement_coeffs.csv
        LOGIT_TYPE: MNL
        """

    telework_arrangement_yaml = example_root / "configs" / "telework_arrangement.yaml"
    telework_arrangement_yaml.write_text(telework_arrangement_settings)

    settings_file = example_root / "configs" / "settings.yaml"
    settings_file.write_text(settings)

    yaml_file = example_root / "configs" / "network_los.yaml"
    yaml_file.write_text(network_los_yaml)

    telework_arrangement_coeffs = (
        example_root / "configs" / "telework_arrangement_coeffs.csv"
    )
    telework_arrangement_coeffs.write_text(coeffs_configs_csv)

    telework_arrangement = example_root / "configs" / "telework_arrangement.csv"
    telework_arrangement.write_text(configs_csv)

    skims = omx.open_file(example_root / "data" / "skims.omx", "w")
    skims["DIST"] = skim_matrix
    taz_equivs = [2103, 2104, 2115, 2142, 2144]
    skims.create_mapping("zone_number", taz_equivs)
    skims.close()

    state = workflow.State.make_default(example_root)

    return state


@pytest.fixture(scope="module")
def persons() -> pd.DataFrame:
    persons = pd.DataFrame(
        {
            "person_id": [
                2664688,
                2664689,
                2668012,
                2668013,
                2701577,
                2701578,
                2860810,
                2860811,
                2865544,
                2865545,
                2865546,
            ],
            "household_id": [
                1080351,
                1080351,
                1081684,
                1081684,
                1094369,
                1094369,
                1156249,
                1156249,
                1158612,
                1158612,
                1158612,
            ],
            "member_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3],
            "sex": [1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2],
            "maz_seqid": [
                22660.0,
                22660.0,
                22670.0,
                22670.0,
                22734.0,
                22734.0,
                22803.0,
                22803.0,
                22799.0,
                22799.0,
                22799.0,
            ],
            "zone_id": [
                2103.0,
                2103.0,
                2104.0,
                2104.0,
                2115.0,
                2115.0,
                2144.0,
                2144.0,
                2142.0,
                2142.0,
                2142.0,
            ],
            "is_worker": [
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
            "home_zone_id": [
                22660.0,
                22660.0,
                22670.0,
                22670.0,
                22734.0,
                22734.0,
                22803.0,
                22803.0,
                22799.0,
                22799.0,
                22799.0,
            ],
            "workplace_zone_id": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        }
    )
    persons = persons.set_index("person_id")

    return persons


@pytest.fixture(scope="module")
def households() -> pd.DataFrame:
    households = pd.DataFrame(
        {
            "household_id": [1156249, 1080351, 1094369, 1081684, 1158612],
            "adjinc": [1010145, 1054606, 1073449, 1031452, 1080470],
            "hht": [7.0, 1.0, 1.0, 1.0, 2.0],
            "maz": [22803.0, 22660.0, 22734.0, 22670.0, 22799.0],
            "taz": [2144.0, 2103.0, 2115.0, 2104.0, 2142.0],
            "auto_ownership": [2, 2, 2, 2, 2],
        }
    )
    households = households.set_index("household_id")

    return households


@pytest.fixture(scope="module")
def land_use() -> pd.DataFrame:
    land_use = pd.DataFrame(
        {
            "MAZ": [22660, 22670, 22734, 22799, 22803],
            "TAZ": [2103, 2104, 2115, 2142, 2144],
        }
    )

    return land_use


@pytest.fixture(scope="module")
def configs_csv():
    csv_content = """Label,Description,Expression,has_in_home_work_activity,no_in_home_work_activity
util_acs,alternative specific constant,1,,coef_acs_no_in_home_work
util_female,female,sex==2,coef_female_has_in_home_work,
"""
    return csv_content


@pytest.fixture(scope="module")
def coeffs_configs_csv():
    csv_content = """coefficient_name,value,constrain
coef_acs_no_in_home_work,0.1,F
coef_female_has_in_home_work,0.5,F
"""
    return csv_content


@pytest.fixture(scope="module")
def network_los(state, persons, households, land_use) -> los.Network_LOS:

    land_use["zone_id"] = land_use["MAZ"]
    land_use.set_index("zone_id", inplace=True)
    households["home_zone_id"] = households["maz"]

    state.add_table("persons", persons)
    state.add_table("households", households)
    state.add_table("land_use", land_use)

    persons_merged = pd.merge(
        persons.reset_index(), households, on="household_id", how="left"
    )
    persons_merged = pd.merge(
        persons_merged, land_use.rename(columns={"TAZ": "taz"}), on="taz", how="left"
    )

    persons_merged["home_zone_id"] = persons_merged["MAZ"]
    persons_merged["TAZ"] = persons_merged["taz"]
    persons_merged.set_index("person_id", inplace=True)

    state.add_table("persons_merged", persons_merged)

    network_los = los.Network_LOS(state)

    network_los.maz_taz_df = land_use[["MAZ", "TAZ"]]

    network_los.skim_dicts["taz"] = network_los.create_skim_dict("taz")
    network_los.skim_dicts["maz"] = network_los.create_skim_dict("maz")

    return network_los


def test_telework_arrangement_real(state, model_settings, network_los):

    persons_merged = state.get_dataframe("persons_merged").copy()

    model.telework_arrangement(
        state=state,
        persons_merged=persons_merged,
        persons=state.get_dataframe("persons").copy(),
        model_settings=model_settings,
    )

    out = state.get_dataframe("persons")["has_in_home_work_activity"]

    assert out.dtype == bool
    assert out.to_dict() == {
        2664688: False,
        2664689: True,
        2668012: False,
        2668013: False,
        2701577: False,
        2701578: True,
        2860810: True,
        2860811: False,
        2865544: True,
        2865545: False,
        2865546: False,
    }

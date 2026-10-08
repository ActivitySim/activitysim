from pathlib import Path

import pandas as pd
import pytest

from activitysim.abm.models import telework_duration as model
from activitysim.abm.models.settings_checker import try_load_and_check_spec_coefs
from activitysim.core import workflow


class DummyFileSystem:
    def __init__(self, probs_path: Path):
        self.probs_path = probs_path

    def _resolve_path(self, file_name):
        candidate = Path(file_name)
        if candidate.exists():
            return candidate
        tmp_candidate = Path("/tmp") / candidate.name
        if tmp_candidate.exists():
            return tmp_candidate
        return candidate

    def read_model_alts(self, state, file_name, set_index=None):
        return pd.DataFrame(
            {
                "alt": ["short", "long"],
                "duration_hours": [2.0, 4.0],
            }
        )

    def read_model_spec(self, file_name, *args, **kwargs):
        path = self._resolve_path(file_name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, comment="#")

    def read_model_coefficients(self, file_name, *args, **kwargs):
        path = self._resolve_path(file_name)
        if not path.exists():
            return pd.DataFrame(columns=["coefficient_name", "value"])
        df = pd.read_csv(path, comment="#")
        if "coefficient_name" in df.columns:
            df = df.set_index("coefficient_name")
        return df

    def get_config_file_path(self, file_name):
        assert file_name == "telework_duration_probs.csv"
        return self.probs_path


class DummyState:
    def __init__(self, probs_path: Path):
        self.filesystem = DummyFileSystem(probs_path)
        self.settings = type("Settings", (), {"trace_hh_id": False})()
        self.added_tables = {}

    def add_table(self, name, table):
        self.added_tables[name] = table.copy()


def _settings():
    return type(
        "ModelSettings",
        (),
        {
            "CHOOSER_FILTER_COLUMN_NAME": "has_in_home_work_activity",
            "DURATION_CATEGORY_COLUMN_NAME": "telework_duration_category",
            "DURATION_HOURS_COLUMN_NAME": "telework_duration_hours",
            "ALTS": "telework_duration_alts.csv",
            "ALT_NAME_COLUMN": "alt",
            "ALT_DURATION_COLUMN": "duration_hours",
            "PROBS_SPEC": "telework_duration_probs.csv",
            "PROBS_JOIN_COLS": None,
            "CHOICE_MODEL": "PROBABILISTIC",
            "compute_settings": None,
        },
    )()


def test_telework_duration_probabilistic_skips_mnl_spec_validation(monkeypatch):
    state = DummyState(Path("/tmp/telework_duration_probs.csv"))

    settings = model.TeleworkDurationSettings(
        CHOICE_MODEL="PROBABILISTIC",
        PROBS_SPEC="telework_duration_probs.csv",
    )

    errors = try_load_and_check_spec_coefs(
        model_name="telework_duration",
        model_settings=settings,
        state=state,
    )

    assert errors == []


def test_telework_duration_probabilistic_maps_choice_to_duration_monkeypatch(
    tmp_path, monkeypatch
):
    probs_path = tmp_path / "telework_duration_probs.csv"
    probs_path.write_text("short,long\n0.2,0.8\n")

    state = DummyState(probs_path)
    persons = pd.DataFrame(index=pd.Index([1, 2], name="person_id"))
    persons_merged = pd.DataFrame(
        {
            "has_in_home_work_activity": [True, False],
        },
        index=persons.index,
    )

    called = {"choosers_index": None}

    monkeypatch.setattr(
        model.estimation.manager, "begin_estimation", lambda *a, **k: None
    )
    monkeypatch.setattr(model.config, "get_model_constants", lambda *_: {})
    monkeypatch.setattr(
        model.expressions, "annotate_preprocessors", lambda *a, **k: None
    )
    monkeypatch.setattr(model.expressions, "annotate_tables", lambda *a, **k: None)
    monkeypatch.setattr(model.tracing, "print_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        model.simulate,
        "read_model_alts",
        lambda *a, **k: pd.DataFrame(
            {"alt": ["short", "long"], "duration_hours": [2.0, 4.0]}
        ),
    )

    def fake_make_choices(state, chooser_probs, trace_label, trace_choosers):
        called["choosers_index"] = trace_choosers.index.tolist()
        return pd.Series([1], index=trace_choosers.index), None

    monkeypatch.setattr(model.logit, "make_choices", fake_make_choices)

    model.telework_duration(
        state=state,
        persons_merged=persons_merged,
        persons=persons,
        model_settings=_settings(),
    )

    assert called["choosers_index"] == [1]

    out = state.added_tables["persons"]
    assert out["telework_duration_category"].astype(str).to_dict() == {
        1: "long",
        2: "",
    }
    assert out["telework_duration_hours"].to_dict() == {
        1: 4.0,
        2: 0.0,
    }


@pytest.fixture(scope="module")
def real_example_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("telework_duration_real")
    config_dir = root / "configs"
    config_dir.mkdir()
    (root / "data").mkdir()

    (config_dir / "settings.yaml").write_text("input_table_list: []\n")
    (config_dir / "telework_duration.yaml").write_text(
        "CHOOSER_FILTER_COLUMN_NAME: has_in_home_work_activity\n"
        "DURATION_CATEGORY_COLUMN_NAME: telework_duration_category\n"
        "DURATION_HOURS_COLUMN_NAME: telework_duration_hours\n"
        "ALTS: telework_duration_alts.csv\n"
        "ALT_NAME_COLUMN: alt\n"
        "ALT_DURATION_COLUMN: duration_hours\n"
        "PROBS_SPEC: telework_duration_probs.csv\n"
    )
    (config_dir / "telework_duration_alts.csv").write_text(
        "alt,duration_hours\n" "short,2.0\n" "long,4.0\n"
    )
    (config_dir / "telework_duration_probs.csv").write_text("short,long\n0.0,1.0\n")
    (config_dir / "telework_duration_mnl.yaml").write_text(
        "CHOOSER_FILTER_COLUMN_NAME: has_in_home_work_activity\n"
        "DURATION_CATEGORY_COLUMN_NAME: telework_duration_category\n"
        "DURATION_HOURS_COLUMN_NAME: telework_duration_hours\n"
        "ALTS: telework_duration_alts.csv\n"
        "ALT_NAME_COLUMN: alt\n"
        "ALT_DURATION_COLUMN: duration_hours\n"
        "CHOICE_MODEL: MNL\n"
        "SPEC: telework_duration_mnl.csv\n"
        "COEFFICIENTS: telework_duration_mnl_coeffs.csv\n"
        "LOGIT_TYPE: MNL\n"
    )
    (config_dir / "telework_duration_mnl.csv").write_text(
        "Label,Description,Expression,short,long\n"
        "util_asc,,1,coef_acs_short,\n"
        "util_female,,sex==2,coef_female_short,\n"
    )
    (config_dir / "telework_duration_mnl_coeffs.csv").write_text(
        "coefficient_name,value,constrain\n"
        "coef_acs_short,-0.1,F\n"
        "coef_female_short,0.5,F\n"
    )

    return root


@pytest.fixture(scope="module")
def real_state(real_example_root):
    return workflow.State.make_default(real_example_root)


def test_telework_duration_probabilistic(real_state):
    model_settings = model.TeleworkDurationSettings.read_settings_file(
        real_state.filesystem, "telework_duration.yaml"
    )

    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    persons_merged = pd.DataFrame(
        {
            "has_in_home_work_activity": [True, False, True],
            "sex": [1, 2, 2],
        },
        index=persons.index,
    )

    real_state.add_table("persons", persons.copy())

    model.telework_duration(
        state=real_state,
        persons_merged=persons_merged,
        persons=persons,
        model_settings=model_settings,
    )

    out = real_state.get_dataframe("persons")
    assert out["telework_duration_category"].astype(str).to_dict() == {
        1: "long",
        2: "",
        3: "long",
    }
    assert out["telework_duration_hours"].to_dict() == {
        1: 4.0,
        2: 0.0,
        3: 4.0,
    }


def test_telework_duration_mnl(real_state):
    model_settings = model.TeleworkDurationSettings.read_settings_file(
        real_state.filesystem, "telework_duration_mnl.yaml"
    )

    persons = pd.DataFrame(index=pd.Index([1, 2, 3], name="person_id"))
    persons_merged = pd.DataFrame(
        {
            "has_in_home_work_activity": [True, False, True],
            "sex": [1, 2, 2],
        },
        index=persons.index,
    )

    real_state.add_table("persons", persons.copy())

    model.telework_duration(
        state=real_state,
        persons_merged=persons_merged,
        persons=persons,
        model_settings=model_settings,
    )

    out = real_state.get_dataframe("persons")
    assert out["telework_duration_category"].astype(str).to_dict() == {
        1: "long",
        2: "",
        3: "long",
    }
    assert out["telework_duration_hours"].to_dict() == {
        1: 4.0,
        2: 0.0,
        3: 4.0,
    }

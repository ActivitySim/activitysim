import pandas as pd

from activitysim.core.workflow import State


def test_parquet_resume_truncates_checkpoint_inventory(tmp_path):
    state = State.make_default(
        tmp_path,
        settings={"checkpoint_format": "parquet"},
    )

    state.add_table("sample", pd.DataFrame({"value": [1]}))
    state.checkpoint.add("target")

    state.add_table("sample", pd.DataFrame({"value": [2]}))
    state.checkpoint.add("later")
    state.checkpoint.close_store()

    try:
        state.checkpoint.restore("target")

        inventory = state.checkpoint.get_inventory()
        checkpoint_names = inventory["checkpoint_name"].tolist()

        assert checkpoint_names == ["target"]
        assert state.checkpoint.store.list_checkpoint_names() == ["target"]
        assert state.get_dataframe("sample")["value"].tolist() == [1]
    finally:
        state.checkpoint.close_store()
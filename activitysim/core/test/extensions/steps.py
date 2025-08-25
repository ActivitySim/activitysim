from __future__ import annotations

import pandas as pd

from activitysim.core import workflow


@workflow.step
def step1(state: workflow.State) -> None:
    table1 = pd.DataFrame({"c": [1, 2, 3]})
    state.add_table("table1", table1)


@workflow.step
def step2(state: workflow.State) -> None:
    table1 = pd.DataFrame({"c": [2, 4, 6]})
    state.add_table("table2", table1)


@workflow.step
def step3(state: workflow.State) -> None:
    table1 = pd.DataFrame({"c": [3, 6, 9]})
    state.add_table("table3", table1)


@workflow.step
def step_add_col(state: workflow.State) -> None:
    table_name = state.get_step_arg("table_name")
    assert table_name is not None

    col_name = state.get_step_arg("column_name")
    assert col_name is not None

    table = state.get_dataframe(table_name)

    assert col_name not in table.columns

    table[col_name] = table.index + (1000 * len(table.columns))

    state.add_table(table_name, table)


@workflow.step
def step_forget_tab(state: workflow.State) -> None:
    table_name = state.get_step_arg("table_name")
    assert table_name is not None

    table = state.get_dataframe(table_name)

    state.drop_table(table_name)


@workflow.step
def create_households(state: workflow.State) -> None:
    df = pd.DataFrame({"household_id": [1, 2, 3], "home_zone_id": {100, 100, 101}})
    state.add_table("households", df)

    state.get_rn_generator().add_channel("households", df)

    state.tracing.register_traceable_table("households", df)

import pandas as pd

from activitysim.core import workflow


@workflow.step
def step1(whale: workflow.Whale):

    table1 = pd.DataFrame({"c": [1, 2, 3]})
    whale.add_table("table1", table1)


@workflow.step
def step2(whale: workflow.Whale):

    table1 = pd.DataFrame({"c": [2, 4, 6]})
    whale.add_table("table2", table1)


@workflow.step
def step3(whale: workflow.Whale):

    table1 = pd.DataFrame({"c": [3, 6, 9]})
    whale.add_table("table3", table1)


@workflow.step
def step_add_col(whale: workflow.Whale):

    table_name = whale.get_step_arg("table_name")
    assert table_name is not None

    col_name = whale.get_step_arg("column_name")
    assert col_name is not None

    table = whale.get_table(table_name)

    assert col_name not in table.columns

    table[col_name] = table.index + (1000 * len(table.columns))

    whale.add_table(table_name, table)


@workflow.step
def step_forget_tab(whale: workflow.Whale):

    table_name = whale.get_step_arg("table_name")
    assert table_name is not None

    table = whale.get_table(table_name)

    whale.drop_table(table_name)


@workflow.step
def create_households(whale: workflow.Whale):

    df = pd.DataFrame({"household_id": [1, 2, 3], "home_zone_id": {100, 100, 101}})
    whale.add_table("households", df)

    whale.get_rn_generator().add_channel("households", df)

    whale.tracing.register_traceable_table("households", df)

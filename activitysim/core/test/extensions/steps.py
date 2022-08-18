import pandas as pd

from activitysim.core import inject, pipeline, tracing


@inject.step()
def step1():

    table1 = pd.DataFrame({"c": [1, 2, 3]})
    inject.add_table("table1", table1)


@inject.step()
def step2():

    table1 = pd.DataFrame({"c": [2, 4, 6]})
    inject.add_table("table2", table1)


@inject.step()
def step3():

    table1 = pd.DataFrame({"c": [3, 6, 9]})
    inject.add_table("table3", table1)


@inject.step()
def step_add_col():

    table_name = inject.get_step_arg("table_name")
    assert table_name is not None

    col_name = inject.get_step_arg("column_name")
    assert col_name is not None

    table = pipeline.get_table(table_name)

    assert col_name not in table.columns

    table[col_name] = table.index + (1000 * len(table.columns))

    pipeline.replace_table(table_name, table)


@inject.step()
def step_forget_tab():

    table_name = inject.get_step_arg("table_name")
    assert table_name is not None

    table = pipeline.get_table(table_name)

    pipeline.drop_table(table_name)


@inject.step()
def create_households(trace_hh_id):

    df = pd.DataFrame({"household_id": [1, 2, 3], "home_zone_id": {100, 100, 101}})
    inject.add_table("households", df)

    pipeline.get_rn_generator().add_channel("households", df)

    tracing.register_traceable_table("households", df)

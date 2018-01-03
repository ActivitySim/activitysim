
import pandas as pd
from activitysim.core import inject
from activitysim.core import pipeline


@inject.step()
def step1():

    table1 = pd.DataFrame({'column1': [1, 2, 3]})
    inject.add_table('table1', table1)


@inject.step()
def step2():

    table_name = inject.get_step_arg('table_name')
    assert table_name is not None

    table2 = pd.DataFrame({'column1': [10, 20, 30]})
    inject.add_table(table_name, table2)

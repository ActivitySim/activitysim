# Orca
# Copyright (C) 2016 UrbanSim Inc.
# See full license in LICENSE.

import os
import tempfile

import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import orca
from ..utils.testing import assert_frames_equal


def setup_function(func):
    orca.clear_all()
    orca.enable_cache()


def teardown_function(func):
    orca.clear_all()
    orca.enable_cache()


@pytest.fixture
def df():
    return pd.DataFrame(
        [[1, 4],
         [2, 5],
         [3, 6]],
        columns=['a', 'b'],
        index=['x', 'y', 'z'])


def test_tables(df):
    wrapped_df = orca.add_table('test_frame', df)

    @orca.table()
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    assert set(orca.list_tables()) == {'test_frame', 'test_func'}

    table = orca.get_table('test_frame')
    assert table is wrapped_df
    assert table.columns == ['a', 'b']
    assert table.local_columns == ['a', 'b']
    assert len(table) == 3
    pdt.assert_index_equal(table.index, df.index)
    pdt.assert_series_equal(table.get_column('a'), df.a)
    pdt.assert_series_equal(table.a, df.a)
    pdt.assert_series_equal(table['b'], df['b'])

    table = orca._TABLES['test_func']
    assert table.index is None
    assert table.columns == []
    assert len(table) is 0
    pdt.assert_frame_equal(table.to_frame(), df / 2)
    pdt.assert_frame_equal(table.to_frame([]), df[[]])
    pdt.assert_frame_equal(table.to_frame(columns=['a']), df[['a']] / 2)
    pdt.assert_frame_equal(table.to_frame(columns='a'), df[['a']] / 2)
    pdt.assert_index_equal(table.index, df.index)
    pdt.assert_series_equal(table.get_column('a'), df.a / 2)
    pdt.assert_series_equal(table.a, df.a / 2)
    pdt.assert_series_equal(table['b'], df['b'] / 2)
    assert len(table) == 3
    assert table.columns == ['a', 'b']


def test_table_func_cache(df):
    orca.add_injectable('x', 2)

    @orca.table(cache=True)
    def table(variable='x'):
        return df * variable

    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 2)
    orca.add_injectable('x', 3)
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 2)
    orca.get_table('table').clear_cached()
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 3)
    orca.add_injectable('x', 4)
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 3)
    orca.clear_cache()
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 4)
    orca.add_injectable('x', 5)
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 4)
    orca.add_table('table', table)
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 5)


def test_table_func_cache_disabled(df):
    orca.add_injectable('x', 2)

    @orca.table('table', cache=True)
    def asdf(x):
        return df * x

    orca.disable_cache()

    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 2)
    orca.add_injectable('x', 3)
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 3)

    orca.enable_cache()

    orca.add_injectable('x', 4)
    pdt.assert_frame_equal(orca.get_table('table').to_frame(), df * 3)


def test_table_copy(df):
    orca.add_table('test_frame_copied', df, copy_col=True)
    orca.add_table('test_frame_uncopied', df, copy_col=False)
    orca.add_table('test_func_copied', lambda: df, copy_col=True)
    orca.add_table('test_func_uncopied', lambda: df, copy_col=False)

    @orca.table(copy_col=True)
    def test_funcd_copied():
        return df

    @orca.table(copy_col=False)
    def test_funcd_uncopied():
        return df

    @orca.table(copy_col=True)
    def test_funcd_copied2(test_frame_copied):
        # local returns original, but it is copied by copy_col.
        return test_frame_copied.local

    @orca.table(copy_col=True)
    def test_funcd_copied3(test_frame_uncopied):
        # local returns original, but it is copied by copy_col.
        return test_frame_uncopied.local

    @orca.table(copy_col=False)
    def test_funcd_uncopied2(test_frame_copied):
        # local returns original.
        return test_frame_copied.local

    @orca.table(copy_col=False)
    def test_funcd_uncopied3(test_frame_uncopied):
        # local returns original.
        return test_frame_uncopied.local

    orca.add_table('test_cache_copied', lambda: df, cache=True, copy_col=True)
    orca.add_table(
        'test_cache_uncopied', lambda: df, cache=True, copy_col=False)

    @orca.table(cache=True, copy_col=True)
    def test_cached_copied():
        return df

    @orca.table(cache=True, copy_col=False)
    def test_cached_uncopied():
        return df

    # Create tables with computed columns.
    orca.add_table(
        'test_copied_columns', pd.DataFrame(index=df.index), copy_col=True)
    orca.add_table(
        'test_uncopied_columns', pd.DataFrame(index=df.index), copy_col=False)
    for column_name in ['a', 'b']:
        label = "test_frame_uncopied.{}".format(column_name)

        def func(col=label):
            return col
        for table_name in ['test_copied_columns', 'test_uncopied_columns']:
            orca.add_column(table_name, column_name, func)

    for name in ['test_frame_uncopied', 'test_func_uncopied',
                 'test_funcd_uncopied', 'test_funcd_uncopied2',
                 'test_funcd_uncopied3', 'test_cache_uncopied',
                 'test_cached_uncopied', 'test_uncopied_columns',
                 'test_frame_copied', 'test_func_copied',
                 'test_funcd_copied', 'test_funcd_copied2',
                 'test_funcd_copied3', 'test_cache_copied',
                 'test_cached_copied', 'test_copied_columns']:
        table = orca.get_table(name)
        table2 = orca.get_table(name)

        # to_frame will always return a copy.
        if 'columns' in name:
            assert_frames_equal(table.to_frame(), df)
        else:
            pdt.assert_frame_equal(table.to_frame(), df)
        assert table.to_frame() is not df
        pdt.assert_frame_equal(table.to_frame(), table.to_frame())
        assert table.to_frame() is not table.to_frame()
        pdt.assert_series_equal(table.to_frame()['a'], df['a'])
        assert table.to_frame()['a'] is not df['a']
        pdt.assert_series_equal(table.to_frame()['a'],
                                table.to_frame()['a'])
        assert table.to_frame()['a'] is not table.to_frame()['a']

        if 'uncopied' in name:
            pdt.assert_series_equal(table['a'], df['a'])
            assert table['a'] is df['a']
            pdt.assert_series_equal(table['a'], table2['a'])
            assert table['a'] is table2['a']
        else:
            pdt.assert_series_equal(table['a'], df['a'])
            assert table['a'] is not df['a']
            pdt.assert_series_equal(table['a'], table2['a'])
            assert table['a'] is not table2['a']


def test_columns_for_table():
    orca.add_column(
        'table1', 'col10', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
    orca.add_column(
        'table2', 'col20', pd.Series([10, 11, 12], index=['x', 'y', 'z']))

    @orca.column('table1')
    def col11():
        return pd.Series([4, 5, 6], index=['a', 'b', 'c'])

    @orca.column('table2', 'col21')
    def asdf():
        return pd.Series([13, 14, 15], index=['x', 'y', 'z'])

    t1_col_names = orca.list_columns_for_table('table1')
    assert set(t1_col_names) == {'col10', 'col11'}

    t2_col_names = orca.list_columns_for_table('table2')
    assert set(t2_col_names) == {'col20', 'col21'}

    t1_cols = orca._columns_for_table('table1')
    assert 'col10' in t1_cols and 'col11' in t1_cols

    t2_cols = orca._columns_for_table('table2')
    assert 'col20' in t2_cols and 'col21' in t2_cols


def test_columns_and_tables(df):
    orca.add_table('test_frame', df)

    @orca.table()
    def test_func(test_frame):
        return test_frame.to_frame() / 2

    orca.add_column('test_frame', 'c', pd.Series([7, 8, 9], index=df.index))

    @orca.column('test_func', 'd')
    def asdf(test_func):
        return test_func.to_frame(columns=['b'])['b'] * 2

    @orca.column('test_func')
    def e(column='test_func.d'):
        return column + 1

    test_frame = orca.get_table('test_frame')
    assert set(test_frame.columns) == set(['a', 'b', 'c'])
    assert_frames_equal(
        test_frame.to_frame(),
        pd.DataFrame(
            {'a': [1, 2, 3],
             'b': [4, 5, 6],
             'c': [7, 8, 9]},
            index=['x', 'y', 'z']))
    assert_frames_equal(
        test_frame.to_frame(columns=['a', 'c']),
        pd.DataFrame(
            {'a': [1, 2, 3],
             'c': [7, 8, 9]},
            index=['x', 'y', 'z']))

    test_func_df = orca._TABLES['test_func']
    assert set(test_func_df.columns) == set(['d', 'e'])
    assert_frames_equal(
        test_func_df.to_frame(),
        pd.DataFrame(
            {'a': [0.5, 1, 1.5],
             'b': [2, 2.5, 3],
             'c': [3.5, 4, 4.5],
             'd': [4., 5., 6.],
             'e': [5., 6., 7.]},
            index=['x', 'y', 'z']))
    assert_frames_equal(
        test_func_df.to_frame(columns=['b', 'd']),
        pd.DataFrame(
            {'b': [2, 2.5, 3],
             'd': [4., 5., 6.]},
            index=['x', 'y', 'z']))
    assert set(test_func_df.columns) == set(['a', 'b', 'c', 'd', 'e'])

    assert set(orca.list_columns()) == {
        ('test_frame', 'c'), ('test_func', 'd'), ('test_func', 'e')}


def test_column_cache(df):
    orca.add_injectable('x', 2)
    series = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
    key = ('table', 'col')

    @orca.table()
    def table():
        return df

    @orca.column(*key, cache=True)
    def column(variable='x'):
        return series * variable

    def c():
        return orca._COLUMNS[key]

    pdt.assert_series_equal(c()(), series * 2)
    orca.add_injectable('x', 3)
    pdt.assert_series_equal(c()(), series * 2)
    c().clear_cached()
    pdt.assert_series_equal(c()(), series * 3)
    orca.add_injectable('x', 4)
    pdt.assert_series_equal(c()(), series * 3)
    orca.clear_cache()
    pdt.assert_series_equal(c()(), series * 4)
    orca.add_injectable('x', 5)
    pdt.assert_series_equal(c()(), series * 4)
    orca.get_table('table').clear_cached()
    pdt.assert_series_equal(c()(), series * 5)
    orca.add_injectable('x', 6)
    pdt.assert_series_equal(c()(), series * 5)
    orca.add_column(*key, column=column, cache=True)
    pdt.assert_series_equal(c()(), series * 6)


def test_column_cache_disabled(df):
    orca.add_injectable('x', 2)
    series = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
    key = ('table', 'col')

    @orca.table()
    def table():
        return df

    @orca.column(*key, cache=True)
    def column(x):
        return series * x

    def c():
        return orca._COLUMNS[key]

    orca.disable_cache()

    pdt.assert_series_equal(c()(), series * 2)
    orca.add_injectable('x', 3)
    pdt.assert_series_equal(c()(), series * 3)

    orca.enable_cache()

    orca.add_injectable('x', 4)
    pdt.assert_series_equal(c()(), series * 3)


def test_update_col(df):
    wrapped = orca.add_table('table', df)

    wrapped.update_col('b', pd.Series([7, 8, 9], index=df.index))
    pdt.assert_series_equal(
        wrapped['b'], pd.Series([7, 8, 9], index=df.index, name='b'))

    a_dtype = wrapped['a'].dtype

    # test 1 - cast the data type before the update
    wrapped.update_col_from_series('a', pd.Series(dtype=a_dtype))
    pdt.assert_series_equal(wrapped['a'], df['a'])

    # test 2 - let the update method do the cast
    wrapped.update_col_from_series('a', pd.Series(), True)
    pdt.assert_series_equal(wrapped['a'], df['a'])

    # test 3 - don't cast, should raise an error
    with pytest.raises(ValueError):
        wrapped.update_col_from_series('a', pd.Series())

    wrapped.update_col_from_series('a', pd.Series([99], index=['y']))
    pdt.assert_series_equal(
        wrapped['a'], pd.Series([1, 99, 3], index=df.index, name='a'))


class _FakeTable(object):
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns


@pytest.fixture
def fta():
    return _FakeTable('a', ['aa', 'ab', 'ac'])


@pytest.fixture
def ftb():
    return _FakeTable('b', ['bx', 'by', 'bz'])


def test_column_map_raises(fta, ftb):
    with pytest.raises(RuntimeError):
        orca.column_map([fta, ftb], ['aa', 'by', 'bz', 'cw'])


def test_column_map_none(fta, ftb):
    assert orca.column_map([fta, ftb], None) == {'a': None, 'b': None}


def test_column_map(fta, ftb):
    result = orca.column_map([fta, ftb], ['aa', 'by', 'bz'])
    assert result['a'] == ['aa']
    assert sorted(result['b']) == ['by', 'bz']

    result = orca.column_map([fta, ftb], ['by', 'bz'])
    assert result['a'] == []
    assert sorted(result['b']) == ['by', 'bz']


def test_is_step():
    @orca.step()
    def test_step():
        pass

    assert orca.is_step('test_step') is True
    assert orca.is_step('not_a_step') is False


def test_steps(df):
    orca.add_table('test_table', df)

    df2 = df / 2
    orca.add_table('test_table2', df2)

    @orca.step()
    def test_step(test_table, test_column='test_table2.b'):
        tt = test_table.to_frame()
        test_table['a'] = tt['a'] + tt['b']
        pdt.assert_series_equal(test_column, df2['b'])

    with pytest.raises(KeyError):
        orca.get_step('asdf')

    step = orca.get_step('test_step')
    assert step._tables_used() == set(['test_table', 'test_table2'])
    step()

    table = orca.get_table('test_table')
    pdt.assert_frame_equal(
        table.to_frame(),
        pd.DataFrame(
            {'a': [5, 7, 9],
             'b': [4, 5, 6]},
            index=['x', 'y', 'z']))

    assert orca.list_steps() == ['test_step']


def test_step_run(df):
    orca.add_table('test_table', df)

    @orca.table()
    def table_func(test_table):
        tt = test_table.to_frame()
        tt['c'] = [7, 8, 9]
        return tt

    @orca.column('table_func')
    def new_col(test_table, table_func):
        tt = test_table.to_frame()
        tf = table_func.to_frame(columns=['c'])
        return tt['a'] + tt['b'] + tf['c']

    @orca.step()
    def test_step1(iter_var, test_table, table_func):
        tf = table_func.to_frame(columns=['new_col'])
        test_table[iter_var] = tf['new_col'] + iter_var

    @orca.step('test_step2')
    def asdf(table='test_table'):
        tt = table.to_frame()
        table['a'] = tt['a'] ** 2

    orca.run(steps=['test_step1', 'test_step2'], iter_vars=[2000, 3000])

    test_table = orca.get_table('test_table')
    assert_frames_equal(
        test_table.to_frame(),
        pd.DataFrame(
            {'a': [1, 16, 81],
             'b': [4, 5, 6],
             2000: [2012, 2015, 2018],
             3000: [3012, 3017, 3024]},
            index=['x', 'y', 'z']))

    m = orca.get_step('test_step1')
    assert set(m._tables_used()) == {'test_table', 'table_func'}


def test_step_func_source_data():
    @orca.step()
    def test_step():
        return 'orca'

    filename, lineno, source = orca.get_step('test_step').func_source_data()

    assert filename.endswith('test_orca.py')
    assert isinstance(lineno, int)
    assert source == (
        "    @orca.step()\n"
        "    def test_step():\n"
        "        return 'orca'\n")


def test_get_broadcast():
    orca.broadcast('a', 'b', cast_on='ax', onto_on='bx')
    orca.broadcast('x', 'y', cast_on='yx', onto_index=True)

    assert orca.is_broadcast('a', 'b') is True
    assert orca.is_broadcast('b', 'a') is False

    with pytest.raises(KeyError):
        orca.get_broadcast('b', 'a')

    ab = orca.get_broadcast('a', 'b')
    assert isinstance(ab, orca.Broadcast)
    assert ab == ('a', 'b', 'ax', 'bx', False, False)

    xy = orca.get_broadcast('x', 'y')
    assert isinstance(xy, orca.Broadcast)
    assert xy == ('x', 'y', 'yx', None, False, True)


def test_get_broadcasts():
    orca.broadcast('a', 'b')
    orca.broadcast('b', 'c')
    orca.broadcast('z', 'b')
    orca.broadcast('f', 'g')

    with pytest.raises(ValueError):
        orca._get_broadcasts(['a', 'b', 'g'])

    assert set(orca._get_broadcasts(['a', 'b', 'c', 'z']).keys()) == \
        {('a', 'b'), ('b', 'c'), ('z', 'b')}
    assert set(orca._get_broadcasts(['a', 'b', 'z']).keys()) == \
        {('a', 'b'), ('z', 'b')}
    assert set(orca._get_broadcasts(['a', 'b', 'c']).keys()) == \
        {('a', 'b'), ('b', 'c')}

    assert set(orca.list_broadcasts()) == \
        {('a', 'b'), ('b', 'c'), ('z', 'b'), ('f', 'g')}


def test_collect_variables(df):
    orca.add_table('df', df)

    @orca.table()
    def df_func():
        return df

    @orca.column('df')
    def zzz():
        return df['a'] / 2

    orca.add_injectable('answer', 42)

    @orca.injectable()
    def injected():
        return 'injected'

    @orca.table('source table', cache=True)
    def source():
        return df

    with pytest.raises(KeyError):
        orca._collect_variables(['asdf'])

    with pytest.raises(KeyError):
        orca._collect_variables(names=['df'], expressions=['asdf'])

    names = ['df', 'df_func', 'answer', 'injected', 'source_label', 'df_a']
    expressions = ['source table', 'df.a']
    things = orca._collect_variables(names, expressions)

    assert set(things.keys()) == set(names)
    assert isinstance(things['source_label'], orca.DataFrameWrapper)
    pdt.assert_frame_equal(things['source_label'].to_frame(), df)
    assert isinstance(things['df_a'], pd.Series)
    pdt.assert_series_equal(things['df_a'], df['a'])


def test_collect_variables_expression_only(df):
    @orca.table()
    def table():
        return df

    vars = orca._collect_variables(['a'], ['table.a'])
    pdt.assert_series_equal(vars['a'], df.a)


def test_injectables():
    orca.add_injectable('answer', 42)

    @orca.injectable()
    def func1(answer):
        return answer * 2

    @orca.injectable('func2', autocall=False)
    def asdf(variable='x'):
        return variable / 2

    @orca.injectable()
    def func3(func2):
        return func2(4)

    @orca.injectable()
    def func4(func='func1'):
        return func / 2

    assert orca._INJECTABLES['answer'] == 42
    assert orca._INJECTABLES['func1']() == 42 * 2
    assert orca._INJECTABLES['func2'](4) == 2
    assert orca._INJECTABLES['func3']() == 2
    assert orca._INJECTABLES['func4']() == 42

    assert orca.get_injectable('answer') == 42
    assert orca.get_injectable('func1') == 42 * 2
    assert orca.get_injectable('func2')(4) == 2
    assert orca.get_injectable('func3') == 2
    assert orca.get_injectable('func4') == 42

    with pytest.raises(KeyError):
        orca.get_injectable('asdf')

    assert set(orca.list_injectables()) == \
        {'answer', 'func1', 'func2', 'func3', 'func4'}


def test_injectables_combined(df):
    @orca.injectable()
    def column():
        return pd.Series(['a', 'b', 'c'], index=df.index)

    @orca.table()
    def table():
        return df

    @orca.step()
    def step(table, column):
        df = table.to_frame()
        df['new'] = column
        orca.add_table('table', df)

    orca.run(steps=['step'])

    table_wr = orca.get_table('table').to_frame()

    pdt.assert_frame_equal(table_wr[['a', 'b']], df)
    pdt.assert_series_equal(table_wr['new'], pd.Series(column(), name='new'))


def test_injectables_cache():
    x = 2

    @orca.injectable(autocall=True, cache=True)
    def inj():
        return x * x

    def i():
        return orca._INJECTABLES['inj']

    assert i()() == 4
    x = 3
    assert i()() == 4
    i().clear_cached()
    assert i()() == 9
    x = 4
    assert i()() == 9
    orca.clear_cache()
    assert i()() == 16
    x = 5
    assert i()() == 16
    orca.add_injectable('inj', inj, autocall=True, cache=True)
    assert i()() == 25


def test_injectables_cache_disabled():
    x = 2

    @orca.injectable(autocall=True, cache=True)
    def inj():
        return x * x

    def i():
        return orca._INJECTABLES['inj']

    orca.disable_cache()

    assert i()() == 4
    x = 3
    assert i()() == 9

    orca.enable_cache()

    assert i()() == 9
    x = 4
    assert i()() == 9

    orca.disable_cache()
    assert i()() == 16


def test_memoized_injectable():
    outside = 'x'

    @orca.injectable(autocall=False, memoize=True)
    def x(s):
        return outside + s

    assert 'x' in orca._MEMOIZED

    def getx():
        return orca.get_injectable('x')

    assert hasattr(getx(), 'cache')
    assert hasattr(getx(), 'clear_cached')

    assert getx()('y') == 'xy'
    outside = 'z'
    assert getx()('y') == 'xy'

    getx().clear_cached()

    assert getx()('y') == 'zy'


def test_memoized_injectable_cache_off():
    outside = 'x'

    @orca.injectable(autocall=False, memoize=True)
    def x(s):
        return outside + s

    def getx():
        return orca.get_injectable('x')('y')

    orca.disable_cache()

    assert getx() == 'xy'
    outside = 'z'
    assert getx() == 'zy'

    orca.enable_cache()
    outside = 'a'

    assert getx() == 'zy'

    orca.disable_cache()

    assert getx() == 'ay'


def test_clear_cache_all(df):
    @orca.table(cache=True)
    def table():
        return df

    @orca.column('table', cache=True)
    def z(table):
        return df.a

    @orca.injectable(cache=True)
    def x():
        return 'x'

    @orca.injectable(autocall=False, memoize=True)
    def y(s):
        return s + 'y'

    orca.eval_variable('table.z')
    orca.eval_variable('x')
    orca.get_injectable('y')('x')

    assert list(orca._TABLE_CACHE.keys()) == ['table']
    assert list(orca._COLUMN_CACHE.keys()) == [('table', 'z')]
    assert list(orca._INJECTABLE_CACHE.keys()) == ['x']
    assert orca._MEMOIZED['y'].value.cache == {(('x',), None): 'xy'}

    orca.clear_cache()

    assert orca._TABLE_CACHE == {}
    assert orca._COLUMN_CACHE == {}
    assert orca._INJECTABLE_CACHE == {}
    assert orca._MEMOIZED['y'].value.cache == {}


def test_clear_cache_scopes(df):
    @orca.table(cache=True, cache_scope='forever')
    def table():
        return df

    @orca.column('table', cache=True, cache_scope='iteration')
    def z(table):
        return df.a

    @orca.injectable(cache=True, cache_scope='step')
    def x():
        return 'x'

    @orca.injectable(autocall=False, memoize=True, cache_scope='iteration')
    def y(s):
        return s + 'y'

    orca.eval_variable('table.z')
    orca.eval_variable('x')
    orca.get_injectable('y')('x')

    assert list(orca._TABLE_CACHE.keys()) == ['table']
    assert list(orca._COLUMN_CACHE.keys()) == [('table', 'z')]
    assert list(orca._INJECTABLE_CACHE.keys()) == ['x']
    assert orca._MEMOIZED['y'].value.cache == {(('x',), None): 'xy'}

    orca.clear_cache(scope='step')

    assert list(orca._TABLE_CACHE.keys()) == ['table']
    assert list(orca._COLUMN_CACHE.keys()) == [('table', 'z')]
    assert orca._INJECTABLE_CACHE == {}
    assert orca._MEMOIZED['y'].value.cache == {(('x',), None): 'xy'}

    orca.clear_cache(scope='iteration')

    assert list(orca._TABLE_CACHE.keys()) == ['table']
    assert orca._COLUMN_CACHE == {}
    assert orca._INJECTABLE_CACHE == {}
    assert orca._MEMOIZED['y'].value.cache == {}

    orca.clear_cache(scope='forever')

    assert orca._TABLE_CACHE == {}
    assert orca._COLUMN_CACHE == {}
    assert orca._INJECTABLE_CACHE == {}
    assert orca._MEMOIZED['y'].value.cache == {}


def test_cache_scope(df):
    orca.add_injectable('x', 11)
    orca.add_injectable('y', 22)
    orca.add_injectable('z', 33)
    orca.add_injectable('iterations', 1)

    @orca.injectable(cache=True, cache_scope='forever')
    def a(x):
        return x

    @orca.injectable(cache=True, cache_scope='iteration')
    def b(y):
        return y

    @orca.injectable(cache=True, cache_scope='step')
    def c(z):
        return z

    @orca.step()
    def m1(iter_var, a, b, c):
        orca.add_injectable('x', iter_var + a)
        orca.add_injectable('y', iter_var + b)
        orca.add_injectable('z', iter_var + c)

        assert a == 11

    @orca.step()
    def m2(iter_var, a, b, c, iterations):
        assert a == 11
        if iter_var == 1000:
            assert b == 22
            assert c == 1033
        elif iter_var == 2000:
            assert b == 1022
            assert c == 3033

        orca.add_injectable('iterations', iterations + 1)

    orca.run(['m1', 'm2'], iter_vars=[1000, 2000])


def test_table_func_local_cols(df):
    @orca.table()
    def table():
        return df
    orca.add_column(
        'table', 'new', pd.Series(['a', 'b', 'c'], index=df.index))

    assert orca.get_table('table').local_columns == ['a', 'b']


def test_is_table(df):
    orca.add_table('table', df)
    assert orca.is_table('table') is True
    assert orca.is_table('asdf') is False


@pytest.fixture
def store_name(request):
    fname = tempfile.NamedTemporaryFile(suffix='.h5').name

    def fin():
        if os.path.isfile(fname):
            os.remove(fname)
    request.addfinalizer(fin)

    return fname


def test_write_tables(df, store_name):
    orca.add_table('table', df)

    @orca.step()
    def step(table):
        pass

    step_tables = orca.get_step_table_names(['step'])

    orca.write_tables(store_name, step_tables, None)
    with pd.HDFStore(store_name, mode='r') as store:
        assert 'table' in store
        pdt.assert_frame_equal(store['table'], df)

    orca.write_tables(store_name, step_tables, 1969)

    with pd.HDFStore(store_name, mode='r') as store:
        assert '1969/table' in store
        pdt.assert_frame_equal(store['1969/table'], df)


def test_write_all_tables(df, store_name):
    orca.add_table('table', df)
    orca.write_tables(store_name)

    with pd.HDFStore(store_name, mode='r') as store:
        for t in orca.list_tables():
            assert t in store


def test_run_and_write_tables(df, store_name):
    orca.add_table('table', df)

    def year_key(y):
        return '{}'.format(y)

    def series_year(y):
        return pd.Series([y] * 3, index=df.index, name=str(y))

    @orca.step()
    def step(iter_var, table):
        table[year_key(iter_var)] = series_year(iter_var)

    orca.run(
        ['step'], iter_vars=range(11), data_out=store_name, out_interval=3)

    with pd.HDFStore(store_name, mode='r') as store:
        for year in range(0, 11, 3):
            key = '{}/table'.format(year)
            assert key in store

            for x in range(year):
                pdt.assert_series_equal(
                    store[key][year_key(x)], series_year(x))

        assert 'base/table' in store

        for x in range(11):
            pdt.assert_series_equal(
                store['10/table'][year_key(x)], series_year(x))


def test_run_and_write_tables_out_tables_provided(df, store_name):
    table_names = ['table', 'table2', 'table3']
    for t in table_names:
        orca.add_table(t, df)

    @orca.step()
    def step(iter_var, table, table2):
        return

    orca.run(
        ['step'],
        iter_vars=range(1),
        data_out=store_name,
        out_base_tables=table_names,
        out_run_tables=['table'])

    with pd.HDFStore(store_name, mode='r') as store:

        for t in table_names:
            assert 'base/{}'.format(t) in store

        assert '0/table' in store
        assert '0/table2' not in store
        assert '0/table3' not in store


def test_get_raw_table(df):
    orca.add_table('table1', df)

    @orca.table()
    def table2():
        return df

    assert isinstance(orca.get_raw_table('table1'), orca.DataFrameWrapper)
    assert isinstance(orca.get_raw_table('table2'), orca.TableFuncWrapper)

    assert orca.table_type('table1') == 'dataframe'
    assert orca.table_type('table2') == 'function'


def test_get_table(df):
    orca.add_table('frame', df)

    @orca.table()
    def table():
        return df

    @orca.table(cache=True)
    def source():
        return df

    fr = orca.get_table('frame')
    ta = orca.get_table('table')
    so = orca.get_table('source')

    with pytest.raises(KeyError):
        orca.get_table('asdf')

    assert isinstance(fr, orca.DataFrameWrapper)
    assert isinstance(ta, orca.DataFrameWrapper)
    assert isinstance(so, orca.DataFrameWrapper)

    pdt.assert_frame_equal(fr.to_frame(), df)
    pdt.assert_frame_equal(ta.to_frame(), df)
    pdt.assert_frame_equal(so.to_frame(), df)


def test_cache_disabled_cm():
    x = 3

    @orca.injectable(cache=True)
    def xi():
        return x

    assert orca.get_injectable('xi') == 3
    x = 5
    assert orca.get_injectable('xi') == 3

    with orca.cache_disabled():
        assert orca.get_injectable('xi') == 5

    # cache still gets updated even when cacheing is off
    assert orca.get_injectable('xi') == 5


def test_injectables_cm():
    orca.add_injectable('a', 'a')
    orca.add_injectable('b', 'b')
    orca.add_injectable('c', 'c')

    with orca.injectables():
        assert orca._INJECTABLES == {
            'a': 'a', 'b': 'b', 'c': 'c'
        }

    with orca.injectables(c='d', x='x', y='y', z='z'):
        assert orca._INJECTABLES == {
            'a': 'a', 'b': 'b', 'c': 'd',
            'x': 'x', 'y': 'y', 'z': 'z'
        }

    assert orca._INJECTABLES == {
        'a': 'a', 'b': 'b', 'c': 'c'
    }


def test_temporary_tables_cm():
    orca.add_table('a', pd.DataFrame())

    with orca.temporary_tables():
        assert sorted(orca._TABLES.keys()) == ['a']

    with orca.temporary_tables(a=pd.DataFrame(), b=pd.DataFrame()):
        assert sorted(orca._TABLES.keys()) == ['a', 'b']

    assert sorted(orca._TABLES.keys()) == ['a']


def test_is_expression():
    assert orca.is_expression('name') is False
    assert orca.is_expression('table.column') is True


def test_eval_variable(df):
    orca.add_injectable('x', 3)
    assert orca.eval_variable('x') == 3

    @orca.injectable()
    def func(x):
        return 'xyz' * x
    assert orca.eval_variable('func') == 'xyzxyzxyz'
    assert orca.eval_variable('func', x=2) == 'xyzxyz'

    @orca.table()
    def table(x):
        return df * x
    pdt.assert_series_equal(orca.eval_variable('table.a'), df.a * 3)


def test_eval_step(df):
    orca.add_injectable('x', 3)

    @orca.step()
    def step(x):
        return df * x

    pdt.assert_frame_equal(orca.eval_step('step'), df * 3)
    pdt.assert_frame_equal(orca.eval_step('step', x=5), df * 5)


def test_always_dataframewrapper(df):
    @orca.table()
    def table():
        return df / 2

    @orca.table()
    def table2(table):
        assert isinstance(table, orca.DataFrameWrapper)
        return table.to_frame() / 2

    result = orca.eval_variable('table2')
    pdt.assert_frame_equal(result.to_frame(), df / 4)


def test_table_func_source_data(df):
    @orca.table()
    def table():
        return df * 2

    t = orca.get_raw_table('table')
    filename, lineno, source = t.func_source_data()

    assert filename.endswith('test_orca.py')
    assert isinstance(lineno, int)
    assert 'return df * 2' in source


def test_column_type(df):
    orca.add_table('test_frame', df)

    @orca.table()
    def test_func():
        return df

    s = pd.Series(range(len(df)), index=df.index)

    def col_func():
        return s

    orca.add_column('test_frame', 'col_series', s)
    orca.add_column('test_func', 'col_series', s)
    orca.add_column('test_frame', 'col_func', col_func)
    orca.add_column('test_func', 'col_func', col_func)

    tframe = orca.get_raw_table('test_frame')
    tfunc = orca.get_raw_table('test_func')

    assert tframe.column_type('a') == 'local'
    assert tframe.column_type('col_series') == 'series'
    assert tframe.column_type('col_func') == 'function'

    assert tfunc.column_type('a') == 'local'
    assert tfunc.column_type('col_series') == 'series'
    assert tfunc.column_type('col_func') == 'function'


def test_get_raw_column(df):
    orca.add_table('test_frame', df)

    s = pd.Series(range(len(df)), index=df.index)

    def col_func():
        return s

    orca.add_column('test_frame', 'col_series', s)
    orca.add_column('test_frame', 'col_func', col_func)

    assert isinstance(
        orca.get_raw_column('test_frame', 'col_series'),
        orca._SeriesWrapper)
    assert isinstance(
        orca.get_raw_column('test_frame', 'col_func'),
        orca._ColumnFuncWrapper)


def test_column_func_source_data(df):
    orca.add_table('test_frame', df)

    @orca.column('test_frame')
    def col_func():
        return pd.Series(range(len(df)), index=df.index)

    s = orca.get_raw_column('test_frame', 'col_func')
    filename, lineno, source = s.func_source_data()

    assert filename.endswith('test_orca.py')
    assert isinstance(lineno, int)
    assert 'def col_func():' in source


def test_is_injectable():
    orca.add_injectable('answer', 42)
    assert orca.is_injectable('answer') is True
    assert orca.is_injectable('nope') is False


def test_injectable_type():
    orca.add_injectable('answer', 42)

    @orca.injectable()
    def inj1():
        return 42

    @orca.injectable(autocall=False, memoize=True)
    def power(x):
        return 42 ** x

    assert orca.injectable_type('answer') == 'variable'
    assert orca.injectable_type('inj1') == 'function'
    assert orca.injectable_type('power') == 'function'


def test_get_injectable_func_source_data():
    @orca.injectable()
    def inj1():
        return 42

    @orca.injectable(autocall=False, memoize=True)
    def power(x):
        return 42 ** x

    def inj2():
        return 'orca'

    orca.add_injectable('inj2', inj2, autocall=False)

    filename, lineno, source = orca.get_injectable_func_source_data('inj1')
    assert filename.endswith('test_orca.py')
    assert isinstance(lineno, int)
    assert '@orca.injectable()' in source

    filename, lineno, source = orca.get_injectable_func_source_data('power')
    assert filename.endswith('test_orca.py')
    assert isinstance(lineno, int)
    assert '@orca.injectable(autocall=False, memoize=True)' in source

    filename, lineno, source = orca.get_injectable_func_source_data('inj2')
    assert filename.endswith('test_orca.py')
    assert isinstance(lineno, int)
    assert 'def inj2()' in source

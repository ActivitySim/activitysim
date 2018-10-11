# Orca
# Copyright (C) 2016 UrbanSim Inc.
# See full license in LICENSE.

import pandas as pd
import pytest

from .. import orca
from ..utils.testing import assert_frames_equal


def setup_function(func):
    orca.clear_all()


def teardown_function(func):
    orca.clear_all()


@pytest.fixture
def dfa():
    return orca.DataFrameWrapper('a', pd.DataFrame(
        {'a1': [1, 2, 3],
         'a2': [4, 5, 6],
         'a3': [7, 8, 9]},
        index=['aa', 'ab', 'ac']))


@pytest.fixture
def dfz():
    return orca.DataFrameWrapper('z', pd.DataFrame(
        {'z1': [90, 91],
         'z2': [92, 93],
         'z3': [94, 95],
         'z4': [96, 97],
         'z5': [98, 99]},
        index=['za', 'zb']))


@pytest.fixture
def dfb():
    return orca.DataFrameWrapper('b', pd.DataFrame(
        {'b1': range(10, 15),
         'b2': range(15, 20),
         'a_id': ['ac', 'ac', 'ab', 'aa', 'ab'],
         'z_id': ['zb', 'zb', 'za', 'za', 'zb']},
        index=['ba', 'bb', 'bc', 'bd', 'be']))


@pytest.fixture
def dfc():
    return orca.DataFrameWrapper('c', pd.DataFrame(
        {'c1': range(20, 30),
         'c2': range(30, 40),
         'b_id': ['ba', 'bd', 'bb', 'bc', 'bb', 'ba', 'bb', 'bc', 'bd', 'bb']},
        index=['ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj']))


@pytest.fixture
def dfg():
    return orca.DataFrameWrapper('g', pd.DataFrame(
        {'g1': [1, 2, 3]},
        index=['ga', 'gb', 'gc']))


@pytest.fixture
def dfh():
    return orca.DataFrameWrapper('h', pd.DataFrame(
        {'h1': range(10, 15),
         'g_id': ['ga', 'gb', 'gc', 'ga', 'gb']},
        index=['ha', 'hb', 'hc', 'hd', 'he']))


def all_broadcasts():
    orca.broadcast('a', 'b', cast_index=True, onto_on='a_id')
    orca.broadcast('z', 'b', cast_index=True, onto_on='z_id')
    orca.broadcast('b', 'c', cast_index=True, onto_on='b_id')
    orca.broadcast('g', 'h', cast_index=True, onto_on='g_id')


def test_recursive_getitem():
    assert orca._recursive_getitem({'a': {}}, 'a') == {'a': {}}
    assert orca._recursive_getitem(
        {'a': {'b': {'c': {'d': {}, 'e': {}}}}}, 'e') == {'d': {}, 'e': {}}

    with pytest.raises(KeyError):
        orca._recursive_getitem({'a': {'b': {'c': {'d': {}, 'e': {}}}}}, 'f')


def test_dict_value_to_pairs():
    assert sorted(orca._dict_value_to_pairs({'c': {'a': 1, 'b': 2}}),
                  key=lambda d: next(iter(d))) == \
        [{'a': 1}, {'b': 2}]


def test_is_leaf_node():
    assert orca._is_leaf_node({'b': {'a': {}}}) is False
    assert orca._is_leaf_node({'a': {}}) is True


def test_next_merge():
    assert orca._next_merge({'d': {'c': {}, 'b': {'a': {}}}}) == \
        {'b': {'a': {}}}
    assert orca._next_merge({'b': {'a': {}, 'z': {}}}) == \
        {'b': {'a': {}, 'z': {}}}


def test_merge_tables_raises(dfa, dfz, dfb, dfg, dfh):
    all_broadcasts()

    with pytest.raises(RuntimeError):
        orca.merge_tables('b', [dfa, dfb, dfz, dfg, dfh])


def test_merge_tables1(dfa, dfz, dfb):
    all_broadcasts()

    merged = orca.merge_tables('b', [dfa, dfz, dfb])

    expected = pd.merge(
        dfa.to_frame(), dfb.to_frame(), left_index=True, right_on='a_id')
    expected = pd.merge(
        expected, dfz.to_frame(), left_on='z_id', right_index=True)

    assert_frames_equal(merged, expected)


def test_merge_tables2(dfa, dfz, dfb, dfc):
    all_broadcasts()

    merged = orca.merge_tables(dfc, [dfa, dfz, dfb, dfc])

    expected = pd.merge(
        dfa.to_frame(), dfb.to_frame(), left_index=True, right_on='a_id')
    expected = pd.merge(
        expected, dfz.to_frame(), left_on='z_id', right_index=True)
    expected = pd.merge(
        expected, dfc.to_frame(), left_index=True, right_on='b_id')

    assert_frames_equal(merged, expected)


def test_merge_tables_cols(dfa, dfz, dfb, dfc):
    all_broadcasts()

    merged = orca.merge_tables(
        'c', [dfa, dfz, dfb, dfc], columns=['a1', 'b1', 'z1', 'c1'])

    expected = pd.DataFrame(
        {'c1': range(20, 30),
         'b1': [10, 13, 11, 12, 11, 10, 11, 12, 13, 11],
         'a1': [3, 1, 3, 2, 3, 3, 3, 2, 1, 3],
         'z1': [91, 90, 91, 90, 91, 91, 91, 90, 90, 91]},
        index=['ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj'])

    assert_frames_equal(merged, expected)


def test_merge_tables3():
    df_a = pd.DataFrame(
        {'a': [0, 1]},
        index=['a0', 'a1'])
    df_b = pd.DataFrame(
        {'b': [2, 3, 4, 5, 6],
         'a_id': ['a0', 'a1', 'a1', 'a0', 'a1']},
        index=['b0', 'b1', 'b2', 'b3', 'b4'])
    df_c = pd.DataFrame(
        {'c': [7, 8, 9]},
        index=['c0', 'c1', 'c2'])
    df_d = pd.DataFrame(
        {'d': [10, 11, 12, 13, 15, 16, 16, 17, 18, 19],
         'b_id': ['b2', 'b0', 'b3', 'b3', 'b1', 'b4', 'b1', 'b4', 'b3', 'b3'],
         'c_id': ['c0', 'c1', 'c1', 'c0', 'c0', 'c2', 'c1', 'c2', 'c1', 'c2']},
        index=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])

    orca.add_table('a', df_a)
    orca.add_table('b', df_b)
    orca.add_table('c', df_c)
    orca.add_table('d', df_d)

    orca.broadcast(cast='a', onto='b', cast_index=True, onto_on='a_id')
    orca.broadcast(cast='b', onto='d', cast_index=True, onto_on='b_id')
    orca.broadcast(cast='c', onto='d', cast_index=True, onto_on='c_id')

    df = orca.merge_tables(target='d', tables=['a', 'b', 'c', 'd'])

    expected = pd.merge(df_a, df_b, left_index=True, right_on='a_id')
    expected = pd.merge(expected, df_d, left_index=True, right_on='b_id')
    expected = pd.merge(df_c, expected, left_index=True, right_on='c_id')

    assert_frames_equal(df, expected)


def test_merge_tables_dup_columns():
    # I'm intentionally setting the zone-ids to something different when joined
    # in a real case they'd likely be the same but the whole point of this
    # test is to see if we can get them back with different names tied to each
    # table and they need to be different to test if that's working
    hh_df = pd.DataFrame({'zone_id': [1, 1, 2], 'building_id': [5, 5, 6]})
    orca.add_table('households', hh_df)

    bldg_df = pd.DataFrame(
        {'zone_id': [2, 3], 'parcel_id': [0, 1]}, index=[5, 6])
    orca.add_table('buildings', bldg_df)

    parcels_df = pd.DataFrame({'zone_id': [4, 5]}, index=[0, 1])
    orca.add_table('parcels', parcels_df)

    orca.broadcast(
        'buildings', 'households', cast_index=True, onto_on='building_id')
    orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')

    df = orca.merge_tables(
        target='households', tables=['households', 'buildings', 'parcels'])

    expected = pd.DataFrame(
        {'building_id': [5, 5, 6], 'parcel_id': [0, 0, 1], 'zone_id': [1, 1, 2]})
    assert_frames_equal(df, expected)

    df = orca.merge_tables(
        target='households',
        tables=['households', 'buildings', 'parcels'],
        drop_intersection=False)

    expected = pd.DataFrame({
        'building_id': [5, 5, 6],
        'parcel_id': [0, 0, 1],
        'zone_id_households': [1, 1, 2],
        'zone_id_buildings': [2, 2, 3],
        'zone_id_parcels': [4, 4, 5]
    })
    assert_frames_equal(df, expected)

    df = orca.merge_tables(
        target='households',
        tables=['households', 'buildings'],
        drop_intersection=False)

    expected = pd.DataFrame({
        'building_id': [5, 5, 6],
        'parcel_id': [0, 0, 1],
        'zone_id_households': [1, 1, 2],
        'zone_id_buildings': [2, 2, 3]
    })
    assert_frames_equal(df, expected)

    df = orca.merge_tables(
        target='households',
        tables=['households', 'buildings']
    )

    expected = pd.DataFrame({
        'building_id': [5, 5, 6],
        'parcel_id': [0, 0, 1],
        'zone_id': [1, 1, 2]
    })
    assert_frames_equal(df, expected)

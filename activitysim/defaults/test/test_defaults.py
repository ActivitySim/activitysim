# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.


import pytest
import os
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
import urbansim.sim.simulation as sim
from .. import __init__
from ..tables import size_terms
import yaml


# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 50


@pytest.fixture(scope="module")
def store(request):
    store = pd.HDFStore(
        os.path.join(os.path.dirname(__file__), 'test.h5'), "r")

    def fin():
        store.close()
    request.addfinalizer(fin)

    return store


@sim.injectable(cache=False)
def settings(configs_dir):
    with open(os.path.join(configs_dir, "configs", "settings.yaml")) as f:
        obj = yaml.load(f)
        obj['households_sample_size'] = HOUSEHOLDS_SAMPLE_SIZE
        return obj


def set_random_seed():
    np.random.seed(0)


def test_size_term():
    data = {
        'a': [1, 2, 3],
        'b': [2, 3, 4]
    }
    coeffs = {
        'a': 1,
        'b': 2
    }
    answer = {
        0: 5,
        1: 8,
        2: 11
    }
    s = size_terms.size_term(pd.DataFrame(data), pd.Series(coeffs))
    pdt.assert_series_equal(s, pd.Series(answer))


def test_run(store):
    sim.add_injectable("configs_dir",
                       os.path.join(os.path.dirname(__file__), '..', '..',
                                    '..', 'example'))

    sim.add_injectable("store", store)

    sim.add_injectable("nonmotskm_matrix", np.ones((1454, 1454)))
    sim.add_injectable("set_random_seed", set_random_seed)

    # grab some of the tables
    sim.get_table("land_use").to_frame().info()
    sim.get_table("households").to_frame().info()
    sim.get_table("persons").to_frame().info()

    assert len(sim.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    # run the models in the expected order
    sim.run(["workplace_location_simulate"])
    sim.run(["auto_ownership_simulate"])
    sim.run(["cdap_simulate"])
    sim.run(['mandatory_tour_frequency'])
    sim.get_table("mandatory_tours").tour_type.value_counts()
    sim.run(['non_mandatory_tour_frequency'])
    sim.get_table("non_mandatory_tours").tour_type.value_counts()
    sim.run(["destination_choice"])
    sim.run(["mandatory_scheduling"])
    sim.run(["non_mandatory_scheduling"])
    sim.run(["mode_choice_simulate"])

    sim.clear_cache()


def test_mini_run(store):
    sim.add_injectable("configs_dir",
                       os.path.join(os.path.dirname(__file__)))

    sim.add_injectable("store", store)

    sim.add_injectable("nonmotskm_matrix", np.ones((1454, 1454)))
    sim.add_injectable("set_random_seed", set_random_seed)

    assert len(sim.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    # run the models in the expected order
    sim.run(["workplace_location_simulate"])
    sim.run(["auto_ownership_simulate"])

    # this is a regression test so that we know if these numbers change
    auto_choice = sim.get_table('households').get_column('auto_ownership')
    pdt.assert_series_equal(
        auto_choice[[2306822, 652072, 651907]],
        pd.Series(
            [2, 1, 1], index=[2306822, 652072, 651907]))

    sim.run(["cdap_simulate"])

    sim.run(['mandatory_tour_frequency'])

    mtf_choice = sim.get_table('persons').get_column(
        'mandatory_tour_frequency')
    print mtf_choice
    pdt.assert_series_equal(
        mtf_choice[[642922, 6203845, 642921]],
        pd.Series(
            ['work1', 'work1', 'school2'],
            index=[642922, 6203845, 642921]))

    sim.clear_cache()

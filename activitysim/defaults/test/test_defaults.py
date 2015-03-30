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


@pytest.fixture(scope="module")
def store(request):
    store = pd.HDFStore(
        os.path.join(os.path.dirname(__file__), 'test.h5'), "r")

    def fin():
        store.close()
    request.addfinalizer(fin)

    return store


def test_run(store):
    sim.add_injectable("configs_dir",
                       os.path.join(os.path.dirname(__file__), '..', '..',
                                    '..', 'example'))

    sim.add_injectable("store", store)

    sim.add_injectable("nonmotskm_matrix", np.ones((1454, 1454)))

    # grab some of the tables
    sim.get_table("land_use").to_frame().info()
    sim.get_table("households").to_frame().info()
    sim.get_table("persons").to_frame().info()

    # run the models in the expected order
    sim.run(["workplace_location_simulate"])
    sim.run(["auto_ownership_simulate"])
    sim.run(['mandatory_tour_frequency'])
    sim.get_table("mandatory_tours").tour_type.value_counts()
    sim.run(['non_mandatory_tour_frequency'])
    sim.get_table("non_mandatory_tours").tour_type.value_counts()
    sim.run(["destination_choice"])
    sim.run(["mandatory_scheduling"])
    sim.run(["non_mandatory_scheduling"])
    sim.run(["mode_choice_simulate"])

    sim.clear_cache()


def set_random_seed():
    np.random.seed(0)


def test_mini_run(store):
    sim.add_injectable("configs_dir",
                       os.path.join(os.path.dirname(__file__)))

    sim.add_injectable("store", store)

    sim.add_injectable("nonmotskm_matrix", np.ones((1454, 1454)))
    sim.add_injectable("set_random_seed", set_random_seed)

    # run the models in the expected order
    sim.run(["workplace_location_simulate"])
    sim.run(["auto_ownership_simulate"])

    # this is a regression test so that we know if these numbers change
    auto_choice = sim.get_table('households').get_column('auto_ownership')
    pdt.assert_series_equal(
        auto_choice[[2306822, 652072, 2542402, 651907, 788657]],
        pd.Series(
            [4, 1, 2, 1, 1], index=[2306822, 652072, 2542402, 651907, 788657]))

    sim.run(['mandatory_tour_frequency'])

    mtf_choice = sim.get_table('persons').get_column(
        'mandatory_tour_frequency')
    pdt.assert_series_equal(
        mtf_choice[[1015, 9998, 28064, 29035, 34241]],
        pd.Series(
            ['school2', 'work2', 'school1', 'work1', 'work1'],
            index=[1015, 9998, 28064, 29035, 34241]))

    sim.clear_cache()

# ActivitySim
# See full license in LICENSE.txt.

import os
import tempfile

import numpy as np
import orca
import pandas as pd
import pandas.util.testing as pdt
import pytest
import yaml
import openmatrix as omx

from .. import __init__
from ..tables import size_terms


# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100


@pytest.fixture(scope="module")
def store(request):
    store = pd.HDFStore(
        os.path.join(os.path.dirname(__file__), 'mtc_asim.h5'), "r")

    def fin():
        store.close()
    request.addfinalizer(fin)

    return store


@pytest.fixture(scope="module")
def omx_file(request):
    omx_file = omx.openFile(os.path.join(os.path.dirname(__file__), "nonmotskm.omx"))

    def fin():
        omx_file.close()
    request.addfinalizer(fin)

    return omx_file


@orca.injectable(cache=False)
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


def test_mini_run(store, omx_file, random_seed):
    orca.add_injectable("configs_dir",
                        os.path.join(os.path.dirname(__file__)))

    orca.add_injectable("omx_file", omx_file)

    orca.add_injectable("store", store)

    orca.add_injectable("set_random_seed", set_random_seed)

    assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    # run the models in the expected order
    orca.run(["workplace_location_simulate"])
    orca.run(["auto_ownership_simulate"])

    # this is a regression test so that we know if these numbers change
    auto_choice = orca.get_table('households').get_column('auto_ownership')

    hh_ids = [1161386, 2666136, 461593]
    choices = [1, 2, 1]
    print "auto_choice\n", auto_choice.head(10)
    pdt.assert_series_equal(
        auto_choice[hh_ids],
        pd.Series(choices, index=pd.Index(hh_ids, name="HHID")))

    orca.run(["cdap_simulate"])
    orca.run(['mandatory_tour_frequency'])

    mtf_choice = orca.get_table('persons').get_column('mandatory_tour_frequency')
    per_ids = [24693, 93217, 297614]
    choices = ['work1', 'school2', 'work2']
    print "mtf_choice\n", mtf_choice.head(20)
    pdt.assert_series_equal(
        mtf_choice[per_ids],
        pd.Series(choices, index=pd.Index(per_ids, name='PERID')))
    orca.clear_cache()


def test_full_run(store, omx_file):
    orca.add_injectable("configs_dir",
                        os.path.join(os.path.dirname(__file__), '..', '..',
                                     '..', 'example'))

    orca.add_injectable("omx_file", omx_file)

    orca.add_injectable("store", store)

    orca.add_injectable("set_random_seed", set_random_seed)

    # grab some of the tables
    orca.get_table("land_use").to_frame().info()
    orca.get_table("households").to_frame().info()
    orca.get_table("persons").to_frame().info()

    assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    # run the models in the expected order
    orca.run(["school_location_simulate"])
    orca.run(["workplace_location_simulate"])
    orca.run(["auto_ownership_simulate"])
    orca.run(["cdap_simulate"])
    orca.run(['mandatory_tour_frequency'])
    orca.get_table("mandatory_tours").tour_type.value_counts()
    orca.run(['non_mandatory_tour_frequency'])
    orca.get_table("non_mandatory_tours").tour_type.value_counts()
    orca.run(["destination_choice"])
    orca.run(["mandatory_scheduling"])
    orca.run(["non_mandatory_scheduling"])
    orca.run(["patch_mandatory_tour_destination"])
    orca.run(["tour_mode_choice_simulate"])
    orca.run(["trip_mode_choice_simulate"])

    orca.clear_cache()

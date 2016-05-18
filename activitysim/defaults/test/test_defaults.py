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


def inject_settings(configs_dir, households_sample_size, preload_3d_skims=None, chunk_size=None):

    with open(os.path.join(configs_dir, "configs", "settings.yaml")) as f:
        settings = yaml.load(f)
        settings['households_sample_size'] = households_sample_size
        if preload_3d_skims is not None:
            settings['preload_3d_skims'] = preload_3d_skims
        if chunk_size is not None:
            settings['chunk_size'] = chunk_size

    orca.add_injectable("settings", settings)


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

    configs_dir = os.path.join(os.path.dirname(__file__))
    orca.add_injectable("configs_dir", configs_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    orca.add_injectable("omx_file", omx_file)

    orca.add_injectable("store", store)

    orca.add_injectable("set_random_seed", set_random_seed)

    orca.clear_cache()

    assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    # run the models in the expected order
    orca.run(["workplace_location_simulate"])
    orca.run(["auto_ownership_simulate"])

    # this is a regression test so that we know if these numbers change
    auto_choice = orca.get_table('households').get_column('auto_ownership')

    hh_ids = [2124015, 961042, 1583271]
    choices = [1, 2, 2]
    print "auto_choice\n", auto_choice.head(3)
    pdt.assert_series_equal(
        auto_choice[hh_ids],
        pd.Series(choices, index=pd.Index(hh_ids, name="HHID")))

    orca.run(["cdap_simulate"])
    orca.run(['mandatory_tour_frequency'])

    mtf_choice = orca.get_table('persons').get_column('mandatory_tour_frequency')
    per_ids = [172616, 172781, 172782]
    choices = ['work1', 'school1', 'work_and_school']
    print "mtf_choice\n", mtf_choice.head(20)
    pdt.assert_series_equal(
        mtf_choice[per_ids],
        pd.Series(choices, index=pd.Index(per_ids, name='PERID')))
    orca.clear_cache()


def full_run(store, omx_file, preload_3d_skims, chunk_size=0):

    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example')
    orca.add_injectable("configs_dir", configs_dir)

    inject_settings(configs_dir,
                    households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
                    preload_3d_skims=preload_3d_skims,
                    chunk_size=chunk_size)

    orca.add_injectable("omx_file", omx_file)
    orca.add_injectable("store", store)
    orca.add_injectable("set_random_seed", set_random_seed)

    orca.clear_cache()

    # grab some of the tables
    orca.get_table("land_use").to_frame().info()
    orca.get_table("households").to_frame().info()
    orca.get_table("persons").to_frame().info()

    assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE
    assert orca.get_injectable("chunk_size") == chunk_size

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

    tours_merged = orca.get_table("tours_merged").to_frame()

    tour_count = len(tours_merged.index)

    orca.clear_cache()

    return tour_count


def test_full_run(store, omx_file):

    tour_count = full_run(store, omx_file, preload_3d_skims=False)

    assert(tour_count == 183)


def test_full_run_with_preload_skims(store, omx_file):

    tour_count = full_run(store, omx_file, preload_3d_skims=True)

    assert(tour_count == 183)


def test_full_run_with_chunks(store, omx_file):

    tour_count = full_run(store, omx_file, preload_3d_skims=True, chunk_size=10)

    # FIXME - different sampling causes slightly different results?
    assert(tour_count == 187)

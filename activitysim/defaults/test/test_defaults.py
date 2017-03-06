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
from . import extensions

from ... import tracing
from ... import pipeline

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100

SKIP_FULL_RUN = True


def inject_settings(configs_dir, households_sample_size, chunk_size=None,
                    trace_hh_id=None, trace_od=None, check_for_variability=None):

    with open(os.path.join(configs_dir, 'settings.yaml')) as f:
        settings = yaml.load(f)
        settings['households_sample_size'] = households_sample_size
        if chunk_size is not None:
            settings['chunk_size'] = chunk_size
        if trace_hh_id is not None:
            settings['trace_hh_id'] = trace_hh_id
        if trace_od is not None:
            settings['trace_od'] = trace_od
        if check_for_variability is not None:
            settings['check_for_variability'] = check_for_variability

    orca.add_injectable("settings", settings)


def set_random_seed():
    pipeline.get_rn_generator().reseed_global_prng(offset=0)


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


def test_mini_run(random_seed):

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    orca.add_injectable("set_random_seed", set_random_seed)

    orca.clear_cache()

    assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    # run the models in the expected order
    orca.run(["compute_accessibility"])
    orca.run(["school_location_simulate"])
    orca.run(["workplace_location_simulate"])
    orca.run(["auto_ownership_simulate"])

    # this is a regression test so that we know if these numbers change
    auto_choice = orca.get_table('households').get_column('auto_ownership')

    hh_ids = [2124015, 961042, 1583271]
    choices = [1, 1, 1]
    print "auto_choice\n", auto_choice.head(3)
    pdt.assert_series_equal(
        auto_choice[hh_ids],
        pd.Series(choices, index=pd.Index(hh_ids, name="HHID")))

    orca.run(["cdap_simulate"])
    orca.run(['mandatory_tour_frequency'])

    mtf_choice = orca.get_table('persons').get_column('mandatory_tour_frequency')
    per_ids = [326914, 172781, 298898]
    choices = ['school1', 'work_and_school', 'work2']
    print "mtf_choice\n", mtf_choice.head(20)
    pdt.assert_series_equal(
        mtf_choice[per_ids],
        pd.Series(choices, index=pd.Index(per_ids, name='PERID')))
    orca.clear_cache()


def full_run(chunk_size=0,
             households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
             trace_hh_id=None, trace_od=None, check_for_variability=None):

    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    inject_settings(configs_dir,
                    households_sample_size=households_sample_size,
                    chunk_size=chunk_size,
                    trace_hh_id=trace_hh_id,
                    trace_od=trace_od,
                    check_for_variability=check_for_variability)

    orca.add_injectable("set_random_seed", set_random_seed)

    orca.clear_cache()

    tracing.config_logger()

    # grab some of the tables
    # orca.get_table("land_use").to_frame().info()
    # orca.get_table("households").to_frame().info()
    # orca.get_table("persons").to_frame().info()

    assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE
    assert orca.get_injectable("chunk_size") == chunk_size

    # run the models in the expected order
    orca.run(["compute_accessibility"])
    orca.run(["school_location_simulate"])
    orca.run(["workplace_location_simulate"])
    orca.run(["auto_ownership_simulate"])
    orca.run(["cdap_simulate"])
    orca.run(['mandatory_tour_frequency'])
    orca.run(['non_mandatory_tour_frequency'])
    orca.run(["destination_choice"])
    orca.run(["mandatory_scheduling"])
    orca.run(["non_mandatory_scheduling"])
    orca.run(["tour_mode_choice_simulate"])
    orca.run(["trip_mode_choice_simulate"])

    tours_merged = orca.get_table("tours_merged").to_frame()

    tour_count = len(tours_merged.index)

    orca.clear_cache()

    return tour_count


def test_full_run():

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(check_for_variability=True)
    assert(tour_count == 230)


def test_full_run_with_chunks():

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(chunk_size=10)

    # different sampling causes slightly different results
    assert(tour_count == 219)


def test_full_run_with_hh_trace():

    if SKIP_FULL_RUN:
        return

    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    hh_fname = os.path.join(output_dir, 'households.csv')
    accessibility_fname = os.path.join(output_dir, 'accessibility.csv')
    goner_fname = os.path.join(output_dir, 'x.csv')

    df = pd.DataFrame(np.random.randn(8, 2), columns=['A', 'B'])
    df.to_csv(hh_fname)
    df.to_csv(goner_fname)
    assert os.path.isfile(goner_fname)

    HH_ID = 961042
    OD = [5, 11]

    tour_count = full_run(trace_hh_id=HH_ID, trace_od=OD)

    assert(tour_count == 230)

    # should delete any csv files from output
    assert not os.path.isfile(goner_fname)

    # should have created household csv trace file
    # first row should be HHID
    h = pd.read_csv(hh_fname)
    assert h.iloc[0][0] == 'HHID'
    assert h.iloc[0][1] == HH_ID

    h = pd.read_csv(os.path.join(output_dir, 'workplace_location.interaction_simulate.choices.csv'))
    assert h.columns[0] == 'PERID'
    assert h.columns[1] == 'workplace_location'
    assert h.iloc[0][0] == 1888694
    assert h.iloc[0][1] == 17

    # should have created accessibility csv trace file
    a = pd.read_csv(accessibility_fname)
    assert a.iloc[1][0] == 'dest'
    assert int(a.iloc[1][1]) == OD[1]
    assert a.iloc[2][0] == 'orig'
    assert int(a.iloc[2][1]) == OD[0]

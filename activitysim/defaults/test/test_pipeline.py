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


def test_mini_pipeline_run():

    pipeline.get_rn_generator().set_base_seed(0)

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    orca.clear_cache()

    # assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    _MODELS = [
        'compute_accessibility',
        'school_location_simulate',
        'workplace_location_simulate',
        'auto_ownership_simulate'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    auto_choice = pipeline.get_table("households").auto_ownership

    hh_ids = [2124015, 961042, 1583271]
    choices = [2, 0, 2]
    print "auto_choice\n", auto_choice.head(3)
    pdt.assert_series_equal(
        auto_choice[hh_ids],
        pd.Series(choices, index=pd.Index(hh_ids, name="HHID"), name='auto_ownership'))

    pipeline.run_model('cdap_simulate')
    pipeline.run_model('mandatory_tour_frequency')

    mtf_choice = pipeline.get_table("persons").mandatory_tour_frequency

    per_ids = [326914, 172781, 298898]
    choices = ['school1', 'school1', 'work1']
    print "mtf_choice\n", mtf_choice.head(20)
    pdt.assert_series_equal(
        mtf_choice[per_ids],
        pd.Series(choices, index=pd.Index(per_ids, name='PERID'), name='mandatory_tour_frequency'))

    pipeline.close()

    orca.clear_cache()


def test_mini_pipeline_run2():

    pipeline.get_rn_generator().set_base_seed(0)

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    orca.clear_cache()

    # assert len(orca.get_table("households").index) == HOUSEHOLDS_SAMPLE_SIZE

    pipeline.start_pipeline('auto_ownership_simulate')

    auto_choice = pipeline.get_table("households").auto_ownership

    hh_ids = [2124015, 961042, 1583271]
    choices = [2, 0, 2]
    print "auto_choice\n", auto_choice.head(3)
    pdt.assert_series_equal(
        auto_choice[hh_ids],
        pd.Series(choices, index=pd.Index(hh_ids, name="HHID"), name='auto_ownership'))

    pipeline.run_model('cdap_simulate')
    pipeline.run_model('mandatory_tour_frequency')

    mtf_choice = pipeline.get_table("persons").mandatory_tour_frequency

    per_ids = [326914, 172781, 298898]
    choices = ['school1', 'school1', 'work1']
    print "mtf_choice\n", mtf_choice.head(20)
    pdt.assert_series_equal(
        mtf_choice[per_ids],
        pd.Series(choices, index=pd.Index(per_ids, name='PERID'), name='mandatory_tour_frequency'))

    pipeline.close()

    orca.clear_cache()

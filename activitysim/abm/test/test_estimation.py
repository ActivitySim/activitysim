import os
import larch  # !conda install larch #for estimation
import larch.util.activitysim as la
import pandas as pd
import pytest
import yaml
import numpy as np

from pytest import approx

@pytest.fixture(scope='module')
def est_data():
    # !activitysim create -e example_estimation_sf -d _test_est
    if os.path.exists("_test_est"):
        retain_test_data = True
    else:
        retain_test_data = False
        from activitysim.cli.create import get_example
        get_example("example_estimation_sf", "_test_est")

    # %cd _test_est
    cwd = os.getcwd()
    os.chdir("_test_est")

    # !activitysim run -c configs_estimation/configs -c configs -o output -d data_sf
    if not retain_test_data:
        import subprocess
        cp = subprocess.run([
            'activitysim', 'run',
            '-c', 'configs_estimation/configs',
            '-c', 'configs',
            '-o', 'output',
            '-d', 'data_sf',
        ], capture_output=True)

    yield

    os.chdir(cwd)
    if not retain_test_data:
        os.remove("_test_est")


def _regression_check(dataframe_regression, df):
    dataframe_regression.check(df.select_dtypes('number').clip(-9e9,9e9))


def test_auto_ownership(est_data, dataframe_regression):
    from larch.util.activitysim.auto_ownership import auto_ownership_model
    m = auto_ownership_model()
    m.load_data()
    assert m.loglike() == approx(-1745.4932922174953)
    r = m.maximize_loglike()
    assert r.loglike == approx(-1724.943397053043)
    _regression_check(dataframe_regression, m.pf)


def test_workplace_location(est_data, dataframe_regression):
    from larch.util.activitysim.location_choice import location_choice_model
    m = location_choice_model(model_selector='workplace')
    m.load_data()
    assert m.loglike() == approx(-13412.269435487342)
    r = m.maximize_loglike(method='SLSQP')
    assert r.loglike == approx(-12624.992531894903)
    _regression_check(dataframe_regression, m.pf)


def test_school_location(est_data, dataframe_regression):
    from larch.util.activitysim.location_choice import location_choice_model
    m = location_choice_model(model_selector='school')
    m.load_data()
    assert m.loglike() == approx(-3729.5884682769906)
    r = m.maximize_loglike(method='BHHH')
    assert r.loglike == approx(-3097.6915329312333)
    _regression_check(dataframe_regression, m.pf)


def test_tour_mode_choice(est_data, dataframe_regression):
    from larch.util.activitysim.tour_mode_choice import tour_mode_choice_model
    m = tour_mode_choice_model()
    m.load_data()
    m.doctor(repair_ch_av='-')
    assert m.loglike() == approx(-6069.594743230453)
    r = m.maximize_loglike(method='SLSQP', options={'maxiter': 1000})
    assert r.loglike == approx(-5877.503582602247)
    _regression_check(dataframe_regression, m.pf)


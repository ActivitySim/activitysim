import os
import larch  # !conda install larch #for estimation
import larch.util.activitysim as la
import pandas as pd
import pytest
import yaml
import numpy as np
import subprocess

from pytest import approx
from activitysim.cli.create import get_example


@pytest.fixture(scope="module")
def est_data():
    # !activitysim create -e example_estimation_sf -d _test_est
    if os.path.exists("_test_est"):
        retain_test_data = True
    else:
        retain_test_data = False
        get_example("example_estimation_sf", "_test_est")

    # %cd _test_est
    cwd = os.getcwd()
    os.chdir("_test_est")

    # !activitysim run -c configs_estimation/configs -c configs -o output -d data_sf
    if not retain_test_data:
        print(f"List of files now in {os.getcwd()}")
        subprocess.run(["find", "."])
        print(f"\n\nrunning activitysim estimation mode in {os.getcwd()}")
        subprocess.run(
            [
                "activitysim",
                "run",
                "-c",
                "configs_estimation/configs",
                "-c",
                "configs",
                "-o",
                "output",
                "-d",
                "data_sf",
            ],
        )
    else:
        print(f"reusing existing data in {os.getcwd()}")

    yield os.getcwd()

    os.chdir(cwd)
    # if not retain_test_data:
    #     os.remove("_test_est")


def _regression_check(dataframe_regression, df):
    dataframe_regression.check(df.select_dtypes("number").clip(-9e9, 9e9))


def test_auto_ownership(est_data, num_regression, dataframe_regression):
    from larch.util.activitysim.auto_ownership import auto_ownership_model

    m = auto_ownership_model()
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike()
    num_regression.check(
        {
            "loglike_prior": loglike_prior,
            "loglike_converge": r.loglike,
        },
        basename="test_auto_ownership_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_workplace_location(est_data, num_regression, dataframe_regression):
    from larch.util.activitysim.location_choice import location_choice_model

    m = location_choice_model(model_selector="workplace")
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP")
    num_regression.check(
        {
            "loglike_prior": loglike_prior,
            "loglike_converge": r.loglike,
        },
        basename="test_workplace_location_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_school_location(est_data, num_regression, dataframe_regression):
    from larch.util.activitysim.location_choice import location_choice_model

    m = location_choice_model(model_selector="school")
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="BHHH")
    num_regression.check(
        {
            "loglike_prior": loglike_prior,
            "loglike_converge": r.loglike,
        },
        basename="test_school_location_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_cdap_model(est_data, num_regression, dataframe_regression):
    from larch.util.activitysim.cdap import cdap_model

    m = cdap_model()
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000})
    num_regression.check(
        {
            "loglike_prior": loglike_prior,
            "loglike_converge": r.loglike,
        },
        basename="test_cdap_model_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_tour_mode_choice(est_data, num_regression, dataframe_regression):
    from larch.util.activitysim.tour_mode_choice import tour_mode_choice_model

    m = tour_mode_choice_model()
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000})
    num_regression.check(
        {
            "loglike_prior": loglike_prior,
            "loglike_converge": r.loglike,
        },
        basename="test_tour_mode_choice_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_nonmand_tour_freq(est_data, num_regression, dataframe_regression):
    from larch.util.activitysim.nonmand_tour_freq import nonmand_tour_freq_model

    m = nonmand_tour_freq_model()
    loglike_prior = {}
    for segment_name in m:
        m[segment_name].load_data()
        m[segment_name].doctor(repair_ch_av="-")
        loglike_prior[segment_name] = m[segment_name].loglike()
    r = {}
    for segment_name in m:
        r[segment_name] = m[segment_name].maximize_loglike(
            method="SLSQP", options={"maxiter": 1000}
        )
    loglike_priors = [value for key, value in sorted(loglike_prior.items())]
    loglike_converge = [value.loglike for key, value in sorted(r.items())]
    num_regression.check(
        {
            "loglike_prior": loglike_priors,
            "loglike_converge": loglike_converge,
        },
        basename="test_nonmand_tour_freq_loglike",
    )
    _regression_check(dataframe_regression, pd.concat([x.pf for x in m.values()]))

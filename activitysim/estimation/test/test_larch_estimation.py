import os
import pandas as pd
import pytest
import subprocess

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


def _regression_check(dataframe_regression, df, basename=None):
    dataframe_regression.check(
        df.select_dtypes("number").drop(columns=["holdfast"], errors='ignore').clip(-9e9, 9e9),
        # pandas 1.3 handles int8 dtypes as actual numbers, so holdfast needs to be dropped manually
        # we're dropping it not adding to the regression check so older pandas will also work.
        basename=basename,
        default_tolerance=dict(atol=1e-6, rtol=1e-2)
        # set a little loose, as there is sometimes a little variance in these
        # results when switching backend implementations.
    )


@pytest.mark.parametrize("name,method", [
    ("free_parking", "BHHH"),
    ("mandatory_tour_frequency", "SLSQP"),
    ("joint_tour_frequency", "SLSQP"),
    ("joint_tour_composition", "SLSQP"),
    ("joint_tour_participation", "SLSQP"),
    ("mandatory_tour_frequency", "BHHH"),
    ("atwork_subtour_frequency", "SLSQP"),
    ("auto_ownership", "BHHH"),
    ("trip_mode_choice", "SLSQP"),
])
def test_simple_simulate(est_data, num_regression, dataframe_regression, name, method):
    from activitysim.estimation.larch import component_model

    m = component_model(name)
    m.load_data()
    m.doctor(repair_ch_av='-')
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method=method, options={"maxiter": 1000})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_simple_simulate_{name}_{method}_loglike",
        default_tolerance=dict(atol=1e-6, rtol=1e-3),
    )
    _regression_check(dataframe_regression, m.pf)


@pytest.mark.parametrize("name,method", [
    ("workplace_location", "SLSQP"),
    ("school_location", "SLSQP"),
    ("non_mandatory_tour_destination", "SLSQP"),
    ("atwork_subtour_destination", "BHHH"),
    ("trip_destination", "SLSQP"),
])
def test_location_model(est_data, num_regression, dataframe_regression, name, method):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model(name, return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method=method)
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_loc_{name}_loglike",
    )
    _regression_check(dataframe_regression, m.pf)
    size_spec = update_size_spec(
        m, data, result_dir=None, output_file=None,
    )
    dataframe_regression.check(
        size_spec,
        basename=f"test_loc_{name}_size_spec",
        default_tolerance=dict(atol=1e-6, rtol=1e-2)
        # set a little loose, as there is sometimes a little variance in these
        # results when switching backend implementations.
    )


@pytest.mark.parametrize("name,method", [
    ("non_mandatory_tour_scheduling", "SLSQP"),
    ("joint_tour_scheduling", "SLSQP"),
    ("atwork_subtour_scheduling", "SLSQP"),
    ("mandatory_tour_scheduling_work", "SLSQP"),
    ("mandatory_tour_scheduling_school", "SLSQP"),
])
def test_scheduling_model(est_data, num_regression, dataframe_regression, name, method):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model(name, return_data=True)
    m.load_data()
    m.doctor(repair_ch_av='-')
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method=method)
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_{name}_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_stop_freq_model(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model

    name = "stop_frequency"
    m, data = component_model(name, return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike()
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_{name}_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_workplace_location(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model("workplace_location", return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP")
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_workplace_location_loglike",
    )
    _regression_check(dataframe_regression, m.pf)
    size_spec = update_size_spec(
        m, data, result_dir=None, output_file=None,
    )
    dataframe_regression.check(
        size_spec,
        basename="test_workplace_location_size_spec",
        default_tolerance=dict(atol=1e-6, rtol=1e-2),
    )


def test_school_location(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model("school_location", return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="BHHH")
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_school_location_loglike",
    )
    _regression_check(dataframe_regression, m.pf)
    size_spec = update_size_spec(
        m, data, result_dir=None, output_file=None,
    )
    dataframe_regression.check(
        size_spec,
        basename="test_school_location_size_spec",
        default_tolerance=dict(atol=1e-6, rtol=1e-2),
    )


def test_cdap_model(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch.cdap import cdap_model

    m = cdap_model()
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_cdap_model_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_nonmand_and_joint_tour_dest_choice(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model

    modelname = ("non_mandatory_tour_destination", "joint_tour_destination")
    m, d = component_model(modelname, return_data=True)
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_nonmand_and_joint_tour_dest_choice_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_tour_and_subtour_mode_choice(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch.mode_choice import tour_mode_choice_model, \
        atwork_subtour_mode_choice_model

    m = tour_mode_choice_model()
    s = atwork_subtour_mode_choice_model()
    m.extend(s)  # join the atwork subtour model to the master group
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_tour_mode_choice_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_nonmand_tour_freq(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch.nonmand_tour_freq import nonmand_tour_freq_model

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
        {"loglike_prior": loglike_priors, "loglike_converge": loglike_converge},
        basename="test_nonmand_tour_freq_loglike",
    )
    _regression_check(dataframe_regression, pd.concat([x.pf for x in m.values()]))

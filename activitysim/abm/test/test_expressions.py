import os

import pandas as pd
import pytest
import yaml

from activitysim.core import inject
from activitysim.abm.models.util import expressions


@pytest.fixture(scope="session")
def config_path():
    return os.path.join(os.path.dirname(__file__), 'configs_test_misc')


def test_30_minute_windows(config_path):
    with open(os.path.join(config_path, 'settings_30_min.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)

    inject.add_injectable("settings", settings)

    assert expressions.skim_time_period_label(1) == 'EA'
    assert expressions.skim_time_period_label(16) == 'AM'
    assert expressions.skim_time_period_label(24) == 'MD'
    assert expressions.skim_time_period_label(36) == 'PM'
    assert expressions.skim_time_period_label(46) == 'EV'

    pd.testing.assert_series_equal(
        expressions.skim_time_period_label(pd.Series([1, 16, 24, 36, 46])),
        pd.Series(['EA', 'AM', 'MD', 'PM', 'EV']))


def test_60_minute_windows(config_path):
    with open(os.path.join(config_path, 'settings_60_min.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)

    inject.add_injectable("settings", settings)

    assert expressions.skim_time_period_label(1) == 'EA'
    assert expressions.skim_time_period_label(8) == 'AM'
    assert expressions.skim_time_period_label(12) == 'MD'
    assert expressions.skim_time_period_label(18) == 'PM'
    assert expressions.skim_time_period_label(23) == 'EV'

    pd.testing.assert_series_equal(
        expressions.skim_time_period_label(pd.Series([1, 8, 12, 18, 23])),
        pd.Series(['EA', 'AM', 'MD', 'PM', 'EV']))


def test_1_week_time_window():
    settings = {
        'skim_time_periods': {
            'time_window': 10080,  # One Week
            'period_minutes': 1440,  # One Day
            'periods': [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7
            ],
            'labels': ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                       'Thursday', 'Friday', 'Saturday']
        }
    }

    inject.add_injectable("settings", settings)

    assert expressions.skim_time_period_label(1) == 'Sunday'
    assert expressions.skim_time_period_label(2) == 'Monday'
    assert expressions.skim_time_period_label(3) == 'Tuesday'
    assert expressions.skim_time_period_label(4) == 'Wednesday'
    assert expressions.skim_time_period_label(5) == 'Thursday'
    assert expressions.skim_time_period_label(6) == 'Friday'
    assert expressions.skim_time_period_label(7) == 'Saturday'

    weekly_series = expressions.skim_time_period_label(pd.Series([1, 2, 3, 4, 5, 6, 7]))

    pd.testing.assert_series_equal(weekly_series,
                                   pd.Series(['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                                              'Thursday', 'Friday', 'Saturday']))


def test_future_warning(config_path):
    with open(os.path.join(config_path, 'settings_60_min.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)

    settings['skim_time_periods']['hours'] = settings['skim_time_periods'].pop('periods')

    inject.add_injectable("settings", settings)

    with pytest.warns(FutureWarning) as warning_test:
        expressions.skim_time_period_label(1)

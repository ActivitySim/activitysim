import os
import pandas as pd
import pytest

from activitysim.abm.models import summarize
from .setup_utils import setup_dirs

@pytest.fixture(scope='module')
def data_path():
    return os.path.join('activitysim', 'examples', 'example_mtc', 'output')

@pytest.fixture(scope='module')
def tours(data_path):
     tour_path = os.path.join(data_path, 'final_tours.csv')
     return pd.read_csv(tour_path)


def test_summarize(tours):
    setup_dirs()
    summarize.summarize(tours)
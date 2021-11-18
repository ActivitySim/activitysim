import pytest
### import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import pipeline


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def module():
    return 'summarize'


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def tables():
    return ['tours', 'trips']


def test_summarize(initialize_pipeline):
    pipeline.run(models=['summarize'])

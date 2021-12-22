import logging

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
    return {
        'land_use': 'zone_id',
        'tours': 'tour_id',
        'trips': 'trip_id',
        'persons': 'person_id',
        'households': 'household_id',
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipeline
@pytest.fixture(scope='module')
def initialize_network_los():
    return True


def test_summarize(initialize_pipeline, caplog):
    caplog.set_level(logging.INFO)
    pipeline.run(models=['summarize'])

import os
import pandas as pd
import pytest

from activitysim.core import pipeline
from activitysim.abm.models.summarize import summarize
from .setup_utils import setup_dirs

@pytest.fixture(scope='module')
def data_path():
    return os.path.join('..', '..', '..', 'examples', 'example_mtc', 'output')

@pytest.fixture(scope='module')
def trips(data_path):
     return pd.read_csv(os.path.join(data_path, 'final_trips.csv'))

@pytest.fixture(scope='module')
def tours(data_path):
     return pd.read_csv(os.path.join(data_path, 'final_tours.csv'))

@pytest.fixture(scope='module')
def tables(trips, tours):
     return {
        'trips': trips,
        'tours': tours,
     }

# def test_summarize(tours):
#     setup_dirs()
#     summarize.summarize(tours)


# @pytest.fixture(scope='module')
# def synthesize_pipeline(tables):
#     if not pipeline.is_open():
#         try:
#             # Open the pipeline from an available h5 file
#             # (this makes sure an available h5 file isn't overwritten by a new pipeline)
#             ipeline.open_pipeline('_')
#         except:
#             # Otherwise, create a new pipeline
#             pipeline.open_pipeline()
#     test_checkpoint = 'test_summarize'
#     # Make a new test checkpoint if it doesn't exist already
#     if test_checkpoint not in pipeline.get_checkpoints().checkpoint_name:
#         pipeline.add_checkpoint('test_summarize')
#     # Otherwise, go to that checkpoint
#     else:
#         pipeline.load_checkpoint('test_summarize')
#     # Replace specified tables with inputs 
#     for name, df in tables.items():
#         pipeline.replace_table(name, df)
    

# def test_summarize(tables=None):
#     setup_dirs()
#     if tables:
#         synthesize_pipeline(tables)
#     else:
#         if not pipeline.is_open():
#             pipeline.open_pipeline('_')
#     summarize()


def test_summarize(tables):
    setup_dirs()
    summarize(tables)




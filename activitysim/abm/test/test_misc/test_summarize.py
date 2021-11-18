import os
import pandas as pd
import pytest

### import models is necessary to initalize the model steps with orca
from activitysim.abm import models

from activitysim.core import pipeline


def test_summarize():
    ### Set to return the tables as they would be after the last step in the model
    pipeline.run(models=['summarize'], resume_after='trip_mode_choice')




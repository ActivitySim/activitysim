import os
import orca
import pandas as pd
### import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import config, inject, pipeline


def test_summarize():
    tours = pd.read_csv(os.path.join('data', 'summarize', 'tours.csv'))
    trips = pd.read_csv(os.path.join('data', 'summarize', 'trips.csv'))

    orca.add_table('trips', trips)
    orca.add_table('tours', tours)
    pipeline.open_pipeline_store(overwrite=True)
    pipeline.add_checkpoint('summarize')

    ### Set to return the tables as they would be after the last step in the model
    pipeline.run(models=['summarize'])
    #pipeline._PIPELINE.pipeline_store.close()

    #pipeline_file_path = os.path.join('output', 'pipeline.h5' )
    #os.unlink(pipeline_file_path)


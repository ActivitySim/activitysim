import pandas as pd

from ..core import pipeline


def load_checkpointed_tables(
    pipeline_file_path,
    tables=None,
    checkpoint_name=None,
):
    pipeline_store = pd.HDFStore(pipeline_file_path, mode="r")

    checkpoints = pipeline_store[pipeline.CHECKPOINT_TABLE_NAME]

    # checkpoint row as series
    if checkpoint_name is None:
        checkpoint = checkpoints.iloc[-1]
        checkpoint_name = checkpoint.loc[pipeline.CHECKPOINT_NAME]
    else:
        i = checkpoints.set_index(pipeline.CHECKPOINT_NAME).index.get_loc(
            checkpoint_name
        )
        checkpoint = checkpoints.iloc[i]

    # series with table name as index and checkpoint_name as value
    checkpoint_tables = checkpoint[~checkpoint.index.isin(pipeline.NON_TABLE_COLUMNS)]

    # omit dropped tables with empty checkpoint name
    checkpoint_tables = checkpoint_tables[checkpoint_tables != ""]

    # hdf5 key is <table_name>/<checkpoint_name>
    checkpoint_tables = {
        table_name: pipeline.pipeline_table_key(table_name, checkpoint_name)
        for table_name, checkpoint_name in checkpoint_tables.items()
    }

    data = {}
    for table_name, table_key in checkpoint_tables.items():
        if tables is None or table_name in tables:
            data[table_name] = pipeline_store[table_key]

    pipeline_store.close()

    # checkpoint name and series mapping table name to hdf5 key for tables in that checkpoint
    return checkpoint_name, data

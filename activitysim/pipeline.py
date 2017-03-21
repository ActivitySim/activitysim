import os
import time
import datetime as dt

import cPickle

import numpy as np
import pandas as pd
import orca

import logging

import prng
from tracing import print_elapsed_time

logger = logging.getLogger(__name__)

_MODELS = [
    'compute_accessibility',
    'school_location_simulate',
    'workplace_location_simulate',
    'auto_ownership_simulate',
    'cdap_simulate',
    'mandatory_tour_frequency',
    'mandatory_scheduling',
    'non_mandatory_tour_frequency',
    'destination_choice',
    'non_mandatory_scheduling',
    'tour_mode_choice_simulate',
    # 'trip_mode_choice_simulate'
]

_MAX_PRNG_OFFSETS = {'households': 2, 'persons': 5, 'tours': 5, 'trips': 5}  # FIXME

_TIMESTAMP_COL = 'timestamp'
_CHECKPOINT_COL = 'checkpoint_name'
_GLOBAL_RANDOM_STATE_COL = 'random_state'
_PRNG_CHANNELS_COL = 'prng_channels'
_NON_TABLE_COLUMNS = [_CHECKPOINT_COL, _TIMESTAMP_COL, _GLOBAL_RANDOM_STATE_COL, _PRNG_CHANNELS_COL]

_RUNTIME_COL = 'runtime_seconds'

_CHECKPOINT_TABLE_NAME = 'checkpoints'
_INITIAL_CHECKPOINT_NAME = 'init'

# most recent checkpoint
_LAST_CHECKPOINT = {}

# array of checkpoint dicts
_CHECKPOINTS = []

_PRNG = prng.Prng(_MAX_PRNG_OFFSETS)


@orca.injectable(cache=True)
def pipeline_path(output_dir, settings):
    pipeline_file_name = settings.get('pipeline', 'pipeline.h5')
    pipeline_file_path = os.path.join(output_dir, pipeline_file_name)
    return pipeline_file_path


def get_pipeline_store():
    store = orca.get_injectable('pipeline_store')
    if not orca.is_injectable('pipeline_store'):
        raise RuntimeError("Pipeline store not initialized! Did you call run the init step?")
    return orca.get_injectable('pipeline_store')


def get_rn_generator():
    return _PRNG


def _open_pipeline_store(overwrite=False):

    if orca.is_injectable('pipeline_store'):
        raise RuntimeError("Pipeline store is already open!")

    pipeline_file_path = orca.get_injectable('pipeline_path')

    if overwrite:
        try:
            if os.path.isfile(pipeline_file_path):
                logger.debug("removing pipeline store: %s" % pipeline_file_path)
                os.unlink(pipeline_file_path)
        except Exception as e:
            print(e)
            logger.warn("Error removing %s: %s" % (e,))

    store = pd.HDFStore(pipeline_file_path, mode='a')
    orca.add_injectable('pipeline_store', store)

    logger.debug("opened pipeline_store")


def read_df(table_name, checkpoint_name=None):

    if checkpoint_name:
        key = "%s/%s" % (table_name, checkpoint_name)
    else:
        key = table_name

    t0 = print_elapsed_time()

    store = get_pipeline_store()
    df = store[key]

    print_elapsed_time("read_df %s shape %s" % (key, df.shape,), t0, debug=True)

    return df


def write_df(df, table_name, checkpoint_name=None):

    if checkpoint_name:
        key = "%s/%s" % (table_name, checkpoint_name)
    else:
        key = table_name

    t0 = print_elapsed_time()

    store = get_pipeline_store()
    store[key] = df

    runtime_seconds = time.time() - t0

    print_elapsed_time("write_df %s shape %s" % (key, df.shape,), t0, debug=True)


def rewrap(table_name, df=None):

    logger.debug("rewrap table %s inplace=%s" % (table_name, (df is None)))

    if orca.is_table(table_name):

        if df is None:
            logger.debug("rewrap - orca.get_table(%s)" % (table_name,))
            t = orca.get_table(table_name)
            df = t.to_frame()
        else:
            logger.debug("rewrap - orca.get_raw_table(%s)" % (table_name,))
            # don't trigger function call of TableFuncWrapper
            t = orca.get_raw_table(table_name)

        t.clear_cached()

        for column_name in orca.list_columns_for_table(table_name):
            # logger.debug("pop %s.%s: %s" % (table_name, column_name, t.column_type(column_name)))
            orca.orca._COLUMNS.pop((table_name, column_name), None)

        # remove from orca's table list
        orca.orca._TABLES.pop(table_name, None)

    logger.debug("rewrap - orca.add_table(%s)" % (table_name,))
    orca.add_table(table_name, df)

    return df


def print_checkpoints():

    print "\nprint_checkpoints"

    for checkpoint in _CHECKPOINTS:

        print "\n "

        print "checkpoint keys:", checkpoint.keys()
        print "checkpoint_name:", checkpoint[_CHECKPOINT_COL]
        print "timestamp:      ", checkpoint[_TIMESTAMP_COL]

        print "gprng_offset:      ", checkpoint.get(_GLOBAL_RANDOM_STATE_COL, None)
        print "prng channels:", cPickle.loads(checkpoint[_PRNG_CHANNELS_COL])

        table_columns = list((set(checkpoint.keys()) - set(_NON_TABLE_COLUMNS)))
        for table_name in table_columns:
            print "table: '%s'" % table_name


def set_checkpoint(checkpoint_name):

    timestamp = dt.datetime.now()

    logger.debug("set_checkpoint %s timestamp %s" % (checkpoint_name, timestamp))

    for table_name in orca_dataframe_tables():

        # if we have not already checkpointed it or it has changed
        if (table_name not in _LAST_CHECKPOINT or len(orca.list_columns_for_table(table_name))):

            df = rewrap(table_name)

            logger.debug("set_checkpoint %s writing %s to store" % (checkpoint_name, table_name, ))

            # write it to store
            write_df(df, table_name, checkpoint_name)

            # remember which checkpoint it was last written
            _LAST_CHECKPOINT[table_name] = checkpoint_name

    _LAST_CHECKPOINT[_CHECKPOINT_COL] = checkpoint_name
    _LAST_CHECKPOINT[_TIMESTAMP_COL] = timestamp
    _LAST_CHECKPOINT[_GLOBAL_RANDOM_STATE_COL] = _PRNG.gprng_offset
    _LAST_CHECKPOINT[_PRNG_CHANNELS_COL] = cPickle.dumps(_PRNG.get_channels())

    _CHECKPOINTS.append(_LAST_CHECKPOINT.copy())

    checkpoints = pd.DataFrame(_CHECKPOINTS)

    write_df(checkpoints, _CHECKPOINT_TABLE_NAME)

    logger.debug("gprng_offset: %s" % _PRNG.gprng_offset)
    for channel_state in _PRNG.get_channels():
        logger.debug("channel_name '%s', step_name '%s', offset: %s" % channel_state)


def orca_dataframe_tables():

    t = []

    for table_name in orca.list_tables():
        if orca.table_type(table_name) == 'dataframe':
            t.append(table_name)

    return t


def checkpointed_tables():

    return list((set(_LAST_CHECKPOINT.keys()) - set(_NON_TABLE_COLUMNS)))


def load_checkpoint(resume_after):

    logger.info("load_checkpoint %s" % (resume_after))

    checkpoints_df = read_df(_CHECKPOINT_TABLE_NAME)

    index_of_resume_after = checkpoints_df[checkpoints_df[_CHECKPOINT_COL] == resume_after].index
    if len(index_of_resume_after) == 0:
        msg = "Couldn't find checkpoint '%s' in checkpoints" % (resume_after,)
        logger.error(msg)
        raise RuntimeError(msg)
    index_of_resume_after = index_of_resume_after[0]

    # truncate rows after resume_after
    checkpoints_df = checkpoints_df.loc[:index_of_resume_after]

    # array of dicts
    checkpoints = checkpoints_df.to_dict(orient='records')

    # drop tables with empty names (they are nans)
    for checkpoint in checkpoints:
        for key in checkpoint.keys():
            if key not in _NON_TABLE_COLUMNS and type(checkpoint[key]) != str:
                del checkpoint[key]

    # patch _CHECKPOINTS array of dicts
    del _CHECKPOINTS[:]
    _CHECKPOINTS.extend(checkpoints)

    # print_checkpoints()

    # patch _CHECKPOINTS dict with latest checkpoint info
    _LAST_CHECKPOINT.clear()
    _LAST_CHECKPOINT.update(_CHECKPOINTS[-1])

    logger.info("load_checkpoint %s timestamp %s" % (resume_after, _LAST_CHECKPOINT['timestamp']))

    for table_name in checkpointed_tables():
        rewrap(table_name, read_df(table_name, checkpoint_name=_LAST_CHECKPOINT[table_name]))

    # set random state to pickled state at end of last checkpoint
    logger.debug("resetting random state")
    _PRNG.set_global_prng_offset(offset=_LAST_CHECKPOINT[_GLOBAL_RANDOM_STATE_COL])
    _PRNG.load_channels(cPickle.loads(_LAST_CHECKPOINT[_PRNG_CHANNELS_COL]))


def run_model(model_name):

    t0 = print_elapsed_time()

    _PRNG.begin_step(model_name)

    orca.run([model_name])

    _PRNG.end_step(model_name)

    runtime_seconds = time.time() - t0

    set_checkpoint(model_name)

    print_elapsed_time("run_model '%s'" % model_name, t0)


def start_pipeline(resume_after=None):

    logger.info("start_pipeline...")

    skims = orca.get_injectable('skim_dict')
    logger.debug("load skim_dict")

    skims = orca.get_injectable('skim_stack')
    logger.debug("load skim_stack")

    if resume_after:
        _open_pipeline_store(overwrite=False)
        load_checkpoint(resume_after)
    else:
        _open_pipeline_store(overwrite=True)
        set_checkpoint(_INITIAL_CHECKPOINT_NAME)

    logger.debug("start_pipeline complete")


def run(models=None, resume_after=None):

    if models is None:
        models = _MODELS

    if resume_after and resume_after in models:
        models = models[models.index(resume_after) + 1:]

    start_pipeline(resume_after)

    for model in models:
        run_model(model)


def close():

    orca.get_injectable('store').close()
    orca.get_injectable('omx_file').close()
    orca.get_injectable('pipeline_store').close()

    logger.info("close_pipeline")


def get_table(table_name, checkpoint_name=None):

    # orca table not in checkpoints (e.g. a merged table)
    if table_name not in _LAST_CHECKPOINT and orca.is_table(table_name):
        if checkpoint_name is not None:
            raise RuntimeError("get_table: checkpoint_name ('%s') not supported"
                               "for non-checkpointed table '%s'" % (checkpoint_name, table_name))

        return orca.get_table(table_name).to_frame()

    if table_name not in _LAST_CHECKPOINT:
        raise RuntimeError("table '%s' not in checkpoints." % table_name)

    if checkpoint_name is None or _LAST_CHECKPOINT[table_name] == checkpoint_name:
        return orca.get_table(table_name).local

    if checkpoint_name not in [checkpoint[_CHECKPOINT_COL] for checkpoint in _CHECKPOINTS]:
        raise RuntimeError("checkpoint '%s' not in checkpoints." % checkpoint_name)

    return read_df(table_name, checkpoint_name)

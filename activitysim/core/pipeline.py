import os
import time
import datetime as dt

import cPickle

import numpy as np
import pandas as pd
import orca

import logging
import inject

import random
import tracing
from tracing import print_elapsed_time

logger = logging.getLogger(__name__)

# name of the checkpoint dict keys
# (which are also columns in the checkpoints dataframe stored in hte pipeline store)
TIMESTAMP = 'timestamp'
CHECKPOINT_NAME = 'checkpoint_name'
PRNG_CHANNELS = 'prng_channels'
NON_TABLE_COLUMNS = [CHECKPOINT_NAME, TIMESTAMP, PRNG_CHANNELS]

# name used for storing the checkpoints dataframe to the pipeline store
CHECKPOINT_TABLE_NAME = 'checkpoints'

# name of the first step/checkpoint created when teh pipeline is started
INITIAL_CHECKPOINT_NAME = 'init'


class Pipeline(object):
    def __init__(self):
        self.init_state()

    def init_state(self):

        # most recent checkpoint
        self.last_checkpoint = {}

        # array of checkpoint dicts
        self.checkpoints = []

        self.replaced_tables = {}

        self.prng = random.Random()

        self.open_files = {}

        self.pipeline_store = None


_PIPELINE = Pipeline()


def close_on_exit(file, name):
    assert name not in _PIPELINE.open_files
    _PIPELINE.open_files[name] = file


def close_open_files():
    for name, file in _PIPELINE.open_files.iteritems():
        print "Closing %s" % name
        file.close()
    _PIPELINE.open_files.clear()


def add_dependent_columns(base_dfname, new_dfname):
    tbl = orca.get_table(new_dfname)
    for col in tbl.columns:
        logger.debug("Adding dependent column %s" % col)
        orca.add_column(base_dfname, col, tbl[col])


def open_pipeline_store(overwrite=False):
    """
    Open the pipeline checkpoint store

    Parameters
    ----------
    overwrite : bool
        delete file before opening (unless resuming)
    """

    if _PIPELINE.pipeline_store is not None:
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

    _PIPELINE.pipeline_store = pd.HDFStore(pipeline_file_path, mode='a')

    logger.debug("opened pipeline_store")


def get_pipeline_store():
    """
    Return the open pipeline hdf5 checkpoint store or return None if it not been opened
    """
    return _PIPELINE.pipeline_store


def get_rn_generator():
    """
    Return the singleton random number object

    Returns
    -------
    activitysim.random.Random
    """

    return _PIPELINE.prng


def set_rn_generator_base_seed(seed):
    """
    Like seed for numpy.random.RandomState, but generalized for use with all random streams.

    Provide a base seed that will be added to the seeds of all random streams.
    The default base seed value is 0, so set_base_seed(0) is a NOP

    set_rn_generator_base_seed(1) will (e.g.) provide a different set of random streams
    than the default, but will provide repeatable results re-running or resuming the simulation

    set_rn_generator_base_seed(None) will set the base seed to a random and unpredictable integer
    and so provides "fully pseudo random" non-repeatable streams with different results every time

    Must be called before open_pipeline() or pipeline.run()

    Parameters
    ----------
    seed : int or None
    """

    if _PIPELINE.last_checkpoint:
        raise RuntimeError("Can only call set_rn_generator_base_seed before the first step.")

    _PIPELINE.prng.set_base_seed(seed)


def read_df(table_name, checkpoint_name=None):
    """
    Read a pandas dataframe from the pipeline store.

    We store multiple versions of all simulation tables, for every checkpoint in which they change,
    so we need to know both the table_name and the checkpoint_name of hte desired table.

    The only exception is the checkpoints dataframe, which just has a table_name

    An error will be raised by HDFStore if the table is not found

    Parameters
    ----------
    table_name : str
    checkpoint_name : str

    Returns
    -------
    df : pandas.DataFrame
        the dataframe read from the store

    """

    if checkpoint_name:
        key = "%s/%s" % (table_name, checkpoint_name)
    else:
        key = table_name

    t0 = print_elapsed_time()

    store = get_pipeline_store()
    df = store[key]

    t0 = print_elapsed_time("read_df %s shape %s" % (key, df.shape,), t0, debug=True)

    return df


def write_df(df, table_name, checkpoint_name=None):
    """
    Write a pandas dataframe to the pipeline store.

    We store multiple versions of all simulation tables, for every checkpoint in which they change,
    so we need to know both the table_name and the checkpoint_name to label the saved table

    The only exception is the checkpoints dataframe, which just has a table_name

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to store
    table_name : str
        also conventionally the orca table name
    checkpoint_name : str
        the checkpoint at which the table was created/modified
    """

    # coerce column names to str as unicode names will cause PyTables to pickle them
    df.columns = df.columns.astype(str)

    if checkpoint_name:
        key = "%s/%s" % (table_name, checkpoint_name)
    else:
        key = table_name

    t0 = print_elapsed_time()

    store = get_pipeline_store()
    store[key] = df

    t0 = print_elapsed_time("write_df %s shape %s" % (key, df.shape,), t0, debug=True)


def rewrap(table_name, df=None):
    """
    Add or replace an orca registered table as a unitary DataFrame-backed DataFrameWrapper table

    if df is None, then get the dataframe from orca (table_name should be registered, or
    an error will be thrown) which may involve evaluating added columns, etc.

    If the orca table already exists, deregister it along with any associated columns before
    re-registering it.

    The net result is that the dataframe is a registered orca DataFrameWrapper table with no
    computed or added columns.

    Parameters
    ----------
    table_name
    df

    Returns
    -------
        the underlying df of the rewrapped table
    """

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

    assert df is not None

    logger.debug("rewrap - orca.add_table(%s)" % (table_name,))
    orca.add_table(table_name, df)

    return df


def add_checkpoint(checkpoint_name):
    """
    Create a new checkpoint with specified name, write all data required to restore the simulation
    to its current state.

    Detect any changed tables , re-wrap them and write the current version to the pipeline store.
    Write the current state of the random number generator.

    Parameters
    ----------
    checkpoint_name : str
    """
    timestamp = dt.datetime.now()

    logger.debug("add_checkpoint %s timestamp %s" % (checkpoint_name, timestamp))

    for table_name in orca_dataframe_tables():

        # if we have not already checkpointed it or it has changed
        # FIXME - this won't detect if the orca table was modified
        if len(orca.list_columns_for_table(table_name)):
            logger.debug("add_checkpoint table_name %s rewrap" % (table_name,))
            # rewrap the changed orca table as a unitary DataFrame-backed DataFrameWrapper table
            df = rewrap(table_name)
        elif table_name not in _PIPELINE.last_checkpoint or table_name in _PIPELINE.replaced_tables:
            df = orca.get_table(table_name).to_frame()
            logger.debug("add_checkpoint table_name %s get_table" % (table_name,))
        else:
            continue

        logger.debug("add_checkpoint %s writing %s to store" % (checkpoint_name, table_name, ))

        # write it to store
        write_df(df, table_name, checkpoint_name)

        # remember which checkpoint it was last written
        _PIPELINE.last_checkpoint[table_name] = checkpoint_name

    _PIPELINE.replaced_tables.clear()

    _PIPELINE.last_checkpoint[CHECKPOINT_NAME] = checkpoint_name
    _PIPELINE.last_checkpoint[TIMESTAMP] = timestamp

    # current state of the random number generator
    _PIPELINE.last_checkpoint[PRNG_CHANNELS] = cPickle.dumps(_PIPELINE.prng.get_channels())

    # append to the array of checkpoint history
    _PIPELINE.checkpoints.append(_PIPELINE.last_checkpoint.copy())

    # create a pandas dataframe of the checkpoint history, one row per checkpoint
    checkpoints = pd.DataFrame(_PIPELINE.checkpoints)

    # convert empty values to str so PyTables doesn't pickle object types
    for c in checkpoints.columns:
        checkpoints[c] = checkpoints[c].fillna('')

    # write it to the store, overwriting any previous version (no way to simply extend)
    write_df(checkpoints, CHECKPOINT_TABLE_NAME)

    for channel_state in _PIPELINE.prng.get_channels():
        logger.debug("channel_name '%s', step_name '%s', offset: %s" % channel_state)


def orca_dataframe_tables():
    """
    Return a list of the neames of all currently registered dataframe tables
    """
    return [name for name in orca.list_tables() if orca.table_type(name) == 'dataframe']


def checkpointed_tables():
    """
    Return a list of the names of all checkpointed tables
    """
    return [name for name in _PIPELINE.last_checkpoint.keys() if name not in NON_TABLE_COLUMNS]


def load_checkpoint(checkpoint_name):
    """
    Load dataframes and restore random number channel state from pipeline hdf5 file.
    This restores the pipeline state that existed at the specified checkpoint in a prior simulation.
    This allows us to resume the simulation after the specified checkpoint

    Parameters
    ----------
    checkpoint_name : str
        model_name of checkpoint to load (resume_after argument to open_pipeline)
    """

    logger.info("load_checkpoint %s" % (checkpoint_name))

    checkpoints = read_df(CHECKPOINT_TABLE_NAME)

    if checkpoint_name == '_':
        checkpoint_name = checkpoints[CHECKPOINT_NAME].iloc[-1]

    try:
        # truncate rows after target checkpoint
        i = checkpoints[checkpoints[CHECKPOINT_NAME] == checkpoint_name].index[0]
        checkpoints = checkpoints.loc[:i]
    except IndexError:
        msg = "Couldn't find checkpoint '%s' in checkpoints" % (checkpoint_name,)
        logger.error(msg)
        raise RuntimeError(msg)

    # convert pandas dataframe back to array of checkpoint dicts
    checkpoints = checkpoints.to_dict(orient='records')

    # drop tables with empty names
    for checkpoint in checkpoints:
        for key in checkpoint.keys():
            if key not in NON_TABLE_COLUMNS and not checkpoint[key]:
                del checkpoint[key]

    # patch _CHECKPOINTS array of dicts
    # del _PIPELINE.checkpoints[:]
    # _PIPELINE.checkpoints.extend(checkpoints)
    _PIPELINE.checkpoints = checkpoints

    # patch _CHECKPOINTS dict with latest checkpoint info
    _PIPELINE.last_checkpoint.clear()
    _PIPELINE.last_checkpoint.update(_PIPELINE.checkpoints[-1])

    logger.info("load_checkpoint %s timestamp %s"
                % (checkpoint_name, _PIPELINE.last_checkpoint['timestamp']))

    # table names in order that tracing.register_traceable_table wants us to register them
    tables = tracing.sort_for_registration(checkpointed_tables())

    for table_name in tables:
        # read dataframe from pipeline store
        df = read_df(table_name, checkpoint_name=_PIPELINE.last_checkpoint[table_name])
        logger.info("load_checkpoint table %s %s" % (table_name, df.shape))
        # register it as an orca table
        rewrap(table_name, df)
        # register for tracing
        tracing.register_traceable_table(table_name, df)

    # set random state to pickled state at end of last checkpoint
    logger.debug("resetting random state")
    _PIPELINE.prng.load_channels(cPickle.loads(_PIPELINE.last_checkpoint[PRNG_CHANNELS]))


def split_arg(s, sep, default=''):
    """
    split str s in two at first sep, returning empty string as second result if no sep
    """
    r = s.split(sep, 2)
    r = map(str.strip, r)

    arg = r[0]

    if len(r) == 1:
        val = default
    else:
        val = r[1]
        val = {'true': True, 'false': False}.get(val.lower(), val)

    return arg, val


def run_model(model_name):
    """
    Run the specified model and add checkpoint for model_name

    Since we use model_name as checkpoint name, the same model may not be run more than once.

    Parameters
    ----------
    model_name : str
        model_name is assumed to be the name of a registered orca step
    """

    if not _PIPELINE.last_checkpoint:
        raise RuntimeError("Pipeline not initialized! Did you call open_pipeline?")

    # can't run same model more than once
    if model_name in [checkpoint[CHECKPOINT_NAME] for checkpoint in _PIPELINE.checkpoints]:
        raise RuntimeError("Cannot run model '%s' more than once" % model_name)

    t0 = print_elapsed_time()
    _PIPELINE.prng.begin_step(model_name)

    # check for args
    if '.' in model_name:
        step_name, arg_string = model_name.split('.', 1)
        args = dict((k, v)
                    for k, v in (split_arg(item, "=", default=True)
                                 for item in arg_string.split(";")))
    else:
        step_name = model_name
        args = {}

    inject.set_step_args(args)

    orca.run([step_name])

    inject.set_step_args(None)

    _PIPELINE.prng.end_step(model_name)
    t0 = print_elapsed_time("run_model '%s'" % model_name, t0)
    add_checkpoint(model_name)
    t0 = print_elapsed_time("add_checkpoint '%s'" % model_name, t0)


def open_pipeline(resume_after=None):
    """
    Start pipeline, either for a new run or, if resume_after, loading checkpoint from pipeline.

    If resume_after, then we expect the pipeline hdf5 file to exist and contain
    checkpoints from a previous run, including a checkpoint with name specified in resume_after

    Parameters
    ----------
    resume_after : str or None
        name of checkpoint to load from pipeline store
    """

    logger.info("open_pipeline...")

    t0 = print_elapsed_time()

    if resume_after:
        # open existing pipeline
        logger.debug("open_pipeline - open existing pipeline")
        open_pipeline_store(overwrite=False)
        load_checkpoint(resume_after)
        t0 = print_elapsed_time("load_checkpoint '%s'" % resume_after, t0)
    else:
        # open new, empty pipeline
        logger.debug("open_pipeline - new, empty pipeline")
        open_pipeline_store(overwrite=True)
        # - not sure why I thought we needed this?
        # could have exogenous tables or prng instantiation under some circumstance??
        _PIPELINE.last_checkpoint[CHECKPOINT_NAME] = INITIAL_CHECKPOINT_NAME
        # add_checkpoint(INITIAL_CHECKPOINT_NAME)
        # t0 = print_elapsed_time("add_checkpoint '%s'" % INITIAL_CHECKPOINT_NAME, t0)

    logger.debug("open_pipeline complete")


def close_pipeline():
    """
    Close any known open files
    """

    close_open_files()

    _PIPELINE.pipeline_store.close()

    _PIPELINE.init_state()

    logger.info("close_pipeline")


def preload_injectables():

    t0 = print_elapsed_time()

    # load skim_stack
    if orca.is_injectable('preload_injectables'):
        orca.get_injectable('preload_injectables')

    t0 = print_elapsed_time("preload_injectables", t0)


def run(models, resume_after=None):
    """
    run the specified list of models, optionally loading checkpoint and resuming after specified
    checkpoint.

    Since we use model_name as checkpoint name, the same model may not be run more than once.

    If resume_after checkpoint is specified and a model with that name appears in the models list,
    then we only run the models after that point in the list. This allows the user always to pass
    the same list of models, but specify a resume_after point if desired.

    Parameters
    ----------
    models : [str]
        list of model_names
    resume_after : str or None
        model_name of checkpoint to load checkpoint and AFTER WHICH to resume model run
    """

    if resume_after and resume_after in models:
        models = models[models.index(resume_after) + 1:]

    open_pipeline(resume_after)

    preload_injectables()

    t0 = print_elapsed_time()
    for model in models:
        run_model(model)
    t0 = print_elapsed_time("run (%s models)" % len(models), t0)

    # don't close the pipeline, as the user may want to read intermediate results from the store


def get_table(table_name, checkpoint_name=None):
    """
    Return pandas dataframe corresponding to table_name

    if checkpoint_name is None, return the current (most recent) version of the table.
    The table can be a checkpointed table or any registered orca table (e.g. function table)

    if checkpoint_name is specified, return table as it was at that checkpoint
    (the most recently checkpointed version of the table at or before checkpoint_name)

    Parameters
    ----------
    table_name : str
    checkpoint_name : str or None

    Returns
    -------
    df : pandas.DataFrame
    """

    # orca table not in checkpoints (e.g. a merged table)
    if table_name not in _PIPELINE.last_checkpoint and orca.is_table(table_name):
        if checkpoint_name is not None:
            raise RuntimeError("get_table: checkpoint_name ('%s') not supported"
                               "for non-checkpointed table '%s'" % (checkpoint_name, table_name))

        return orca.get_table(table_name).to_frame()

    # was table ever checkpointed?
    if table_name not in checkpointed_tables():
        raise RuntimeError("table '%s' not in checkpointed tables." % table_name)

    # if they want current version of table, no need to read from pipeline store
    if checkpoint_name is None or _PIPELINE.last_checkpoint[table_name] == checkpoint_name:
        # return orca.get_table(table_name).local
        return orca.get_table(table_name).to_frame()

    if checkpoint_name not in [checkpoint[CHECKPOINT_NAME] for checkpoint in _PIPELINE.checkpoints]:
        raise RuntimeError("checkpoint '%s' not in checkpoints." % checkpoint_name)

    return read_df(table_name, checkpoint_name)


def get_checkpoints():
    """
    Get pandas dataframe of info about all checkpoints stored in pipeline

    Returns
    -------
    checkpoints_df : pandas.DataFrame

    """

    store = get_pipeline_store()

    if store:
        df = store[CHECKPOINT_TABLE_NAME]
    else:
        pipeline_file_path = orca.get_injectable('pipeline_path')
        df = pd.read_hdf(pipeline_file_path, CHECKPOINT_TABLE_NAME)

    return df

    # non-table columns first (column order in df is random because created from a dict)
    table_names = [name for name in df.columns.values if name not in NON_TABLE_COLUMNS]

    # omit human-illegible PRNG_CHANNELS
    df = df[[CHECKPOINT_NAME, TIMESTAMP] + table_names]

    df.index.name = 'step_num'

    return df


def replace_table(table_name, df):
    """
    Add or replace a orca table, removing any existing added orca columns

    The use case for this function is a method that calls to_frame on an orca table, modifies
    it and then saves the modified.

    orca.to_frame returns a copy, so no changes are saved, and adding multiple column with
    add_column adds them in an indeterminate order.

    Simply replacing an existing the table "behind the pipeline's back" by calling orca.add_table
    risks pipeline to failing to detect that it has changed, and thus not checkpoint the changes.

    Parameters
    ----------
    table_name : str
        orca/pipeline table name
    df : pandas DataFrame
    """

    rewrap(table_name, df)

    _PIPELINE.replaced_tables[table_name] = True


def extend_table(table_name, df):
    """
    add new table or extend (add rows) to an existing table

    Parameters
    ----------
    table_name : str
        orca/inject table name
    df : pandas DataFrame
    """

    if orca.is_table(table_name):

        extend_df = orca.get_table(table_name).to_frame()

        # don't expect indexes to overlap
        assert len(extend_df.index.intersection(df.index)) == 0

        # preserve existing column order (concat reorders columns)
        columns = list(extend_df.columns) + [c for c in df.columns if c not in extend_df.columns]

        df = pd.concat([extend_df, df])[columns]

    replace_table(table_name, df)

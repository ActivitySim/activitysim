# ActivitySim
# See full license in LICENSE.txt.
import logging
import multiprocessing
import os
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml

from activitysim.core import config, inject, mem, pipeline, tracing, util
from activitysim.core.config import setting

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "_"

MEM_TRACE_TICKS = 5

"""
mp_tasks - activitysim multiprocessing overview

Activitysim runs a list of models sequentially, performing various computational operations
on tables. Model steps can modify values in existing tables, add columns, or create additional
tables. Activitysim provides the facility, via expression files, to specify vectorized operations
on data tables. The ability to vectorize operations depends upon the independence of the
computations performed on the vectorized elements.

Python is agonizingly slow performing scalar operations sequentially on large datasets, so
vectorization (using pandas and/or numpy) is essential for good performance.

Fortunately most activity based model simulation steps are row independent at the household,
person, tour, or trip level. The decisions for one household are independent of the choices
made by other households. Thus it is (generally speaking) possible to run an entire simulation
on a household sample with only one household, and get the same result for that household as
you would running the simulation on a thousand households. (See the shared data section below
for an exception to this highly convenient situation.)

The random number generator supports this goal by providing streams of random numbers
for each households and person that are mutually independent and repeatable across model runs
and processes.

To the extent that simulation model steps are row independent, we can implement most simulations
as a series of vectorized operations on pandas DataFrames and numpy arrays. These vectorized
operations are much faster than sequential python because they are implemented by native code
(compiled C) and are to some extent multi-threaded. But the benefits of numpy multi-processing are
limited because they only apply to atomic numpy or pandas calls, and as soon as control returns
to python it is single-threaded and slow.

Multi-threading is not an attractive strategy to get around the python performance problem because
of the limitations imposed by python's global interpreter lock (GIL). Rather than struggling with
python multi-threading, this module uses the python multiprocessing to parallelize certain models.

Because of activitysim's modular and extensible architecture, we don't hardwire the multiprocessing
architecture. The specification of which models should be run in parallel, how many processers
should be used, and the segmentation of the data between processes are all specified in the
settings config file. For conceptual simplicity, the single processing model as treated as
dominant (because even though in practice multiprocessing may be the norm for production runs,
the single-processing model will be used in development and debugging and keeping it dominant
will tend to concentrate the multiprocessing-specific code in one place and prevent multiprocessing
considerations from permeating the code base obscuring the model-specific logic.

The primary function of the multiprocessing settings are to identify distinct stages of
computation, and to specify how many simultaneous processes should be used to perform them,
and how the data to be treated should be apportioned between those processes. We assume that
the data can be apportioned between subprocesses according to the index of a single primary table
(e.g. households) or else are by derivative or dependent tables that reference that table's index
(primary key) with a ref_col (foreign key) sharing the name of the primary table's key.

Generally speaking, we assume that any new tables that are created are directly dependent on the
previously existing tables, and all rows in new tables are either attributable to previously
existing rows in the pipeline tables, or are global utility tables that are identical across
sub-processes.

Note: There are a few exceptions to 'row independence', such as school and location choice models,
where the model behavior is externally constrained or adjusted. For instance, we want school
location choice to match known aggregate school enrollments by zone. Similarly, a parking model
(not yet implemented) might be constrained by availability. These situations require special
handling.

::

    models:
      ### mp_initialize step
      - initialize_landuse
      - compute_accessibility
      - initialize_households
      ### mp_households step
      - school_location
      - workplace_location
      - auto_ownership_simulate
      - free_parking
      ### mp_summarize step
      - write_tables

    multiprocess_steps:
      - name: mp_initialize
        begin: initialize_landuse
      - name: mp_households
        begin: school_location
        num_processes: 2
        slice:
          tables:
            - households
            - persons
      - name: mp_summarize
        begin: write_tables

The multiprocess_steps setting above annotates the models list to indicate that the simulation
should be broken into three steps.

The first multiprocess_step (mp_initialize) begins with the initialize_landuse step and is
implicity single-process because there is no 'slice' key indicating how to apportion the tables.
This first step includes all models listed in the 'models' setting up until the first step
in the next multiprocess_steps.

The second multiprocess_step (mp_households) starts with the school location model and continues
through auto_ownership_simulate. The 'slice' info indicates that the tables should be sliced by
households, and that persons is a dependent table and so and persons with a ref_col (foreign key
column with the same name as the Households table index) referencing a household record should be
taken to 'belong' to that household. Similarly, any other table that either share an index
(i.e. having the same name) with either the households or persons table, or have a ref_col to
either of their indexes, should also be considered a dependent table.

The num_processes setting of 2 indicates that the pipeline should be split in two, and half of the
households should be apportioned into each subprocess pipeline, and all dependent tables should
likewise be apportioned accordingly. All other tables (e.g. land_use) that do share an index (name)
or have a ref_col should be considered mirrored and be included in their entirety.

The primary table is sliced by num_processes-sized strides. (e.g. for num_processes == 2, the
sub-processes get every second record starting at offsets 0 and 1 respectively. All other dependent
tables slices are based (directly or indirectly) on this primary stride segmentation of the primary
table index.

Two separate sub-process are launched (num_processes == 2) and each passed the name of their
apportioned pipeline file. They execute independently and if they terminate successfully, their
contents are then coalesced into a single pipeline file whose tables should then be essentially
the same as it had been generated by a single process.

We assume that any new tables that are created by the sub-processes are directly dependent on the
previously primary tables or are mirrored. Thus we can coalesce the sub-process pipelines by
concatenating the primary and dependent tables and simply retaining any copy of the mirrored tables
(since they should all be identical.)

The third multiprocess_step (mp_summarize) then is handled in single-process mode and runs the
write_tables model, writing the results, but also leaving the tables in the pipeline, with
essentially the same tables and results as if the whole simulation had been run as a single process.

"""

"""
shared data

Although multiprocessing subprocesses each have their (apportioned) pipeline, they also share some
data passed to them by the parent process. There are essentially two types of shared data.

read-only shared data

Skim files are read-only and take up a lot of RAM, so we share them across sub-processes, loading
them into shared-memory (multiprocessing.sharedctypes.RawArray) in the parent process and passing
them to the child sub-processes when they are launched/forked. (unlike ordinary python data,
sharedctypes are not pickled and reconstituted, but passed through to the subprocess by address
when launch/forked multiprocessing.Process. Since they are read-only, no Locks are required to
access their data safely. The receiving process needs to know to wrap them using numpy.frombuffer
but they can thereafter be treated as ordinary numpy arrays.

read-write shared memory

There are a few circumstances in which the assumption of row independence breaks down.
This happens if the model must respect some aggregated resource or constraint such as school
enrollments or parking availability. In these cases, the individual choice models have to be
influenced or constrained in light of aggregate choices.

Currently school and workplace location choice are the only such aggregate constraints.
The details of these are handled by the shadow_pricing module (q.v.), and our only concern here
is the need to provide shared read-write data buffers for communication between processes.
It is worth noting here that the shared buffers are instances of multiprocessing.Array which
incorporates a multiprocessing.Lock object to mediate access of the underlying data. You might
think that the existence of such a lock would make shared access pretty straightforward, but
this is not the case as the level of locking is very low, reportedly not very performant, and
essentially useless in any event since we want to use numpy.frombuffer to wrap and handle them
as numpy arrays. The Lock is a convenient bundled locking primative, but shadow_pricing rolls
its own semaphore system using the Lock.

FIXME - The code below knows that it need to allocate skim and shadow price buffers by calling
the appropriate methods in abm.tables.skims and abm.tables.shadow_pricing to allocate shared
buffers. This is not very extensible and should be generalized.

"""

# FIXME - pathological knowledge of abm.tables.skims and abm.tables.shadow_pricing (see note above)


def log(msg, level, write_to_log_file=True):

    process_name = multiprocessing.current_process().name

    if not write_to_log_file:
        print(f"############ mp_tasks - {process_name} - {msg}")

    if write_to_log_file:
        with config.open_log_file("mp_tasks_log.txt", "a") as log_file:
            print(f"mp_tasks - {process_name} - {msg}", file=log_file)

    if write_to_log_file:
        # logger.log(level, f"mp_tasks - {process_name} - {msg}")
        logger.log(level, msg)


def debug(msg, write_to_log_file=True):
    log(msg, level=logging.DEBUG, write_to_log_file=write_to_log_file)


def info(msg, write_to_log_file=True):
    log(msg, level=logging.INFO, write_to_log_file=write_to_log_file)


def warning(msg, write_to_log_file=True):
    log(msg, level=logging.WARNING, write_to_log_file=write_to_log_file)


def error(msg, write_to_log_file=True):
    log(msg, level=logging.ERROR, write_to_log_file=write_to_log_file)


def exception(msg, write_to_log_file=True):

    process_name = multiprocessing.current_process().name

    if not write_to_log_file:
        print(f"mp_tasks - {process_name} - {msg}")
        print(f"---\n{traceback.format_exc()}---")

    with config.open_log_file("mp_tasks_log.txt", "a") as log_file:
        print(f"---\nmp_tasks - {process_name} - {msg}", file=log_file)
        traceback.print_exc(limit=10, file=log_file)
        print("---", file=log_file)

    if write_to_log_file:
        logger.log(logging.ERROR, f"mp_tasks - {process_name} - {msg}")
        logger.log(logging.ERROR, f"\n---\n{traceback.format_exc()}---\n")


"""
### child process methods (called within sub process)
"""


def pipeline_table_keys(pipeline_store):
    """
    return dict of current (as of last checkpoint) pipeline tables
    and their checkpoint-specific hdf5_keys

    This facilitates reading pipeline tables directly from a 'raw' open pandas.HDFStore without
    opening it as a pipeline (e.g. when apportioning and coalescing pipelines)

    We currently only ever need to do this from the last checkpoint, so the ability to specify
    checkpoint_name is not required, and thus omitted.

    Parameters
    ----------
    pipeline_store : open hdf5 pipeline_store

    Returns
    -------
    checkpoint_name : name of the checkpoint
    checkpoint_tables : dict {<table_name>: <table_key>}

    """

    checkpoints = pipeline_store[pipeline.CHECKPOINT_TABLE_NAME]

    # don't currently need this capability...
    # if checkpoint_name:
    #     # specified checkpoint row as series
    #     i = checkpoints[checkpoints[pipeline.CHECKPOINT_NAME] == checkpoint_name].index[0]
    #     checkpoint = checkpoints.loc[i]
    # else:

    # last checkpoint row as series
    checkpoint = checkpoints.iloc[-1]
    checkpoint_name = checkpoint.loc[pipeline.CHECKPOINT_NAME]

    # series with table name as index and checkpoint_name as value
    checkpoint_tables = checkpoint[~checkpoint.index.isin(pipeline.NON_TABLE_COLUMNS)]

    # omit dropped tables with empty checkpoint name
    checkpoint_tables = checkpoint_tables[checkpoint_tables != ""]

    # hdf5 key is <table_name>/<checkpoint_name>
    checkpoint_tables = {
        table_name: pipeline.pipeline_table_key(table_name, checkpoint_name)
        for table_name, checkpoint_name in checkpoint_tables.items()
    }

    # checkpoint name and series mapping table name to hdf5 key for tables in that checkpoint
    return checkpoint_name, checkpoint_tables


def build_slice_rules(slice_info, pipeline_tables):
    """
    based on slice_info for current step from run_list, generate a recipe for slicing
    the tables in the pipeline (passed in tables parameter)

    slice_info is a dict with two well-known keys:
        'tables': required list of table names (order matters!)
        'except': optional list of tables not to slice even if they have a sliceable index name

    Note: tables listed in slice_info must appear in same order and before any others in tables dict

    The index of the first table in the 'tables' list is the primary_slicer.

    Any other tables listed are dependent tables with either ref_cols to the primary_slicer
    or with the same index (i.e. having an index with the same name). This cascades, so any
    tables dependent on the primary_table can in turn have dependent tables that will be sliced
    by index or ref_col.

    For instance, if the primary_slicer is households, then persons can be sliced because it
    has a ref_col to (column with the same same name as) the household table index. And the
    tours table can be sliced since it has a ref_col to persons. Tables can also be sliced
    by index. For instance the person_windows table can be sliced because it has an index with
    the same names as the persons table.

    slice_info from multiprocess_steps

    ::

        slice:
          tables:
            - households
            - persons

    tables from pipeline

    +-----------------+--------------+---------------+
    | Table Name      | Index        | ref_col       |
    +=================+==============+===============+
    | households      | household_id |               |
    +-----------------+--------------+---------------+
    | persons         | person_id    | household_id  |
    +-----------------+--------------+---------------+
    | person_windows  | person_id    |               |
    +-----------------+--------------+---------------+
    | accessibility   | zone_id      |               |
    +-----------------+--------------+---------------+

    generated slice_rules dict

    ::

        households:
           slice_by: primary       <- primary table is sliced in num_processors-sized strides
        persons:
           source: households
           slice_by: column
           column:  household_id   <- slice by ref_col (foreign key) to households
        person_windows:
           source: persons
           slice_by: index         <- slice by index of persons table
        accessibility:
           slice_by:               <- mirrored (non-dependent) tables don't get sliced
        land_use:
           slice_by:


    Parameters
    ----------
    slice_info : dict
        'slice' info from run_list for this step

    pipeline_tables : dict {<table_name>, <pandas.DataFrame>}
        dict of all tables from the pipeline keyed by table name

    Returns
    -------
    slice_rules : dict
    """

    slicer_table_names = slice_info["tables"]
    slicer_table_exceptions = slice_info.get("except", [])
    primary_slicer = slicer_table_names[0]

    # - ensure that tables listed in slice_info appear in correct order and before any others
    tables = OrderedDict([(table_name, None) for table_name in slicer_table_names])

    for table_name in pipeline_tables.keys():
        tables[table_name] = pipeline_tables[table_name]

    if primary_slicer not in tables:
        raise RuntimeError("primary slice table '%s' not in pipeline" % primary_slicer)

    # allow wildcard 'True' to avoid slicing (or coalescing) any tables no explicitly listed in slice_info.tables
    # populationsim uses slice.except wildcards to avoid listing control tables (etc) that should not be sliced,
    # followed by a slice.coalesce directive to explicitly list the omnibus tables created by the subprocesses.
    # So don't change this behavior withoyt testing populationsim multiprocess!
    if slicer_table_exceptions is True:
        debug(
            f"slice.except wildcard (True): excluding all tables not explicitly listed in slice.tables"
        )
        slicer_table_exceptions = [t for t in tables if t not in slicer_table_names]

    # dict mapping slicer table_name to index name
    # (also presumed to be name of ref col name in referencing table)
    slicer_ref_cols = OrderedDict()

    if slicer_table_exceptions == "*":
        slicer_table_exceptions = [t for t in tables if t not in slicer_table_names]

    # build slice rules for loaded tables
    slice_rules = OrderedDict()
    for table_name, df in tables.items():

        rule = {}
        if table_name == primary_slicer:
            # slice primary apportion table
            rule = {"slice_by": "primary"}
        elif table_name in slicer_table_exceptions:
            rule["slice_by"] = None
        else:
            for slicer_table_name in slicer_ref_cols:
                if df.index.name is not None and (
                    df.index.name == tables[slicer_table_name].index.name
                ):
                    # slice df with same index name as a known slicer
                    rule = {"slice_by": "index", "source": slicer_table_name}
                else:
                    # if df has a column with same name as the ref_col (index) of a slicer?
                    try:
                        source, ref_col = next(
                            (t, c)
                            for t, c in slicer_ref_cols.items()
                            if c in df.columns
                        )
                        # then we can use that table to slice this df
                        rule = {
                            "slice_by": "column",
                            "column": ref_col,
                            "source": source,
                        }
                    except StopIteration:
                        rule["slice_by"] = None

        if rule["slice_by"]:
            # cascade sliceability
            slicer_ref_cols[table_name] = df.index.name

        slice_rules[table_name] = rule

    for table_name, rule in slice_rules.items():
        if rule["slice_by"] is not None:
            debug(
                f"### table_name: {table_name} slice_rules: {slice_rules[table_name]}"
            )
    debug(f"### slicer_ref_cols: {slicer_ref_cols}")

    return slice_rules


def apportion_pipeline(sub_proc_names, step_info):
    """
    apportion pipeline for multiprocessing step

    create pipeline files for sub_procs, apportioning data based on slice_rules

    Called at the beginning of a multiprocess step prior to launching the sub-processes
    Pipeline files have well known names (pipeline file name prefixed by subjob name)

    Parameters
    ----------
    sub_proc_names : list of str
        names of the sub processes to apportion
    step_info : dict
        step_info from multiprocess_steps for step we are apportioning pipeline tables for

    Returns
    -------
    creates apportioned pipeline files for each sub job
    """

    slice_info = step_info.get("slice", None)
    multiprocess_step_name = step_info.get("name", None)

    pipeline_file_name = inject.get_injectable("pipeline_file_name")

    # ensure that if we are resuming, we don't apportion any tables from future model steps
    last_checkpoint_in_previous_multiprocess_step = step_info.get(
        "last_checkpoint_in_previous_multiprocess_step", None
    )
    assert last_checkpoint_in_previous_multiprocess_step is not None
    pipeline.open_pipeline(resume_after=last_checkpoint_in_previous_multiprocess_step)

    # ensure all tables are in the pipeline
    checkpointed_tables = pipeline.checkpointed_tables()
    for table_name in slice_info["tables"]:
        if table_name not in checkpointed_tables:
            raise RuntimeError(f"slicer table {table_name} not found in pipeline")

    checkpoints_df = pipeline.get_checkpoints()

    # for the subprocess pipelines, keep only the last row of checkpoints and patch the last checkpoint name
    checkpoints_df = checkpoints_df.tail(1).copy()

    # load all tables from pipeline
    checkpoint_name = multiprocess_step_name
    tables = {}
    for table_name in checkpointed_tables:
        # patch last checkpoint name for all tables
        checkpoints_df[table_name] = checkpoint_name
        # load the dataframe
        tables[table_name] = pipeline.get_table(table_name)

        debug(f"loaded table {table_name} {tables[table_name].shape}")

    pipeline.close_pipeline()

    # should only be one checkpoint (named <multiprocess_step_name>)
    assert len(checkpoints_df) == 1

    # - build slice rules for loaded tables
    slice_rules = build_slice_rules(slice_info, tables)

    # - allocate sliced tables for each sub_proc
    num_sub_procs = len(sub_proc_names)
    for i in range(num_sub_procs):

        # use well-known pipeline file name
        process_name = sub_proc_names[i]
        pipeline_path = config.build_output_file_path(
            pipeline_file_name, use_prefix=process_name
        )

        # remove existing file
        try:
            os.unlink(pipeline_path)
        except OSError:
            pass

        with pd.HDFStore(pipeline_path, mode="a") as pipeline_store:

            # remember sliced_tables so we can cascade slicing to other tables
            sliced_tables = {}

            # - for each table in pipeline
            for table_name, rule in slice_rules.items():

                df = tables[table_name]

                if rule["slice_by"] is not None and num_sub_procs > len(df):

                    # almost certainly a configuration error
                    raise RuntimeError(
                        f"apportion_pipeline: multiprocess step {multiprocess_step_name} "
                        f"slice table {table_name} has fewer rows {df.shape} "
                        f"than num_processes ({num_sub_procs})."
                    )

                if rule["slice_by"] == "primary":
                    # slice primary apportion table by num_sub_procs strides
                    # this hopefully yields a more random distribution
                    # (e.g.) households are ordered by size in input store
                    # we are assuming that the primary table index is unique
                    # otherwise we should slice by strides in df.index.unique
                    # we could easily work around this, but it seems likely this was an error on the user's part
                    assert not df.index.duplicated().any()

                    primary_df = df[
                        np.asanyarray(list(range(df.shape[0]))) % num_sub_procs == i
                    ]
                    sliced_tables[table_name] = primary_df
                elif rule["slice_by"] == "index":
                    # slice a table with same index name as a known slicer
                    source_df = sliced_tables[rule["source"]]
                    sliced_tables[table_name] = df.loc[source_df.index]
                elif rule["slice_by"] == "column":
                    # slice a table with a recognized slicer_column
                    source_df = sliced_tables[rule["source"]]
                    sliced_tables[table_name] = df[
                        df[rule["column"]].isin(source_df.index)
                    ]
                elif rule["slice_by"] is None:
                    # don't slice mirrored tables
                    sliced_tables[table_name] = df
                else:
                    raise RuntimeError(
                        "Unrecognized slice rule '%s' for table %s"
                        % (rule["slice_by"], table_name)
                    )

                # - write table to pipeline
                hdf5_key = pipeline.pipeline_table_key(table_name, checkpoint_name)
                pipeline_store[hdf5_key] = sliced_tables[table_name]

            debug(
                f"writing checkpoints ({checkpoints_df.shape}) "
                f"to {pipeline.CHECKPOINT_TABLE_NAME} in {pipeline_path}"
            )
            pipeline_store[pipeline.CHECKPOINT_TABLE_NAME] = checkpoints_df


def coalesce_pipelines(sub_proc_names, slice_info):
    """
    Coalesce the data in the sub_processes apportioned pipelines back into a single pipeline

    We use slice_rules to distinguish sliced (apportioned) tables from mirrored tables.

    Sliced tables are concatenated to create a single omnibus table with data from all sub_procs
    but mirrored tables are the same across all sub_procs, so we can grab a copy from any pipeline.

    Parameters
    ----------
    sub_proc_names : list of str
    slice_info : dict
        slice_info from multiprocess_steps

    Returns
    -------
    creates an omnibus pipeline with coalesced data from individual sub_proc pipelines
    """

    pipeline_file_name = inject.get_injectable("pipeline_file_name")

    debug(f"coalesce_pipelines to: {pipeline_file_name}")

    # - read all tables from first process pipeline
    # FIXME - note: assumes any new tables will be present in ALL subprocess pipelines
    tables = {}
    pipeline_path = config.build_output_file_path(
        pipeline_file_name, use_prefix=sub_proc_names[0]
    )

    with pd.HDFStore(pipeline_path, mode="r") as pipeline_store:

        # hdf5_keys is a dict mapping table_name to pipeline hdf5_key
        checkpoint_name, hdf5_keys = pipeline_table_keys(pipeline_store)

        for table_name, hdf5_key in hdf5_keys.items():
            debug(f"loading table {table_name} {hdf5_key}")
            tables[table_name] = pipeline_store[hdf5_key]

    # slice.coalesce is an override  list of omnibus tables created by subprocesses that should be coalesced,
    # whether or not they satisfy the slice rules. Ordinarily all tables qualify for slicing by the slice rules
    # will be coalesced, including any new tables created by the subprocess that have sliceable indexes or ref_cols.
    # Any other new tables that don't match the slice rules will be considered mirrored. This is usually the desired
    # behavior, especially in activitysim abm models. However, if the "slice.except: True" wildcard is used, it
    # prevents the inference for newly generated tables, and this directive permits explicit specification of
    # which new tables to coalesce. Populationsim uses this wildcard except directives to avoid having to list
    # many slice exceptions, and just lists weigh tables to coalesce. So don't change this behavior without testing
    # populationsim multiprocessing!
    coalesce_tables = slice_info.get("coalesce", [])

    # report absence of any slice_info.coalesce tables not in pipeline
    # we don't require their presence in case there are tracing tables that will only be present if tracing is enabled
    for table_name in coalesce_tables:
        if table_name not in tables:
            logger.warning(
                "slicer coalesce.table %s not found in pipeline" % table_name
            )

    # - use slice rules followed by apportion_pipeline to identify mirrored tables
    # (tables that are identical in every pipeline and so don't need to be concatenated)
    slice_rules = build_slice_rules(slice_info, tables)

    # table is mirrored if no slice rule or explicitly listed in slice_info.coalesce setting
    mirrored_table_names = [
        t
        for t, rule in slice_rules.items()
        if rule["slice_by"] is None and t not in coalesce_tables
    ]
    mirrored_tables = {t: tables[t] for t in mirrored_table_names}
    omnibus_keys = {t: k for t, k in hdf5_keys.items() if t not in mirrored_table_names}

    debug(f"coalesce_pipelines to: {pipeline_file_name}")
    debug(f"mirrored_table_names: {mirrored_table_names}")
    debug(f"omnibus_keys: {omnibus_keys}")

    # assemble lists of omnibus tables from all sub_processes
    omnibus_tables = {table_name: [] for table_name in omnibus_keys}
    for process_name in sub_proc_names:
        pipeline_path = config.build_output_file_path(
            pipeline_file_name, use_prefix=process_name
        )
        logger.info(f"coalesce pipeline {pipeline_path}")

        with pd.HDFStore(pipeline_path, mode="r") as pipeline_store:
            for table_name, hdf5_key in omnibus_keys.items():
                omnibus_tables[table_name].append(pipeline_store[hdf5_key])

    # open pipeline, preserving existing checkpoints (so resume_after will work for prior steps)
    pipeline.open_pipeline("_")

    # - add mirrored tables to pipeline
    for table_name in mirrored_tables:
        df = mirrored_tables[table_name]
        info(f"adding mirrored table {table_name} {df.shape}")
        pipeline.replace_table(table_name, df)

    # - concatenate omnibus tables and add them to pipeline
    for table_name in omnibus_tables:
        df = pd.concat(omnibus_tables[table_name], sort=False)
        info(f"adding omnibus table {table_name} {df.shape}")
        pipeline.replace_table(table_name, df)

    pipeline.add_checkpoint(checkpoint_name)

    pipeline.close_pipeline()


def setup_injectables_and_logging(injectables, locutor=True):
    """
    Setup injectables (passed by parent process) within sub process

    we sometimes want only one of the sub-processes to perform an action (e.g. write shadow prices)
    the locutor flag indicates that this sub process is the designated singleton spokesperson

    Parameters
    ----------
    injectables : dict {<injectable_name>: <value>}
        dict of injectables passed by parent process
    locutor : bool
        is this sub process the designated spokesperson

    Returns
    -------
    injects injectables
    """

    # register abm steps and other abm-specific injectables
    # by default, assume we are running activitysim.abm
    # other callers (e.g. piopulationsim) will have to arrange to register their own steps and injectables
    # (presumably) in a custom run_simulation.py instead of using the 'activitysim run' command
    if not inject.is_injectable("preload_injectables"):
        from activitysim import (  # register abm steps and other abm-specific injectables
            abm,
        )

    try:

        for k, v in injectables.items():
            inject.add_injectable(k, v)

        inject.add_injectable("is_sub_task", True)
        inject.add_injectable("locutor", locutor)

        config.filter_warnings()

        process_name = multiprocessing.current_process().name
        inject.add_injectable("log_file_prefix", process_name)

    except Exception as e:
        exception(
            f"{type(e).__name__} exception while setting up injectables: {str(e)}",
            write_to_log_file=False,
        )
        raise e

    try:
        tracing.config_logger()
    except Exception as e:
        exception(f"{type(e).__name__} exception while configuring logger: {str(e)}")
        raise e


def adjust_chunk_size_for_shared_memory(chunk_size, data_buffers, num_processes):

    # even if there is only one subprocess,
    # we are separate from parent who allocated the shared memory
    # so we still need to compensate for it

    if chunk_size == 0:
        return chunk_size

    shared_memory_size = mem.shared_memory_size(data_buffers)

    if shared_memory_size == 0:
        return chunk_size

    fair_share_of_shared_memory = int(shared_memory_size / num_processes)

    adjusted_chunk_size = chunk_size - fair_share_of_shared_memory

    logger.info(
        f"adjust_chunk_size_for_shared_memory "
        f"adjusted_chunk_size {util.INT(adjusted_chunk_size)} "
        f"chunk_size {util.INT(chunk_size)} "
        f"shared_memory_size {util.INT(shared_memory_size)} "
        f"num_processes {num_processes} "
        f"fair_share_of_shared_memory {util.INT(fair_share_of_shared_memory)} "
    )

    if adjusted_chunk_size <= 0:
        raise RuntimeError(
            f"adjust_chunk_size_for_shared_memory: chunk_size too small for shared memory.  "
            f"adjusted_chunk_size: {adjusted_chunk_size}"
        )

    return adjusted_chunk_size


def run_simulation(queue, step_info, resume_after, shared_data_buffer):
    """
    run step models as subtask

    called once to run each individual sub process in multiprocess step

    Unless actually resuming resuming, resume_after will be None for first step,
    and then FINAL for subsequent steps so pipelines opened to resume where previous step left off

    Parameters
    ----------
    queue : multiprocessing.Queue
    step_info : dict
        step_info for current step from multiprocess_steps
    resume_after : str or None
    shared_data_buffer : dict
        dict of shared data (e.g. skims and shadow_pricing)
    """

    # step_label = step_info['name']

    models = step_info["models"]
    chunk_size = step_info["chunk_size"]
    num_processes = step_info["num_processes"]

    chunk_size = adjust_chunk_size_for_shared_memory(
        chunk_size, shared_data_buffer, num_processes
    )

    inject.add_injectable("data_buffers", shared_data_buffer)
    inject.add_injectable("chunk_size", chunk_size)
    inject.add_injectable("num_processes", num_processes)

    if resume_after:
        info(f"resume_after {resume_after}")

        # if they specified a resume_after model, check to make sure it is checkpointed
        if (
            resume_after != LAST_CHECKPOINT
            and resume_after
            not in pipeline.get_checkpoints()[pipeline.CHECKPOINT_NAME].values
        ):
            # if not checkpointed, then fall back to last checkpoint
            info(f"resume_after checkpoint '{resume_after}' not in pipeline.")
            resume_after = LAST_CHECKPOINT

    pipeline.open_pipeline(resume_after)
    last_checkpoint = pipeline.last_checkpoint()

    if last_checkpoint in models:
        info(f"Resuming model run list after {last_checkpoint}")
        models = models[models.index(last_checkpoint) + 1 :]

    assert inject.get_injectable("preload_injectables", None)

    t0 = tracing.print_elapsed_time()
    for model in models:

        t1 = tracing.print_elapsed_time()

        try:
            pipeline.run_model(model)
        except Exception as e:
            warning(f"{type(e).__name__} exception running {model} model: {str(e)}")
            raise e

        tracing.log_runtime(model_name=model, start_time=t1)
        queue.put({"model": model, "time": time.time() - t1})

    tracing.print_elapsed_time("run (%s models)" % len(models), t0)

    # add checkpoint with final tables even if not intermediate checkpointing
    checkpoint_name = step_info["name"]
    pipeline.add_checkpoint(checkpoint_name)

    pipeline.close_pipeline()


"""
### multiprocessing sub-process entry points
"""


def mp_run_simulation(locutor, queue, injectables, step_info, resume_after, **kwargs):
    """
    mp entry point for run_simulation

    Parameters
    ----------
    locutor
    queue
    injectables
    step_info
    resume_after : bool
    kwargs : dict
        shared_data_buffers passed as kwargs to avoid picking dict
    """

    setup_injectables_and_logging(injectables, locutor=locutor)

    debug(
        f"mp_run_simulation {step_info['name']} locutor={inject.get_injectable('locutor', False)} "
    )

    try:

        if step_info["num_processes"] > 1:
            pipeline_prefix = multiprocessing.current_process().name
            logger.debug(f"injecting pipeline_file_prefix '{pipeline_prefix}'")
            inject.add_injectable("pipeline_file_prefix", pipeline_prefix)

        shared_data_buffer = kwargs
        run_simulation(queue, step_info, resume_after, shared_data_buffer)

        mem.log_global_hwm()  # subprocess

    except Exception as e:
        exception(f"{type(e).__name__} exception caught in mp_run_simulation: {str(e)}")
        raise e


def mp_apportion_pipeline(injectables, sub_proc_names, step_info):
    """
    mp entry point for apportion_pipeline

    Parameters
    ----------
    injectables : dict
        injectables from parent
    sub_proc_names : list of str
        names of the sub processes to apportion
    step_info : dict
        step_info for multiprocess_step we are apportioning
    """

    setup_injectables_and_logging(injectables)

    try:
        apportion_pipeline(sub_proc_names, step_info)
    except Exception as e:
        exception(
            f"{type(e).__name__} exception caught in mp_apportion_pipeline: {str(e)}"
        )
        raise e


def mp_setup_skims(injectables, **kwargs):
    """
    Sub process to load skim data into shared_data

    There is no particular necessity to perform this in a sub process instead of the parent
    except to ensure that this heavyweight task has no side-effects (e.g. loading injectables)

    Parameters
    ----------
    injectables : dict
        injectables from parent
    kwargs : dict
        shared_data_buffers passed as kwargs to avoid picking dict
    """

    setup_injectables_and_logging(injectables)

    info("mp_setup_skims")

    try:
        shared_data_buffer = kwargs

        network_los_preload = inject.get_injectable("network_los_preload", None)

        if network_los_preload is not None:
            network_los_preload.load_shared_data(shared_data_buffer)

    except Exception as e:
        exception(f"{type(e).__name__} exception caught in mp_setup_skims: {str(e)}")
        raise e


def mp_coalesce_pipelines(injectables, sub_proc_names, slice_info):
    """
    mp entry point for coalesce_pipeline

    Parameters
    ----------
    injectables : dict
        injectables from parent
    sub_proc_names : list of str
        names of the sub processes to apportion
    slice_info : dict
        slice_info from multiprocess_steps
    """

    setup_injectables_and_logging(injectables)

    try:
        coalesce_pipelines(sub_proc_names, slice_info)
    except Exception as e:
        exception(
            f"{type(e).__name__} exception caught in coalesce_pipelines: {str(e)}"
        )
        raise e


"""
### main (parent) process methods
"""


def allocate_shared_skim_buffers():
    """
    This is called by the main process to allocate shared memory buffer to share with subprocs

    Note: Buffers must be allocated BEFORE network_los.load_data

    Returns
    -------
    skim_buffers : dict {<skim_tag>: <multiprocessing.RawArray>}

    """

    info("allocate_shared_skim_buffer")

    network_los = inject.get_injectable("network_los_preload", None)
    if network_los is not None:
        skim_buffers = network_los.allocate_shared_skim_buffers()
    else:
        skim_buffers = {}

    return skim_buffers


def allocate_shared_shadow_pricing_buffers():
    """
    This is called by the main process to allocate memory buffer to share with subprocs

    Returns
    -------
        multiprocessing.RawArray
    """

    info("allocate_shared_shadow_pricing_buffers")

    shadow_pricing_info = inject.get_injectable("shadow_pricing_info", None)

    if shadow_pricing_info is not None:
        from activitysim.abm.tables import shadow_pricing

        shadow_pricing_buffers = shadow_pricing.buffers_for_shadow_pricing(
            shadow_pricing_info
        )
    else:
        shadow_pricing_buffers = {}

    return shadow_pricing_buffers


def run_sub_simulations(
    injectables,
    shared_data_buffers,
    step_info,
    process_names,
    resume_after,
    previously_completed,
    fail_fast,
):
    """
    Launch sub processes to run models in step according to specification in step_info.

    If resume_after is LAST_CHECKPOINT, then pick up where previous run left off, using breadcrumbs
    from previous run. If some sub-processes completed in the prior run, then skip rerunning them.

    If resume_after specifies a checkpiont, skip checkpoints that precede the resume_after

    Drop 'completed' breadcrumbs for this run as sub-processes terminate

    Wait for all sub-processes to terminate and return list of those that completed successfully.

    Parameters
    ----------
    injectables : dict
        values to inject in subprocesses
    shared_data_buffers : dict
        dict of shared_data for sub-processes (e.g. skim and shadow pricing data)
    step_info : dict
        step_info from run_list
    process_names : list of str
        list of sub process names to in parallel
    resume_after : str or None
        name of simulation to resume after, or LAST_CHECKPOINT to resume where previous run left off
    previously_completed : list of str
        names of processes that successfully completed in previous run
    fail_fast : bool
        whether to raise error if a sub process terminates with nonzero exitcode

    Returns
    -------
    completed : list of str
        names of sub_processes that completed successfully

    """

    def log_queued_messages():
        for process, queue in zip(procs, queues):
            while not queue.empty():
                msg = queue.get(block=False)
                model_name = msg["model"]
                info(
                    f"{process.name} {model_name} : {tracing.format_elapsed_time(msg['time'])}"
                )
                mem.trace_memory_info(f"{process.name}.{model_name}.completed")

    def check_proc_status():
        # we want to drop 'completed' breadcrumb when it happens, lest we terminate
        # if fail_fast flag is set raise
        for p in procs:
            if p.exitcode is None:
                pass  # still running
            elif p.exitcode == 0:
                # completed successfully
                if p.name not in completed:
                    info(f"process {p.name} completed")
                    completed.add(p.name)
                    drop_breadcrumb(step_name, "completed", list(completed))
                    mem.trace_memory_info(f"{p.name}.completed")
            else:
                # process failed
                if p.name not in failed:
                    warning(f"process {p.name} failed with exitcode {p.exitcode}")
                    failed.add(p.name)
                    mem.trace_memory_info(f"{p.name}.failed")
                    if fail_fast:
                        warning(f"fail_fast terminating remaining running processes")
                        for op in procs:
                            if op.exitcode is None:
                                try:
                                    info(f"terminating process {op.name}")
                                    op.terminate()
                                except Exception as e:
                                    info(f"error terminating process {op.name}: {e}")
                        raise RuntimeError("Process %s failed" % (p.name,))

    step_name = step_info["name"]

    t0 = tracing.print_elapsed_time()
    info(f"run_sub_simulations step {step_name} models resume_after {resume_after}")

    # if resuming and some processes completed successfully in previous run
    if previously_completed:
        assert resume_after is not None
        assert set(previously_completed).issubset(set(process_names))

        if resume_after == LAST_CHECKPOINT:
            # if we are resuming where previous run left off, then we can skip running
            # any subprocudures that successfully complete the previous run
            process_names = [
                name for name in process_names if name not in previously_completed
            ]
            info(
                f"step {step_name}: skipping {len(previously_completed)} previously completed subprocedures"
            )
        else:
            # if we are resuming after a specific model, then force all subprocesses to run
            # (assuming if they specified a model, they really want everything after that to run)
            previously_completed = []

    # if not the first step, resume_after the last checkpoint from the previous step
    if resume_after is None and step_info["step_num"] > 0:
        resume_after = LAST_CHECKPOINT

    num_simulations = len(process_names)
    procs = []
    queues = []

    completed = set(previously_completed)
    failed = set([])  # so we can log process failure first time it happens
    drop_breadcrumb(step_name, "completed", list(completed))

    for i, process_name in enumerate(process_names):
        q = multiprocessing.Queue()
        locutor = i == 0

        args = OrderedDict(
            locutor=locutor,
            queue=q,
            injectables=injectables,
            step_info=step_info,
            resume_after=resume_after,
        )

        # debug(f"create_process {process_name} target={mp_run_simulation}")
        # for k in args:
        #     debug(f"create_process {process_name} arg {k}={args[k]}")
        # for k in shared_data_buffers:
        #     debug(f"create_process {process_name} shared_data_buffers {k}={shared_data_buffers[k]}")

        p = multiprocessing.Process(
            target=mp_run_simulation,
            name=process_name,
            args=(
                locutor,
                q,
                injectables,
                step_info,
                resume_after,
            ),
            kwargs=shared_data_buffers,
        )

        procs.append(p)
        queues.append(q)

    # - start processes
    for i, p in zip(list(range(num_simulations)), procs):
        info(f"start process {p.name}")
        p.start()

        """
        windows mmap does not handle multiple simultaneous calls from different processes for the same tagname.
        Process start causes a call to mmap to initialize the wrapper for the anonymous shared memory arrays
        in the shared_data_buffers kwargs. some of the processses fail with WinError 1450 (or similar error)
        OSError: [WinError 1450] Insufficient system resources exist to complete the requested service.
        Judging by the commented-out assert, this (or a related) issue may have been around in some form for a while.

        def __setstate__(self, state):
            self.size, self.name = self._state = state
            # Reopen existing mmap
            self.buffer = mmap.mmap(-1, self.size, tagname=self.name)
            # XXX Temporarily preventing buildbot failures while determining
            # XXX the correct long-term fix. See issue 23060
            #assert _winapi.GetLastError() == _winapi.ERROR_ALREADY_EXISTS
        """
        if sys.platform == "win32":
            time.sleep(1)

        mem.trace_memory_info(f"{p.name}.start")

    while multiprocessing.active_children():
        # log queued messages as they are received
        log_queued_messages()
        # monitor sub process status and drop breadcrumbs or fail_fast as they terminate
        check_proc_status()
        # monitor memory usage
        mem.trace_memory_info(
            "run_sub_simulations.idle", trace_ticks=mem.MEM_PARENT_TRACE_TICK_LEN
        )
        time.sleep(1)

    # clean up any messages or breadcrumbs that occurred while we slept
    log_queued_messages()
    check_proc_status()

    # no need to join() explicitly since multiprocessing.active_children joins completed procs

    for p in procs:
        assert p.exitcode is not None
        if p.exitcode:
            error(f"Process %s failed with exitcode {p.exitcode}")
            assert p.name in failed
        else:
            info(f"Process {p.name} completed with exitcode {p.exitcode}")
            assert p.name in completed

    t0 = tracing.print_elapsed_time("run_sub_simulations step %s" % step_name, t0)

    return list(completed)


def run_sub_task(p):
    """
    Run process p synchroneously,

    Return when sub process terminates, or raise error if exitcode is nonzero

    Parameters
    ----------
    p : multiprocessing.Process
    """
    info(f"#run_model running sub_process {p.name}")

    mem.trace_memory_info(f"{p.name}.start")

    t0 = tracing.print_elapsed_time()
    p.start()

    while multiprocessing.active_children():
        mem.trace_memory_info(
            "run_sub_simulations.idle", trace_ticks=mem.MEM_PARENT_TRACE_TICK_LEN
        )
        time.sleep(1)

    # no need to join explicitly since multiprocessing.active_children joins completed procs
    # p.join()

    t0 = tracing.print_elapsed_time("#run_model sub_process %s" % p.name, t0)
    # info(f'{p.name}.exitcode = {p.exitcode}')

    mem.trace_memory_info(f"run_model {p.name} completed")

    if p.exitcode:
        error(f"Process {p.name} returned exitcode {p.exitcode}")
        raise RuntimeError("Process %s returned exitcode %s" % (p.name, p.exitcode))


def drop_breadcrumb(step_name, crumb, value=True):
    """
    Add (crumb: value) to specified step in breadcrumbs and flush breadcrumbs to file
    run can be resumed with resume_after

    Breadcrumbs provides a record of steps that have been run for use when resuming
    Basically, we want to know which steps have been run, which phases completed
    (i.e. apportion, simulate, coalesce). For multi-processed simulate steps, we also
    want to know which sub-processes completed successfully, because if resume_after
    is LAST_CHECKPOINT we don't have to rerun the successful ones.

    Parameters
    ----------
    step_name : str
    crumb : str
    value : yaml-writable value

    Returns
    -------

    """
    breadcrumbs = inject.get_injectable("breadcrumbs", OrderedDict())
    breadcrumbs.setdefault(step_name, {"name": step_name})[crumb] = value
    inject.add_injectable("breadcrumbs", breadcrumbs)
    write_breadcrumbs(breadcrumbs)


def run_multiprocess(injectables):
    """
    run the steps in run_list, possibly resuming after checkpoint specified by resume_after

    we never open the pipeline since that is all done within multi-processing steps -
    mp_apportion_pipeline, run_sub_simulations, mp_coalesce_pipelines -
    each of which opens the pipeline/s and closes it/them within the sub-process
    This 'feature' makes the pipeline state a bit opaque to us, for better or worse...

    Steps may be either single or multi process.
    For multi-process steps, we need to apportion pipelines before running sub processes
    and coalesce them afterwards

    injectables arg allows propagation of setting values that were overridden on the command line
    (parent process command line arguments are not available to sub-processes in Windows)

    * allocate shared data buffers for skims and shadow_pricing
    * load shared skim data from OMX files
    * run each (single or multiprocess) step in turn

    Drop breadcrumbs along the way to facilitate resuming in a later run

    Parameters
    ----------
    run_list : dict
        annotated run_list  (including prior run breadcrumbs if resuming)
    injectables : dict
        dict of values to inject in sub-processes
    """

    mem.trace_memory_info("run_multiprocess.start")

    run_list = get_run_list()

    if not run_list["multiprocess"]:
        raise RuntimeError(
            "run_multiprocess called but multiprocess flag is %s"
            % run_list["multiprocess"]
        )

    old_breadcrumbs = run_list.get("breadcrumbs", {})

    # raise error if any sub-process fails without waiting for others to complete
    fail_fast = setting("fail_fast")
    info(f"run_multiprocess fail_fast: {fail_fast}")

    def skip_phase(phase):
        skip = old_breadcrumbs and old_breadcrumbs.get(step_name, {}).get(phase, False)
        if skip:
            info(f"Skipping {step_name} {phase}")
        return skip

    def find_breadcrumb(crumb, default=None):
        return old_breadcrumbs.get(step_name, {}).get(crumb, default)

    # - allocate shared data
    shared_data_buffers = {}

    mem.trace_memory_info("allocate_shared_skim_buffer.before")

    t0 = tracing.print_elapsed_time()
    shared_data_buffers.update(allocate_shared_skim_buffers())
    t0 = tracing.print_elapsed_time("allocate shared skim buffer", t0)
    mem.trace_memory_info("allocate_shared_skim_buffer.completed")

    # combine shared_skim_buffer and shared_shadow_pricing_buffer in shared_data_buffer
    t0 = tracing.print_elapsed_time()
    shared_data_buffers.update(allocate_shared_shadow_pricing_buffers())
    t0 = tracing.print_elapsed_time("allocate shared shadow_pricing buffer", t0)
    mem.trace_memory_info("allocate_shared_shadow_pricing_buffers.completed")

    # - mp_setup_skims
    if len(shared_data_buffers) > 0:
        run_sub_task(
            multiprocessing.Process(
                target=mp_setup_skims,
                name="mp_setup_skims",
                args=(injectables,),
                kwargs=shared_data_buffers,
            )
        )
        t0 = tracing.print_elapsed_time("setup shared_data_buffers", t0)
        mem.trace_memory_info("mp_setup_skims.completed")

    # - for each step in run list
    for step_info in run_list["multiprocess_steps"]:

        step_name = step_info["name"]

        num_processes = step_info["num_processes"]
        slice_info = step_info.get("slice", None)

        if num_processes == 1:
            sub_proc_names = [step_name]
        else:
            sub_proc_names = ["%s_%s" % (step_name, i) for i in range(num_processes)]

        # - mp_apportion_pipeline
        if not skip_phase("apportion") and num_processes > 1:
            run_sub_task(
                multiprocessing.Process(
                    target=mp_apportion_pipeline,
                    name="%s_apportion" % step_name,
                    args=(injectables, sub_proc_names, step_info),
                )
            )
        drop_breadcrumb(step_name, "apportion")

        # - run_sub_simulations
        if not skip_phase("simulate"):
            resume_after = step_info.get("resume_after", None)

            previously_completed = find_breadcrumb("completed", default=[])

            completed = run_sub_simulations(
                injectables,
                shared_data_buffers,
                step_info,
                sub_proc_names,
                resume_after,
                previously_completed,
                fail_fast,
            )

            if len(completed) != num_processes:
                raise RuntimeError(
                    "%s processes failed in step %s"
                    % (num_processes - len(completed), step_name)
                )
        drop_breadcrumb(step_name, "simulate")

        # - mp_coalesce_pipelines
        if not skip_phase("coalesce") and num_processes > 1:
            run_sub_task(
                multiprocessing.Process(
                    target=mp_coalesce_pipelines,
                    name="%s_coalesce" % step_name,
                    args=(injectables, sub_proc_names, slice_info),
                )
            )
        drop_breadcrumb(step_name, "coalesce")

    # add checkpoint with final tables even if not intermediate checkpointing
    if not pipeline.intermediate_checkpoint():
        pipeline.open_pipeline("_")
        pipeline.add_checkpoint(pipeline.FINAL_CHECKPOINT_NAME)
        pipeline.close_pipeline()

    mem.log_global_hwm()  # main process


def get_breadcrumbs(run_list):
    """
    Read, validate, and annotate breadcrumb file from previous run

    if resume_after specifies a model name, we need to determine which step it falls within,
    drop any subsequent steps, and set the 'simulate' and 'coalesce' to None so

    Extract from breadcrumbs file showing completed mp_households step with 2 processes:
    ::

        - apportion: true
          completed: [mp_households_0, mp_households_1]
          name: mp_households
          simulate: true
          coalesce: true


    Parameters
    ----------
    run_list : dict
        validated and annotated run_list from settings

    Returns
    -------
    breadcrumbs : dict
        validated and annotated breadcrumbs file from previous run
    """

    resume_after = run_list["resume_after"]
    assert resume_after is not None

    # - read breadcrumbs file from previous run
    breadcrumbs = read_breadcrumbs()

    # - can't resume multiprocess without breadcrumbs file
    if not breadcrumbs:
        error(f"empty breadcrumbs for resume_after '{resume_after}'")
        raise RuntimeError("empty breadcrumbs for resume_after '%s'" % resume_after)

    # if resume_after is specified by name
    if resume_after != LAST_CHECKPOINT:

        # breadcrumbs for steps from previous run
        previous_steps = list(breadcrumbs.keys())

        # find the run_list step resume_after is in
        resume_step = next(
            (
                step
                for step in run_list["multiprocess_steps"]
                if resume_after in step["models"]
            ),
            None,
        )

        resume_step_name = resume_step["name"]

        if resume_step_name not in previous_steps:
            error(f"resume_after model '{resume_after}' not in breadcrumbs")
            raise RuntimeError(
                "resume_after model '%s' not in breadcrumbs" % resume_after
            )

        # drop any previous_breadcrumbs steps after resume_step
        for step in previous_steps[previous_steps.index(resume_step_name) + 1 :]:
            del breadcrumbs[step]

        # if resume_after is not the last model in the step
        # then we need to rerun the simulations in that step, even if they succeeded
        if resume_after in resume_step["models"][:-1]:
            if "simulate" in breadcrumbs[resume_step_name]:
                breadcrumbs[resume_step_name]["simulate"] = None
            if "coalesce" in breadcrumbs[resume_step_name]:
                breadcrumbs[resume_step_name]["coalesce"] = None

    multiprocess_step_names = [step["name"] for step in run_list["multiprocess_steps"]]
    if list(breadcrumbs.keys()) != multiprocess_step_names[: len(breadcrumbs)]:
        raise RuntimeError(
            "last run steps don't match run list: %s" % list(breadcrumbs.keys())
        )

    return breadcrumbs


def get_run_list():
    """
    validate and annotate run_list from settings

    Assign defaults to missing settings (e.g. chunk_size)
    Build individual step model lists based on step starts
    If resuming, read breadcrumbs file for info on previous run execution status

    # annotated run_list with two steps, the second with 2 processors

    ::

        resume_after: None
        multiprocess: True
        models:
          -  initialize_landuse
          -  compute_accessibility
          -  initialize_households
          -  school_location
          -  workplace_location

        multiprocess_steps:
          step: mp_initialize
            begin: initialize_landuse
            name: mp_initialize
            models:
               - initialize_landuse
               - compute_accessibility
               - initialize_households
            num_processes: 1
            chunk_size: 0
            step_num: 0
          step: mp_households
            begin: school_location
            slice: {'tables': ['households', 'persons']}
            name: mp_households
            models:
               - school_location
               - workplace_location
            num_processes: 2
            chunk_size: 10000
            step_num: 1

    Returns
    -------
    run_list : dict
        validated and annotated run_list
    """

    models = setting("models", [])
    multiprocess_steps = setting("multiprocess_steps", [])

    resume_after = inject.get_injectable("resume_after", None) or setting(
        "resume_after", None
    )
    multiprocess = inject.get_injectable("multiprocess", False) or setting(
        "multiprocess", False
    )

    # default settings that can be overridden by settings in individual steps
    global_chunk_size = setting("chunk_size", 0) or 0
    default_mp_processes = setting("num_processes", 0) or int(
        1 + multiprocessing.cpu_count() / 2.0
    )

    if multiprocess and multiprocessing.cpu_count() == 1:
        warning("Can't multiprocess because there is only 1 cpu")

    run_list = {
        "models": models,
        "resume_after": resume_after,
        "multiprocess": multiprocess,
        # 'multiprocess_steps': multiprocess_steps  # add this later if multiprocess
    }

    if not models or not isinstance(models, list):
        raise RuntimeError("No models list in settings file")
    if resume_after == models[-1]:
        raise RuntimeError(
            "resume_after '%s' is last model in models list" % resume_after
        )

    if multiprocess:

        if not multiprocess_steps:
            raise RuntimeError(
                "multiprocess setting is %s but no multiprocess_steps setting"
                % multiprocess
            )

        # check step name, num_processes, chunk_size and presence of slice info
        num_steps = len(multiprocess_steps)
        step_names = set()
        for istep in range(num_steps):
            step = multiprocess_steps[istep]

            step["step_num"] = istep

            # - validate step name
            name = step.get("name", None)
            if not name:
                raise RuntimeError(
                    "missing name for step %s" " in multiprocess_steps" % istep
                )
            if name in step_names:
                raise RuntimeError(
                    "duplicate step name %s" " in multiprocess_steps" % name
                )
            if name in models:
                raise RuntimeError(
                    f"multiprocess_steps step name '{name}' cannot also be a model name"
                )

            step_names.add(name)

            # - validate num_processes and assign default
            num_processes = step.get("num_processes", 0)

            if not isinstance(num_processes, int) or num_processes < 0:
                raise RuntimeError(
                    "bad value (%s) for num_processes for step %s"
                    " in multiprocess_steps" % (num_processes, name)
                )

            if "slice" in step:
                if num_processes == 0:
                    info(f"Setting num_processes = {num_processes} for step {name}")
                    num_processes = default_mp_processes
                if num_processes > multiprocessing.cpu_count():
                    warning(
                        f"num_processes setting ({num_processes}) "
                        f"greater than cpu count ({ multiprocessing.cpu_count()})"
                    )
            else:
                if num_processes == 0:
                    num_processes = 1
                if num_processes > 1:
                    raise RuntimeError(
                        "num_processes > 1 but no slice info for step %s"
                        " in multiprocess_steps" % name
                    )

            multiprocess_steps[istep]["num_processes"] = num_processes

            # - validate chunk_size and assign default
            chunk_size = step.get("chunk_size", None)
            if chunk_size is None:
                if global_chunk_size > 0 and num_processes > 1:
                    chunk_size = int(round(global_chunk_size / num_processes))
                    chunk_size = max(chunk_size, 1)
                else:
                    chunk_size = global_chunk_size

            multiprocess_steps[istep]["chunk_size"] = chunk_size

        # - determine index in models list of step starts
        start_tag = "begin"
        starts = [0] * len(multiprocess_steps)
        for istep in range(num_steps):
            step = multiprocess_steps[istep]

            name = step["name"]

            slice = step.get("slice", None)
            if slice:
                if "tables" not in slice:
                    raise RuntimeError(
                        "missing tables list for step %s"
                        " in multiprocess_steps" % istep
                    )

            start = step.get(start_tag, None)
            if not name:
                raise RuntimeError(
                    "missing %s tag for step '%s' (%s)"
                    " in multiprocess_steps" % (start_tag, name, istep)
                )
            if start not in models:
                raise RuntimeError(
                    "%s tag '%s' for step '%s' (%s) not in models list"
                    % (start_tag, start, name, istep)
                )

            starts[istep] = models.index(start)

            if istep == 0 and starts[istep] != 0:
                raise RuntimeError(
                    "%s tag '%s' for first step '%s' (%s)"
                    " is not first model in models list"
                    % (start_tag, start, name, istep)
                )

            if istep > 0 and starts[istep] <= starts[istep - 1]:
                raise RuntimeError(
                    "%s tag '%s' for step '%s' (%s)"
                    " falls before that of prior step in models list"
                    % (start_tag, start, name, istep)
                )

            # remember there should always be a final checkpoint with same name as multiprocess_step name
            multiprocess_steps[istep][
                "last_checkpoint_in_previous_multiprocess_step"
            ] = (multiprocess_steps[istep - 1].get("name") if istep > 0 else None)

        # - build individual step model lists based on starts
        starts.append(len(models))  # so last step gets remaining models in list
        for istep in range(num_steps):
            step_models = models[starts[istep] : starts[istep + 1]]

            if step_models[-1][0] == LAST_CHECKPOINT:
                raise RuntimeError(
                    "Final model '%s' in step %s models list not checkpointed"
                    % (step_models[-1], name)
                )

            multiprocess_steps[istep]["models"] = step_models

        run_list["multiprocess_steps"] = multiprocess_steps

        # - add resume breadcrumbs
        if resume_after:
            breadcrumbs = get_breadcrumbs(run_list)
            if breadcrumbs:
                run_list["breadcrumbs"] = breadcrumbs

                # - add resume_after to last step
                if resume_after is not None:
                    # get_breadcrumbs should have deleted breadcrumbs for any subsequent steps
                    istep = len(breadcrumbs) - 1
                    assert (
                        resume_after == LAST_CHECKPOINT
                        or resume_after in multiprocess_steps[istep]["models"]
                    )
                    multiprocess_steps[istep]["resume_after"] = resume_after

    # - write run list to output dir
    # use log_file_path so we use (optional) log subdir and prefix process name
    with config.open_log_file("run_list.txt", "w") as f:
        print_run_list(run_list, f)

    return run_list


def print_run_list(run_list, output_file=None):
    """
    Print run_list to stdout or file (informational - not read back in)

    Parameters
    ----------
    run_list : dict
    output_file : open file
    """

    if output_file is None:
        output_file = sys.stdout

    print("resume_after:", run_list["resume_after"], file=output_file)
    print("multiprocess:", run_list["multiprocess"], file=output_file)

    print("models:", file=output_file)
    for m in run_list["models"]:
        print("  - ", m, file=output_file)

    # - print multiprocess_steps
    if run_list["multiprocess"]:
        print("\nmultiprocess_steps:", file=output_file)
        for step in run_list["multiprocess_steps"]:
            print("  step:", step["name"], file=output_file)
            for k in step:
                if isinstance(step[k], list):
                    print("    %s:" % k, file=output_file)
                    for v in step[k]:
                        print("       -", v, file=output_file)
                else:
                    print("    %s: %s" % (k, step[k]), file=output_file)

    # - print breadcrumbs
    breadcrumbs = run_list.get("breadcrumbs")
    if breadcrumbs:
        print("\nbreadcrumbs:", file=output_file)
        for step_name in breadcrumbs:
            step = breadcrumbs[step_name]
            print("  step:", step_name, file=output_file)
            for k in step:
                if isinstance(k, str):
                    print("    ", k, step[k], file=output_file)
                else:
                    print("    ", k, file=output_file)
                    for v in step[k]:
                        print("      ", v, file=output_file)


def breadcrumbs_file_path():
    # return path to breadcrumbs file in output_dir
    return config.build_output_file_path("breadcrumbs.yaml")


def read_breadcrumbs():
    """
    Read breadcrumbs file from previous run

    write_breadcrumbs wrote OrderedDict steps as list so ordered is preserved
    (step names are duplicated in steps)

    Returns
    -------
    breadcrumbs : OrderedDict
    """
    file_path = breadcrumbs_file_path()
    if not os.path.exists(file_path):
        raise IOError("Could not find saved breadcrumbs file '%s'" % file_path)
    with open(file_path, "r") as f:
        breadcrumbs = yaml.load(f, Loader=yaml.SafeLoader)
    # convert array to ordered dict keyed by step name
    breadcrumbs = OrderedDict([(step["name"], step) for step in breadcrumbs])
    return breadcrumbs


def write_breadcrumbs(breadcrumbs):
    """
    Write breadcrumbs file with execution history of multiprocess run

    Write steps as array so order is preserved (step names are duplicated in steps)

    Extract from breadcrumbs file showing completed mp_households step with 2 processes:
    ::

        - apportion: true
          coalesce: true
          completed: [mp_households_0, mp_households_1]
          name: mp_households
          simulate: true

    Parameters
    ----------
    breadcrumbs : OrderedDict
    """
    with open(breadcrumbs_file_path(), "w") as f:
        # write ordered dict as array
        breadcrumbs = [step for step in list(breadcrumbs.values())]
        yaml.dump(breadcrumbs, f)


def if_sub_task(if_is, if_isnt):
    """
    select one of two values depending whether current process is primary process or subtask

    This is primarily intended for use in yaml files to select between (e.g.) logging levels
    so main log file can display only warnings and errors from subtasks

    In yaml file, it can be used like this:

    level: !!python/object/apply:activitysim.core.mp_tasks.if_sub_task [WARNING, NOTSET]


    Parameters
    ----------
    if_is : (any type) value to return if process is a subtask
    if_isnt : (any type) value to return if process is not a subtask

    Returns
    -------
    (any type) (one of parameters if_is or if_isnt)
    """

    return if_is if inject.get_injectable("is_sub_task", False) else if_isnt

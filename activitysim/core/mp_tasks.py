# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from future.utils import iteritems

import sys
import os
import time
import logging
import multiprocessing
import cProfile

from collections import OrderedDict

import yaml
import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config

from activitysim.core import chunk
from activitysim.core import mem

from activitysim.core.config import setting

# activitysim.abm imported for its side-effects (dependency injection)
from activitysim import abm

from activitysim.abm.tables.skims import get_skim_info
from activitysim.abm.tables.skims import buffer_for_skims
from activitysim.abm.tables.skims import load_skims


logger = logging.getLogger('activitysim')


"""
    child process methods (called within sub process)
"""


def filter_warnings():

    if setting('strict', False):  # noqa: E402
        import warnings
        warnings.filterwarnings('error', category=Warning)
        warnings.filterwarnings('default', category=PendingDeprecationWarning, module='future')
        warnings.filterwarnings('default', category=FutureWarning, module='pandas')
        warnings.filterwarnings('default', category=RuntimeWarning, module='numpy')


def pipeline_table_keys(pipeline_store, checkpoint_name=None):

    checkpoints = pipeline_store[pipeline.CHECKPOINT_TABLE_NAME]

    if checkpoint_name:
        # specified checkpoint row as series
        i = checkpoints[checkpoints[pipeline.CHECKPOINT_NAME] == checkpoint_name].index[0]
        checkpoint = checkpoints.loc[i]
    else:
        # last checkpoint row as series
        checkpoint = checkpoints.iloc[-1]
        checkpoint_name = checkpoint.loc[pipeline.CHECKPOINT_NAME]

    # series with table name as index and checkpoint_name as value
    checkpoint_tables = checkpoint[~checkpoint.index.isin(pipeline.NON_TABLE_COLUMNS)]

    # omit dropped tables with empty checkpoint name
    checkpoint_tables = checkpoint_tables[checkpoint_tables != '']

    # hdf5 key is <table_name>/<checkpoint_name>
    checkpoint_tables = {table_name: pipeline.pipeline_table_key(table_name, checkpoint_name)
                         for table_name, checkpoint_name in iteritems(checkpoint_tables)}

    # checkpoint name and series mapping table name to hdf5 key for tables in that checkpoint
    return checkpoint_name, checkpoint_tables


def build_slice_rules(slice_info, tables):

    slicer_table_names = slice_info['tables']
    slicer_table_exceptions = slice_info.get('except', [])
    primary_slicer = slicer_table_names[0]

    if primary_slicer not in tables:
        raise RuntimeError("primary slice table '%s' not in pipeline" % primary_slicer)

    logger.debug("build_slice_rules tables %s", list(tables.keys()))
    logger.debug("build_slice_rules primary_slicer %s", primary_slicer)
    logger.debug("build_slice_rules slicer_table_names %s", slicer_table_names)
    logger.debug("build_slice_rules slicer_table_exceptions %s", slicer_table_exceptions)

    # dict mapping slicer table_name to index name
    # (also presumed to be name of ref col name in referencing table)
    slicer_ref_cols = OrderedDict()

    # build slice rules for loaded tables
    slice_rules = OrderedDict()
    for table_name, df in iteritems(tables):

        rule = {}
        if table_name == primary_slicer:
            # slice primary apportion table
            rule = {'slice_by': 'primary'}
        elif table_name in slicer_table_exceptions:
            rule['slice_by'] = None
        else:
            for slicer_table_name in slicer_ref_cols:
                if df.index.name == tables[slicer_table_name].index.name:
                    # slice df with same index name as a known slicer
                    rule = {'slice_by': 'index', 'source': slicer_table_name}
                else:
                    # if df has a column with same name as the ref_col (index) of a slicer?
                    try:
                        source, ref_col = next((t, c)
                                               for t, c in iteritems(slicer_ref_cols)
                                               if c in df.columns)
                        # then we can use that table to slice this df
                        rule = {'slice_by': 'column',
                                'column': ref_col,
                                'source': source}
                    except StopIteration:
                        rule['slice_by'] = None

        if rule['slice_by']:
            # cascade sliceability
            slicer_ref_cols[table_name] = df.index.name

        slice_rules[table_name] = rule

    for table_name in slice_rules:
        logger.debug("%s: %s", table_name, slice_rules[table_name])

    return slice_rules


def apportion_pipeline(sub_job_names, slice_info):

    pipeline_file_name = inject.get_injectable('pipeline_file_name')

    tables = OrderedDict([(table_name, None) for table_name in slice_info['tables']])

    # get last checkpoint from first job pipeline
    pipeline_path = config.build_output_file_path(pipeline_file_name)

    logger.debug("apportion_pipeline pipeline_path: %s", pipeline_path)

    # load all tables from pipeline
    with pd.HDFStore(pipeline_path, mode='r') as pipeline_store:

        checkpoints_df = pipeline_store[pipeline.CHECKPOINT_TABLE_NAME]

        # hdf5_keys is a dict mapping table_name to pipeline hdf5_key
        checkpoint_name, hdf5_keys = pipeline_table_keys(pipeline_store)

        # ensure presence of slicer tables in pipeline
        for table_name in tables:
            if table_name not in hdf5_keys:
                raise RuntimeError("slicer table %s not found in pipeline" % table_name)

        # load all tables from pipeline
        for table_name, hdf5_key in iteritems(hdf5_keys):
            # new checkpoint for all tables the same
            checkpoints_df[table_name] = checkpoint_name
            # load the dataframe
            tables[table_name] = pipeline_store[hdf5_key]

            logger.debug("loaded table %s %s", table_name, tables[table_name].shape)

    # keep only the last row of checkpoints and patch the last checkpoint name
    checkpoints_df = checkpoints_df.tail(1).copy()
    checkpoints_df[list(tables.keys())] = checkpoint_name

    # build slice rules for loaded tables
    slice_rules = build_slice_rules(slice_info, tables)

    # allocate sliced tables
    num_sub_jobs = len(sub_job_names)
    for i in range(num_sub_jobs):

        process_name = sub_job_names[i]
        pipeline_path = config.build_output_file_path(pipeline_file_name, use_prefix=process_name)

        # remove existing file
        try:
            os.unlink(pipeline_path)
        except OSError:
            pass

        with pd.HDFStore(pipeline_path, mode='a') as pipeline_store:

            # remember sliced_tables so we can cascade slicing to other tables
            sliced_tables = {}
            for table_name, rule in iteritems(slice_rules):

                df = tables[table_name]

                if rule['slice_by'] == 'primary':
                    # slice primary apportion table by num_sub_jobs strides
                    # this hopefully yields a more random distribution
                    # (e.g.) households are ordered by size in input store
                    primary_df = df[np.asanyarray(list(range(df.shape[0]))) % num_sub_jobs == i]
                    sliced_tables[table_name] = primary_df
                elif rule['slice_by'] == 'index':
                    # slice a table with same index name as a known slicer
                    source_df = sliced_tables[rule['source']]
                    sliced_tables[table_name] = df.loc[source_df.index]
                elif rule['slice_by'] == 'column':
                    # slice a table with a recognized slicer_column
                    source_df = sliced_tables[rule['source']]
                    sliced_tables[table_name] = df[df[rule['column']].isin(source_df.index)]
                elif rule['slice_by'] is None:
                    # not all tables should be sliced (e.g. land_use)
                    sliced_tables[table_name] = df
                else:
                    raise RuntimeError("Unrecognized slice rule '%s' for table %s" %
                                       (rule['slice_by'], table_name))

                hdf5_key = pipeline.pipeline_table_key(table_name, checkpoint_name)

                logger.debug("writing %s (%s) to %s in %s",
                             table_name, sliced_tables[table_name].shape, hdf5_key, pipeline_path)
                pipeline_store[hdf5_key] = sliced_tables[table_name]

            logger.debug("writing checkpoints (%s) to %s in %s",
                         checkpoints_df.shape, pipeline.CHECKPOINT_TABLE_NAME, pipeline_path)
            pipeline_store[pipeline.CHECKPOINT_TABLE_NAME] = checkpoints_df


def coalesce_pipelines(sub_process_names, slice_info, use_prefix=True):

    def sub_pipeline_path(name):
        if use_prefix:
            # name is prefix
            path = config.build_output_file_path(pipeline_file_name, use_prefix=name)
        elif os.path.exists(name):
            # check if name is valid path
            path = name
        else:
            # otherwise expect to find it on output dir
            path = config.build_output_file_path(name)

        return path

    pipeline_file_name = inject.get_injectable('pipeline_file_name')

    logger.debug("coalesce_pipelines to: %s", pipeline_file_name)

    # tables that are identical in every pipeline and so don't need to be concatenated

    tables = OrderedDict([(table_name, None) for table_name in slice_info['tables']])

    # read all tables from first process pipeline
    pipeline_path = sub_pipeline_path(sub_process_names[0])
    with pd.HDFStore(pipeline_path, mode='r') as pipeline_store:

        # hdf5_keys is a dict mapping table_name to pipeline hdf5_key
        checkpoint_name, hdf5_keys = pipeline_table_keys(pipeline_store)

        for table_name, hdf5_key in iteritems(hdf5_keys):
            logger.debug("loading table %s %s", table_name, hdf5_key)
            tables[table_name] = pipeline_store[hdf5_key]

    # use slice rules followed by apportion_pipeline to identify singleton tables
    slice_rules = build_slice_rules(slice_info, tables)
    singleton_table_names = [t for t, rule in iteritems(slice_rules) if rule['slice_by'] is None]
    singleton_tables = {t: tables[t] for t in singleton_table_names}
    omnibus_keys = {t: k for t, k in iteritems(hdf5_keys) if t not in singleton_table_names}

    logger.debug("coalesce_pipelines to: %s", pipeline_file_name)
    logger.debug("singleton_table_names: %s", singleton_table_names)
    logger.debug("omnibus_keys: %s", omnibus_keys)

    # concat omnibus tables from all sub_processes
    omnibus_tables = {table_name: [] for table_name in omnibus_keys}
    for process_name in sub_process_names:
        pipeline_path = sub_pipeline_path(process_name)
        logger.info("coalesce pipeline %s", pipeline_path)

        with pd.HDFStore(pipeline_path, mode='r') as pipeline_store:
            for table_name, hdf5_key in iteritems(omnibus_keys):
                omnibus_tables[table_name].append(pipeline_store[hdf5_key])

    pipeline.open_pipeline()

    for table_name in singleton_tables:
        df = singleton_tables[table_name]
        logger.info("adding singleton table %s %s", table_name, df.shape)
        pipeline.replace_table(table_name, df)
    for table_name in omnibus_tables:
        df = pd.concat(omnibus_tables[table_name], sort=False)
        logger.info("adding omnibus table %s %s", table_name, df.shape)
        pipeline.replace_table(table_name, df)

    pipeline.add_checkpoint(checkpoint_name)

    pipeline.close_pipeline()


def load_skim_data(skim_buffer):

    logger.info("load_skim_data")

    omx_file_path = config.data_file_path(setting('skims_file'))
    tags_to_load = setting('skim_time_periods')['labels']

    skim_info = get_skim_info(omx_file_path, tags_to_load)
    load_skims(omx_file_path, skim_info, skim_buffer)


def allocate_shared_skim_buffer():
    """
    This is called by the main process and allocate memory buffer to share with subprocs

    Returns
    -------
        multiprocessing.RawArray
    """

    logger.info("allocate_shared_skim_buffer")

    omx_file_path = config.data_file_path(setting('skims_file'))
    tags_to_load = setting('skim_time_periods')['labels']

    # select the skims to load
    skim_info = get_skim_info(omx_file_path, tags_to_load)
    skim_buffer = buffer_for_skims(skim_info, shared=True)

    return skim_buffer


def setup_injectables_and_logging(injectables):

    for k, v in iteritems(injectables):
        inject.add_injectable(k, v)

    inject.add_injectable("is_sub_task", True)

    filter_warnings()

    process_name = multiprocessing.current_process().name
    inject.add_injectable("log_file_prefix", process_name)
    tracing.config_logger()


def run_simulation(queue, step_info, resume_after, skim_buffer):

    models = step_info['models']
    chunk_size = step_info['chunk_size']
    step_label = step_info['name']

    inject.add_injectable('skim_buffer', skim_buffer)
    inject.add_injectable("chunk_size", chunk_size)

    if resume_after:
        logger.info('resume_after %s', resume_after)

        # if they specified a resume_after model, check to make sure it is checkpointed
        if resume_after != '_' \
                and resume_after not in pipeline.get_checkpoints()[pipeline.CHECKPOINT_NAME].values:
            # if not checkpointed, then fall back to last checkpoint
            logger.info("resume_after checkpoint '%s' not in pipeline.", resume_after)
            resume_after = '_'

    pipeline.open_pipeline(resume_after)
    last_checkpoint = pipeline.last_checkpoint()

    if last_checkpoint in models:
        logger.info("Resuming model run list after %s", last_checkpoint)
        models = models[models.index(last_checkpoint) + 1:]

    # preload any bulky injectables (e.g. skims) not in pipeline
    inject.get_injectable('preload_injectables', None)

    t0 = tracing.print_elapsed_time()
    for model in models:

        t1 = tracing.print_elapsed_time()
        pipeline.run_model(model)

        queue.put({'model': model, 'time': time.time()-t1})

    t0 = tracing.print_elapsed_time("run (%s models)" % len(models), t0)

    pipeline.close_pipeline()


def profile_path():
    path = config.output_file_path('%s.prof' % multiprocessing.current_process().name)
    return path


"""
    multiprocessing entry points
"""


def mp_run_simulation(queue, injectables, step_info, resume_after, **kwargs):

    skim_buffer = kwargs
    # handle_standard_args()

    setup_injectables_and_logging(injectables)

    mem.init_trace(setting('mem_tick'))

    if step_info['num_processes'] > 1:
        pipeline_prefix = multiprocessing.current_process().name
        logger.debug("injecting pipeline_file_prefix '%s'", pipeline_prefix)
        inject.add_injectable("pipeline_file_prefix", pipeline_prefix)

    if setting('profile', False):
        cProfile.runctx('run_simulation(queue, step_info, resume_after, skim_buffer)',
                        globals(), locals(), filename=profile_path())
    else:
        run_simulation(queue, step_info, resume_after, skim_buffer)

    chunk.log_write_hwm()
    mem.log_hwm()


def mp_apportion_pipeline(injectables, sub_job_proc_names, slice_info):
    setup_injectables_and_logging(injectables)

    if setting('profile', False):
        cProfile.runctx('apportion_pipeline(sub_job_proc_names, slice_info)',
                        globals(), locals(), filename=profile_path())
    else:
        apportion_pipeline(sub_job_proc_names, slice_info)


def mp_setup_skims(injectables, **kwargs):
    skim_buffer = kwargs
    setup_injectables_and_logging(injectables)

    if setting('profile', False):
        cProfile.runctx('load_skim_data(skim_buffer)',
                        globals(), locals(), filename=profile_path())
    else:
        load_skim_data(skim_buffer)


def mp_coalesce_pipelines(injectables, sub_job_proc_names, slice_info):
    setup_injectables_and_logging(injectables)

    if setting('profile', False):
        cProfile.runctx('coalesce_pipelines(sub_job_proc_names, slice_info)',
                        globals(), locals(), filename=profile_path())
    else:
        coalesce_pipelines(sub_job_proc_names, slice_info)


"""
    main (parent) process methods
"""


def run_sub_simulations(injectables, shared_skim_buffer, step_info, process_names,
                        resume_after, previously_completed):

    def log_queued_messages():
        for i, process, queue in zip(list(range(num_simulations)), procs, queues):
            while not queue.empty():
                msg = queue.get(block=False)
                logger.info("%s %s : %s", process.name, msg['model'],
                            tracing.format_elapsed_time(msg['time']))
                mem.trace_memory_info("%s.%s.completed" % (process.name, msg['model']))

    def check_proc_status():
        # we want to drop 'completed' breadcrumb when it happens, lest we terminate
        for p in procs:
            if p.exitcode is None:
                pass  # still running
            elif p.exitcode == 0:
                # completed successfully
                if p.name not in completed:
                    logger.info("process %s completed", p.name)
                    completed.add(p.name)
                    drop_breadcrumb(step_name, 'completed', list(completed))
                    mem.trace_memory_info("%s.completed" % p.name)
            else:
                # process failed
                if p.name not in failed:
                    logger.info("process %s failed with exitcode %s", p.name, p.exitcode)
                    failed.add(p.name)
                    mem.trace_memory_info("%s.failed" % p.name)

    def idle(seconds):
        log_queued_messages()
        check_proc_status()
        mem.trace_memory_info()
        for _ in range(seconds):
            time.sleep(1)
            log_queued_messages()
            check_proc_status()
            mem.trace_memory_info()

    step_name = step_info['name']

    t0 = tracing.print_elapsed_time()
    logger.info('run_sub_simulations step %s models resume_after %s', step_name, resume_after)

    if previously_completed:
        assert resume_after == '_'
        assert set(previously_completed).issubset(set(process_names))
        process_names = [name for name in process_names if name not in previously_completed]
        logger.info('run_sub_simulations step %s: skipping %s previously completed subprocedures',
                    step_name, len(previously_completed))

    # if not the first step, resume_after the last checkpoint from the previous step
    if resume_after is None and step_info['step_num'] > 0:
        resume_after = '_'

    num_simulations = len(process_names)
    procs = []
    queues = []
    stagger_starts = step_info['stagger']

    completed = set(previously_completed)
    failed = set([])  # so we can log process failure when it happens
    drop_breadcrumb(step_name, 'completed', list(completed))

    for process_name in process_names:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=mp_run_simulation, name=process_name,
                                    args=(q, injectables, step_info, resume_after),
                                    kwargs=shared_skim_buffer)
        procs.append(p)
        queues.append(q)

    # - start processes
    for i, p in zip(list(range(num_simulations)), procs):
        if stagger_starts > 0 and i > 0:
            logger.info("stagger process %s by %s seconds", p.name, stagger_starts)
            idle(seconds=stagger_starts)
        logger.info("start process %s", p.name)
        p.start()
        mem.trace_memory_info("%s.start" % p.name)

    # - idle logging queued messages and proc completion
    while multiprocessing.active_children():
        idle(seconds=1)
    idle(seconds=0)

    # no need to join explicitly since multiprocessing.active_children joins completed procs
    # for p in procs:
    #     p.join()

    for p in procs:
        assert p.exitcode is not None
        if p.exitcode:
            logger.error("Process %s failed with exitcode %s", p.name, p.exitcode)
            assert p.name in failed
        else:
            logger.info("Process %s completed with exitcode %s", p.name, p.exitcode)
            assert p.name in completed

    t0 = tracing.print_elapsed_time('run_sub_simulations step %s' % step_name, t0)

    return list(completed)


def run_sub_task(p):
    logger.info("running sub_process %s", p.name)

    mem.trace_memory_info("%s.start" % p.name)

    t0 = tracing.print_elapsed_time()
    p.start()

    while multiprocessing.active_children():
        mem.trace_memory_info()
        time.sleep(1)

    # no need to join explicitly since multiprocessing.active_children joins completed procs
    # p.join()

    t0 = tracing.print_elapsed_time('sub_process %s' % p.name, t0)
    # logger.info('%s.exitcode = %s' % (p.name, p.exitcode))

    mem.trace_memory_info("%s.completed" % p.name)

    if p.exitcode:
        logger.error("Process %s returned exitcode %s", p.name, p.exitcode)
        raise RuntimeError("Process %s returned exitcode %s" % (p.name, p.exitcode))


def drop_breadcrumb(step_name, crumb, value=True):
    breadcrumbs = inject.get_injectable('breadcrumbs', OrderedDict())
    breadcrumbs.setdefault(step_name, {'name': step_name})[crumb] = value
    inject.add_injectable('breadcrumbs', breadcrumbs)
    write_breadcrumbs(breadcrumbs)


def run_multiprocess(run_list, injectables):

    mem.init_trace(setting('mem_tick'))
    mem.trace_memory_info("run_multiprocess.start")

    if not run_list['multiprocess']:
        raise RuntimeError("run_multiprocess called but multiprocess flag is %s" %
                           run_list['multiprocess'])

    old_breadcrumbs = run_list.get('breadcrumbs', {})

    def skip_phase(phase):
        skip = old_breadcrumbs and old_breadcrumbs.get(step_name, {}).get(phase, False)
        if skip:
            logger.info("Skipping %s %s", step_name, phase)
        return skip

    def find_breadcrumb(crumb, default=None):
        return old_breadcrumbs.get(step_name, {}).get(crumb, default)

    t0 = tracing.print_elapsed_time()
    shared_skim_buffer = allocate_shared_skim_buffer()
    t0 = tracing.print_elapsed_time('allocate shared skim buffer', t0)
    mem.trace_memory_info("allocate_shared_skim_buffer.completed")

    # - mp_setup_skims
    run_sub_task(
        multiprocessing.Process(
            target=mp_setup_skims, name='mp_setup_skims', args=(injectables,),
            kwargs=shared_skim_buffer)
    )
    t0 = tracing.print_elapsed_time('setup skims', t0)

    for step_info in run_list['multiprocess_steps']:

        step_name = step_info['name']

        num_processes = step_info['num_processes']
        slice_info = step_info.get('slice', None)

        if num_processes == 1:
            sub_proc_names = [step_name]
        else:
            sub_proc_names = ["%s_%s" % (step_name, i) for i in range(num_processes)]

        # - mp_apportion_pipeline
        if not skip_phase('apportion') and num_processes > 1:
            run_sub_task(
                multiprocessing.Process(
                    target=mp_apportion_pipeline, name='%s_apportion' % step_name,
                    args=(injectables, sub_proc_names, slice_info))
            )
        drop_breadcrumb(step_name, 'apportion')

        # - run_sub_simulations
        if not skip_phase('simulate'):
            resume_after = step_info.get('resume_after', None)

            completed = find_breadcrumb('completed', default=[]) if resume_after == '_' else []

            completed = run_sub_simulations(injectables, shared_skim_buffer, step_info,
                                            sub_proc_names, resume_after, completed)

            if len(completed) != num_processes:
                raise RuntimeError("%s processes failed in step %s" %
                                   (num_processes - len(completed), step_name))
        drop_breadcrumb(step_name, 'simulate')

        # - mp_coalesce_pipelines
        if not skip_phase('coalesce') and num_processes > 1:
            run_sub_task(
                multiprocessing.Process(
                    target=mp_coalesce_pipelines, name='%s_coalesce' % step_name,
                    args=(injectables, sub_proc_names, slice_info))
            )
        drop_breadcrumb(step_name, 'coalesce')

    mem.log_hwm()


def get_breadcrumbs(run_list):

    resume_after = run_list['resume_after']
    assert resume_after is not None

    breadcrumbs = read_breadcrumbs()

    if not breadcrumbs:
        logger.error("empty breadcrumbs for resume_after '%s'", resume_after)
        raise RuntimeError("empty breadcrumbs for resume_after '%s'" % resume_after)

    if resume_after == '_':
        resume_step_name = list(breadcrumbs.keys())[-1]
    else:

        previous_steps = list(breadcrumbs.keys())

        # run_list step resume_after is in
        resume_step_name = next((step['name'] for step in run_list['multiprocess_steps']
                                 if resume_after in step['models']), None)

        if resume_step_name not in previous_steps:
            logger.error("resume_after model '%s' not in breadcrumbs", resume_after)
            raise RuntimeError("resume_after model '%s' not in breadcrumbs" % resume_after)

        # drop any previous_breadcrumbs steps after resume_step
        for step in previous_steps[previous_steps.index(resume_step_name) + 1:]:
            del breadcrumbs[step]

    multiprocess_step = next((step for step in run_list['multiprocess_steps']
                              if step['name'] == resume_step_name), [])  # type: dict

    if resume_after in multiprocess_step['models'][:-1]:

        # if resume_after is specified by name, and is not the last model in the step
        # then we need to rerun the simulations, even if they succeeded

        if breadcrumbs[resume_step_name].get('simulate', None):
            breadcrumbs[resume_step_name]['simulate'] = None

        if breadcrumbs[resume_step_name].get('coalesce', None):
            breadcrumbs[resume_step_name]['coalesce'] = None

    multiprocess_step_names = [step['name'] for step in run_list['multiprocess_steps']]
    if list(breadcrumbs.keys()) != multiprocess_step_names[:len(breadcrumbs)]:
        raise RuntimeError("last run steps don't match run list: %s" %
                           list(breadcrumbs.keys()))

    return breadcrumbs


def get_run_list():

    models = setting('models', [])
    resume_after = inject.get_injectable('resume_after', None) or setting('resume_after', None)
    multiprocess = inject.get_injectable('multiprocess', False) or setting('multiprocess', False)
    global_chunk_size = setting('chunk_size', 0)
    default_mp_processes = setting('num_processes', 0) or int(1 + multiprocessing.cpu_count() / 2.0)
    default_stagger = setting('stagger', 0)
    multiprocess_steps = setting('multiprocess_steps', [])

    if multiprocess and multiprocessing.cpu_count() == 1:
        logger.warning("Can't multiprocess because there is only 1 cpu")

    run_list = {
        'models': models,
        'resume_after': resume_after,
        'multiprocess': multiprocess,
        # 'multiprocess_steps': multiprocess_steps  # add this later if multiprocess
    }

    if not models or not isinstance(models, list):
        raise RuntimeError('No models list in settings file')
    if resume_after not in models + ['_', None]:
        raise RuntimeError("resume_after '%s' not in models list" % resume_after)
    if resume_after == models[-1]:
        raise RuntimeError("resume_after '%s' is last model in models list" % resume_after)

    if multiprocess:

        if not multiprocess_steps:
            raise RuntimeError("multiprocess setting is %s but no multiprocess_steps setting" %
                               multiprocess)

        # check step name, num_processes, chunk_size and presence of slice info
        num_steps = len(multiprocess_steps)
        step_names = set()
        for istep in range(num_steps):
            step = multiprocess_steps[istep]

            step['step_num'] = istep

            # - validate step name
            name = step.get('name', None)
            if not name:
                raise RuntimeError("missing name for step %s"
                                   " in multiprocess_steps" % istep)
            if name in step_names:
                raise RuntimeError("duplicate step name %s"
                                   " in multiprocess_steps" % name)
            step_names.add(name)

            # - validate num_processes and assign default
            num_processes = step.get('num_processes', 0)

            if not isinstance(num_processes, int) or num_processes < 0:
                raise RuntimeError("bad value (%s) for num_processes for step %s"
                                   " in multiprocess_steps" % (num_processes, name))

            if 'slice' in step:
                if num_processes == 0:
                    logger.info("Setting num_processes = %s for step %s", num_processes, name)
                    num_processes = default_mp_processes
                # if num_processes == 1:
                #     raise RuntimeError("num_processes = 1 but found slice info for step %s"
                #                        " in multiprocess_steps" % name)
                if num_processes > multiprocessing.cpu_count():
                    logger.warning("num_processes setting (%s) greater than cpu count (%s",
                                   num_processes, multiprocessing.cpu_count())
            else:
                if num_processes == 0:
                    num_processes = 1
                if num_processes > 1:
                    raise RuntimeError("num_processes > 1 but no slice info for step %s"
                                       " in multiprocess_steps" % name)

            multiprocess_steps[istep]['num_processes'] = num_processes

            # - validate chunk_size and assign default
            chunk_size = step.get('chunk_size', None)
            if chunk_size is None:
                if global_chunk_size > 0 and num_processes > 1:
                    chunk_size = int(round(global_chunk_size / num_processes))
                    chunk_size = max(chunk_size, 1)
                else:
                    chunk_size = global_chunk_size

            multiprocess_steps[istep]['chunk_size'] = chunk_size

            # - validate stagger and assign default
            multiprocess_steps[istep]['stagger'] = max(int(step.get('stagger', default_stagger)), 0)

        # - determine index in models list of step starts
        start_tag = 'begin'
        starts = [0] * len(multiprocess_steps)
        for istep in range(num_steps):
            step = multiprocess_steps[istep]

            name = step['name']

            slice = step.get('slice', None)
            if slice:
                if 'tables' not in slice:
                    raise RuntimeError("missing tables list for step %s"
                                       " in multiprocess_steps" % istep)

            start = step.get(start_tag, None)
            if not name:
                raise RuntimeError("missing %s tag for step '%s' (%s)"
                                   " in multiprocess_steps" %
                                   (start_tag, name, istep))
            if start not in models:
                raise RuntimeError("%s tag '%s' for step '%s' (%s) not in models list" %
                                   (start_tag, start, name, istep))

            starts[istep] = models.index(start)

            if istep == 0 and starts[istep] != 0:
                raise RuntimeError("%s tag '%s' for first step '%s' (%s)"
                                   " is not first model in models list" %
                                   (start_tag, start, name, istep))

            if istep > 0 and starts[istep] <= starts[istep - 1]:
                raise RuntimeError("%s tag '%s' for step '%s' (%s)"
                                   " falls before that of prior step in models list" %
                                   (start_tag, start, name, istep))

        # - build step model lists
        starts.append(len(models))  # so last step gets remaining models in list
        for istep in range(num_steps):
            step_models = models[starts[istep]: starts[istep + 1]]

            if step_models[-1][0] == '_':
                raise RuntimeError("Final model '%s' in step %s models list not checkpointed" %
                                   (step_models[-1], name))

            multiprocess_steps[istep]['models'] = step_models

        run_list['multiprocess_steps'] = multiprocess_steps

        # - add resume_breadcrumbs
        if resume_after:
            breadcrumbs = get_breadcrumbs(run_list)
            if breadcrumbs:
                run_list['breadcrumbs'] = breadcrumbs
                # - add resume_after to resume_step
                # FIXME - are we assuming it is in last step?
                istep = len(breadcrumbs) - 1
                multiprocess_steps[istep]['resume_after'] = resume_after

    # - write run list to output dir
    # use log_file_path so we use (optional) log subdir and prefix process name
    with config.open_log_file('run_list.txt', 'w') as f:
        print_run_list(run_list, f)

    return run_list


def print_run_list(run_list, output_file=None):

    if output_file is None:
        output_file = sys.stdout

    s = 'print_run_list'
    print(s, file=output_file)

    print("resume_after:", run_list['resume_after'], file=output_file)
    print("multiprocess:", run_list['multiprocess'], file=output_file)

    print("models", file=output_file)
    for m in run_list['models']:
        print("  - ", m, file=output_file)

    if run_list['multiprocess']:
        print("\nmultiprocess_steps:", file=output_file)
        for step in run_list['multiprocess_steps']:
            print("  step:", step['name'], file=output_file)
            for k in step:
                if isinstance(step[k], list):
                    print("    ", k, file=output_file)
                    for v in step[k]:
                        print("       -", v, file=output_file)
                else:
                    print("    %s: %s" % (k, step[k]), file=output_file)

        if run_list.get('breadcrumbs'):
            print("\nbreadcrumbs:", file=output_file)
            print_breadcrumbs(run_list['breadcrumbs'], output_file)


def print_breadcrumbs(breadcrumbs, output_file=None):

    if output_file is None:
        output_file = sys.stdout

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
    return config.build_output_file_path('breadcrumbs.yaml')


def read_breadcrumbs():
    file_path = breadcrumbs_file_path()
    if not os.path.exists(file_path):
        raise IOError("Could not find saved breadcrumbs file '%s'" % file_path)
    with open(file_path, 'r') as f:
        breadcrumbs = yaml.load(f)

    breadcrumbs = OrderedDict([(step['name'], step) for step in breadcrumbs])
    return breadcrumbs


def write_breadcrumbs(breadcrumbs):
    with open(breadcrumbs_file_path(), 'w') as f:
        breadcrumbs = [step for step in list(breadcrumbs.values())]
        yaml.dump(breadcrumbs, f)


def is_sub_task():

    return inject.get_injectable('is_sub_task', False)


def if_sub_task(if_is, if_isnt):

    return if_is if is_sub_task() else if_isnt

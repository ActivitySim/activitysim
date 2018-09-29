import os
import time
import logging

from collections import OrderedDict

import numpy as np
import pandas as pd
import multiprocessing as mp

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config

from activitysim.core.config import setting
from activitysim.core.config import handle_standard_args

from activitysim import abm
from activitysim.abm.tables.skims import skims_to_load
from activitysim.abm.tables.skims import shared_buffer_for_skims
from activitysim.abm.tables.skims import load_skims


logger = logging.getLogger('activitysim')


def load_skim_data(skim_buffer):

    logger.info("load_skim_data")

    data_dir = inject.get_injectable('data_dir')
    omx_file_path = os.path.join(data_dir, setting('skims_file'))
    tags_to_load = setting('skim_time_periods')['labels']

    skim_keys, skims_shape, skim_dtype = skims_to_load(omx_file_path, tags_to_load)

    skim_data = np.frombuffer(skim_buffer, dtype=skim_dtype).reshape(skims_shape)

    load_skims(omx_file_path, skim_keys, skim_data)

    return skim_data


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
    # FIXME - pathologically knows the format used by pipeline.pipeline_table_key()
    checkpoint_tables = {table_name: table_name + '/' + checkpoint_name
                         for table_name, checkpoint_name in checkpoint_tables.iteritems()}

    # checkpoint name and series mapping table name to hdf5 key for tables in that checkpoint
    return checkpoint_name, checkpoint_tables


def build_slice_rules(slice_info, tables):

    slicer_table_names = slice_info['tables']
    slicer_table_exceptions = slice_info.get('except', [])
    primary_slicer = slicer_table_names[0]

    if primary_slicer not in tables:
        raise RuntimeError("primary slice table '%s' not in pipeline" % primary_slicer)

    logger.debug("build_slice_rules tables %s" % tables.keys())
    logger.debug("build_slice_rules primary_slicer %s" % primary_slicer)
    logger.debug("build_slice_rules slicer_table_names %s" % slicer_table_names)
    logger.debug("build_slice_rules slicer_table_exceptions %s" % slicer_table_exceptions)

    # dict mapping slicer table_name to index name
    # (also presumed to be name of ref col name in referencing table)
    slicer_ref_cols = OrderedDict()

    # build slice rules for loaded tables
    slice_rules = {}
    for table_name, df in tables.iteritems():

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
                                               for t, c in slicer_ref_cols.iteritems()
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

        print "## rule %s: %s" % (table_name, rule)

    for table_name in slice_rules:
        logger.debug("%s: %s" % (table_name, slice_rules[table_name]))

    return slice_rules


def apportion_pipeline(sub_job_names, slice_info):

    pipeline_file_name = inject.get_injectable('pipeline_file_name')

    tables = OrderedDict([(table_name, None) for table_name in slice_info['tables']])

    # get last checkpoint from first job pipeline
    pipeline_path = config.build_output_file_path(pipeline_file_name)

    logger.debug("apportion_pipeline pipeline_path: %s" % pipeline_path)

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
        for table_name, hdf5_key in hdf5_keys.iteritems():
            # new checkpoint for all tables the same
            checkpoints_df[table_name] = checkpoint_name
            # load the dataframe
            tables[table_name] = pipeline_store[hdf5_key]

            logger.debug("loaded table %s %s" % (table_name, tables[table_name].shape))

    # keep only the last row of checkpoints and patch the last checkpoint name
    checkpoints_df = checkpoints_df.tail(1).copy()
    checkpoints_df[tables.keys()] = checkpoint_name

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
        except OSError as e:
            pass

        with pd.HDFStore(pipeline_path, mode='a') as pipeline_store:

            # remember sliced_tables so we can cascade slicing to other tables
            sliced_tables = {}
            for table_name, rule in slice_rules.iteritems():

                df = tables[table_name]

                if rule['slice_by'] == 'primary':
                    # slice primary apportion table by num_sub_jobs strides
                    # this hopefully yields a more random distribution
                    # (e.g.) households are ordered by size in input store
                    primary_df = df[np.asanyarray(range(df.shape[0])) % num_sub_jobs == i]
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

                logger.debug("writing %s (%s) to %s in %s" %
                             (table_name, sliced_tables[table_name].shape, hdf5_key, pipeline_path))
                pipeline_store[hdf5_key] = sliced_tables[table_name]

            logger.debug("writing checkpoints (%s) to %s in %s" %
                         (checkpoints_df.shape, pipeline.CHECKPOINT_TABLE_NAME, pipeline_path))
            pipeline_store[pipeline.CHECKPOINT_TABLE_NAME] = checkpoints_df


def coalesce_pipelines(sub_process_names, slice_info):

    pipeline_file_name = inject.get_injectable('pipeline_file_name')

    logger.debug("coalesce_pipelines to: %s" % pipeline_file_name)

    # tables that are identical in every pipeline and so don't need to be concatenated

    tables = OrderedDict([(table_name, None) for table_name in slice_info['tables']])

    # read all tables from first process pipeline
    pipeline_path = \
        config.build_output_file_path(pipeline_file_name, use_prefix=sub_process_names[0])
    with pd.HDFStore(pipeline_path, mode='r') as pipeline_store:

        # hdf5_keys is a dict mapping table_name to pipeline hdf5_key
        checkpoint_name, hdf5_keys = pipeline_table_keys(pipeline_store)

        for table_name, hdf5_key in hdf5_keys.iteritems():
            print "loading table", table_name, hdf5_key
            tables[table_name] = pipeline_store[hdf5_key]

    # use slice rules followed by apportion_pipeline to identify singleton tables
    slice_rules = build_slice_rules(slice_info, tables)
    singleton_table_names = [t for t, rule in slice_rules.iteritems() if rule['slice_by'] is None]
    singleton_tables = {t: tables[t] for t in singleton_table_names}
    omnibus_keys = {t: k for t, k in hdf5_keys.iteritems() if t not in singleton_table_names}

    logger.debug("coalesce_pipelines to: %s" % pipeline_file_name)
    logger.debug("singleton_table_names: %s" % singleton_table_names)
    logger.debug("omnibus_keys: %s" % omnibus_keys)

    # concat omnibus tables from all sub_processes
    omnibus_tables = {table_name: [] for table_name in omnibus_keys}
    for process_name in sub_process_names:
        pipeline_path = config.build_output_file_path(pipeline_file_name, use_prefix=process_name)
        logger.info("coalesce pipeline %s" % pipeline_path)

        with pd.HDFStore(pipeline_path, mode='r') as pipeline_store:
            for table_name, hdf5_key in omnibus_keys.iteritems():
                omnibus_tables[table_name].append(pipeline_store[hdf5_key])

    pipeline.open_pipeline()

    for table_name in singleton_tables:
        df = singleton_tables[table_name]
        logger.info("adding singleton table %s %s" % (table_name, df.shape))
        pipeline.replace_table(table_name, df)
    for table_name in omnibus_tables:
        df = pd.concat(omnibus_tables[table_name], sort=False)
        logger.info("adding omnibus table %s %s" % (table_name, df.shape))
        pipeline.replace_table(table_name, df)

    pipeline.add_checkpoint(checkpoint_name)

    pipeline.close_pipeline()

    # pipeline_path = config.build_output_file_path(pipeline_file_name)
    # with pd.HDFStore(pipeline_path, mode='r') as pipeline_store:
    #     checkpoint_name, checkpoint_tables = pipeline_table_keys(pipeline_store)
    #     print "checkpoint_tables\n", checkpoint_tables


def run_simulation(models, resume_after=None):

    pipeline.run(models=models, resume_after=resume_after)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()


def allocate_shared_data():
    logger.info("allocate_shared_data")

    data_dir = inject.get_injectable('data_dir')
    omx_file_path = os.path.join(data_dir, setting('skims_file'))
    tags_to_load = setting('skim_time_periods')['labels']

    # select the skims to load
    skim_keys, skims_shape, skim_dtype = skims_to_load(omx_file_path, tags_to_load)

    skim_buffer = shared_buffer_for_skims(skims_shape, skim_dtype)

    return skim_buffer


def run_mp_simulation(skim_buffer, models, resume_after, num_processes, pipeline_prefix=False):

    handle_standard_args()

    # do this before config_logger so log file is named appropriately
    process_name = mp.current_process().name

    logger.info("run_mp_simulation %s num_processes %s" % (process_name, num_processes))

    inject.add_injectable("log_file_prefix", process_name)
    if pipeline_prefix:
        pipeline_prefix = process_name if pipeline_prefix is True else pipeline_prefix
        logger.info("injecting pipeline_file_prefix '%s'" % pipeline_prefix)
        inject.add_injectable("pipeline_file_prefix", pipeline_prefix)

    tracing.config_logger()

    inject.add_injectable('skim_buffer', skim_buffer)

    if num_processes > 1:
        chunk_size = inject.get_injectable('chunk_size')

        if chunk_size:
            new_chunk_size = int(round(chunk_size / float(num_processes)))
            new_chunk_size = max(new_chunk_size, 1)
            logger.info("run_mp_simulation adjusting chunk_size from %s to %s" %
                        (chunk_size, new_chunk_size))
            inject.add_injectable("chunk_size", new_chunk_size)

    run_simulation(models, resume_after)

    # try:
    #     run_simulation(models, resume_after)
    # except Exception as e:
    #     print(e)
    #     logger.error("Error running simulation: %s" % (e,))
    #     raise e


def mp_apportion_pipeline(sub_job_proc_names, slice_info):
    process_name = mp.current_process().name
    inject.add_injectable("log_file_prefix", process_name)
    tracing.config_logger()

    apportion_pipeline(sub_job_proc_names, slice_info)


def mp_setup_skims(skim_buffer):
    process_name = mp.current_process().name
    inject.add_injectable("log_file_prefix", process_name)
    tracing.config_logger()

    skim_data = load_skim_data(skim_buffer)


def mp_coalesce_pipelines(sub_job_proc_names, slice_info):
    process_name = mp.current_process().name
    inject.add_injectable("log_file_prefix", process_name)
    tracing.config_logger()

    coalesce_pipelines(sub_job_proc_names, slice_info)


def mp_debug(injectables):

    for k,v in injectables.iteritems():
        inject.add_injectable(k, v)

    process_name = mp.current_process().name
    inject.add_injectable("log_file_prefix", process_name)
    tracing.config_logger()

    print "configs_dir", inject.get_injectable('configs_dir')
    print "households_sample_size", setting('households_sample_size')

def run_sub_process(p):
    logger.info("running sub_process %s" % p.name)
    p.start()
    p.join()
    # logger.info('%s.exitcode = %s' % (p.name, p.exitcode))

    if p.exitcode:
        logger.error("Process %s returned exitcode %s" % (p.name, p.exitcode))
        raise RuntimeError("Process %s returned exitcode %s" % (p.name, p.exitcode))


def run_sub_procs(procs):
    for p in procs:
        logger.info("start process %s" % p.name)
        p.start()

    while mp.active_children():
        logger.info("%s active processes" % len(mp.active_children()))
        time.sleep(15)

    for p in procs:
        p.join()

    error_procs = [p for p in procs if p.exitcode]

    return error_procs


def run_multiprocess(run_list):

    #fixme
    # logger.info('running mp_debug')
    # run_sub_process(
    #     mp.Process(target=mp_debug, name='mp_debug',
    #                args=({},))
    # )
    # bug

    logger.info('setup shared skim data')
    shared_skim_data = allocate_shared_data()
    run_sub_process(
        mp.Process(target=mp_setup_skims, name='mp_setup_skims', args=(shared_skim_data,))
    )

    resume_after = None

    for step_info in run_list['multiprocess_steps']:

        label = step_info['label']
        step_models = step_info['models']
        slice_info = step_info.get('slice', None)

        if not slice_info:

            num_processes = step_info['num_processes']
            assert num_processes == 1

            logger.info('running step %s single process with %s models' % (label, len(step_models)))

            # unsliced steps run single-threaded
            sub_proc_name = label

            run_sub_process(
                mp.Process(target=run_mp_simulation, name=sub_proc_name,
                           args=(shared_skim_data, step_models, resume_after, num_processes))
            )

        else:

            num_processes = step_info['num_processes']

            logger.info('running step %s multiprocess with %s processes and %s models' %
                        (label, num_processes, len(step_models)))

            sub_proc_names = ["%s_sub-%s" % (label, i) for i in range(num_processes)]

            logger.info('apportioning households to sub_processes')
            run_sub_process(
                mp.Process(target=mp_apportion_pipeline, name='%s_apportion' % label,
                           args=(sub_proc_names, slice_info))
            )

            logger.info('starting sub_processes')
            error_procs = run_sub_procs([
                mp.Process(target=run_mp_simulation, name=process_name,
                           args=(shared_skim_data, step_models, resume_after, num_processes),
                           kwargs={'pipeline_prefix': True})
                for process_name in sub_proc_names
            ])

            if error_procs:
                for p in error_procs:
                    logger.error("Process %s returned exitcode %s" % (p.name, p.exitcode))
                raise RuntimeError("%s processes failed in %s" % (len(error_procs), label))

            logger.info('coalescing sub_process pipelines')
            run_sub_process(
                mp.Process(target=mp_coalesce_pipelines, name='%s_coalesce' % label,
                           args=(sub_proc_names, slice_info))
            )

        resume_after = '_'


def get_run_list():

    models = setting('models', [])
    resume_after = inject.get_injectable('resume_after', None) or setting('resume_after', None)
    multiprocess = inject.get_injectable('multiprocess', False) or setting('multiprocess', False)
    multiprocess_steps = setting('multiprocess_steps', [])

    if multiprocess and mp.cpu_count() == 1:
        logger.warn("Can't multiprocess because there is only 1 cpu")
        multiprocess = False

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

    if multiprocess:

        if resume_after:
            raise RuntimeError("resume_after not implemented for multiprocessing")

        if not multiprocess_steps:
            raise RuntimeError("multiprocess setting is %s but no multiprocess_steps setting" %
                               multiprocess)

        # check label, num_processes value and presence of slice info
        labels = set()
        for istep in range(len(multiprocess_steps)):
            step = multiprocess_steps[istep]

            # check label
            label = step.get('label', None)
            if not label:
                raise RuntimeError("missing label for step %s"
                                   " in multiprocess_steps" % istep)
            if label in labels:
                raise RuntimeError("duplicate step label %s"
                                   " in multiprocess_steps" % label)
            labels.add(label)

            # validate num_processes and assign default
            num_processes = step.get('num_processes', 0)

            if not isinstance(num_processes, int) or num_processes < 0:
                raise RuntimeError("bad value (%s) for num_processes for step %s"
                                   " in multiprocess_steps" % (num_processes, label))

            if 'slice' in step:
                if num_processes == 0:
                    logger.info("Setting num_processes = %s for step %s" %
                                (num_processes, label))
                    num_processes = mp.cpu_count()
                if num_processes == 1:
                    raise RuntimeError("num_processes = 1 but found slice info for step %s"
                                       " in multiprocess_steps" % label)
                if num_processes > mp.cpu_count():
                    logger.warn("num_processes setting (%s) greater than cpu count (%s" %
                                (num_processes, mp.cpu_count()))
            else:
                if num_processes == 0:
                    num_processes = 1
                if num_processes > 1:
                    raise RuntimeError("num_processes > 1 but no slice info for step %s"
                                       " in multiprocess_steps" % label)

            multiprocess_steps[istep]['num_processes'] = num_processes

        # determine index in models list of step starts
        START = 'begin'
        starts = [0] * len(multiprocess_steps)
        for istep in range(len(multiprocess_steps)):
            step = multiprocess_steps[istep]

            label = step['label']

            slice = step.get('slice', None)
            if slice:
                if 'tables' not in slice:
                    raise RuntimeError("missing tables list for step %s"
                                       " in multiprocess_steps" % istep)

            start = step.get(START, None)
            if not label:
                raise RuntimeError("missing %s tag for step '%s' (%s)"
                                   " in multiprocess_steps" %
                                   (START, label, istep))
            if start not in models:
                raise RuntimeError("%s tag '%s' for step '%s' (%s) not in models list" %
                                   (START, start, label, istep))

            starts[istep] = models.index(start)

            if istep == 0 and starts[istep] != 0:
                raise RuntimeError("%s tag '%s' for first is not first model in models list" %
                                   (START, start, label, istep))

            if istep > 0 and starts[istep] <= starts[istep - 1]:
                raise RuntimeError("%s tag '%s' for step '%s' (%s)"
                                   " falls before that of prior step in models list" %
                                   (START, start, label, istep))

        # build step model lists
        starts.append(len(models))  # so last step gets remaining models in list
        for istep in range(len(multiprocess_steps)):
            multiprocess_steps[istep]['models'] = models[starts[istep]: starts[istep + 1]]

        run_list['multiprocess_steps'] = multiprocess_steps

    return run_list


def print_run_list(run_list):

    print "resume_after:", run_list['resume_after']
    print "multiprocess:", run_list['multiprocess']

    if run_list['multiprocess']:
        for step in run_list['multiprocess_steps']:
            print "step:", step['label']
            print "   num_processes:", step.get('num_processes', None)
            print "   models"
            for m in step['models']:
                print "     - ", m
    else:
        print "models"
        for m in run_list['models']:
            print "  - ", m
# 'multiprocess_steps': multiprocess_steps,


def console_logger_format(format):

    if inject.get_injectable('log_file_prefix', None):
        format = "%(processName)-10s " + format

    return format

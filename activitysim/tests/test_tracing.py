# ActivitySim
# See full license in LICENSE.txt.

import os.path
import logging

import pytest

import orca
import pandas as pd

from .. import tracing as tracing


def close_handlers():
    for logger_name in ['activitysim', 'orca']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def add_canonical_dirs():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)


def test_bad_custom_config_file(capsys):

    add_canonical_dirs()

    custom_config_file = os.path.join(os.path.dirname(__file__), 'configs', 'xlogging.yaml')
    tracing.config_logger(custom_config_file=custom_config_file)

    logger = logging.getLogger('activitysim')

    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    asim_logger_baseFilename = file_handlers[0].baseFilename

    logger = logging.getLogger(__name__)
    logger.info('test_bad_custom_config_file')
    logger.info('log_info')
    logger.warn('log_warn1')

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert "could not find conf file" in out
    assert 'log_warn1' in out
    assert 'log_info' not in out

    close_handlers()

    logger.warn('log_warn2')

    with open(asim_logger_baseFilename, 'r') as content_file:
        content = content_file.read()
    assert 'log_warn1' in content
    assert 'log_warn2' not in content


def test_config_logger(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    logger = logging.getLogger('activitysim')

    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    asim_logger_baseFilename = file_handlers[0].baseFilename

    print "handlers:", logger.handlers

    logger.info('test_config_logger')
    logger.info('log_info')
    logger.warn('log_warn1')

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert "could not find conf file" not in out
    assert 'log_warn1' in out
    assert 'log_info' not in out

    close_handlers()

    logger = logging.getLogger(__name__)
    logger.warn('log_warn2')

    with open(asim_logger_baseFilename, 'r') as content_file:
        content = content_file.read()
        print content
    assert 'log_warn1' in content
    assert 'log_warn2' not in content


def test_custom_config_logger(capsys):

    add_canonical_dirs()

    custom_config_file = os.path.join(os.path.dirname(__file__), 'configs', 'custom_logging.yaml')
    tracing.config_logger(custom_config_file)

    logger = logging.getLogger('activitysim')

    logger.warn('custom_log_warn')

    asim_logger_filename = os.path.join(os.path.dirname(__file__), 'output', 'xasim.log')

    with open(asim_logger_filename, 'r') as content_file:
        content = content_file.read()
    assert 'custom_log_warn' in content

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert 'custom_log_warn' in out


def test_basic(capsys):

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    # remove existing handlers or basicConfig is a NOP
    logging.getLogger().handlers = []

    tracing.config_logger(basic=True)

    logger = logging.getLogger()
    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 0

    logger = logging.getLogger('activitysim')

    logger.info('test_basic')
    logger.debug('log_debug')
    logger.info('log_info')
    logger.warn('log_warn')

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert 'log_warn' in out
    assert 'log_info' in out
    assert 'log_debug' not in out

    close_handlers()


def test_print_summary(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    tracing.print_summary('label', df=None, describe=False, value_counts=False)

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert 'print_summary neither value_counts nor describe' in out

    close_handlers()


def test_register_households(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    df = pd.DataFrame({'zort': ['a', 'b', 'c']}, index=[1, 2, 3])

    tracing.register_households(df, 5)

    out, err = capsys.readouterr()

    # don't consume output
    print out

    # should warn that household id not in index
    assert 'trace_hh_id 5 not in dataframe' in out

    # should warn and rename index if index name is None
    assert "households table index had no name. renamed index 'household_id'" in out

    close_handlers()


def test_register_tours(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    df = pd.DataFrame({'zort': ['a', 'b', 'c']}, index=[1, 2, 3])

    tracing.register_tours(df, 5)

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert "no person ids registered for trace_hh_id 5" in out

    close_handlers()


def test_register_persons(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    df = pd.DataFrame({'household_id': [1, 2, 3]}, index=[11, 12, 13])

    tracing.register_persons(df, 5)

    out, err = capsys.readouterr()

    # don't consume output
    print out

    # should warn that household id not in index
    assert 'trace_hh_id 5 not found' in out

    # should warn and rename index if index name is None
    assert "persons table index had no name. renamed index 'person_id'" in out

    close_handlers()


def test_write_csv(capsys):

    add_canonical_dirs()

    tracing.config_logger()

    # should complain if df not a DataFrame or Series
    tracing.write_csv(df='not a df or series', file_name='baddie')

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert "write_df_csv object 'baddie' of unexpected type" in out

    close_handlers()


def test_slice_ids():

    df = pd.DataFrame({'household_id': [1, 2, 3]}, index=[11, 12, 13])

    # slice by named column
    sliced_df = tracing.slice_ids(df, [1, 3, 6], column='household_id')
    assert len(sliced_df.index) == 2

    # slice by index
    sliced_df = tracing.slice_ids(df, [6, 12], column=None)
    assert len(sliced_df.index) == 1

    # attempt to slice by non-existent column
    with pytest.raises(RuntimeError) as excinfo:
        sliced_df = tracing.slice_ids(df, [5, 6], column='baddie')
    assert "slice_ids slicer column 'baddie' not in dataframe" in str(excinfo.value)

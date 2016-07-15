# ActivitySim
# See full license in LICENSE.txt.

import os.path
import logging

import orca

from .. import tracing as tracing


def close_handlers():
    for logger_name in ['activitysim', 'orca']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_bad_custom_config_file(capsys):

    configs_dir = os.path.join(os.path.dirname(__file__))
    orca.add_injectable("configs_dir", configs_dir)

    orca.add_injectable("output_dir", '.')

    custom_config_file = os.path.join(os.path.dirname(__file__), 'data', 'xlogging.yaml')
    tracing.config_logger(custom_config_file=custom_config_file)

    logger = logging.getLogger('activitysim')

    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    asim_logger_baseFilename = file_handlers[0].baseFilename

    logger = logging.getLogger(__name__)
    logger.info('log_info')
    logger.warn('log_warn1')

    tracing.debug(__name__, 'test-debug-tracing')
    tracing.info(__name__, 'test-info-tracing')
    tracing.warn(__name__, 'test-warn-tracing')
    tracing.error(__name__, 'test-error-tracing')

    out, err = capsys.readouterr()

    # don't consume output
    print out

    assert "could not find conf file" in out
    assert 'log_warn1' in out
    assert 'log_info' not in out

    assert 'test-debug-tracing' not in out
    assert 'test-info-tracing' not in out
    assert 'test-warn-tracing' in out
    assert 'test-error-tracing' in out

    close_handlers()

    logger.warn('log_warn2')

    with open(asim_logger_baseFilename, 'r') as content_file:
        content = content_file.read()
    assert 'log_warn1' in content
    assert 'log_warn2' not in content


def test_config_logger(capsys):

    configs_dir = os.path.join(os.path.dirname(__file__))
    orca.add_injectable("configs_dir", configs_dir)

    orca.add_injectable("output_dir", '.')

    tracing.config_logger()

    logger = logging.getLogger('activitysim')

    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    asim_logger_baseFilename = file_handlers[0].baseFilename

    print "handlers:", logger.handlers

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


def test_basic(capsys):

    configs_dir = os.path.join(os.path.dirname(__file__))
    orca.add_injectable("configs_dir", configs_dir)

    orca.add_injectable("output_dir", '.')

    # remove existing handlers or basicConfig is a NOP
    logging.getLogger().handlers = []

    tracing.config_logger(basic=True)

    logger = logging.getLogger()
    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 0

    logger = logging.getLogger('activitysim')
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

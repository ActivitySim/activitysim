# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import textwrap

import pytest
import yaml

from activitysim.core import workflow


def close_handlers():
    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


logging_config_content = {
    "simple": """
        ---
        logging:
          version: 1
          disable_existing_loggers: true
          loggers:
            activitysim:
              level: DEBUG
              handlers: [logfile, console]
              propagate: false
          handlers:
            logfile:
              class: logging.FileHandler
              filename: activitysim.log
              mode: w
              formatter: simpleFormatter
              level: NOTSET
            console:
              class: logging.StreamHandler
              stream: ext://sys.stdout
              formatter: simpleFormatter
              level: WARNING
          formatters:
            simpleFormatter:
              class: logging.Formatter
              format: '%(levelname)s - %(name)s - %(message)s'
              datefmt: '%d/%m/%Y %H:%M:%S'
        ...
    """,
    "functional": """
    ---
    logging:
      version: 1
      disable_existing_loggers: true
      loggers:
        activitysim:
          level: DEBUG
          handlers: [logfile, console]
          propagate: false
      handlers:
        logfile:
          class: logging.FileHandler
          filename:
            get_log_file_path: 'activitysim_from_func.log'
          mode: w
          formatter: simpleFormatter
          level: NOTSET
        console:
          class: logging.StreamHandler
          stream: ext://sys.stdout
          formatter: simpleFormatter
          level: WARNING
      formatters:
        simpleFormatter:
          class: logging.Formatter
          format: '%(levelname)s - %(name)s - %(message)s'
          datefmt: '%d/%m/%Y %H:%M:%S'
    ...
    """,
    "unsecure": """
    ---
    logging:
      version: 1
      disable_existing_loggers: true
      loggers:
        activitysim:
          level: DEBUG
          handlers: [logfile, console]
          propagate: false
      handlers:
        logfile:
          class: logging.FileHandler
          filename: !!python/object/apply:activitysim.core.config.log_file_path ['activitysim_unsecure.log']
          mode: w
          formatter: simpleFormatter
          level: NOTSET
        console:
          class: logging.StreamHandler
          stream: ext://sys.stdout
          formatter: simpleFormatter
          level: WARNING
      formatters:
        simpleFormatter:
          class: logging.Formatter
          format: '%(levelname)s - %(name)s - %(message)s'
          datefmt: '%d/%m/%Y %H:%M:%S'
    ...
    """,
}


@pytest.mark.parametrize("logging_yaml", logging_config_content.keys())
def test_config_logger(capsys, logging_yaml):
    print(logging_config_content[logging_yaml])

    state = workflow.State.make_temp()
    state.filesystem.get_configs_dir()[0].joinpath("logging.yaml").write_text(
        textwrap.dedent(logging_config_content[logging_yaml])
    )

    if logging_yaml == "unsecure":
        with pytest.raises(yaml.constructor.ConstructorError):
            state.logging.config_logger()
        return

    state.logging.config_logger()

    logger = logging.getLogger("activitysim")

    file_handlers = [h for h in logger.handlers if type(h) is logging.FileHandler]
    assert len(file_handlers) == 1
    asim_logger_baseFilename = file_handlers[0].baseFilename

    logger.info("test_config_logger")
    logger.info("log_info")
    logger.warning("log_warn1")

    out, err = capsys.readouterr()

    assert "could not find conf file" not in out
    assert "log_warn1" in out
    assert "log_info" not in out

    close_handlers()

    logger = logging.getLogger(__name__)
    logger.warning("log_warn2")

    with open(asim_logger_baseFilename) as content_file:
        content = content_file.read()
        print(content)
    assert "log_warn1" in content
    assert "log_warn2" not in content

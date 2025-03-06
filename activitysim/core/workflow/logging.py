from __future__ import annotations

import logging
import logging.config
import os
import sys
from collections.abc import Mapping, MutableMapping

import yaml

from activitysim.core.workflow.accessor import StateAccessor

logger = logging.getLogger(__name__)

ASIM_LOGGER = "activitysim"
CSV_FILE_TYPE = "csv"
LOGGING_CONF_FILE_NAME = "logging.yaml"


def _rewrite_config_dict(state, x):
    if isinstance(x, Mapping):
        # When a log config is a mapping of a single key that is `get_log_file_path`
        # we apply the get_log_file_path method to the value, which can add a
        # prefix (usually for multiprocessing)
        if len(x.keys()) == 1 and "get_log_file_path" in x.keys():
            return _rewrite_config_dict(
                state, state.get_log_file_path(x["get_log_file_path"])
            )
        # When a log config is a mapping of two keys that are `is_sub_task`
        # and `is_not_sub_task`, we check the `is_sub_task` value in this context,
        # and choose the appropriate value
        elif (
            len(x.keys()) == 2
            and "is_sub_task" in x.keys()
            and "is_not_sub_task" in x.keys()
        ):
            is_sub_task = state.get_injectable("is_sub_task", False)
            return _rewrite_config_dict(
                state, x["is_sub_task"] if is_sub_task else x["is_not_sub_task"]
            )
        # accept alternate spelling "if_sub_task" in addition to "is_sub_task"
        elif (
            len(x.keys()) == 2
            and "if_sub_task" in x.keys()
            and "if_not_sub_task" in x.keys()
        ):
            is_sub_task = state.get_injectable("is_sub_task", False)
            return _rewrite_config_dict(
                state, x["if_sub_task"] if is_sub_task else x["if_not_sub_task"]
            )
        else:
            return {k: _rewrite_config_dict(state, v) for (k, v) in x.items()}
    elif isinstance(x, list):
        return [_rewrite_config_dict(state, v) for v in x]
    elif isinstance(x, tuple):
        return tuple(_rewrite_config_dict(state, v) for v in x)
    else:
        return x


class Logging(StateAccessor):
    """
    This accessor provides logging tools.
    """

    def __get__(self, instance, objtype=None) -> Logging:
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def config_logger(self, basic=False):
        """
        Configure logger

        look for conf file in configs_dir, if not found use basicConfig

        Parameters
        ----------
        basic : bool or int, default False
            When set, ignore configuration file and just set logging to
            use streaming to stdout.  True implies logging level INFO,
            or set to a different integer for that level.

        """

        # look for conf file in configs_dir
        if basic:
            log_config_file = None
        else:
            log_config_file = self._obj.filesystem.get_config_file_path(
                LOGGING_CONF_FILE_NAME, mandatory=False
            )

        if log_config_file:
            try:
                with open(log_config_file) as f:
                    config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            except Exception as e:
                print(f"Unable to read logging config file {log_config_file}")
                raise e

            config_dict = _rewrite_config_dict(self._obj, config_dict)

            try:
                config_dict = config_dict["logging"]
                config_dict.setdefault("version", 1)
                logging.config.dictConfig(config_dict)
            except Exception as e:
                logging.warning(
                    f"Unable to config logging as specified in {log_config_file}"
                )
                logging.warning(
                    "ActivitySim now requires YAML files to be loaded in "
                    "safe mode, check your file for unsafe tags such as "
                    "`!!python/object/apply`"
                )
                raise e

        else:
            if basic is True:
                basic = logging.INFO
            logging.basicConfig(level=basic, stream=sys.stdout)

        logger = logging.getLogger(ASIM_LOGGER)

        if log_config_file:
            logger.info("Read logging configuration from: %s" % log_config_file)
        else:
            logger.log(basic, "Configured logging using basicConfig")

    def rotate_log_directory(self):
        output_dir = self._obj.filesystem.get_output_dir()
        log_dir = output_dir.joinpath("log")
        if not log_dir.exists():
            return

        from datetime import datetime
        from stat import ST_CTIME

        old_log_time = os.stat(log_dir)[ST_CTIME]
        rotate_name = os.path.join(
            output_dir,
            datetime.fromtimestamp(old_log_time).strftime("log--%Y-%m-%d--%H-%M-%S"),
        )
        try:
            os.rename(log_dir, rotate_name)
        except Exception as err:
            # if Windows fights us due to permissions or whatever,
            print(f"unable to rotate log file, {err!r}")
        else:
            # on successful rotate, create new empty log directory
            os.makedirs(log_dir)

import logging
import logging.config
import sys
from collections.abc import Mapping, MutableMapping

import yaml

from .accessor import WhaleAccessor

logger = logging.getLogger(__name__)

ASIM_LOGGER = "activitysim"
CSV_FILE_TYPE = "csv"
LOGGING_CONF_FILE_NAME = "logging.yaml"


def _rewrite_config_dict(whale, x):
    if isinstance(x, Mapping):
        # When a log config is a mapping of a single key that is `get_log_file_path`
        # we apply the get_log_file_path method to the value, which can add a
        # prefix (usually for multiprocessing)
        if len(x.keys()) == 1 and "get_log_file_path" in x.keys():
            return _rewrite_config_dict(
                whale, whale.get_log_file_path(x["get_log_file_path"])
            )
        # When a log config is a mapping of two keys that are `is_sub_task`
        # and `is_not_sub_task`, we check the `is_sub_task` value in this context,
        # and choose the appropriate value
        elif (
            len(x.keys()) == 2
            and "is_sub_task" in x.keys()
            and "is_not_sub_task" in x.keys()
        ):
            is_sub_task = whale.get_injectable("is_sub_task", False)
            return _rewrite_config_dict(
                whale, x["is_sub_task"] if is_sub_task else x["is_not_sub_task"]
            )
        # accept alternate spelling "if_sub_task" in addition to "is_sub_task"
        elif (
            len(x.keys()) == 2
            and "if_sub_task" in x.keys()
            and "if_not_sub_task" in x.keys()
        ):
            is_sub_task = whale.get_injectable("is_sub_task", False)
            return _rewrite_config_dict(
                whale, x["if_sub_task"] if is_sub_task else x["if_not_sub_task"]
            )
        else:
            return {k: _rewrite_config_dict(whale, v) for (k, v) in x.items()}
    elif isinstance(x, list):
        return [_rewrite_config_dict(whale, v) for v in x]
    elif isinstance(x, tuple):
        return tuple(_rewrite_config_dict(whale, v) for v in x)
    else:
        return x


class Logging(WhaleAccessor):
    def config_logger(self, basic=False):
        """
        Configure logger

        look for conf file in configs_dir, if not found use basicConfig

        Returns
        -------
        Nothing
        """

        # look for conf file in configs_dir
        if basic:
            log_config_file = None
        else:
            log_config_file = self.obj.filesystem.get_config_file_path(
                LOGGING_CONF_FILE_NAME, mandatory=False
            )

        if log_config_file:
            try:
                with open(log_config_file) as f:
                    config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            except Exception as e:
                print(f"Unable to read logging config file {log_config_file}")
                raise e

            config_dict = _rewrite_config_dict(self.obj, config_dict)

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
            logging.basicConfig(level=logging.INFO, stream=sys.stdout)

        logger = logging.getLogger(ASIM_LOGGER)

        if log_config_file:
            logger.info("Read logging configuration from: %s" % log_config_file)
        else:
            print("Configured logging using basicConfig")
            logger.info("Configured logging using basicConfig")

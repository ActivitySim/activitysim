# ActivitySim
# See full license in LICENSE.txt.
import argparse
import glob
import logging
import os
import struct
import time
import warnings

import yaml

from activitysim.core import inject

logger = logging.getLogger(__name__)

"""
    default injectables
"""


@inject.injectable(cache=True)
def locutor():
    # when multiprocessing, sometimes you only want one process to write trace files
    # mp_tasks overrides this definition to designate a single sub-process as locutor
    return True


@inject.injectable(cache=True)
def configs_dir():
    if not os.path.exists("configs"):
        raise RuntimeError("'configs' directory does not exist")
    return "configs"


@inject.injectable(cache=True)
def data_dir():
    if not os.path.exists("data"):
        raise RuntimeError("'data' directory does not exist")
    return "data"


@inject.injectable(cache=True)
def output_dir():
    if not os.path.exists("output"):
        print(
            f"'output' directory does not exist - current working directory: {os.getcwd()}"
        )
        raise RuntimeError("'output' directory does not exist")
    return "output"


@inject.injectable()
def output_file_prefix():
    return ""


@inject.injectable(cache=True)
def pipeline_file_name(settings):

    pipeline_file_name = settings.get("pipeline_file_name", "pipeline.h5")

    return pipeline_file_name


@inject.injectable()
def rng_base_seed():
    return 0


@inject.injectable(cache=True)
def settings_file_name():
    return "settings.yaml"


@inject.injectable(cache=True)
def settings(settings_file_name):
    settings_dict = read_settings_file(settings_file_name, mandatory=True)

    return settings_dict


# def testing():
#
#     assert ("pytest" in sys.modules) == ("PYTEST_CURRENT_TEST" in os.environ)
#     return "PYTEST_CURRENT_TEST" in os.environ


def get_cache_dir():
    """
    return path of cache directory in output_dir (creating it, if need be)

    cache directory is used to store
        skim memmaps created by skim+dict_factories
        tvpb tap_tap table cache

    Returns
    -------
    str path
    """
    cache_dir = setting("cache_dir", default=None)
    if cache_dir is None:
        cache_dir = setting(
            "cache_dir", os.path.join(inject.get_injectable("output_dir"), "cache")
        )

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    assert os.path.isdir(cache_dir)

    return cache_dir


def setting(key, default=None):
    return inject.get_injectable("settings").get(key, default)


def override_setting(key, value):
    new_settings = inject.get_injectable("settings")
    new_settings[key] = value
    inject.add_injectable("settings", new_settings)


def get_global_constants():
    """
    Read global constants from settings file

    Returns
    -------
    constants : dict
        dictionary of constants to add to locals for use by expressions in model spec
    """
    return read_settings_file("constants.yaml", mandatory=False)


def read_model_settings(file_name, mandatory=False):
    """

    Parameters
    ----------
    file_name : str
        yaml file name
    mandatory : bool
        throw error if file empty or not found
    Returns
    -------

    """

    model_settings = read_settings_file(file_name, mandatory=mandatory)

    return model_settings


def future_model_settings(model_name, model_settings, future_settings):
    """
    Warn users of new required model settings, and substitute default values

    Parameters
    ----------
    model_name: str
        name of model
    model_settings: dict
        model_settings from settigns file
    future_settings: dict
        default values for new required settings

    Returns
    -------
        dict
            model_settings with any missing future_settings added

    """
    model_settings = model_settings.copy()
    for key, setting in future_settings.items():
        if key not in model_settings.keys():
            warnings.warn(
                f"Setting '{key}' not found in {model_name} model settings."
                f"Replacing with default value: {setting}."
                f"This setting will be required in future versions",
                FutureWarning,
            )
            model_settings[key] = setting

    return model_settings


def get_model_constants(model_settings):
    """
    Read constants from model settings file

    Returns
    -------
    constants : dict
        dictionary of constants to add to locals for use by expressions in model spec
    """
    return model_settings.get("CONSTANTS", {})


def get_logit_model_settings(model_settings):
    """
    Read nest spec (for nested logit) from model settings file

    Returns
    -------
    nests : dict
        dictionary specifying nesting structure and nesting coefficients

    constants : dict
        dictionary of constants to add to locals for use by expressions in model spec
    """
    nests = None

    if model_settings is not None:

        # default to MNL
        logit_type = model_settings.get("LOGIT_TYPE", "MNL")

        if logit_type not in ["NL", "MNL"]:
            logger.error("Unrecognized logit type '%s'" % logit_type)
            raise RuntimeError("Unrecognized logit type '%s'" % logit_type)

        if logit_type == "NL":
            nests = model_settings.get("NESTS", None)
            if nests is None:
                logger.error("No NEST found in model spec for NL model type")
                raise RuntimeError("No NEST found in model spec for NL model type")

    return nests


def build_output_file_path(file_name, use_prefix=None):
    output_dir = inject.get_injectable("output_dir")

    if use_prefix:
        file_name = "%s-%s" % (use_prefix, file_name)

    file_path = os.path.join(output_dir, file_name)

    return file_path


def cascading_input_file_path(
    file_name, dir_list_injectable_name, mandatory=True, allow_glob=False
):

    dir_paths = inject.get_injectable(dir_list_injectable_name)
    dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths

    file_path = None
    for dir in dir_paths:
        p = os.path.join(dir, file_name)
        if os.path.isfile(p):
            file_path = p
            break

        if allow_glob and len(glob.glob(p)) > 0:
            file_path = p
            break

    if mandatory and not file_path:
        raise RuntimeError(
            "file_path %s: file '%s' not in %s"
            % (dir_list_injectable_name, file_name, dir_paths)
        )

    return file_path


def data_file_path(file_name, mandatory=True, allow_glob=False):

    return cascading_input_file_path(
        file_name, "data_dir", mandatory=mandatory, allow_glob=allow_glob
    )


def expand_input_file_list(input_files):
    """
    expand list by unglobbing globs globs
    """

    # be nice and accept a string as well as a list of strings
    if isinstance(input_files, str):
        input_files = [input_files]

    expanded_files = []
    ungroked_files = 0

    for file_name in input_files:

        file_name = data_file_path(file_name, allow_glob=True)

        if os.path.isfile(file_name):
            expanded_files.append(file_name)
            continue

        if os.path.isdir(file_name):
            logger.warning(
                "WARNING: expand_input_file_list skipping directory: "
                "(use glob instead): %s",
                file_name,
            )
            ungroked_files += 1
            continue

        # - glob
        logger.debug(f"expand_input_file_list trying {file_name} as glob")
        globbed_files = glob.glob(file_name)
        for globbed_file in globbed_files:
            if os.path.isfile(globbed_file):
                expanded_files.append(globbed_file)
            else:
                logger.warning(
                    "WARNING: expand_input_file_list skipping: " "(does not grok) %s",
                    file_name,
                )
                ungroked_files += 1

        if len(globbed_files) == 0:
            logger.warning(
                "WARNING: expand_input_file_list file/glob not found: %s", file_name
            )

    assert ungroked_files == 0, f"{ungroked_files} ungroked file names"

    return sorted(expanded_files)


def config_file_path(file_name, mandatory=True):

    return cascading_input_file_path(file_name, "configs_dir", mandatory)


def output_file_path(file_name):

    prefix = inject.get_injectable("output_file_prefix", None)
    return build_output_file_path(file_name, use_prefix=prefix)


def trace_file_path(file_name):

    output_dir = inject.get_injectable("output_dir")

    # - check for trace subfolder, create it if missing
    trace_dir = os.path.join(output_dir, "trace")
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    # construct a unique tail string from the time
    # this is a convenience for opening multiple similarly named trace files
    tail = hex(struct.unpack("<Q", struct.pack("<d", time.time()))[0])[-6:]

    file_parts = file_name.split(".")

    file_path = os.path.join(trace_dir, *file_parts[:-1]) + f"-{tail}.{file_parts[-1]}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def log_file_path(file_name, prefix=True):

    output_dir = inject.get_injectable("output_dir")

    # - check if running asv and if so, log to commit-specific subfolder
    asv_commit = os.environ.get("ASV_COMMIT", None)
    if asv_commit:
        output_dir = os.path.join(output_dir, f"log-{asv_commit}")
        os.makedirs(output_dir, exist_ok=True)

    # - check for optional log subfolder
    if os.path.exists(os.path.join(output_dir, "log")):
        output_dir = os.path.join(output_dir, "log")

    # - check for optional process name prefix
    prefix = prefix and inject.get_injectable("log_file_prefix", None)
    if prefix:
        file_name = "%s-%s" % (prefix, file_name)

    file_path = os.path.join(output_dir, file_name)

    return file_path


def open_log_file(file_name, mode, header=None, prefix=False):

    file_path = log_file_path(file_name, prefix)

    want_header = header and not os.path.exists(file_path)

    f = open(file_path, mode)

    if want_header:
        assert mode in [
            "a",
            "w",
        ], f"open_log_file: header requested but mode was {mode}"
        print(header, file=f)

    return f


def pipeline_file_path(file_name):

    prefix = inject.get_injectable("pipeline_file_prefix", None)
    return build_output_file_path(file_name, use_prefix=prefix)


class SettingsFileNotFound(Exception):
    def __init__(self, file_name, configs_dir):
        self.file_name = file_name
        self.configs_dir = configs_dir

    def __str__(self):
        return repr(f"Settings file '{self.file_name}' not found in {self.configs_dir}")


def read_settings_file(
    file_name, mandatory=True, include_stack=False, configs_dir_list=None
):
    """

    look for first occurence of yaml file named <file_name> in directories in configs_dir list,
    read settings from yaml file and return as dict.

    Settings file may contain directives that affect which file settings are returned:

    inherit_settings: boolean
        backfill settings in the current file with values from the next settings file in configs_dir list
    include_settings: string <include_file_name>
        read settings from specified include_file in place of the current file settings
        (to avoid confusion, this directive must appea ALONE in fiel, without any additional settings or directives.)

    Parameters
    ----------
    file_name
    mandatory: booelan
        if true, raise SettingsFileNotFound exception if no settings file, otherwise return empty dict
    include_stack: boolean or list
        only used for recursive calls to provide list of files included so far to detect cycles

    Returns: dict
        settings from speciified settings file/s
    -------

    """

    def backfill_settings(settings, backfill):
        new_settings = backfill.copy()
        new_settings.update(settings)
        return new_settings

    if configs_dir_list is None:
        configs_dir_list = inject.get_injectable("configs_dir")
        configs_dir_list = (
            [configs_dir_list]
            if isinstance(configs_dir_list, str)
            else configs_dir_list
        )
        assert isinstance(configs_dir_list, list)
        assert len(configs_dir_list) == len(
            set(configs_dir_list)
        ), f"repeating file names not allowed in config_dir list: {configs_dir_list}"

    if not file_name.lower().endswith(".yaml"):
        file_name = "%s.yaml" % (file_name,)

    inheriting = False
    settings = {}
    if isinstance(include_stack, list):
        source_file_paths = include_stack.copy()
    else:
        source_file_paths = []
    for dir in configs_dir_list:
        file_path = os.path.join(dir, file_name)
        if os.path.exists(file_path):
            if inheriting:
                # we must be inheriting
                logger.debug(
                    "inheriting additional settings for %s from %s"
                    % (file_name, file_path)
                )
                inheriting = True

            assert (
                file_path not in source_file_paths
            ), f"read_settings_file - recursion in reading 'file_path' after loading: {source_file_paths}"

            with open(file_path) as f:

                s = yaml.load(f, Loader=yaml.SafeLoader)
                if s is None:
                    s = {}

            settings = backfill_settings(settings, s)

            # maintain a list of files we read from to improve error message when an expected setting is not found
            source_file_paths += [file_path]

            include_file_name = s.get("include_settings", False)
            if include_file_name:
                # FIXME - prevent users from creating borgesian garden of branching paths?
                # There is a lot of opportunity for confusion if this feature were over-used
                # Maybe we insist that a file with an include directive is the 'end of the road'
                # essentially the current settings firle is an alias for the included file
                if len(s) > 1:
                    logger.error(
                        f"'include_settings' must appear alone in settings file."
                    )
                    additional_settings = list(
                        set(s.keys()).difference({"include_settings"})
                    )
                    logger.error(
                        f"Unexpected additional settings: {additional_settings}"
                    )
                    raise RuntimeError(
                        f"'include_settings' must appear alone in settings file."
                    )

                logger.debug(
                    "including settings for %s from %s" % (file_name, include_file_name)
                )

                # recursive call to read included file INSTEAD of the file  with include_settings sepcified
                s, source_file_paths = read_settings_file(
                    include_file_name, mandatory=True, include_stack=source_file_paths
                )

                # FIXME backfill with the included file
                settings = backfill_settings(settings, s)

            # we are done as soon as we read one file successfully
            # unless if inherit_settings is set to true in this file

            if not s.get("inherit_settings", False):
                break

            # if inheriting, continue and backfill settings from the next existing settings file configs_dir_list

            inherit_settings = s.get("inherit_settings")
            if isinstance(inherit_settings, str):
                inherit_file_name = inherit_settings
                assert (
                    os.path.join(dir, inherit_file_name) not in source_file_paths
                ), f"circular inheritance of {inherit_file_name}: {source_file_paths}: "
                # make a recursive call to switch inheritance chain to specified file

                logger.debug(
                    "inheriting additional settings for %s from %s"
                    % (file_name, inherit_file_name)
                )
                s, source_file_paths = read_settings_file(
                    inherit_file_name,
                    mandatory=True,
                    include_stack=source_file_paths,
                    configs_dir_list=configs_dir_list,
                )

                # backfill with the inherited file
                settings = backfill_settings(settings, s)
                break  # break the current inheritance chain (not as bad luck as breaking a chain-letter chain?...)

    if len(source_file_paths) > 0:
        settings["source_file_paths"] = source_file_paths

    if mandatory and not settings:
        raise SettingsFileNotFound(file_name, configs_dir_list)

    if include_stack:
        # if we were called recursively, return an updated list of source_file_paths
        return settings, source_file_paths

    else:
        return settings


def base_settings_file_path(file_name):
    """

    Parameters
    ----------
    file_name

    Returns
    -------
        path to base settings file or None if not found
    """

    if not file_name.lower().endswith(".yaml"):
        file_name = "%s.yaml" % (file_name,)

    configs_dir = inject.get_injectable("configs_dir")
    configs_dir = [configs_dir] if isinstance(configs_dir, str) else configs_dir

    for dir in configs_dir:
        file_path = os.path.join(dir, file_name)
        if os.path.exists(file_path):
            return file_path

    raise RuntimeError("base_settings_file %s not found" % file_name)


def filter_warnings():
    """
    set warning filter to 'strict' if specified in settings
    """

    if setting("strict", False):  # noqa: E402
        warnings.filterwarnings("error", category=Warning)
        warnings.filterwarnings(
            "default", category=PendingDeprecationWarning, module="future"
        )
        warnings.filterwarnings("default", category=FutureWarning, module="pandas")
        warnings.filterwarnings("default", category=RuntimeWarning, module="numpy")

    # pandas pytables.py __getitem__ (e.g. df = store['any_string'])
    # indirectly raises tables DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="tables", message="tostring"
    )

    #   File "tables/hdf5extension.pyx", line 1450, in tables.hdf5extension.Array._open_array
    # DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`.
    # Deprecated in NumPy 1.20;
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="tables",
        message="`np.object` is a deprecated alias",
    )

    # beginning pandas version 1.3, various places emit a PerformanceWarning that is
    # caught in the "strict" filter above, but which are currently unavoidable for complex models.
    # These warning are left as warnings as an invitation for future enhancement.
    from pandas.errors import PerformanceWarning

    warnings.filterwarnings("default", category=PerformanceWarning)


def handle_standard_args(parser=None):

    from activitysim.cli import run

    warnings.warn(
        "config.handle_standard_args() has been moved to the command line "
        "module and will be removed in future versions.",
        FutureWarning,
    )

    if parser is None:
        parser = argparse.ArgumentParser()

    run.add_run_args(parser)
    args = parser.parse_args()
    run.handle_standard_args(args)

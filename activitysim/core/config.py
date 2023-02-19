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

from activitysim.core import inject, util, workflow
from activitysim.core.exceptions import SettingsFileNotFoundError
from activitysim.core.workflow.util import get_formatted_or_default

logger = logging.getLogger(__name__)

"""
    default injectables
"""


@workflow.cached_object
def locutor(whale: workflow.Whale):
    # when multiprocessing, sometimes you only want one process to write trace files
    # mp_tasks overrides this definition to designate a single sub-process as locutor
    return True


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
    if file_name is not None:
        for dir in dir_paths:
            p = os.path.join(dir, file_name)
            if os.path.isfile(p):
                file_path = p
                break

            if allow_glob and len(glob.glob(p)) > 0:
                file_path = p
                break

    if mandatory and not file_path:
        raise FileNotFoundError(
            "file_path %s: file '%s' not in %s"
            % (dir_list_injectable_name, file_name, dir_paths)
        )

    return file_path


def data_file_path(file_name, mandatory=True, allow_glob=False):

    return cascading_input_file_path(
        file_name, "data_dir", mandatory=mandatory, allow_glob=allow_glob
    )


def expand_input_file_list(input_files, whale=None):
    """
    expand list by unglobbing globs globs
    """

    # be nice and accept a string as well as a list of strings
    if isinstance(input_files, str):
        input_files = [input_files]

    expanded_files = []
    ungroked_files = 0

    for file_name in input_files:

        if whale is None:
            file_name = data_file_path(file_name, allow_glob=True)
        else:
            file_name = str(
                whale.filesystem.get_data_file_path(file_name, allow_glob=True)
            )

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


# def output_file_path(file_name):
#
#     prefix = inject.get_injectable("output_file_prefix", None)
#     return build_output_file_path(file_name, use_prefix=prefix)


def log_file_path(file_name, prefix=True, whale: workflow.Whale = None):

    if whale is not None:
        output_dir = whale.filesystem.get_output_dir()
        prefix = prefix and get_formatted_or_default(
            whale.context, "log_file_prefix", None
        )
    else:
        output_dir = inject.get_injectable("output_dir")
        prefix = prefix and inject.get_injectable("log_file_prefix", None)

    # - check if running asv and if so, log to commit-specific subfolder
    asv_commit = os.environ.get("ASV_COMMIT", None)
    if asv_commit:
        output_dir = os.path.join(output_dir, f"log-{asv_commit}")
        os.makedirs(output_dir, exist_ok=True)

    # - check for optional log subfolder
    if os.path.exists(os.path.join(output_dir, "log")):
        output_dir = os.path.join(output_dir, "log")

    # - check for optional process name prefix
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


def rotate_log_directory(whale=None):

    if whale is not None:
        output_dir = whale.context.get_formatted("output_dir")
    else:
        output_dir = inject.get_injectable("output_dir")
    log_dir = os.path.join(output_dir, "log")
    if not os.path.exists(log_dir):
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
        if true, raise SettingsFileNotFoundError if no settings file, otherwise return empty dict
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

    args = util.parse_suffix_args(file_name)
    file_name = args.filename

    assert isinstance(args.ROOTS, list)
    assert (args.SUFFIX is not None and args.ROOTS) or (
        args.SUFFIX is None and not args.ROOTS
    ), ("Expected to find both 'ROOTS' and 'SUFFIX' in %s, missing one" % args.filename)

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
                        "'include_settings' must appear alone in settings file."
                    )
                    additional_settings = list(
                        set(s.keys()).difference({"include_settings"})
                    )
                    logger.error(
                        f"Unexpected additional settings: {additional_settings}"
                    )
                    raise RuntimeError(
                        "'include_settings' must appear alone in settings file."
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
        raise SettingsFileNotFoundError(file_name, configs_dir_list)

    # Adds proto_ suffix for disaggregate accessibilities
    if args.SUFFIX is not None and args.ROOTS:
        settings = util.suffix_tables_in_settings(settings, args.SUFFIX, args.ROOTS)

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


def filter_warnings(whale=None):
    """
    set warning filter to 'strict' if specified in settings
    """

    if whale is None:
        strict = False
    else:
        strict = whale.settings.treat_warnings_as_errors

    if strict:  # noqa: E402
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

    # Numba triggers a DeprecationWarning from numpy about np.MachAr
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="numba",
        message=".np.MachAr. is deprecated",
    )

    # beginning pandas version 1.3, various places emit a PerformanceWarning that is
    # caught in the "strict" filter above, but which are currently unavoidable for complex models.
    # These warning are left as warnings as an invitation for future enhancement.
    from pandas.errors import PerformanceWarning

    warnings.filterwarnings("default", category=PerformanceWarning)

    # pandas 1.5
    # beginning in pandas version 1.5, a new warning is emitted when a column is set via iloc
    # from an array of different dtype, the update will eventually be done in-place in future
    # versions. This is actually the preferred outcome for ActivitySim and no code changes are
    # needed.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=(
            ".*will attempt to set the values inplace instead of always setting a new array. "
            "To retain the old behavior, use either.*"
        ),
    )
    # beginning in pandas version 1.5, a warning is emitted when using pandas.concat on dataframes
    # that contain object-dtype columns with all-bool values.  ActivitySim plans to address dtypes
    # and move away from object-dtype columns anyhow, so this is not a critical problem.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=(
            ".*object-dtype columns with all-bool values will not be included in reductions.*"
        ),
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*will attempt to set the values inplace instead of always setting a new array.*",
    )

    # beginning in sharrow version 2.5, a CacheMissWarning is emitted when a sharrow
    # flow cannot be loaded from cache and needs to be compiled.  These are performance
    # warnings for production runs and totally expected when running test or on new
    # machines
    try:
        from sharrow import CacheMissWarning
    except ImportError:
        pass
    else:
        warnings.filterwarnings("default", category=CacheMissWarning)


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

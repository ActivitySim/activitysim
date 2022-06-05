from pypyr.errors import KeyNotInContextError
from pypyr.steps.fetchyaml import run_step as _fetch
from pypyr.steps.filewriteyaml import run_step as _write
from pypyr.steps.py import run_step as _run_step

from .progression import progress, progress_overall, progress_step


def run_step(context):
    """Update yaml file with payload.

    If you do not set encoding, will use the system default, which is utf-8
    for everything except windows.

    Args:
        context: pypyr.context.Context. Mandatory.
                 The following context keys expected:
                - updateYaml
                    - path. mandatory. path-like. Read and write file
                      here.
                    - payload. Add this to output file.
                    - encoding. string. Defaults None (platform default,
                      usually 'utf-8').

    Returns:
        None.

    Raises:
        pypyr.errors.KeyNotInContextError: fileWriteYaml or
            fileWriteYaml['path'] missing in context.
        pypyr.errors.KeyInContextHasNoValueError: fileWriteYaml or
            fileWriteYaml['path'] exists but is None.

    """
    context.assert_key_has_value("updateYaml", __name__)
    fetch_yaml_input = context.get_formatted("updateYaml")
    context["fetchYaml"] = {
        "path": fetch_yaml_input["path"],
        "key": "file_payload",
        "encoding": fetch_yaml_input.get("encoding", None),
    }
    _fetch(context)
    payload = fetch_yaml_input["payload"]
    context["file_payload"].update(payload)
    context["fileWriteYaml"] = {
        "path": fetch_yaml_input["path"],
        "payload": context["file_payload"],
        "encoding": fetch_yaml_input.get("encoding", None),
    }
    _write(context)

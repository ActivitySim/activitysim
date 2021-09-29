import os

_directory = None


def get_dir():
    global _directory
    if _directory is None:
        _directory = os.environ.get("ASIM_ASV_WORKSPACE", None)
    if _directory is None:
        _directory = os.environ.get("ASV_CONF_DIR", None)
    return _directory


def set_dir(directory):
    global _directory
    if directory:
        _directory = directory
    else:
        _directory = os.environ.get("ASIM_ASV_WORKSPACE", None)

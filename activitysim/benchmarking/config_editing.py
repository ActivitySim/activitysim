import sys
import os
import shutil
from pathlib import Path
from ruamel.yaml import YAML


def copy_to_original(filename):
    """
    If a ".original" does not exist, make a copy of a file to be the original.

    Parameters
    ----------
    filename : Path-like
        The base file to original-ize.

    Returns
    -------
    basefilename, originalfilename : Path
    """
    basefilename = Path(filename)
    if ".original" in basefilename.stem:
        basefilename = basefilename.parent / basefilename.name.replace(".original", "")
    originalfilename = basefilename.parent / (
        basefilename.stem + ".original" + basefilename.suffix
    )
    if basefilename.exists() and not originalfilename.exists():
        shutil.copyfile(basefilename, originalfilename)
    return basefilename, originalfilename


def modify_yaml(filename, changes=None, **kwargs):
    """
    Modify the settings in a yaml file.

    The file to be changed is first memorialized with `copy_to_original`
    before changes are applied.  Repeated changes are applied repeatedly to
    the original (which is not itself changed) so they do not stack.

    Parameters
    ----------
    filename : Path-like
        The settings file to manipulate
    changes, kwargs : Mapping
        Changes to apply to the file.

    """
    yaml = YAML()
    doc = Path(filename)
    doc, doc2 = copy_to_original(doc)
    settings = yaml.load(doc2)
    if changes is not None:
        kwargs.update(changes)
    settings.update(kwargs)
    yaml.dump(settings, doc)
    yaml.dump(kwargs, sys.stdout)

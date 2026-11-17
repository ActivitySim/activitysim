"""Shared extension loading for the CLI, workflow states, and workers."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def resolve_extension(extension: str | os.PathLike, working_dir=None) -> str:
    """Freeze the search directory while retaining the Python module name.

    The final path component is a module name (possibly dotted), not a Python
    filename. Like Python imports, a module may also be found elsewhere on
    sys.path. Store an absolute location so workers need not share the caller's
    current directory. abspath normalizes separators and trailing slashes without
    resolving symlinks, which could change the name of a package being imported.
    """
    return os.path.abspath(Path(working_dir or Path.cwd()) / extension)


def import_extension(extension: str | os.PathLike) -> ModuleType:
    """Import a normalized extension location, restoring sys.path on failure too."""
    location = Path(extension)
    original_path = sys.path[:]
    sys.path.insert(0, str(location.parent))
    try:
        return importlib.import_module(location.name)
    finally:
        sys.path[:] = original_path

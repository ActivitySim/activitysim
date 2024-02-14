from __future__ import annotations

from pathlib import Path

from yaml import Dumper, SafeDumper, dump_all


def _Path(dumper: Dumper, data: Path):
    """Dump a Path as a string."""
    return dumper.represent_str(str(data))


SafeDumper.add_multi_representer(Path, _Path)


def safe_dump(data, stream=None, **kwds):
    """
    Serialize a Python object into a YAML stream.
    Produce only basic YAML tags.
    If stream is None, return the produced string instead.
    """
    return dump_all([data], stream, Dumper=SafeDumper, **kwds)

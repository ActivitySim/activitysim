# ActivitySim
# See full license in LICENSE.txt.

__doc__ = "Activity-Based Travel Modeling"

try:
    from ._generated_version import __version__, __version_tuple__
except ImportError:
    # Package is not "installed", parse git tag at runtime
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version(__package__)
    except PackageNotFoundError:
        # package is not installed
        __version__ = "999"

    __version_tuple__ = __version__.split(".")

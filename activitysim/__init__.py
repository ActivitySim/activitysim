# ActivitySim
# See full license in LICENSE.txt.

__doc__ = "Activity-Based Travel Modeling"

try:
    from ._generated_version import __version__, __version_tuple__
except ImportError:
    # Package is not installed, parse git tag at runtime
    try:
        import setuptools_scm

        __version__ = setuptools_scm.get_version("../", relative_to=__file__)
    except ImportError:
        __version__ = "999"
    __version_tuple__ = __version__.split(".")

# ActivitySim
# See full license in LICENSE.txt.


__doc__ = "Activity-Based Travel Modeling"

try:
    from ._generated_version import __version__
except ImportError:
    # Package is not installed, parse git tag at runtime
    try:
        import setuptools_scm

        __version__ = setuptools_scm.get_version("../", relative_to=__file__)
    except ImportError:
        __version__ = None

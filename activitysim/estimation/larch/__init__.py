from __future__ import annotations

from typing import Iterable

from packaging.version import Version

from .cdap import *
from .data_maker import *
from .general import *
from .location_choice import *
from .mode_choice import *
from .nonmand_tour_freq import *
from .scheduling import *
from .simple_simulate import *
from .stop_frequency import *

try:
    import larch as lx
except ImportError:
    lx = None
else:
    # if larch is installed, require larch version 6.0.0 or later
    if Version(lx.__version__) < Version("6.0.0"):
        # when installed from source without a full git history including version
        # tags, sometimes development versions of larch default to version 0.1.devX
        if not lx.__version__.startswith("0.1.dev"):
            raise ImportError(
                f"activitysim estimation mode requires larch version 6.0.0 or later. Found {lx.__version__}"
            )


def component_model(name: str | Iterable[str], *args, **kwargs):
    """
    Load a component model from the estimation data bundle (EDB).

    Parameters
    ----------
    name : str or iterable of str
        The name of the model to load. If an iterable is provided, a ModelGroup
        will be returned, which will be the collection of models named.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the model constructor.
        Usually these will include `edb_directory` (a path to the EDB directory),
        and `return_data` (a boolean indicating whether to return the data objects.

    Returns
    -------
    Model or ModelGroup
        The loaded model or ModelGroup.
    Dict or list[Dict], optional
        The data objects associated with the model. If `return_data` is True, this may
        include the coefficients, chooser data, and alternative values, and possibly
        other data objects depending on the model. If `name` is a single string, a
        single data object will be returned. If `name` is an iterable, a list of data
        objects will be returned, one for each model in the ModelGroup. Note these
        data objects are primarily for the user to review, and are not the same
        objects used by the Model or ModelGroup for estimation. If some data transformation
        or manipulation is needed, it should be done on the data embedded in the
        Model(s).
    """
    if isinstance(name, str):
        m = globals().get(f"{name}_model")
        if m:
            return m(*args, **kwargs)
        raise KeyError(f"no known {name}_model")
    else:
        models = []
        all_data = []
        for n in name:
            model, *data = component_model(n, *args, **kwargs)
            models.append(model)
            all_data.extend(data)
        if all_data:
            return lx.ModelGroup(models), all_data
        else:
            return lx.ModelGroup(models)

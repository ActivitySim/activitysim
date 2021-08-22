import larch

from .cdap import *
from .data_maker import *
from .general import *
from .location_choice import *
from .mode_choice import *
from .nonmand_tour_freq import *
from .scheduling import *
from .simple_simulate import *
from .stop_frequency import *


def component_model(name, *args, **kwargs):
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
            return ModelGroup(models), all_data
        else:
            return ModelGroup(models)

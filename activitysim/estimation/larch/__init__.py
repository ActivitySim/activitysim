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
    m = globals().get(f"{name}_model")
    if m:
        return m(*args, **kwargs)
    raise KeyError(f"no known {name}_model")

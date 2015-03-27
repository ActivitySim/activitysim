import urbansim.sim.simulation as sim
import warnings
import os
import yaml
import pandas as pd
import numpy as np


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.injectable()
def set_random_seed():
    pass


@sim.injectable()
def configs_dir():
    return '.'


@sim.injectable()
def data_dir():
    return '.'


@sim.injectable(cache=True)
def settings(configs_dir):
    with open(os.path.join(configs_dir, "configs", "settings.yaml")) as f:
        return yaml.load(f)


@sim.injectable(cache=True)
def store(data_dir, settings):
    return pd.HDFStore(os.path.join(data_dir, "data", settings["store"]),
                       mode='r')


# these are the alternatives for the workplace choice, among other things
@sim.table()
def zones():
    # I grant this is a weird idiom but it helps to name the index
    return pd.DataFrame({"TAZ": np.arange(1454)+1}).set_index("TAZ")


# this is a common merge so might as well define it once here and use it
@sim.table()
def households_merged(households, land_use, accessibility):
    return sim.merge_tables(households.name, tables=[households,
                                                     land_use,
                                                     accessibility])


# another common merge for persons
@sim.table()
def persons_merged(persons, households, land_use, accessibility):
    return sim.merge_tables(persons.name, tables=[persons,
                                                  households,
                                                  land_use,
                                                  accessibility])


@sim.table()
def mandatory_tours_merged(mandatory_tours, persons_merged):
    return sim.merge_tables(mandatory_tours.name,
                            [mandatory_tours, persons_merged])


@sim.table()
def non_mandatory_tours_merged(non_mandatory_tours, persons_merged):
    tours = non_mandatory_tours
    return sim.merge_tables(tours.name, tables=[tours,
                                                persons_merged])

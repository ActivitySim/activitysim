# ActivitySim
# See full license in LICENSE.txt.


import logging

import numpy as np
import orca
import pandas as pd


@orca.table(cache=True)
def taz_table(store):
    return store["/TAZ"]


@orca.table(cache=True)
def tap_table(store):
    return store["/TAP"]


@orca.table(cache=True)
def maz_table(store):
    return store["/MAZ"]

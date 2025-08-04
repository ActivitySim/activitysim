import pandas as pd
import numpy as np
import ctypes
import logging
import multiprocessing

TALLY_CHECKIN = (0, -1)
TALLY_CHECKOUT = (1, -1)
TALLY_PENDING_PERSONS = (2, -1)

class ParkAndRideCapacity:
    """
    Class to handle park-and-ride lot capacity calculations.

    This class is used to get and set park-and-ride lot choices for tours
    and handles sending the choices across other multiprocess steps.
    It works very similarly to shadow pricing -- see `activitysim/abm/tables/shadow_pricing.py`
    """

    def __init__(self, model_settings, network_los):
        self.model_settings = model_settings
        self.network_los = network_los

    def get_capacity(self, pnr_alts, choosers, choosers_dest_col_name, state):
        """
        Calculate the park-and-ride lot capacity based on choosers and their destinations.
        """

        pass


def create_park_and_ride_capacity_data_buffers(state, model_settings):
    """
    Sets up multiprocessing buffers for park-and-ride lot choice.

    One buffer for adding up the number of park-and-ride lot choice zone ids to calculate capacities.
    One other buffer keeping track of the choices for each tour so we can choose which ones to resimulate.
    This function is called before the multiprocesses are kicked off in activitysim/core/mp_tasks.py
    """

    # get landuse and person tables to determine the size of the buffers
    land_use = state.get_dataframe("land_use")
    persons = state.get_dataframe("persons")

    shared_data_buffer_capacity = multiprocessing.Array(ctypes.c_int64, len(land_use))

    # don't know a-priori how many park-and-ride tours there are at the start of the model run
    # giving the buffer a size equal to the number of persons should be sufficient
    # need column for tour_id and column for choice -- length of persons * 2
    shared_data_buffer_choices = multiprocessing.Array(ctypes.c_int64, len(persons)*2)

    data_buffers = {
        "pnr_capacity": shared_data_buffer_capacity,
        "pnr_choices": shared_data_buffer_choices,
    }
    return data_buffers

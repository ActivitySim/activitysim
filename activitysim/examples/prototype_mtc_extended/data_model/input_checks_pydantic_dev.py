"""
Data Model for ActivitySim Inputs

Instructions: customize these example values for your own ActivitySim implementation
"""
from typing import List, Optional
import os, sys, logging

from pydantic import BaseModel, validator
import pandera as pa
import numpy as np
import pandas as pd
import openmatrix as omx

import enums as e

from activitysim.abm.models.input_checker import TABLE_STORE

logger = logging.getLogger(__name__)


class Person(BaseModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    person_id: int
    household_id: int
    age: int
    sex: int
    ptype: int


class Household(BaseModel):
    """
    Household data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    household_id: int
    home_zone_id: int
    hhsize: int
    income: int
    auto_ownership: int
    HHT: int
    persons: list[Person]


class TravelAnalysisZoneData(BaseModel):
    """
    Land use data.
    Customize as needed for your application.

    zone_id: TAZ of the zone
    DISTRICT: District the zone relies in
    SD: Super District
    COUNTY: County of zone, see enums.County
    TOTHH: Total households
    TOTEMP: Total Employment
    RETEMPN: Retail trade employment
    FPSEMPN: Financial and processional services employment
    HEREMPN: Health, educational, and recreational service employment
    OTHEMPN: Other employment
    AGREMPN: Agricultural and natural resources employment
    MWTEMPN: Manufacturing, wholesale trade, and transporation employment

    """

    zone_id: int
    DISTRICT: int
    SD: int
    county_id: int
    area_type: int
    TOTHH: int
    TOTEMP: int
    RETEMPN: int
    FPSEMPN: int
    HEREMPN: int
    OTHEMPN: int
    AGREMPN: int
    MWTEMPN: int


class PersonValidator(BaseModel):
    """
    Helper class to validate a list of persons.
    See the example notebooks for details on how this works.
    """

    list_of_persons: List[Person]


class HouseholdValidator(BaseModel):
    """
    Helper class to validate a list of households.
    See the example notebooks for details on how this works.
    """

    list_of_households: List[Household]


class TazValidator(BaseModel):
    """
    Helper class to validate a list of zonal data.
    See the example notebooks for details on how this works.
    """

    list_of_zones: List[TravelAnalysisZoneData]

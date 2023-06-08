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

# for skim name parsing
import re
import csv

from activitysim.core import config, inject, simulate

import enums as e

from activitysim.abm.models.input_checker import TABLE_STORE

logger = logging.getLogger(__name__)


class Household(pa.DataFrameModel):
    """
    Household data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    household_id: int = pa.Field(unique=True, gt=0)
    home_zone_id: int = pa.Field(ge=0)
    hhsize: int = pa.Field(gt=0)
    income: int = pa.Field(ge=0, raise_warning=True)
    auto_ownership: int = pa.Field(ge=0, le=6)
    HHT: int = pa.Field(ge=0) # FIXME add to enums

    @pa.dataframe_check(name="Household size equals the number of persons?")
    def check_persons_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE['persons']
        hhsize = persons.groupby('household_id')['person_id'].count().reindex(households.household_id)
        return (hhsize.values == households.hhsize.values).all()


class Person(pa.DataFrameModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    person_id: int = pa.Field(unique=True, ge=0)
    household_id: int = pa.Field(nullable=False)
    age: int = pa.Field(ge=0, le=100)
    sex: int = pa.Field(isin=e.Gender)
    ptype: int = pa.Field(isin=e.PersonType)

    @pa.dataframe_check(name="All persons in households table?")
    def check_persons_in_households(cls, persons: pd.DataFrame):
        households = TABLE_STORE['households']
        return persons.household_id.isin(households.household_id)
    
    @pa.dataframe_check(name="Every household has a person?")
    def check_households_have_persons(cls, persons: pd.DataFrame):
        households = TABLE_STORE['households']
        return households.household_id.isin(persons.household_id)


class Landuse(pa.DataFrameModel):
    """
    Land use data.
    Customize as needed for your application.

    Fields checked include:
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

    zone_id: int = pa.Field(unique=True, ge=0)
    DISTRICT: int = pa.Field(ge=0)
    SD: int = pa.Field(ge=0)
    county_id: int = pa.Field(isin=e.County)
    area_type: int = pa.Field(isin=e.AreaType)
    TOTHH: int = pa.Field(ge=0)
    TOTEMP: int = pa.Field(ge=0)
    RETEMPN: int = pa.Field(ge=0)
    FPSEMPN: int = pa.Field(ge=0)
    HEREMPN: int = pa.Field(ge=0)
    OTHEMPN: int = pa.Field(ge=0)
    AGREMPN: int = pa.Field(ge=0)
    MWTEMPN: int = pa.Field(ge=0)

    @pa.dataframe_check(name="Total employment is sum of employment categories?")
    def check_persons_in_households(cls, land_use: pd.DataFrame):
        tot_emp = land_use[["RETEMPN", "FPSEMPN", "HEREMPN", "OTHEMPN", "AGREMPN", "MWTEMPN"]].sum(axis=1)
        return (tot_emp == land_use.TOTEMP).all()
    



    @pa.dataframe_check(name="Dummy to check literally anything!")
    def dummy_example(cls, land_use: pd.DataFrame):
        return True
    
    @pa.dataframe_check(name="All skims in File?", raise_warning=True)
    def check_all_skims_exist(cls, land_use: pd.DataFrame):

        # FIXME code duplicated from skim_dict_factory.py but need to copy here to not load skim data
        los_settings = config.read_settings_file("network_los.yaml")
        omx_file_paths = config.expand_input_file_list(los_settings['taz_skims']['omx'])
        omx_manifest = dict()

        # FIXME getting numpy deprication warning from below omx read
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for omx_file_path in omx_file_paths:
            with omx.open_file(omx_file_path, mode="r") as omx_file:
                for skim_name in omx_file.listMatrices():
                        omx_manifest[skim_name] = omx_file_path
        
        omx_keys = []
        for skim_name in omx_manifest.keys():
            key1, sep, key2 = skim_name.partition("__")
            omx_keys.append(key1)

        tour_mode_choice_spec = config.read_settings_file('tour_mode_choice.yaml')['SPEC']
        skim_names = extract_skim_names(config.config_file_path(tour_mode_choice_spec))

        # Adding breaking change!
        skim_names.append('break')

        missing_skims = [skim_name for skim_name in skim_names if skim_name not in omx_keys]
        if len(missing_skims) > 0:
            logger.warning(f"Missing skims {missing_skims} found in {tour_mode_choice_spec}")

        return len(missing_skims) == 0


def extract_skim_names(file_path):
    skim_names = []
    
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row_string = ','.join(row)
            matches = re.findall(r"skims\[['\"]([^'\"]+)['\"]\]", row_string)
            skim_names.extend(matches)
    
    return skim_names
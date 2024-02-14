"""
Data Model for ActivitySim Inputs

Instructions: customize these example values for your own ActivitySim implementation
"""
from __future__ import annotations

import csv
import logging
import os

# for skim name parsing
import re
import sys
from typing import List, Optional

import enums as e
import numpy as np
import openmatrix as omx
import pandas as pd
import pandera as pa
from pydantic import BaseModel, validator

from activitysim.abm.models.input_checker import TABLE_STORE, log_info
from activitysim.core import config

logger = logging.getLogger(__name__)


class Household(pa.DataFrameModel):
    """
    Household data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.

    Fields:
    household_id: unique number identifying each household
    home_zone_id: zone number where household resides, MAZ in two zone systems, TAZ in one zone
    hhsize: number of people in the household
    income: Annual income in $
    auto_ownership: Seeding for initial number of autos owned by the household
    HHT: Household type, see enums.HHT
    """

    household_id: int = pa.Field(unique=True, gt=0)
    home_zone_id: int = pa.Field(ge=0)
    hhsize: int = pa.Field(gt=0)
    income: int = pa.Field(ge=0, raise_warning=True)
    auto_ownership: int = pa.Field(ge=0, le=6)
    HHT: int = pa.Field(isin=e.HHT, raise_warning=True)

    @pa.dataframe_check(
        name="Do household sizes equal the number of persons in that household?",
        raise_warning=True,
    )
    def check_persons_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE["persons"]
        hhsize = (
            persons.groupby("household_id")["person_id"]
            .count()
            .reindex(households.household_id)
        )
        log_info("test logging info")
        return (hhsize == households.set_index("household_id").hhsize).reindex(
            households.index
        )

    @pa.dataframe_check(
        name="Are all households' home_zone_ids found in the landuse file?"
    )
    def check_home_zone_in_landuse(cls, households: pd.DataFrame):
        land_use = TABLE_STORE["land_use"]
        return households.home_zone_id.isin(land_use.zone_id)

    @pa.dataframe_check(name="Example setup of a passing error check.")
    def dummy_example(cls, households: pd.DataFrame):
        return True

    @pa.dataframe_check(name="Example of a failed warning check.", raise_warning=True)
    def dummy_warning_example(cls, households: pd.DataFrame):
        return False

    @pa.dataframe_check(
        name="Household workers equals number of workers in persons table?",
        raise_warning=True,
    )
    def check_workers_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE["persons"]
        num_workers = (
            persons[persons.pemploy.isin([1, 2])]  # count full- and part-time workers
            .groupby("household_id")
            .count()
            .pemploy.reindex(households.household_id)
            .fillna(0)
        )

        return (
            num_workers == households.set_index("household_id").num_workers
        ).reindex(households.index)


class Person(pa.DataFrameModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.

    person_id: Unique person ID
    household_id: household ID of the person
    age: Person age
    sex: Person sex (see enums.py::Gender)
    ptype: Person type (see enums.py::PersonType)
    """

    person_id: int = pa.Field(unique=True, ge=0)
    household_id: int = pa.Field(nullable=False)
    age: int = pa.Field(ge=0, le=100)
    sex: int = pa.Field(isin=e.Gender)
    ptype: int = pa.Field(isin=e.PersonType)

    @pa.dataframe_check(
        name="Do each persons' household IDs exist in the households table?"
    )
    def check_persons_in_households(cls, persons: pd.DataFrame):
        households = TABLE_STORE["households"]
        return persons.household_id.isin(households.household_id)

    @pa.dataframe_check(name="Does every household ID have a matching person ID?")
    def check_households_have_persons(cls, persons: pd.DataFrame):
        households = TABLE_STORE["households"]
        return households.household_id.isin(persons.household_id)

    @pa.dataframe_check(
        name="Are all workers' and college students' ages >=18?", raise_warning=True
    )
    def check_worker_college_student_age(cls, persons: pd.DataFrame):
        return (~persons.ptype.isin([1, 2, 3])) | (persons.age >= 18)

    @pa.dataframe_check(
        name="Are all non-workers' ages in [18,65)?", raise_warning=True
    )
    def check_nonworker_age(cls, persons: pd.DataFrame):
        return (~(persons.ptype == 4)) | ((persons.age >= 18) & (persons.age < 65))

    @pa.dataframe_check(name="Are all retirees' ages >=65?", raise_warning=True)
    def check_retiree_age(cls, persons: pd.DataFrame):
        return (~(persons.ptype == 5)) | (persons.age >= 65)

    @pa.dataframe_check(
        name="Are all driving age students' ages in [16,18)?", raise_warning=True
    )
    def check_driving_student_age(cls, persons: pd.DataFrame):
        return (~(persons.ptype == 6)) | (persons.age.isin(range(16, 18)))

    @pa.dataframe_check(
        name="Are all non-driving age students' ages in [6,17)?", raise_warning=True
    )
    def check_nondriving_student_age(cls, persons: pd.DataFrame):
        return (~(persons.ptype == 7)) | (persons.age.isin(range(6, 17)))

    @pa.dataframe_check(
        name="Are all preschool children's ages in [0,6)?", raise_warning=True
    )
    def check_preschooler_student_age(cls, persons: pd.DataFrame):
        return (~(persons.ptype == 8)) | (persons.age.isin(range(0, 6)))


class Landuse(pa.DataFrameModel):
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

    @pa.dataframe_check(
        name="Is total employment equal to the sum of all employment categories?"
    )
    def check_tot_employment(cls, land_use: pd.DataFrame):
        tot_emp = land_use[
            ["RETEMPN", "FPSEMPN", "HEREMPN", "OTHEMPN", "AGREMPN", "MWTEMPN"]
        ].sum(axis=1)
        return (tot_emp == land_use.TOTEMP).reindex(land_use.index)

    @pa.dataframe_check(
        name="Do zones' total HH equal number of HH in households table?",
        raise_warning=True,
    )
    def check_hh_per_zone(cls, land_use: pd.DataFrame):
        households = TABLE_STORE["households"]
        num_hh = (
            households.groupby("home_zone_id")
            .household_id.nunique()
            .reindex(land_use.zone_id)
            .fillna(0)
        )
        return (land_use.set_index("zone_id").TOTHH == num_hh).reindex(land_use.index)

    @pa.dataframe_check(
        name="Do zones' populations equal number of people in persons table?",
        raise_warning=True,
    )
    def check_pop_per_zone(cls, land_use: pd.DataFrame):
        persons = TABLE_STORE["persons"]
        households = TABLE_STORE["households"]
        persons_per_household = persons.groupby("household_id").size()
        hh = households[["household_id", "home_zone_id"]].merge(
            persons_per_household.rename("persons_per_household"), on="household_id"
        )
        pop = hh.groupby(households.home_zone_id)["persons_per_household"].sum()
        lu = land_use.set_index("zone_id")
        return pop.reindex(lu.index) == lu.TOTPOP


class NetworkLinks(pa.DataFrameModel):
    """
    Example Network data.
    Only including some columns here for illustrative purposes.

    ID: network link ID
    Dir: Direction
    Length: length of link in miles
    AB_LANES: number of lanes in the A-node to B-node direction
    BA_LANES: number of lanes in the B-node to A-node direction
    FENAME: street name
    """

    ID: int = pa.Field(unique=True, ge=0)
    Dir: int = pa.Field(isin=[-1, 0, 1])
    Length: float = pa.Field(ge=0)
    AB_LANES: int = pa.Field(ge=0, le=10)
    BA_LANES: int = pa.Field(ge=0, le=10)
    FENAME: str = pa.Field()

    @pa.dataframe_check(name="All skims in File?", raise_warning=True)
    def check_all_skims_exist(cls, land_use: pd.DataFrame):
        state = TABLE_STORE["state"]

        # code duplicated from skim_dict_factory.py but need to copy here to not load skim data
        los_settings = state.filesystem.read_settings_file("network_los.yaml")
        omx_file_paths = state.filesystem.expand_input_file_list(
            los_settings["taz_skims"]["omx"]
        )
        omx_manifest = dict()

        for omx_file_path in omx_file_paths:
            with omx.open_file(omx_file_path, mode="r") as omx_file:
                for skim_name in omx_file.listMatrices():
                    omx_manifest[skim_name] = omx_file_path
        omx_keys = []
        for skim_name in omx_manifest.keys():
            key1, sep, key2 = skim_name.partition("__")
            omx_keys.append(key1)

        tour_mode_choice_spec = state.filesystem.read_settings_file(
            "tour_mode_choice.yaml"
        )["SPEC"]

        def extract_skim_names(file_path):
            """
            Helper function to grab the names of the matrices from a given file_path.

            e.g. grabbing 'DRIVEALONE_TIME' from instance of skims['DRIVEALONE_TIME'] in tour_mode_choice.csv
            """
            skim_names = []

            with open(file_path) as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    row_string = ",".join(row)
                    matches = re.findall(r"skims\[['\"]([^'\"]+)['\"]\]", row_string)
                    skim_names.extend(matches)

            return skim_names

        skim_names = extract_skim_names(
            state.filesystem.get_config_file_path(tour_mode_choice_spec)
        )

        missing_skims = [
            skim_name for skim_name in skim_names if skim_name not in omx_keys
        ]
        if len(missing_skims) > 0:
            log_info(f"Missing skims {missing_skims} found in {tour_mode_choice_spec}")
        else:
            log_info(f"Found all skimms in {tour_mode_choice_spec}")
        return len(missing_skims) == 0

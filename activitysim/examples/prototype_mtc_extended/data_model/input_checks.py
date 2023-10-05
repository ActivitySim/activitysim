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

from activitysim.core import config

import enums as e

from activitysim.abm.models.input_checker import TABLE_STORE, append_to_logfile

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
    bug1: int
    bug2: int

    @pa.dataframe_check(
        name="Do household sizes equal the number of persons in that household?"
    )
    def check_persons_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE["persons"]
        hhsize = (
            persons.groupby("household_id")["person_id"]
            .count()
            .reindex(households.household_id)
        )
        return (hhsize.values == households.hhsize.values).all()

    @pa.dataframe_check(
        name="Are all households' home_zone_ids found in the landuse file?"
    )
    def check_home_zone_in_landuse(cls, households: pd.DataFrame):
        land_use = TABLE_STORE["land_use"]
        return households.home_zone_id.isin(land_use.zone_id).all()

    @pa.dataframe_check(name="Example setup of a passing error check.")
    def dummy_example(cls, households: pd.DataFrame):
        return True

    @pa.dataframe_check(name="Example of a failed warning check.", raise_warning=True)
    def dummy_warning_example(cls, households: pd.DataFrame):
        return False


class Person(pa.DataFrameModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    person_id: int = pa.Field(unique=True, ge=0)
    household_id: int = pa.Field(nullable=False)
    age: int = pa.Field(ge=5, le=100)
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
        return (tot_emp == land_use.TOTEMP).all()


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

    @pa.dataframe_check(
        name="Are all skims listed in the tour mode choice config found in the taz_skims OMX file?",
        raise_warning=True,
    )
    def check_all_skims_exist(cls, land_use: pd.DataFrame):
        state = TABLE_STORE["state"]

        # code duplicated from skim_dict_factory.py but need to copy here to not load skim data
        los_settings = state.filesystem.read_settings_file("network_los.yaml")
        omx_file_paths = state.filesystem.expand_input_file_list(
            los_settings["taz_skims"]["omx"]
        )
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

        # Adding breaking change for testing!
        skim_names.append("break")

        missing_skims = [
            skim_name for skim_name in skim_names if skim_name not in omx_keys
        ]
        if len(missing_skims) > 0:
            logger.warning(
                f"Missing skims {missing_skims} found in {tour_mode_choice_spec}"
            )

        return len(missing_skims) == 0

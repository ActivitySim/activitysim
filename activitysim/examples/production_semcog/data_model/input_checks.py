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

from activitysim.abm.models.input_checker import TABLE_STORE, log_info


class Household(pa.DataFrameModel):
    """
    Household data from PopulationSim and input to ActivitySim.

    Fields:
    household_id: unique number identifying each household
    age_of_head: age of the head of household
    auto_ownership: Seeding for initial number of autos owned by the household
    hhsize: number of people in the household
    race_id:
    children: Number of children in household
    home_zone_id: zone number where household resides, MAZ in two zone systems, TAZ in one zone
    income: Annual income in $
    adjinc: Adjusted income
    HHT: Household type, see enums.HHT
    home_zone_id: MAZ of household
    TAZ: TAZ of household
    """

    household_id: int = pa.Field(unique=True, gt=0)
    age_of_head: int = pa.Field(ge=0, coerce=True)
    auto_ownership: int = pa.Field(
        isin=(list([-9] + list(range(0, 7)))), coerce=True
    )  # cars
    hhsize: int = pa.Field(gt=0)  # persons
    race_id: int = pa.Field(gt=0, le=4, coerce=True)
    children: int = pa.Field(ge=0, raise_warning=True, coerce=True)
    type: int = pa.Field(gt=0)
    hincp: float
    adjinc: int = pa.Field(ge=0, raise_warning=True)
    hht: int = pa.Field(isin=e.HHT, raise_warning=True, coerce=True)
    home_zone_id: int = pa.Field(gt=0, nullable=False, coerce=True)
    TAZ: int = pa.Field(gt=0, nullable=False, coerce=True)

    @pa.dataframe_check(
        name="Household size equals the number of persons?", raise_warning=True
    )
    def check_persons_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE["persons"]
        hhsize = (
            persons.groupby("household_id")["person_id"]
            .count()
            .reindex(households.household_id)
        )
        mismatched_indices = hhsize != households.set_index("household_id").hhsize
        mismatched_cases = households.set_index("household_id").loc[mismatched_indices]
        if len(mismatched_cases) > 0:
            log_info(
                f"Household size does not equal the number of persons at \n{mismatched_cases}.\n"
            )
        else:
            log_info(f"Household size equals the number of persons.\n")
        return ~mismatched_indices

    @pa.dataframe_check(name="taz in landuse file?", raise_warning=True)
    def check_home_zone_in_landuse(cls, households: pd.DataFrame):
        land_use = TABLE_STORE["land_use"]
        # Perform the check as before
        result = households.TAZ.isin(land_use.TAZ).all()
        output = households[~households["TAZ"].isin(land_use["TAZ"])]["TAZ"].tolist()
        if result != True:
            log_info(f"Tazes are not in landuse file at \n{output}.\n")
        else:
            log_info(f"All tazes are in landuse file.\n")
        return result

    @pa.dataframe_check(
        name="Household children equals the number of child (age<=17) in persons?",
        raise_warning=True,
    )
    def check_children_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE["persons"]
        children = (
            persons[persons["age"] <= 17]  # Filter rows where 'age' <= 17
            .groupby("household_id")  # Group by 'householder' column
            .size()  # Count the number of rows in each group
            .reindex(households.household_id)  # Reindex based on household IDs
            .fillna(0)  # Fill NaN values with 0 if there are no matching groups
        )
        mismatched_indices = children != households.set_index("household_id").children
        mismatched_cases = households.set_index("household_id").loc[mismatched_indices]
        if len(mismatched_cases) > 0:
            log_info(
                f"Household children does not equal the number of children in persons at \n{mismatched_cases}.\n"
            )
        else:
            log_info(f"Household children equals the number of children in persons.\n")
        return ~mismatched_indices


class Person(pa.DataFrameModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.

    person_id: unique person identification number
    relate:
    age: person age
    sex: person sex
    race_id: person race
    member_id: person number in the household
    household_id: household identification number
    esr: Employment status recode (from PUMS)
    wkhp: Usual hours worked per week past 12 months (from PUMS)
    wkw: Weeks worked during past 12 months (from PUMS)
    schg: Grade Level Attending (from PUMS)
    mil: Military Service (from PUMS)
    naicsp: North American Industry Classification System recode (from PUMS)
    industry: Employment industry
    zone_id: MAZ of the household
    """

    person_id: int = pa.Field(unique=True, gt=0)
    relate: int = pa.Field(ge=0, le=17)
    age: int = pa.Field(ge=0, le=100, coerce=True)
    sex: int = pa.Field(ge=1, le=2)
    race_id: int = pa.Field(gt=0, le=4)
    member_id: int = pa.Field(gt=0)
    household_id: int = pa.Field(nullable=False)
    esr: float = pa.Field(isin=e.ESR)
    wkhp: float = pa.Field(isin=(set([-9.0] + [float(x) for x in range(0, 100)])))
    wkw: float = pa.Field(isin=(set([-9.0] + [float(x) for x in range(0, 7)])))
    schg: float = pa.Field(isin=(set([-9.0] + [float(x) for x in range(0, 17)])))
    mil: float = pa.Field(isin=(set([-9.0] + [float(x) for x in range(0, 5)])))
    naicsp: str
    industry: int = pa.Field(
        isin=(set([-9.0] + [float(x) for x in range(0, 19)])), coerce=True
    )
    zone_id: int = pa.Field(gt=0, nullable=False, coerce=True)

    @pa.dataframe_check(name="All person's households in households table?")
    def check_persons_in_households(cls, persons: pd.DataFrame):
        households = TABLE_STORE["households"]
        result = persons.household_id.isin(households.household_id)
        output = persons[~persons["household_id"].isin(households["household_id"])][
            "household_id"
        ].tolist()
        if len(output) > 0:
            log_info(f"Person's household are not in households table at \n{output}.\n")
        else:
            log_info(f"All person's households are in households table.\n")
        return result

    @pa.dataframe_check(name="Every household has a person?")
    def check_households_have_persons(cls, persons: pd.DataFrame):
        households = TABLE_STORE["households"]
        result = households.household_id.isin(persons.household_id).all()
        output = households[~households["household_id"].isin(persons["household_id"])][
            "household_id"
        ].tolist()
        if result != True:
            log_info(f"Household does not have a person at \n{output}.\n")
        else:
            log_info(f"Every household has a person.\n")
        return result


class Landuse(pa.DataFrameModel):
    """
    Land use data.

    zone_id: MAZ ID
    tot_acres: Acres of the zone
    TAZ: TAZ ID
    tot_hhs: Number of households
    hhs_pop: Non-Group Quarters population
    grppop: Group-Quarters population
    tot_pop: Total population
    K_8: Preschool through 8th grade enrollment
    G9_12: High school enrollment
    e01_nrm:
    e02_constr: contrsruction employment
    e03_manuf: manufacturing employment
    e04_whole: wholsesale employment
    e05_retail: retail employment
    e06_trans: transportation employment
    e07_utility: Utility employment
    e08_infor: information services employment
    e09_finan: financial services employment
    e10_pstsvc: postal services employment(?)
    e11_compmgt: management services employment
    e12_admsvc: administrative services employment
    e13_edusvc: educational services employment
    e14_medfac: medical employment
    e15_hospit: hospital employment
    e16_leisure: leisure employment
    e17_othsvc: other services employment
    e18_pubadm: public administration employment
    tot_emp: total employment
    """

    zone_id: int = pa.Field(gt=0, le=22818, nullable=False)
    tot_acres: float = pa.Field(gt=0)
    TAZ: int = pa.Field(gt=0, le=2811, nullable=False)
    tot_hhs: float = pa.Field(ge=0, nullable=False, coerce=True)
    hhs_pop: float = pa.Field(ge=0, coerce=True)
    grppop: float = pa.Field(ge=0, coerce=True)
    tot_pop: float = pa.Field(ge=0, coerce=True)
    K_8: float = pa.Field(ge=0, coerce=True)
    G9_12: float = pa.Field(ge=0, coerce=True)
    e01_nrm: float = pa.Field(ge=0, coerce=True)
    e02_constr: float = pa.Field(ge=0, coerce=True)
    e03_manuf: float = pa.Field(ge=0, coerce=True)
    e04_whole: float = pa.Field(ge=0, coerce=True)
    e05_retail: float = pa.Field(ge=0, coerce=True)
    e06_trans: float = pa.Field(ge=0, coerce=True)
    e07_utility: float = pa.Field(ge=0, coerce=True)
    e08_infor: float = pa.Field(ge=0, coerce=True)
    e09_finan: float = pa.Field(ge=0, coerce=True)
    e10_pstsvc: float = pa.Field(ge=0, coerce=True)
    e11_compmgt: float = pa.Field(ge=0, coerce=True)
    e12_admsvc: float = pa.Field(ge=0, coerce=True)
    e13_edusvc: float = pa.Field(ge=0, coerce=True)
    e14_medfac: float = pa.Field(ge=0, coerce=True)
    e15_hospit: float = pa.Field(ge=0, coerce=True)
    e16_leisure: float = pa.Field(ge=0, coerce=True)
    e17_othsvc: float = pa.Field(ge=0, coerce=True)
    e18_pubadm: float = pa.Field(ge=0, coerce=True)
    tot_emp: float = pa.Field(ge=0, coerce=True)

    @pa.dataframe_check(name="Total employment is sum of employment categories?")
    def check_tot_employment(cls, land_use: pd.DataFrame):
        tot_emp = land_use[
            [
                "e01_nrm",
                "e02_constr",
                "e03_manuf",
                "e04_whole",
                "e05_retail",
                "e06_trans",
                "e07_utility",
                "e08_infor",
                "e09_finan",
                "e10_pstsvc",
                "e11_compmgt",
                "e12_admsvc",
                "e13_edusvc",
                "e14_medfac",
                "e15_hospit",
                "e16_leisure",
                "e17_othsvc",
                "e18_pubadm",
            ]
        ].sum(axis=1)
        result = (tot_emp == land_use.tot_emp).all()
        output = land_use.loc[~(tot_emp == land_use.tot_emp), "zone_id"].tolist()
        if len(output) > 0:
            log_info(
                f"Total employment is not sum of 18 employment categories at zone_id \n{output}.\n."
            )
        else:
            log_info(f"Total employment is sum of 18 employment categories.\n")
        return tot_emp == land_use.tot_emp

    @pa.dataframe_check(name="Zonal households equals the number of households?")
    def check_hh_per_zone(cls, land_use: pd.DataFrame):
        land_use = land_use.sort_values(by="zone_id", ascending=True)
        households = TABLE_STORE["households"]
        households = households[households["hht"] > 0]
        hh = (
            households.groupby("home_zone_id")["household_id"]
            .nunique()
            .reset_index(name="count")  # Rename the result column to "count"
        )
        hh = hh.set_index("home_zone_id").reindex(land_use.zone_id).fillna(0)
        mismatched_indices = hh["count"] != land_use.set_index("zone_id")["tot_hhs"]
        mismatched_cases = hh.loc[mismatched_indices]
        if len(mismatched_cases) > 0:
            log_info(
                f"Zonal households does not equal the number of households at \n{mismatched_cases}.\n"
            )
        else:
            log_info(f"Zonal households equals the number of households.\n")
        return ~mismatched_indices

    @pa.dataframe_check(name="Zonal pop equals the number of persons?")
    def check_pop_per_zone(cls, land_use: pd.DataFrame):
        land_use = land_use.sort_values(by="zone_id", ascending=True)
        persons = TABLE_STORE["persons"]
        persons = (
            persons.groupby("maz_seqid")["person_id"]
            .nunique()
            .reset_index(name="count")  # Rename the result column to "count"
        )
        persons = persons.set_index("maz_seqid").reindex(land_use.zone_id).fillna(0)

        mismatched_indices = (
            persons["count"] != land_use.set_index("zone_id")["tot_pop"]
        )
        mismatched_cases = persons.loc[mismatched_indices]
        if len(mismatched_cases) > 0:
            log_info(
                f"Zonal pop does not equal the number of persons at \n{mismatched_cases}.\n"
            )
        else:
            log_info(f"Zonal pop equals the number of persons.\n")
        return persons["count"] == land_use.set_index("zone_id").tot_pop


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

    ID: int = pa.Field(unique=True, gt=0)
    Dir: int = pa.Field(isin=[-1, 0, 1])
    Length: float = pa.Field(ge=0)
    AB_LANES: int = pa.Field(ge=0, le=10)
    BA_LANES: int = pa.Field(ge=0, le=10)
    FENAME: str = pa.Field(nullable=True)

    @pa.dataframe_check(name="All skims in File?", raise_warning=True)
    def check_all_skims_exist(cls, land_use: pd.DataFrame):
        state = TABLE_STORE["state"]

        # code duplicated from skim_dict_factory.py but need to copy here to not load skim data
        los_settings = state.filesystem.read_settings_file("network_los.yaml")
        omx_file_paths = state.filesystem.expand_input_file_list(
            los_settings["taz_skims"]
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

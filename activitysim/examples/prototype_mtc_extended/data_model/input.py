"""
Data Model for ActivitySim Inputs

Instructions: customize these example values for your own ActivitySim implementation
"""
import pandera as pa
import numpy as np
import pandas as pd

import enums as e

from activitysim.abm.models.input_checker import TABLE_STORE

class Person(pa.DataFrameModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    person_id: np.int32 = pa.Field(unique=True, ge=0, coerce=True)
    household_id: np.int32 = pa.Field(nullable=False, coerce=True)
    age: np.int8 = pa.Field(ge=0, le=100, coerce=True)
    sex: np.int8 = pa.Field(isin=e.Gender, coerce=True)
    ptype: np.int8 = pa.Field(isin=e.PersonType, coerce=True)

    @pa.dataframe_check(name="all persons in household")
    def check_persons_in_households(cls, persons: pd.DataFrame):
        households = TABLE_STORE['households']
        return persons.household_id.isin(households.household_id).all()



class Household(pa.DataFrameModel):
    """
    Household data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    household_id: int = pa.Field(unique=True, gt=0)
    home_zone_id: int = pa.Field(ge=0)
    hhsize: int = pa.Field(gt=0)
    income: int = pa.Field(ge=0)
    hinccat1: int = pa.Field(ge=0)
    auto_ownership: int = pa.Field(ge=0, le=6)
    HHT: int = pa.Field(ge=0) # FIXME add to enums

    @pa.dataframe_check(name='Income category matches income')
    def income_matches_category(cls, households: pd.DataFrame):
        inc_cat = pd.cut(households.income, bins=[-np.inf, 20000, 50000, 100000, np.inf], labels=[1, 2, 3, 4], right=False).astype(int)
        return ((inc_cat == households.hinccat1) | (households.income == 0)).all()
    
    @pa.dataframe_check(name="Household size equals the number of persons")
    def check_persons_per_household(cls, households: pd.DataFrame):
        persons = TABLE_STORE['persons']
        hhsize = persons.groupby('household_id')['person_id'].count().reindex(households.household_id)
        return (hhsize.values == households.hhsize.values).all()
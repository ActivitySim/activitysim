"""
Data Model for ActivitySim Inputs

Instructions: customize these example values for your own ActivitySim implementation
"""
from typing import List, Optional

from pydantic import BaseModel, validator
import pandera as pa
import numpy as np
import pandas as pd

import enums as e
from parameters import Parameters

p = Parameters()

from activitysim.abm.models.input_checker import TABLE_STORE


# class TravelAnalysisZoneData(BaseModel):
#     """
#     TAZ or socio-economic data. Customize as needed for your application.
#     """

#     id: int
#     centroid_latitude: Optional[float]
#     centroid_longitude: Optional[float]
#     households: float
#     household_population: float
#     employment_agriculture: float
#     employment_mining: float
#     employment_utilities: float
#     employment_construction: float
#     employment_manufacturing: float
#     employment_wholesale: float
#     employment_retail: float
#     employment_transport: float
#     employment_communication: float
#     employment_finance: float
#     employment_rental: float
#     employment_professional: float
#     employment_administrative: float
#     employment_education: float
#     employment_health: float
#     employment_social: float
#     employment_accommodation: float
#     employment_public_administration: float
#     employment_other: float
#     enrollment_secondary: float
#     enrollment_primary: float
#     enrollment_tertiary: float
#     parking_cost_per_hour_usd2019: float
#     area_type: e.AreaType
#     valid_values_for_internal_travel: Optional[List[int]] = range(
#         1, p.maximum_internal_zone_number
#     )

#     @property
#     def employment_total(self) -> float:
#         """
#         This is an example of how to compute a new variable from the variables defined above.
#         In this case `employment_total` is the sum across employment categories.
#         Modify as needed and add other variables as desired.
#         """
#         return (
#             self.employment_agriculture
#             + self.employment_mining
#             + self.employment_construction
#             + self.employment_manufacturing
#             + self.employment_wholesale
#             + self.employment_retail
#             + self.employment_transport
#             + self.employment_communication
#             + self.employment_finance
#             + self.employment_rental
#             + self.employment_professional
#             + self.employment_administrative
#             + self.employment_education
#             + self.employment_health
#             + self.employment_social
#             + self.employment_accommodation
#             + self.employment_public_administration
#             + self.employment_other
#         )

#     @validator("parking_cost_per_hour_usd2019")
#     def parking_cost_is_too_high(cls, value):
#         """
#         This is an example of of a custom validation method. In this case,
#         the method returns an error if `parking_cost_per_hour_usd2019` is higher than
#         the value set in the `parameters` file.
#         """
#         if value > p.maximum_parking_cost_per_hour:
#             raise ValueError("Parking cost too high")
#         return value

    # @validator("new_validator_example")
    # def define_something(cls, value):


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
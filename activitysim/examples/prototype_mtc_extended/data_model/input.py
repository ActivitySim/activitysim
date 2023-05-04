"""
Data Model for ActivitySim Inputs

Instructions: customize these example values for your own ActivitySim implementation
"""
from typing import List, Optional

from pydantic import BaseModel, validator

import enums as e
from parameters import Parameters

p = Parameters()


class TravelAnalysisZoneData(BaseModel):
    """
    TAZ or socio-economic data. Customize as needed for your application.
    """

    id: int
    centroid_latitude: Optional[float]
    centroid_longitude: Optional[float]
    households: float
    household_population: float
    employment_agriculture: float
    employment_mining: float
    employment_utilities: float
    employment_construction: float
    employment_manufacturing: float
    employment_wholesale: float
    employment_retail: float
    employment_transport: float
    employment_communication: float
    employment_finance: float
    employment_rental: float
    employment_professional: float
    employment_administrative: float
    employment_education: float
    employment_health: float
    employment_social: float
    employment_accommodation: float
    employment_public_administration: float
    employment_other: float
    enrollment_secondary: float
    enrollment_primary: float
    enrollment_tertiary: float
    parking_cost_per_hour_usd2019: float
    area_type: e.AreaType
    valid_values_for_internal_travel: Optional[List[int]] = range(
        1, p.maximum_internal_zone_number
    )

    @property
    def employment_total(self) -> float:
        """
        This is an example of how to compute a new variable from the variables defined above.
        In this case `employment_total` is the sum across employment categories.
        Modify as needed and add other variables as desired.
        """
        return (
            self.employment_agriculture
            + self.employment_mining
            + self.employment_construction
            + self.employment_manufacturing
            + self.employment_wholesale
            + self.employment_retail
            + self.employment_transport
            + self.employment_communication
            + self.employment_finance
            + self.employment_rental
            + self.employment_professional
            + self.employment_administrative
            + self.employment_education
            + self.employment_health
            + self.employment_social
            + self.employment_accommodation
            + self.employment_public_administration
            + self.employment_other
        )

    @validator("parking_cost_per_hour_usd2019")
    def parking_cost_is_too_high(cls, value):
        """
        This is an example of of a custom validation method. In this case,
        the method returns an error if `parking_cost_per_hour_usd2019` is higher than
        the value set in the `parameters` file.
        """
        if value > p.maximum_parking_cost_per_hour:
            raise ValueError("Parking cost too high")
        return value

    # @validator("new_validator_example")
    # def define_something(cls, value):
        


class Person(BaseModel):
    """
    Person data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    id: int
    age: int
    sex: e.Gender
    person_type: e.PersonType
    occupation: e.Occupation

    @property
    def gender(self) -> e.Gender:
        """
        Sex is used as a proxy for gender in the travel model
        """
        return self.sex


class Household(BaseModel):
    """
    Household data from PopulationSim and input to ActivitySim.
    Customize as needed for your application.
    """

    id: int
    persons: List[Person]
    income_in_usd2011: float
    home_location: int

    @property
    def income_quartile(self) -> int:
        """
        Create income quartiles from the income using the
        `income_quartile_breakpoints_in_aud2019` Parameter
        """
        quartile = 1
        for break_point in p.income_quartile_breakpoints_in_usd2011:
            if self.income_in_usd2011 < break_point:
                return quartile

    @property
    def household_size(self) -> int:
        """
        Number of persons in the household from the persons array length
        """
        return len(self.persons)


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

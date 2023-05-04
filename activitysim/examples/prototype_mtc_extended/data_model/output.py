"""
Data Model for ActivitySim Outputs

Instructions: customize these example values for your own ActivitySim implementation
"""
from typing import List, Optional

from pydantic import BaseModel

import enums as e
from input import Household as InputHousehold
from input import Person as InputPerson


class Trip(BaseModel):
    """
    Relevant variables for ActivitySim `trips` output data file.
    Customize as needed for your application.
    """

    id: int
    purpose: e.Purpose
    origin: int
    destination: int
    depart_from_origin: e.ModelTime
    arrive_at_destination: e.ModelTime
    mode: e.Mode


class Tour(BaseModel):
    """
    Relevant variables for ActivitySim `tours` output data file.
    Customize as needed for your application.
    """

    id: int
    purpose: e.Purpose
    origin: int
    destination: int
    depart_from_origin: e.ModelTime
    return_to_origin: e.ModelTime
    trips: List[Trip]
    mode: e.Mode


class Person(InputPerson):
    """
    Relevant variables for ActivitySim `persons` output data file.
    Customize as needed for your application.
    """

    usual_work_location: Optional[int]
    usual_school_location: Optional[int]
    individual_tours: Optional[List[Tour]]


class JointTour(Tour):
    """
    Relevant variables for ActivitySim `joint_tours` output data file.
    Customize as needed for your application.
    """

    participants: List[Person]


class Household(InputHousehold):
    """
    Relevant variables for ActivitySim `household` output data file.
    Customize as needed for your application.
    """

    joint_tours: Optional[List[JointTour]] = None

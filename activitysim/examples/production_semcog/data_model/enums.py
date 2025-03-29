"""
Data Model Enumerated Variables

Instructions: modify these enumerated variables as needed for your ActivitySim implementation.
"""
from enum import IntEnum


class PersonType(IntEnum):
    """
    Provides integer mapping to the person type variable. A person type is
    used as a co-variate in numerous ActivitySim modules to explain behavior.
    """

    FULL_TIME_WORKER = 1
    PART_TIME_WORKER = 2
    ADULT_STUDENT = 3
    NON_WORKING_ADULT = 4
    RETIRED = 5
    SECONDARY_SCHOOL_STUDENT = 6
    PRIMARY_SCHOOL_STUDENT = 7
    PRE_SCHOOL_CHILD = 8


class HHT(IntEnum):
    """
    Provide an integer mapping for household/family type.
    """

    GROUP_QUARTERS = 0
    FAMILY_MARRIED_COUPLE = 1
    FAMILY_MALE_HOUSEHOLDER_NO_WIFE = 2
    FAMILY_FEMANLE_HOUSEHOLDER_NO_HUSBAND = 3
    NON_FAMILY_MALE_ALONE = 4
    NON_FAMILY_MALE_NOT_ALONE = 5
    NON_FAMILY_FEMALE_ALONE = 6
    NON_FAMILY_FEMALE_NOT_ALONE = 7
    UNKNOWN = -9


class ESR(IntEnum):
    """
    Employment Status Recode
    """

    NA = -9
    EMPLOYED_AT_WORK = 1
    EMPLOYED_NOT_AT_WORK = 2
    UNEMPLOYED = 3
    ARMED_FORCES_AT_WORK = 4
    ARMED_FORCES_NOT_AT_WORK = 5
    NOT_IN_LABOR_FORCE = 6

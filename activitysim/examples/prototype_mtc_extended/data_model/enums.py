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


class Gender(IntEnum):
    """
    Provides an integer mapping for gender.
    """

    MALE = 1
    FEMALE = 2
    OTHER = 3


class Occupation(IntEnum):
    """
    Provides an integer mapping for a persons occupation.
    """

    WHITE_COLLAR = 1
    SERVICES = 2
    HEALTH = 3
    RETAIL = 4
    BLUE_COLLAR = 5
    NOT_EMPLOYED = 6


class DailyActivityPattern(IntEnum):
    """
    Provides an integer mapping to the daily activity pattern variable.
    """

    MANDATORY = 1
    NON_MANDATORY = 2
    HOME = 3


class Purpose(IntEnum):
    """
    Provides an integer mapping for trip and tour purpose.
    """

    WORK = 1
    ADULT_SCHOOL = 2
    CHILD_SCHOOL = 3
    SHOPPING = 4
    ESCORT = 5
    MAINTENANCE = 6
    DISCRETIONARY = 7


class AreaType(IntEnum):
    """
    Provide an integer mapping for area type, which serves as a proxy for urban form.
    """

    REGIONAL_CORE = 0
    CBD = 1
    URBAN_BUSINESS = 2
    URBAN = 3
    SUBURBAN = 4
    RURAL = 5


class County(IntEnum):
    """
    Provide an integer mapping for county in which a landuse zone resides.
    """

    SAN_FRANSISCO = 1
    SAN_MATEO = 2
    SANTA_CLARA = 3
    ALAMEDA = 4
    CONTRA_COSTA = 5
    SOLANO = 6
    NAPA = 7
    SONOMA = 8
    MARIN = 9


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


class Mode(IntEnum):
    """
    Provides an integer mapping for travel mode.
    """

    DRIVE_ALONE_FREE = 1
    DRIVE_ALONE_PAY = 2
    SHARED_RIDE_2_FREE = 3
    SHARED_RIDE_2_PAY = 4
    SHARED_RIDE_3_FREE = 5
    SHARED_RIDE_3_PAY = 6
    WALK = 7
    BICYCLE = 8
    WALK_TO_TRANSIT_ALL = 9
    WALK_TO_TRANSIT_PREMIUM_ONLY = 10
    PARK_AND_RIDE_TRANSIT_ALL = 11
    PARK_AND_RIDE_TRANSIT_PREMIUM_ONLY = 12
    KISS_AND_RIDE_TRANSIT_ALL = 13
    KISS_AND_RIDE_TRANSIT_PREMIUM_ONLY = 14
    SCHOOL_BUS = 15


class ModelTime(IntEnum):
    """
    Provides an integer mapping from military time to model time interval index.
    The name represents the starting point of the interval. So the interval from
    3:00 am to 3:30 am is represented by index 1, which is named `ZERO_THREE`.
    """

    ZERO_THREE = 1
    ZERO_THREE_THIRTY = 2
    ZERO_FOUR = 3
    ZERO_FOUR_THIRTY = 4
    ZERO_FIVE = 5
    ZERO_FIVE_THIRTY = 6
    ZERO_SIX = 7
    ZERO_SIX_THIRTY = 8
    ZERO_SEVEN = 9
    ZERO_SEVEN_THIRTY = 10
    ZERO_EIGHT = 11
    ZERO_EIGHT_THIRTY = 12
    ZERO_NINE = 13
    ZERO_NINE_THIRTY = 14
    TEN = 15
    TEN_THIRTY = 16
    ELEVEN = 17
    ELEVEN_THIRTY = 18
    TWELVE = 19
    TWELVE_THIRTY = 20
    THIRTEEN = 21
    THIRTEEN_THIRTY = 22
    FOURTEEN = 23
    FOURTEEN_THIRTY = 24
    FIFTEEN = 25
    FIFTEEN_THIRTY = 26
    SIXTEEN = 27
    SIXTEEN_THIRTY = 28
    SEVENTEEN = 29
    SEVENTEEN_THIRTY = 30
    EIGHTEEN = 31
    EIGHTEEN_THIRTY = 32
    NINETEEN = 33
    NINETEEN_THIRTY = 34
    TWENTY = 35
    TWENTY_THIRTY = 36
    TWENTY_ONE = 37
    TWENTY_ONE_THIRTY = 38
    TWENTY_TWO = 39
    TWENTY_TWO_THIRTY = 40
    TWENTY_THREE = 41
    TWENTY_THREE_THIRTY = 42

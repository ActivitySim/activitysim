# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

HHT_NONE = 0
HHT_FAMILY_MARRIED = 1
HHT_FAMILY_MALE = 2
HHT_FAMILY_FEMALE = 3
HHT_NONFAMILY_MALE_ALONE = 4
HHT_NONFAMILY_MALE_NOTALONE = 5
HHT_NONFAMILY_FEMALE_ALONE = 6
HHT_NONFAMILY_FEMALE_NOTALONE = 7

# convenience for expression files
HHT_NONFAMILY = [4, 5, 6, 7]
HHT_FAMILY = [1, 2, 3]

PSTUDENT_GRADE_OR_HIGH = 1
PSTUDENT_UNIVERSITY = 2
PSTUDENT_NOT = 3

GRADE_SCHOOL_MAX_AGE = 14
GRADE_SCHOOL_MIN_AGE = 5

SCHOOL_SEGMENT_NONE = 0
SCHOOL_SEGMENT_GRADE = 1
SCHOOL_SEGMENT_HIGH = 2
SCHOOL_SEGMENT_UNIV = 3

INCOME_SEGMENT_LOW = 1
INCOME_SEGMENT_MED = 2
INCOME_SEGMENT_HIGH = 3
INCOME_SEGMENT_VERYHIGH = 4

PEMPLOY_FULL = 1
PEMPLOY_PART = 2
PEMPLOY_NOT = 3
PEMPLOY_CHILD = 4

PTYPE_FULL = 1
PTYPE_PART = 2
PTYPE_UNIVERSITY = 3
PTYPE_NONWORK = 4
PTYPE_RETIRED = 5
PTYPE_DRIVING = 6
PTYPE_SCHOOL = 7
PTYPE_PRESCHOOL = 8


# these appear as column headers in non_mandatory_tour_frequency.csv
PTYPE_NAME = {
    PTYPE_FULL: 'PTYPE_FULL',
    PTYPE_PART: 'PTYPE_PART',
    PTYPE_UNIVERSITY: 'PTYPE_UNIVERSITY',
    PTYPE_NONWORK: 'PTYPE_NONWORK',
    PTYPE_RETIRED: 'PTYPE_RETIRED',
    PTYPE_DRIVING: 'PTYPE_DRIVING',
    PTYPE_SCHOOL: 'PTYPE_SCHOOL',
    PTYPE_PRESCHOOL: 'PTYPE_PRESCHOOL'
}

CDAP_ACTIVITY_MANDATORY = 'M'
CDAP_ACTIVITY_NONMANDATORY = 'N'
CDAP_ACTIVITY_HOME = 'H'

# for use in string expressions (e.g. so we can change from a string to an int)
CDAP_ACTIVITY_MANDATORY_Q = "'M'"
CDAP_ACTIVITY_NONMANDATORY_Q = "'N'"
CDAP_ACTIVITY_HOME_Q = "'H'"

# joint tour types
# shopping,othmaint,eatout,social,othdiscr
#
# atwork subtour tour types
# eat,business,maint
#
# mandatory tour types
# work,school
#
# nonmandatory tour types
# escort,shopping,othmaint,othdiscr,eatout,social

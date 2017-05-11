# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import orca
import pandas as pd

from activitysim.core import pipeline

from activitysim.core import tracing
from activitysim.core.util import other_than, reindex


logger = logging.getLogger(__name__)


@orca.table()
def persons(store, households_sample_size, households, trace_hh_id):

    df = store["persons"]

    if households_sample_size > 0:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    logger.info("loaded persons %s" % (df.shape,))

    # replace table function with dataframe
    orca.add_table('persons', df)

    pipeline.get_rn_generator().add_channel(df, 'persons')

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "persons", warn_if_empty=True)

    return df


# another common merge for persons
@orca.table()
def persons_merged(persons, households, land_use, accessibility):
    return orca.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])


# this is the placeholder for all the columns to update after the
# non-mandatory tour frequency model
@orca.table()
def persons_nmtf(persons):
    return pd.DataFrame(index=persons.index)


@orca.column("persons_nmtf")
def num_escort_tours(persons, non_mandatory_tours):
    nmt = non_mandatory_tours.to_frame()
    return nmt[nmt.tour_type == "escort"].groupby("person_id").size()\
        .reindex(persons.index).fillna(0)


@orca.column("persons_nmtf")
def num_non_escort_tours(persons, non_mandatory_tours):
    nmt = non_mandatory_tours.to_frame()
    return nmt[nmt.tour_type != "escort"].groupby("person_id").size()\
        .reindex(persons.index).fillna(0)


# this is the placeholder for all the columns to update after the
# mandatory tour frequency model
@orca.table()
def persons_mtf(persons):
    return pd.DataFrame(index=persons.index)


# count the number of mandatory tours for each person
@orca.column("persons_mtf")
def num_mand(persons):

    s = persons.mandatory_tour_frequency.map({
        "work1": 1,
        "work2": 2,
        "school1": 1,
        "school2": 2,
        "work_and_school": 2
    }, na_action='ignore')
    return s.fillna(0)


@orca.column("persons_mtf")
def work_and_school_and_worker(persons):

    s = (persons.mandatory_tour_frequency == "work_and_school").\
        reindex(persons.index).fillna(False)

    return s & persons.is_worker


@orca.column("persons_mtf")
def work_and_school_and_student(persons):

    s = (persons.mandatory_tour_frequency == "work_and_school").\
        reindex(persons.index).fillna(False)

    return s & persons.is_student


# this is the placeholder for all the columns to update after the
# workplace location choice model
@orca.table()
def persons_workplace(persons):
    return pd.DataFrame(index=persons.index)


# this use the distance skims to compute the raw distance to work from home
@orca.column("persons_workplace")
def distance_to_work(persons, skim_dict):
    distance_skim = skim_dict.get('DIST')
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.workplace_taz),
                     index=persons.index)


# this uses the free flow travel time in both directions
# MTC TM1 was MD and MD since term is free flow roundtrip_auto_time_to_work
@orca.column("persons_workplace")
def roundtrip_auto_time_to_work(persons, skim_dict):
    sovmd_skim = skim_dict.get(('SOV_TIME', 'MD'))
    return pd.Series(sovmd_skim.get(persons.home_taz,
                                    persons.workplace_taz) +
                     sovmd_skim.get(persons.workplace_taz,
                                    persons.home_taz),
                     index=persons.index)


@orca.column('persons_workplace')
def workplace_in_cbd(persons, land_use, settings):
    s = reindex(land_use.area_type, persons.workplace_taz)
    return s < settings['cbd_threshold']


# this is the placeholder for all the columns to update after the
# school location choice model
@orca.table()
def persons_school(persons):
    return pd.DataFrame(index=persons.index)


# same deal as distance_to_work but to school
@orca.column("persons_school")
def distance_to_school(persons, skim_dict):
    logger.debug("eval computed column persons_school.roundtrip_auto_time_to_school")
    distance_skim = skim_dict.get('DIST')
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.school_taz),
                     index=persons.index)


# this uses the free flow travel time in both directions
# MTC TM1 was MD and MD since term is free flow roundtrip_auto_time_to_school
@orca.column("persons_school")
def roundtrip_auto_time_to_school(persons, skim_dict):
    sovmd_skim = skim_dict.get(('SOV_TIME', 'MD'))
    return pd.Series(sovmd_skim.get(persons.home_taz,
                                    persons.school_taz) +
                     sovmd_skim.get(persons.school_taz,
                                    persons.home_taz),
                     index=persons.index)


# this is an idiom to grab the person of the specified type and check to see if
# there is 1 or more of that kind of person in each household
def presence_of(ptype, persons, at_home=False):
    if at_home:
        # if at_home, they need to be of given type AND at home
        bools = (persons.ptype_cat == ptype) & (persons.cdap_activity == "H")
    else:
        bools = persons.ptype_cat == ptype

    return other_than(persons.household_id, bools)


# this is the placeholder for all the columns to update after the
# workplace location choice model
@orca.table()
def persons_cdap(persons):
    return pd.DataFrame(index=persons.index)


@orca.column("persons_cdap")
def under16_not_at_school(persons):
    return (persons.ptype_cat.isin(["school", "preschool"]) &
            persons.cdap_activity.isin(["N", "H"]))


@orca.column('persons_cdap')
def has_preschool_kid_at_home(persons):
    return presence_of("preschool", persons, at_home=True)


@orca.column('persons_cdap')
def has_school_kid_at_home(persons):
    return presence_of("school", persons, at_home=True)

# ActivitySim
# See full license in LICENSE.txt.

"""
reindex household_id  in households and persons tables
legacy tables have household_ids starting at 930000000
which causes headaches for activitysim's automatic generation of trip and tour ids based on hosuehold_id
(predictable trip and tour ids are used for repeatable random number stream generation)
"""

import os
import sys

import numpy as np
import pandas as pd

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

file_names = {
    "households": "households.csv",
    "persons": "persons.csv",
    "land_use": "land_use.csv",
}

land_use_zone_col = "ZONE"
hh_zone_col = "zone_id"


def drop_and_dump(df, drop, msg, tag, output_dir):

    print("Checking for %s" % msg)
    if drop.any():
        print(
            "WARNING: dropping %s out of %s %s (%s)" % (drop.sum(), len(df), msg, tag)
        )
        df[drop].to_csv(os.path.join(output_dir, "%s.csv" % tag), index=False)
        df = df[~drop]

    return df


def create_subset(input_dir, output_dir, drop_dir):

    ###
    # land_use
    ###
    land_use_df = pd.read_csv(os.path.join(input_dir, file_names["land_use"]))
    land_use_df = land_use_df.sort_values(by=land_use_zone_col)
    land_use_df.to_csv(os.path.join(output_dir, file_names["land_use"]), index=False)

    print("zones: %s" % len(land_use_df))

    ###
    # households
    ###

    households = pd.read_csv(
        os.path.join(input_dir, file_names["households"]),
        dtype={"household_id": np.int64},
    )
    households = households.sort_values(by="household_id")
    households.rename(columns={"household_id": "legacy_household_id"}, inplace=True)

    raw_household_count = len(households)

    # all households must have a zone_id
    null_zones = households[hh_zone_col].isnull()
    households = drop_and_dump(
        households,
        null_zones,
        msg="households with null zones",
        tag="households_with_null_zones",
        output_dir=drop_dir,
    )
    households[hh_zone_col] = households[hh_zone_col].astype(np.int64)

    # all households zone_ids must be in land_use
    orphan_zones = ~households[hh_zone_col].isin(land_use_df[land_use_zone_col])
    households = drop_and_dump(
        households,
        orphan_zones,
        msg="households with unknown zones",
        tag="households_with_unknown_zones",
        output_dir=drop_dir,
    )

    # reindexed household_id as both index and column
    households.index = np.arange(1, len(households) + 1)
    households["household_id"] = households.index

    ###
    # persons
    ###
    persons = pd.read_csv(
        os.path.join(input_dir, file_names["persons"]),
        dtype={"household_id": np.int64, "person_id": np.int64},
    )
    persons = persons.sort_values(by=["household_id", "member_id"])
    persons.rename(
        columns={
            "person_id": "legacy_person_id",
            "household_id": "legacy_household_id",
        },
        inplace=True,
    )
    persons.legacy_household_id = persons.legacy_household_id.astype(np.int64)

    raw_person_count = len(persons)

    assert not persons.legacy_household_id.isnull().any()

    orphan_persons = ~persons.legacy_household_id.isin(households.legacy_household_id)
    persons = drop_and_dump(
        persons,
        orphan_persons,
        msg="persons without households",
        tag="persons_without_households",
        output_dir=drop_dir,
    )

    persons = pd.merge(
        persons,
        households[["legacy_household_id", "household_id"]],
        left_on="legacy_household_id",
        right_on="legacy_household_id",
        how="left",
    )
    assert not persons.household_id.isnull().any()
    persons.household_id = persons.household_id.astype(np.int64)

    # reindexed person_id as both index and column
    persons.index = np.arange(1, len(persons) + 1)
    persons["person_id"] = persons.index

    # check that we have the right number of persons in every household"
    assert (persons.groupby("household_id").size() == households.persons).all()

    # check that all persons in household have different member_id"
    persons_with_dupe_member_id = persons.duplicated(
        ["household_id", "member_id"], keep="first"
    )
    household_ids_with_dupe_member_id = persons.household_id[
        persons_with_dupe_member_id
    ].unique()
    households_with_dupe_members = households.household_id.isin(
        household_ids_with_dupe_member_id
    )
    persons_in_households_with_dupe_members = persons.household_id.isin(
        household_ids_with_dupe_member_id
    )

    print(
        "%s of %s persons_with_dupe_member_id"
        % (persons_with_dupe_member_id.sum(), len(persons))
    )
    persons = drop_and_dump(
        persons,
        persons_in_households_with_dupe_members,
        msg="persons in households with duplicate (household_id, member_id)",
        tag="persons_in_households_with_dupe_member_id",
        output_dir=drop_dir,
    )

    households = drop_and_dump(
        households,
        households_with_dupe_members,
        msg="households with duplicate persons.member_id",
        tag="households_with_dupe_member_id",
        output_dir=drop_dir,
    )

    missing_member1 = ~households.household_id.isin(
        persons.household_id[persons.member_id == 1]
    )
    # print("%s of %s households missing member_id 1" % (missing_member1.sum(), len(households)))
    assert not missing_member1.any()

    print(
        "Writing %s households. Dropped %s"
        % (len(households), raw_household_count - len(households))
    )
    households.to_csv(os.path.join(output_dir, file_names["households"]), index=False)

    print(
        "Writing %s persons. Dropped %s"
        % (len(persons), raw_person_count - len(persons))
    )
    persons.to_csv(os.path.join(output_dir, file_names["persons"]), index=False)


create_subset(input_dir="data_raw/", output_dir="data/", drop_dir="data_raw/dropped")

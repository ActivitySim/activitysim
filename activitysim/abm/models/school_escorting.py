# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing
from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.core.util import reindex

from .util import estimation, school_escort_tours_trips

logger = logging.getLogger(__name__)

# setting global defaults for max number of escortees and escortees in model
NUM_ESCORTEES = 3
NUM_CHAPERONES = 2


def determine_escorting_participants(choosers, persons, model_settings):
    """
    Determining which persons correspond to chauffer 1..n and escortee 1..n.
    Chauffers are those with the highest weight given by:
    weight = 100 * person type +  10 * gender + 1*(age > 25)
    and escortees are selected youngest to oldest.
    """
    global NUM_ESCORTEES
    global NUM_CHAPERONES
    NUM_ESCORTEES = model_settings.get("NUM_ESCORTEES", NUM_ESCORTEES)
    NUM_CHAPERONES = model_settings.get("NUM_CHAPERONES", NUM_CHAPERONES)

    ptype_col = model_settings.get("PERSONTYPE_COLUMN", "ptype")
    sex_col = model_settings.get("GENDER_COLUMN", "sex")
    age_col = model_settings.get("AGE_COLUMN", "age")

    escortee_age_cutoff = model_settings.get("ESCORTEE_AGE_CUTOFF", 16)
    chaperone_age_cutoff = model_settings.get("CHAPERONE_AGE_CUTOFF", 18)

    escortees = persons[
        persons.is_student
        & (persons[age_col] < escortee_age_cutoff)
        & (persons.cdap_activity == "M")
    ]
    households_with_escortees = escortees["household_id"]

    # can specify different weights to determine chaperones
    persontype_weight = model_settings.get("PERSON_WEIGHT", 100)
    gender_weight = model_settings.get("PERSON_WEIGHT", 10)
    age_weight = model_settings.get("AGE_WEIGHT", 1)

    # can we move all of these to a config file?
    chaperones = persons[
        (persons[age_col] > chaperone_age_cutoff)
        & persons.household_id.isin(households_with_escortees)
    ]

    chaperones["chaperone_weight"] = (
        (persontype_weight * chaperones[ptype_col])
        + (gender_weight * np.where(chaperones[sex_col] == 1, 1, 2))
        + (age_weight * np.where(chaperones[age_col] > 25, 1, 0))
    )

    chaperones["chaperone_num"] = (
        chaperones.sort_values("chaperone_weight", ascending=False)
        .groupby("household_id")
        .cumcount()
        + 1
    )
    escortees["escortee_num"] = (
        escortees.sort_values("age", ascending=True).groupby("household_id").cumcount()
        + 1
    )

    participant_columns = []
    for i in range(1, NUM_CHAPERONES + 1):
        choosers["chauf_id" + str(i)] = (
            chaperones[chaperones["chaperone_num"] == i]
            .reset_index()
            .set_index("household_id")
            .reindex(choosers.index)["person_id"]
        )
        participant_columns.append("chauf_id" + str(i))
    for i in range(1, NUM_ESCORTEES + 1):
        choosers["child_id" + str(i)] = (
            escortees[escortees["escortee_num"] == i]
            .reset_index()
            .set_index("household_id")
            .reindex(choosers.index)["person_id"]
        )
        participant_columns.append("child_id" + str(i))

    return choosers, participant_columns


def check_alts_consistency(alts):
    """
    Checking to ensure that the alternatives file is consistent with
    the number of chaperones and escortees set in the model settings.
    """
    for i in range(1, NUM_ESCORTEES + 1):
        chauf_col = f"chauf{i}"
        # The number of chauf columns should equal the number of escortees
        assert chauf_col in alts.columns, f"Missing {chauf_col} in alternatives file"

        # Each escortee should be able to be escorted by each chaperone with ride hail or pure escort
        assert alts[chauf_col].max() == (NUM_CHAPERONES * 2)
    return


def add_prev_choices_to_choosers(choosers, choices, alts, stage):
    # adding choice details to chooser table
    escorting_choice = "school_escorting_" + stage
    choosers[escorting_choice] = choices

    stage_alts = alts.copy()
    stage_alts.columns = stage_alts.columns + "_" + stage

    choosers = (
        choosers.reset_index()
        .merge(
            stage_alts,
            how="left",
            left_on=escorting_choice,
            right_on=stage_alts.index.name,
        )
        .set_index("household_id")
    )

    return choosers


def create_bundle_attributes(row):
    """
    Parse a bundle to determine escortee numbers and tour info.
    """
    escortee_str = ""
    escortee_num_str = ""
    school_dests_str = ""
    school_starts_str = ""
    school_ends_str = ""
    school_tour_ids_str = ""
    num_escortees = 0

    for child_num in row["child_order"]:
        child_num = str(child_num)
        child_id = int(row["bundle_child" + child_num])

        if child_id > 0:
            num_escortees += 1
            school_dest = str(int(row["school_destination_child" + child_num]))
            school_start = str(int(row["school_start_child" + child_num]))
            school_end = str(int(row["school_end_child" + child_num]))
            school_tour_id = str(int(row["school_tour_id_child" + child_num]))

            if escortee_str == "":
                escortee_str = str(child_id)
                escortee_num_str = str(child_num)
                school_dests_str = school_dest
                school_starts_str = school_start
                school_ends_str = school_end
                school_tour_ids_str = school_tour_id
            else:
                escortee_str = escortee_str + "_" + str(child_id)
                escortee_num_str = escortee_num_str + "_" + str(child_num)
                school_dests_str = school_dests_str + "_" + school_dest
                school_starts_str = school_starts_str + "_" + school_start
                school_ends_str = school_ends_str + "_" + school_end
                school_tour_ids_str = school_tour_ids_str + "_" + school_tour_id

    row["escortees"] = escortee_str
    row["escortee_nums"] = escortee_num_str
    row["num_escortees"] = num_escortees
    row["school_destinations"] = school_dests_str
    row["school_starts"] = school_starts_str
    row["school_ends"] = school_ends_str
    row["school_tour_ids"] = school_tour_ids_str
    return row


def create_school_escorting_bundles_table(choosers, tours, stage):
    """
    Creates a table that has one row for every school escorting bundle.
    Additional calculations are performed to help facilitate tour and
    trip creation including escortee order, times, etc.

    Parameters
    ----------
    choosers : pd.DataFrame
        households pre-processed for the school escorting model
    tours : pd.Dataframe
        mandatory tours
    stage : str
        inbound or outbound_cond

    Returns
    -------
    bundles : pd.DataFrame
        one school escorting bundle per row
    """
    # making a table of bundles
    choosers = choosers.reset_index()
    choosers = choosers.loc[choosers.index.repeat(choosers["nbundles"])]

    bundles = pd.DataFrame()
    # bundles.index = choosers.index
    bundles["household_id"] = choosers["household_id"]
    bundles["home_zone_id"] = choosers["home_zone_id"]
    bundles["school_escort_direction"] = (
        "outbound" if "outbound" in stage else "inbound"
    )
    bundles["bundle_num"] = bundles.groupby("household_id").cumcount() + 1

    # initialize values
    bundles["chauf_type_num"] = 0

    # getting bundle school start times and locations
    school_tours = tours[(tours.tour_type == "school") & (tours.tour_num == 1)]

    school_starts = school_tours.set_index("person_id").start
    school_ends = school_tours.set_index("person_id").end
    school_destinations = school_tours.set_index("person_id").destination
    school_origins = school_tours.set_index("person_id").origin
    school_tour_ids = school_tours.reset_index().set_index("person_id").tour_id

    for child_num in range(1, NUM_ESCORTEES + 1):
        i = str(child_num)
        bundles["bundle_child" + i] = np.where(
            choosers["bundle" + i] == bundles["bundle_num"],
            choosers["child_id" + i],
            -1,
        )
        bundles["chauf_type_num"] = np.where(
            (choosers["bundle" + i] == bundles["bundle_num"]),
            choosers["chauf" + i],
            bundles["chauf_type_num"],
        )
        bundles["time_home_to_school" + i] = np.where(
            (choosers["bundle" + i] == bundles["bundle_num"]),
            choosers["time_home_to_school" + i],
            np.NaN,
        )

        bundles["school_destination_child" + i] = reindex(
            school_destinations, bundles["bundle_child" + i]
        )
        bundles["school_origin_child" + i] = reindex(
            school_origins, bundles["bundle_child" + i]
        )
        bundles["school_start_child" + i] = reindex(
            school_starts, bundles["bundle_child" + i]
        )
        bundles["school_end_child" + i] = reindex(
            school_ends, bundles["bundle_child" + i]
        )
        bundles["school_tour_id_child" + i] = reindex(
            school_tour_ids, bundles["bundle_child" + i]
        )

    # each chauffeur option has ride share or pure escort
    bundles["chauf_num"] = np.ceil(bundles["chauf_type_num"].div(2)).astype(int)

    # getting bundle chauffeur id based on the chauffeur num
    bundles["chauf_id"] = -1
    for i in range(1, NUM_CHAPERONES + 1):
        bundles["chauf_id"] = np.where(
            bundles["chauf_num"] == i,
            choosers["chauf_id" + str(i)],
            bundles["chauf_id"],
        )
    bundles["chauf_id"] = bundles["chauf_id"].astype(int)
    assert (
        bundles["chauf_id"] > 0
    ).all(), "Invalid chauf_id's for school escort bundles!"

    # odd chauf_type_num means ride share, even means pure escort
    # this comes from the way the alternatives file is constructed where chauf_id is
    # incremented for each possible chauffeur and for each tour type
    bundles["escort_type"] = np.where(
        bundles["chauf_type_num"].mod(2) == 1, "ride_share", "pure_escort"
    )

    # This is just pulled from the pre-processor. Will break if removed or renamed in pre-processor
    # I think this is still a better implmentation than re-calculating here...
    school_time_cols = [
        "time_home_to_school" + str(i) for i in range(1, NUM_ESCORTEES + 1)
    ]
    bundles["outbound_order"] = list(bundles[school_time_cols].values.argsort() + 1)
    bundles["inbound_order"] = list(
        (-1 * bundles[school_time_cols]).values.argsort() + 1
    )  # inbound gets reverse order
    bundles["child_order"] = np.where(
        bundles["school_escort_direction"] == "outbound",
        bundles["outbound_order"],
        bundles["inbound_order"],
    )

    bundles = bundles.apply(lambda row: create_bundle_attributes(row), axis=1)

    # getting chauffer mandatory times
    mandatory_escort_tours = tours[
        (tours.tour_category == "mandatory") & (tours.tour_num == 1)
    ]
    # bundles["first_mand_tour_start_time"] = reindex(
    #     mandatory_escort_tours.set_index("person_id").start, bundles["chauf_id"]
    # )
    # bundles["first_mand_tour_end_time"] = reindex(
    #     mandatory_escort_tours.set_index("person_id").end, bundles["chauf_id"]
    # )
    bundles["first_mand_tour_id"] = reindex(
        mandatory_escort_tours.reset_index().set_index("person_id").tour_id,
        bundles["chauf_id"],
    )
    bundles["first_mand_tour_dest"] = reindex(
        mandatory_escort_tours.reset_index().set_index("person_id").destination,
        bundles["chauf_id"],
    )
    bundles["first_mand_tour_purpose"] = reindex(
        mandatory_escort_tours.reset_index().set_index("person_id").tour_type,
        bundles["chauf_id"],
    )

    bundles["Alt"] = choosers["Alt"]
    bundles["Description"] = choosers["Description"]

    return bundles


@inject.step()
def school_escorting(
    households, households_merged, persons, tours, chunk_size, trace_hh_id
):
    """
    school escorting model

    The school escorting model determines whether children are dropped-off at or
    picked-up from school, simultaneously with the driver responsible for
    chauffeuring the children, which children are bundled together on half-tours,
    and the type of tour (pure escort versus rideshare).

    Run iteratively for an outbound choice, an inbound choice, and an outbound choice
    conditional on the inbound choice. The choices for inbound and outbound conditional
    are used to create school escort tours and trips.

    Updates / adds the following tables to the pipeline:

    ::

        - households with school escorting choice
        - tours including pure school escorting
        - school_escort_tours which contains only pure school escort tours
        - school_escort_trips
        - timetable to avoid joint tours scheduled over school escort tours

    """
    trace_label = "school_escorting_simulate"
    model_settings_file_name = "school_escorting.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    persons = persons.to_frame()
    households = households.to_frame()
    households_merged = households_merged.to_frame()
    tours = tours.to_frame()

    alts = simulate.read_model_alts(model_settings["ALTS"], set_index="Alt")

    households_merged, participant_columns = determine_escorting_participants(
        households_merged, persons, model_settings
    )

    check_alts_consistency(alts)

    constants = config.get_model_constants(model_settings)
    locals_dict = {}
    locals_dict.update(constants)

    school_escorting_stages = ["outbound", "inbound", "outbound_cond"]
    escort_bundles = []
    choices = None
    for stage_num, stage in enumerate(school_escorting_stages):
        stage_trace_label = trace_label + "_" + stage
        estimator = estimation.manager.begin_estimation("school_escorting_" + stage)

        model_spec_raw = simulate.read_model_spec(
            file_name=model_settings[stage.upper() + "_SPEC"]
        )
        coefficients_df = simulate.read_model_coefficients(
            file_name=model_settings[stage.upper() + "_COEFFICIENTS"]
        )
        model_spec = simulate.eval_coefficients(
            model_spec_raw, coefficients_df, estimator
        )

        # allow for skipping sharrow entirely in this model with `sharrow_skip: true`
        # or skipping stages selectively with a mapping of the stages to skip
        sharrow_skip = model_settings.get("sharrow_skip", False)
        stage_sharrow_skip = False  # default is false unless set below
        if sharrow_skip:
            if isinstance(sharrow_skip, dict):
                stage_sharrow_skip = sharrow_skip.get(stage.upper(), False)
            else:
                stage_sharrow_skip = True
        if stage_sharrow_skip:
            locals_dict["_sharrow_skip"] = True
        else:
            locals_dict.pop("_sharrow_skip", None)

        # reduce memory by limiting columns if selected columns are supplied
        chooser_columns = model_settings.get("SIMULATE_CHOOSER_COLUMNS", None)
        if chooser_columns is not None:
            chooser_columns = chooser_columns + participant_columns
            choosers = households_merged[chooser_columns]
        else:
            choosers = households_merged

        # add previous data to stage
        if stage_num >= 1:
            choosers = add_prev_choices_to_choosers(
                choosers, choices, alts, school_escorting_stages[stage_num - 1]
            )

        locals_dict.update(coefficients_df)

        logger.info("Running %s with %d households", stage_trace_label, len(choosers))

        preprocessor_settings = model_settings.get("preprocessor_" + stage, None)
        if preprocessor_settings:
            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=stage_trace_label,
            )

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        log_alt_losers = config.setting("log_alt_losers", False)

        choices = interaction_simulate(
            choosers=choosers,
            alternatives=alts,
            spec=model_spec,
            log_alt_losers=log_alt_losers,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            trace_label=stage_trace_label,
            trace_choice_name="school_escorting_" + "stage",
            estimator=estimator,
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "households", "school_escorting_" + stage
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # no need to reindex as we used all households
        escorting_choice = "school_escorting_" + stage
        households[escorting_choice] = choices

        # tracing each step -- outbound, inbound, outbound_cond
        tracing.print_summary(
            escorting_choice, households[escorting_choice], value_counts=True
        )

        if trace_hh_id:
            tracing.trace_df(households, label=escorting_choice, warn_if_empty=True)

        if stage_num >= 1:
            choosers["Alt"] = choices
            choosers = choosers.join(alts, how="left", on="Alt")
            bundles = create_school_escorting_bundles_table(
                choosers[choosers["Alt"] > 1], tours, stage
            )
            escort_bundles.append(bundles)

    escort_bundles = pd.concat(escort_bundles)
    escort_bundles["bundle_id"] = (
        escort_bundles["household_id"] * 10
        + escort_bundles.groupby("household_id").cumcount()
        + 1
    )
    escort_bundles.sort_values(
        by=["household_id", "school_escort_direction"],
        ascending=[True, False],
        inplace=True,
    )

    school_escort_tours = school_escort_tours_trips.create_pure_school_escort_tours(
        escort_bundles
    )
    chauf_tour_id_map = {
        v: k for k, v in school_escort_tours["bundle_id"].to_dict().items()
    }
    escort_bundles["chauf_tour_id"] = np.where(
        escort_bundles["escort_type"] == "ride_share",
        escort_bundles["first_mand_tour_id"],
        escort_bundles["bundle_id"].map(chauf_tour_id_map),
    )

    tours = school_escort_tours_trips.add_pure_escort_tours(tours, school_escort_tours)
    tours = school_escort_tours_trips.process_tours_after_escorting_model(
        escort_bundles, tours
    )

    school_escort_trips = school_escort_tours_trips.create_school_escort_trips(
        escort_bundles
    )

    # update pipeline
    pipeline.replace_table("households", households)
    pipeline.replace_table("tours", tours)
    pipeline.get_rn_generator().drop_channel("tours")
    pipeline.get_rn_generator().add_channel("tours", tours)
    pipeline.replace_table("escort_bundles", escort_bundles)
    # save school escorting tours and trips in pipeline so we can overwrite results from downstream models
    pipeline.replace_table("school_escort_tours", school_escort_tours)
    pipeline.replace_table("school_escort_trips", school_escort_trips)

    # updating timetable object with pure escort tours so joint tours do not schedule ontop
    timetable = inject.get_injectable("timetable")

    # Need to do this such that only one person is in nth_tours
    # thus, looping through tour_category and tour_num
    # including mandatory tours because their start / end times may have
    # changed to match the school escort times
    for tour_category in tours.tour_category.unique():
        for tour_num, nth_tours in tours[tours.tour_category == tour_category].groupby(
            "tour_num", sort=True
        ):
            timetable.assign(
                window_row_ids=nth_tours["person_id"], tdds=nth_tours["tdd"]
            )

    timetable.replace_table()

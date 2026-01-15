import os
from pathlib import Path
import numpy as np
import pandas as pd

import pytest
import openmatrix as omx


from activitysim.abm.tables import shadow_pricing
from activitysim.core import workflow, los
from activitysim.core.configuration.logit import TourLocationComponentSettings
from activitysim.abm.models.location_choice import run_location_choice


LAND_USE_FIELDS = [
    "e01_nrm",
    "e02_constr",
    "e03_manuf",
    "e04_whole",
    "e05_retail",
    "e06_trans",
    "e07_utility",
    "e08_infor",
    "e09_finan",
    "e10_pstsvc",
    "e11_compmgt",
    "e12_admsvc",
    "e13_edusvc",
    "e14_medfac",
    "e15_hospit",
    "e16_leisure",
    "e17_othsvc",
    "e18_pubadm",
]


@pytest.fixture(scope="session")
def example_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("example")
    config_dir = root / "configs"
    config_dir.mkdir()

    data_dir = root / "data"
    data_dir.mkdir()

    return root


@pytest.fixture(scope="module")
def model_settings(example_root, state):

    model_settings = TourLocationComponentSettings.read_settings_file(
        state.filesystem, "school_location.yaml"
    )

    return model_settings


@pytest.fixture(scope="module")
def state(
    example_root, location_coeffs_configs_csv, location_sample_configs_csv
) -> workflow.State:

    settings = """
        input_table_list:
            - tablename: households
            - tablename: persons
            - tablename: land_use
        """

    network_los_yaml = """
                zone_system: 2
                taz_skims: skims*.omx
                skim_time_periods:
                    time_window: 1440
                    period_minutes: 30
                    periods: [12]
                    labels: &skim_time_period_labels ['AM']
                    """
    shadow_pricing_settings = """
                shadow_pricing_models:
                    school: school_location

                PERCENT_TOLERANCE: 5

                school_segmentation_targets:
                    highschool: K_8
                    gradeschool: G9_12
                    university: Univ_Enrollment

                SHADOW_PRICE_METHOD: simulation
                """
    school_location_csv = f"""
{location_sample_configs_csv}
util_mode_choice_logsum,Mode choice logsum,mode_choice_logsum,coef_mode_logsum_uni,coef_mode_logsum,coef_mode_logsum
util_sample_of_corrections_factor,Sample of alternatives correction factor,"@np.minimum(np.log(df.pick_count/df.prob), 60)",1,1,1
"""

    skim_matrix = np.array(
        [
            [0.42, 0.89, 4.33, 10.31, 9.98],
            [0.89, 0.39, 3.76, 10.05, 9.72],
            [4.19, 3.61, 0.85, 10.02, 9.69],
            [10.57, 9.99, 9.81, 0.16, 0.37],
            [10.19, 9.61, 9.43, 0.37, 0.16],
        ]
    )

    choice_sizes = pd.DataFrame(
        {
            "model_selector": ["school", "school", "school"],
            "segment": ["university", "gradeschool", "highschool"],
            "tot_hhs": [0, 0, 0],
            "K_8": [0, 1, 0],
            "G9_12": [0, 0, 1],
            "Univ_Enrollment": [1, 0, 0],
        }
    )

    for col in LAND_USE_FIELDS:
        choice_sizes[col] = 0

    school_location_settings = """
        SIMULATE_CHOOSER_COLUMNS:
            - home_zone_id
            - school_segment
            - household_id
            - is_student
            - age_0_to_5
            - age_6_to_12
            - pemploy
        CHOOSER_ORIG_COL_NAME: home_zone_id
        ALT_DEST_COL_NAME: alt_dest
        SAMPLE_SIZE: 30
        IN_PERIOD: 14
        OUT_PERIOD: 8
        DEST_CHOICE_COLUMN_NAME: school_zone_id
        SAMPLE_SPEC: school_location_sample.csv
        SPEC: school_location.csv
        COEFFICIENTS: school_location_coeffs.csv
        CHOOSER_TABLE_NAME: persons
        MODEL_SELECTOR: school
        CHOOSER_SEGMENT_COLUMN_NAME: school_segment
        CHOOSER_FILTER_COLUMN_NAME: is_student
        SEGMENT_IDS:
            university: 3
            highschool: 2
            gradeschool: 1
        SHADOW_PRICE_TABLE: school_shadow_prices
        MODELED_SIZE_TABLE: school_modeled_size
            """

    school_loc_yaml = example_root / "configs" / "school_location.yaml"
    school_loc_yaml.write_text(school_location_settings)

    taz_equivs = [2103, 2104, 2115, 2142, 2144]

    # example_root = tmp_path_factory.mktemp("example")

    settings_file = example_root / "configs" / "settings.yaml"
    settings_file.write_text(settings)

    yaml_file = example_root / "configs" / "network_los.yaml"
    yaml_file.write_text(network_los_yaml)

    sp_yaml = example_root / "configs" / "shadow_pricing.yaml"
    sp_yaml.write_text(shadow_pricing_settings)

    choice_sizes.to_csv(
        example_root / "configs" / "destination_choice_size_terms.csv", index=False
    )

    location_coeffs = example_root / "configs" / "school_location_coeffs.csv"
    location_coeffs.write_text(location_coeffs_configs_csv)

    location_sample = example_root / "configs" / "school_location_sample.csv"
    location_sample.write_text(location_sample_configs_csv)

    school_location = example_root / "configs" / "school_location.csv"
    school_location.write_text(school_location_csv)

    skims = omx.open_file(example_root / "data" / "skims.omx", "w")
    skims["DIST"] = skim_matrix
    skims.create_mapping("zone_number", taz_equivs)
    skims.close()

    state = workflow.State.make_default(example_root)

    return state


@pytest.fixture(scope="module")
def persons() -> pd.DataFrame:
    persons = pd.DataFrame(
        {
            "person_id": [
                2664688,
                2664689,
                2668012,
                2668013,
                2701577,
                2701578,
                2860810,
                2860811,
                2865544,
                2865545,
                2865546,
            ],
            "age": [13.0, 12.0, 13.0, 12.0, 11.0, 10.0, 15.0, 14.0, 10.0, 14.0, 15.0],
            "household_id": [
                1080351,
                1080351,
                1081684,
                1081684,
                1094369,
                1094369,
                1156249,
                1156249,
                1158612,
                1158612,
                1158612,
            ],
            "member_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3],
            "sex": [1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2],
            "relate": [0, 1, 0, 1, 0, 1, 0, 12, 0, 2, 13],
            "race_id": [1, 1, 2, 2, 4, 4, 4, 4, 1, 1, 1],
            "esr": [6.0, 6.0, 2.0, 6.0, 1.0, 6.0, 1.0, 6.0, 1.0, 1.0, 1.0],
            "wkhp": [-9.0, -9.0, 3.0, -9.0, 50.0, -9.0, 40.0, -9.0, 50.0, 37.0, 40.0],
            "wkw": [-9.0, -9.0, 3.0, -9.0, 1.0, -9.0, 3.0, -9.0, 1.0, 1.0, 1.0],
            "schg": [-9.0, -9.0, -9.0, -9.0, -9.0, -9.0, 16.0, 16.0, -9.0, -9.0, -9.0],
            "mil": [4.0, 4.0, 4.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "naicsp": [
                "-9",
                "-9",
                "5121",
                "-9",
                "8139Z",
                "-9",
                "611M1",
                "-9",
                "7112",
                "6241",
                "5417",
            ],
            "industry": [0.0, 0.0, 8.0, 0.0, 17.0, 0.0, 13.0, 0.0, 16.0, 14.0, 10.0],
            "maz_seqid": [
                22660.0,
                22660.0,
                22670.0,
                22670.0,
                22734.0,
                22734.0,
                22803.0,
                22803.0,
                22799.0,
                22799.0,
                22799.0,
            ],
            "zone_id": [
                2103.0,
                2103.0,
                2104.0,
                2104.0,
                2115.0,
                2115.0,
                2144.0,
                2144.0,
                2142.0,
                2142.0,
                2142.0,
            ],
            "school_segment": [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2],
            "age_0_to_5": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "age_6_to_12": [
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                False,
            ],
            "pemploy": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "value_of_time": [
                3.729377,
                3.729377,
                33.35,
                33.35,
                3.030231,
                3.030231,
                4.512599,
                4.512599,
                11.620478,
                11.620478,
                11.620478,
            ],
            "is_student": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "home_zone_id": [
                22660.0,
                22660.0,
                22670.0,
                22670.0,
                22734.0,
                22734.0,
                22803.0,
                22803.0,
                22799.0,
                22799.0,
                22799.0,
            ],
            "school_zone_id": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        }
    )

    return persons


@pytest.fixture(scope="module")
def households() -> pd.DataFrame:
    households = pd.DataFrame(
        {
            "household_id": [1156249, 1080351, 1094369, 1081684, 1158612],
            "persons": [2, 2, 2, 2, 3],
            "age_of_head": [26.0, 67.0, 68.0, 72.0, 66.0],
            "race_id": [4.0, 1.0, 4.0, 2.0, 1.0],
            "cars": [2.0, 2.0, 2.0, 2.0, 2.0],
            "children": [0.0, 0.0, 0.0, 0.0, 0.0],
            "type": [1, 1, 1, 1, 1],
            "hincp": [96200.0, 93100.0, 153000.0, 89000.0, 114280.0],
            "adjinc": [1010145, 1054606, 1073449, 1031452, 1080470],
            "hht": [7.0, 1.0, 1.0, 1.0, 2.0],
            "maz": [22803.0, 22660.0, 22734.0, 22670.0, 22799.0],
            "taz": [2144.0, 2103.0, 2115.0, 2104.0, 2142.0],
            "auto_ownership": [2, 2, 2, 2, 2],
            "home_zone_id": [22803.0, 22660.0, 22734.0, 22670.0, 22799.0],
        }
    )

    return households


@pytest.fixture(scope="module")
def land_use() -> pd.DataFrame:
    land_use = pd.DataFrame(
        {
            "MAZ": [22660, 22670, 22734, 22799, 22803],
            "TAZ": [2103, 2104, 2115, 2142, 2144],
            "hhs_pop": [2, 2, 2, 3, 2],
            "K_8": [150, 100, 200, 300, 250],
            "G9_12": [150, 100, 200, 300, 250],
            "Univ_Enrollment": [0, 0, 0, 0, 0],
            "tot_pop": [2, 2, 2, 3, 2],
            "tot_hhs": [1, 1, 1, 1, 1],
        }
    )

    for col in LAND_USE_FIELDS:
        land_use[col] = 0

    return land_use


@pytest.fixture(scope="module")
def location_sample_configs_csv():
    csv_content = """Label,Description,Expression,university,highschool,gradeschool
local_dist,,_DIST@skims['DIST'],1,1,1
util_dist_0_1,"Distance, piecewise linear from 0 to 1 miles","@_DIST.clip(0,1)",coef_univ_dist_0_1,0,0
util_dist_1_2,"Distance, piecewise linear from 1 to 2 miles","@(_DIST-1).clip(0,1)",coef_univ_dist_1_2,0,0
util_dist_2_5,"Distance, piecewise linear from 2 to 5 miles","@(_DIST-2).clip(0,3)",coef_univ_dist_2_5,0,0
util_dist_5_15,"Distance, piecewise linear from 5 to 15 miles","@(_DIST-5).clip(0,10)",coef_univ_dist_5_15,0,0
util_dist_15_up,"Distance, piecewise linear for 15+ miles",@(_DIST-15.0).clip(0),coef_univ_dist_15_up,0,0
util_size_variable,Size variable,@(df['size_term'] * df['shadow_price_size_term_adjustment']).apply(np.log1p),1,1,1
util_utility_adjustment,utility adjustment,@df['shadow_price_utility_adjustment'],1,1,1
util_no_attractions,No attractions,@df['size_term']==0,-999,-999,-999
util_dist,Distance,@_DIST,0,coef_dist,coef_dist
util_dist_squared,"Distance squared, capped at 20 miles","@(_DIST).clip(0,20)**2",0,coef_dist_squared,coef_dist_squared
util_dist_cubed,"Distance cubed, capped at 20 miles","@(_DIST).clip(0,20)**3",0,coef_dist_cubed,coef_dist_cubed
util_dist_logged,Distance logged,@(_DIST).apply(np.log1p),0,coef_dist_logged,coef_dist_logged
util_dist_part_time,"Distance,part time",@(df['pemploy']==2) * _DIST,0,coef_dist_part_time,coef_dist_part_time
util_dist_child_0_5,"Distance,child 0  to 5",@(df['age_0_to_5']==True) * _DIST,0,coef_dist_child_0_5,coef_dist_child_0_5
util_dist_child_6_12,"Distance,child 6 to 12",@(df['age_6_to_12']==True) * _DIST,0,coef_dist_child_6_12,coef_dist_child_6_12
"""
    return csv_content


@pytest.fixture(scope="module")
def location_coeffs_configs_csv():
    csv_content = """coefficient_name,value,constrain
coef_univ_dist_0_1,-3.2451,F
coef_univ_dist_1_2,-2.7011,F
coef_univ_dist_2_5,-0.5707,F
coef_univ_dist_5_15,-0.5002,F
coef_univ_dist_15_up,-0.073,F
coef_mode_logsum_uni,0.5358,F
coef_dist,-0.1560,F
coef_dist_squared,-0.0116,F
coef_dist_cubed,0.0005,F
coef_dist_logged,-0.9316,F
coef_dist_part_time,-0.0985,F
coef_dist_child_0_5,0.0236,F
coef_dist_child_6_12,-0.0657,F
coef_mode_logsum,0.4,F
"""
    return csv_content


@pytest.fixture(scope="module")
def network_los(state, persons, households, land_use) -> los.Network_LOS:

    persons["is_student"] = True
    land_use["zone_id"] = land_use["MAZ"]
    land_use.set_index("zone_id", inplace=True)
    households["home_zone_id"] = households["maz"]

    state.add_table("persons", persons)
    state.add_table("households", households)
    state.add_table("land_use", land_use)

    persons_merged = pd.merge(persons, households, on="household_id")
    persons_merged = pd.merge(
        persons_merged, land_use.rename(columns={"TAZ": "taz"}), on="taz"
    )

    persons_merged["home_zone_id"] = persons_merged["MAZ"]
    persons_merged["TAZ"] = persons_merged["taz"]
    persons_merged = persons_merged.set_index("person_id")

    state.add_table("persons_merged", persons_merged)

    network_los = los.Network_LOS(state)
    state.settings.use_shadow_pricing = True

    shadow_pricing.add_size_tables(state, None, scale=False)

    network_los.maz_taz_df = land_use[["MAZ", "TAZ"]]

    network_los.skim_dicts["taz"] = network_los.create_skim_dict("taz")
    network_los.skim_dicts["maz"] = network_los.create_skim_dict("maz")

    return network_los


def check_shadow_prices(spc, iteration):
    """
    Check shadow prices against expected values for each iteration.
    Only checking the highschool segment here as an example.
    Highschool segment was chosen because it has some zones that had
    shadow prices change from open to closed to open over the iterations.
    (See ActivitySim Issue #820)
    """
    if iteration == 1:
        # initial iteration, all shadow prices should be zero
        assert (spc.shadow_prices["highschool"] == 0).all()
    elif iteration == 2:
        highschool_expected = pd.Series(
            index=[22660, 22670, 22734, 22799, 22803],
            data=[0.0, -999.0, 0.0, 0.0, -999.0],
        )
        assert (spc.shadow_prices["highschool"] == highschool_expected).all()
    elif iteration == 3:
        highschool_expected = pd.Series(
            index=[22660, 22670, 22734, 22799, 22803],
            data=[-999.0, -999.0, -999.0, 0.0, -999.0],
        )
        assert (spc.shadow_prices["highschool"] == highschool_expected).all()
    elif iteration == 4:
        # converged from here onward
        assert (spc.shadow_prices["highschool"] == -999).all()
    elif iteration == 5:
        assert (spc.shadow_prices["highschool"] == -999).all()
    else:
        assert False, "Unexpected iteration number in shadow pricing test"


def test_shadow_pricing_simulate(state, model_settings, network_los):
    """
    We iterate the location choice algorithm with shadow pricing and check if any closed zone
    is repoening after they are updated.
    """
    model_settings.LOGSUM_SETTINGS = None

    spc = shadow_pricing.load_shadow_price_calculator(state, model_settings)

    MAX_ITERATIONS = 5

    chooser_segment_column = "school_segment"

    save_sample_df = choices_df = None

    persons_merged = state.get_dataframe("persons_merged")

    for iteration in range(1, MAX_ITERATIONS + 1):

        old_shadow_prices = spc.shadow_prices["highschool"].values

        persons_merged_df_ = persons_merged.copy()

        if spc.use_shadow_pricing and iteration > 1:
            spc.update_shadow_prices(state)

            if spc.shadow_settings.SHADOW_PRICE_METHOD == "simulation":
                # filter from the sampled persons
                persons_merged_df_ = persons_merged_df_[
                    persons_merged_df_.index.isin(spc.sampled_persons.index)
                ]
                persons_merged_df_ = persons_merged_df_.sort_index()

        choices_df_, save_sample_df = run_location_choice(
            state,
            persons_merged_df_,
            network_los,
            shadow_price_calculator=spc,
            want_logsums=False,
            want_sample_table=False,
            estimator=None,
            model_settings=model_settings,
            chunk_size=0,
            chunk_tag="school_location",
            trace_label=f"school_location_{iteration}",
        )
        if spc.use_shadow_pricing:
            # handle simulation method
            if (
                spc.shadow_settings.SHADOW_PRICE_METHOD == "simulation"
                and iteration > 1
            ):
                # if a process ends up with no sampled workers in it, hence an empty choice_df_, then choice_df wil be what it was previously
                if len(choices_df_) != 0:
                    choices_df = pd.concat([choices_df, choices_df_], axis=0)
                    choices_df_index = choices_df_.index.name
                    choices_df = choices_df.reset_index()
                    # update choices of workers/students
                    choices_df = choices_df.drop_duplicates(
                        subset=[choices_df_index], keep="last"
                    )
                    choices_df = choices_df.set_index(choices_df_index)
                    choices_df = choices_df.sort_index()
            else:
                choices_df = choices_df_.copy()

        new_shadow_prices = spc.shadow_prices["highschool"].values

        assert not any((old_shadow_prices == -999) & (new_shadow_prices != -999))
        check_shadow_prices(spc, iteration)

        spc.set_choices(
            choices=choices_df["choice"],
            segment_ids=persons_merged[chooser_segment_column].reindex(
                choices_df.index
            ),
        )

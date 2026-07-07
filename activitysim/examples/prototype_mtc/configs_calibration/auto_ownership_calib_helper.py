import pandas as pd
import os

SURVEY_DATA_FOLDER = r"U:\projects\clients\ASIM\autocalibration\calibration_test_mtc\data"

def report_auto_ownership():
    model_hhs = households
    survey_hhs = None
    try:
        survey_hhs = pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_households.csv"))
    except FileNotFoundError:
        raise FileNotFoundError(f"No survey file override_households.csv found in {SURVEY_DATA_FOLDER}!")

    model_summary = model_hhs.auto_ownership.value_counts(normalize=True).sort_index().fillna(0)
    # survey_summary = survey_hhs.groupby("auto_ownership").household_weight.sum()
    # survey_summary = survey_summary / survey_hhs.household_weight.sum()
    survey_summary = survey_hhs.auto_ownership.value_counts(normalize=True).sort_index().fillna(0)

    summary_df = (
        pd.DataFrame({"model": model_summary, "survey": survey_summary})
        .reset_index()
        .rename(columns={"index": "num_autos"})
    )

    print(summary_df)

    # plot comparing model and survey distributions
    import matplotlib.pyplot as plt

    summary_df.plot(x="auto_ownership", y=["model", "survey"], kind="bar")
    plt.title("Auto Ownership Distribution: Model vs Survey")
    plt.xlabel("Number of Autos")
    plt.ylabel("Proportion of Households")
    plt.legend(title="Data Source")
    plt.savefig(os.path.join(component_output_dir, "auto_ownership_comparison.png"))
    plt.close()
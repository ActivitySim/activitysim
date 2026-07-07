import matplotlib.pyplot as plt
import pandas as pd
import os

SURVEY_DATA_FOLDER = "activitysim/examples/example_estimation/data_sf/survey_data"

def report_auto_ownership(context):
    model_hhs = context["households"]
    survey_hhs = pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_households.csv"))

    model_summary = model_hhs.auto_ownership.value_counts(normalize=True).sort_index().fillna(0)
    survey_summary = survey_hhs.auto_ownership.value_counts(normalize=True).sort_index().fillna(0)
    summary_df = (
        pd.DataFrame({"model": model_summary, "survey": survey_summary})
        .reset_index()
        .rename(columns={"index": "num_autos"})
    )

    # plot comparing model and survey distributions
    summary_df.plot(x="auto_ownership", y=["model", "survey"], kind="bar")
    plt.title("Auto Ownership Distribution: Model vs Survey")
    plt.xlabel("Number of Autos")
    plt.ylabel("Proportion of Households")
    plt.legend(title="Data Source")
    plt.savefig(os.path.join(context["component_output_dir"], "auto_ownership_comparison.png"))
    plt.close()
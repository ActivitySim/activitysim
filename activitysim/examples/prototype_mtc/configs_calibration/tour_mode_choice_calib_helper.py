import pandas as pd
import matplotlib.pyplot as plt
import os

SURVEY_DATA_FOLDER = r"C:\Users\david.hensle\OneDrive - Resource Systems Group, Inc\Documents\projects\activitysim\rsg_activitysim\activitysim\examples\example_estimation\data_test\survey_data"

def report_tour_mode_choice(context):
    model_tours = context["tours"]
    survey_tours = None
    survey_tours = pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_tours.csv"))

    model_summary = model_tours.tour_mode.value_counts(normalize=True).sort_index().fillna(0)
    survey_summary = survey_tours.groupby("tour_mode").tour_weight.sum()
    survey_summary = survey_summary / survey_tours.tour_weight.sum()

    summary_df = (
        pd.DataFrame({"model": model_summary, "survey": survey_summary})
        .reset_index()
        .rename(columns={"index": "tour_mode"})
    )

    # plot comparing model and survey distributions
    summary_df.plot(x="tour_mode", y=["model", "survey"], kind="bar")
    plt.title("Tour Mode Choice Distribution: Model vs Survey")
    plt.xlabel("Tour Mode")
    plt.ylabel("Proportion of Tours")
    plt.legend(title="Data Source")
    plt.savefig(os.path.join(context["component_output_dir"], "tour_mode_choice_comparison.png"))
    plt.close()
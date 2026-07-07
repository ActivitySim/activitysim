import pandas as pd
import os

def report_tour_mode_choice():
    model_tours = state.get_table("tours")
    survey_tours = None
    for data_dir in state.filesystem.data_dir:
        try:
            survey_tours = pd.read_csv(os.path.join(data_dir, component_settings.survey_file))
            break
        except FileNotFoundError:
            pass
    assert survey_tours is not None, f"No survey file {component_settings.survey_file} found in data dirs!"

    model_summary = model_tours.tour_mode.value_counts(normalize=True).sort_index().fillna(0)
    survey_summary = survey_tours.groupby("tour_mode").tour_weight.sum()
    survey_summary = survey_summary / survey_tours.tour_weight.sum()

    summary_df = (
        pd.DataFrame({"model": model_summary, "survey": survey_summary})
        .reset_index()
        .rename(columns={"index": "tour_mode"})
    )

    print(summary_df)

    # plot comparing model and survey distributions
    import matplotlib.pyplot as plt

    summary_df.plot(x="tour_mode", y=["model", "survey"], kind="bar")
    plt.title("Tour Mode Choice Distribution: Model vs Survey")
    plt.xlabel("Tour Mode")
    plt.ylabel("Proportion of Tours")
    plt.legend(title="Data Source")
    plt.savefig("tour_mode_choice_comparison.png")
    plt.close()
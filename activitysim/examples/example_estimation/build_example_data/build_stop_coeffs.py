#  python ~/work/activitysim/activitysim/examples/example_estimation/build_example_data/build_stop_coeffs.py

import numpy as np
import pandas as pd

FIRST_RUN = True

# work, school, univ, social, shopping, eatout, escort,atwork,othmaint,othdiscr
for what in [
    "work",
    "school",
    "univ",
    "social",
    "shopping",
    "eatout",
    "escort",
    "atwork",
    "othmaint",
    "othdiscr",
]:

    if FIRST_RUN:
        df = pd.read_csv(f"stop_frequency_{what}.csv", comment="#")
        df.to_csv(f"stop_frequency_backup_{what}.csv", index=False)
    else:
        df = pd.read_csv(f"stop_frequency_backup_{what}.csv", comment="#")

    del df["Expression"]

    df = df.set_index("Description").unstack()

    # drop empty coefficients
    df = df[~df.isnull()]

    # want index as columns
    df = df.reset_index().rename(columns={"level_0": "alt", 0: "value"})

    # drop duplicate coefficients on same spec row
    df = df[~df[["Description", "value"]].duplicated(keep="first")]

    dupes = df[["Description"]].duplicated(keep=False)
    df["coefficient_name"] = np.where(
        dupes, "coef_" + df.Description + "_" + df["alt"], "coef_" + df.Description
    )
    df["coefficient_name"] = df["coefficient_name"].str.lower()
    df["coefficient_name"] = df["coefficient_name"].str.replace(
        "[^a-zZ-Z0-9]+", "_", regex=True
    )
    del df["alt"]

    df.to_csv(f"stop_frequency_coefficients_{what}.csv", index=False)

    spec = pd.read_csv(f"stop_frequency_backup_{what}.csv", comment="#")

    alt_cols = spec.columns[2:].values

    for index, row in df.iterrows():
        m = {row["value"]: row["coefficient_name"]}
        alts = spec.loc[spec.Description == row["Description"], alt_cols].values[0]
        alts = [m.get(a, a) for a in alts]

        spec.loc[spec.Description == row["Description"], alt_cols] = [
            m.get(a, a) for a in alts
        ]

    spec.insert(loc=0, column="Label", value="util_" + spec.Description)

    spec["Label"] = spec["Label"].str.lower()
    spec["Label"] = spec["Label"].str.replace("[^a-zZ-Z0-9]+", "_", regex=True)

    df.to_csv(f"stop_frequency_coefficients_{what}.csv", index=False)
    spec.to_csv(f"stop_frequency_{what}.csv", index=False)

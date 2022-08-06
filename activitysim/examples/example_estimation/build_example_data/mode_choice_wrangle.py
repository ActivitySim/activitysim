import numpy as np
import pandas as pd

df = pd.read_csv(f"trip_mode_coefficients_p.csv", comment="#")

alts = list(df.drop(columns="Expression").columns.values.astype(str))
alts_str = "_".join(alts)

df = df.set_index("Expression").unstack()

df = df.reset_index().rename(columns={"level_0": "alts", 0: "value"})

df = df.groupby(["Expression", "value"]).agg(lambda col: "_".join(col)).reset_index()

df["coefficient_name"] = "coef_" + np.where(
    df.alts == alts_str, df["Expression"], df["Expression"] + "_" + df.alts
)

coefficients_df = df


df = pd.read_csv(f"trip_mode_coefficients_p.csv", comment="#")

for alt in alts:
    alt_df = pd.merge(
        df[["Expression", alt]].rename(columns={alt: "value"}),
        coefficients_df[["Expression", "value", "coefficient_name"]],
        left_on=["Expression", "value"],
        right_on=["Expression", "value"],
        how="left",
    )
    df[alt] = alt_df["coefficient_name"]

coefficients_df = coefficients_df[["coefficient_name", "value"]]
coefficients_df.to_csv(f"trip_mode_choice_coefficients.csv", index=False)


df.to_csv(f"trip_mode_choice_coefficients_template.csv", index=False)

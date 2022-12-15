# ActivitySim
# See full license in LICENSE.txt.
import logging
import os

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from sklearn.naive_bayes import CategoricalNB
from activitysim.core import inject, config, pipeline, util, input

logger = logging.getLogger(__name__)


def find_nearest_accessibility_zone(choosers, accessibility_df, method="skims"):
    """
    Matches choosers zone to the nearest accessibility zones.
    Can be achieved by querying the skims or by nearest neighbor of centroids
    """
    origin_col = "home_zone_id"

    def weighted_average(df, values, weights):
        return df[values].T.dot(df[weights]) / df[weights].sum()

    def nearest_skim(oz, zones):
        # need to pass equal # of origins and destinations to skim_dict
        orig_zones = np.full(shape=len(zones), fill_value=oz, dtype=int)
        return (
            oz,
            zones[np.argmin(skim_dict.lookup(orig_zones, zones, "DIST"))],
        )

    def nearest_node(oz, zones_df):
        _idx = util.nearest_node_index(_centroids.loc[oz].XY, zones_df.to_list())
        return oz, zones_df.index[_idx]

    unique_origin_zones = choosers[origin_col].unique()
    accessibility_zones = list(set(accessibility_df[origin_col]))

    # First find any choosers zones that are missing from accessibility zones
    matched_zones = list(set(unique_origin_zones).intersection(accessibility_zones))
    unmatched_zones = list(set(unique_origin_zones).difference(accessibility_zones))

    # Store choosers index to ensure consistency
    _idx = choosers.index

    if method == "centroids":
        # Extract and vectorize TAZ centroids
        centroids = inject.get_table("maz_centroids").to_frame()

        # TODO.NF This is a bit hacky, needs some work for variable zone names
        if "TAZ" in centroids.columns:
            # Find the TAZ centroid as weighted average of MAZ centroids
            _centroids = centroids[centroids.TAZ.isin(accessibility_zones)]
            _centroids = _centroids[["TAZ", "X", "Y", "Area"]].set_index("TAZ")
            _centroids = _centroids.groupby("TAZ").apply(
                weighted_average, ["X", "Y"], "Area"
            )
        else:
            _centroids = centroids

        # create XY tuple columns to find the nearest node with
        _centroids["XY"] = list(zip(_centroids.X, _centroids.Y))
        nearest = [nearest_node(Oz, _centroids.XY) for Oz in unmatched_zones]

    else:
        skim_dict = inject.get_injectable("skim_dict")
        nearest = [nearest_skim(Oz, accessibility_zones) for Oz in unmatched_zones]

    # Add the nearest zones to the matched zones
    matched = [(x, x) for x in matched_zones]
    matched += nearest

    # Create a DF and merge to choosers
    matched_df = pd.DataFrame(
        matched, columns=[origin_col, "nearest_accessibility_zone_id"]
    )
    matched_df = choosers.reset_index().merge(matched_df, on=origin_col)
    matched_df = matched_df.set_index(choosers.index.name)

    return matched_df.loc[_idx]


@inject.injectable()
def disaggregate_suffixes():
    return {"SUFFIX": None, "ROOTS": []}


@inject.table()
def maz_centroids():
    df = input.read_input_table("maz_centroids")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded maz_centroids %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table("maz_centroids", df)

    return df


@inject.table()
def proto_disaggregate_accessibility():

    # Read existing accessibilities, but is not required to enable model compatibility
    df = input.read_input_table("proto_disaggregate_accessibility", required=False)

    # If no df, return empty dataframe to skip this model
    if not df:
        return pd.DataFrame()

    # Ensure canonical index order
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded proto_disaggregate_accessibility %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table("proto_disaggregate_accessibility", df)

    return df


@inject.table()
def disaggregate_accessibility(persons, households, land_use, accessibility):
    """
    This step initializes pre-computed disaggregate accessibility and merges it onto the full synthetic population.
    Function adds merged all disaggregate accessibility tables to the pipeline but returns nothing.

    """

    # If disaggregate_accessibilities do not exist in the pipeline, it will try loading csv of that name
    proto_accessibility_df = pipeline.get_table("proto_disaggregate_accessibility")

    # If there is no table, skip. We do this first to skip as fast as possible
    if proto_accessibility_df.empty:
        return pd.DataFrame()

    # Get persons merged manually
    persons_merged_df = inject.merge_tables(
        persons.name, tables=[persons, households, land_use, accessibility]
    )

    # Extract model settings
    model_settings = config.read_model_settings("disaggregate_accessibility.yaml")
    merging_params = model_settings.get("MERGE_ON")
    nearest_method = model_settings.get("NEAREST_METHOD", "skims")
    accessibility_cols = [
        x for x in proto_accessibility_df.columns if "accessibility" in x
    ]

    # Parse the merging parameters
    assert merging_params is not None

    # Check if already assigned!
    if set(accessibility_cols).intersection(persons_merged_df.columns) == set(
        accessibility_cols
    ):
        return

    # Find the nearest zone (spatially) with accessibilities calculated
    # Note that from here on the 'home_zone_id' is the matched name
    if "nearest_accessibility_zone_id" not in persons_merged_df.columns:
        persons_merged_df = find_nearest_accessibility_zone(
            persons_merged_df, proto_accessibility_df, nearest_method
        )

    # Copy home_zone_id in proto-table to match the temporary 'nearest_zone_id'
    proto_accessibility_df[
        "nearest_accessibility_zone_id"
    ] = proto_accessibility_df.home_zone_id

    # Set up the useful columns
    exact_cols = merging_params.get("by", [])
    if "home_zone_id" in exact_cols:
        exact_cols.remove("home_zone_id")
    exact_cols.insert(0, "nearest_accessibility_zone_id")

    nearest_cols = merging_params.get("asof", [])
    merge_cols = exact_cols + nearest_cols

    assert len(nearest_cols) <= 1

    # Setup and left and right tables. If asof join is used, it must be sorted.
    # Drop duplicate accessibilities once filtered (may expect duplicates on households).
    # Simply dropping duplicate households won't work if sampling is not 100%
    # because it will get slightly different logsums for households in the same zone.
    # This is because different destination zones were selected. To resolve, get mean by cols.
    right_df = (
        proto_accessibility_df.groupby(merge_cols)[accessibility_cols]
        .mean()
        .sort_values(nearest_cols)
        .reset_index()
    )

    left_df = persons_merged_df[merge_cols].sort_values(nearest_cols)

    if merging_params.get("method") == "soft":
        # a 'soft' merge is possible by finding the nearest neighbor
        x_pop, x_proto = left_df[exact_cols], right_df[exact_cols]
        y = x_proto.index

        # Note: Naive Bayes is fast but discretely constrained. Some refinement may be necessary
        # Index error here means data ranges don't match (e.g., age or hh veh is 0,1,2,3 but proto only has 0,1,2)
        # The proto pop must at least be the same size or bigger.
        clf = CategoricalNB()
        clf.fit(x_proto, y)

        assert not any(
            x_proto.duplicated()
        )  # Ensure no duplicates, would mean we're missing a variable
        # assert all(clf.predict(x_proto) == y)  # Ensure it can at least predict on itself. If not there is a problem
        # Also can just relax this constraint and report the accuracy to the user
        accuracy = round(100 * sum(clf.predict(x_proto) == y) / len(y), 2)
        print(
            "Disaggregate accessibility merge training accuracy:"
            " {}% (<100% typically means insufficient merge-on features.)".format(
                accuracy
            )
        )

        # Predict the nearest person ID and pull the logsums
        matched_logsums_df = right_df.loc[clf.predict(x_pop)][
            accessibility_cols
        ].reset_index(drop=True)
        merge_df = pd.concat(
            [left_df.reset_index(drop=False), matched_logsums_df], axis=1
        ).set_index("person_id")

    else:
        # merge_asof is sensitive to dataframe data types. Ensure consistency by 'upgrading' any int32 to int64
        for col in merge_cols:
            if left_df[col].dtype is not right_df[col].dtype:
                assert ptypes.is_integer_dtype(
                    left_df[col]
                ) and ptypes.is_integer_dtype(right_df[col])
                datatype = np.max([left_df[col].dtype, right_df[col].dtype])
                left_df[col] = left_df[col].astype(datatype)
                right_df[col] = right_df[col].astype(datatype)

        if nearest_cols:
            merge_df = pd.merge_asof(
                left=left_df,
                right=right_df,
                by=exact_cols,
                on=nearest_cols,
                direction="nearest",
            )
        else:
            merge_df = pd.merge(
                left=left_df.reset_index(), right=right_df, on=exact_cols, how="left"
            )
        merge_df = merge_df.set_index("person_id")

    # Check that it was correctly left-joined
    assert all(persons_merged_df[merge_cols] == merge_df[merge_cols])
    assert any(merge_df[accessibility_cols].isnull())

    # Inject merged accessibilities so that it can be included in persons_merged function
    inject.add_table("disaggregate_accessibility", merge_df[accessibility_cols])

    return merge_df[accessibility_cols]


inject.broadcast(
    "disaggregate_accessibility", "persons", cast_index=True, onto_on="person_id"
)

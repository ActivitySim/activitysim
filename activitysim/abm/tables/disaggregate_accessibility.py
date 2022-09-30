# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from sklearn.naive_bayes import CategoricalNB

from activitysim.core import (inject,
                              config,
                              pipeline,
                              util,
                              input)

logger = logging.getLogger(__name__)


def find_nearest_accessibility_zone(choosers, accessibility_df, origin_col='home_zone_id', method='skims'):
    """
    Matches choosers zone to the nearest accessibility zones.
    Can be achieved by querying the skims or by nearest neighbor of centroids
    """

    def weighted_average(df, values, weights):
        return df[values].T.dot(df[weights]) / df[weights].sum()

    def nearest_skim(oz, zones):
        return oz, zones[np.argmin([skim_dict.lookup([oz], [az], 'DIST') for az in zones])]

    def nearest_node(oz, zones_df):
        _idx = util.nearest_node_index(_centroids.loc[oz].XY, zones_df.to_list())
        return oz, zones_df.index[_idx]

    unique_origin_zones = choosers[origin_col].unique()
    accessibility_zones = list(set(accessibility_df[origin_col]))

    # First find any choosers zones that are missing from accessibility zones
    matched_zones = list(set(unique_origin_zones).intersection(accessibility_zones))
    unmatched_zones = list(set(unique_origin_zones).difference(accessibility_zones))

    if method == 'centroids':
        # Extract and vectorize TAZ centroids
        centroids = inject.get_table('maz_centroids').to_frame()

        # TODO.NF This is a bit hacky, needs some work for variable zone names
        if 'TAZ' in centroids.columns:
            # Find the TAZ centroid as weighted average of MAZ centroids
            _centroids = centroids[centroids.TAZ.isin(accessibility_zones)]
            _centroids = _centroids[['TAZ', 'X', 'Y', 'Area']].set_index('TAZ')
            _centroids = _centroids.groupby('TAZ').apply(weighted_average, ['X', 'Y'], 'Area')
        else:
            _centroids = centroids

        # create XY tuple columns to find the nearest node with
        _centroids['XY'] = list(zip(_centroids.X, _centroids.Y))
        nearest = [nearest_node(Oz, _centroids.XY) for Oz in unmatched_zones]

    else:
        skim_dict = inject.get_injectable('skim_dict')
        nearest = [nearest_skim(Oz, accessibility_zones) for Oz in unmatched_zones]

    # Add the nearest zones to the matched zones
    matched = [(x, x) for x in matched_zones]
    matched += nearest

    # Create a DF and merge to choosers
    matched_df = pd.DataFrame(matched, columns=[origin_col, 'nearest_accessibility_zone_id'])
    matched_df = choosers.reset_index().merge(matched_df, on=origin_col)
    matched_df = matched_df.set_index(choosers.index.name)

    return matched_df


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
    df = input.read_input_table("proto_disaggregate_accessibility", required=False)

    if not df:
        return None

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    logger.info("loaded proto_disaggregate_accessibility %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table("maz_centroids", df)

    return df


@inject.step()
def initialize_disaggregate_accessibility():
    """
    This step initializes pre-computed disaggregate accessibility and merges it onto the full synthetic population.
    Function adds merged all disaggregate accessibility tables to the pipeline but returns nothing.

    """

    # If disaggregate_accessibilities do not exist in the pipeline, it will try loading csv of that name
    proto_accessibility_df = pipeline.get_table('proto_disaggregate_accessibility')
    persons_merged_df = pipeline.get_table('persons_merged')

    # If there is no table, skip. We do this first to skip as fast as possible
    if proto_accessibility_df is None:
        return

    # Extract model settings
    model_settings = config.read_model_settings('disaggregate_accessibility.yaml')
    merging_params = model_settings.get('MERGE_ON')
    origin_col = merging_params.get('origin_col', 'home_zone_id')
    logsums_cols = [x for x in proto_accessibility_df.columns if 'logsums' in x]

    # Parse the merging parameters
    assert merging_params is not None

    # Check if already assigned!
    if all([True for x in logsums_cols if x in persons_merged_df.columns]):
        return

    # Find the nearest zone (spatially) with accessibilities calculated
    if 'nearest_accessibility_zone_id' not in persons_merged_df.columns:
        persons_merged_df = find_nearest_accessibility_zone(persons_merged_df, proto_accessibility_df, origin_col)

    # Rename the home_zone_id in proto-table to match the temporary 'nearest_zone_id'
    proto_accessibility_df['nearest_accessibility_zone_id'] = proto_accessibility_df[origin_col]

    # Set up the useful columns
    exact_cols = ['nearest_accessibility_zone_id'] + merging_params.get('by', [])
    nearest_cols = merging_params.get('asof', [])
    merge_cols = exact_cols + nearest_cols

    # Perform an ASOF join, which first matches exact values, then finds the "nearest" values
    right_df = proto_accessibility_df[merge_cols + logsums_cols].sort_values(nearest_cols)
    left_df = persons_merged_df[merge_cols].sort_values(nearest_cols)

    if merging_params.get('method') == 'hard':
        # merge_asof is sensitive to dataframe data types. Ensure consistency by 'upgrading' any int32 to int64
        for col in merge_cols:
            if left_df[col].dtype is not right_df[col].dtype:
                assert ptypes.is_integer_dtype(left_df[col]) and ptypes.is_integer_dtype(right_df[col])
                datatype = np.max([left_df[col].dtype, right_df[col].dtype])
                left_df[col] = left_df[col].astype(datatype)
                right_df[col] = right_df[col].astype(datatype)

        if nearest_cols:
            merge_df = pd.merge_asof(left=left_df, right=right_df, by=exact_cols, on=nearest_cols, direction='nearest')
            merge_df = merge_df.set_index('person_id')
        else:
            merge_df = pd.merge(left=left_df.reset_index(), right=right_df, on=exact_cols, how='left')

    else:
        # A little worried that full model is too big, may need segment on zone
        # for zone in persons_merged_df[origin_col].unique():
        # x_pop = left_df[left_df.nearest_accessibility_zone_id==zone][merge_cols[1:]]
        # x_proto = right_df[right_df.nearest_accessibility_zone_id == zone][merge_cols[1:]]
        x_pop, x_proto = left_df[exact_cols], right_df[exact_cols]
        y = x_proto.index

        # Note: Naive Bayes is fast but discretely constrained. Some refinement may be necessary
        # Index error here means data ranges don't match (e.g., age or hh veh is 0,1,2,3 but proto only has 0,1,2)
        # The proto pop must at least be the same size or bigger.
        clf = CategoricalNB()
        clf.fit(x_proto, y)

        assert not any(x_proto.duplicated())   # Ensure no duplicates, would mean we're missing a variable
        # assert all(clf.predict(x_proto) == y)  # Ensure it can at least predict on itself. If not there is a problem
        # Also can just relax this constraint and report the accuracy to the user
        accuracy = round(100*sum(clf.predict(x_proto) == y) / len(y), 2)
        print('Disaggregate accessibility merge training accuracy:'
              ' {}% (<100% typically means insufficient merge-on features.)'.format(accuracy))

        # Predict the nearest person ID and pull the logsums
        matched_logsums_df = right_df.loc[clf.predict(x_pop)][logsums_cols].reset_index(drop=True)
        merge_df = pd.concat([left_df.reset_index(drop=False), matched_logsums_df], axis=1).set_index('person_id')

    # Check that it was correctly left-joined
    assert persons_merged_df[merge_cols].sort_index().equals(merge_df[merge_cols].sort_index())

    for col in logsums_cols:
        if merge_df[col].isnull().values.any():
            print("WARNING: NaNs found in disaggregate accessibilities '{}'".format(col))

    # Drop the temporary ID zone?
    # persons_merged_df.drop(columns='nearest_accessibility_zone_id')

    # Merge the accessibilities to the persons_merged table and update the pipeline
    pipeline.replace_table('persons_merged', persons_merged_df.join(merge_df[logsums_cols]))

    return

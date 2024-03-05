# ActivitySim
# See full license in LICENSE.txt.
import os
from ast import literal_eval
import pandas as pd
import numpy as np
import pandas.testing as pdt

from activitysim.abm.models.school_escorting import create_bundle_attributes


def optimized_create_bundle_attributes(bundles):
    # Initialize columns
    bundles['escortees'] = ''
    bundles['escortee_nums'] = ''
    bundles['num_escortees'] = ''
    bundles['school_destinations'] = ''
    bundles['school_starts'] = ''
    bundles['school_ends'] = ''
    bundles['school_tour_ids'] = ''

    bundles[['first_child', 'second_child', 'third_child']] = pd.DataFrame(bundles['child_order'].to_list(), index=bundles.index).astype(int)

    # index needs to be unique for filtering below
    original_idx = bundles.index
    bundles = bundles.reset_index(drop=True)

    def join_strings(row):
        return '_'.join([x for x in row if ~pd.isna(x)])


    # Loop through all possible combinations of child order
    # once the order is known, we can fill in the escorting information in the child order
    for first_child in [1, 2, 3]:
        for second_child in [1, 2, 3]:
            for third_child in [1, 2, 3]:
                if (first_child == second_child) | (first_child == third_child) | (second_child == third_child):
                    # children order is not unique
                    continue

                filtered_bundles = bundles[(bundles.first_child == first_child) & (bundles.second_child == second_child) & (bundles.third_child == third_child)]

                if len(filtered_bundles) == 0:
                    # no bundles for this combination of child order
                    continue 

                # escortees contain the child id of the escortees concatenated with '_'
                # special treatment here for -1.0, which is the value for no escortee
                bundles.loc[filtered_bundles.index, 'escortees'] = (
                    filtered_bundles[f"bundle_child{first_child}"].astype(str).str.replace('-1.0', '', regex=False) + '_' +
                    filtered_bundles[f"bundle_child{second_child}"].astype(str).str.replace('-1.0', '', regex=False)+ '_' +
                    filtered_bundles[f"bundle_child{third_child}"].astype(str).str.replace('-1.0', '', regex=False)
                ).str.replace(r'^_+', '', regex=True).str.replace(r'_+$', '', regex=True)
                
                # escortee_nums contain the child number of the escortees concatenated with '_'
                escortee_num1 = pd.Series(np.where(filtered_bundles[f"bundle_child{first_child}"] > 0, first_child, ''), index=filtered_bundles.index).astype(str)
                escortee_num2 = pd.Series(np.where(filtered_bundles[f"bundle_child{second_child}"] > 0, second_child, ''), index=filtered_bundles.index).astype(str)
                escortee_num3 = pd.Series(np.where(filtered_bundles[f"bundle_child{third_child}"] > 0, third_child, ''), index=filtered_bundles.index).astype(str)
                bundles.loc[filtered_bundles.index, 'escortee_nums'] = (
                    escortee_num1 + '_' + escortee_num2 + '_' + escortee_num3
                )

                # num_escortees contain the number of escortees
                bundles.loc[filtered_bundles.index, 'num_escortees'] = (filtered_bundles[
                    [f"bundle_child{first_child}", f"bundle_child{second_child}", f"bundle_child{third_child}"]
                ] > 0).sum(axis=1)

                # school_destinations contain the school destination of the escortees concatenated with '_'
                bundles.loc[filtered_bundles.index, 'school_destinations'] = (
                    filtered_bundles[f"school_destination_child{first_child}"].astype(str).fillna('') + '_' +
                    filtered_bundles[f"school_destination_child{second_child}"].astype(str).fillna('') + '_' +
                    filtered_bundles[f"school_destination_child{third_child}"].astype(str).fillna('')
                ).str.replace(r'^_+', '', regex=True).str.replace(r'_+$', '', regex=True)
                
                bundles.loc[filtered_bundles.index, 'school_starts'] = filtered_bundles[
                    [f"school_start_child{first_child}", f"school_start_child{second_child}", f"school_start_child{third_child}"]
                ].astype(str).agg(lambda row: join_strings(row), axis=1)

                bundles.loc[filtered_bundles.index, 'school_ends'] = filtered_bundles[
                    [f"school_end_child{first_child}", f"school_end_child{second_child}", f"school_end_child{third_child}"]
                ].astype(str).agg(lambda row: join_strings(row), axis=1)

                bundles.loc[filtered_bundles.index, 'school_tour_ids'] = filtered_bundles[
                    [f"school_end_child{first_child}", f"school_end_child{second_child}", f"school_end_child{third_child}"]
                ].astype(str).agg(lambda row: join_strings(row), axis=1)

    bundles.drop(columns=['first_child', 'second_child', 'third_child'], inplace=True)

    return bundles.set_index(original_idx)


def test_create_bundle_attributes():

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    dtype_dict = {'escortees': 'str', 'escortee_nums': 'str', 'school_destinations': 'str', 'school_starts': 'str', 'school_ends': 'str', 'school_tour_ids': 'str'}

    inbound_input = pd.read_csv(os.path.join(data_dir, "create_bundle_attributes__input_inbound.csv"), index_col=0)
    inbound_output = pd.read_csv(os.path.join(data_dir, "create_bundle_attributes__output_inbound.csv"), index_col=0, dtype=dtype_dict)

    outbound_input = pd.read_csv(os.path.join(data_dir, "create_bundle_attributes__input_outbound_cond.csv"), index_col=0)
    outbound_output = pd.read_csv(os.path.join(data_dir, "create_bundle_attributes__output_outbound_cond.csv"), index_col=0, dtype=dtype_dict)

    # need to convert columns from string back to list
    list_columns = ['outbound_order', 'inbound_order', 'child_order']
    for col in list_columns:
        inbound_input[col] = inbound_input[col].apply(lambda x: x.strip('[]').split(' '))
        outbound_input[col] = outbound_input[col].apply(lambda x: x.strip('[]').split(' '))
        inbound_output[col] = inbound_output[col].apply(lambda x: x.strip('[]').split(' '))
        outbound_output[col] = outbound_output[col].apply(lambda x: x.strip('[]').split(' '))

    inbound_result = inbound_input.apply(lambda row: create_bundle_attributes(row), axis=1)
    # inbound_result = optimized_create_bundle_attributes(inbound_input)
    pdt.assert_frame_equal(inbound_result, inbound_output, check_dtype=False)
    
    outbound_result = outbound_input.apply(lambda row: create_bundle_attributes(row), axis=1)
    # outbound_result = optimized_create_bundle_attributes(outbound_input)
    pdt.assert_frame_equal(outbound_result, outbound_output, check_dtype=False)


if __name__ == '__main__':
    test_create_bundle_attributes()

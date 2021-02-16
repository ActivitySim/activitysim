# # ActivitySim
# # See full license in LICENSE.txt.
# import os
# import logging
# import pkg_resources
#
# import pandas as pd
# import pandas.testing as pdt
#
# from activitysim.core import pipeline
# from activitysim.core import inject
# from activitysim.core import config
#
# from activitysim.cli.run import cleanup_output_files
#
# from activitysim.abm.test.test_pipeline import setup_dirs
# from activitysim.abm.test.test_pipeline import inject_settings
#
# from activitysim import abm  # register injectables
#
#
# def example_path(dirname):
#     resource = os.path.join('examples', 'example_psrc', dirname)
#     return pkg_resources.resource_filename('activitysim', resource)
#
# def test_path(dirname):
#     return os.path.join(os.path.dirname(__file__), dirname)
#
#
# def get_regression_df(file_name):
#     file_path = os.path.join(os.path.dirname(__file__), file_name)
#     df = pd.read_csv(file_path)
#     df = df.set_index(df.columns[0])
#     return df


# def test_full_run():
#
#     configs_dir = [test_path('configs'), example_path('configs')]
#     data_dir = example_path('data')
#
#     setup_dirs(configs_dir, data_dir)
#
#     inject.add_injectable('output_dir', test_path('output'))
#
#     cleanup_output_files()
#
#     pipeline.run(models=config.setting('models'))
#
#     trips_df = pipeline.get_table('trips')
#
#     regress_trips_df = get_regression_df('regress_trips.csv')
#
#     # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
#     # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
#     # compare_cols = []
#     pdt.assert_frame_equal(trips_df, regress_trips_df)
#
#     pipeline.close_pipeline()

import os
import sys
import argparse
import pkg_resources

import pandas as pd
import pandas.testing as pdt

from activitysim.core import pipeline

from activitysim.cli.run import add_run_args, run


def example_path(dirname):
    resource = os.path.join('examples', 'example_psrc', dirname)
    return pkg_resources.resource_filename('activitysim', resource)


def test_path(dirname):
    return os.path.join(os.path.dirname(__file__), dirname)


def get_regression_df(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    df = pd.read_csv(file_path)
    df = df.set_index(df.columns[0])
    return df


def regress_psrc():

    pipeline.open_pipeline('_')
    trips_df = pipeline.get_table('trips')

    regress_trips_df = get_regression_df('regress_trips.csv')

    # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
    # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
    # compare_cols = []
    pdt.assert_frame_equal(trips_df, regress_trips_df)

    pipeline.close_pipeline()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_run_args(parser)

    args = parser.parse_args([
        '-c', test_path('configs'),
        '-c', example_path('configs'),
        '-d', example_path('data'),
        '-o', test_path('output')
    ])

    run(args)

    regress_psrc()

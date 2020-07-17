# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import numpy as np
import pandas as pd

from activitysim.core import assign
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import simulate

from activitysim.abm.models.util import expressions


logger = logging.getLogger('activitysim')


def compute_logsums(model_settings,
                    choosers, parameters, spec,
                    model_constants, network_los,
                    chunk_size, trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'compute_logsums')

    logger.debug("Running compute_logsums with %d choosers" % choosers.shape[0])

    locals_dict = {
        'np': np,
        'los': network_los
    }
    locals_dict.update(model_constants)
    locals_dict.update(parameters)

    # - run preprocessor to annotate choosers
    preprocessor_settings = model_settings.get('PREPROCESSOR')
    if preprocessor_settings:

        # don't want to alter caller's dataframe
        choosers = choosers.copy()

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    logsums = simulate.simple_simulate_logsums(
        choosers,
        spec,
        nest_spec=None,
        skims=None,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return logsums


def compute_tap_tap_utilities(network_los, model_settings, chunk_size, trace_label):

    tap_tap_settings = network_los.setting('tap_to_tap.logsums')

    model_constants = config.get_model_constants(model_settings)

    demographic_segments = network_los.setting('DEMOGRAPHIC_SEGMENTS')

    spec = simulate.read_model_spec(file_name=tap_tap_settings['SPEC'])

    taps = network_los.tap_df['TAP'].values
    tap_tap_df = pd.DataFrame({
        'btap': np.repeat(taps, len(taps)),
        'atap': np.tile(taps, len(taps)),
    })

    # FIXME drop diagonal?
    tap_tap_df = tap_tap_df[tap_tap_df.btap != tap_tap_df.atap]

    for seg, demographic_specific_parameters in demographic_segments.items():

        print(f"seg {seg} demographic_specific_parameters: {demographic_specific_parameters}")
        bug

        logsums = compute_logsums(
            tap_tap_settings,
            choosers=tap_tap_df,
            parameters=demographic_specific_parameters,
            spec=spec,
            model_constants=model_constants,
            network_los=network_los,
            chunk_size=chunk_size,
            trace_label=trace_label)

        tap_tap_df[seg] = logsums

    result_df = tap_tap_df.set_index(['btap', 'atap'], drop=True).drop(columns='tod')

    return result_df


def compute_maz_tap_utilities(network_los, model_settings, chunk_size, trace_label):

    model_constants = config.get_model_constants(model_settings)
    modes = network_los.setting('maz_tap_modes')

    demographic_segments = model_settings['DEMOGRAPHIC_SEGMENTS']

    maz_to_tap_utilities = {}

    for mode, mode_settings in modes.items():

        mode_specific_spec = simulate.read_model_spec(file_name=mode_settings['SPEC'])

        choosers = network_los.maz_to_tap_dfs[mode]
        if 'CHOOSER_COLUMNS' in mode_settings:
            choosers = choosers[mode_settings.get('CHOOSER_COLUMNS')]

        utilities_df = pd.DataFrame(index=choosers.index)

        for demographic_segment, demographic_parameters in demographic_segments.items():

            utilities_df[demographic_segment] = compute_logsums(
                mode_settings,
                choosers,
                parameters=demographic_parameters,
                spec=mode_specific_spec,
                model_constants=model_constants,
                network_los=network_los,
                chunk_size=chunk_size,
                trace_label=trace_label)

        maz_to_tap_utilities[mode] = utilities_df

    DUMP = True
    if DUMP:
        for mode in maz_to_tap_utilities:
            print(f"\nmode: {mode}")
            print(f"\nmode: {maz_to_tap_utilities[mode]}")
        bug

    return maz_to_tap_utilities


def generate_paths(orig_maz_tap_utilities, tap_tap_utilities, dest_maz_tap_utilities):
    """

    Parameters
    ----------
    orig_maz_tap_utilities:  Series with index: MAZ, TAP
    tap_tap_utilities:       Series with index: btap, atap
    dest_maz_tap_utilities:  Series with index: MAZ, TAP

    Returns
    -------

    """

    path_df = \
        pd.merge(
            orig_maz_tap_utilities.to_frame('orig_utility').
            reset_index(drop=False).
            rename(columns={'MAZ': 'omaz', 'TAP': 'btap'}),
            tap_tap_utilities.to_frame('tap_utility').reset_index(drop=False),
            on='btap', how='inner')

    path_df = \
        pd.merge(
            path_df,
            dest_maz_tap_utilities.to_frame('dest_utility').
            reset_index(drop=False).
            rename(columns={'MAZ': 'dmaz', 'TAP': 'atap'}),
            on='atap', how='inner')

    # drop paths with same omaz and dmaz
    path_df = path_df[path_df.omaz != path_df.dmaz]

    path_df['utility'] = path_df.orig_utility + path_df.tap_utility + path_df.dest_utility

    # columns = ['omaz', 'orig_utility', 'btap', 'tap_utility', 'atap', 'dmaz', 'dest_utility']
    columns = ['omaz', 'btap', 'atap', 'dmaz', 'utility']
    return path_df[columns]


@inject.step()
def transit_virtual_path_builder(network_los, path_builder, chunk_size, output_dir):

    trace_label = 'tvpb'

    model_settings = config.read_model_settings('tvpb.yaml')

    path_builder.build_tap_tap_utilities(chunk_size, trace_label)

    modes = network_los.maz_to_tap_modes
    demographic_segments = network_los.setting('DEMOGRAPHIC_SEGMENTS').keys()


    # DataFrame indexed by atap, btap with utilities for each demographic_segment (vot varies by segment)
    tap_tap_utilities_df = \
        compute_tap_tap_utilities(network_los, model_settings, chunk_size, trace_label)

    # dict keyed by mode of DataFrame with index ['MAZ', 'TAP'] and one column of utilities per demographic segment
    maz_tap_utilities = \
        compute_maz_tap_utilities(network_los, model_settings, chunk_size, trace_label)

    logger.debug(f"{trace_label} tap_tap_utilities_df shape: {tap_tap_utilities_df.shape}")

    mazs = network_los.maz_df['MAZ'].values
    choosers_df = pd.DataFrame({
        'orig_maz': np.repeat(mazs, len(mazs)),
        'dest_maz': np.tile(mazs, len(mazs)),
    })


    for orig_mode in modes:
        for dest_mode in modes:
            for demographic_segment in demographic_segments:

                print(f"demographic_segment: {demographic_segment} orig_mode: {orig_mode} dest_mode: {dest_mode}")

                orig_maz_tap_df = maz_tap_utilities[orig_mode]
                dest_maz_tap_df = maz_tap_utilities[dest_mode]

                # path_df has columns omaz, orig_utility, btap, tap_utility, atap , dmaz, dest_utility
                path_df = generate_paths(
                    orig_maz_tap_utilities=orig_maz_tap_df[demographic_segment],
                    tap_tap_utilities=tap_tap_utilities_df[demographic_segment],
                    dest_maz_tap_utilities=dest_maz_tap_df[demographic_segment],
                )

                num_best_paths_to_keep = 1
                path_df = path_df.\
                    sort_values(by='utility', ascending=False).\
                    groupby(['omaz', 'dmaz']).\
                    head(num_best_paths_to_keep)

                if (len(modes) > 1):
                    # don't need this if only one mode is being used...
                    path_df['orig_mode'] = orig_mode
                    path_df['dest_mode'] = dest_mode
                path_df['demographic_segment'] = demographic_segment

                logger.debug(f"{trace_label} path_df shape: {path_df.shape} size: {path_df.size} "
                             f"for {orig_mode} {dest_mode} {demographic_segment}")
                print(path_df.head())

                path_df.to_csv(os.path.join(output_dir, f'best_path_{orig_mode}_{dest_mode}_{demographic_segment}.csv'),
                               index=False)

                # FIXME then what?

    bug

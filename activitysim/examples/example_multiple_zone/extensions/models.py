# ActivitySim
# See full license in LICENSE.txt.

import logging

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

    sub_model_name = 'TAP_TAP'
    tap_tap_settings = model_settings[sub_model_name]
    model_constants = config.get_model_constants(model_settings)

    demographic_segments = model_settings.get('DEMOGRAPHIC_SEGMENTS')

    spec = simulate.read_model_spec(file_name=tap_tap_settings['SPEC'])
    time_of_day = tap_tap_settings['TIME_OF_DAY']

    taps = network_los.tap_df['TAP'].values
    tap_tap_df = pd.DataFrame({
        'btap': np.repeat(taps, len(taps)),
        'atap': np.tile(taps, len(taps)),
        'tod': time_of_day
    })

    # FIXME drop diagonal?
    tap_tap_df = tap_tap_df[tap_tap_df.btap != tap_tap_df.atap]

    for seg, demographic_specific_parameters in demographic_segments.items():

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


class MazMazUtilityStore(object):

    def __init__(self, modes, demographic_segments):
        self.modes = {}
        for mode in modes:
            self.modes[mode] = {}
            for seg in demographic_segments:
                self.modes[mode][seg] = None

    def set(self, mode, seg, df):
        assert self.modes[mode][seg] is None
        self.modes[mode][seg] = df

    def get(self, mode, seg):
        assert self.modes[mode][seg] is not None
        return self.modes[mode][seg].dropna()


def compute_maz_tap_utilities(network_los, model_settings, chunk_size, trace_label):

    model_constants = config.get_model_constants(model_settings)
    modes = model_settings['MAZ_TAP_MODES']
    demographic_segments = model_settings['DEMOGRAPHIC_SEGMENTS']

    m2m_utilities = MazMazUtilityStore(modes, demographic_segments)

    for mode, mode_settings in modes.items():

        mode_specific_spec = simulate.read_model_spec(file_name=mode_settings['SPEC'])

        choosers = network_los.maz_to_tap_df
        if 'CHOOSER_COLUMNS' in mode_settings:
            choosers = choosers[mode_settings.get('CHOOSER_COLUMNS')]

        for demographic_segment, demographic_parameters in demographic_segments.items():

            utilities = compute_logsums(
                mode_settings,
                choosers,
                parameters=demographic_parameters,
                spec=mode_specific_spec,
                model_constants=model_constants,
                network_los=network_los,
                chunk_size=chunk_size,
                trace_label=trace_label)

            m2m_utilities.set(mode, demographic_segment, utilities)

    return m2m_utilities


def generate_paths(orig_maz_tap_df, tap_tap_df, dest_maz_tap_df):

    path_df = \
        pd.merge(
            orig_maz_tap_df.to_frame('orig_utility').
            reset_index(drop=False).
            rename(columns={'MAZ': 'omaz', 'TAP': 'btap'}),
            tap_tap_df.to_frame('tap_utility').reset_index(drop=False),
            on='btap', how='inner')

    path_df = \
        pd.merge(
            path_df,
            dest_maz_tap_df.to_frame('dest_utility').
            reset_index(drop=False).
            rename(columns={'MAZ': 'dmaz', 'TAP': 'atap'}),
            on='atap', how='inner')

    # drop paths with same omaz and dmaz
    path_df = path_df[path_df.omaz != path_df.dmaz]

    columns = ['omaz', 'orig_utility', 'btap', 'tap_utility', 'atap', 'dmaz', 'dest_utility']
    return path_df[columns]


@inject.step()
def transit_virtual_path_builder(network_los, chunk_size):

    trace_label = 'tvpb'

    model_settings = config.read_model_settings('tvpb.yaml')
    modes = model_settings['MAZ_TAP_MODES'].keys()
    demographic_segments = model_settings['DEMOGRAPHIC_SEGMENTS'].keys()

    tap_tap_utilities_df = \
        compute_tap_tap_utilities(network_los, model_settings, chunk_size, trace_label)

    maz_tap_utilities = \
        compute_maz_tap_utilities(network_los, model_settings, chunk_size, trace_label)

    for demographic_segment in demographic_segments:
        for orig_mode in modes:
            for dest_mode in modes:

                print(f"demographic_segment: {demographic_segment} orig_mode: {orig_mode} dest_mode: {dest_mode}")

                path_df = generate_paths(
                    orig_maz_tap_df=maz_tap_utilities.get(orig_mode, demographic_segment),
                    tap_tap_df=tap_tap_utilities_df[demographic_segment],
                    dest_maz_tap_df=maz_tap_utilities.get(dest_mode, demographic_segment)
                )

                print(path_df)
                bug

    bug

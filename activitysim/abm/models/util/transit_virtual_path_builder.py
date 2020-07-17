# ActivitySim
# See full license in LICENSE.txt.
from builtins import range
from builtins import int

import sys
import os
import logging
import multiprocessing

from collections import OrderedDict
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
import openmatrix as omx

from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import simulate
from activitysim.core import los

from activitysim.core.util import reindex

from activitysim.abm.models.util import expressions


logger = logging.getLogger(__name__)


# def compute_logsums(model_settings,
#                     choosers, spec,
#                     model_constants, network_los,
#                     chunk_size, trace_label):
#
#     trace_label = tracing.extend_trace_label(trace_label, 'compute_logsums')
#
#     logger.debug("Running compute_logsums with %d choosers" % choosers.shape[0])
#
#     locals_dict = {
#         'np': np,
#         'los': network_los
#     }
#     locals_dict.update(model_constants)
#
#     # - run preprocessor to annotate choosers
#     preprocessor_settings = model_settings.get('PREPROCESSOR')
#     if preprocessor_settings:
#
#         # don't want to alter caller's dataframe
#         choosers = choosers.copy()
#
#         expressions.assign_columns(
#             df=choosers,
#             model_settings=preprocessor_settings,
#             locals_dict=locals_dict,
#             trace_label=trace_label)
#
#     logsums = simulate.simple_simulate_logsums(
#         choosers,
#         spec,
#         nest_spec=None,
#         skims=None,
#         locals_d=locals_dict,
#         chunk_size=chunk_size,
#         trace_label=trace_label)
#
#     return logsums

def compute_utilities(
        model_settings,
        choosers, spec,
        model_constants, network_los,
        chunk_size, trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'compute_utilities')

    logger.debug("Running compute_utilities with %d choosers" % choosers.shape[0])

    locals_dict = {
        'np': np,
        'los': network_los
    }
    locals_dict.update(model_constants)

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

    utilities = simulate.eval_utilities(
        spec,
        choosers,
        locals_d=locals_dict,
        trace_label=trace_label)

    return utilities


class TransitVirtualPathBuilder(object):

    def __init__(self, network_los):

        self.network_los = network_los
        self.chunk_size = inject.get_injectable('chunk_size')

        assert network_los.zone_system == los.THREE_ZONE, \
            f"TransitVirtualPathBuilder: network_los zone_system not THREE_ZONE"

    def compute_maz_tap_utilities(self, maz_od_df, chooser_attributes, path_type, leg, trace_label):

        trace_label = tracing.extend_trace_label(trace_label, f'compute_maz_tap_utilities.{leg}')

        path_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{path_type}')
        mode = path_settings[leg]

        if leg == 'access':
            maz_col = 'omaz'
            tap_col = 'btap'
        else:
            maz_col = 'dmaz'
            tap_col = 'atap'

        # compute access maz_to_tap utilities
        # deduped access_df - one row per chooser for each boarding tap (btap) accessable from omaz
        access_df = self.network_los.maz_to_tap_dfs[mode]
        chooser_columns = self.network_los.setting(f'maz_to_tap.utilities.{mode}.CHOOSER_COLUMNS')
        access_df = access_df[chooser_columns].\
            reset_index(drop=False).\
            rename(columns={'MAZ': maz_col, 'TAP': tap_col})
        access_df = pd.merge(
            maz_od_df[['idx', maz_col]].drop_duplicates(),
            access_df,
            on=maz_col, how='inner')
        # compute maz_to_tap utilities for demographic segment
        attribute_columns = list(chooser_attributes.columns)
        for c in attribute_columns:
            access_df[c] = reindex(chooser_attributes[c], access_df['idx'])

        model_constants = self.network_los.setting(f'TVPB_CONSTANTS')
        mode_settings = self.network_los.setting(f'maz_to_tap.utilities.{mode}')
        mode_specific_spec = simulate.read_model_spec(file_name=mode_settings['SPEC'])
        access_df[leg] = compute_utilities(
            mode_settings,
            access_df,
            spec=mode_specific_spec,
            model_constants=model_constants,
            network_los=self.network_los,
            chunk_size=self.chunk_size,
            trace_label=trace_label)

        # drop utility computation columns ('tod', 'demographic_segment' and maz_to_tap_df time/distance columns)
        access_df.drop(columns=attribute_columns + chooser_columns, inplace=True)

        return access_df

    def compute_tap_tap_utilities(self, access_df, egress_df, chooser_attributes, trace_label):

        trace_label = tracing.extend_trace_label(trace_label, 'compute_tap_tap_utilities')

        # compute tap_to_tap utilities
        # deduped transit_df has one row per chooser for each boarding (btap) and alighting (atap) pair
        transit_df = pd.merge(
            access_df[['idx', 'btap']],
            egress_df[['idx', 'atap']],
            on='idx').drop_duplicates()

        for c in list(chooser_attributes.columns):
            transit_df[c] = reindex(chooser_attributes[c], transit_df['idx'])

        model_constants = self.network_los.setting(f'TVPB_CONSTANTS')
        tap_tap_settings = self.network_los.setting('tap_to_tap.utilities')
        spec = simulate.read_model_spec(file_name=tap_tap_settings['SPEC'])

        transit_utilities = compute_utilities(
            tap_tap_settings,
            choosers=transit_df,
            spec=spec,
            model_constants=model_constants,
            network_los=self.network_los,
            chunk_size=self.chunk_size,
            trace_label=trace_label)

        transit_sets = self.network_los.setting(f'tap_to_tap.sets')
        assert set(transit_sets) == set(transit_utilities.columns)

        transit_df = pd.concat([transit_df[['idx', 'btap', 'atap']], transit_utilities], axis=1)

        return transit_df

    def best_paths(self, path_type, maz_od_df, access_df, egress_df, transit_df):

        maz_od_df['seq'] = maz_od_df.index
        # maz_od_df has one row per chooser
        # inner join to add rows for each access, egress, and transit segment combination
        path_df = maz_od_df. \
            merge(access_df, on=['idx', 'omaz'], how='inner'). \
            merge(egress_df, on=['idx', 'dmaz'], how='inner'). \
            merge(transit_df, on=['idx', 'atap', 'btap'], how='inner')

        transit_sets = self.network_los.setting(f'tap_to_tap.sets')
        for c in transit_sets:
            path_df[c] = path_df[c] + path_df['access'] + path_df['egress']
        path_df.drop(columns=['access', 'egress'], inplace=True)

        path_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{path_type}')
        max_best_paths_across_tap_sets = path_settings.get('max_best_paths_across_tap_sets', 3)
        max_paths_for_logsum_per_tap_set = path_settings.get('max_paths_for_logsum_per_tap_set', 1)

        # best paths by tap set
        keep = pd.Series(False, index=path_df.index)
        for c in transit_sets:
            keep |= path_df.index.isin(
                path_df[['seq', c]].sort_values(by=c, ascending=False).
                groupby(['seq']).head(max_paths_for_logsum_per_tap_set).index
            )
        path_df = path_df[keep]

        # best paths overall by seq
        path_df['utility'] = path_df[transit_sets].max(axis=1)
        path_df.drop(columns=transit_sets, inplace=True)
        path_df = path_df.sort_values(by=['seq', 'utility'], ascending=[True, False])
        keep = path_df.index.isin(
            path_df.
            groupby(['seq']).head(max_best_paths_across_tap_sets).index
        )
        path_df = path_df[keep]

        return path_df
    
    def get_tvpb_logsum(self, orig, dest, tod, demographic_segment, path_type):

        trace_label = ('get_tvpb_logsum')

        # maz od pairs requested
        #orig_index_name = orig.index.name
        maz_od_df = pd.DataFrame({
            'idx': orig.index.values,
            'omaz': orig.values,
            'dmaz': dest.values,
            'seq': range(len(orig))
        })
        #maz_od_df['seq'] = maz_od_df.index

        # tod and demographic_segment should be the same for all chooser rows (unique orig index values)
        # if are called from interaction_simulate, there will be multiple alternative rows per chooser
        # knowing this allows us to eliminate redundant computations (e.g. utilities of maz_tap pairs)
        duplicated = orig.index.duplicated(keep='first')
        chooser_attributes = \
            pd.DataFrame({
                'tod': tod.loc[~duplicated],
                'demographic_segment': demographic_segment.loc[~duplicated]},
                index=orig.index[~duplicated])

        # compute access deduped maz_to_tap utilities
        access_df = self.compute_maz_tap_utilities(
            maz_od_df,
            chooser_attributes,
            path_type=path_type, leg='access',
            trace_label=trace_label)

        # compute egress deduped maz_to_tap utilities
        egress_df = self.compute_maz_tap_utilities(
            maz_od_df,
            chooser_attributes,
            path_type=path_type, leg='egress',
            trace_label=trace_label)

        transit_df = self.compute_tap_tap_utilities(
            access_df, egress_df,
            chooser_attributes,
            trace_label=trace_label)

        path_df = self.best_paths(path_type, maz_od_df, access_df, egress_df, transit_df)
        
        # logsums
        # aone row per seq with utilities in columns
        path_df['path_num'] = path_df.groupby('seq').cumcount() + 1

        utilities_df = path_df[['seq', 'path_num', 'utility']].set_index(['seq', 'path_num']).unstack()

        # paths with fewer than the max number of paths will have Nan values for missing data
        # but there shouold always be at least one path/utility per seq
        assert not utilities_df.isnull().all(axis=1).any()
        # logsums = np.where(utilities_df.isnull().all(axis=1), -999, np.log(np.nansum(np.exp(x.values), axis=1)))

        logsums = pd.Series(np.log(np.nansum(np.exp(utilities_df.values), axis=1)), index=utilities_df.index)

        #utilities_df['logsums'] = logsums
        #print(f"utilities_df\n{utilities_df}")
        #assert (utilities_df.index == maz_od_df.index).all()
        #bug

        #utilities_df['logsums']= logsums
        #print(f"utilities_df\n{utilities_df}")
        print(logsums)

        return logsums


    def wrap(self, orig_key, dest_key, tod_key, segment_key):

        return TransitVirtualPathWrapper(self, orig_key, dest_key, tod_key, segment_key)


class TransitVirtualPathWrapper(object):

    def __init__(self, transit_virtual_path_builder, orig_key, dest_key, tod_key, segment_key):

        self.tvpb = transit_virtual_path_builder
        assert hasattr(transit_virtual_path_builder, 'get_tvpb_logsum')

        self.orig_key = orig_key
        self.dest_key = dest_key
        self.tod_key = tod_key
        self.segment_key = segment_key
        self.segment_key = segment_key
        self.df = None

        assert isinstance(dest_key, str)
        assert isinstance(dest_key, str)
        assert isinstance(tod_key, str)
        assert isinstance(segment_key, str)

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitiate chaining)
        """

        self.df = df
        return self

    def __getitem__(self, path_type):
        """
        Get an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """

        assert(self.orig_key in self.df), \
            f"TransitVirtualPathWrapper: orig_key '{self.orig_key}' not in df"
        assert(self.dest_key in self.df), \
            f"TransitVirtualPathWrapper: dest_key '{self.dest_key}' not in df"
        assert(self.tod_key in self.df), \
            f"TransitVirtualPathWrapper: tod_key '{self.tod_key}' not in df"
        assert(self.segment_key in self.df), \
            f"TransitVirtualPathWrapper: segment_key '{self.segment_key}' not in df"

        assert self.df is not None, "Call set_df first"
        orig = self.df[self.orig_key].astype('int')
        dest = self.df[self.dest_key].astype('int')
        tod = self.df[self.tod_key]
        segment = self.df[self.segment_key]

        skim_values = \
            self.tvpb.get_tvpb_logsum(
                orig,
                dest,
                tod,
                segment,
                path_type)

        return pd.Series(skim_values, self.df.index)

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
from activitysim.core import config

from activitysim.core.util import reindex

from activitysim.abm.models.util import expressions
from activitysim.core import assign

logger = logging.getLogger(__name__)


def compute_utilities(
        model_settings,
        choosers, spec,
        locals_dict, network_los,
        chunk_size, trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'compute_utilities')

    logger.debug("Running compute_utilities with %d choosers" % choosers.shape[0])

    locals_dict = locals_dict.copy()  # don't clobber argument
    locals_dict.update({
        'np': np,
        'los': network_los
    })

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

        # FIXME - need to recompute headroom?
        self.chunk_size = inject.get_injectable('chunk_size')

        assert network_los.zone_system == los.THREE_ZONE, \
            f"TransitVirtualPathBuilder: network_los zone_system not THREE_ZONE"

    def compute_maz_tap_utilities(self, recipe, maz_od_df, chooser_attributes, leg, mode, trace_label):

        trace_label = tracing.extend_trace_label(trace_label, f'compute_maz_tap_utilities.{leg}')

        maz_tap_settings = \
            self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.maz_tap_expressions.{mode}')
        chooser_columns = maz_tap_settings['CHOOSER_COLUMNS']
        attribute_columns = list(chooser_attributes.columns) if chooser_attributes is not None else []
        model_constants = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.CONSTANTS')
        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')

        if leg == 'access':
            maz_col = 'omaz'
            tap_col = 'btap'
        else:
            maz_col = 'dmaz'
            tap_col = 'atap'

        # maz_to_tap access/egress
        # deduped access_df - one row per chooser for each boarding tap (btap) accessible from omaz
        access_df = self.network_los.maz_to_tap_dfs[mode]

        access_df = access_df[chooser_columns]. \
            reset_index(drop=False). \
            rename(columns={'MAZ': maz_col, 'TAP': tap_col})
        access_df = pd.merge(
            maz_od_df[['idx', maz_col]].drop_duplicates(),
            access_df,
            on=maz_col, how='inner')
        # add any supplemental chooser attributes (e.g. demographic_segment, tod)
        for c in attribute_columns:
            access_df[c] = reindex(chooser_attributes[c], access_df['idx'])

        if units == 'utility':

            mode_specific_spec = simulate.read_model_spec(file_name=maz_tap_settings['SPEC'])

            access_df[leg] = compute_utilities(
                maz_tap_settings,
                access_df,
                spec=mode_specific_spec,
                locals_dict=model_constants,
                network_los=self.network_los,
                chunk_size=self.chunk_size,
                trace_label=trace_label)
        else:

            assignment_spec = assign.read_assignment_spec(file_name=config.config_file_path(maz_tap_settings['SPEC']))

            results, _, _ = assign.assign_variables(assignment_spec, access_df, model_constants)
            assert len(results.columns == 1)
            access_df[leg] = results

        # drop utility computation columns ('tod', 'demographic_segment' and maz_to_tap_df time/distance columns)
        access_df.drop(columns=attribute_columns + chooser_columns, inplace=True)

        return access_df

    def compute_tap_tap_utilities(self, recipe, access_df, egress_df, chooser_attributes, path_info, trace_label):

        trace_label = tracing.extend_trace_label(trace_label, 'compute_tap_tap_utilities')

        model_constants = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.CONSTANTS')
        tap_tap_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.tap_tap_expressions')
        attribute_columns = list(chooser_attributes.columns) if chooser_attributes is not None else []
        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')

        # FIXME some expressions may want to know access mode -
        locals_dict = path_info.copy()
        locals_dict.update(model_constants)

        # compute tap_to_tap utilities
        # deduped transit_df has one row per chooser for each boarding (btap) and alighting (atap) pair
        transit_df = pd.merge(
            access_df[['idx', 'btap']],
            egress_df[['idx', 'atap']],
            on='idx').drop_duplicates()

        for c in list(attribute_columns):
            transit_df[c] = reindex(chooser_attributes[c], transit_df['idx'])

        if units == 'utility':
            spec = simulate.read_model_spec(file_name=tap_tap_settings['SPEC'])

            transit_utilities = compute_utilities(
                tap_tap_settings,
                choosers=transit_df,
                spec=spec,
                locals_dict=locals_dict,
                network_los=self.network_los,
                chunk_size=self.chunk_size,
                trace_label=trace_label)

            transit_df = pd.concat([transit_df[['idx', 'btap', 'atap']], transit_utilities], axis=1)
        else:

            locals_d = {'los': self.network_los}
            locals_d.update(model_constants)

            assignment_spec = assign.read_assignment_spec(file_name=config.config_file_path(tap_tap_settings['SPEC']))

            results, _, _ = assign.assign_variables(assignment_spec, transit_df, locals_d)
            assert len(results.columns == 1)
            transit_df['transit'] = results

            # filter out unavailable btap_atap pairs
            logger.debug(f"{(transit_df['transit'] <= 0).sum()} unavailable tap_tap pairs out of {len(transit_df)}")
            transit_df = transit_df[transit_df.transit > 0]

            transit_df.drop(columns=attribute_columns, inplace=True)

        return transit_df

    def best_paths(self, recipe, path_type, maz_od_df, access_df, egress_df, transit_df):

        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')

        maz_od_df['seq'] = maz_od_df.index
        # maz_od_df has one row per chooser
        # inner join to add rows for each access, egress, and transit segment combination
        path_df = maz_od_df. \
            merge(access_df, on=['idx', 'omaz'], how='inner'). \
            merge(egress_df, on=['idx', 'dmaz'], how='inner'). \
            merge(transit_df, on=['idx', 'atap', 'btap'], how='inner')

        transit_sets = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.tap_tap_expressions.sets')
        for c in transit_sets:
            path_df[c] = path_df[c] + path_df['access'] + path_df['egress']
        path_df.drop(columns=['access', 'egress'], inplace=True)

        path_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}')
        max_paths_across_tap_sets = path_settings.get('max_paths_across_tap_sets', 1)
        max_paths_per_tap_set = path_settings.get('max_paths_per_tap_set', 1)

        # best paths by tap set
        keep = pd.Series(False, index=path_df.index)
        for c in transit_sets:
            keep |= path_df.index.isin(
                path_df[['seq', c]].sort_values(by=c, ascending=False).
                groupby(['seq']).head(max_paths_per_tap_set).index
            )
        path_df = path_df[keep]

        # best paths overall by seq
        path_df[units] = path_df[transit_sets].max(axis=1)
        path_df.drop(columns=transit_sets, inplace=True)
        path_df = path_df.sort_values(by=['seq', units], ascending=[True, False])
        keep = path_df.index.isin(
            path_df.
            groupby(['seq']).head(max_paths_across_tap_sets).index
        )
        path_df = path_df[keep]

        return path_df

    def build_virtual_path(self, recipe, path_type, orig, dest, tod, demographic_segment, trace_label):

        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')
        assert units in ['utility', 'time'], f"unrecognized units: {units}. Expected either 'time' or 'utility'."

        access_mode = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}.access')
        egress_mode = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}.egress')

        # maz od pairs requested
        maz_od_df = pd.DataFrame({
            'idx': orig.index.values,
            'omaz': orig.values,
            'dmaz': dest.values,
            'seq': range(len(orig))
        })

        # for location choice, there will be multiple alt dest rows per chooser and duplicate orig.index values
        # but tod and demographic_segment should be the same for all chooser rows (unique orig index values)
        # knowing this allows us to eliminate redundant computations (e.g. utilities of maz_tap pairs)
        duplicated = orig.index.duplicated(keep='first')
        chooser_attributes = pd.DataFrame(index=orig.index[~duplicated])
        chooser_attributes['tod'] = tod if isinstance(tod, str) else tod.loc[~duplicated]
        if demographic_segment is not None:
            chooser_attributes['demographic_segment'] = demographic_segment.loc[~duplicated]

        access_df = self.compute_maz_tap_utilities(
            recipe,
            maz_od_df,
            chooser_attributes,
            leg='access',
            mode=access_mode,
            trace_label=trace_label)

        egress_df = self.compute_maz_tap_utilities(
            recipe,
            maz_od_df,
            chooser_attributes,
            leg='egress',
            mode=egress_mode,
            trace_label=trace_label)

        # path_info for use by expressions (e.g. penalty for drive access if no parking at access tap)
        path_info = {'path_type': path_type, 'access_mode': access_mode, 'egress_mode': egress_mode}
        transit_df = self.compute_tap_tap_utilities(
            recipe,
            access_df,
            egress_df,
            chooser_attributes,
            path_info=path_info,
            trace_label=trace_label)

        path_df = self.best_paths(recipe, path_type, maz_od_df, access_df, egress_df, transit_df)

        if units == 'utility':
            # logsums
            # one row per seq with utilities in columns
            path_df['path_num'] = path_df.groupby('seq').cumcount() + 1

            utilities_df = path_df[['seq', 'path_num', units]].set_index(['seq', 'path_num']).unstack()

            # paths with fewer than the max number of paths will have Nan values for missing data
            # but there should always be at least one path/utility per seq
            assert not utilities_df.isnull().all(axis=1).any()

            # logsum of utilities_df columns
            result = np.log(np.nansum(np.exp(utilities_df.values), axis=1))

        else:

            result = pd.Series(path_df[units].values, index=path_df['idx'])

            # zero-fill rows for O-D pairs where no best path exists because there was no tap-tap transit availability
            result = reindex(result, maz_od_df.idx).fillna(0.0)

        assert len(result) == len(orig)

        # diagnostic
        # maz_od_df['DIST'] = self.network_los.get_default_skim_dict().get('DIST').get(maz_od_df.omaz, maz_od_df.dmaz)
        # maz_od_df[units] = result if units == 'utility' else result.values
        # print(f"maz_od_df\n{maz_od_df}")

        return result

    def get_tvpb_logsum(self, path_type, orig, dest, tod, demographic_segment=None):

        recipe = 'tour_mode_choice'
        return self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment,
                                       trace_label='get_tvpb_logsum')

    def get_tvpb_best_transit_time(self, orig, dest, tod):

        recipe = 'accessibility'
        path_type = 'WTW'

        return self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment=None,
                                       trace_label='get_tvpb_best_transit_time')

    def wrap_logsum(self, orig_key, dest_key, tod_key, segment_key):

        return TransitVirtualPathLogsumWrapper(self, orig_key, dest_key, tod_key, segment_key)


class TransitVirtualPathLogsumWrapper(object):

    def __init__(self, transit_virtual_path_builder, orig_key, dest_key, tod_key, segment_key):

        self.tvpb = transit_virtual_path_builder
        assert hasattr(transit_virtual_path_builder, 'get_tvpb_logsum')

        self.orig_key = orig_key
        self.dest_key = dest_key
        self.tod_key = tod_key
        self.segment_key = segment_key
        self.df = None

        assert isinstance(orig_key, str)
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

        assert self.df is not None, "Call set_df first"
        assert(self.orig_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: orig_key '{self.orig_key}' not in df"
        assert(self.dest_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: dest_key '{self.dest_key}' not in df"
        assert(self.tod_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: tod_key '{self.tod_key}' not in df"
        assert(self.segment_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: segment_key '{self.segment_key}' not in df"

        orig = self.df[self.orig_key].astype('int')
        dest = self.df[self.dest_key].astype('int')
        tod = self.df[self.tod_key]
        segment = self.df[self.segment_key]

        skim_values = \
            self.tvpb.get_tvpb_logsum(
                path_type,
                orig,
                dest,
                tod,
                segment
                )

        skim_values = pd.Series(skim_values, self.df.index)

        return skim_values

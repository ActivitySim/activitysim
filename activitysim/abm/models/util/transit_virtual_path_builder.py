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
from activitysim.core import config

from activitysim.core import logit
from activitysim.core import simulate
from activitysim.core import los

from activitysim.core.util import reindex

from activitysim.abm.models.util import expressions
from activitysim.core import assign

logger = logging.getLogger(__name__)


class TransitVirtualPathBuilder(object):

    def __init__(self, network_los):

        self.network_los = network_los

        # FIXME - need to recompute headroom?
        self.chunk_size = inject.get_injectable('chunk_size')

        assert network_los.zone_system == los.THREE_ZONE, \
            f"TransitVirtualPathBuilder: network_los zone_system not THREE_ZONE"

    def trace_df(self, df, trace_label, extension=None, bug=False):

        if extension:
            trace_label = tracing.extend_trace_label(trace_label, extension)

        assert len(df) > 0

        tracing.trace_df(df, label=trace_label, slicer='NONE', transpose=False)

        if bug:
            print(f"{trace_label}\n{df}")
            bug_out

    def compute_utilities(self, model_settings, choosers, spec, locals_dict, network_los, trace_label, trace):

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

            if trace:
                self.trace_df(choosers, trace_label, 'choosers')

        utilities = simulate.eval_utilities(
            spec,
            choosers,
            locals_d=locals_dict,
            trace_all_rows=trace,
            trace_label=trace_label)

        return utilities

    def compute_maz_tap_utilities(self, recipe, maz_od_df, chooser_attributes, leg, mode, trace_label, trace):

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

            maz_tap_spec = simulate.read_model_spec(file_name=maz_tap_settings['SPEC'])

            access_df[leg] = self.compute_utilities(
                maz_tap_settings,
                access_df,
                spec=maz_tap_spec,
                locals_dict=model_constants,
                network_los=self.network_los,
                trace_label=trace_label, trace=trace)

        else:

            assignment_spec = assign.read_assignment_spec(file_name=config.config_file_path(maz_tap_settings['SPEC']))

            results, _, _ = assign.assign_variables(assignment_spec, access_df, model_constants)
            assert len(results.columns == 1)
            access_df[leg] = results

        # drop utility computation columns ('tod', 'demographic_segment' and maz_to_tap_df time/distance columns)
        access_df.drop(columns=attribute_columns + chooser_columns, inplace=True)

        if trace:
            self.trace_df(access_df, trace_label, 'access_df')

        return access_df

    def compute_tap_tap_utilities(self, recipe, access_df, egress_df, chooser_attributes, path_info,
                                  trace_label, trace):

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

        # don't want transit trips that start and stop in same tap
        transit_df = transit_df[transit_df.atap != transit_df.btap]

        for c in list(attribute_columns):
            transit_df[c] = reindex(chooser_attributes[c], transit_df['idx'])

        if units == 'utility':
            spec = simulate.read_model_spec(file_name=tap_tap_settings['SPEC'])

            transit_utilities = self.compute_utilities(
                tap_tap_settings,
                choosers=transit_df,
                spec=spec,
                locals_dict=locals_dict,
                network_los=self.network_los,
                trace_label=trace_label, trace=trace)

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

        if trace:
            self.trace_df(transit_df, trace_label, 'transit_df')

        return transit_df

    def best_paths(self, recipe, path_type, maz_od_df, access_df, egress_df, transit_df, trace_label, trace=False):

        trace_label = tracing.extend_trace_label(trace_label, 'best_paths')

        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')
        transit_sets = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.tap_tap_expressions.sets')

        path_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}')
        max_paths_per_tap_set = path_settings.get('max_paths_per_tap_set', 1)
        max_paths_across_tap_sets = path_settings.get('max_paths_across_tap_sets', 1)

        assert units in ['utility', 'time'], f"unrecognized units: {units}. Expected either 'time' or 'utility'."
        smaller_is_better = (units in ['time'])

        maz_od_df['seq'] = maz_od_df.index
        # maz_od_df has one row per chooser
        # inner join to add rows for each access, egress, and transit segment combination
        path_df = maz_od_df. \
            merge(access_df, on=['idx', 'omaz'], how='inner'). \
            merge(egress_df, on=['idx', 'dmaz'], how='inner'). \
            merge(transit_df, on=['idx', 'atap', 'btap'], how='inner')

        for c in transit_sets:
            path_df[c] = path_df[c] + path_df['access'] + path_df['egress']
        path_df.drop(columns=['access', 'egress'], inplace=True)

        # best paths by tap set
        keep = pd.Series(False, index=path_df.index)
        for c in transit_sets:
            keep |= path_df.index.isin(
                path_df[['seq', c]].sort_values(by=c, ascending=smaller_is_better).
                groupby(['seq']).head(max_paths_per_tap_set).index
            )
        path_df = path_df[keep]

        # (only trace if we actually might remove some sets)
        if trace and max_paths_across_tap_sets < (max_paths_per_tap_set * len(transit_sets)):
            self.trace_df(path_df, trace_label, 'best_paths.max_per_set')

        # best paths overall by seq
        path_df[units] = path_df[transit_sets].min(axis=1) if smaller_is_better else path_df[transit_sets].max(axis=1)
        path_df.drop(columns=transit_sets, inplace=True)
        path_df = path_df.sort_values(by=['seq', units], ascending=[True, smaller_is_better])
        keep = path_df.index.isin(
            path_df.
            groupby(['seq']).head(max_paths_across_tap_sets).index
        )

        path_df = path_df[keep]

        if trace:
            self.trace_df(path_df, trace_label, 'best_paths.best_paths')

        return path_df

    def build_virtual_path(self, recipe, path_type, orig, dest, tod, demographic_segment,
                           want_choices, trace_label,
                           trace_targets=None, override_choices=None):

        trace_label = tracing.extend_trace_label(trace_label, 'build_virtual_path')

        trace = trace_targets is not None
        if trace:
            assert trace_targets.any()
            orig = orig[trace_targets]
            dest = dest[trace_targets]
            assert len(orig) > 0
            assert len(dest) > 0

            if not isinstance(tod, str):
                tod = tod[trace_targets]
            if demographic_segment is not None:
                demographic_segment = demographic_segment[trace_targets]
                assert len(demographic_segment) > 0

            if want_choices:
                assert override_choices is not None
                override_choices = override_choices[trace_targets]

        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')
        assert units in ['utility', 'time'], f"unrecognized units: {units}. Expected either 'time' or 'utility'."
        assert units == 'utility' or not want_choices, "'want_choices' only supported supported if units is utility"

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
            trace_label=trace_label, trace=trace)

        egress_df = self.compute_maz_tap_utilities(
            recipe,
            maz_od_df,
            chooser_attributes,
            leg='egress',
            mode=egress_mode,
            trace_label=trace_label, trace=trace)

        # path_info for use by expressions (e.g. penalty for drive access if no parking at access tap)
        path_info = {'path_type': path_type, 'access_mode': access_mode, 'egress_mode': egress_mode}
        transit_df = self.compute_tap_tap_utilities(
            recipe,
            access_df,
            egress_df,
            chooser_attributes,
            path_info=path_info,
            trace_label=trace_label, trace=trace)

        path_df = self.best_paths(recipe, path_type, maz_od_df, access_df, egress_df, transit_df, trace_label, trace)

        if units == 'utility':
            # logsums
            # one row per seq with utilities in columns
            # path_num 0-based to aligh with logit.make_choices 0-based choice indexes
            path_df['path_num'] = path_df.groupby('seq').cumcount()

            utilities_df = path_df[['seq', 'path_num', units]].set_index(['seq', 'path_num']).unstack()

            # paths with fewer than the max number of paths will have Nan values for missing data
            # but there should always be at least one path/utility per seq
            # FIXME what if there is no tap-tap transit availability?
            assert not utilities_df.isnull().all(axis=1).any()

            assert (utilities_df.index == maz_od_df.seq).all()  # should be aligned with maz_od_df and so with orig

            # logsum of utilities_df columns
            logsum = np.log(np.nansum(np.exp(utilities_df.values), axis=1))

            if want_choices:
                # FIXME - for aid debugging, increase the likelihood of getting 2nd choice
                # if len(utilities_df.columns)>1:
                #     utilities_df.columns = utilities_df.columns.droplevel()
                #     utilities_df[1] = utilities_df[1]/2.5

                # utilities for missing paths will be Nan
                utilities_df = utilities_df.fillna(-999.0)
                # orig index to identify appropriate random number channel to use making choices
                utilities_df.index = orig.index

                probs = logit.utils_to_probs(utilities_df, trace_label=trace_label)

                if trace:
                    choices = override_choices
                else:
                    choices, rands = logit.make_choices(probs, trace_label=trace_label)

                # we need to get btap and atap from path_df with same seq and path_num
                columns_to_cache = ['btap', 'atap']
                logsum_df = \
                    pd.merge(pd.DataFrame({'seq': range(len(orig)), 'path_num': choices.values}),
                             path_df[['seq', 'path_num'] + columns_to_cache],
                             on=['seq', 'path_num'], how='left')

                # keep path_num choice for caller to pass as override_choices when tracing
                logsum_df.drop(columns=['seq'], inplace=True)

                logsum_df.index = orig.index
                logsum_df['logsum'] = logsum
                results = logsum_df

                if trace:
                    utilities_df['choices'] = choices
                    self.trace_df(utilities_df, trace_label, 'utilities_df')
                    self.trace_df(logsum_df, trace_label, 'logsum_df')
            else:

                assert len(logsum) == len(orig)
                results = pd.DataFrame({'logsum': logsum}, index=orig.index)

        elif units == 'time':

            # return a series
            results = pd.Series(path_df[units].values, index=path_df['idx'])

            # zero-fill rows for O-D pairs where no best path exists because there was no tap-tap transit availability
            results = reindex(results, maz_od_df.idx).fillna(0.0)
        else:
            raise RuntimeError(f"Unrecognized units: '{units}")

        assert len(results) == len(orig)

        # diagnostic
        # maz_od_df['DIST'] = self.network_los.get_default_skim_dict().get('DIST').get(maz_od_df.omaz, maz_od_df.dmaz)
        # maz_od_df[units] = results.logsum if units == 'utility' else results.values
        # print(f"maz_od_df\n{maz_od_df}")

        return results

    def get_tvpb_logsum(self, path_type, orig, dest, tod, demographic_segment, want_choices, trace_label=None):

        # assume they have given us a more specific name (since there may be more than one active wrapper)
        trace_label = trace_label or 'get_tvpb_logsum'

        recipe = 'tour_mode_choice'
        logsum_df = \
            self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment,
                                    want_choices, trace_label)

        trace_hh_id = inject.get_injectable("trace_hh_id", None)
        if trace_hh_id:
            trace_targets = tracing.trace_targets(orig)
            # choices from preceding run (because random numbers)
            override_choices = logsum_df['path_num'] if want_choices else None
            if trace_targets.any():
                self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment,
                                        want_choices=want_choices, override_choices=override_choices,
                                        trace_label=trace_label, trace_targets=trace_targets,
                                        )

        return logsum_df

    def get_tvpb_best_transit_time(self, orig, dest, tod):

        # FIXME lots of pathological knowledge here as we are only called by accessibility directly from expressions

        trace_label = tracing.extend_trace_label('accessibility.get_tvpb_best_transit_time', tod)
        recipe = 'accessibility'
        path_type = 'WTW'
        result = \
            self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment=None,
                                    want_choices=False, trace_label=trace_label)

        trace_od = inject.get_injectable("trace_od", None)
        if trace_od:
            trace_targets = (orig == trace_od[0]) & (dest == trace_od[1])
            if trace_targets.any():
                self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment=None,
                                        want_choices=False, trace_label=trace_label, trace_targets=trace_targets)

        return result

    def wrap_logsum(self, orig_key, dest_key, tod_key, segment_key,
                    cache_choices=False, trace_label=None):

        return TransitVirtualPathLogsumWrapper(self, orig_key, dest_key, tod_key, segment_key,
                                               cache_choices, trace_label)


class TransitVirtualPathLogsumWrapper(object):

    def __init__(self, transit_virtual_path_builder, orig_key, dest_key, tod_key, segment_key,
                 cache_choices, trace_label):

        self.tvpb = transit_virtual_path_builder
        assert hasattr(transit_virtual_path_builder, 'get_tvpb_logsum')

        self.orig_key = orig_key
        self.dest_key = dest_key
        self.tod_key = tod_key
        self.segment_key = segment_key
        self.df = None

        self.cache_choices = cache_choices
        self.cache = {} if cache_choices else None

        self.base_trace_label = trace_label or 'tvpb_logsum'
        self.trace_label = self.base_trace_label

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

    def extend_trace_label(self, extension=None):
        if extension:
            self.trace_label = tracing.extend_trace_label(self.base_trace_label, extension)
        else:
            self.trace_label = self.base_trace_label

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

        logsum_df = \
            self.tvpb.get_tvpb_logsum(path_type, orig, dest, tod, segment,
                                      want_choices=self.cache_choices,
                                      trace_label=self.trace_label)

        if self.cache_choices:

            # not tested on duplicate index because not currently needed
            # caching strategy does not require unique indexes but care would need to be taken to maintain alignment
            assert not orig.index.duplicated().any()

            # we only need to cache taps
            choices_df = logsum_df[['atap', 'btap']]

            if path_type in self.cache:
                assert len(self.cache.get(path_type).index.intersection(logsum_df.index)) == 0
                choices_df = pd.concat([self.cache.get(path_type), choices_df])

            self.cache[path_type] = choices_df

        return logsum_df.logsum

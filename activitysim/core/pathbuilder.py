# ActivitySim
# See full license in LICENSE.txt.
import logging
import warnings
from builtins import range

import numpy as np
import pandas as pd

from activitysim.core import (
    assign,
    chunk,
    config,
    expressions,
    inject,
    logit,
    los,
    pathbuilder_cache,
    simulate,
    tracing,
)
from activitysim.core.pathbuilder_cache import memo
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)

TIMING = True
TRACE_CHUNK = True
ERR_CHECK = True
TRACE_COMPLEXITY = False  # diagnostic: log the omaz,dmaz pairs with the greatest number of virtual tap-tap paths

UNAVAILABLE = -999

# used as base file name for cached files and as shared buffer tag
CACHE_TAG = "tap_tap_utilities"


def compute_utilities(
    network_los,
    model_settings,
    choosers,
    model_constants,
    trace_label,
    trace=False,
    trace_column_names=None,
):
    """
    Compute utilities
    """
    trace_label = tracing.extend_trace_label(trace_label, "compute_utils")

    with chunk.chunk_log(trace_label):

        logger.debug(
            f"{trace_label} Running compute_utilities with {choosers.shape[0]} choosers"
        )

        locals_dict = {"np": np, "los": network_los}
        locals_dict.update(model_constants)

        # we don't grok coefficients, but allow them to use constants in spec alt columns
        spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
        for c in spec.columns:
            if c != simulate.SPEC_LABEL_NAME:
                spec[c] = spec[c].map(lambda s: model_constants.get(s, s)).astype(float)

        # - run preprocessor to annotate choosers
        preprocessor_settings = model_settings.get("PREPROCESSOR")
        if preprocessor_settings:

            # don't want to alter caller's dataframe
            choosers = choosers.copy()

            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=trace_label,
            )

        utilities = simulate.eval_utilities(
            spec,
            choosers,
            locals_d=locals_dict,
            trace_all_rows=trace,
            trace_label=trace_label,
            trace_column_names=trace_column_names,
        )

    return utilities


class TransitVirtualPathBuilder(object):
    """
    Transit virtual path builder for three zone systems
    """

    def __init__(self, network_los):

        self.network_los = network_los

        self.uid_calculator = pathbuilder_cache.TapTapUidCalculator(network_los)

        # note: pathbuilder_cache is lightweight until opened
        self.tap_cache = pathbuilder_cache.TVPBCache(
            self.network_los, self.uid_calculator, CACHE_TAG
        )

        assert (
            network_los.zone_system == los.THREE_ZONE
        ), f"TransitVirtualPathBuilder: network_los zone_system not THREE_ZONE"

    def trace_df(self, df, trace_label, extension):
        assert len(df) > 0
        tracing.trace_df(
            df,
            label=tracing.extend_trace_label(trace_label, extension),
            slicer="NONE",
            transpose=False,
        )

    def trace_maz_tap(self, maz_od_df, access_mode, egress_mode):
        def maz_tap_stats(mode, name):
            maz_tap_df = self.network_los.maz_to_tap_dfs[mode].reset_index()
            logger.debug(f"TVPB access_maz_tap {maz_tap_df.shape}")
            MAZ_count = len(maz_tap_df.MAZ.unique())
            TAP_count = len(maz_tap_df.TAP.unique())
            MAZ_PER_TAP = MAZ_count / TAP_count
            logger.debug(
                f"TVPB maz_tap_stats {name} {mode} MAZ {MAZ_count} TAP {TAP_count} ratio {MAZ_PER_TAP}"
            )

        logger.debug(f"TVPB maz_od_df {maz_od_df.shape}")

        maz_tap_stats(access_mode, "access")
        maz_tap_stats(egress_mode, "egress")

    def units_for_recipe(self, recipe):
        units = self.network_los.setting(f"TVPB_SETTINGS.{recipe}.units")
        assert units in [
            "utility",
            "time",
        ], f"unrecognized units: {units} for {recipe}. Expected either 'time' or 'utility'."
        return units

    def compute_maz_tap_utilities(
        self, recipe, maz_od_df, chooser_attributes, leg, mode, trace_label, trace
    ):

        trace_label = tracing.extend_trace_label(trace_label, f"maz_tap_utils.{leg}")

        with chunk.chunk_log(trace_label):
            maz_tap_settings = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.maz_tap_settings.{mode}"
            )
            chooser_columns = maz_tap_settings["CHOOSER_COLUMNS"]
            attribute_columns = (
                list(chooser_attributes.columns)
                if chooser_attributes is not None
                else []
            )
            model_constants = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.CONSTANTS"
            )

            if leg == "access":
                maz_col = "omaz"
                tap_col = "btap"
            else:
                maz_col = "dmaz"
                tap_col = "atap"

            # maz_to_tap access/egress utilities
            # deduped utilities_df - one row per chooser for each boarding tap (btap) accessible from omaz
            utilities_df = self.network_los.maz_to_tap_dfs[mode]

            utilities_df = (
                utilities_df[chooser_columns]
                .reset_index(drop=False)
                .rename(columns={"MAZ": maz_col, "TAP": tap_col})
            )
            utilities_df = pd.merge(
                maz_od_df[["idx", maz_col]].drop_duplicates(),
                utilities_df,
                on=maz_col,
                how="inner",
            )

            if len(utilities_df) == 0:
                trace = False
            # add any supplemental chooser attributes (e.g. demographic_segment, tod)
            for c in attribute_columns:
                utilities_df[c] = reindex(chooser_attributes[c], utilities_df["idx"])

            chunk.log_df(trace_label, "utilities_df", utilities_df)

            if self.units_for_recipe(recipe) == "utility":

                utilities_df[leg] = compute_utilities(
                    self.network_los,
                    maz_tap_settings,
                    utilities_df,
                    model_constants=model_constants,
                    trace_label=trace_label,
                    trace=trace,
                    trace_column_names=["idx", maz_col, tap_col] if trace else None,
                )

                chunk.log_df(trace_label, "utilities_df", utilities_df)  # annotated

            else:

                assignment_spec = assign.read_assignment_spec(
                    file_name=config.config_file_path(maz_tap_settings["SPEC"])
                )

                results, _, _ = assign.assign_variables(
                    assignment_spec, utilities_df, model_constants
                )
                assert len(results.columns == 1)
                utilities_df[leg] = results

            chunk.log_df(trace_label, "utilities_df", utilities_df)

            if trace:
                self.trace_df(utilities_df, trace_label, "utilities_df")

            # drop utility computation columns ('tod', 'demographic_segment' and maz_to_tap_df time/distance columns)
            utilities_df.drop(columns=attribute_columns + chooser_columns, inplace=True)

        return utilities_df

    def all_transit_paths(
        self, access_df, egress_df, chooser_attributes, trace_label, trace
    ):

        trace_label = tracing.extend_trace_label(trace_label, "all_transit_paths")

        # deduped transit_df has one row per chooser for each boarding (btap) and alighting (atap) pair
        transit_df = pd.merge(
            access_df[["idx", "btap"]], egress_df[["idx", "atap"]], on="idx"
        ).drop_duplicates()

        # don't want transit trips that start and stop in same tap
        transit_df = transit_df[transit_df.atap != transit_df.btap]

        for c in list(chooser_attributes.columns):
            transit_df[c] = reindex(chooser_attributes[c], transit_df["idx"])

        transit_df = transit_df.reset_index(drop=True)

        if trace:
            self.trace_df(transit_df, trace_label, "all_transit_df")

        return transit_df

    def compute_tap_tap_utilities(
        self,
        recipe,
        access_df,
        egress_df,
        chooser_attributes,
        path_info,
        trace_label,
        trace,
    ):
        """
        create transit_df and compute utilities for all atap-btap pairs between omaz in access and dmaz in egress_df
        compute the utilities using the tap_tap utility expressions file specified in tap_tap_settings

        transit_df contains all possible access omaz/btap to egress dmaz/atap transit path pairs for each chooser

        trace should be True as we don't encourage/support dynamic utility computation except when tracing
        (precompute being fairly fast)

        Parameters
        ----------
        recipe: str
           'recipe' key in network_los.yaml TVPB_SETTINGS e.g. tour_mode_choice
        access_df: pandas.DataFrame
            dataframe with 'idx' and 'omaz' columns
        egress_df: pandas.DataFrame
            dataframe with 'idx' and 'dmaz' columns
        chooser_attributes: dict
        path_info
        trace_label: str
        trace: boolean

        Returns
        -------
        transit_df: pandas.dataframe
        """

        assert trace

        trace_label = tracing.extend_trace_label(trace_label, "compute_tap_tap_utils")

        with chunk.chunk_log(trace_label):

            model_constants = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.CONSTANTS"
            )
            tap_tap_settings = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.tap_tap_settings"
            )

            with memo("#TVPB CACHE compute_tap_tap_utilities all_transit_paths"):
                transit_df = self.all_transit_paths(
                    access_df, egress_df, chooser_attributes, trace_label, trace
                )
                # note: transit_df index is arbitrary
            chunk.log_df(trace_label, "transit_df", transit_df)

            # FIXME some expressions may want to know access mode -
            locals_dict = path_info.copy()
            locals_dict.update(model_constants)

            # columns needed for compute_utilities
            chooser_columns = ["btap", "atap"] + list(chooser_attributes.columns)

            # deduplicate transit_df to unique_transit_df
            with memo("#TVPB compute_tap_tap_utilities deduplicate transit_df"):

                attribute_segments = self.network_los.setting(
                    "TVPB_SETTINGS.tour_mode_choice.tap_tap_settings.attribute_segments"
                )
                scalar_attributes = {
                    k: locals_dict[k]
                    for k in attribute_segments.keys()
                    if k not in transit_df
                }

                transit_df["uid"] = self.uid_calculator.get_unique_ids(
                    transit_df, scalar_attributes
                )

                unique_transit_df = transit_df.loc[
                    ~transit_df.uid.duplicated(), chooser_columns + ["uid"]
                ]
                logger.debug(
                    f"#TVPB CACHE deduped transit_df from {len(transit_df)} to {len(unique_transit_df)}"
                )

                unique_transit_df.set_index("uid", inplace=True)
                chunk.log_df(trace_label, "unique_transit_df", unique_transit_df)

                transit_df = transit_df[
                    ["idx", "btap", "atap", "uid"]
                ]  # don't need chooser columns
                chunk.log_df(trace_label, "transit_df", transit_df)

            logger.debug(
                f"#TVPB CACHE compute_tap_tap_utilities dedupe transit_df "
                f"from {len(transit_df)} to {len(unique_transit_df)} rows"
            )

            num_unique_transit_rows = len(unique_transit_df)  # errcheck
            logger.debug(
                f"#TVPB CACHE compute_tap_tap_utilities compute_utilities for {len(unique_transit_df)} rows"
            )

            with memo("#TVPB compute_tap_tap_utilities compute_utilities"):
                unique_utilities_df = compute_utilities(
                    self.network_los,
                    tap_tap_settings,
                    choosers=unique_transit_df,
                    model_constants=locals_dict,
                    trace_label=trace_label,
                    trace=trace,
                    trace_column_names=chooser_columns if trace else None,
                )
                chunk.log_df(trace_label, "unique_utilities_df", unique_utilities_df)
                chunk.log_df(
                    trace_label, "unique_transit_df", unique_transit_df
                )  # annotated

                if trace:
                    # combine unique_transit_df with unique_utilities_df for legibility
                    omnibus_df = pd.merge(
                        unique_transit_df,
                        unique_utilities_df,
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    self.trace_df(omnibus_df, trace_label, "unique_utilities_df")
                    chunk.log_df(trace_label, "omnibus_df", omnibus_df)
                    del omnibus_df
                    chunk.log_df(trace_label, "omnibus_df", None)

            assert num_unique_transit_rows == len(unique_utilities_df)  # errcheck

            # redupe unique_transit_df back into transit_df
            with memo("#TVPB compute_tap_tap_utilities redupe transit_df"):

                # idx = transit_df.index
                transit_df = pd.merge(
                    transit_df, unique_utilities_df, left_on="uid", right_index=True
                )
                del transit_df["uid"]
                # transit_df.index = idx
                # note: left merge on columns does not preserve index,
                # but transit_df index is arbitrary so no need to restore

                chunk.log_df(trace_label, "transit_df", transit_df)

            for c in unique_utilities_df:
                assert ERR_CHECK and not transit_df[c].isnull().any()

            if len(unique_transit_df) > 0:
                # if all rows were cached, then unique_utilities_df is just a ref to cache
                del unique_utilities_df
                chunk.log_df(trace_label, "unique_utilities_df", None)

            chunk.log_df(trace_label, "transit_df", None)

            if trace:
                self.trace_df(transit_df, trace_label, "transit_df")

        return transit_df

    def lookup_tap_tap_utilities(
        self,
        recipe,
        maz_od_df,
        access_df,
        egress_df,
        chooser_attributes,
        path_info,
        trace_label,
    ):
        """
        create transit_df and compute utilities for all atap-btap pairs between omaz in access and dmaz in egress_df
        look up the utilities in the precomputed tap_cache data (which is indexed by uid_calculator unique_ids)
        (unique_id can used as a zero-based index into the data array)

        transit_df contains all possible access omaz/btap to egress dmaz/atap transit path pairs for each chooser

        Parameters
        ----------
        recipe
        maz_od_df
        access_df
        egress_df
        chooser_attributes
        path_info
        trace_label

        Returns
        -------

        """

        trace_label = tracing.extend_trace_label(trace_label, "lookup_tap_tap_utils")

        with chunk.chunk_log(trace_label):

            with memo("#TVPB CACHE lookup_tap_tap_utilities all_transit_paths"):
                transit_df = self.all_transit_paths(
                    access_df, egress_df, chooser_attributes, trace_label, trace=False
                )
                # note: transit_df index is arbitrary
                chunk.log_df(trace_label, "transit_df", transit_df)

            if TRACE_COMPLEXITY:
                # diagnostic: log the omaz,dmaz pairs with the greatest number of virtual tap-tap paths
                num_paths = transit_df.groupby(["idx"]).size().to_frame("n")
                num_paths = pd.merge(
                    maz_od_df, num_paths, left_on="idx", right_index=True
                )
                num_paths = num_paths[["omaz", "dmaz", "n"]].drop_duplicates(
                    subset=["omaz", "dmaz"]
                )
                num_paths = num_paths.sort_values("n", ascending=False).reset_index(
                    drop=True
                )
                logger.debug(f"num_paths\n{num_paths.head(10)}")

            # FIXME some expressions may want to know access mode -
            locals_dict = path_info.copy()

            # add uid column to transit_df
            with memo("#TVPB lookup_tap_tap_utilities assign uid"):
                attribute_segments = self.network_los.setting(
                    "TVPB_SETTINGS.tour_mode_choice.tap_tap_settings.attribute_segments"
                )
                scalar_attributes = {
                    k: locals_dict[k]
                    for k in attribute_segments.keys()
                    if k not in transit_df
                }

                transit_df.index = self.uid_calculator.get_unique_ids(
                    transit_df, scalar_attributes
                )
                transit_df = transit_df[
                    ["idx", "btap", "atap"]
                ]  # just needed chooser_columns for uid calculation
                chunk.log_df(trace_label, "transit_df add uid index", transit_df)

            with memo("#TVPB lookup_tap_tap_utilities reindex transit_df"):
                utilities = self.tap_cache.data
                i = 0
                for column_name in self.uid_calculator.set_names:
                    transit_df[column_name] = utilities[transit_df.index.values, i]
                    i += 1

            for c in self.uid_calculator.set_names:
                assert ERR_CHECK and not transit_df[c].isnull().any()

            chunk.log_df(trace_label, "transit_df", None)

        return transit_df

    def compute_tap_tap_time(
        self,
        recipe,
        access_df,
        egress_df,
        chooser_attributes,
        path_info,
        trace_label,
        trace,
    ):

        trace_label = tracing.extend_trace_label(trace_label, "compute_tap_tap_time")

        with chunk.chunk_log(trace_label):

            model_constants = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.CONSTANTS"
            )
            tap_tap_settings = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.tap_tap_settings"
            )

            with memo("#TVPB CACHE compute_tap_tap_utilities all_transit_paths"):
                transit_df = self.all_transit_paths(
                    access_df, egress_df, chooser_attributes, trace_label, trace
                )
                # note: transit_df index is arbitrary
                chunk.log_df(trace_label, "transit_df", transit_df)

            # some expressions may want to know access mode -
            locals_dict = path_info.copy()
            locals_dict["los"] = self.network_los
            locals_dict.update(model_constants)

            assignment_spec = assign.read_assignment_spec(
                file_name=config.config_file_path(tap_tap_settings["SPEC"])
            )

            DEDUPE = True
            if DEDUPE:

                # assign uid for reduping
                max_atap = transit_df.atap.max() + 1
                transit_df["uid"] = transit_df.btap * max_atap + transit_df.atap

                # dedupe
                chooser_attribute_columns = list(chooser_attributes.columns)
                unique_transit_df = transit_df.loc[
                    ~transit_df.uid.duplicated(),
                    ["btap", "atap", "uid"] + chooser_attribute_columns,
                ]
                unique_transit_df.set_index("uid", inplace=True)
                chunk.log_df(trace_label, "unique_transit_df", unique_transit_df)

                logger.debug(
                    f"#TVPB CACHE deduped transit_df from {len(transit_df)} to {len(unique_transit_df)}"
                )

                # assign_variables
                results, _, _ = assign.assign_variables(
                    assignment_spec, unique_transit_df, locals_dict
                )
                assert len(results.columns == 1)
                unique_transit_df["transit"] = results

                # redupe results back into transit_df
                with memo("#TVPB compute_tap_tap_time redupe transit_df"):
                    transit_df["transit"] = reindex(
                        unique_transit_df.transit, transit_df.uid
                    )

                del transit_df["uid"]
                del unique_transit_df
                chunk.log_df(trace_label, "transit_df", transit_df)
                chunk.log_df(trace_label, "unique_transit_df", None)

            else:
                results, _, _ = assign.assign_variables(
                    assignment_spec, transit_df, locals_dict
                )
                assert len(results.columns == 1)
                transit_df["transit"] = results

            # filter out unavailable btap_atap pairs
            logger.debug(
                f"{(transit_df['transit'] <= 0).sum()} unavailable tap_tap pairs out of {len(transit_df)}"
            )
            transit_df = transit_df[transit_df.transit > 0]

            transit_df.drop(columns=chooser_attributes.columns, inplace=True)

            chunk.log_df(trace_label, "transit_df", None)

            if trace:
                self.trace_df(transit_df, trace_label, "transit_df")

        return transit_df

    def compute_tap_tap(
        self,
        recipe,
        maz_od_df,
        access_df,
        egress_df,
        chooser_attributes,
        path_info,
        trace_label,
        trace,
    ):

        if self.units_for_recipe(recipe) == "utility":

            if not self.tap_cache.is_open:
                with memo("#TVPB compute_tap_tap tap_cache.open"):
                    self.tap_cache.open()

            if trace:
                result = self.compute_tap_tap_utilities(
                    recipe,
                    access_df,
                    egress_df,
                    chooser_attributes,
                    path_info,
                    trace_label,
                    trace,
                )
            else:
                result = self.lookup_tap_tap_utilities(
                    recipe,
                    maz_od_df,
                    access_df,
                    egress_df,
                    chooser_attributes,
                    path_info,
                    trace_label,
                )
            return result
        else:
            assert self.units_for_recipe(recipe) == "time"

            with memo("#TVPB compute_tap_tap_time"):
                result = self.compute_tap_tap_time(
                    recipe,
                    access_df,
                    egress_df,
                    chooser_attributes,
                    path_info,
                    trace_label,
                    trace,
                )
        return result

    def best_paths(
        self,
        recipe,
        path_type,
        maz_od_df,
        access_df,
        egress_df,
        transit_df,
        trace_label,
        trace=False,
    ):

        trace_label = tracing.extend_trace_label(trace_label, "best_paths")

        with chunk.chunk_log(trace_label):

            path_settings = self.network_los.setting(
                f"TVPB_SETTINGS.{recipe}.path_types.{path_type}"
            )
            max_paths_per_tap_set = path_settings.get("max_paths_per_tap_set", 1)
            max_paths_across_tap_sets = path_settings.get(
                "max_paths_across_tap_sets", 1
            )

            units = self.units_for_recipe(recipe)
            smaller_is_better = units in ["time"]

            maz_od_df["seq"] = maz_od_df.index
            # maz_od_df has one row per chooser
            # inner join to add rows for each access, egress, and transit segment combination
            path_df = (
                maz_od_df.merge(access_df, on=["idx", "omaz"], how="inner")
                .merge(egress_df, on=["idx", "dmaz"], how="inner")
                .merge(transit_df, on=["idx", "atap", "btap"], how="inner")
            )

            chunk.log_df(trace_label, "path_df", path_df)

            # transit sets are the transit_df non-join columns
            transit_sets = [
                c for c in transit_df.columns if c not in ["idx", "atap", "btap"]
            ]

            if trace:
                # be nice and show both tap_tap set utility and total_set = access + set + egress
                for c in transit_sets:
                    path_df[f"total_{c}"] = (
                        path_df[c] + path_df["access"] + path_df["egress"]
                    )
                self.trace_df(path_df, trace_label, "best_paths.full")
                for c in transit_sets:
                    del path_df[f"total_{c}"]

            for c in transit_sets:
                path_df[c] = path_df[c] + path_df["access"] + path_df["egress"]
            path_df.drop(columns=["access", "egress"], inplace=True)

            # choose best paths by tap set
            best_paths_list = []
            for c in transit_sets:
                keep = path_df.index.isin(
                    path_df[["seq", c]]
                    .sort_values(by=c, ascending=smaller_is_better)
                    .groupby(["seq"])
                    .head(max_paths_per_tap_set)
                    .index
                )

                best_paths_for_set = path_df[keep]
                best_paths_for_set["path_set"] = c  # remember the path set
                best_paths_for_set[units] = path_df[keep][c]
                best_paths_for_set.drop(columns=transit_sets, inplace=True)
                best_paths_list.append(best_paths_for_set)

            path_df = pd.concat(best_paths_list).sort_values(
                by=["seq", units], ascending=[True, smaller_is_better]
            )

            # choose best paths overall by seq
            path_df = path_df.sort_values(
                by=["seq", units], ascending=[True, smaller_is_better]
            )
            path_df = path_df[
                path_df.index.isin(
                    path_df.groupby(["seq"]).head(max_paths_across_tap_sets).index
                )
            ]

            if trace:
                self.trace_df(path_df, trace_label, "best_paths")

        return path_df

    def build_virtual_path(
        self,
        recipe,
        path_type,
        orig,
        dest,
        tod,
        demographic_segment,
        want_choices,
        trace_label,
        filter_targets=None,
        trace=False,
        override_choices=None,
    ):

        trace_label = tracing.extend_trace_label(trace_label, "build_virtual_path")

        # Tracing is implemented as a seperate, second call that operates ONLY on filter_targets
        assert not (trace and filter_targets is None)
        if filter_targets is not None:
            assert filter_targets.any()

            # slice orig and dest
            orig = orig[filter_targets]
            dest = dest[filter_targets]
            assert len(orig) > 0
            assert len(dest) > 0

            # slice tod and demographic_segment if not scalar
            if not isinstance(tod, str):
                tod = tod[filter_targets]
            if demographic_segment is not None:
                demographic_segment = demographic_segment[filter_targets]
                assert len(demographic_segment) > 0

            # slice choices
            # (requires actual choices from the previous call lest rands change on second call)
            assert want_choices == (override_choices is not None)
            if want_choices:
                override_choices = override_choices[filter_targets]

        units = self.units_for_recipe(recipe)
        assert (
            units == "utility" or not want_choices
        ), "'want_choices' only supported supported if units is utility"

        access_mode = self.network_los.setting(
            f"TVPB_SETTINGS.{recipe}.path_types.{path_type}.access"
        )
        egress_mode = self.network_los.setting(
            f"TVPB_SETTINGS.{recipe}.path_types.{path_type}.egress"
        )
        path_types_settings = self.network_los.setting(
            f"TVPB_SETTINGS.{recipe}.path_types.{path_type}"
        )
        attributes_as_columns = self.network_los.setting(
            f"TVPB_SETTINGS.{recipe}.tap_tap_settings.attributes_as_columns", []
        )

        path_info = {
            "path_type": path_type,
            "access_mode": access_mode,
            "egress_mode": egress_mode,
        }

        # maz od pairs requested
        with memo("#TVPB build_virtual_path maz_od_df"):
            maz_od_df = pd.DataFrame(
                {
                    "idx": orig.index.values,
                    "omaz": orig.values,
                    "dmaz": dest.values,
                    "seq": range(len(orig)),
                }
            )
            chunk.log_df(trace_label, "maz_od_df", maz_od_df)
            self.trace_maz_tap(maz_od_df, access_mode, egress_mode)

        # for location choice, there will be multiple alt dest rows per chooser and duplicate orig.index values
        # but tod and demographic_segment should be the same for all chooser rows (unique orig index values)
        # knowing this allows us to eliminate redundant computations (e.g. utilities of maz_tap pairs)
        duplicated = orig.index.duplicated(keep="first")
        chooser_attributes = pd.DataFrame(index=orig.index[~duplicated])
        if not isinstance(tod, str):
            chooser_attributes["tod"] = tod.loc[~duplicated]
        elif "tod" in attributes_as_columns:
            chooser_attributes["tod"] = tod
        else:
            path_info["tod"] = tod
        if demographic_segment is not None:
            chooser_attributes["demographic_segment"] = demographic_segment.loc[
                ~duplicated
            ]

        with memo("#TVPB build_virtual_path access_df"):
            access_df = self.compute_maz_tap_utilities(
                recipe,
                maz_od_df,
                chooser_attributes,
                leg="access",
                mode=access_mode,
                trace_label=trace_label,
                trace=trace,
            )
        chunk.log_df(trace_label, "access_df", access_df)

        with memo("#TVPB build_virtual_path egress_df"):
            egress_df = self.compute_maz_tap_utilities(
                recipe,
                maz_od_df,
                chooser_attributes,
                leg="egress",
                mode=egress_mode,
                trace_label=trace_label,
                trace=trace,
            )
        chunk.log_df(trace_label, "egress_df", egress_df)

        # L200 will drop all rows if all trips are intra-tap.
        if np.array_equal(access_df["btap"].values, egress_df["atap"].values):
            trace = False

        # path_info for use by expressions (e.g. penalty for drive access if no parking at access tap)
        with memo("#TVPB build_virtual_path compute_tap_tap"):
            if len(access_df) * len(egress_df) == 0:
                trace = False
            transit_df = self.compute_tap_tap(
                recipe,
                maz_od_df,
                access_df,
                egress_df,
                chooser_attributes,
                path_info=path_info,
                trace_label=trace_label,
                trace=trace,
            )
        chunk.log_df(trace_label, "transit_df", transit_df)

        # Cannot trace if df is empty. Prob happened at L200
        if len(transit_df) == 0:
            want_choices = False

        with memo("#TVPB build_virtual_path best_paths"):
            path_df = self.best_paths(
                recipe,
                path_type,
                maz_od_df,
                access_df,
                egress_df,
                transit_df,
                trace_label,
                trace,
            )
        chunk.log_df(trace_label, "path_df", path_df)

        # now that we have created path_df, we are done with the dataframes for the separate legs
        del access_df
        chunk.log_df(trace_label, "access_df", None)
        del egress_df
        chunk.log_df(trace_label, "egress_df", None)
        del transit_df
        chunk.log_df(trace_label, "transit_df", None)

        if units == "utility":

            # logsums
            with memo("#TVPB build_virtual_path logsums"):
                # one row per seq with utilities in columns
                # path_num 0-based to aligh with logit.make_choices 0-based choice indexes
                path_df["path_num"] = path_df.groupby("seq").cumcount()
                chunk.log_df(trace_label, "path_df", path_df)

                utilities_df = (
                    path_df[["seq", "path_num", units]]
                    .set_index(["seq", "path_num"])
                    .unstack()
                )
                utilities_df.columns = (
                    utilities_df.columns.droplevel()
                )  # for legibility

                # add rows missing because no access or egress availability
                utilities_df = pd.concat(
                    [pd.DataFrame(index=maz_od_df.seq), utilities_df], axis=1
                )
                utilities_df = utilities_df.fillna(
                    UNAVAILABLE
                )  # set utilities for missing paths to UNAVAILABLE

                chunk.log_df(trace_label, "utilities_df", utilities_df)

                with warnings.catch_warnings(record=True) as w:
                    # Cause all warnings to always be triggered.
                    # most likely "divide by zero encountered in log" caused by all transit sets non-viable
                    warnings.simplefilter("always")

                    paths_nest_nesting_coefficient = path_types_settings.get(
                        "paths_nest_nesting_coefficient", 1
                    )
                    exp_utilities = np.exp(
                        utilities_df.values / paths_nest_nesting_coefficient
                    )
                    logsums = np.maximum(
                        np.log(np.nansum(exp_utilities, axis=1)), UNAVAILABLE
                    )

                    if len(w) > 0:
                        for wrn in w:
                            logger.warning(
                                f"{trace_label} - {type(wrn).__name__} ({wrn.message})"
                            )

                        DUMP = False
                        if DUMP:
                            zero_utilities_df = utilities_df[
                                np.nansum(np.exp(utilities_df.values), axis=1) == 0
                            ]
                            zero_utilities_df.to_csv(
                                config.output_file_path("warning_utilities_df.csv"),
                                index=True,
                            )

            if want_choices:

                # orig index to identify appropriate random number channel to use making choices
                utilities_df.index = orig.index

                with memo("#TVPB build_virtual_path make_choices"):

                    probs = logit.utils_to_probs(
                        utilities_df, allow_zero_probs=True, trace_label=trace_label
                    )
                    chunk.log_df(trace_label, "probs", probs)

                    if trace:
                        choices = override_choices

                        utilities_df["choices"] = choices
                        self.trace_df(utilities_df, trace_label, "utilities_df")

                        probs["choices"] = choices
                        self.trace_df(probs, trace_label, "probs")
                    else:

                        choices, rands = logit.make_choices(
                            probs, allow_bad_probs=True, trace_label=trace_label
                        )

                        chunk.log_df(trace_label, "rands", rands)
                        del rands
                        chunk.log_df(trace_label, "rands", None)

                    del probs
                    chunk.log_df(trace_label, "probs", None)

                # we need to get path_set, btap, atap from path_df row with same seq and path_num
                # drop seq join column, but keep path_num of choice to override_choices when tracing
                columns_to_cache = ["btap", "atap", "path_set", "path_num"]
                logsum_df = (
                    pd.merge(
                        pd.DataFrame(
                            {"seq": range(len(orig)), "path_num": choices.values}
                        ),
                        path_df[["seq"] + columns_to_cache],
                        on=["seq", "path_num"],
                        how="left",
                    )
                    .drop(columns=["seq"])
                    .set_index(orig.index)
                )

                logsum_df["logsum"] = logsums

            else:

                assert len(logsums) == len(orig)
                logsum_df = pd.DataFrame({"logsum": logsums}, index=orig.index)

            chunk.log_df(trace_label, "logsum_df", logsum_df)

            del utilities_df
            chunk.log_df(trace_label, "utilities_df", None)

            if trace:
                self.trace_df(logsum_df, trace_label, "logsum_df")

            chunk.log_df(trace_label, "logsum_df", logsum_df)
            results = logsum_df

        else:
            assert units == "time"

            # return a series
            results = pd.Series(path_df[units].values, index=path_df["idx"])

            # zero-fill rows for O-D pairs where no best path exists because there was no tap-tap transit availability
            results = reindex(results, maz_od_df.idx).fillna(0.0)

            chunk.log_df(trace_label, "results", results)

        assert len(results) == len(orig)

        del path_df
        chunk.log_df(trace_label, "path_df", None)

        # diagnostic
        # maz_od_df['DIST'] = self.network_los.get_default_skim_dict().get('DIST').get(maz_od_df.omaz, maz_od_df.dmaz)
        # maz_od_df[units] = results.logsum if units == 'utility' else results.values
        # print(f"maz_od_df\n{maz_od_df}")

        return results

    def get_tvpb_logsum(
        self,
        path_type,
        orig,
        dest,
        tod,
        demographic_segment,
        want_choices,
        recipe="tour_mode_choice",
        trace_label=None,
    ):

        # assume they have given us a more specific name (since there may be more than one active wrapper)
        trace_label = trace_label or "get_tvpb_logsum"
        trace_label = tracing.extend_trace_label(trace_label, path_type)

        with chunk.chunk_log(trace_label):

            logsum_df = self.build_virtual_path(
                recipe,
                path_type,
                orig,
                dest,
                tod,
                demographic_segment,
                want_choices=want_choices,
                trace_label=trace_label,
            )

            trace_hh_id = inject.get_injectable("trace_hh_id", None)
            if (all(logsum_df["logsum"] == UNAVAILABLE)) or (len(logsum_df) == 0):
                trace_hh_id = False

            if trace_hh_id:
                filter_targets = tracing.trace_targets(orig)
                # choices from preceding run (because random numbers)
                override_choices = logsum_df["path_num"] if want_choices else None
                if filter_targets.any():
                    self.build_virtual_path(
                        recipe,
                        path_type,
                        orig,
                        dest,
                        tod,
                        demographic_segment,
                        want_choices=want_choices,
                        override_choices=override_choices,
                        trace_label=trace_label,
                        filter_targets=filter_targets,
                        trace=True,
                    )

        return logsum_df

    def get_tvpb_best_transit_time(self, orig, dest, tod):

        # FIXME lots of pathological knowledge here as we are only called by accessibility directly from expressions

        trace_label = tracing.extend_trace_label("accessibility.tvpb_best_time", tod)
        recipe = "accessibility"
        path_type = "WTW"

        with chunk.chunk_log(trace_label):
            result = self.build_virtual_path(
                recipe,
                path_type,
                orig,
                dest,
                tod,
                demographic_segment=None,
                want_choices=False,
                trace_label=trace_label,
            )

            trace_od = inject.get_injectable("trace_od", None)
            if trace_od:
                filter_targets = (orig == trace_od[0]) & (dest == trace_od[1])
                if filter_targets.any():
                    self.build_virtual_path(
                        recipe,
                        path_type,
                        orig,
                        dest,
                        tod,
                        demographic_segment=None,
                        want_choices=False,
                        trace_label=trace_label,
                        filter_targets=filter_targets,
                        trace=True,
                    )

        return result

    def wrap_logsum(
        self,
        orig_key,
        dest_key,
        tod_key,
        segment_key,
        recipe="tour_mode_choice",
        cache_choices=False,
        trace_label=None,
        tag=None,
    ):

        return TransitVirtualPathLogsumWrapper(
            self,
            orig_key,
            dest_key,
            tod_key,
            segment_key,
            recipe,
            cache_choices,
            trace_label,
            tag,
        )


class TransitVirtualPathLogsumWrapper(object):
    """
    Transit virtual path builder logsum wrapper for three zone systems
    """

    def __init__(
        self,
        pathbuilder,
        orig_key,
        dest_key,
        tod_key,
        segment_key,
        recipe,
        cache_choices,
        trace_label,
        tag,
    ):

        self.tvpb = pathbuilder
        assert hasattr(pathbuilder, "get_tvpb_logsum")

        self.orig_key = orig_key
        self.dest_key = dest_key
        self.tod_key = tod_key
        self.segment_key = segment_key
        self.recipe = recipe
        self.df = None

        self.cache_choices = cache_choices
        self.cache = {} if cache_choices else None

        self.base_trace_label = (
            tracing.extend_trace_label(trace_label, tag) or f"tvpb_logsum.{tag}"
        )
        self.trace_label = self.base_trace_label
        self.tag = tag

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
            self.trace_label = tracing.extend_trace_label(
                self.base_trace_label, extension
            )
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
        assert (
            self.orig_key in self.df
        ), f"TransitVirtualPathLogsumWrapper: orig_key '{self.orig_key}' not in df"
        assert (
            self.dest_key in self.df
        ), f"TransitVirtualPathLogsumWrapper: dest_key '{self.dest_key}' not in df"
        assert (
            self.tod_key in self.df
        ), f"TransitVirtualPathLogsumWrapper: tod_key '{self.tod_key}' not in df"
        assert (
            self.segment_key in self.df
        ), f"TransitVirtualPathLogsumWrapper: segment_key '{self.segment_key}' not in df"

        orig = self.df[self.orig_key].astype("int")
        dest = self.df[self.dest_key].astype("int")
        tod = self.df[self.tod_key]
        segment = self.df[self.segment_key]

        logsum_df = self.tvpb.get_tvpb_logsum(
            path_type,
            orig,
            dest,
            tod,
            segment,
            want_choices=self.cache_choices,
            recipe=self.recipe,
            trace_label=self.trace_label,
        )

        if (self.cache_choices) and (not all(logsum_df["logsum"] == UNAVAILABLE)):

            # not tested on duplicate index because not currently needed
            # caching strategy does not require unique indexes but care would need to be taken to maintain alignment
            assert not orig.index.duplicated().any()

            # we only need to cache taps and path_set
            choices_df = logsum_df[["atap", "btap", "path_set"]]

            if path_type in self.cache:
                assert (
                    len(self.cache.get(path_type).index.intersection(logsum_df.index))
                    == 0
                )
                choices_df = pd.concat([self.cache.get(path_type), choices_df])

            self.cache[path_type] = choices_df

        return logsum_df.logsum

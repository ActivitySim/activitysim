# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import os
import warnings
from typing import Any

from activitysim.abm.tables import disaggregate_accessibility, shadow_pricing
from activitysim.core import chunk, expressions, tracing, workflow
from activitysim.core.configuration.base import PydanticReadable
from activitysim.core.configuration.logit import PreprocessorSettings

# We are using the naming conventions in the mtc_asim.h5 example
# file for our default list. This provides backwards compatibility
# with previous versions of ActivitySim in which only 'input_store'
# is given in the settings file.
DEFAULT_TABLE_LIST = [
    {
        "tablename": "households",
        "h5_tablename": "households",
        "index_col": "household_id",
    },
    {"tablename": "persons", "h5_tablename": "persons", "index_col": "person_id"},
    {"tablename": "land_use", "h5_tablename": "land_use_taz", "index_col": "TAZ"},
]

logger = logging.getLogger(__name__)


def annotate_tables(state: workflow.State, model_settings, trace_label, chunk_sizer):
    """

    Parameters
    ----------
    state : workflow.State
    model_settings : PydanticReadable
    trace_label : str
    chunk_sizer : ChunkSizer

    Returns
    -------

    """

    trace_label = tracing.extend_trace_label(trace_label, "annotate_tables")

    chunk_sizer.log_rss(trace_label)

    annotate_tables = model_settings.annotate_tables
    print(annotate_tables)

    if not annotate_tables:
        logger.warning(
            f"{trace_label} - annotate_tables setting is empty - nothing to do!"
        )

    assert isinstance(
        annotate_tables, list
    ), f"annotate_tables settings should be a list but is {type(annotate_tables)}"

    t0 = tracing.print_elapsed_time()

    for table_info in annotate_tables:
        tablename = table_info.tablename

        chunk_sizer.log_rss(f"{trace_label}.pre-get_table.{tablename}")

        df = state.get_dataframe(tablename)
        chunk_sizer.log_df(trace_label, tablename, df)

        # - rename columns
        column_map = table_info.column_map
        if column_map:
            warnings.warn(
                f"Setting 'column_map' has been changed to 'rename_columns'. "
                f"Support for 'column_map' in annotate_tables  will be removed in future versions.",
                FutureWarning,
            )

            logger.info(f"{trace_label} - renaming {tablename} columns {column_map}")
            df.rename(columns=column_map, inplace=True)

        # - annotate
        annotate = table_info.annotate
        if annotate:
            logger.info(f"{trace_label} - annotating {tablename} SPEC {annotate.SPEC}")
            expressions.assign_columns(
                state, df=df, model_settings=annotate, trace_label=trace_label
            )

        chunk_sizer.log_df(trace_label, tablename, df)

        # - write table to pipeline
        state.add_table(tablename, df)

        del df
        chunk_sizer.log_df(trace_label, tablename, None)


class AnnotateTableSettings(PydanticReadable):
    tablename: str
    annotate: PreprocessorSettings
    column_map: dict[str, str] | None = None


class InitializeTableSettings(PydanticReadable):
    """
    Settings for the `initialize_landuse` component.
    """

    annotate_tables: list[AnnotateTableSettings] = []


@workflow.step
def initialize_landuse(
    state: workflow.State,
    model_settings: InitializeTableSettings | None = None,
    model_settings_file_name: str = "initialize_landuse.yaml",
    trace_label: str = "initialize_landuse",
) -> None:
    """
    Initialize the land use table.

    Parameters
    ----------
    state : State
    """
    with chunk.chunk_log(state, trace_label, base=True) as chunk_sizer:
        if model_settings is None:
            model_settings = InitializeTableSettings.read_settings_file(
                state.filesystem,
                model_settings_file_name,
                mandatory=True,
            )

        annotate_tables(state, model_settings, trace_label, chunk_sizer)

        # instantiate accessibility (must be checkpointed to be be used to slice accessibility)
        accessibility = state.get_dataframe("accessibility")
        chunk_sizer.log_df(trace_label, "accessibility", accessibility)


@workflow.step
def initialize_households(
    state: workflow.State,
    model_settings_file_name: str = "initialize_households.yaml",
    trace_label: str = "initialize_households",
) -> None:
    with chunk.chunk_log(state, trace_label, base=True) as chunk_sizer:
        chunk_sizer.log_rss(f"{trace_label}.inside-yield")

        households = state.get_dataframe("households")
        assert not households._is_view
        chunk_sizer.log_df(trace_label, "households", households)
        del households
        chunk_sizer.log_df(trace_label, "households", None)

        persons = state.get_dataframe("persons")
        assert not persons._is_view
        chunk_sizer.log_df(trace_label, "persons", persons)
        del persons
        chunk_sizer.log_df(trace_label, "persons", None)

        model_settings = InitializeTableSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
            mandatory=True,
        )
        annotate_tables(state, model_settings, trace_label, chunk_sizer)

        # - initialize shadow_pricing size tables after annotating household and person tables
        # since these are scaled to model size, they have to be created while single-process
        # this can now be called as a stand alone model step instead, add_size_tables
        # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
        suffixes = disaggregate_accessibility.disaggregate_suffixes(state)
        shadow_pricing.add_size_tables(state, suffixes)

        # create disaggregate_accessibility table if not model was run
        if state.is_table("proto_disaggregate_accessibility"):
            disaggregate_accessibility.disaggregate_accessibility(state)

        # - preload person_windows
        person_windows = state.get_dataframe("person_windows")
        chunk_sizer.log_df(trace_label, "person_windows", person_windows)


@workflow.cached_object
def preload_injectables(state: workflow.State):
    """
    preload bulky injectables up front - stuff that isn't inserted into the pipeline
    """

    logger.info("preload_injectables")

    # FIXME undocumented feature
    if state.settings.write_raw_tables:
        # write raw input tables as csv (before annotation)
        csv_dir = state.get_output_file_path("raw_tables")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)  # make directory if needed

        # default ActivitySim table names and indices
        if state.settings.input_table_list is None:
            raise ValueError(
                "no `input_table_list` found in settings, " "cannot `write_raw_tables`."
            )

        table_names = [t["tablename"] for t in state.settings.input_table_list]
        for t in table_names:
            df = state.get_dataframe(t)
            df.to_csv(os.path.join(csv_dir, "%s.csv" % t), index=True)

    t0 = tracing.print_elapsed_time()

    if state.settings.benchmarking:
        # we don't want to pay for skim_dict inside any model component during
        # benchmarking, so we'll preload skim_dict here.  Preloading is not needed
        # for regular operation, as activitysim components can load-on-demand.
        if state.get_injectable("skim_dict", None) is not None:
            t0 = tracing.print_elapsed_time("preload skim_dict", t0, debug=True)

    return True

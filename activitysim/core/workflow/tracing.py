import logging
import logging.config
import sys
from collections.abc import Mapping, MutableMapping, Sequence

import pandas as pd
import yaml

from activitysim.core.workflow.accessor import FromWhale, WhaleAccessor

logger = logging.getLogger(__name__)


DEFAULT_TRACEABLE_TABLES = [
    "households",
    "persons",
    "tours",
    "joint_tour_participants",
    "trips",
    "vehicles",
]


class Tracing(WhaleAccessor):
    traceable_tables: list[str] = FromWhale(default_value=DEFAULT_TRACEABLE_TABLES)
    traceable_table_ids: dict[str, Sequence] = FromWhale(default_init=True)
    traceable_table_indexes: dict[str, str] = FromWhale(default_init=True)

    def initialize(self):
        self.traceable_table_ids = {}

    def register_traceable_table(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Register traceable table

        Parameters
        ----------
        table_name : str
        df: pandas.DataFrame
            The traced dataframe.
        """

        # add index name to traceable_table_indexes

        logger.debug(f"register_traceable_table {table_name}")

        traceable_tables = self.traceable_tables
        if table_name not in traceable_tables:
            logger.error("table '%s' not in traceable_tables" % table_name)
            return

        idx_name = df.index.name
        if idx_name is None:
            logger.error("Can't register table '%s' without index name" % table_name)
            return

        traceable_table_ids = self.traceable_table_ids
        traceable_table_indexes = self.traceable_table_indexes

        if (
            idx_name in traceable_table_indexes
            and traceable_table_indexes[idx_name] != table_name
        ):
            logger.error(
                "table '%s' index name '%s' already registered for table '%s'"
                % (table_name, idx_name, traceable_table_indexes[idx_name])
            )
            return

        # update traceable_table_indexes with this traceable_table's idx_name
        if idx_name not in traceable_table_indexes:
            traceable_table_indexes[idx_name] = table_name
            logger.debug(
                "adding table %s.%s to traceable_table_indexes" % (table_name, idx_name)
            )
            self.traceable_table_indexes = traceable_table_indexes

        # add any new indexes associated with trace_hh_id to traceable_table_ids

        trace_hh_id = self.obj.settings.trace_hh_id
        if trace_hh_id is None:
            return

        new_traced_ids = []
        # if table_name == "households":
        if table_name in ["households", "proto_households"]:
            if trace_hh_id not in df.index:
                logger.warning("trace_hh_id %s not in dataframe" % trace_hh_id)
                new_traced_ids = []
            else:
                logger.info(
                    "tracing household id %s in %s households"
                    % (trace_hh_id, len(df.index))
                )
                new_traced_ids = [trace_hh_id]
        else:
            # find first already registered ref_col we can use to slice this table
            ref_col = next(
                (c for c in traceable_table_indexes if c in df.columns), None
            )

            if ref_col is None:
                logger.error(
                    "can't find a registered table to slice table '%s' index name '%s'"
                    " in traceable_table_indexes: %s"
                    % (table_name, idx_name, traceable_table_indexes)
                )
                return

            # get traceable_ids for ref_col table
            ref_col_table_name = traceable_table_indexes[ref_col]
            ref_col_traced_ids = traceable_table_ids.get(ref_col_table_name, [])

            # inject list of ids in table we are tracing
            # this allows us to slice by id without requiring presence of a household id column
            traced_df = df[df[ref_col].isin(ref_col_traced_ids)]
            new_traced_ids = traced_df.index.tolist()
            if len(new_traced_ids) == 0:
                logger.warning(
                    "register %s: no rows with %s in %s."
                    % (table_name, ref_col, ref_col_traced_ids)
                )

        # update the list of trace_ids for this table
        prior_traced_ids = traceable_table_ids.get(table_name, [])

        if new_traced_ids:
            assert not set(prior_traced_ids) & set(new_traced_ids)
            traceable_table_ids[table_name] = prior_traced_ids + new_traced_ids
            self.traceable_table_ids = traceable_table_ids

        logger.debug(
            "register %s: added %s new ids to %s existing trace ids"
            % (table_name, len(new_traced_ids), len(prior_traced_ids))
        )
        logger.debug(
            "register %s: tracing new ids %s in %s"
            % (table_name, new_traced_ids, table_name)
        )

    def deregister_traceable_table(self, table_name: str) -> None:
        """
        un-register traceable table

        Parameters
        ----------
        table_name : str
        """
        traceable_table_ids = self.traceable_table_ids
        traceable_table_indexes = self.traceable_table_indexes

        if table_name not in self.traceable_tables:
            logger.error("table '%s' not in traceable_tables" % table_name)

        else:
            self.traceable_table_ids = {
                k: v for k, v in traceable_table_ids.items() if k != table_name
            }
            self.traceable_table_indexes = {
                k: v for k, v in traceable_table_indexes.items() if v != table_name
            }

from __future__ import annotations

from activitysim.core.workflow.accessor import StateAccessor


class Extend(StateAccessor):
    """
    Methods to extend the capabilities of ActivitySim.
    """

    def __get__(self, instance, objtype=None) -> "Extend":
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def declare_table(
        self, table_name, traceable=True, random_channel=True, index_name=None
    ):
        """
        Declare a new table.

        Parameters
        ----------
        table_name : str
        traceable : bool, default True
        random_channel : bool, default True
        index_name : str, optional

        """

        traceable_tables = self._obj.tracing.traceable_tables
        if traceable and table_name not in traceable_tables:
            traceable_tables.append(table_name)
        self._obj.set("traceable_tables", traceable_tables)

        from activitysim.abm.models.util import canonical_ids

        rng_channels = self._obj.get("rng_channels")
        if random_channel and table_name not in rng_channels:
            rng_channels.append(table_name)
        self._obj.set("rng_channels", rng_channels)

        canonical_table_index_names = self._obj.get("canonical_table_index_names")
        if index_name is not None and table_name not in canonical_table_index_names:
            canonical_table_index_names[table_name] = index_name
        self._obj.set("canonical_table_index_names", canonical_table_index_names)

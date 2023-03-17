import logging

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep(updates_context=True)
def join_table_data(
    tablesets,
    tablename,
    from_tablename,
    columns,
    on,
) -> dict:
    if isinstance(columns, str):
        columns = [columns]
    if len(columns) == 1:
        columns_note = columns[0]
    else:
        columns_note = f"{len(columns)} columns"

    reset_progress_step(description=f"join table data / {tablename} <- {columns_note}")

    for key, tableset in tablesets.items():
        other = tableset[from_tablename][columns]
        tablesets[key][tablename] = tablesets[key][tablename].join(other, on=on)

    return dict(tablesets=tablesets)

from xmle import NumberedCaption, Reporter

from ..progression import reset_progress_step
from ..wrapping import workstep


@workstep(updates_context=True)
def init_report(
    title,
):
    reset_progress_step(description="initialize report")
    return dict(
        report=Reporter(title=title),
        fig=NumberedCaption("Figure", level=2, anchor=True),
        tab=NumberedCaption("Table", level=2, anchor=True),
    )

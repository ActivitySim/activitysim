from pypyr.context import Context
from xmle import Reporter, NumberedCaption
from ..progression import reset_progress_step


def run_step(context: Context) -> None:
    reset_progress_step(description="initialize report")

    title = context.get_formatted('title')
    context['report'] = Reporter(title=title)
    context['fig'] = NumberedCaption("Figure", level=2, anchor=True)
    context['tab'] = NumberedCaption("Table", level=2, anchor=True)


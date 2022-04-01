from pypyr.context import Context
from xmle import Reporter, NumberedCaption


def run_step(context: Context) -> None:
    title = context.get_formatted('title')
    context['report'] = Reporter(title=title)
    context['fig'] = NumberedCaption("Figure", level=2, anchor=True)

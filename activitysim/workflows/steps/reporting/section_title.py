import logging

from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep(returns_names="report")
def section_title(
    report,
    title,
    level=2,
):
    with report:
        t = "#" * level
        report << f"{t} {title}"
    return report

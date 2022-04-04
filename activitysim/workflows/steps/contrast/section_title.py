
import logging
from pypyr.context import Context
from pypyr.errors import KeyNotInContextError
from ..error_handler import error_logging

logger = logging.getLogger(__name__)


@error_logging
def run_step(context: Context) -> None:

    context.assert_key_has_value(key='report', caller=__name__)
    report = context.get('report')

    title = context.get_formatted('title')
    try:
        level = context.get_formatted('level')
    except (KeyError, KeyNotInContextError):
        level = 2

    with report:
        t = "#" * level
        report << f"{t} {title}"



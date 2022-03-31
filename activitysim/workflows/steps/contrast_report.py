import logging
import os
from pypyr.context import Context
from .progression import reset_progress_step
from .error_handler import error_logging

@error_logging
def run_step(context: Context) -> None:

    reset_progress_step(description="contrast report")

    context.assert_key_has_value(key='workspace', caller=__name__)
    context.assert_key_has_value(key='example_name', caller=__name__)
    context.assert_key_has_value(key='tag', caller=__name__)
    context.assert_key_has_value(key='archive_dir', caller=__name__)

    workspace = context.get_formatted('workspace')
    example_name = context.get_formatted('example_name')
    tag = context.get_formatted('tag')
    archive_base = context.get_formatted('archive_base')

    from activitysim.standalone.render import render_comparison
    if example_name in {"example_mtc", "example_mtc_full"}:
        dist_name = "DIST"
        county_id = "county_id"
    elif example_name in {"example_arc", "example_arc_full"}:
        dist_name = "SOV_FREE_DISTANCE"
        county_id = "county"
    else:
        dist_name = None
        county_id = None

    cwd = os.getcwd()
    try:
        os.chdir(f"{workspace}/{example_name}")
        render_comparison(
            f"{archive_base}/report-{tag}.html",
            title=example_name,
            dist_skim=dist_name,
            county_id=county_id,
            timing_log=f"combined_timing_log-{tag}.csv"
        )
    except FileNotFoundError as err:
        logging.error(f"FileNotFoundError in contrast report: {err.filename}")
    finally:
        os.chdir(cwd)


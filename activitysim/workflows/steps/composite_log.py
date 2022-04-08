import os
from pypyr.context import Context
from .progression import reset_progress_step
from .error_handler import error_logging


def to_csv_safe(obj, filename, *args, **kwargs):
    """
    Write a csv as normal, changing the filename if a permission error occurs.
    """
    try:
        obj.to_csv(filename, *args, **kwargs)
    except PermissionError:
        n = 1
        f1, f2 = os.path.splitext(filename)
        while os.path.exists(f"{f1} ({n}){f2}"):
            n += 1
        obj.to_csv(f"{f1} ({n}){f2}", *args, **kwargs)



@error_logging
def run_step(context: Context) -> None:

    reset_progress_step(description="composite timing and memory logs")

    context.assert_key_has_value(key='tag', caller=__name__)
    context.assert_key_has_value(key='archive_dir', caller=__name__)

    tag = context.get_formatted('tag')
    archive_dir = context.get_formatted('archive_dir')

    import pandas as pd
    timings = {}
    compares = ['compile', 'sharrow', 'legacy']
    for t in compares:
        filename = f"{archive_dir}/output-{t}/log/timing_log.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = df.set_index("model_name")['seconds']
            timings[t] = df.loc[~df.index.duplicated()]
    if timings:
        composite_timing = pd.concat(timings, axis=1)
        to_csv_safe(composite_timing, f"{archive_dir}/combined_timing_log-{tag}.csv")
    mems = {}
    for t in compares:
        filename = f"{archive_dir}/output-{t}/log/mem.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = df.set_index('event')[['rss', 'full_rss', 'uss']]
            mems[t] = df.loc[~df.index.duplicated()]
    if mems:
        composite_mem = pd.concat(mems, axis=1)
        to_csv_safe(composite_mem, f"{archive_dir}/combined_mem_log-{tag}.csv")

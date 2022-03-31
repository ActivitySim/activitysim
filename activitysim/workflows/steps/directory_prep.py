import os
from pypyr.context import Context
from .progression import reset_progress_step
from .error_handler import error_logging
from pathlib import Path


def _prep_dir(directory):
    directory = Path(directory)
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory/"log", exist_ok=True)
    os.makedirs(directory/"trace", exist_ok=True)
    gitignore = directory / ".gitignore"
    if not os.path.exists(gitignore):
        with open(gitignore, 'wt') as f:
            f.write("/*")


@error_logging
def run_step(context: Context) -> None:

    context.assert_key_has_value(key='tag', caller=__name__)
    context.assert_key_has_value(key='workspace', caller=__name__)
    context.assert_key_has_value(key='example_name', caller=__name__)

    compile = context.get('compile', True)
    sharrow = context.get('sharrow', True)
    legacy = context.get('legacy', True)
    tag = context.get_formatted('tag')
    workspace = context.get_formatted('workspace')
    example_name = context.get_formatted('example_name')

    archive_dir = f"{workspace}/{example_name}/output-{tag}"
    os.makedirs(archive_dir, exist_ok=True)
    if compile:
        _prep_dir(f"{archive_dir}/output-compile")
    if sharrow:
        _prep_dir(f"{archive_dir}/output-sharrow")
    if legacy:
        _prep_dir(f"{archive_dir}/output-legacy")

    context['archive_dir'] = archive_dir
    context['archive_base'] = os.path.basename(archive_dir)
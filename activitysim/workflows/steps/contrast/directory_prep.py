import os
from pathlib import Path

from ..wrapping import workstep


def _prep_dir(directory):
    directory = Path(directory)
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory / "log", exist_ok=True)
    os.makedirs(directory / "trace", exist_ok=True)
    gitignore = directory / ".gitignore"
    if not os.path.exists(gitignore):
        with open(gitignore, "wt") as f:
            f.write("/*")


@workstep(updates_context=True)
def directory_prep(
    tag,
    example_name,
    workspace,
    compile=True,
    sharrow=True,
    legacy=True,
    reference=True,
    chunk_training=None,
):
    archive_dir = f"{workspace}/{example_name}/output-{tag}"
    os.makedirs(archive_dir, exist_ok=True)
    if compile:
        _prep_dir(f"{archive_dir}/output-compile")
    if sharrow:
        _prep_dir(f"{archive_dir}/output-sharrow")
        if chunk_training:
            _prep_dir(f"{archive_dir}/output-sharrow-training")
    if legacy:
        _prep_dir(f"{archive_dir}/output-legacy")
        if chunk_training:
            _prep_dir(f"{archive_dir}/output-legacy-training")
    if reference:
        _prep_dir(f"{archive_dir}/output-reference")
    return dict(
        archive_dir=archive_dir,
        archive_base=os.path.basename(archive_dir),
    )

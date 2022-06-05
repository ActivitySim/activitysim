import os
import shlex

from ...standalone.utils import chdir
from .progression import reset_progress_step
from .wrapping import workstep


@workstep
def create(example_name, destination=None) -> None:
    """
    Install a functioning example from the ActivitySim resources collection.

    The behavior of this workstep differs slightly from the CLI of
    `create`: on the command line, if the destination directory does not
    already exist, it is created and becomes the target location to install
    the example model. This can result in the example model being installed
    into a (newly created) directory with any name, not necessarily the name
    of the example being installed.

    This workstep function creates the destination directory if it does not
    exist, but then always creates a subdirectory within the destination, named
    according to the `example_name`.  This ensures stability and guarantees the
    resulting installed example is in the same location whether that directory
    existed before or not.

    Parameters
    ----------
    example_name : str
        The name of the example to be installed, which should be listed in
        the example manifest.
    destination : path-like, optional
        The directory where the example should be installed, defaulting to the
        current working directory.  The example is always then installed into
        "destination/example_name".

    """
    reset_progress_step(
        description=f"activitysim create {example_name}", prefix="[bold green]"
    )

    args = f"create -e {example_name} --link -d ."
    if destination:
        os.makedirs(destination, exist_ok=True)
    else:
        destination = "."

    # Call the run program inside this process
    from activitysim.cli.main import prog

    with chdir(destination):
        namespace = prog().parser.parse_args(shlex.split(args))
        namespace.afunc(namespace)

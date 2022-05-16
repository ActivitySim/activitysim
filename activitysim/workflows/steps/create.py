from .progression import reset_progress_step
from .wrapping import workstep
import shlex


@workstep
def create_activitysim(
    example_name,
    destination=None,
    label=None,
) -> None:
    args = f"create -e {example_name} --link"
    if destination:
        args += f' -d "{destination}"'
    if label is None:
        label = f"activitysim {args}"

    reset_progress_step(description=f"{label}", prefix="[bold green]")

    # Call the run program inside this process
    from activitysim.cli.main import prog
    namespace = prog().parser.parse_args(shlex.split(args))
    namespace.afunc(namespace)


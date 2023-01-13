from .progression import update_progress_overall
from .wrapping import workstep


@workstep
def title(
    label="ActivitySim Workflow",
    formatting="bold blue",
) -> None:
    update_progress_overall(label, formatting)
